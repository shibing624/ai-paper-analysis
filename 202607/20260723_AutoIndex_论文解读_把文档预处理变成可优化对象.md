---
title: "AutoIndex 论文解读：把文档预处理变成可优化对象，BM25 也能涨 8 个点"
date: 2026-07-23
arxiv: 2607.18603
authors: Sam O'Nuallain, Nithya Rajkumar, Ramya Narayanasamy, Hanna Jiang, Shreyas Chaudhari, Andrew Drozdov
---

你有没有这种感觉：搭 RAG 系统的时候，retriever 换了又换、reranker 加了一层又一层、prompt 调了几十版，检索质量就是死活提不上去。然后你回头看了一眼索引构建——chunk size 512、overlap 64、标题保留、停用词默认——这些参数从项目第一天起就没动过。

这篇论文就在戳这个痛点：**文档表示（document representation）一直被当作"基础设施"，从来没人把它当作优化目标**。AutoIndex 的思路很直接：把索引前那段预处理代码变成可搜索的程序空间，让两个 LLM Agent 协作去改它，改完用 BM25 跑一遍验证集，按 Recall@100 选优。

听起来很暴力对吧？结果在 CRUMB 8 个任务上，BM25 一行没动，平均 Recall@100 涨了 **8.4 个百分点**，nDCG@10 涨了 **8.3 个点**。在 SetOpEntity（集合操作实体查询）上，nDCG@10 直接涨了 **43.6 个点**。我没写错，是 43.6。

---

## 核心摘要

AutoIndex 是一个把"文档表示"当作可执行程序来搜索的框架。它不调 retriever、不调 reranker、不加 embedding，只改索引前那段把原始文档变成可索引单元的代码。两个 Agent 协作：一个负责诊断当前索引下的检索失败模式（Analysis Agent），一个负责据此写新的预处理代码（Code Agent）。候选程序跑在沙箱里，重建索引，用验证集 Recall@100 选优。在 CRUMB 8 个任务上，相对固定的全文档 BM25 基线，平均 Recall@100 提升 8.4 个点，nDCG@10 提升 8.3 个点，最大提升分别达到 30.5 和 43.6 个点。

我的判断：这是一篇"框架驱动"的论文，单点创新不算大（程序搜索 + 工具调用 LLM 都不是新东西），但把这两者焊到文档索引这个被忽视的优化层上，确实切到了一个真实痛点。消融实验做得很老实，承认了 1 次迭代不够、搜索历史起稳定作用而非提升作用、Analysis Agent 是关键信号源。**最大的隐忧是：把所有赌注压在 BM25 的词项匹配上，dense retriever 的可迁移性只验证了一个任务**。

---

## 论文信息

- **标题**：AutoIndex: Learning Representation Programs for Retrieval
- **作者**：Sam O'Nuallain, Nithya Rajkumar, Ramya Narayanasamy, Hanna Jiang, Shreyas Chaudhari, Andrew Drozdov
- **机构**：从作者署名看是工业界团队（提交邮箱对应 Andrew Drozdov），GitHub 仓库在 [auto-index/autoindex](https://github.com/auto-index/autoindex)
- **arXiv**：[2607.18603](https://arxiv.org/abs/2607.18603)
- **提交日期**：2026 年 7 月 21 日
- **代码**：已开源

---

## 为什么要把这块做成"可优化对象"

先说清楚作者在反对什么。检索系统的优化空间，传统上被切成这几块：

| 优化层 | 典型操作 | 论文的态度 |
|--------|----------|------------|
| Retriever | 换 BM25/Dense/Hybrid、改 embedding 模型 | 固定不动 |
| Reranker | 加 cross-encoder、调 rerank 阈值 | 固定不动 |
| Prompt / Generator | 改 system prompt、换 LLM | 固定不动 |
| **文档表示** | **chunk size、overlap、停用词、metadata 模板** | **这篇要优化的就是这个** |

作者的核心论点是这样的：现有 RAG 优化工作（论文里点名了 Zeng et al. 2026 和 Kartal et al. 2025）搜的是 retriever × reranker × prompt × generator 的组合，但**没人去搜"文档在进入索引之前应该被怎么切、怎么改写、怎么重组"这件事**。这块被当成"基础设施"，配置一次就再也不碰。

但实际上，文档表示决定了哪些词项被暴露给 BM25、哪些上下文被绑定到每个 chunk、哪些结构信息被保留。作者做了一个有意思的类比——如果把检索系统比作编译器，文档表示就是它的前端 IR（中间表示）。编译器优化几十年都在做 pass，没人觉得 IR 选错了也能出好代码；但检索系统的 IR 选择，几乎没人系统研究过。

这个观察我认。**在工业界 RAG 项目里，我见过的最常见的"调不动"现象就是 chunking 策略定得太死**——产品经理说"就按 512 切吧，行业标准"，然后后面所有问题都让 retriever 背锅。

---

## 方法：一个双 Agent 的程序搜索循环

AutoIndex 的核心结构很简洁，我用论文的 Figure 1 来说清楚：

![Figure 1: AutoIndex 优化循环示意图](https://arxiv.org/html/2607.18603v1/images/698_figure_1-cropped.png)

*图 1：AutoIndex 的优化循环。输入是源语料、验证查询、固定 retriever；当前程序构建索引 → Analysis Agent 诊断失败 → Code Agent 合成候选更新 → Selection 选优 → 输出学习到的表示程序和检索质量指标。红色箭头是"用更好的程序替换 incumbent 并重复"的反馈回路。*

整个流程跑 5 轮，每轮里有三个角色：

### Analysis Agent：诊断失败模式

这个 Agent 拿到当前程序 θ(t)、一组分层采样过的验证查询 Qc、和一个**只读工具集** {bm25_retrieve, read_file, grep_search}。它的产出是一段自然语言诊断摘要 s(t)。

诊断查询被分成三类：
1. **Anchors**：在初始程序下能召回 gold，但当前程序召不回的查询
2. **Recall Violations**：当前程序召不回 gold 的查询
3. **Small-Margin Positives**：当前程序能把 gold 召进 top-k，但召不进 top-1

每类抽 5 个。这个设计很关键——**它强制 Agent 去关注"程序改动后变差"和"始终没改好"这两类问题**，而不是平均用力。作者说是 priority order：先填 Anchors，再填剩下的 Recall Violations。

### Code Agent：写新程序

拿到诊断摘要 s(t) 和搜索历史 H(t)（所有之前评估过的程序及其验证分数），这个 Agent **一次性生成 N=4 个候选程序**。每个候选都要：

1. 语法验证
2. 沙箱执行（超时 15 分钟）
3. 重建索引
4. 跑验证集 Recall@100
5. 算 ΔJ = J(θ_new) - J(θ_current)

只有 ΔJ ≥ 10⁻⁵ 的候选才进入"采纳集" 𝒜(t)。

### Selection：决定下一轮的 incumbent

- 𝒜(t) 为空 → incumbent 不变
- 只有一个候选 → 它就是新的 incumbent
- 多个候选 → 让 LLM 写一个"组合这些改动的合成程序"，只有当合成版本严格超过最优单候选时才采纳，否则退回最优单候选

最后在验证集上表现最好的那个程序，在 held-out 测试集上评估一次。

### 关于"搜索历史"这个设计

我特别想强调一下这个细节。H(t) = {(θi, ΔJi)}_{i<t}，就是所有之前试过的程序和它们的结果。Code Agent 拿到这个历史，能避免重复犯同样的错。消融里能看到这个组件的作用——去掉历史，5/8 任务仍然正收益，但 CodeRetrieval 涨了 **5.9 个点**、StackExchange 涨了 **4.7 个点**，说明历史在"防破坏性编辑"上比"找新增益"更值钱。

---

## 实验设置：CRUMB + BM25 + 5 次迭代

CRUMB（Complex Retrieval Unified Multi-task Benchmark，Killingback & Zamani 2025）选了 8 个异构任务：

| 缩写 | 任务 | 查询数 |
|------|------|--------|
| CT | ClinicalTrial | 84 |
| CR | CodeRetrieval | 2510 |
| LQA | LegalQA | 4569 |
| PR | PaperRetrieval | 53 |
| SOE | SetOpEntity | 314 |
| SE | StackExchange | 79 |
| TR | TheoremRetrieval | 51 |
| TOT | TipOfTongue | 100 |

基线两个：
- **BM25 Full-Doc**：把整个 markdown 文档作为一个单元索引
- **CRUMB Passage Corpus**：用 512-BERT-token 的固定切分

两个 retriever backbones：Claude Sonnet 4.6（n=2）和 qwen3-coder（n=3）。每轮 N=4 候选，5 轮迭代。

---

## 主实验结果：8/8 全线提升

### Table 1：Recall@100（qwen3-coder）

| Split | BM25 Full-Doc | Passage | AutoIndex | Δ vs Full-Doc | Δ vs Passage |
|-------|---------------|---------|-----------|---------------|--------------|
| CT | 20.7 | 8.4 | 21.2 ± 0.1 | +2.1% | +152.8% |
| LQA | 55.1 | 22.4 | 60.8 ± 6.0 | **10.4 个点** | +171.8% |
| SOE | 25.7 | 15.1 | 33.5 ± 5.8 | **30.5 个点** | +122.4% |
| SE | 67.1 | 49.0 | 69.8 ± 1.3 | +4.1% | +42.6% |
| TOT | 25.0 | 5.8 | 26.7 ± 1.1 | +6.7% | +361.4% |
| CR | 4.7 | — | 5.0 ± 0.4 | +6.5% | — |
| PR | 33.7 | — | 33.7 ± 0.02 | +0.1% | — |
| TR | 8.5 | — | 10.1 ± 1.5 | **19.2 个点** | — |
| **AVG** | **30.1** | — | **32.6** | **8.4 个点** | — |

几个值得注意的点：

- **SOE 涨 30.5 个点**：集合操作实体查询（比如"找出同时出现在 X 和 Y 里的实体"），BM25 词项匹配天然吃亏，因为查询里的实体和文档里的实体不在同一句。AutoIndex 学到了某种能提升这类查询召回的表示方式
- **TR 涨 19.2 个点**：定理检索，定理往往被嵌入在长证明里，full-doc 索引让 BM25 词频被稀释
- **PR 几乎不变（+0.1%）**：论文检索的语料结构相对规整，预处理优化的空间本来就小
- **相对 Passage Corpus 基线的提升更大**（平均 +100% 以上），说明问题不是"chunk 不够细"，而是"怎么切、怎么重组"需要 corpus-specific 的设计

### Table 2：nDCG@10（qwen3-coder）

| Split | BM25 Full-Doc | Passage | AutoIndex | Δ vs Full-Doc |
|-------|---------------|---------|-----------|---------------|
| CT | 52.2 | 42.3 | 53.0 ± 0.1 | +1.7% |
| LQA | 16.4 | 4.5 | 23.4 ± 8.3 | **42.5 个点** |
| SOE | 12.2 | 11.9 | 17.5 ± 4.4 | **43.6 个点** |
| SE | 21.9 | 11.8 | 24.5 ± 1.9 | **11.8 个点** |
| TOT | 12.0 | 2.5 | 11.5 ± 0.5 | 负 3.5 个点 |
| CR | 4.4 | — | 4.9 ± 0.7 | +9.6% |
| PR | 67.3 | — | 66.8 ± 0.6 | 负 0.7 个点 |
| TR | 0.5 | — | 0.7 ± 0.2 | **27.3 个点** |
| **AVG** | **23.4** | — | **25.3** | **8.3 个点** |

**注意一个反直觉的点**：AutoIndex 的目标函数是验证集 Recall@100，但 nDCG@10 几乎在所有任务上都同步提升。**这意味着学到的程序不是简单"把更多候选塞进 top-100"，而是确实提升了排序质量**。LQA 和 SOE 的 nDCG@10 提升超过 40 个点，这幅度在检索领域是非常罕见的。

TOT 的 nDCG@10 反而掉了 3.5%，但 Recall@100 涨了 6.7%。这说明该任务的程序在扩大召回的同时引入了一些排序噪声，作者没回避这个点。

### Dense Retriever 的迁移性（论文里的一段小实验）

作者只测了一个任务：StackExchange 上，用 BM25 学到的程序直接套到 Qwen3-Embedding-0.6B 上：

- BM25 baseline：0.7391
- BM25 学的程序 + Dense retriever：**0.8741（+18.3%）**

这个结果暗示：**学到的预处理程序不只服务于词项匹配，它捕获的某种"语料结构适配"对 dense retriever 也有帮助**。但样本量只有 1 个任务，作者很诚实地标了"broader dense, hybrid, and reranking evaluations remain future work"。

---

## 消融实验：每个组件到底贡献了多少

Table 3 是消融，四个条件：Full / 1 iter / w/o history / w/o analysis。

| Split | Full | 1 iter | w/o history | w/o analysis |
|-------|------|--------|-------------|--------------|
| CT | +2.1% | -1.1% | -1.3% | +0.8% |
| CR | +6.5% | +0.4% | +5.9% | -0.3% |
| LQA | +10.4% | +1.6% | +1.7% | -4.1% |
| PR | +0.1% | 0.0% | -0.4% | +0.4% |
| SOE | +30.5% | +7.0% | +4.5% | +2.0% |
| SE | +4.1% | -3.1% | +4.7% | +4.1% |
| TR | +19.2% | 0.0% | 0.0% | +7.7% |
| TOT | +6.7% | -10.0% | +4.0% | +2.0% |
| **正收益任务数** | **8/8** | **3/8** | **5/8** | **6/8** |

几个关键发现：

1. **迭代不是装饰，是核心**。1 iter 只能让 3/8 任务正收益，TOT 直接 -10%。**有用的程序往往要好几轮分析-提案-评估才能被找到**，不是一次 prompt 重写能搞定的。
2. **搜索历史是稳定器，不是放大器**。去掉历史，5/8 仍然正收益，但 CodeRetrieval 从 +6.5% 跌到 +5.9%，StackExchange 反而从 +4.1% 涨到 +4.7%。**说明历史在防止破坏性编辑，而不是直接贡献增益**——这个区分很重要。
3. **Analysis Agent 是信号源**。去掉它，6/8 仍正收益但幅度大幅缩小（LQA 从 +10.4% 跌到 -4.1%）。**只给 Code Agent 看聚合指标数字，它根本不知道往哪改**。

---

## 搜索动态：不是一蹴而就的

Figure 2 展示了三个代表性任务在 5 轮迭代中的 ΔnDCG@10 轨迹：

![Figure 2: 迭代动态](https://arxiv.org/html/2607.18603v1/x1.png)

*图 2：三个 CRUMB 任务在 5 轮迭代中的 ΔnDCG@10 轨迹。实心点是被采纳的候选，空心点是被拒绝的（ΔJ < 10⁻⁵）。StackExchange 是"早赢后精修"型、SetOpEntity 是"逐轮累积"型、ClinicalTrial 是"反复接近阈值被拒"型。*

这个图透露了几个信息：
- **StackExchange 在第 1 轮就涨了 12 个点**——说明有些增益是 trivial 的（去掉 LaTeX 噪声），1 iter 也能拿到一部分
- **SetOpEntity 是慢热型**——前 2 轮几乎没动，第 3 轮才跳起来涨 10 个点。这种模式 1 iter 完全拿不到
- **ClinicalTrial 多次被拒**——候选程序在阈值边缘反复试探但都没跨过 ΔJ ≥ 10⁻⁵。这种"接近但不达标"的轨迹是优化过程里最让人抓狂的，作者没有回避

**注意：被拒的候选不是"没用"那么简单**——其实是"没用得不够"。这其实暴露了一个潜在问题——阈值 10⁻⁵ 可能太严了，很多有意义的改进被浪费了。但作者没讨论这个超参。

---

## Case Study：AutoIndex 学到了什么

### StackExchange：LaTeX 是检索噪声

![Figure 3: Case study](https://arxiv.org/html/2607.18603v1/images/loop_diagram.png)

*图 3：StackExchange 上一个 LaTeX-heavy 文章的 case study。Analysis Agent 标记了"80+ inline LaTeX tokens"作为问题，Code Agent 写了一个阈值门控的 `strip_latex` step，把 chunk 从 1847 tokens 砍到 1102 tokens。Top-3 检索从"math wiki、textbook、physics"（都不相关）翻转到"pricing、monopoly、supply"（全相关）。R@100 从 0.42 涨到 0.68（+62%），nDCG@10 从 0.11 涨到 0.24（+118%）。*

这个 case 很有说服力。注意几个细节：

1. **不是无脑 strip_latex**——是用 `has_heavy_latex(text)` 做阈值门控，避免对正常文章误伤
2. **BM25 的失败原因被直接命中**——LaTeX 标记（`\displaystyle`、`\frac{p}{q}`）占用了 chunk 空间、稀释了语义词的 TF-IDF 权重
3. **检索结果发生了质变**——不是从"第 5 名正确"挪到"第 1 名正确"，是从"全是数学公式页面"翻到"全是经济学/定价相关页面"

这个发现其实很工程化。**在 LaTeX 重的语料上做 BM25 检索，LaTeX 标记就是噪声**——我之前没意识到这点，但看完成果觉得理所当然。

### TipOfTongue：场景描述 vs 抽象摘要的失配

TipOfTongue 的查询是"我看过一部电影/书，开头是 xxx 场景，结尾是 xxx"，但 Wikipedia 文档是"本片讲述了一个关于 xxx 的故事"这种抽象摘要。Analysis Agent 发现了这个 mismatch，Code Agent **重写了 Plot 和 Cast 部分的权重**——不是删除其他部分，是**增加 query-relevant narrative content 的词频**，同时保留全文上下文。

这个改法挺巧妙的：**不是减法，是加法**。它意识到问题不在"chunk 太大"，而在"查询想要的叙事描述词被通用摘要稀释了"。

---

## 几个我比较在意的细节

### 1. 选 Recall@100 当目标函数

作者选 J = validation Recall@100 作为优化目标。这是个**务实但不完美**的选择：
- **务实**：Recall@100 评估便宜，对索引变化敏感
- **不完美**：nDCG@10 的提升是"附带"的，没有被直接优化。如果生产环境关心 top-5 质量，这套流程可能不够

### 2. 阈值 τ = 10⁻⁵

ΔJ ≥ 10⁻⁵ 才能被采纳。这个数字在 5 轮、每轮 4 候选的设置下，意味着总共 20 次尝试里只有**显著**提升的才能进。Figure 2 里 ClinicalTrial 多次"接近但被拒"就是这个阈值的副作用。

### 3. 固定 BM25 作为 retriever

整篇论文的 retriever 都是 BM25。**这既是最强论据（retriever 都没动，提升纯来自预处理），也是最大软肋（dense/ColBERT/reranker 上能不能 work 是 open question）**。作者只在 StackExchange 一个任务上做了 dense 迁移实验。

### 4. 算力账

Table 5（附录）给了 token 消耗：qwen3-coder 每跑平均 18K analysis tokens + 24K code tokens，LLM 调用 ~30 次/任务。**对一个 5 轮迭代的程序搜索来说，这个开销不便宜**——但比起微调一个 embedding 模型，仍然便宜得多。

---

## 我的判断

**这篇论文做对了什么**：
- **戳到了一个真实痛点**。RAG 系统的优化者花 90% 时间调 retriever 和 prompt，剩下 10% 调 chunking，然后以为问题在前面那 90%
- **方法学诚实**。消融把每个组件的作用拆得很清楚，承认 1 iter 不够、历史是稳定器而非增益器、Analysis 是关键信号
- **Case study 质量高**。Figure 3 的 strip_latex 不是 toy example，是真问题真解
- **代码开源**。框架可以直接拿来改

**这篇论文的局限**：
- **BM25 是单一验证场景**。dense retriever 上只有一个任务的迁移证据
- **8 个任务是异构但量都不大**。最大的 LQA 有 4569 个查询，但 PR 只有 53、TR 只有 51——这种小样本上的提升需要谨慎对待
- **没有和强 baseline 比**。比如和 late-chunking（Gunther et al. 2024）、LLM-based chunking（Duarte et al. 2024; Zhao et al. 2025）这些"也表示层优化"的方法直接对比。论文的 related work 提了它们，但实验里没碰
- **程序可迁移性没验证**。一个 corpus 学到的程序，能不能直接套到另一个 corpus？作者明确标了 future work

**对工程实践的启发**：
- 如果你做 RAG 的语料里有大量结构化噪声（LaTeX、HTML 标签、代码块、表格），**AutoIndex 思路值得手动试一下**——分析哪些 token 占据了 chunk 空间但对 BM25 无意义
- 如果你的 retriever 是 dense 的，**AutoIndex 的程序可能也能 work**（StackExchange 的 +18.3% 是个积极信号），但需要自己验证
- 这个框架最大的工程价值是**强制你用诊断驱动的方式思考文档表示**——即使你不用 LLM Agent，手动做一遍 Analysis Agent 的工作（分 Anchor/Recall Violation/Small-Margin Positive 检查）也能发现很多问题

**一句话总结**：AutoIndex 不是新工具，是新视角。它把"文档怎么变成 BM25 的输入"从工程经验问题变成了可计算优化问题。在 dense retriever 主导的今天，这个视角对 BM25 仍然有意义，对 sparse+lexical-aware 的混合系统可能更有意义。

---

## 收尾

这篇论文最让我意外的是 +43.6% 这个数字。在 2026 年看到 BM25 在一个标准 benchmark 上 nDCG@10 涨四成，本能反应是"是不是测错了"——但看完 case study 和消融，这个提升是真实的、可解释的、来自语料结构适配的。

AutoIndex 的哲学是：不要假设你的索引构造是最优的，让数据自己告诉你应该怎么构造。这个思路不只适用于检索——任何"输入 X → 固定管道 → 输出 Y"的系统，X 的预处理都值得被当作一等优化目标来对待。

如果你也在做 RAG，**下次调不动 retriever 的时候，先回头看看你的 chunk**。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
