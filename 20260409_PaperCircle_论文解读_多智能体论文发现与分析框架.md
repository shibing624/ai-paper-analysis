# 找论文这件事，是时候让 Agent 替你干了 -- Paper Circle: 多智能体学术发现与分析框架

你每天花多长时间找论文？

我说的不是那种"看到一篇引用顺藤摸瓜"的找法，而是"老板让你三天内做一份某个方向的 literature survey"那种。打开 arXiv 搜一遍，再去 Semantic Scholar 搜一遍，Google Scholar 再来一遍——结果发现三个平台返回的论文重合度不到 30%，每个都有对方没有的。去重、排序、导出 BibTeX、整理表格……说实话，这套流程每次做完我都觉得，这不该是人干的活。

Paper Circle 这篇论文就是冲着这个痛点来的。它搞了一套多智能体系统，把"找论文"和"读论文"两件事都自动化了，而且是 ACL 2026 主会 Oral。

坦率讲，我看完之后的感受是：工程完成度很高，系统设计有章法，但"多智能体"这个标签本身可能没有听起来那么性感。

---

## 核心摘要

Paper Circle 用 6 个协作 Agent 组成论文发现流水线（Discovery Pipeline），整合 arXiv、Semantic Scholar 等多源检索，再用 BM25 + 多准则排序实现 98.18% 的命中率。另一条分析流水线（Analysis Pipeline）把 PDF 转成带溯源的知识图谱（KG），支持问答和覆盖度检查。在 500 条 benchmark 查询上 MRR 达到 0.8824，平均 21 秒返回结果。真实用户测试中 NASA-TLX 认知负荷仅 1.2/7，可用性评分 7.6/10。这不是一个"概念验证"，而是一个跑通了全链路的开源系统。

---

## 论文信息

- **标题**: Paper Circle: An Open-source Multi-agent Research Discovery and Analysis Framework
- **作者**: Komal Kumar, Aman Chadha, Salman Khan, Fahad Shahbaz Khan, Hisham Cholakkal
- **机构**: Mohamed bin Zayed University of Artificial Intelligence (MBZUAI); AWS Generative AI Innovation Center, Amazon Web Services
- **发表**: ACL 2026 Main Conference (Oral)
- **链接**: https://arxiv.org/abs/2604.06170

---

## 问题动机：为什么单源搜索已经不够用了？

做过 survey 的人都知道一个残酷的事实：**没有任何单一论文搜索引擎能覆盖所有你需要的论文。**

Paper Circle 在实际使用中统计了 21,115 篇论文的来源覆盖率，结论相当扎心：

| 来源 | 检索缺失率 |
|------|-----------|
| arXiv | 70.9% |
| Semantic Scholar | 80.4% |
| Google Scholar | 36.9% |
| **Paper Circle**（多源融合） | **9.0%** |

也就是说，如果你只用 Semantic Scholar 搜，有 80% 的相关论文你根本看不到。arXiv 好一点但也漏了 70%。即使用 Google Scholar 也有将近 37% 的漏网之鱼。

这个数据我觉得还是有说服力的。虽然"缺失率"的定义可能有一些争议（后面批判性分析里会聊），但方向是对的——多源融合确实是刚需。

除了覆盖率问题，还有一个更现实的痛点：**检索完了还得人肉去重、排序、导出。** Paper Circle 的 Discovery Pipeline 自动去除了 43.5% 的重复论文，直接输出 JSON/CSV/BibTeX/Markdown/HTML 五种格式。

---

## 系统架构：两条流水线，各司其职

Paper Circle 的整体架构分为两个独立的子系统。

![图1: Paper Circle 系统架构总览，展示了 Discovery Orchestrator 和 Paper Mind Orchestrator 两条主线，底层由 Query Agent、Search Agent、Sorting Agent、Analysis Agent、Export Agent 五个核心 Agent 驱动，通过 Multi-agent Code Orchestrator Tracker 统一调度](https://arxiv.org/html/2604.06170v1/x3.png)

*图1: 系统架构总览 -- 上半部分是 Discovery（发现）和 Paper Mind（分析）两个编排器，下半部分是五个核心 Agent 和它们共享的工具层*

整个系统的设计哲学其实挺清晰的：**Discovery Pipeline 负责"找到对的论文"，Analysis Pipeline 负责"读懂一篇论文"**。两条线可以独立使用，也可以串联。

---

## Discovery Pipeline：用"去噪扩散"的思路找论文

这部分的设计我觉得是全文最有意思的地方。

![图2: Discovery Pipeline 的工作流程，借鉴了扩散模型的"噪声-去噪"思想。从一个空白草稿（Empty Draft）出发，经过多轮 Agent Step 的迭代"去噪"，逐步从 Noised Draft 收敛到 Denoised Draft，最终生成 Final Draft](https://arxiv.org/html/2604.06170v1/x4.png)

*图2: Discovery Pipeline 的核心流程 -- 一个 Orchestrator Agent 在底层调度 Query Decomposition、Paper Search、Ranking & Sorting、Analysis & Insights、Export 五个子 Agent*

作者把论文发现过程类比为"去噪扩散"（Denoising Diffusion）：一开始你的检索结果是一团"噪声"（不相关的论文、重复的论文、排序混乱的论文），每一轮 Agent 操作都是在做"去噪"，逐步收敛到你真正想要的结果。

这个类比有点花哨，但底层逻辑是实在的。具体来说，五个 Agent 分工如下：

**Query Agent** 解析用户自然语言查询，拆解成结构化检索参数（关键词、时间范围、会议筛选等）。

**Search Agent** 同时查 arXiv、Semantic Scholar 等多个源，做跨源去重。去重不是简单比标题，而是用 DOI、arXiv ID 等唯一标识做精确匹配，再加模糊标题匹配兜底。

**Sorting Agent** 是排序的核心，用多准则加权打分。BM25 做文本相关性，再叠加时间衰减、引用数归一化、来源权重等维度。还有一个 diversity-aware 的重排，防止前 K 个结果全是同一个方向的。

**Analysis Agent** 对检索结果做统计分析——发表趋势、作者分布、关键词共现之类的。

**Export Agent** 生成同步输出：JSON、CSV、BibTeX、Markdown、HTML，附带完整的检索过程日志。

说实话，每个单独的组件都不新鲜。BM25 是信息检索的老家伙了，多源融合也有人做过。但 Paper Circle 的工程价值在于：**它把这些东西完整地串起来了，而且每一步都有日志、可复现。** 这在学术工具里其实挺少见的。

---

## Analysis Pipeline：把论文变成知识图谱

另一条线就是论文深度分析。给它一篇 PDF，它帮你拆解成结构化的知识图谱。

![图3: Analysis Pipeline 的层级架构。Paper Analysis Orchestrator 统一调度 Concept Extractor、Method Extractor、Experiment Extractor、Linkage Agent 四个专业 Agent，底层共享 Ingestion（PDF解析）、Chunking（段落/图表/公式分块）、Graph Builder（概念/方法/实验/链接节点构建）、QnA（RAG问答/定位/导出/验证）四个基础模块](https://arxiv.org/html/2604.06170v1/x5.png)

*图3: Analysis Pipeline 架构 -- 四个专业化的 Extractor Agent 分别负责提取概念、方法、实验和关联关系*

这个设计的特点是**类型化知识图谱**（Typed Knowledge Graph）。节点不是泛泛的"实体"，而是分为：

| 节点类型 | 说明 | 示例 |
|---------|------|------|
| Concept | 论文涉及的核心概念 | "Attention Mechanism", "BM25" |
| Method | 提出或使用的方法 | "Diversity-aware Reranking" |
| Experiment | 实验设置和结果 | "50-query benchmark, MRR=0.627" |
| Dataset | 使用的数据集 | "ICLR 2024 papers" |
| Figure | 论文中的图表 | "Figure 3: Architecture" |

每个节点和边都带有**溯源信息**（Provenance），可以追溯到论文的具体段落和页码。这个特性在 Table 1 的系统对比中，是 Paper Circle 独有的——PaperQA、PaperQA2、STORM 都不支持。

![图4: 完整的 Analysis Pipeline 细节流程。Orchestrator Agent 接收 PDF/URL 输入后，先调用 PDF Processor 处理，再分发到 Deep Analyzer、Critic Agent、Literature Expert Agent、Knowledge Graph Agent 等多个分析智能体，每个 Agent 下面挂载 Contribution Analyzer、Reproducibility Checker、Summarizer 等子 Agent，最终汇总生成 Final Report](https://arxiv.org/html/2604.06170v1/x6.png)

*图4: Analysis Pipeline 的更详细视图 -- 包含 Deep Analyzer、Critic Agent、Literature Expert Agent、Knowledge Graph Agent 等分工明确的分析智能体*

分析完之后，用户可以在前端直接看知识图谱、按类型筛选节点、提问题：

![图5: Paper Circle 的分析前端界面。左侧（A区域）展示论文的 Mindmap 思维导图可视化，中间部分（C区域）是问答界面，用户提问后系统基于知识图谱返回答案并引用对应的 Figure。右侧（B/D区域）展示提取的 Concepts 列表和详细描述。底部（E区域）提供 Flowchart、Concepts、Methods、Experiments、Interactive Graph 等多种视图切换](https://arxiv.org/html/2604.06170v1/x7.png)

*图5: 分析前端 -- 支持思维导图、概念列表、问答、多视图切换，看起来完成度确实不错*

---

## 跟现有工具比，Paper Circle 到底强在哪？

Table 1 做了一个比较全面的横向对比：

| 能力 | Paper Circle | PaperQA | PaperQA2 | STORM | SciSage | alphaXiv |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| 多智能体编排 | Y | -- | -- | Y | Y | -- |
| 多源检索 | Y | Y | Y | -- | ~ | ~ |
| 类型化知识图谱 | Y | -- | -- | ~ | Y | -- |
| 节点/边溯源 | Y | -- | -- | -- | ~ | -- |
| 覆盖度验证 | Y | -- | -- | -- | ~ | -- |
| 图谱感知QA | Y | Y | Y | -- | Y | -- |
| 确定性运行 | Y | -- | -- | ~ | ~ | -- |
| 结构化导出 | Y | ~ | ~ | Y | ~ | -- |

Paper Circle 的差异化主要在三个点：(1) 多源融合 + 去重；(2) 带溯源的类型化知识图谱；(3) 确定性运行和完整的过程日志。

不过我要说，PaperQA2 在单论文问答的准确率上其实相当厉害，它用的是 RAG + 引用验证的范式，跟 Paper Circle 的定位不太一样。Paper Circle 更偏"发现"（找到论文），PaperQA2 更偏"理解"（读懂论文）。两者其实可以互补。

---

## 实验结果：数据说话

### 主实验：50 条语义查询 benchmark

作者用 312 篇来自 ICLR/NeurIPS/ICML/CVPR 等会议的论文建了一个本地语料库，然后用 50 条自然语言查询做检索评测。

| 模型 | 类型 | Hit Rate | MRR | R@1 | Time(s) |
|------|------|----------|------|------|---------|
| **Qwen3-Coder-30B-Q3KM** | Agent | **0.80** | **0.627** | **0.58** | 22.2 |
| qwen3-coder:30b | Agent | 0.80 | 0.518 | 0.46 | 21.1 |
| **BM25 Baseline** | Baseline | **0.78** | **0.541** | **0.48** | -- |
| deepseek-coder-v3:16b | Agent | 0.66 | 0.396 | 0.32 | 47.9 |
| Semantic Baseline | Baseline | 0.54 | 0.279 | 0.22 | -- |

看到这个数据，我的第一反应是：**BM25 baseline 也太能打了吧？**

Hit Rate 0.78 vs Agent 的 0.80，MRR 0.541 vs 0.627——差距有，但远没到"Agent 碾压传统方法"的程度。而且 BM25 不需要 GPU，不需要等 22 秒，这个性价比……

更有意思的是，小模型（qwen2.5-coder:3b/7b）的 Agent 表现甚至不如 BM25。这说明 Agent 框架本身的价值很大程度上取决于底层 LLM 的能力。如果 LLM 不够强，加了 Agent 架构反而引入了更多的出错环节。

### 扩展实验：500 条查询

到了 500 条查询的大规模测试，数据就好看多了：

| 配置 | Hit Rate | MRR | R@1 | Time(s) |
|------|----------|------|------|---------|
| Default（完整 Agent） | **0.9818** | **0.8824** | **0.8381** | 21.54 |
| 带过滤 + 离线 | 0.9600 | 0.8485 | 0.7800 | 22.76 |
| 纯离线 | 0.9200 | 0.6476 | 0.5600 | 41.45 |
| 无 mention | 0.6400 | 0.4316 | 0.3600 | 38.35 |

98.18% 的命中率和 0.88 的 MRR——这个数确实很能打。但这里有个细节值得注意：500 条查询是用什么标准构造的？如果查询本身就是从语料库里的论文标题改写来的，那高命中率多少有点"开卷考试"的味道。

### 检索消融实验

Table 7 的消融实验揭示了一个有趣的现象：

| 配置 | Hit Rate | MRR | Time(s) |
|------|----------|------|---------|
| BM25 Full | 0.9600 | 0.8629 | 33.75 |
| BM25 + Reranker | 0.9600 | **0.8692** | **935.07** |
| Semantic Full | 0.9400 | 0.7097 | 31.28 |

BM25 + Reranker 的 MRR 只比纯 BM25 高了 0.006（0.8692 vs 0.8629），但耗时从 34 秒飙到了 935 秒——**28 倍的时间换来 0.7% 的提升**。这波投入产出比属实不太划算。

而纯语义检索（Semantic）反而比 BM25 低了 15 个点的 MRR。在学术论文检索这个场景下，BM25 这种精确匹配的方法依然很有竞争力，因为论文标题、摘要里的关键术语本身就是高质量的检索锚点。

### 论文评审预测：老实说效果不行

Paper Circle 还尝试了一个有意思的任务——用 LLM 预测论文评审分数。在 50 篇 ICLR 2024 论文上的结果：

![图6: 论文评审预测的四组对比图。(A) 平均MSE：GPT-120B为1.42，Qwen-30B-Q3为3.79，GPT-20B为1.34，Qwen-30B为3.44；(B) 平均相关系数：所有模型的Pearson和Spearman相关系数都接近0甚至为负（最高仅0.09）；(C) 不同容错阈值下的准确率：GPT-120B和GPT-20B在+-1.5阈值下可达约95%，但Qwen模型仅约65%；(D) 模型成功率：GPT-120B为96%，GPT-20B为84%，Qwen模型为76%](https://arxiv.org/html/2604.06170v1/figures/simple_dashboard.png)

*图6: 论文评审分数预测结果 -- 相关系数接近0，说明 LLM 目前还不能可靠地排序论文质量*

图 6(B) 里的数据很说明问题：所有模型的 Pearson/Spearman 相关系数都在 -0.15 到 0.09 之间徘徊。这意味着**LLM 给的评审分数跟人类评审的排序基本没有相关性**。

GPT-120B 的 MAE 是 1.42（满分通常是 10 分制），看绝对误差还行，但排序能力约等于随机。作者很坦诚地承认了这一点，我觉得这个诚实度值得点赞——很多论文会选择不报这种"难看"的结果。

---

## 真实用户反馈

81 个用户 session，横跨 9 个研究方向，处理了 21,115 篇论文。关键可用性指标：

- **NASA-TLX 认知负荷**: 1.2/7（几乎无感）
- **SUS 积极评分**: 7.6/10
- **SUS 消极评分**: 2.6/10
- **可学习性**: 8/10
- **中位运行时间**: 2.3 分钟

这组数据说明系统的上手门槛确实低。但 81 个 session 的样本量偏小，而且不知道这些用户是不是团队内部的——如果是的话，评分可能偏乐观。

---

## 数据库设计：一个被忽略的亮点

![图7: Paper Circle 的数据库 ER 图，包含 users、papers、communities、paper_analysis、paper_engagement、paper_discussions、community_papers 七张表。paper_analysis 表存储了知识图谱的 Nodes & Edges（JSON格式）以及 Markdown Summary、Mermaid 思维导图和流程图](https://arxiv.org/html/2604.06170v1/figures/analysis_frontend.png)

*图7: 数据库关系图 -- 支持社区（Communities）、论文互动（Engagement）、讨论（Discussions）等社交功能*

这张 ER 图其实透露了不少信息。Paper Circle 不只是一个检索工具，它还内置了社区功能——用户可以创建 Community（比如"NLP 2026"），把论文加进去，还能做讨论和互动（like、view、save）。paper_analysis 表用 JSON 直接存知识图谱的 Nodes & Edges，简单粗暴但够用。

---

## 我的判断：工程价值大于学术贡献

说实话，读完整篇论文之后，我的感受是**这是一个工程质量很高的系统论文，但"多智能体"这个卖点被包装得有点过了**。

**亮点**：

1. **多源融合的覆盖率提升是实打实的**。从单源 20-64% 的覆盖率提升到 91%，这个数据对做 survey 的人来说有真实价值。
2. **系统完成度很高**。从检索到分析到前端到导出，全链路跑通了，而且开源。这在学术工具类论文里属于上乘。
3. **诚实地报告了论文评审预测的失败**。相关系数接近 0 这个结果，很多人会选择不放出来。

**问题**：

1. **BM25 baseline 的尴尬**。在 50 条查询的实验里，BM25 跟最好的 Agent 差距很小（MRR 0.541 vs 0.627）。这让人怀疑：Agent 架构带来的增益，到底是来自"多智能体协作"，还是仅仅来自"在线搜索补充了更多论文"？
2. **Benchmark 的公正性存疑**。500 条查询是怎么构造的？如果是从已有论文生成的，那高命中率可能只是在测"信息检索"而不是"论文发现"。真实场景下，用户的查询往往模糊、不完整，这种 benchmark 可能过于理想化。
3. **"多智能体"的必要性没有充分论证**。消融实验里，去掉 Intent Agent（no_intent）后 MRR 只从 0.8629 降到 0.8554，去掉 Sorting 的单独排序步骤后也几乎没影响。这说明很多 Agent 的边际贡献并不大。
4. **论文评审预测这部分有点"凑"**。跟主系统的论文发现和分析定位不太一致，而且结果确实不好看。

**对工程实践的启发**：

如果你正在做学术检索工具或者 AI 辅助研究的产品，Paper Circle 有几个设计值得借鉴：
- 多源融合 + 跨源去重是必做的，单源覆盖率太低了
- BM25 在学术论文检索场景下依然是超强 baseline，不要一上来就 all-in 语义检索
- Reranker 的时间开销要认真评估，28 倍耗时换 0.7% 提升大概率不值得
- 知识图谱的溯源（Provenance）对用户信任度很关键

---

## 跟同期工作的定位对比

| 工具 | 核心定位 | 优势 | 局限 |
|------|---------|------|------|
| **Paper Circle** | 多源发现 + 知识图谱分析 | 覆盖率高，全链路，开源 | Agent增益有限 |
| **PaperQA2** | 单论文精准问答 | RAG + 引用验证，问答准确 | 不做发现，单源 |
| **STORM** | 综述生成 | 自动写 Wikipedia 式综述 | 不做结构化分析 |
| **SciSage** | 知识图谱 + QA | 图谱能力强 | 社区生态弱 |

Paper Circle 填了一个"多源发现 + 图谱分析"的空位。跟 PaperQA2 比，它更偏"找论文"；跟 STORM 比，它更偏"结构化理解"而非"自由文本生成"。

---

## 收尾

Paper Circle 这篇论文给我最大的收获，其实不是"多智能体"本身，而是那组覆盖率数据——arXiv 漏了 70.9%，Semantic Scholar 漏了 80.4%。这两个数字让我重新审视了自己日常找论文的方式。

至于"用 Agent 做论文检索"这个方向，我的判断是：**核心价值不在 Agent 架构本身，而在数据源整合和智能排序。** Agent 只是一种组织代码的方式，真正让用户受益的是"一次搜八个源然后帮你去重排序"。如果你把同样的逻辑写成一个普通的 Python 脚本而不叫它"Agent"，效果不会差太多。

但话说回来，系统开源了，用起来也方便——如果你每周都要跟踪某个方向的新论文，试试 Paper Circle 还是值得的。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我
