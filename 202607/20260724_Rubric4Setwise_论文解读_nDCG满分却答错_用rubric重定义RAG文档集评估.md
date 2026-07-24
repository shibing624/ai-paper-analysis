# nDCG@5=100%却答错题目？用 rubric 重定义 RAG 文档集评估

> 一篇把"文档集质量"从直觉推到工程中心的工作。读完你会发现，过去几年我们一直在优化一个其实跑偏了的指标。

## 核心摘要

RAG 时代，生成质量的天花板不再由模型决定，而是由塞进 context window 的那堆文档决定。但行业里用了二十年的 nDCG 还在按"单文档相关性"打分——这相当于让一个不能识别矛盾的裁判去给一堆互相打架的证据打分。

这篇来自腾讯元宝 + 中科大的论文（[arXiv:2607.19747](https://arxiv.org/abs/2607.19747)）做了三件事：

1. **SetwiseEvalKit**：把"文档集质量"拆成 3 个层级、9 个维度（Doc-Level: 相关性/真实性/质量；Set-Level: 互补性/冗余/冲突；Global-Level: 完整性/密度/可达性），基于 28K 高质量 query-specific rubric 评估。诊断价值是 nDCG 永远给不了的——它能告诉你的检索器为什么差、差在哪、怎么修。
2. **系统性体检 12 个 reranker**：结果是惊人的——连最强的 SetR 也只有 45.85% 的覆盖率，而且短文本（SetR/Rank4Gen 赢）和长文本场景（ReasonRank/Rearank 赢）**没有同一个方法能通吃**。这意味着每个现有方案都有结构性盲区。
3. **Rubric4Setwise**：把评估 rubric 直接当成选择信号，无训练地用 Qwen3-8B 做 rubric-guided 集合选择。短文本 EM 比 Vanilla 高 7.47 个点、F1 高 8.22 个点，长文本 LLM-judge 比 ReasonRank 高 2.42 个点；而且用的文档更少（短文 2.66 vs 固定 5 篇；长文 20.52 篇/4.52 轮 vs SetR 29.23 篇/4.73 轮）。

我的判断：**这是一篇"工作框架型"的论文，不是某个 SOTA 的微小 trick**。它的真正价值不在 Rubric4Setwise 本身（坦白说，oracle 假设下用 rubric 引导选择，效果好是应该的），而在 SetwiseEvalKit 把"评估"和"优化"在同一个 rubric 空间里打通——这给整个 setwise reranking 领域提供了一个可以量化进步的方向。

---

## 论文信息

- **标题**：Beyond Relevance-Centric Retrieval: Rubric-Oriented Document Set Selection and Ranking
- **arXiv**：[2607.19747](https://arxiv.org/abs/2607.19747) （2026-07-22 v1，2026-07-23 v2）
- **作者**：Kailin Jiang¹²*、Lei Liu¹†、Jian Xi²、Hui Xu²、Junlin Liu³、Baochen Fu⁴、Bin Li¹、Vichwang²、Yu Lu²、Haibo Shi²
- **机构**：¹中国科学技术大学　²腾讯元宝团队（Yuanbao Team, Tencent）　³中国科学院大学　⁴山东大学
- **项目页**：https://rubric4setwise.github.io
- **代码**：https://github.com/Rubric4Setwise/Rubric4Setwise
- **数据集**：https://huggingface.co/datasets/kailinjiang/SetwiseEvalKit

> *第一作者是在腾讯元宝实习期间完成的工作。†通信作者是中科大刘磊教授。

---

## 痛点：nDCG 看不到的"集合级缺陷"

先抛个具体的例子——这是论文 Figure 1 里的场景：

> Query: 在 2015 年美国小姐（Miss USA 2015）颁奖典礼上，Olivia Jordan 是被谁戴上后冠的？这人本身也是 2014 年美国小姐冠军、电视主持、模特、跆拳道教练。
> Answer: **Nia Sanchez**。

一个传统 reranker 选了 5 篇文档，按相关性排序，nDCG@5 = 100%，**完美得分**。看起来完美，对吧？图 1 告诉你真相：

![Figure 1: 传统评估的盲区。即使 nDCG@5=100%，文档集仍存在 4 个结构缺陷：信息冲突、冗余、关键信息缺失](https://arxiv.org/html/2607.19747v2/figures/motivation.png)

*图 1：传统评估范式对文档集结构性缺陷完全无感。nDCG@5=100% 的"完美"集合，实际上有 4 处致命缺陷。*

具体说：

- **Doc [11] 信息冲突**：把 Nia Sanchez 说成"French pageant winner"，但她实际是内华达州（American）小姐——直接把生成器带偏。
- **Doc [11] & [12] 信息冗余**：两篇都在讲"第 64 届美国小姐颁奖典礼"那点事，重叠段落挤占 context。
- **所有文档都缺 Miss USA 2014 冠军信息**——查询里另一个关键人物身份完全没出现在文档集里。
- **Doc [8] & [5] 信息冗余**：又一对重复。

后果就是图 1 底部的连锁反应：

> **低信息增益 → 更多搜索轮次 → 浪费 context window**

看到这个例子你就能理解为什么 nDCG 在 RAG 时代越来越力不从心：它假设"集合质量 = 单文档质量之和"，把"五个相关文档"和"五个互补文档"打成同样的分。但前者可能让 LLM 答错题，后者才是真正有用的。

我之前在调 RAG 的时候也撞过这堵墙。nDCG 涨了，nDCG@5 接近满分，模型回答质量却肉眼可见地变差。当时我们怀疑是 reranker 选了一堆同质化的"高相关"文档，把 LLM 困在信息茧房里。看完这篇论文，我觉得这个猜测基本坐实了。

---

## 框架总览：3 级 9 维 + 28K Rubric

论文给的解法是搭一个完整的 **evaluate-diagnose-optimize** 闭环。先看 SetwiseEvalKit 整体长什么样（Figure 2）：

![Figure 2: SetwiseEvalKit 整体框架。❶ 三级九维 rubric 分类法；❷ 评估流程：候选文档→Reranker→选中集合→Judger→Rubric Coverage Score](https://arxiv.org/html/2607.19747v2/figures/setwiseevalkit.png)

*图 2：SetwiseEvalKit 评估框架。上半部分定义三级九维 rubric，下半部分是 Reranker→Judger 的打分流程。*

### 9 个维度的设计逻辑

| 层级 | 维度 | 含义 | 为什么需要 |
|---|---|---|---|
| **Doc-Level** | Relevance 相关性 | 文档是否真正回答查询（不靠表面关键词） | 替代 nDCG 的相关性 |
| | Authenticity 真实性 | 文档陈述是否与参考答案一致 | 检测"自信地说错话"的文档 |
| | Quality 质量 | 文档结构是否清晰，能否直接抽信息 | 拒绝"信息正确但藏在一堆废话里"的文档 |
| **Set-Level** | Complementarity 互补性 | 集合是否覆盖了所有关键信息元素 | nDCG 完全看不到这一层 |
| | Redundancy 冗余性 | 是否有重复 | nDCG 默认无关，实际挤占 context |
| | Conflict 冲突性 | 不同文档是否给出矛盾陈述 | 直接导致 LLM 答错 |
| **Global-Level** | Completeness 完整性 | 集合能否完整回答查询 | 集合级天花板 |
| | Density 密度 | 有用内容是不是占了文档主体 | 排除"99% 是模板废话、1% 是答案" |
| | Reachability 可达性 | 没有任何外部知识的模型能否仅凭此集合推出正确答案 | 终极检验标准 |

这个分层有个很聪明的地方：**三级分别对应"个体—关系—整体"的认知视角**，从最细粒度（单文档）逐步过渡到最粗粒度（集合对 LLM 的整体效用）。这跟人类评审一份资料包时的思考顺序完全一致——先看单份材料质量，再看材料之间是否互补、有没有重复打架，最后看整体能不能拼出答案。

### Rubric 是怎么生成的？

**Hybrid Rubrics Generation**——多模型生成 + 聚合：

1. **两个前沿 LLM**（GPT 5.1 和 Gemini 3.1-Pro-Preview）独立给同一个 (query, reference answer) 生成候选 rubric。
2. **DeepSeek-V4 Pro** 做聚合：去重、过滤低质量。
3. 每个 rubric 必须是**针对该 query 的具体实体、事实、数值**——禁止"相关的内容"这种含糊措辞。

产出量：短文本 ~24K，长文本 ~4K，合计约 28K。

论文也做了 human study（PhD 级专家）验证 rubric 质量：聚合后大约 **70% 被判为 High/Critical discriminative**（GPT 5.1 单独只有 37%，Gemini 3.1 单独只有 43%），浪费率仅约 8%。这就是说，**多模型聚合不是简单的"取并集"，而是真的把低质量 rubric 筛掉了**。

### 两个评估场景

- **Short-form**：多跳 QA 场景。从 HotpotQA、2WikiMultihopQA、MuSiQue、Bamboogle 采样 2061 个 query，BM25 拉 20 篇候选，每个 reranker 选 top-5（setwise 类自适应选数量）。
- **Long-form**：长答案研究查询。从 ResearchQA 选 200 个 query，用开源搜索 agent **DR.Tulu-8B**（后面接 Google Search API）跑多轮搜索轨迹，所有 reranker 通过 MCP 协议接进去，每轮 retrieval 后做 rerank。

---

## 实验一：12 个 Reranker 的"集体体检"

先看 **SetwiseEvalKit 是否真的能预测下游生成质量**——这一步如果不过，整个 benchmark 都没意义。

Figure 4 把 8 个 reranker 的 Overall Coverage Score 和下游生成平均分画在一张散点图上：

![Figure 4: Overall Coverage Score 与下游生成质量的强相关（Pearson r=0.92, p=0.0013）](https://arxiv.org/html/2607.19747v2/figures/coverage_vs_answer_consistency.png)

*图 4：8 个 reranker 的 Overall Coverage Score 与下游生成质量（Downstream Avg）的散点图。Pearson r=0.92, p=0.0013——评估分数真的能预测生成质量。*

**0.92 的 Pearson 相关系数，p=0.0013**。这个数是相当能打的——基本上"coverage 高的方法，下游也好"是稳的。SetR (8B) 在右上角一骑绝尘，Vanilla BM25 在左下角垫底，跟我们直觉完全一致。

benchmark 站得住脚了，那 12 个 reranker 的体检结果呢？先看短文本主表（Table 2 简化版）：

| 方法 | Doc-Level Avg | Set-Level Avg | Global-Level Avg | **Overall** |
|---|---:|---:|---:|---:|
| Only Retrieval (BM25) | 16.30 | 60.91 | 32.08 | 36.43 |
| BGE-Reranker-L (550M) | 20.30 | 63.05 | 38.34 | 40.57 |
| MonoT5 (3B) | 19.90 | 62.76 | 38.17 | 40.28 |
| RankLlama (7B) | 19.63 | 63.08 | 37.38 | 40.03 |
| Setwise (7B) | 19.52 | 63.17 | 37.51 | 40.07 |
| Rank1 (7B) | 19.86 | 62.93 | 37.75 | 40.18 |
| ReasonRank (7B) | 19.23 | 64.99 | 38.82 | 41.01 |
| Rank4Gen (8B) | 25.85 | 67.65 | 38.12 | 43.87 |
| **SetR 8B** | **33.44** | 64.04 | **40.08** | **45.85** |

先说一个扎心的数字：**最强方法 SetR 也只有 45.85% 覆盖率**。我们以为 RAG 检索已经很成熟了，结果摆在"集合质量"这个尺子下，超过一半的维度都没做好。

更值得玩味的是**三个层级的分化**：

- **Doc-Level**：SetR (33.44) 比第二名 Rank4Gen (25.85) 高了快 8 个点，Adhoc/Reasoning 类方法都在 19-20 区间徘徊——**SetR 在单文档质量上有碾压级优势**。
- **Set-Level**：所有方法差距很小（60-68），**跨文档协调是普遍弱项**。
- **Global-Level**：ReasonRank 反而追上来了（38.82 vs SetR 40.08），说明 reasoning 能力在"整体效用"上更管用。

长文本主表（Table 3 简化版）：

| 方法 | Doc-Level Avg | Set-Level Avg | Global-Level Avg | **Overall** |
|---|---:|---:|---:|---:|
| Only Search (Google) | 19.20 | 60.02 | 23.41 | 30.20 |
| BGE-Reranker-L (550M) | 21.07 | 57.71 | 25.20 | 32.26 |
| ReasonRank (7B) | 22.01 | **60.77** | **26.83** | **33.05** |
| Rearank (7B) | 21.77 | 59.96 | 25.36 | 32.78 |
| Rank4Gen (8B) | **23.05** | 57.60 | 23.41 | 31.97 |
| SetR (8B) | 20.58 | 59.04 | 24.06 | 31.83 |

**这是真正让我眉头一挑的地方**：短文本里 SetR/Rank4Gen 横扫，**到长文本场景 ReasonRank/Rearank 反而反超了**。

论文把这个现象总结成"❼ 缺乏跨场景通用方法"——setwise 类在单轮检索里赢，但多轮累积场景下 reasoning 能力更重要。**没有单一方法在两个场景都最强**。这其实是一个挺扎心的结论：我们以为 setwise reranking 是终点，但实际它只在"短问短答"里够用。

还有一点：**长文本方法间 Overall 差距只有 3 个点左右**（30.20 到 33.05），短文本是 9 个点（36.43 到 45.85）。论文的解读是"ranker 和 agent 当前是隔离的，累积 context 没有信息回流到文档选择"——这其实暗示了接下来要突破的方向是 **ranker-agent 协同优化**。

### 跨文档协调是"集体短板"

Figure 11 把 9 个维度分别对下游生成质量做 Pearson 相关性分析，揭示了一个非常一致的规律：

![Figure 11: 9 个 rubric 维度分别与下游生成质量的 Pearson 相关性。Doc-Level 三维度都显著（r=0.88-0.94），Set-Level 三维度都不显著（p>0.09），Global-Level 维度表现中等](https://arxiv.org/html/2607.19747v2/figures/dims_vs_answer.png)

*图 11：9 个维度分数与下游生成质量的相关性散点图。Doc-Level 三个维度都强相关且显著（r=0.88-0.94），但 Set-Level 三个维度都不显著（p=0.093-0.590）——这印证了"跨文档协调是普遍盲区"。*

具体数字：

| 层级 | 维度 | Pearson r | p 值 | 显著？ |
|---|---|---:|---:|:---:|
| Doc | Relevance | 0.94 | 0.000 | ✅ |
| Doc | Authenticity | 0.88 | 0.004 | ✅ |
| Doc | Quality | 0.90 | 0.002 | ✅ |
| Set | Complementarity | 0.51 | 0.193 | ❌ |
| Set | Redundancy | 0.63 | 0.093 | ❌ |
| Set | Conflict | 0.23 | 0.590 | ❌ |
| Global | Completeness | 0.52 | 0.191 | ❌ |
| Global | Density | 0.83 | 0.010 | ✅ |
| Global | Reachability | 0.43 | 0.292 | ❌ |

Set-Level 三个维度全军覆没。这意味着**现有 reranker 都没真正在做集合级优化**——它们还在用"五个高相关文档"代替"五个互补文档"。Conflict 维度的 r 只有 0.23，几乎没相关性，论文解释为"短文本多跳 QA 里事实冲突不是主要矛盾"（所有方法 Conflict 分都在 90+），但**互补性和冗余性真的做得很差**。

### Long-form 场景的"轮次衰减"

Figure 5 展示了长文本场景下每轮 Reranker 选出来的集合的 coverage score 变化：

![Figure 5: Long-form 场景下，每个 Reranker 在 5 轮搜索过程中的 Coverage Score 变化。整体呈下降趋势，因为后续轮次从"广撒网"转向"补缺口"](https://arxiv.org/html/2607.19747v2/figures/long_form_round.png)

*图 5：长文本场景下，6 个 Reranker 在 5 轮搜索中的 Coverage Score 变化。整体衰减是因为早期轮次是广撒网、后期轮次转向针对性补缺，候选池自然变窄。*

这个图其实说明了长文本检索的一个固有难题：**轮次越多、每轮的候选集越窄**（因为是要补特定缺口），所以分数下降是结构性的，不是方法在退化。论文也指出，他们最终用"跨轮次去重并集打分"来评估——测的是 agent 累计交付给 LLM 的那个集合，不是某一轮的瞬时结果。

---

## 实验二：Rubric4Setwise —— 把评估信号直接当选择信号

这一节是论文真正的"杀招"。前面的诊断是"现有方法都不行"，但凭什么**评估 rubric 能反过来当选择信号**？

论文的论证很直接：

> 如果 rubrics 能量化集合质量，那它们就同样能指导集合选择。

公式化就是：

$$S^* = \arg\max_{S \subseteq \mathcal{C}} f(S; q, \mathcal{R})$$

其中 $f(\cdot)$ 是 rubric-based set utility function，$\mathcal{R}$ 是 query-specific rubrics。**关键差异**：传统方法固定选 top-k，Rubric4Setwise 让集合大小**由 rubric 满足度自适应决定**。

实现上用 Qwen3-8B 做 chain-of-thought 推理，**完全无训练**。看下游生成结果（Table 4）：

| 方法 | 短文 EM | 短文 F1 | 短文 #Psg | 长文 LLM-judge | 长文 #Psg | 长文 Rounds |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla (No Rerank) | 18.63 | 21.10 | 5 | 67.08 | 20.94 | 4.49 |
| BGE-Reranker-L | 19.94 | 22.27 | 5 | 69.48 | 20.84 | 4.50 |
| MonoT5 (3B) | 20.23 | 23.20 | 5 | 70.33 | 20.42 | 4.31 |
| RankLlama (7B) | 20.28 | 23.00 | 5 | 69.10 | 22.24 | 4.78 |
| ReasonRank (7B) | 20.14 | 22.75 | 5 | 68.36 | 20.30 | 4.30 |
| Rearank (7B) | 21.11 | 24.25 | 5 | 68.43 | 21.39 | 4.63 |
| SetR (8B) | 25.13 | 28.70 | 2.75 | 70.54 | 29.23 | 4.73 |
| Rank4Gen (8B) | 25.08 | 28.65 | 3.05 | 68.32 | 12.13 | 5.01 |
| **Rubric4Setwise 8B** | **26.10** | **29.32** | **2.66** | **70.57** | **20.52** | **4.52** |

**短文本**：Rubric4Setwise EM 26.10，比 SetR (25.13) 高 0.97 个点，比 Vanilla 高 7.47 个点。F1 29.32，比 SetR 高 0.62 个点，比 Vanilla 高 8.22 个点。**用的是 2.66 篇文档，比固定 5 篇少了一半，但效果更好**——这就是"互补性选对，少而精"的实证。

**长文本**：LLM-judge 70.57，比 SetR (70.54) 高 0.03（基本打平），但比 ReasonRank (68.36) 高 2.21 个点、比 Rearank (68.43) 高 2.14 个点。**用 20.52 篇文档 vs SetR 29.23 篇、4.52 轮 vs 4.73 轮**——少 30% 文档、少 4% 轮次，达到同等或更好效果。

**关键意义**：Rubric4Setwise 是 9 个方法里**唯一在两个场景都 SOTA** 的。短文本场景它的对手是 setwise 类（R1/Rearank 这种 reasoning 类排第二集团），长文本场景它的对手是 reasoning 类（setwise 类掉到第二集团）。**Rubric4Setwise 横跨两场景都站住了**——这正好解决了论文自己诊断出的"❼ 缺乏通用方法"问题。

---

## 案例：rubric 怎么精确诊断集合缺陷

Figure 6 是个值得细看的诊断案例——把文章开头那个 Miss USA 2015 的例子用 9 个 rubric 维度全维度分析：

![Figure 6: 案例研究。9 个 rubric 维度对同一组文档分别打分，精准定位每个维度的缺陷](https://arxiv.org/html/2607.19747v2/figures/case.png)

*图 6：9 个 rubric 维度对同一文档集分别打分。Relevance 拿到 [3/4]，但 Reachability 只有 [1/4]——揭示了"看起来相关、但 LLM 用不上"的真正瓶颈。*

具体打分（这个例子里 Doc-Level 各项都是 **0/4** 起步的低分，但 Set-Level / Global-Level 还能给出有信息量的诊断）：

- **Relevance [3/4]**：勉强能用。
- **Complementarity [2/4]**：缺关键信息。
- **Redundancy [2/4]**：两对文档重复。
- **Conflict [2/4]**：Doc[11] 把 Nia Sanchez 描述错误。
- **Reachability [1/4]**：没有任何外部知识的 LLM 根本推不出答案。
- **最终 Response**：Nancy O'Dell（错误答案）—— 集合缺陷直接反映在输出上。

每个维度的 rubric 都"点名"了具体哪个 Doc 哪句话出了问题——这不是 nDCG 能给的信息。**这是 SetwiseEvalKit 真正的工程价值**：它把"reranker 为什么差"从黑箱变成可操作清单。

---

## 我的判断：值不值得深挖？

读完后我的整体感觉是：**这是一篇给"setwise retrieval"研究社区立标杆的工作，不是某个新 SOTA 的单点突破**。

### 亮点

1. **3 级 9 维 rubric 体系**是真正的方法论贡献。它把"集合质量"从一个抽象概念拆成可量化、可对比、可定位的 9 个子维度。每个维度都有具体含义和人类可解释的 rubric 描述。
2. **28K query-specific rubric + 多模型聚合**是个相当扎实的工程。仅靠单模型生成，rubric 质量是 37-43%；多模型聚合后跳到 70%——这给"LLM-as-Judge"领域一个重要经验：**单一裁判不如多裁判仲裁**。
3. **Coverage Score 与下游生成 Pearson r=0.92**——这是个非常强的可信度证据。说明 rubric 评估不是自说自话，是真的能预测实际效果。
4. **Evaluate-Diagnose-Optimize 闭环**——这是论文最重要的方法论贡献。**评估信号可以直接当选择信号**，这把"评估"和"优化"在同一个 rubric 空间里打通了。你想想看，**你不需要为同一个问题维护两套标准**——评估的尺子就是优化的目标。

### 问题与限制

1. **Oracle 假设**：Rubric4Setwise 用了参考答案生成 rubric。这是论文自己点明的 limitation（Section 6 最后一段）——rubric 是在"知道答案"的前提下生成的，所以"upper bound"。要真正部署到工业 RAG 系统里，必须**在不知道答案的情况下生成 rubric**——这一步怎么做、效果怎么衰减，论文没给出答案。

2. **Set-Level 维度普遍弱**：我看到这个数据其实有点忧虑。Set-Level 三个维度（Complementarity/Redundancy/Conflict）和下游质量的相关性都不显著（p=0.09-0.59），这意味着论文自己设计的 rubric 里**最有"集合级"特色的那部分，恰恰和下游关系最弱**。这是个需要解释的问题——是真的"集合级优化"对下游没那么重要，还是"现有方法都没做好，所以相关性拉不开"？我倾向于后者，但论文没展开论证。

3. **长文本分差太小**：长文本场景所有方法 Overall 在 30-33 之间，差距只有 3 个点。说实话这个分差很难下"显著优势"的结论。Rubric4Setwise 在长文本里比 SetR 只高 0.03（基本打平），核心优势其实是"用的文档更少"，但 LLM-judge 主观性这么强的情况下，这个"更少文档+同等质量"是不是真的稳，还需要更多验证。

4. **ReFreeKV 风格的工程 trade-off**：用 9 维度 rubric 引导选择，听起来很美，但**每选一篇文档都要让 LLM 评估整个集合**——这个推理成本比传统 reranker 高一个数量级。论文没说 latency/成本数字，这其实是工程上很关键的一环。

5. **和 Whole-Pool Setwise、Rank-R1、SetR/Rank4Gen 的边界**：现在 setwise reranking 是个挺拥挤的方向，2026 年已经有不少工作（SetR、Rank4Gen、Rank-R1、Whole-Pool Setwise Reranking 等）。Rubric4Setwise 跟他们比，**优势是闭环思路，劣势是推理成本**。到底用哪个，要看具体业务场景。

### 落地建议

如果你在做 RAG 检索系统，这篇论文的实操价值至少有两层：

- **诊断层**：用 SetwiseEvalKit 跑一遍你的当前 reranker，看看 Set-Level 三个维度（特别是 Complementarity 和 Redundancy）的分数。如果这两项低于 60%，基本可以确定你的 reranker 在做"高相关同质化文档堆叠"，赶紧试试 setwise 类方法。
- **优化层**：如果对延迟不敏感（比如企业内部知识库、低 QPS 的高价值场景），Rubric4Setwise 这种 rubric-guided 选择的思路是值得尝试的——但**重点不是复制它的 prompt，而是要解决 oracle 假设**。看看能不能用 question 本身的意图分类（不需要答案）来生成 rubric 的近似版。

---

## 收尾

如果让我用一句话总结这篇论文，我会说：

> **nDCG 是给"人浏览搜索结果"设计的尺子；RAG 时代我们需要的尺子，是能告诉 LLM "这堆文档够不够回答这个问题"——SetwiseEvalKit 终于把这个尺子造出来了，而 Rubric4Setwise 顺手证明了"用尺子本身就能挑出更好的文档"。**

下一个有意思的问题（论文作者在 Limitation 里也提到了）是：**能不能把 rubric preference 蒸馏到可训练的奖励里**，让 reranker 在没有参考答案的情况下也能学到这种"集合级判断"？如果做到了，setwise retrieval 才能真正走出学术 benchmark，进入工业 RAG 流水线。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我。*
