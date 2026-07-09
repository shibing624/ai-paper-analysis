# PaperPilot：把文献搜索重写成可编辑的 DAG 工作流

> 论文：Multi-Turn Agentic Scientific Literature Search via Workflow Induction
> arXiv: [2607.00597](https://arxiv.org/abs/2607.00597)
> 作者：Jisen Li、Bingxuan Li、Nanyi Jiang、Xuying Ning、Xiyao Wang、Yifan Shen、Heng Wang、Yuqing Jian、Xiaoxia Wu、Ben Athiwaratkun、Pan Lu、Jiaxuan You、Bingxin Zhao
> 提交：2026 年 7 月 1 日（v1）/ 7 月 3 日（v2）

---

## 核心摘要

你有没有过这种经历：让 AI 帮你找论文，输入一句"找一下这篇 Transformer 的 follow-up 工作"，出来的结果要么太宽泛（把整个 NLP 领域都端上来），要么方向不对（要的是架构 follow-up，它给的是 vision 应用）？你跟它说"再聚焦一点，2020 年之后的，要架构创新"，它要么完全 get 不到，要么每次都从头来一遍。

问题出在哪？传统的"AI 搜索 agent"——不管是 DeepResearch 那种黑盒推理，还是给一堆 tool 让模型自己拼——说到底就是 **把搜索策略埋在自然语言里**。模型怎么想、怎么调参、怎么过滤，全是不可见的隐性决策。你想纠偏，只能再写一句 prompt；想复用上次的工作流？没门。

这篇 [arXiv:2607.00597](https://arxiv.org/abs/2607.00597) 提出的 **PaperPilot** 走了一条完全不一样的路：把文献搜索重写成 **workflow induction**（工作流归纳）问题。给定一个 anchor paper 和一个用户 query，agent 不再闷头生成自然语言推理，而是 **显式地构造一个有向无环图（DAG）**——节点是 keyword search、citation expand、filter、score、rerank 这些可执行的算子，边是数据流。然后用户可以像审代码一样审这个 DAG，添加节点、删掉节点、调整参数，agent 再据此精炼。

具体效果：在多轮交互下，**PaperPilot-9B 把 Hit@5 从 58.0 干到 77.0（涨 19 个点），MRR 从 47.5 提到 59.4，nDCG@10 从 26.8 涨到 32.5**，最夸张的是 **workflow 执行错误率从 9.5% 直接砸到 0%**。而成本只要 OpenAI DeepResearch 的 1/300 左右。

我的判断：这是一篇 **工程上很扎实、思路有真实增量** 的工作。核心贡献不是"在某个榜单上 SOTA"，而是 **把搜索过程从"黑盒自然语言"提升到"可检查、可编辑、可复用的结构化工作流"**。这个抽象本身就值很多钱——不管你做不做文献搜索。

---

## 现有方案的痛点：搜索过程为什么这么难管

先说清楚为什么这事值得专门写一篇论文。

科学文献搜索跟"找一下 React 教程"完全不是一个量级的事。学术搜索有几个特殊性质：

**第一，意图欠规格（underspecified）**。"找 follow-up 工作"到底是什么意思？直接引用？同方向扩展？同应用领域？同一个方法组件的不同尝试？这事用户自己都说不清楚，得看几个例子才能慢慢摸到边界。

**第二，偏好强依赖（preference-dependent）**。什么样的论文算"相关"，强烈依赖用户的研究口味。有人只在乎方法创新，有人只在乎实验 benchmark，有人必须跟自己领域相关。客观的"相关度"是不存在的。

**第三，意图会演化（evolves through interaction）**。用户给的反馈是模糊的——"这些太宽了"、"能不能再聚焦一点"、"要看 2022 年之后的"——这种反馈其实是在 **调整搜索策略本身**，而不是在追加查询条件。

但你去看现在的 AI 搜索 agent，几乎都是这两类套路：

- **固定流水线（fixed pipeline）**：代码写死 `keyword_search → citation_expand → rerank → top_k`，每一步的参数（比如 citation 是往前还是往后追）也是固定的。遇到偏好不同的用户，流水线不会变。
- **纯语言推理（implicit language-only reasoning）**：把工具丢给模型，让它自己 chain-of-thought 自己规划。听起来灵活，实际上你完全看不到模型在做啥决策，错了也不知道错在哪，纠偏只能重新发 prompt。

更糟的是，用户给的反馈——"focus on more recent work"——在这些系统里 **只能被当成新的查询文本去叠加**，而不是被解读为"修改检索行为本身的指令"。这就导致每次反馈都得"重新理解一遍"，效率低到没法用。

PaperPilot 想解的就是这个：**把搜索过程显式化、结构化、可编辑化**，让用户反馈能直接作用在工作流结构上，而不是淹没在自然语言里。

---

## PaperPilot 的核心想法：把搜索当成 DAG 来归纳

![图1：单轮 vs 多轮搜索的对比](https://arxiv.org/html/2607.00597v2/x1.png)

*图1：上面是传统的单轮搜索，模型看到 anchor paper + query + direction 就直接吐结果，过程中没有澄清机制，结果要么不相关要么要重做。下面是 PaperPilot 的多轮流程：先生成初始 workflow，agent 主动向用户提澄清问题（"pretraining 还是 fine-tuning？时间窗口？应用领域？"），根据用户回答编辑 DAG，最终产出相关且成本可控的结果。*

一句话讲清楚 PaperPilot 的核心思想：

> **给定一个 anchor paper 和一个用户 query，agent 归纳出一个可执行的 DAG 工作流。每个节点是一个符号算子，每条边是数据流。用户可以像改代码一样编辑这个 DAG，反馈直接作用在工作流结构上。**

整个系统跑起来就是三步循环：

1. **Induce**：根据 query、anchor paper、历史交互，生成一个初始 DAG $G_t$
2. **Execute**：执行这个 DAG，得到下一轮的 paper 池 $P_{t+1}$ 和结果 $y_t$
3. **Refine**：把用户反馈 $f_t$ 翻译成对 DAG 的局部编辑（增/删/改节点），得到 $G_{t+1}$

用形式化语言说就是：
$$G_t = \text{Induce}(q, p_0, \mathcal{P}_t, \mathcal{H}_t)$$
$$(\mathcal{P}_{t+1}, y_t) = \text{Execute}(G_t, \mathcal{P}_t)$$
$$G_{t+1} = \text{Refine}(G_t, f_t, \mathcal{H}_{t+1})$$

这套抽象为啥有杀伤力？两个灵活性：

- **算子选择灵活**："要 strong baseline" 自然会强调 citation_expand + rerank，"要 emerging follow-up" 会偏重 recent keyword search + citation tracing，DAG 自动适配。
- **算子参数灵活**：关键词、引用方向（前/后）、过滤谓词、评分公式、rerank 准则，全都是可调参数。

---

## 系统长什么样

![图2：PaperPilot 系统架构](https://arxiv.org/html/2607.00597v2/x2.png)

*图2：完整的 PaperPilot 工作流程。从左边的 User Context（anchor paper 是 "Attention Is All You Need"、query 是 "find follow-up work"、"successor" 方向）开始，PaperPilot Agent 先 generate 初始 DAG，再 ask 澄清问题（"fine-tuning or pretraining？2020 年之后？仅 Transformer？"），用户回答后 modify DAG，最终产出 6-15 篇带排名的相关论文。中间那个 Editable Typed-Operator DAG 是核心：用 keyword_search 拉候选、用 citation_expand 扩引用、用 filter 加约束、用 score 打分、用 llm_filter 做语义过滤、用 llm_rerank 排最终顺序。*

看一眼 Figure 2 你就能 get 到这个系统的工程感。它不是空喊口号，每个节点都是实际可调用的算子。

**算子库（PaperPilot-Toolset）一共 17 个算子**，分成 7 类：

| 类别 | 算子示例 | 干啥用的 |
|-----|---------|---------|
| **A. 数据源** | `keyword_search`, `citation_expand`, `web_resolve` | 拉候选论文 |
| **B. 合并/过滤** | `union`, `dedupe`, `filter` | 组合和清洗 paper 集 |
| **C. 打分** | `score` | 按特征（年份、引用数等）打分 |
| **D. 截取** | `top_k`, `above` | 按分数切前 K 或阈值以上 |
| **E. 重排** | `llm_rerank`, `nli_filter`, `fine_read` | 用 LLM/NLI 重新排序 |
| **F. LLM 工具** | `llm_keywords`, `llm_keywords_from` | 让 LLM 生成关键词 |
| **G. 输出** | `extract_evidence`, `pairwise_nli`, `build_graph` | 抽取证据、构图 |

注意一个工程上的关键约束：**DAG 的边必须类型一致**——前驱节点的输出类型必须匹配后继节点的输入类型。这看起来是个细节，实际上是防止 agent 生成"垃圾工作流"的核心护栏。没有这个约束，agent 很容易拼出"把分数喂给 keyword_search"这种完全跑不通的 DAG。

---

## 训练：让 9B 模型学会"听话地编 DAG"

训练分两阶段，思路跟现在做 alignment 的主流打法一致——SFT 先模仿，再用偏好优化纠偏。但这里的关键 trick 在偏好数据怎么造。

### 阶段一：监督微调（SFT）

数据来源是 2,723 个 anchor-query 训练样本，覆盖 5 个搜索方向（predecessor / successor / sibling / benchmark / survey）。用 strong teacher model 跑出完整轨迹，提取"gold paper 出现在 top-5 且方向条件满足"的高质量回合，最后得到 **5,540 个 workflow supervision examples**。

训练配置：3 epochs，lr 2e-4，sequence length 14336，LoRA 适配 attention + MLP projection。这套配置是"刚好够"的状态，没在算力上炫技。

### 阶段二：偏好优化（DPO/IPO）

这一步有意思了。**怎么造 chosen-rejected pair？**

答案是：**人为破坏工作流**。

具体来说，把上面那些成功的 teacher workflow 当 chosen，然后用"corruptions"造 rejected。Corruptions 分两类：

- **结构有效性错误**：无效引用（节点指向不存在的输入）、缺失输入（节点没接上游）
- **搜索质量错误**：错误算子、丢失关键节点、过滤器偏移、NLI 维度模糊

过滤掉太容易区分的 pair 后，留下 **1,733 个 hard pairs**。训练用 IPO-style DPO objective，β=0.2，3 epochs。

这个设计挺聪明：把"工作流是否合理"这种本来很模糊的判断，**用结构化的方式具象化**了。你不用去标"这个工作流好不好"，你只要让系统见过各种"坏工作流"长啥样就行。

---

## 主实验：把数字摆出来

### 检索质量

主表非常密，我把它浓缩成对比表，挑出最关键的几个点：

| 模型 | 配置 | Hit@5 | MRR | nDCG@10 | 成本/case |
|------|------|-------|-----|---------|-----------|
| OpenAI DeepResearch | 1 turn | 72.0 | 53.0 | 29.2 | $6.090 |
| GPT-5.4 Web Search | 1 turn | 72.5 | 60.2 | 33.5 | $0.370 |
| GPT-5.4 Web Search + Toolset | 5 turns | **84.0** | **71.8** | **41.6** | $0.150 |
| Qwen3.5-9B Toolset | 4 turns | 58.0 | 47.5 | 26.8 | $0.013 |
| **PaperPilot-9B** | **7 turns** | **77.0** | **59.4** | **32.5** | **$0.018** |

几个值得展开的观察：

**1. PaperPilot-9B 把 Qwen3.5-9B 全面碾压**：

![图3：PaperPilot-9B vs Qwen3.5-9B 提升幅度](https://arxiv.org/html/2607.00597v2/x5.png)

*图3：柱状图直接展示 PaperPilot-9B 相对 Qwen3.5-9B（Toolset）MT 的提升：Hit@5 +19.0（58.0→77.0），R@50 +5.2（34.8→40.0），MRR +11.9（47.5→59.4），nDCG@10 +5.7（26.8→32.5），最关键的是 err% 从 9.5% 直接砸到 0%。*

注意最后一个柱子的意义——workflow execution error rate 从 9.5% 砸到 0%。也就是说，**未经训练的 toolset agent 有 1/10 的概率生成的 DAG 根本跑不通**（类型不匹配、引用不存在的节点、缺输入），经过 workflow induction 训练后这个错误完全消失。这事比涨点还重要：它意味着你敢把这个 agent 部署到生产环境，不用担心它突然给你抛个 exception。

**2. 同尺寸下 PaperPilot-9B 击败 GPT-5.4 单轮**：

Hit@5 上 77.0 vs 72.5，Hit@10 83.5 vs 79.0，Hit@15 直接平 89.5 vs 81.0。一个 9B 模型，靠多轮工作流精炼 + 显式 DAG 结构，在没刷大模型的前提下硬刚过闭源大模型。MRR 略输（59.4 vs 60.2），说明最强的相关论文还是要靠大模型的"理解力"，但前 K 召回已经全面胜出。

**3. 成本把商业系统吊起来打**：

PaperPilot-9B 每 case 成本 $0.018，OpenAI DeepResearch $6.09，**差 338 倍**。GPT-5.4 Web Search + Toolset 5 轮配置虽然检索质量最高（Hit@5 84.0），但每 case 成本 $0.15，是 PaperPilot 的 8.3 倍。对学术界和中小公司来说，这价差是真金白银的。

### 人类研究

| 系统 | 成功率 | Top-1 距离↓ | 用户满意度 | 收敛轮数↓ |
|------|--------|------------|-----------|-----------|
| GPT-5.4 | 32.0% | 7.8 | 2.4/5 | 4.0 |
| OpenAI DeepResearch | 8.0% | 27.4 | – | 1.0 |
| **PaperPilot** | **74.7%** | **2.4** | **4.2/5** | **3.8** |

人类研究结果更夸张——PaperPilot 成功率 74.7%，是 GPT-5.4 的 2.3 倍，是 DeepResearch 的 9.3 倍。Top-1 Distance（最优结果实际排名）从 7.8 砍到 2.4，等于你点开结果的前几条基本就有你要的。

不过这里我要提个醒：**这个人类研究的样本量没在摘要里说**，小样本的人类研究很容易出现统计波动。但结合主表的数据，这个量级的差距不太可能是纯噪声。

### 关键词池规模消融

![图4：不同关键词池规模下的性能](https://arxiv.org/html/2607.00597v2/x8.png)

*图4：K=8 到 K=20（候选池从 1.0× 到 2.5×）的热力图。Hit@1 在 K=12 达到峰值 41.0%，Hit@5 在 K=8/10 都是 70%+。整体规律是 K=8-10 达到性能峰值，再大反而下降——池子越大干扰项越多，下游 rerank 的负担越重。*

这个消融挺有意思，反直觉的：**候选论文越多反而越差**。直觉上"我给 1000 个候选让 rerank 慢慢选"应该更好，但实际数据告诉你，K=8-10 的小池子反而跑出最高 Hit@5。原因也很朴素——池子越大，垃圾越多，下游 filter 和 rerank 的负担越重，最终把真正相关的论文也搅浑了。**这是个工程上很反常识但很有用的发现**：在文献搜索这种召回质量依赖下游精排的场景，"宁少勿滥"往往更对。

---

## 我的判断：值不值得花时间深读

### 亮点

**1. 抽象本身值钱**。把搜索过程从"模型内心戏"提升到"显式 DAG"，这件事的工程意义远大于论文里那 19 个点。**可解释、可检查、可编辑、可复用**——这四个属性同时满足的 agent 设计，在学术搜索这个场景里是头一次。如果你做过 AI 搜索相关的工程，你会知道"模型给你一坨结果，你不知道它为啥给你这些，也没法改"是有多令人抓狂。

**2. 数据合成范式可复用**。用"对成功 workflow 施加受控 corruption"来造偏好对，这个 trick 完全可以搬到别的 agent 训练里——不管是代码生成、SQL 编写，还是工具调用。核心思想是：**你不用人工标"好/坏"，你只要定义"失败模式"，让系统见过这些失败模式就行**。

**3. 9B 模型成本可接受**。在学术搜索这种对延迟不敏感（用户等得起 7 轮交互）、对成本敏感（个人研究者、实验室用不起 DeepResearch）的场景，9B 模型 + 显式工作流是个非常合理的设计点。

### 问题与保留

**1. 算子库是封闭的**。Toolset 是预定义的 17 个算子，覆盖的是"标准文献搜索"流程。如果你的领域需要非常特殊的搜索行为（比如医学文献要看 MeSH 术语、专利搜索要查法律状态），这个 toolset 不够用。论文也承认了这一点，列在 Limitations 里。

**2. 教师模型偏差**。SFT 数据全是从 strong teacher（应该是某个大模型）跑出来的轨迹，teacher 的盲点会原样继承。论文没明说 teacher 是哪个模型，但从上下文推测可能是 GPT 系列。**这是个没法完全避免但值得警惕的依赖**。

**3. 用户模拟器 vs 真实用户**。多轮评估用的是 Qwen3.5-397B 模拟的用户，论文用了三层泄露控制（prompt 约束、字符串匹配、leak-checker 模型），但模拟反馈毕竟不是真人。真实用户给的反馈可能更模糊、更不一致，对系统鲁棒性是更大的考验。

**4. 领域覆盖偏 CS**。基准主要在 CS 文献上，跨领域的泛化没验证。CS 文献的元数据（arXiv、S2、citation graph）相对干净，医学、法律、社会科学的文献生态完全不同，搜索行为也会差很多。

### 对工程实践的启发

如果你正在做 AI 搜索相关的产品（不管是文献、代码、还是商品），这套思路有几个直接可以借鉴的点：

- **把不可见的策略显式化**。哪怕你不搞 DAG 这么重的结构，至少把"模型对每个候选做了什么"打到 log 里。
- **用户反馈要有结构化的"落点"**。"再聚焦一点"这种反馈不能只被当成新 query，得有办法映射到具体的检索行为调整上。
- **训练数据别全靠人工标**。SFT 用成功的轨迹 + 偏好对用受控 corruption，这个套路比"让标号员写 chosen/rejected"成本低得多，也更可控。

---

## 写在最后

这篇论文给的最深一层启发是：**AI agent 的下一个提升点可能不在"模型更大"或"推理更长"，而在"过程结构化"**。把搜索、规划、决策这些过程从隐性提升到显性，带来的可解释性、可控性、可复用性，对真实落地远比单点性能 SOTA 重要。

PaperPilot 只是一个具体的例子，但这个思路——把任意 AI 任务都重写成一个结构化工作流，再让用户去编辑它——可能比我们想的更普适。

> 觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
