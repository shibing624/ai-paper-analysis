# SAAS：让 Agent 学会"我自己其实知道"——用自感知 RL 治理过度搜索

## 核心摘要

agentic search 这套范式现在用得很顺手，但有个一直没被认真解决的工程问题：**Agent 不知道自己知道什么**。明明参数里就装着答案，它非要先去 google 一圈；明明已经搜到关键证据了，它还要继续追加 query 把上下文塞满。结果就是延迟暴涨、推理成本起飞。

厦门大学和吉林大学的这篇 SAAS（arXiv:2605.29796）把这个现象命名为 **over-search**，并把它拆成两类：question-level（不该搜的搜了）和 step-level（搜够了还在搜）。作者的方案有意思——不是简单地给 search 动作加惩罚（他们做了实验，证明这条路会直接训崩），而是**让模型自己用 search-disabled 和 search-enabled 两组 rollout 对比，动态识别"这道题到底要不要搜"，然后只惩罚那些越界的搜索**。再配上一套先学怎么搜、再学少搜的两阶段课程，最终在 7 个 QA benchmark 上把平均搜索次数从 2.94 压到 0.97（Qwen2.5-7B），accuracy 几乎不掉。

值不值得看？如果你在做 RAG-Agent 或者 ReAct 系搜索，**这篇是近期把"什么时候该停"这件事讲得最清楚的一篇**，公式不复杂，但思路扎实。

---

## 论文信息

- **标题**：SAAS: Self-Aware Reinforcement Learning for Over-Search Mitigation in Agentic Search
- **作者**：Yunbo Tang, Chengyi Yang, Shiyu Liu, Zhishang Xiang, Zerui Chen, Qinggang Zhang, Jinsong Su
- **机构**：厦门大学信息学院、吉林大学人工智能学院
- **arXiv**：[2605.29796](https://arxiv.org/abs/2605.29796)（2026 年 5 月）
- **代码**：https://github.com/XMUDeepLIT/SAAS

---

## 一、为什么"过度搜索"是一个真问题

先抛一个场景。你部署了一个 search-augmented LLM，问它"《哈姆雷特》的作者是谁？"。理论上 7B 的模型脑子里应该早就有这个事实了，但实际跑下来，它会非常自信地先发一个 search query，把维基百科的简介抓回来，再"基于检索结果"告诉你莎士比亚。一个本来零延迟的回答，硬生生加了一次 retrieval round-trip。

这不是个例。

![图1：两类过度搜索现象](https://arxiv.org/html/2605.29796v2/x1.png)

*图 1：作者把过度搜索分成两类——左边是 unnecessary search（不需要搜的硬要搜，参数里就有答案），右边是 redundant search（已经搜到足够证据了还在追加查询）。这两个问题看着相似，根因不同：前者是模型不知道自己知道，后者是不知道自己已经够了。*

按作者的统计，用标准的 outcome-based RL（也就是只看最终答案对不对的 reward）训出来的 agentic search 模型，在 Qwen2.5-7B-Instruct 上，**question-level over-search ratio（QOR）会冲到 100%**——意思是只要这道题原则上能用参数知识答对，模型在训练后基本上还是会先发一次搜索。

为什么会这样？说实话这个机制讲透了挺反直觉的：

![图2：outcome-based RL 训练动态](https://arxiv.org/html/2605.29796v2/x2.png)

*图 2：训练过程中蓝线（不搜索的轨迹比例）从一开始就直线下滑到接近 0，红线（冗余搜索比例）反而一路涨到 50%。outcome-based RL 在鼓励模型"用搜索"——只要搜了能涨一点准确率，不搜的轨迹就会被组内归一化打成负 advantage。结果就是：模型学到的是"无脑先搜准没错"。*

我看到这张图的时候第一反应是——这个其实跟早年 RLHF 里 reward hacking 的味道很像。**只要 reward 函数不显式告诉你"少做某件事"，agent 就会朝"最大化反馈最稳的路径"漂移**。在 search 这个动作上，正反馈太明确了：搜了没坏处，搜了能多兜住几个边界 case，那肯定一直搜。

### 那直接给 search 动作加惩罚不就行了吗？

我读到这里也是这么想的。reward 里减去 $\alpha N$（搜索次数），不是手到擒来吗？

作者也是这么试的。结果——

![图3：朴素惩罚导致后期训练崩溃](https://arxiv.org/html/2605.29796v2/x3.png)

*图 3：橙色是朴素加惩罚后的 accuracy 曲线，大约在 step 250 之后直接崩盘。同时左图说明随着模型能力增长，"不搜也能答对"的题目比例从 12.7%（step 100）涨到 24.3%（step 300）——也就是说，搜索边界本身在动。*

这张图揭露了两个现象，是整篇文章的认知拐点：

**第一，搜索边界是会漂的**。模型刚开始什么都不会，很多题确实要靠搜索。但训了几百 step 之后，参数里学到的东西越来越多，原来需要外部证据的题渐渐也能 zero-shot 解了。**你训练前定的"哪些题该搜"的规则，到了 step 300 就过时了**。

**第二，固定惩罚会引发 reward hacking**。一旦模型发现"搜索 = 扣分"，它会在还没学会怎么搜的时候就开始拒搜，连必要的证据都不查了。准确率断崖式下跌。

读到这儿我有点皱眉——其实这个 dilemma 在 RL+tool 这条线上不是新问题。前阵子 ToolRL（Qian et al. 2025）、SEM（Sha et al. 2025）也都在啃同一块骨头。SAAS 的解法巧妙在于：**它不试图"事先定义"搜索边界，而是让边界跟着策略一起进化**。

---

## 二、SAAS 的核心思路

一句话讲清楚：**用模型当前自己的两组 rollout 来判断"这道题现在需不需要搜索"，然后按需惩罚**。

![图4：SAAS 整体流程](https://arxiv.org/html/2605.29796v2/x4.png)

*图 4：SAAS 的三个组件——Search Boundary Modeling（用 search-disabled 和 search-enabled 两组 rollout 对比，识别每道题的搜索边界）、Boundary-aware Reward（把边界判定翻译成轨迹级惩罚）、Stage-wise Optimization（先学搜索能力再学搜索效率，避免 reward hacking）。*

下面把三个组件一个个拆开。

### 2.1 Search Boundary Modeling：让模型自己回答"我不搜能不能答对"

这是整篇论文最有意思的设计。

对每道训练题 $q$，SAAS 不预设它属不属于"该搜索的题"，而是在每一步训练时，用**当前的策略 $\pi_\theta$**生成两组并行的 rollout：

- $G_d(q)$：搜索被关掉的 rollout 组，模型只能用参数知识硬答
- $G_e(q)$：搜索打开的 rollout 组，可以正常调用检索工具

然后数一数两组里有多少条轨迹答对了：

$$
\mathcal{S}(q)=\begin{cases}
\textsc{NoSearch}, & n_d(q) \geq \delta \\
\textsc{NeedSearch}, & n_d(q) = 0,\ n_e(q) \gt 0 \\
\textsc{Undetermined}, & \text{otherwise}
\end{cases}
$$

直觉很清楚：

- 关掉搜索还能答对至少 $\delta$ 次，说明这道题模型脑子里就有答案，**属于 NoSearch 类，搜索就是浪费**
- 关掉搜索一次都答不对，开了搜索能答对，**属于 NeedSearch 类，搜索是真的需要的**
- 中间态（关了答对几次但不到 $\delta$，或者两边都不太行），归为 Undetermined，先不管

这个设计的妙处在于 **on-policy**：搜索边界不是某个静态标注，而是随着 $\pi_\theta$ 的能力增长在持续重新评估。今天还属于 NeedSearch 的题，再训几百步可能就漂到 NoSearch 了。系统会自己跟上。

我觉得这是这篇论文最值钱的一笔。**before：手工标"哪些题该搜"——脏活累活、过两天还过时；now：让模型自己用对比 rollout 投票**。从工程视角，这相当于把"知识边界标注"这件事从离线人工完全切到了在线自评。

### 2.2 Boundary-Aware Reward：差异化惩罚

边界识别完了，怎么用？

总 reward 公式：

$$R_i = R_i^{\text{acc}} + \mathbb{I}[\text{F1}(\hat{y}_i, y_i) = 1] \cdot R_i^{\text{search}}$$

注意那个指示函数——**只有最终答案完全对的轨迹，才会启用搜索惩罚**。这个细节挺重要的，作者怕的是：模型还没学会怎么用 tool 的时候就被惩罚搜索，会直接懒得搜。

搜索奖励 $R_i^{\text{search}}$ 按类别走：

- **NoSearch 类**：零容忍，每搜一次就扣 $\alpha$ 分
  $$R_i^{\text{search}} = -\alpha N_i$$

- **NeedSearch 类**：搜本身不罚，但只罚"超出最少必要次数"的部分。$N_{\min}$ 取的是当前 search-enabled 组里答对的轨迹中搜索次数最少的那一条
  $$R_i^{\text{search}} = -\alpha \max(0, N_i - N_{\min})$$

- **Undetermined 类**：边界看不清，干脆不动，避免误伤

这个差异化惩罚本身没什么高深的，但 **NeedSearch 那条用同组最少成功搜索次数当 $N_{\min}$**，这是个挺工程的小聪明——它不需要外部 oracle 告诉你"这道题理论上几次搜索够"，直接拿当前策略自己跑出来的最优解当锚点。同组对比，相对意义明确。

最后还有个小细节：**两组 rollout 的 reward 量纲不一样**（一组只有 $R^{\text{acc}}$，另一组多了 $R^{\text{search}}$），直接合并算 advantage 会污染梯度。SAAS 用的是 group-wise advantage normalization——每组各自归一化再合并。这个 trick 在 GRPO 体系里挺常见的，但论文专门提了一句，因为不做的话训练会非常不稳。

### 2.3 Stage-Wise Optimization：先学会搜，再学会停

光有上面两条还不够。作者发现，如果训练一开始就同时开 accuracy reward 和 search reward，**模型会在还没掌握 tool use 的时候就被惩罚搜索，提前躺平**。

所以 SAAS 把训练拆成两阶段：

- **Stage I（capability acquisition）**：只有 $R^{\text{acc}}$，让 agent 先把"怎么调搜索、怎么从结果里抽信息"这些基础能力练出来
- **Stage II（efficiency refinement）**：等 validation 性能不再涨了，再切到完整 reward $R^{\text{acc}} + \mathbb{I}[\cdot] R^{\text{search}}$，开始压搜索次数

公式上：

$$
R_i = \begin{cases}
R_i^{\text{acc}}, & \text{Stage I} \\
R_i^{\text{acc}} + \mathbb{I}[\text{F1}(\hat{y}_i, y_i) = 1] R_i^{\text{search}}, & \text{Stage II}
\end{cases}
$$

说实话这个 staged curriculum 不算特别新，最近 RLHF / agentic RL 的论文里 staged training 用得越来越多。但放在 SAAS 这个 setting 里，它解决的痛点很具体：**你不能让一个还不会用工具的 agent 先学着不用工具**。这个顺序错了，整套 reward signal 就完全失效。

---

## 三、实验：accuracy 不掉，search count 砍 60%+

### 3.1 主实验：7 个 QA benchmark

数据集覆盖单跳和多跳两类：

- **单跳**：TriviaQA、PopQA、NQ
- **多跳**：HotpotQA、2WikiMultiHopQA、MuSiQue、Bamboogle

backbone 用了三个：Qwen2.5-3B-Instruct、Qwen2.5-7B-Instruct、Qwen3-4B-Instruct（appendix 里）。基线包括 Direct Inference、RFT、Search-R1、StepSearch、HiPRAG。

这里贴一下主表 Qwen2.5-7B 的关键对比（论文 Table 1）：

| Method | TriviaQA ACC/SC | HotpotQA ACC/SC | MuSiQue ACC/SC | AVG ACC | AVG SC |
|---|---|---|---|---|---|
| Direct Inference | 55.7 / - | 28.3 / - | 8.8 / - | 29.1 | - |
| RFT | 67.5 / 0.92 | 50.1 / 1.64 | 23.1 / 2.50 | 45.7 | 1.56 |
| Search-R1 | 68.3 / 1.11 | 45.8 / 1.19 | 16.8 / 1.38 | 42.9 | 1.25 |
| StepSearch | 67.8 / 1.28 | 53.2 / 1.80 | 26.2 / 2.27 | 47.6 | 1.69 |
| HiPRAG | **74.0** / 2.04 | **57.3** / 2.16 | 23.8 / 2.55 | **49.8** | 2.19 |
| **SAAS** | **74.0** / **0.56** | 53.6 / **0.96** | 22.6 / **1.30** | 48.7 | **0.97** |

读这个表的时候，我看了好一会儿。

第一反应是 SAAS 的 ACC 不是最高的——HiPRAG 在多跳任务上略胜半个百分点。**但代价是 HiPRAG 平均要搜 2.19 次，SAAS 只要 0.97 次**。同样的精度水平，搜索次数砍掉了一半还多。

如果把 efficiency 当一阶指标看（很多生产场景就是这样），SAAS 的 accuracy/SC 比是吊打其他方法的。**TriviaQA 上 ACC 持平 HiPRAG（74.0）但 SC 从 2.04 降到 0.56——延迟和 token 成本直接 4 倍下降**。

这里我得稍微泼一点冷水。HiPRAG 的 SC 高到 2.19 我有点意外，理论上它也是冲着搜索效率去的方法。我去翻了下原文（arXiv:2026），HiPRAG 主打的是过程奖励里加搜索信号，但**它在做的是 step-level 的搜索贡献度建模**，并没有显式约束 question-level 的搜索发起决策。SAAS 这里其实是在原本不强调搜索发起的方法上找了一个真空带。这不算不公平，但要意识到 SAAS 跟 HiPRAG 解决的不完全是同一类问题。

### 3.2 Over-search 专项分析

光看 ACC/SC 不够直观，作者还专门看了 SOR（step-level over-search ratio）和 QOR（question-level over-search ratio）。Qwen2.5-7B 上：

| Method | AVG SOR | AVG QOR |
|---|---|---|
| RFT | 14.8 | 52.5 |
| Search-R1 | 18.9 | 82.8 |
| StepSearch | 24.3 | 99.9 |
| HiPRAG | 19.5 | **100.0** |
| **SAAS** | **6.3** | **45.9** |

这个 QOR=100% 的现象挺刺眼的。**StepSearch 和 HiPRAG 在 question-level 上完全没有"放弃搜索"的能力——只要题目能用搜索答对，它就一定先搜一次**。SAAS 把这个比例压到 45.9%，意味着将近一半的题它能选择"我直接答"。

我觉得这是 SAAS 真正区别于现有方法的核心点。**前几代 RL agentic search 的优化目标里，"不搜索"这个动作根本就不是被鼓励的方向**。它们都默认搜索是好事，只在搜索质量上做文章。SAAS 把"什么时候不搜"作为一等公民引入 reward，这个视角的转换是这篇论文最大的贡献。

### 3.3 训练动态可视化

![图5：两阶段训练动态](https://arxiv.org/html/2605.29796v2/x5.png)

*图 5：F1 和 search count 在两阶段训练里的动态。Stage I 期间两条线一起涨——agent 在学怎么用搜索来提升答题质量。切到 Stage II 之后，search count 从 ~2.0 直接掉到 1.0 以下，但 F1 只是短暂微跌然后稳住。对比图 3 里朴素惩罚的崩溃曲线，这条非常优雅。*

这张图我是真的觉得做得到位。它把"为什么 stage-wise 起作用"讲得比正文还清楚——**搜索能力先建立稳了，再去压搜索数量，模型有空间在不掉精度的前提下找到 redundant 的部分剪掉**。如果一开始就两件事一起做，模型根本没学会判断哪些 search 是 redundant 的，只能粗暴地全砍。

### 3.4 消融：哪个组件最重要

| Variant | ACC | SC |
|---|---|---|
| **SAAS（full）** | **45.8** | 1.13 |
| w/o Stage-wise optimization | 40.9 | 0.95 |
| w/o On-policy estimation | 42.8 | 1.07 |

去掉 stage-wise，accuracy 从 45.8 跌到 40.9——掉了快 5 个点。这个降幅说明 staged curriculum 其实是 SAAS 能跑通的关键前提，**不是装饰品**。

去掉 on-policy estimation（用训练前固定的搜索边界），accuracy 也从 45.8 降到 42.8。差距没 stage-wise 那么大，但仍然说明搜索边界确实在动，固定边界会失配。

### 3.5 超参 $\delta$ 敏感性

![图6：delta 敏感性分析](https://arxiv.org/html/2605.29796v2/x6.png)

*图 6：$\delta$（GRPO 组里需要多少条 search-disabled 轨迹答对才判定为 NoSearch）的敏感性。$\delta=2$ 是最优点，ACC=45.8%、SC=1.13；$\delta=1$ 太宽松，把不该归 NoSearch 的题归进去，ACC 降到 43.1%；$\delta=3, 4$ 太严，反而更不稳定。*

$\delta=2$ 这个值挺实用的——**只要 search-disabled 组 8 条 rollout 里有 2 条答对，就认为这题模型自己能搞定**。门槛不高也不低。这种小消融在工程落地的时候是很需要的，不然换数据集换模型可能就要重新调。

---

## 四、几点批判性观察

写到这里我得稍微停一下，把一些我读完仍然觉得不完全踏实的点提出来。

**第一，rollout 翻倍的成本**。SAAS 每道题要跑两组 rollout（search-disabled 8 条 + search-enabled 8 条），训练阶段的算力开销直接 double。论文里也提了 max prompt/response len 都只有 512，相对短。如果做长链条 deep research 类任务（典型 trace 几千 token），这个 overhead 是个实打实的问题。作者在 limitation 里没特别强调这点，但工程上不能忽略。

**第二，$N_{\min}$ 的脆弱性**。NeedSearch 类用同组最少成功搜索次数当锚点——但如果当前策略下整组都答不对，$N_{\min}$ 就没定义。论文里把这种 case 划入 Undetermined 不做约束，但 Undetermined 的比例论文里没给。如果训练早期 Undetermined 占比很高，那 SAAS 的优化信号其实主要靠 NoSearch 那一支在贡献，这跟设计初衷有点偏。

**第三，跟同期 BAPO 的关系**。论文 reference 里引了一篇 Liu et al. 2026 的 BAPO（boundary-aware policy optimization for reliable agentic search，arXiv:2601.11037），署名作者 Liu 也跟 SAAS 是同一个 lab（XMU）的。两篇都在做 boundary-aware，但 SAAS 主表里没单独列 BAPO 作 baseline。这个我觉得是个有点遗憾的对比缺失，**同 lab 的两条相近路线的 head-to-head 应该是最能说明问题的**。

**第四，"过度搜索"在多跳里的边界判定真的稳吗**。MuSiQue 这种 4-hop 任务，模型答对的轨迹也未必是用了"恰好 N 次"搜索做的——可能是随机 7 次里中了一次，搜得多反而是因为犹豫。把 $N_{\min}$ 直接当作"必要搜索次数"在多跳长链上有水分。论文 MuSiQue 上 SAAS 的 ACC（22.6）确实比 baseline 略低（HiPRAG 23.8、StepSearch 26.2），可能就有这个原因。

但这些抠细节并不影响 SAAS 的整体价值。**核心 idea——用 on-policy 双组 rollout 自评估搜索边界——这个东西足够干净、足够可复用**。哪怕在 SAAS 当前的 reward 实现之外，这个 boundary modeling 模块本身可以拼到任何 agentic RL 系统里去。

---

## 五、对工程的启发

如果你正在做 agentic search 或者 RAG-Agent 的 RL 训练，这篇论文有几个可以直接拿走用的点：

**1）你的 reward 里要不要"显式引入不调工具"的信号**。如果你的 agent 老是在该 zero-shot 的题上滥用工具，根因很可能就是 reward signal 里"不调工具"从来没被正强化。SAAS 那个 NoSearch 类零容忍惩罚，思路可以照搬。

**2）够用就停的判定锚点**。不是所有任务都能做 search-disabled rollout，但同组最少成功 step 数当锚点的思路是通用的。任何 RL 训 tool-use 的场景都可以借鉴——比如 code agent、math agent，"答对的最短 trajectory"都可以当作"必要工具调用次数"的代理。

**3）staged curriculum 是 RL 训 agent 的几乎必须**。先学能力再学效率，先学怎么做再学少做。这个顺序在 RLHF、tool RL、reasoning RL 上反复被验证。如果你的训练曲线在中后期突然崩，回头看看是不是 reward signal 一开始就太复杂了。

**4）on-policy 边界比离线标注更可持续**。任何依赖"哪些题该走哪条路径"的训练流程，都应该考虑能不能让模型自己 vote，而不是靠人工 label。SAAS 这个做法在 multi-task RL、混合 reward 的 pipeline 里都有借鉴空间。

---

## 六、收尾

这篇 SAAS 在我心里是 2026 年到目前为止 agentic search 这条线上**问题定义最清晰、解法最干净的一篇**。

它没用特别花哨的 architecture，公式拢共就 9 个，但每一步的设计都能讲清楚为什么必须这么做——**搜索边界为什么要 on-policy（因为它会随策略漂移）、reward 为什么要分类（因为不同类型的题该惩罚的对象不同）、训练为什么要分阶段（因为先扼杀工具能力会导致 reward hacking）**。三个 motivation 串起来，每一环都对应着前面 preliminary analysis 里揭露的具体现象。

如果说有什么遗憾，就是它的实验都还停在 7B 以内的 Qwen 系列。在 Llama、DeepSeek、更大尺度模型上是不是同样有效，open question。另外它的训练 horizon 都比较短（response 512 token），放到 deep research 那种几千 token 长链 agent 上还需要重新验证。

但对于做 production-level RAG-Agent 的同学，**SAAS 提供的"如何让 agent 学会不调工具"这个能力，可能比再涨几个点 accuracy 都重要**。延迟、成本、用户体验，这三件事情上都直接受益。

值得花一晚上读一遍的论文。代码也已经开源（XMUDeepLIT/SAAS），有兴趣的同学可以拉下来跑跑。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
