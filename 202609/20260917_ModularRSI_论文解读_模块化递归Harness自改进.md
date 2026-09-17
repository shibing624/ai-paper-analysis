# ModularRSI：把 Agent 的"脚手架"拆成五个模块各自进化，避免在评测集上偷偷作弊

## 一个让人不太舒服的观察

前阵子读 Meta-Harness 那篇工作的时候就注意到一个细节：它在 Terminal-Bench 2 上搜索 harness 变体，但搜索集和测试集是同一批 89 个任务。作者自己也坦白了——基准太小、跑一轮太贵，只能这么干，再靠人工检查和正则审计来防任务特定泄漏。

说实话，这几乎是目前 harness 自进化（Harness RSI）方向的通病。Darwin Gödel Machine 把 SWE-bench Verified 从 20% 干到 50%，AHE 在 Terminal-Bench 2 上跑赢手工设计的 Codex-CLI，数字都很漂亮，但一个始终悬着的问题是：进化出来的 harness，到底学到了"可复用的执行机制"，还是只是"记住了这个评测集长什么样"？

上周刷到这篇 ModularRSI（arXiv:2609.14857），我的第一反应是：终于有人把这个问题当成一等公民来正面处理了。

## 🎯 核心摘要

Harness 递归自我改进（RSI）——让 agent 从执行经验中自动改进自己的运行脚手架——一直有个难以回答的问题：你怎么知道进化出的改进能泛化，而不是过拟合了评测基准？ModularRSI 给了三层回答：数据上，自建 2000 个与下游评测基准完全不相交的可执行进化任务；轨迹上，同一任务跑多次，把成功和失败的轨迹配对做对比分析，再用跨任务投票过滤掉实例特定的噪声；机制上，把 harness 拆成五个功能模块（Agent Loop、Tool Use、Observation Management、Context Management、Task Completion Detection），各自在受限范围内独立进化，最后再做一次跨模块集成。结果相当能打：统一协议下 AHE 和 Meta-Harness 在 Terminal-Bench 2 上只比基线高约 1 个点，ModularRSI 高出 5.6 个点；进化出的冻结 harness 还能跨领域、跨基础模型迁移。我的判断：这不是底层范式突破，但它把 harness 进化从"黑盒炼丹"变成了"有模块化、有验证门、有数据隔离的工程流程"，这个规范化本身就值钱。

## 📖 论文信息

- **标题**：ModularRSI: Modular and Generalizable Recursive Harness Self-Improvement
- **作者**：Siwei Wu, Jincheng Ren, Yizhi Li, Haau-Sing Li, Chengran Yang, Yuxuan Zhang, Weicheng Gu, Jian Yang, Riza Batista-Navarro, Chuanyi Zhang, Xianglong Liu, Ming Zhou, Bryan Dai, Chenghua Lin
- **机构**：Beihang University、University of Manchester、IQuest Research、M-A-P、Langboat、Hohai University
- **发表**：2026 年 9 月 14 日，arXiv:2609.14857v1 [cs.CL]
- **代码与数据**：https://github.com/IQuestLab/ModularRSI

---

## 🤔 为什么 harness 进化的"泛化"是个真问题

先把背景捋一下。所谓 harness，是包在基础模型外面的那层执行脚手架：怎么组织推理-动作循环、怎么解析工具调用、怎么压缩上下文、怎么判断任务做完了没有。最近这波工作（Meta-Harness、AHE、Self-Harness、DGM、Living-Harness 等）的核心发现是：**模型权重不动，光进化这层脚手架，benchmark 分数就能涨几十个点**。DGM 那个 20% → 50% 的 SWE-bench 结果就是最扎眼的例子。

但问题恰恰藏在这里。ModularRSI 把泛化困难拆成三个相互耦合的挑战，我觉得这个拆解本身就很值得抄走：

**挑战一：进化数据和评测数据纠缠不清。** 大多数现有方法直接在下游评测基准（或其子集）上做进化。理由是现成的：带可执行环境、带可靠判分器的长周期任务很难造。但代价是，你没法区分进化出的改进是"通用的工程经验"还是"基准特有模式"。Lilian Weng 在她那篇 harness engineering 综述里也提过类似的担忧——Self-Harness 学出的修改是模型特定的，帮 Qwen 的修改和帮 GLM 的修改根本不是同一套。

**挑战二：单条轨迹说不清"该改什么"。** 一个任务失败了，是 harness 有系统性缺陷，还是 agent 恰好在这道题上推理失误？单侧轨迹（只成功或只失败）把这两种证据混在一起。直接基于它优化，很容易把任务特定的解题细节缝进 harness，迁移到新任务就翻车。

**挑战三：改哪一块不清楚。** harness 是个整体代码库，即便诊断出了反复出现的行为缺陷，信用分配问题依然存在——到底该改哪个组件？很多方法干脆重写大段 harness 代码，无关机制被搅在一起，修改既难归因也难验证。

这三个挑战分别对应数据、轨迹、机制三个层面。ModularRSI 的设计就是对着这三层逐一下药的。

---

## 🏗️ 方法：对比分析 × 模块化 × 验证门

![ModularRSI 框架总览](https://arxiv.org/html/2609.14857v1/main.png)

*图 1：ModularRSI 总览。上方是"对比轨迹采样与分析"：任务经 K 次 rollout 后按奖励分为 Positive / Contrastive / Negative 三组，配对的成败轨迹做对比分析，单侧轨迹做效率或诊断分析，产出结构化 findings。左下是"逐模块修改"：五个模块由 Code-Modify Agent 在 Evolution History 的约束下独立进化。右下是"验证门"：执行验证、程序验证、diff 审查三道关卡，任何一道不过就回滚。*

一句话概括核心思路：**把粗粒度的任务级结果，转化为局部化的函数级进化信号**。被进化的 agent 自己充当 Code-Modify Agent，既分析轨迹也动手改代码。

### 对比轨迹分析：同一个任务，为什么这次成了那次没成

这是最对我的胃口的设计。对每个进化任务 $x_i$，agent 跑 $K$ 次 rollout，每条轨迹拿到二元奖励 $r_i^k$。按成功率把任务分三组：

$$\mathcal{G}_i = \begin{cases} \mathrm{Positive}, & \sum_k r_i^k = K \\ \mathrm{Contrastive}, & 0 \lt \sum_k r_i^k \lt K \\ \mathrm{Negative}, & \sum_k r_i^k = 0 \end{cases}$$

你想想看，一个任务跑 5 次，3 次成功 2 次失败——**模型是同一个，harness 是同一个，区别只在执行过程中的具体行为**。这种对比组才是诊断 harness 缺陷的金矿：把成功轨迹和失败轨迹配对比较，找出与不同结果相关的函数级因素。

三组各有各的分析策略。全成功组（Positive）不做因果诊断，转而找效率改进空间——冗余动作、重复探索、不必要的工具调用。全失败组（Negative）先去查 Trajectory Memory（跨 epoch 的轨迹记忆库）里有没有这个任务的历史成功轨迹，有就配对做对比，没有才退化为单侧诊断，识别重复循环、工具误用、过早终止这类明显缺陷。纯基础设施失败的任务（`infra_only`）直接被排除在 harness 诊断之外——环境挂了就别赖 harness。

分析结果以结构化 JSON findings 输出，每条 finding 带几个很讲究的字段：`is_culprit`（因果判定）、`divergence`（轨迹分歧证据）、`would_change_outcome`（反事实判断——改这个模块能不能改变任务结果）、`suggested_change`。

还有一处细节让我挺欣赏。分析提示词里明确写着"这不是责任归因"：**只有当存在一个具体的模块修改能合理地把任务从失败推向成功时，才允许报告 culprit**；否则就报"无 culprit 不修复"，并且不许"为了有发现而发明一个微不足道的调整"。做过自动进化系统的人都知道，优化器最大的恶习就是没病开药——必须改点什么来证明自己干了活。这条约束是拿工程血泪换来的。

### 跨任务聚合：多数任务都踩的坑，才是 harness 的坑

单个任务的诊断可能是巧合，怎么过滤？两个机制。

一个是**投票**。针对同一函数的语义相似诊断被整合成候选修改，每个候选按"提供支持证据的不同任务数量"计票，优先处理票数最高的。被多个任务共同支持的修改，大概率是系统性问题；只出现在一个任务里的，先靠边站。修改提示词里写得更直白：修改必须是通用机制，"只拯救某一个具名任务的修改在这里毫无价值"。

另一个是**进化历史**。每个函数维护变更历史，记录以往每次修订引入了什么功能，并暴露给 Code-Modify Agent。这样后续更新能保住已进化的功能，避免改来改去反复横跳（论文叫 evolution oscillation）。这其实就是版本控制思想在进化循环里的落地——没有历史的进化系统，就像没有 git 的团队协作。

### 五模块拆解：各自进化，互不干扰

ModularRSI 把初始 harness（Harbor 的 Terminus-2）重组为五个模块，进化被限制在直接介导 agent-环境交互的行为机制上，沙箱初始化、并行执行、LLM 通信这些基础设施不动：

| 模块 | 职责 | 典型接口 |
|---|---|---|
| Agent Loop | 推理-动作-观察主循环，协调其他模块，处理重试与终止覆盖 | `AgentLoop.run` |
| Tool Use | 把模型输出解析为命令和完成信号，执行工具调用 | `ToolSet.parse_llm_response / execute` |
| Observation Management | 处理当前终端输出，过滤压缩噪声观察 | `Observation.capture` |
| Context Management | 维护跨步骤的累积对话历史，压缩与检索 | `ContextMgmt.maybe_compress / force_summarize` |
| Task Completion Detection | 判断是否可停止，只输出"建议"，决定权在 Agent Loop | `VerificationLoop.should_terminate` |

注意 Observation 和 Context 的切分：前者管"当前这一刻环境吐了什么东西进来"，后者管"历史对话怎么保留和压缩"。这个边界划得挺干净——很多手工 harness 里这两件事是糊在一起的。

五模块独立进化、不共享中间更新，因此可以并行。全部进化完后再跑一个跨模块集成 epoch：用集成后的 harness 跑任务，分析轨迹找出跨模块冲突，删掉重复机制、理清模块职责。然后函数库冻结，评测期间不许再改。

另外有函数库管理：Function Merge 合并语义重复的函数；Task-Aware Function Composition 按任务描述选出相关函数子集激活。底层函数库可以越长越大，但实际跑起来的 harness 保持紧凑。

### 三道验证门：不过就回滚

每个修改提案要连过三关：

1. **程序检查**：AST 验证、导入检查、协议合规等静态检查，失败就按 diff 回滚到上一版本；
2. **Diff 审查**：Code-Modify Agent 审查 diff 是否编码了任务特定的解决方案或启发式——不像能泛化的，拒；
3. **执行验证**：从当前批次随机抽 2 个任务实际跑一遍更新后的 harness，引入运行时错误就回滚。

这套东西不新鲜，但组合起来解决了 harness 进化里最烦人的问题：改坏的代码不能污染后续进化。

### 进化数据：2000 个与评测集"隔离"的任务

这是论文标题里 benchmark-disjoint 的落点。构建流程：从 Terminal-Bench 和 SWE-Bench 系列**只提取任务类别标签**作为领域指导（不用任何基准实例本身），人工标注者按标签从 GitHub、Hugging Face、Kaggle、Linux kernel 文档等外部公开来源检索材料，构建成 Harbor 格式的可执行任务。

质控管道有四层：LLM 过滤（环境完整性、非平凡性、评估器必须测功能正确性而非表面完成信号）、可执行验证（参考 `solution.sh` 必须全过测试，同时 no-op 提交不得获得正奖励）、人工审查、LLM 语义相似度筛选剔除与下游基准高度重叠的实例。最终 2000 个实例，TB 相关与 SWE 相关大体平衡。

---

## 🧪 实验：数字比故事更有说服力

评测在 Harbor 框架下进行，两个基准：Terminal-Bench 2.0（89 个长周期终端任务）和 SWE-Bench Verified（500 个真实仓库任务）。指标四个：Acc（平均成功率）、Pass@3（三次里至少成一次）、Pass³（三次全成，衡量稳定性）、StepNum（平均交互步数，越低越高效）。进化主力模型是 DeepSeek-V4-Flash-Preview / DeepSeek-V4-Flash-0731，因算力限制实际用了 120 个 TB 相关 + 120 个 SWE 相关实例做进化。

### 主实验：域内涨，跨域也涨

冻结后的 harness 跨基准迁移（backbone: DeepSeek-V4-Flash-Preview）：

| 评测基准 | 进化集 | Acc ↑ | Pass@3 ↑ | Pass³ ↑ |
|---|---|---|---|---|
| SWE-Bench Verified | 无进化 | 73.40 | 83.20 | 62.80 |
| SWE-Bench Verified | TB 相关（跨域） | 75.80 | 84.67 | 66.20 |
| SWE-Bench Verified | SWE 相关（域内） | **76.45** | **85.30** | **66.80** |
| Terminal-Bench 2.0 | 无进化 | 47.57 | 58.43 | 30.34 |
| Terminal-Bench 2.0 | TB 相关（域内） | **52.43** | **65.17** | **35.96** |
| Terminal-Bench 2.0 | SWE 相关（跨域） | 49.40 | 60.67 | 30.34 |

几个点值得拆开说。域内：TB 2.0 上 Acc 从 47.57 到 52.43，涨了近 5 个点；SWE-Bench Verified 从 73.40 到 76.45。跨域同样有效——用 TB 任务进化出的 harness 拿到 SWE-Bench 上还能涨 2.4 个点，反过来也成立。这正是 benchmark-disjoint 设计想买的东西：进化没有见过任何评测任务，提升却依然迁移过去了。

还有一个容易被忽略的数字：TB 2.0 上 Pass³ 从 30.34 涨到 35.96。Pass³ 要求三次独立 rollout 全成功，它涨说明进化不只是"碰运气多蒙对几题"，而是真的降低了随机失败的概率。

### 跟同类方法硬碰硬：5.6 个点 vs 1 个点

这张表是全文最有分量的。统一协议：所有方法从同一个 Terminus-2 出发、用同一批 120 个进化实例、同一个模型（DeepSeek-V4-Flash-0731）、同样的 TPM 限制、全部禁用 web 搜索。ModularRSI 每模块进化 3 epoch 后合并；为匹配总进化轮数，AHE 和 Meta-Harness 进化了 16 个 epoch。

| 方法（Terminal-Bench 2.0） | Acc ↑ | Pass@3 ↑ | Pass³ ↑ |
|---|---|---|---|
| Baseline（Terminus-2） | 61.79 | 73.03 | 50.56 |
| Meta-Harness | 62.92 | 74.16 | 50.56 |
| AHE | 62.54 | 73.03 | 51.69 |
| ModularRSI | **67.42** | **78.65** | **56.18** |

看到这个表我愣了一下。AHE 和 Meta-Harness 在各自原论文里都是跑赢手工 harness 的明星方法，换到 benchmark-disjoint 协议下，收益缩水到 1 个点上下。说实话这个对比对它们不完全公平——这套协议本来就是 ModularRSI 的主场，且原论文里的高收益很可能部分来自评测集内的进化。但反过来想，这恰恰印证了论文的核心论点：**一旦堵住"用基准数据进化"这条路，大多数方法的泛化收益就所剩无几**。5.6 个点 vs 1 个点，差距不在优化能力，在方法论。

### 消融：模块化不是花架子

先看进化策略对比（Terminal-Bench 2.0）：

| 进化策略 | Acc ↑ | Pass@3 ↑ | Pass³ ↑ | StepNum ↓ |
|---|---|---|---|---|
| Baseline | 47.57 | 58.43 | 30.34 | 34.70 |
| 非模块化进化 | 46.44 | 64.04 | 24.72 | 29.03 |
| 全模块联合进化 | 44.19 | 61.80 | 24.72 | 44.34 |
| ModularRSI（独立进化+集成） | **52.43** | **65.17** | **35.96** | 35.57 |

这个结果挺反直觉的：非模块化和联合进化不但没涨，Acc 反而比基线还低，Pass³ 掉了近 6 个点。改的范围越大，效果越差——修改空间一大，无关机制纠缠进来， harness 被越改越坏。只有"各管各的再合并"这条路走通了。

再看逐模块贡献：

| 单模块进化 | Acc ↑ | Pass@3 ↑ | Pass³ ↑ | StepNum ↓ |
|---|---|---|---|---|
| Baseline | 47.57 | 58.43 | 30.34 | 34.70 |
| Context Management | 49.44 | 61.80 | 31.40 | 35.10 |
| Tool Use | 50.19 | 62.92 | 30.34 | 41.28 |
| Agent Loop | 50.56 | 64.04 | 34.83 | 40.40 |
| Observation Management | 49.81 | 65.17 | 33.70 | **22.50** |
| Task Completion Detection | 49.44 | 65.17 | 31.40 | 31.06 |
| **五模块集成** | **52.43** | **65.17** | **35.96** | 35.57 |

五个模块各自单独进化都超过基线，但贡献的形状不同：Agent Loop 对 Acc 贡献最大（50.56），Observation Management 则把平均步数从 34.70 压到 22.50——过滤噪声观察省下了大量无效交互。集成后 Acc 52.43 高于任何单模块，说明各模块学到的是互补的改进，而不是重复造轮子。

进化数据难度分布也做了消融（SWE-Bench Verified，Acc）：中等难度为主 76.45，高难度+低难度混合 74.25。差 2.2 个点。解释很顺：太简单的任务没有失败轨迹可对比，太难的任务几乎不出成功轨迹，两头都提供不了对比证据。

### 跨模型迁移：学的是机制，不是模型癖好

用 DeepSeek-V4-Flash-Preview 在 TB 相关集上进化的 harness，冻结后换不同模型推理：

| 推理模型 | 方法 | Acc ↑ | Pass@3 ↑ | Pass³ ↑ |
|---|---|---|---|---|
| GLM-5.2 | Baseline | 59.55 | 70.79 | 46.07 |
| GLM-5.2 | ModularRSI | 61.80 | 74.16 | 49.44 |
| MiniMax-2.5 | Baseline | 41.57 | 56.18 | 24.72 |
| MiniMax-2.5 | ModularRSI | 44.94 | 57.30 | 30.34 |
| DeepSeek-V4-Flash | Baseline | 47.57 | 58.43 | 30.34 |
| DeepSeek-V4-Flash | ModularRSI | 52.43 | 65.17 | 35.96 |

三个模型全部一致提升。这回应了 Self-Harness 暴露的问题（修改是模型特定的），说明对比分析 + 投票机制确实在往"机制级改进"上收敛，而不是抓住某个模型的怪癖打补丁。

---

## 🔬 我的判断

这篇论文最值钱的不是某个精巧的算法，而是**把 harness 进化的评估纪律立起来了**。

回到开头的痛点。之前这个方向的叙事很混乱：每篇论文都报大涨，但你不知道涨的是能力还是记忆。ModularRSI 做了三件规范化的工作——进化数据与评测集隔离、统一协议下复现 baseline、冻结后再评测。做完这三件事，数字反而更可信了。TB 2.0 上 5.6 个点的收益是在最严苛的协议下拿到的，含金量比那些两位数涨幅高。

方法论上，"对比轨迹 + 跨任务投票"这个组合我很认可，说到底就是给粗粒度奖励信号加了一个因果过滤器。模块化拆解 + 三道验证门则是把软件工程里"限定 blast radius"的直觉搬进了进化系统——消融里联合进化反而掉分的结果，是对这个设计最有力的辩护。

但问题也得说清楚：

**一，没有对比分析组件的独立消融。** 论文自己在局限性里承认了。模块化消融做得很漂亮，但"对比分析 vs 单侧分析"这条主线缺一张硬表。目前只有"随进化推进对比轨迹占比下降"的间接证据和案例研究撑着，说服力差点意思。

**二，进化规模偏小。** 2000 个实例只用了 240 个。论文归因于算力限制，可以理解，但"数据规模上去后收益是否持续"也因此成了个开放问题。另外那 2000 个任务的构建有人工标注参与，"benchmark-disjoint"的含金量依赖人工审查的质量——LLM 语义相似度筛选能挡多重，论文没有给量化数据。

**三，跟 AHE / Meta-Harness 的对比要辩证看。** 把对手放进自己的协议里打，赢是意料之中。更有趣的问题其实是：如果让 AHE 也用上 benchmark-disjoint 数据，差距还剩多少？论文没有回答。

工程上的启发倒是很直接：如果你在搭自己的 agent 自进化系统，先别急着设计花活，把"数据隔离、模块边界、验证门、进化历史"这四件事做了，大概率比换个更聪明的优化器收益大。另外 Observation Management 那个 22.50 步的结果也提醒手工 harness 的设计者——噪声观察过滤可能是最被低估的效率杠杆。

harness 这层东西，Lilian Weng 的判断是"模型和真实世界之间的接口跟模型本身一样重要"。ModularRSI 没有挑战这个判断，它做的事是让"接口的自我进化"变得可验证、可归因。这类规范化工作不性感，但一个领域要从炼丹走向工程，靠的就是这种论文。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
