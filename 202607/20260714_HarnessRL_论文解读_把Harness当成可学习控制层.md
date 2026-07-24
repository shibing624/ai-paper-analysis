# 把 Harness 当成可学习控制层：Offline RL 让冻结的 LLM 学会"何时检查"

你有没有这种感觉：同一个 GPT-4，外面套的执行框架（harness）不一样，出来的 agent 效果能差出十几个百分点？比如让它做编码任务，有的 harness 会先跑测试再提交，有的直接交卷；做研究问答时，有的会先检索证据再下结论，有的拍脑袋就给答案。

但我们现在的优化思路几乎都集中在改 prompt、改模型、或者干脆重写一套 multi-agent 脚手架。harness 这一层？默认它是基础设施，工程师手写完就完事了。arXiv:2607.05458 这篇论文想说的恰恰是：harness 本身就是一个可学习的控制层，而且——重点来了——**不用动 LLM 的一个参数**。

## 核心摘要

这篇论文把 LLM agent 的执行框架（harness）重新定义为一个有限时域 MDP：一个 64 单元的 MLP controller 在七个结构动作里挑一个（observe / retrieve / call-tool / draft / check / revise / submit），LLM executor 全程冻结。训练用 offline advantage-weighted regression（AWR），奖励只来自终端任务 rubric 分数。论文还把"最终任务质量 G"和"Harness Maturity Score（HMS）"彻底解耦——过程好不好和答案对不对，是两件事。

在 6 个受控领域加 2 个公开基准适配器（τ-bench retail、AgentBench DB-Bench）上，learned controller 普遍把 CheckBeforeSubmit 率从 0% 拉到 5.6% 到 17.8%，但最终任务质量只在 buffer 里"有高回报支撑"的设置上明显提升（τ-bench retail +18.2 pp、DB-Bench +13.2 pp、coding +10.0 pp）。**这篇论文最值钱的地方不是又刷了一个 SOTA，而是把"过程可控"和"结果可控"在 offline RL 框架下做了形式化分离，并且明确告诉你：offline buffer 的支持范围，决定了 process 改善什么时候能转成 final answer 提升**。

## 论文信息

- **标题**：Learning to Control LLM Agent Harnesses with Offline Reinforcement Learning
- **作者**：Haiwen Yi, Xinyuan Song
- **链接**：https://arxiv.org/abs/2607.05458
- **分类**：cs.LG / cs.AI
- **发表**：2026 年 7 月 5 日，17 页，7 张图

---

## 为什么 harness 值得学

作者在第一节抛了一个判断，我觉得挺关键的：

> The harness is therefore not a cosmetic wrapper around inference; it is part of the policy that determines how work is carried out

翻译成人话：一个 LLM agent 不等于"语言模型 + 提示词"。在部署系统里，harness 决定了模型什么时候观察环境、什么时候检索证据、什么时候调工具、什么时候起草答案、什么时候检查中间工作、什么时候停手。在保持 executor 不变的前提下，光改这一层控制逻辑，就能大幅改变 agent 行为。

现在的工程实践呢？基本是手写规则 + 提示模板 + 固定执行图，把 harness 当成静态工件搭起来。这里面有个根本问题：**对简单和困难任务编码相同的控制逻辑、对早期和晚期轨迹状态编码相同的控制逻辑、对成功和失败的中间尝试编码相同的控制逻辑**。

举两个例子就懂了：

- 固定规则"always check"在解决方案已经完成时浪费预算；
- 固定规则"submit after drafting"在草稿合理但没验证时直接翻车。

作者的原话更狠：

> The failure mode is often not missing model knowledge, but a poor sequence of external control decisions

失败常常不是模型知识不够，而是外部控制决策的顺序没排对。这其实和我之前在做项目时的体感一致：模型换了几个版本，prompt 改了几十遍，最后 agent 质量的天花板很多时候卡在"什么时候做这件事"上，而不是"做这件事本身"。

---

## 框架长什么样

先看 Figure 1 的全局直觉图——三层堆叠的结构非常清晰：

![图1：HarnessRL 框架概览——冻结的 LLM executor、训练式 controller、终端 rubric 奖励信号](https://arxiv.org/html/2607.05458v1/figures/g4_global_intuition.png)

*图1：HarnessRL 框架概览。最底层是冻结的 LLM executor（GPT-4、Claude 之类的），中间是轻量级 RL Harness Controller（一个小策略网络学习动作选择），最上面是终端的 Task Rubric Reward Signal。右侧标出 Outcome Distribution：大多数领域 ΔG 约等于 0，但在 buffer 支持下两个 benchmark 能涨 10 到 22 个百分点。底部那行字最关键——Decoupled Axes: harness improves process discipline without modifying the frozen LLM*

作者想传达的就是一件事：过程改进和结果改进在两个轴上，你可以只动其中一个。这个观察是后面所有实验设计的出发点。

---

## Harness MDP：把 harness 形式化成一个 MDP

作者做了三件事来形式化 harness。

**状态**：状态向量总结外部控制所需的信息——轨迹进度、草稿可用性、证据覆盖、工具输出、验证器反馈、最近的失败、剩余预算、上一个动作、领域特定的结构标志。关键约束：状态不暴露 LLM 的隐藏激活，也不更新 LLM。

**动作空间**：controller 在七个候选动作里挑一个：

| 动作 | 干啥 |
|------|------|
| `observe` | 观察环境 |
| `retrieve` | 检索证据 |
| `call-tool` | 调用工具 |
| `draft` | 起草答案 |
| `check` | 检查中间工作 |
| `revise` | 修改 |
| `submit` | 提交 |

我注意到这里没有作者之前工作中常见的 `decompose`、`backtrack` 之类的动作——动作集刻意保持小而正交。Controller 就是一个单隐藏层 MLP，64 个隐藏单元，softmax 输出七个动作，无效动作由领域适配器掩码。

**奖励**：每个领域把终端工件映射成标准 rubric 分数和标量任务奖励。**只给终端奖励**，中间动作不发奖金。

这一点论文里有 Theorem 1 撑腰：任务 rubric 奖励可以被基于势函数的中间信号增强而不改变任务最优策略集。论文实验取特例 Φ_t ≡ 0。但 Proposition 1 立刻给了个警告：直接给 check、revise 动作发奖金可能改变最优策略集，favor 那些"不改善最终工件、但过程形式好看"的策略。

其实，**为什么只用 terminal reward 不是偷懒，是有理论原因的**。check、revise、call-tool 有没有价值，只看它有没有真的改善终端任务质量。

---

## 训练：Offline Advantage-Weighted Regression

看 Figure 2 的算法流水线——标准的 AWR 五步：

![图2：Offline Advantage-Weighted Policy 算法流水线](https://arxiv.org/html/2607.05458v1/figures/g1_offline_aw_pipeline.png)

*图2：Algorithm 1 流程图。从 Behavior Buffer（每域 1600 条轨迹）开始，先做 Return Computation 得到 G_i，然后做 Advantage Estimation（A_i = G_i - mean(G | task)，b(task) 是同任务的均值回报），接着做 Exponential Weighting 得到权重 w_i = exp(A_i / β)，其中 β=0.2 且 clip 到 [0.1, 10.0]，最后 Policy Update 用加权负对数似然 ∑ w_i · log π(a|s) 做梯度下降，熵正则系数 β_H = 0.01*

目标函数写出来就是经典的 AWR 形式：

$$\mathcal{L}(\theta) = -\mathbb{E}_{(s,a)\sim\mathcal{D}}\left[\exp(A(s,a)/\beta) \cdot \log\pi_\theta(a \mid s)\right]$$

训练超参：Adam 优化器，学习率 10⁻³，batch size 256，20 个 epoch，每个领域跑 3 个独立随机种子。优化器层面的选择没有任何花哨，**重点是 controller 本身只是一个 64 单元的 MLP**——训一个 epoch 在笔记本上可能都能跑完。

---

## Harness Maturity Score：把过程好不好变成可测量的东西

论文最有意思的设计是把过程质量形式化成 HMS——一个覆盖七个过程事件的归一化加权诊断指标：

| 过程事件 | 说明 |
|----------|------|
| CheckBeforeSubmit | 提交前检查 |
| EvidenceBeforeClaim | 声明前获取证据 |
| TestBeforeSubmit | 提交前测试 |
| RevisionAfterFailure | 失败后修改 |
| ValidToolUse | 有效工具使用 |
| StopWhenSufficient | 充分后停止 |
| EarlySubmit（惩罚项）| 过早提交 |

**HMS 训练时从来不用作奖励，只在评估时报告**。这就把训练目标和评估指标切干净了——controller 学的是终端 rubric 分数，HMS 是事后看的过程有没有变好。

理论层面有 Theorem 2 撑这个分离：

- **Outcome bound（结果界限）**：$\mathbb{E}_{q_{AW}}[G] \leq G_B^\star$，最终质量的提升上限是 buffer slack $\sigma_B$——即 offline buffer 中最佳轨迹回报和均值回报的差。
- **Process identity（过程恒等式）**：$\Delta\Psi_B = \frac{\operatorname{Cov}_{\mu_B}(w, \Psi)}{\mathbb{E}_{\mu_B}[w]}$——过程行为的改变由过程统计量和 AW 权重之间的协方差决定。$\Delta\Psi_B \geq 0$ 当且仅当协方差非负。

这两条放一起就是论文的核心洞察：**过程改善可以广泛发生（只要和 AW 权重正相关），但结果改善需要 buffer 里真有更好的轨迹撑着**。Corollary 1 和 Proposition 2 还分别指出：过程统计量对优势单调时过程一定改善；终端回报和过程统计量负相关时过程仍可改善（因为过程改变由 AW 权重协方差决定，不是由回报相关性决定）。

---

## 实验结果：6 个受控域 + 2 个公开基准

### 主实验：过程普遍改善，结果看 buffer 脸色

每个受控领域 100 个人工标注任务（80 训 + 20 评），按难度分层（简单 20 / 标准 60 / 困难 20）。每个设置 3 种子 × 3 rollouts × 20 任务 = 180 个评估 episode。

最大的亮点是 CheckBeforeSubmit 率——这个指标在所有 8 个设置上都被学到的 controller 拉起来了：

| 域 | Base CBS | AW CBS |
|------|----------|--------|
| Knowledge-work | 0% | 5.6% |
| Coding | 0% | 17.8% |
| Research | 0% | 13.9% |
| Multi-tool | 0% | 8.9% |
| Long-memory | 0% | 10.0% |
| Planning | 0.6% | 6.7% |
| τ-bench retail | 0% | 17.2% |
| AgentBench DB-Bench | 0% | 16.7% |

从 1080 个 base episode 中的 1 个 CBS 增长到 1080 个 AW episode 中的 113 个，**整体 CBS 提升是 universal 的**——但这只是过程。

最终任务质量的提升就挑剔多了。只有 3 个设置显著：

| 设置 | Base G | AW G | ΔG（pp）| 95% CI | CBS |
|------|--------|------|---------|---------|-----|
| τ-bench retail | 0.337 | 0.519 | +**18.2** | [+15.1, +21.3] | 17.2% |
| AgentBench DB-Bench | 0.415 | 0.547 | +**13.2** | [+10.2, +16.2] | 16.7% |
| Coding | 0.712 | 0.812 | +**10.0** | [+5.9, +14.6] | 17.8% |

其余受控领域（Knowledge-work、Research、Multi-tool、Long-memory、Planning）的 ΔG 都在噪声范围内——具体到 Table 5 的数据：Knowledge-work +0.014（CI [-0.002, +0.031]，p=0.097）、Multi-tool -0.013（p=0.217）、Research -0.003（p=0.238）、Long-memory -0.003（p=0.449）、Planning +0.026（p=0.175）。图 3 把这种"过程涨 vs 结果涨"的对比画得非常直观：

![图3：8 个设置的 outcome 和 process 增益——adapters 和 coding 在结果上最强，过程改善更广](https://arxiv.org/html/2607.05458v1/x1.png)

*图3：Figure 3 主结果图。左侧是 final-quality change（ΔG），右侧是 process-maturity change（ΔHMS）。adapters（τ-bench retail、DB-Bench）和 coding 是 ΔG 最大的三组；ΔHMS 普遍正向，主要由 CheckBeforeSubmit 驱动*

### 消融：AW 不是 BC，也不是机械加 check

论文把 AW 和两个直观的 baseline 对比：

- **BC（Behavior Cloning）**：复制观察到的轨迹就够吗？
- **FC（Forced CHECK）**：统一插一个 check 就够吗？

| 设置 | AW lift | BC lift | FC lift | ΔΔ_AW-BC | ΔΔ_AW-FC |
|------|---------|---------|---------|-----------|-----------|
| Knowledge-work | +1.4 | -0.4 | +0.1 | +**1.8** | +**1.3** |
| Coding | +10.0 | -8.3 | +0.0 | +**18.3** | +**10.0** |
| Research | -0.3 | -4.2 | -0.2 | +**3.8** | -0.1 |
| Multi-tool | -1.3 | -6.8 | +0.3 | +**5.5** | -1.6 |
| Long-memory | -0.3 | -5.8 | +0.0 | +**5.5** | -0.3 |
| Planning | +2.6 | +0.0 | -2.3 | +**2.6** | +**4.9** |
| τ-bench retail | +18.2 | +8.2 | +0.1 | +**10.0** | +**18.1** |
| DB-Bench | +13.2 | +5.8 | -0.5 | +**7.4** | +**13.7** |

两个结论都很硬：

- **AW 在所有 8 个设置上都赢 BC**——这不是模仿能解释的；
- **AW 在 5 个设置上同时赢 BC 和 FC**（Knowledge-work、coding、planning、τ-bench retail、DB-Bench）——这也不是机械加个 check 能解释的。

特别有意思的是 Coding：BC lift 是 -8.3——纯模仿的策略反而比 base harness 还差。这说明 buffer 里收集的轨迹本身可能就有偏，AW 之所以能涨 +10.0，是它学会了**在什么状态加 check 比不 check 好**，而不是机械地模仿哪些轨迹加 check、哪些不加。Figure 4 把这种 policy-level 对比画成了散点图：

![图4：AW vs BC vs Forced CHECK 的策略对比散点图](https://arxiv.org/html/2607.05458v1/x2.png)

*图4：Figure 4 策略对比图。每个点是相对自己 Base 的 within-run lift。点越偏上说明 AW 优势越大——AW 在所有 8 个 setting 上都高于 BC 线（点全部在对角线左侧或上方），并在 5 个 setting 上同时高于 Forced CHECK*

### 为什么有的域不涨：Buffer slack 说了算

论文的 Table 9 给了一个非常漂亮的诊断——把每个域的 buffer slack σ_D 和 ΔG 摆在一起：

| 域 | σ_D | Base G | ΔG | ΔHMS |
|------|------|--------|-----|------|
| Knowledge-work | 0.279 | 0.450 | +0.014 | +0.059 |
| Coding | 0.217 | 0.712 | +**0.100** | +0.054 |
| Research | 0.060 | 0.314 | -0.003 | -0.026 |
| Multi-tool | 0.175 | 0.528 | -0.013 | +0.058 |
| Long-memory | **0.000** | 0.453 | -0.003 | +0.011 |
| Planning | **0.552** | 0.385 | +0.026 | +0.010 |

σ_D 是 offline buffer 中最佳轨迹回报和均值回报的差——也就是 buffer 里还有多少改进空间。

- Long-memory 的 σ_D = 0.000，**buffer 里根本没有比均值更好的轨迹**，AW 自然什么都学不到；
- Research 的 σ_D = 0.060，几乎也是没有，ΔG 也就 0；
- Planning 的 σ_D = 0.552，**buffer 里改进空间最大**，但 ΔG 只 +0.026——slack 大不等于一定涨，还得那些高回报轨迹里的动作真的能复现到新轨迹上；
- Coding 的 σ_D = 0.217，不算最大，但 ΔG +10.0 是受控域里最强的——**因为 buffer 里那些跑测试再提交的轨迹动作模式是清晰可学的**。

这一段是整篇论文最值钱的 insight：**offline RL 学到过程改善是普遍现象，学到结果改善是稀缺现象——而且这个稀缺性的根因在数据不在算法**。Figure 5 把这个 slack–outcome 关系画在了不同 verifier 配置下：

![图5：Buffer slack 与 final-quality change 在两种 verifier 下的关系](https://arxiv.org/html/2607.05458v1/x3.png)

*图5：Figure 5 散点图。横轴是 buffer slack σ_D，纵轴是 ΔG。空心圆是原始严格 verifier（包含接近饱和的确定性 coding rubric），实心圆是 calibrated structural verifier。只有当 σ_D 够大且 verifier 留有空间时，AW 才能把 slack 转化为 final-quality 提升*

---

## 三个我特别想吐槽的点

### 1. Research 域的过程改善被 EarlySubmit 抵消

Research 域是 8 个设置里**唯一 ΔHMS 为负**的（-0.026）。论文 Table 7 给出了原因：CheckBeforeSubmit 从 0% 涨到 13.9%，看起来很好；但 EarlySubmit 从 0% 涨到 25.0%——controller 在 research 任务里学到做更多事反而鼓励了差不多就交。其实，**controller 找到了一个 buffer 里看似更优、但实际更糟的策略模式**。

这种过程指标变好但实际策略在变差的情况在 RLHF 里也经常出现，论文至少把它显式量化了。但我个人的疑虑是：terminal rubric 评的是最终答案，EarlySubmit 涨了为什么 ΔG 还是 -0.003 而不是更负？可能 rubric 本身对完整流程和早期提交都给分了——这块论文没展开，我也不是 100% 确定。

### 2. 验证器饱和问题

作者在 Figure 5 里专门花了一节讨论：在原始严格编码 rubric 下，base harness 已经拿到 0.929，**接近天花板**，AW 提升只有 -0.006。只有切换到校准结构验证器（base 降到 0.712）才能看到 +10.0 的提升。

这其实是个老问题——**rubric 的天花板决定了你能看到多少 AW 的真实能力**。论文用了 calibrated structural verifier 来统一域之间的可比性，但这里有一个隐含的工程风险：你训的 controller 强不强，取决于你的 verifier 准不准。如果 verifier 给假阳性的高 reward，AW 会照单全收地学到垃圾。

### 3. Lifting LLM 这个隐含假设

论文明确说 LLM executor 保持冻结，但 base 和 exploratory harnesses 是怎么收集的 buffer？如果 buffer 本身就是在冻结 LLM 上 rollout 出来的，那 controller 学到的就是在这套 LLM 上怎么 harness 最优。**一旦换 LLM（比如从 GPT-4o 换到 Claude 4），controller 要不要重训？** 论文没明说。但我猜 controller 的策略和 LLM 的能力分布强相关，换 LLM 必然要重训或至少 fine-tune controller。

不过这是 offline RL 的通病，不算论文的锅。Figure 6 把 process-outcome 相关性在不同域下的差异画了出来，提醒你别被简单的跨域平均数骗了：

![图6：Within-policy 过程-结果相关性，不同域方向不一致](https://arxiv.org/html/2607.05458v1/x4.png)

*图6：Figure 6 散点图。每个域一个 marker，横轴是 final-quality G，纵轴是 HMS。pooled within-domain 估计是 0.183，但 naive pooling across all episodes 是 0.456——跨域异质性会显著高估 G 和 HMS 的真实相关性。这条图其实是在警告：别拿全 episode 的 correlation 当因果*

---

## 我的判断

**这篇论文最大的价值是把 harness 是可学习层从直觉变成了一个可操作的框架**。形式化做得很干净：Harness MDP、AWR、HMS、buffer slack 诊断，每一层都对应一个具体问题。

但我也要坦率地说几点：

- **结果层面，它不是 SOTA 故事**。在大多数受控域上 ΔG 都在噪声内。τ-bench retail 和 DB-Bench 涨了十几个点，**用的是 adapted Harness MDP scoring protocols 而不是官方 upstream 评分**——论文 Table 3 注释里也明确说了这一点。所以这两个数字的可比性要打折扣。
- **它更像一个机制分析论文而不是工程胜利论文**。最值钱的产出是 Theorem 2 加 buffer slack 诊断——告诉你 offline RL 什么时候能、什么时候不能改 final answer。这个 insight 比刷分更能影响工程决策。
- **和工业界的 agent 框架怎么对比？** LangChain、AutoGen、LangGraph、CrewAI 这些框架本质都在做 harness 编排，但都是手写的。论文的 controller 是 64 单元 MLP，**工程门槛远低于训一个 SFT 或 RLHF 的 LLM**。如果你手头有一批 agent rollout 数据（其实大多数 agent 公司都有），这个方法可以低成本试一下。

**什么时候值得上这个方法？**

- 你有一批 offline rollout 加 terminal 评分（甚至人工 rubric 都行）；
- 你想优化的是什么时候做这件事而不是做这件事的提示词；
- 你的 domain 有可定义的过程事件（check、test、revise、retrieve...）；
- 你不想动 LLM，也不想重训 prompt。

**什么时候不值得上？**

- 你的 buffer 本身没有高回报支撑（论文里 Long-memory 就是反例，σ_D=0.000）；
- 你的过程事件无法自动检测（HMS 需要 7 个事件 detector）；
- 你的 LLM 在 base harness 下已经接近 rubric 天花板。

Figure 7 是论文的"封底"图，把每个设置的 outcome 增益和 process 增益放在同一个散点空间里看：

![图7：每设置 outcome 增益 vs process 增益的散点关系](https://arxiv.org/html/2607.05458v1/x5.png)

*图7：Figure 7 散点图。public-benchmark adapters（τ-bench、DB-Bench）在两轴上都有显著增益；受控域 setting 主要在 process 轴上有进展，outcome 轴接近零。视觉上一眼就能看出两类 setting 的策略空间差异*

最后留一个开放问题：**offline AWR 把 controller 训出来了，那能不能再做一次 online fine-tuning 突破 buffer slack 的限制？** 论文没做，但这是最自然的下一步。如果有人能把 offline AW pretraining 加 online RL finetuning 这条路打通，那 harness 控制就真的变成 agent 训练的标准件了。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
