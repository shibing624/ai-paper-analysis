# 把 Rollout 预算花在刀刃上：TRACE 如何用一棵"树"重塑 Agentic RL 的探索效率

> 论文标题：TRACE: Tree Rollout Allocation for Contrastive Exploration
> arXiv ID：2606.11119
> 关键词：Agentic RL、RLVR、Rollout 预算分配、奖励对比、树状结构

## 一句话总结

TRACE 把多轮 agent 的 ReAct 交互过程建模成一棵"树"，然后用一个**全局到局部的两阶段预算分配框架**，把有限的 rollout 采样预算优先投到"最可能同时产生成功与失败结果"的位置上，从而显著放大训练信号中的**奖励对比（reward contrast）**，让 agentic 强化学习训得更快、更好。

---

## 一、问题从哪来：Agentic RL 里"白跑"的 Rollout 太多了

近两年，用**可验证奖励的强化学习（RLVR, Reinforcement Learning with Verifiable Rewards）**来训练大模型 agent 已经成为主流路线。它的核心循环很简单：让模型对同一个 prompt 多次采样（rollout），根据最终结果（答案对不对、任务成不成功）给一个奖励，再用 GRPO 这类算法做组内对比、更新策略。

这里的"组内对比"是关键。GRPO 之所以有效，靠的是同一个 prompt 的多条 rollout 之间存在**奖励差异**——有的成功有的失败，梯度才有方向。一旦一组 rollout 全成功或全失败，组内 advantage 全为零，这一组样本就**白跑了**，对策略更新毫无贡献。

论文把这个核心痛点称为**奖励对比不足（insufficient reward contrast）**，并指出它来自两个被忽视的层面：

**1. Prompt 层面的信息量浪费。** 过简的题目模型每次都对，过难的题目模型每次都错，这两类 prompt 都产生低方差反馈。过往工作（如课程学习类方法）已经注意到这一点，会去筛选"难度适中"的 prompt。

**2. Prefix 层面的信息量被完全忽略。** 这是 TRACE 最关键的洞察：在一个多轮 rollout **内部**，不同的中间状态（prefix，即"走到某一步时的历史轨迹"）其实蕴含着**截然不同的对比潜力**。一个 outcome-only 的奖励，会把同一条 rollout 上的每一个决策都赋予**相同的终端评估**——哪怕其中某些 turn 是关键的胜负手，某些 turn 只是无关紧要的过渡。换句话说，传统方法只在"题目"这个粗粒度上做文章，却看不到"同一道题、走到不同岔路口时"信息量的天壤之别。

下面这张对比图很直观地说明了问题：

![Uniform Rollout vs Adaptive Rollout 的树状对比](https://arxiv.org/html/2606.11119v1/x1.png)

- **左侧（Uniform Rollout，均匀分配）**：对每个 prompt 平均撒预算，结果是很多子树要么叶子全是 ✓（全成功），要么全是 ✗（全失败），形成 **Weak Contrast（弱对比）**，梯度信号被浪费。
- **右侧（Adaptive Rollout，自适应分配）**：把预算重新分配后，让同一个 anchor（锚点，即某个中间状态）的后代里**同时出现成功与失败**，形成 **Strong Contrast（强对比）**。图中绿色边代表成功路径，红色边代表失败路径，✓/✗ 标记终端结果。

TRACE 要做的，就是把右图这种"高对比"的采样结构，系统化、可计算地生产出来。

---

## 二、核心建模：把 ReAct 轨迹变成一棵树

TRACE 的第一步是换一个看世界的角度——**把 agent 的交互轨迹建模成树**。

在 ReAct 范式里，agent 的一次交互由若干个 turn 组成，每个 turn 是一个"思考-行动-观察"三元组。TRACE 把每个 turn 封装为树上的一个**节点**：

$$n_t := \langle \tau_t, a_t, o_t \rangle$$

其中 $\tau_t$ 是 thought（思考），$a_t$ 是 action（行动），$o_t$ 是 observation（环境观察）。走到第 $t$ 步时累积的历史记为 $\mathcal{H}_t$。整个 prompt 本身是这棵树深度为 0 的根节点（root），作为所有 rollout 的公共 anchor。

这样建模之后，"在某个中间状态继续采样"就等价于"从树上某个节点向下扩展分支"。**Prefix 层面的信息量差异**，也就自然地变成了"树上不同节点的扩展价值差异"——可以被量化、被优化。

---

## 三、TRACE 的方法：Mixed-Reward Contrast 与两阶段分配

### 3.1 指导原则：Mixed-Reward Contrast（混合奖励对比）

TRACE 的预算分配遵循一个核心原则：

> **把预算分配给那些"后代集合最可能同时包含成功与失败结果"的 anchor。**

直觉很清楚：如果一个节点往下扩展，结果要么全成功要么全失败，那这些采样就是浪费；只有当它的后代成败混合时，才能贡献出有效的组内对比。所以 TRACE 衡量一个节点的价值，本质上是在衡量它"产生混合结果的概率"。

### 3.2 整体框架

![TRACE 两阶段框架图](https://arxiv.org/html/2606.11119v1/figures/figure2/figure2_v3.png)

如上图，TRACE 的工作流分为三大块：

1. **Stage 1 — Global Root Allocation（全局根分配）**：用 Root Allocation Solver，根据 Predictor 的估计，把总预算 $\mathcal{M}$ 在各个 prompt（root）之间分配为 $\{m_1, ..., m_B\}$，先产出一批 **Bare Rollouts（裸 rollout）**。
2. **Stage 2 — Local Prefix Expansion（局部前缀扩展）**：用 Prefix Expansion Solver，在已有 rollout 内部挑选高价值的 prefix 节点，对它们做额外的 continuations（续写）$\{K_{i,j,t}\}$。
3. **Predictor & Policy Updates（预测器与策略更新）**：计算递归回溯目标 → 更新 Predictor $\tilde{V}_\psi$ → 做 Tree-Aware 的策略优化 → 更新策略模型 $\pi_\theta$。

这就是论文反复强调的 **Global-to-Local（全局到局部）、initialize-then-expand（先初始化再扩展）** 的两阶段设计。

### 3.3 Stage 1：全局根分配

第一阶段在 prompt 之间分配预算。对每个 prompt $x_i$，给它分配 $m$ 条 rollout 的价值函数定义为：

$$V_{\mathrm{root}}(x_i, m) = 1 - v_i^m - (1-v_i)^m$$

这里 $v_i$ 是该 prompt 的预估成功率。这个式子的含义非常优雅：$v_i^m$ 是 $m$ 条全成功的概率，$(1-v_i)^m$ 是 $m$ 条全失败的概率，**$1$ 减去这两者，正好就是"至少出现一次成功且至少出现一次失败"的混合概率**——也就是这一组能贡献有效对比的概率。求解器会在这个目标下，把预算优先给那些"投入边际收益最高"的 prompt。

### 3.4 Stage 2：局部前缀扩展

第二阶段在单条 rollout 内部，挑选有价值的 prefix 继续向下扩展。某个 prefix 节点扩展 $k$ 条 continuation 的价值定义为：

$$V_{\mathrm{pref}}(i,j,t,k) := 1 - \left[r_{i,j}\tilde{V}_\psi(\mathcal{H}_{i,j,t}) + (1-r_{i,j})(1-\tilde{V}_\psi(\mathcal{H}_{i,j,t}))\right]^k$$

其中 $\tilde{V}_\psi(\mathcal{H}_{i,j,t})$ 是共享预测器对"从历史 $\mathcal{H}_{i,j,t}$ 出发的条件成功概率"的估计，$r_{i,j}$ 是该条 rollout 的实际结果。形式上与 Stage 1 同源——依旧是在最大化"混合结果出现概率"，只不过粒度从 prompt 下沉到了 prefix。

### 3.5 共享预测器 $\tilde{V}_\psi$

支撑两个阶段决策的，是一个**可泛化的共享预测器** $\tilde{V}_\psi$，它负责估计任意状态下的条件成功概率。它的训练目标通过树上的**递归回溯**得到：

$$\widehat{V}(y) = \frac{1}{n_y}\sum_{c\in\mathcal{C}(y)} n_c \widehat{V}(c)$$

即一个节点的目标值是其所有子节点 $\mathcal{C}(y)$ 价值的、按访问次数 $n_c$ 加权的平均。

需要特别强调的是：**这个预测器只服务于 allocator（分配器）做预算决策，本身不参与策略优化**，因此不会污染 policy 的梯度，保持了训练的纯净性。

### 3.6 三个理论命题

论文从理论上为该框架提供了支撑，给出三个命题：

1. **Prefix 信息能改善组难度预测**：引入 prefix 层面的信息后，对一组 rollout 难度的预测更准。
2. **Prefix 的不确定性等价于剩余对比潜力**：一个状态的对比潜力，正比于其成功概率的伯努利方差 $V_t^\pi(1 - V_t^\pi)$——成功率越接近 0.5，对比潜力越大。这为"为什么要把预算投到中等难度状态"提供了数学解释。
3. **激活分配优于均匀分配**：在理论上证明了 TRACE 的自适应（激活）分配策略，其期望对比收益严格优于均匀分配。

---

## 四、实验结果：又快又好

TRACE 在 **数学推理（Math Reasoning）** 和 **多跳问答（Multi-Hop QA）** 两类 agentic 任务上，基于 Qwen3-8B 和 Qwen3-14B 做了系统验证，对比的 baseline 包括 ReAct、GRPO、PCL（Prompt Curriculum Learning）和 TreePO。

### 4.1 主结果

![训练准确率曲线](https://arxiv.org/html/2606.11119v1/x5.png)

- **数学推理**：Qwen3-8B 上 GRPO 70.0 → TRACE **71.1**；Qwen3-14B 上 73.5 → TRACE **74.9**。
- **多跳问答**：Qwen3-14B 上 TRACE 比 baseline 提升 **2.8 个点**。

提升幅度看似温和，但结合下面的"效率"指标看，含金量就出来了。

### 4.2 Effective Ratio：核心效率指标的大幅跃升

论文定义了一个直击要害的指标 **Effective Ratio（有效比例）**：一个 batch 内，rollout 树里**同时包含成功与失败终端叶子**的 prompt 所占的比例——这正是"没白跑、能贡献梯度"的样本占比。

![Effective Ratio 对比](https://arxiv.org/html/2606.11119v1/x2.png)

- Qwen3-8B：**26.8% → 60.6%**
- Qwen3-14B：**34.7% → 59.7%**

有效样本占比直接翻倍，这意味着在相同采样预算下，TRACE 喂给策略优化器的"有用对比信号"多了一倍以上。这正是它的效率来源。

### 4.3 消融实验：两阶段都不可或缺

Table 1（Qwen3-8B，Multi-Hop QA）对比了 Stage 1 / Stage 2 各自用 Uniform（均匀）还是 Active（自适应）：

| Stage1 / Stage2 分配方式 | 指标 A | 指标 B |
|---|---|---|
| Uniform / Uniform | 49.5 | 42.8 |
| Active / Active（TRACE） | **50.6** | **52.3** |

两个阶段都换成自适应分配后，第二项指标从 42.8 跃升到 52.3，提升近 10 个点，充分说明**全局与局部两层分配缺一不可**。

### 4.4 预算兼容性：在不同采样配置下都稳健

Table 2 验证了 TRACE 在不同 (总预算, 每 prompt rollout 数) 配置下的表现：

| 预算配置 (M, m) | TRACE 指标 A | TRACE 指标 B |
|---|---|---|
| (512, 2) | 49.7 | 42.4 |
| (512, 6) | 50.3 | 47.8 |
| (1024, 2) | 50.6 | 52.3 |

无论是小预算还是大预算配置，TRACE 都能稳定发挥，说明它的分配机制对预算规模具有良好的适应性。

---

## 五、为什么这篇值得读

1. **视角创新**：第一次系统性地把"奖励对比"从 prompt 粒度下沉到 **prefix 粒度**，并用树结构把这个抽象概念变得可计算、可优化。这是对 GRPO 类方法本质的一次深刻补充。

2. **框架优雅**：Global-to-Local 两阶段分配，配合一个统一形式的"混合结果概率"价值函数（Stage 1 和 Stage 2 同源），数学上自洽，工程上可落地。

3. **效率证据扎实**：Effective Ratio 翻倍这个指标，比单纯的准确率提升更能说明问题——它直接量化了"采样预算的利用率"，这正是大规模 RL 训练中最稀缺的资源。

4. **理论与实证兼备**：三个命题从理论上回答了"为什么自适应分配更优""为什么要瞄准中等难度状态"，与实验结果相互印证。

---

## 六、局限与思考

- **预测器的额外开销**：维护并训练共享预测器 $\tilde{V}_\psi$ 会引入额外计算与工程复杂度，论文虽强调它不参与策略优化，但其估计质量直接决定分配效果，预测不准时可能误导预算。
- **树状 rollout 的实现成本**：从线性 rollout 切换到树状 rollout，对训练基础设施（KV cache 复用、分支管理）提出了更高要求。
- **任务范围**：目前验证集中在数学推理与多跳问答，对更长程、更开放的 agentic 任务（如复杂工具使用、长程规划）的泛化性仍有待观察。

---

## 结语

TRACE（arXiv:2606.11119）的核心贡献，是用一棵"树"重新组织 agent 的探索过程，并把有限的 rollout 预算精准地投向"最可能产生强对比"的位置。它告诉我们：在 agentic RL 里，**采样不是越多越好，而是越"对比鲜明"越好**。当有效样本占比从 27% 提升到 60%，效率的提升便不言而喻。对于任何在做大规模 RLVR 训练、苦于"白跑 rollout"的团队来说，这套"把预算花在刀刃上"的思路都极具借鉴价值。
