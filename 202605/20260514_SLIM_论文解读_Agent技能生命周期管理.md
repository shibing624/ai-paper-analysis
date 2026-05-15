# SLIM：Agent 的"技能仓库"不应该一味变大或变小——技能要有生命周期

## 核心摘要

LLM Agent 的 skill bank 这两年是个热门话题。给 Agent 配一堆外部 skill（可以是 prompt module、procedural rule、工具调用模板），Agent 在执行时检索相关 skill 注入到 context 里——这套思路确实管用。

但有个问题一直让我困惑：**skill bank 到底应该越大越好，还是应该把 skill "internalize" 到模型参数里，最后变成 zero-skill inference？**

现在的方法分成两派。**SkillRL** 那一派认为 skill 是 persistent augmentation，越积越多越好。**Skill0** 那一派认为 skill 是 temporary scaffold，最终应该全部消化进参数里。两派都假设 skill set 是 **monotonic**——要么一直涨要么一直减。

这篇 SLIM 直接戳穿这个二元假设：**最优 skill set 应该是非单调的、task-dependent、stage-dependent**。论文提出三个 lifecycle 操作——**Retain / Retire / Expand**——通过 leave-one-skill-out validation 估计每个 skill 的 marginal external contribution (MEC)，动态决定哪些保留、哪些淘汰、哪些扩展。

在 ALFWorld 和 SearchQA 两个 benchmark 上，SLIM 比最强 baseline 平均提升 **7.1 个点**。在 ALFWorld 上 SLIM† 达到 87.5，比 SkillRL† 的 75.0 高 12.5 个点。最关键的训练 dynamics 显示：active skill 数量先扩到 46，最后稳定在 21，**既不是 SkillRL 那种一路涨到 73，也不是 Skill0 那种降到 0**。

读完我的感觉是——这篇论文做了一件**简单但被忽视的事**：把"管理 skill 集合"提升为一个 first-class 的优化变量。这个 framing 比 method 本身更值得记住。

---

## 论文信息

- **标题**：Dynamic Skill Lifecycle Management for Agentic Reinforcement Learning
- **作者**：Junhao Shen, Teng Zhang, Xiaoyan Zhao, Hong Cheng
- **机构**：香港中文大学、佛罗里达大学
- **arXiv**：https://arxiv.org/abs/2605.10923
- **代码**：https://github.com/ejhshen/SLIM

---

## 问题：单调假设到底有什么问题？

我先复述一下两派的逻辑：

**SkillRL 派**：skill 越多，agent 能力覆盖越广。每次遇到新问题，generate 一个新 skill 加进去。bank 会持续增长。

**Skill0 派**：skill 只是 temporary scaffold。训练过程中，skill 应该被 internalize 到 model 参数里，最终达到 zero-skill inference——agent 不需要外部 skill 也能解决问题。

这两派各自都有 reasonable 的 motivation，但都隐含一个假设：**最优 skill set 是 monotonic 的**。要么持续加，要么持续减。

SLIM 的核心 critique 是：**这个假设过于 restrictive**。原因：

1. **Model 参数容量是有限的**。不是每个 skill 都应该被强行塞进参数。Long-tail、low-frequency 的 procedure 留在 external 反而合理
2. **Skill 之间的 marginal value 不均匀**。有些 skill 用得多、值得 internalize；有些用得少但关键，留在 external 反而更可靠
3. **Skill 太多本身有副作用**。大 skill bank 会引入 routing noise、长 context 影响 skill 选择可靠性

![图1：训练过程中 ALFWorld 上 skill 数量与成功率的关系](https://arxiv.org/html/2605.10923v1/x1.png)

*图1：横轴是训练过程中 active skill 的数量，纵轴是 validation success rate。SkillRL（粉色）skill 数从 38 涨到 73 一直涨；Skill0（绿色）从 38 降到 0；SLIM（蓝色）走的是非单调路径——先扩到 46 再降到 21，最终稳定。最关键的是 SLIM 最终成功率最高——既不是"全保留"也不是"全消化"。*

所以核心问题不是"保留还是消化"，而是**"如何确定 agent 的 external skill 边界"**。

---

## SLIM 的核心机制

![图2：SLIM 框架总览](https://arxiv.org/html/2605.10923v1/x2.png)

*图2：三步走——(1) Hierarchical Skill Retrieval：从当前 active set 里检索 task-conditioned skill；(2) Marginal External Contribution Estimation：用 leave-one-skill-out validation 估计每个 active skill 的边际价值；(3) Dynamic Skill Lifecycle Management：根据 MEC 决定 retain / retire / expand，与 GRPO 交替优化。*

### 数学 framing

把 skill set 优化形式化成一个 capacity-constrained allocation 问题：

$$\max_{\theta, \mathcal{A}, \mathcal{I}} \mathbb{E}_{x \sim \mathcal{X}}[\text{Perf}(x; \pi_\theta, \mathcal{A})] - \Omega(\mathcal{A}) \quad \text{s.t.} \quad \sum_{s \in \mathcal{I}} m(s) \leq \mathcal{C}_\theta$$

其中：
- $\mathcal{A}$：active external skill set
- $\mathcal{I}$：internalized skill set（已经吸收到参数里）
- $\Omega(\mathcal{A})$：维持 external skill 的"成本"（context 长度、routing noise 等）
- $m(s)$：internalize 一个 skill 的参数代价
- $\mathcal{C}_\theta$：model 参数容量

直接优化这个 mixed-space 问题不可行（$\mathcal{A}$ 是离散变量，$\Omega$ 是 black-box）。SLIM 做了三个 tractable approximation。

### Step 1：Hierarchical Skill Retrieval

把 skill 分成 general 和 task-specific 两层。对每个 task $x$（type $k$），检索：

$$\mathcal{Q}_t(x) = \text{TopK}\left(\{s \in \mathcal{A}_t^k : \cos(e_x, e_s) \geq \tau_{\text{emb}}\}, K\right)$$

这把全局 skill 选择变成了 task-conditioned 检索。检索阈值 $\tau_{\text{emb}}=0.45$，最多 K=3 个 task-specific skill 注入 prompt。

### Step 2：Marginal External Contribution (MEC) 估计

这是 SLIM 的核心 idea。对每个 active skill $s$，用 leave-one-skill-out 估计它的边际价值：

$$\Delta_t(s) = \text{Perf}(\mathcal{V}_t(s); \mathcal{A}_t) - \text{Perf}(\mathcal{V}_t(s); \mathcal{A}_t \setminus \{s\})$$

简单说就是：**在那些用到 skill $s$ 的 validation task 上，去掉 $s$ 之后性能掉多少**。

注意这是 **local 估计**——不是全局 ablation，而是在当前 policy 和 active set 下，移除单个 skill 的影响。用 EMA 平滑减少 audit noise：

$$\bar{\Delta}_t(s) = \alpha \Delta_t(s) + (1-\alpha) \bar{\Delta}_{t-1}(s)$$

**MEC > 0**：policy 仍然依赖这个 skill 提供 external support
**MEC ≈ 0 或 < 0**：skill 可能已经被 internalize，或者变 redundant 了

### Step 3：Retain / Retire / Expand

三个操作的触发条件：

**Retain**：MEC 显著正
$$\bar{\Delta}_t(s) \geq \tau_{\text{keep}} \Rightarrow s \in \mathcal{A}_{t+1}$$

**Retire**：MEC 持续低 + 足够 exposure
$$\bar{\Delta}_t(s) < \tau_{\text{retire}}, u_t(s) \geq n_{\min}, \ell_t(s) \geq p \Rightarrow s \notin \mathcal{A}_{t+1}$$

这里 $u_t(s)$ 是累计 exposure 次数，$\ell_t(s)$ 是 low-contribution streak。这两个条件保护 low-frequency skill 不被过早删除——一个 skill 用得少，不代表它没用。

**Expand**：当前 active skill 持续 fail
$$\text{Perf}(\mathcal{V}_t(s); \mathcal{A}_t) < \tau_{\text{expand}}, N_t(s) \geq n_{\text{expand}}, \bar{\Delta}_t(s) < \tau_{\text{keep}} \Rightarrow \mathcal{A}_{t+1} = \mathcal{A}_t \cup \{s_{\text{new}}\}$$

也就是说：某个 skill 被频繁路由但失败率高，说明这个区域 capability 覆盖不足，触发新 skill 生成（用 Anthropic-style 的 skill-creator 工作流）。

### 交替优化

每 d=10 个 GRPO step 做一次 lifecycle audit。GRPO step 里固定 active set 优化 policy；audit step 里固定 policy 调整 active set。

---

## 实验结果

### 主表对比

| Method | ALFWorld Avg | SearchQA Avg |
|--------|--------------|--------------|
| GRPO | 67.2 | 37.5 |
| GRPO† | 68.8 | 37.9 |
| EvolveR | 39.8 | 37.4 |
| SkillRL† | 75.0 | 38.1 |
| Skill0 | 74.2 | 39.3 |
| **SLIM** | 72.7 | **41.0** |
| **SLIM†** | **87.5** | **41.0** |

注：† 表示推理时使用 retrieved external skill。

### 关键观察

**ALFWorld 和 SearchQA 是两种不同 regime**：
- **ALFWorld**：SLIM† 87.5 vs SLIM 72.7，差距巨大。说明这个 domain 有大量 long-horizon procedural behavior，必须保留外部 skill。比 SkillRL† 高 12.5 个点
- **SearchQA**：SLIM 和 SLIM† 都是 41.0，差距几乎为零。说明 skill 的价值已经通过训练 internalize 进了 policy，不需要 inference-time 外部支持

这个差异印证了论文的核心 claim——**skill-based RL 的 endpoint 是 task-dependent 的**。某些任务需要保留外部 procedural skill，某些任务可以基本 internalize。

### 训练 dynamics

![图3：ALFWorld 上的训练曲线](https://arxiv.org/html/2605.10923v1/x3.png)

*图3：左图是 with-skill 和 no-skill 的 validation 曲线。Skill0（右图青色）在 epoch 90 active skill 降到 0 之后，validation 从 92.2% 跌到 76.6%——这是论文非常关键的一个发现：**强制 zero-skill inference 会损失 long-tail 的外部支持**。SLIM 最终稳定在 21 个 skill，no-skill 性能从 29.7% 涨到 84.4%（说明 policy 真的学了东西），with-skill 性能 90.6%。*

这张图说明 SLIM 的核心 thesis 是对的：**policy learning 和 external skill 不是对立的**——可以同时进行，让 policy 学到通用能力，同时保留 long-tail 的外部 skill。

### Ablation

| Variant | ALFWorld |
|---------|----------|
| SLIM | 87.5 |
| w/o Retirement | 73.4 (-14.1) |
| w/o Expansion | 78.9 (-8.6) |
| Random Audit | 68.8 (-18.7) |
| Fixed Active Set Size | 75.6 (-11.9) |

读这张表的几个观察：

**观察 1**：去掉 Retirement 掉 14 个点，说明**纯加 skill 是有 cost 的**——不是越多越好。这印证了 $\Omega(\mathcal{A})$ 的 monotonicity 假设

**观察 2**：去掉 Expansion 也掉 8 个点，说明**有些 task region 需要新 skill 才能覆盖**，剪枝单独搞不定

**观察 3**：Random Audit 掉 18 个点——这是最大的 drop，说明**MEC 估计是关键**。如果 retain/retire 是随机的，那这套机制就完全没用

**观察 4**：Fixed Active Set Size 掉 12 个点，说明**关键不是 skill 数量，而是 which skills**。这一点很重要——SLIM 不是简单的 prompt budget 控制

![图4：训练 reward 曲线对比](https://arxiv.org/html/2605.10923v1/x4.png)

*图4：完整 SLIM（蓝色）和各 ablation variant 的 training reward。完整 SLIM 收敛最快、最稳。Random Audit（红色）一直 stuck 在低 reward，说明随机 audit 完全 break 了系统。*

### Skill lifecycle 的 case study

![图5：技能生命周期 case study](https://arxiv.org/html/2605.10923v1/x5.png)

*图5：每个点是一个 skill，横轴是 selection count（被路由次数），纵轴是 MEC。**Retained skills**（绿色）：高频 + 高 MEC——核心 procedure，必须保留。**Retired skills**（橙色）：高频 + 低 MEC——可能已经 internalize 了。**Internalized 候选**（其他）：低频 + 低 MEC——本来用得就少，且 MEC 不高。*

这个 case study 给了一个非常实用的 insight——**MEC 是一个比"使用频率"更准确的 skill 价值指标**。一个 skill 用得多不代表它有价值（可能 policy 已经学会了，但还是按惯性调用）；一个 skill 用得少也不代表它没用（可能它解决的是 long-tail 关键 case）。

---

## 我的判断：值不值得读？

**强推**，特别是如果你在做 agent skill / tool retrieval / RAG-like 增强系统。

**亮点**：

1. **问题 framing 非常 sharp**。"Skill set 是 monotonic 还是 non-monotonic"这个问题，之前没人正经讨论过。SLIM 把这个 problem statement 摆出来本身就值这篇 paper
2. **MEC via leave-one-skill-out** 是一个简单但有效的 trick。完全 tractable，且能直接关联到"这个 skill 当前还有没有用"
3. **Retain/Retire/Expand 三个操作的非对称设计**：Retire 需要 cumulative exposure + low-streak 保护，Expand 需要持续失败触发——这些细节让 lifecycle 不会激进或过保守
4. **统一了 SkillRL 和 Skill0**——通过参数调整，SLIM 可以退化成任何一派。这是个好的理论性质
5. **训练 dynamics 的可视化非常有说服力**。Skill0 在 epoch 90 跳水那张图基本就是"强制 zero-skill"思路的一个 counterexample

**问题**：

1. **MEC 估计 cost 不低**。每个 audit 要做 leave-one-skill-out validation——如果有 N 个 active skill，audit 一次就是 N 次额外 inference。论文用 budget M=4 控制，但 M 偏小有 sampling bias
2. **Skill 之间的依赖性没考虑**。如果 skill A 和 B 必须配合使用，那 leave-one-out 估计会低估每个单独 skill 的价值。Combined MEC（leave-two-out）会更准但 cost 更高
3. **Expand 用 skill-creator 工作流——这个过程没充分讨论**。新 skill 是从 failure case 生成的，质量怎么保证？没说
4. **只在 ALFWorld 和 SearchQA 上测**。这俩 benchmark 风格差异很大但都不算特别复杂的 agentic 场景。希望看到 webagent、software engineering agent 上的表现

**对工程实践的启发**：

- **不要无脑积累 skill bank**。每个 skill 都有 context 成本和 routing 噪声成本，要定期清理
- **不要把 skill 全部强制 internalize**。Long-tail capability 留在 external 反而更稳定
- **用 leave-one-out 评估 skill 价值** 是个通用 trick，可以推广到 retrieval-based 系统的 component 评估
- **训练阶段就要管理 skill set**，不要等部署时再发现 skill bank 噪声太大——那时候 policy 已经训完了

---

## 收尾

回头看 agent skill 这个方向：从 Voyager 用 LLM 自动生成 skill，到 ExpeL 把经验抽象成 skill，到 SkillRL 把 skill 注入 RL training，再到 Skill0 把 skill internalize 进 policy，再到 SLIM 提出"skill 应该有 lifecycle"——可以看出这个方向在快速收敛到一个共识：**skill 是 agent 能力组织的核心 abstraction，但它不应该是 monotonic 增长或消失的**。

下一步值得追的问题：

1. **跨任务的 skill transfer**：A 任务训练出来的 skill bank，在 B 任务上 zero-shot 应用效果如何？SLIM 的 lifecycle 机制能否帮助 cross-task generalization？
2. **MEC 的 sample-efficient 估计**：能不能用 importance sampling 或者 multi-armed bandit 思路减少 audit 成本？
3. **Skill 之间的依赖建模**：Skill A 和 B 一起用才有价值的情况下，单独 leave-one-out 会错杀。这块需要 hierarchical 或 cluster-level 的 audit
4. **Inference cost vs accuracy 的 Pareto frontier**：SLIM 保留 21 个 skill 和 SkillRL 保留 73 个 skill，inference cost 差距巨大。这块论文没充分讨论

如果你也在做 agent 系统，这篇文章的"skill 要 lifecycle 管理"这个思想值得放进你的工程 toolkit。我自己回去要重新审视下我们之前那个 skill bank——估计有一半是该 retire 的。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
