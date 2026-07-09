---
title: 把 prompt 调成动态难度：LLM-as-a-Tutor 让 rubric RL 自己续命
date: 2026-07-09
arxiv_id: 2607.04412
tags: [LLM, RL, Non-Verifiable, Rubric, Prompt Adaptation]
---

你有没有过这种感觉：你给 RL 训练准备了一堆 prompt，跑了两轮发现 reward 在涨，但模型生成质量肉眼可见地卡住了——reward 越来越漂亮，分布越来越集中，标准差越来越小，policy gradient 实际上在原地踏步。

这就是这篇论文（arXiv 2607.04412）想捅破的那层窗户纸：**在用 LLM judge + rubric 做奖励的非可验证 RL 里，policy 涨到一定水平之后，静态 prompt 集会系统性地"过易"，所有 rollouts 在 rubric 上都得满分，judge 再聪明也分不出高下，reward 信号就崩了。** 作者管这叫 *policy-prompt mismatch*。

解决思路也很反直觉——既然 prompt 太简单导致 rollout 没区分度，那就让 LLM 在每轮训练里**先判别哪些 prompt 已经"过易"，再给它们加一句新约束，让难度跟着 policy 长**。这就是 LLM-as-a-Tutor。

---

## 核心摘要

把 LLM judge 的角色从"打分器"扩展成"导师"：同一模型既当 *examiner*（成对比较两个 rollout，判断 prompt 是否对当前 policy 已无挑战性），又当 *generator*（在原 prompt 后面追加一条原子约束，配套加一条 rubric 准则）。**追加而不是改写**保证难度单调上升，且不破坏种子任务分布。

在 FollowBench、AdvancedIF、InfoBench 三个复杂指令跟随 benchmark 上，以 Qwen3-1.7B-Thinking 为 policy、Qwen3-8B-Thinking 为 tutor + judge，平均分 **51.96**，比静态 rubric baseline（Base rubrics 50.51、WildChecklists 50.72）高 1.4 个点，比 policy-adaptive rubric baseline（51.04）高 0.9 个点，比 policy-adaptive prompt 修改方法 EVA（51.04）高 0.9 个点。在六个细粒度指标上拿下五个第一。

这是个"小而美"的工作：核心 idea 简单、工程可落地、平均提升 0.9 个点不算炸裂，但论证链条完整、消融扎实、并且指出了一个被忽视的方向——在 rubric-based non-verifiable RL 里，**prompt 本身也是 policy-aware 的必要维度**，光改 rubric 不够。

---

## 论文信息

- **标题**：LLM-as-a-Tutor: Policy-Aware Prompt Adaptation for Non-Verifiable RL
- **作者**：Yujin Kim, Namgyu Ho, Sangmin Hwang, Joonkee Kim, Yongjin Yang, Sangmin Bae, Seungone Kim, Jaehun Jung, Se-Young Yun, Hwanjun Song
- **机构**：KAIST AI, LG AI Research, Seoul National University（依据作者署名常见机构推断，论文未在首页完整列出；arXiv ID 2607.04412）
- **链接**：https://arxiv.org/abs/2607.04412
- **发表日期**：2026-07-05

---

## 问题：为什么静态 prompt 会"过期"？

先讲清楚背景。在 math、code 这类**可验证**任务里，RL 拿到的 reward 信号天然是"答对/答错"——能直接写个 checker 出来。但**非可验证**的开放指令（写邮件、写故事、写代码解释……）没有标准答案，传统做法是训练一个 Bradley-Terry RM 打分。RM 评分太粗，容易被长度、套话等表面特征钻空子，复杂指令任务上很容易卡 plateau。

最近一年方向变了：用 **LLM judge + per-prompt rubric**（一份针对该 prompt 的"评分细则清单"）打精细分，代表工作有 Rubrics as Rewards（Gunjal et al., 2026）、WildChecklists（Viswanathan et al., 2025）、DR Tulu（Shao et al., 2025）。rubric 确实把判别力拉上去了，但**这套方案有个隐藏前提被大家默认接受了**——

> 训练 prompt 必须 *对当前 policy 有挑战性*。

所谓"有挑战性"，是说同一个 prompt 在当前 policy 下能采出 G 个 rollout，这 G 个 rollout 在 rubric 上得有分数差异（方差非零）。**否则 GRPO 的 advantage 估计是 0，policy gradient 直接白算。**

![图 1：LLM-as-a-Tutor 的整体思路](https://mulanai-1253870827.cos.ap-shanghai.myqcloud.com/llm-tutor-260704412/concept.jpg)

*图 1：LLM-as-a-Tutor 的整体思路。左侧是非可验证 RL 训练语料库，里面混合着"有挑战性"（How to center a div in CSS?）和"无挑战性"（Recommend a movie）两类 prompt。中间是导师 LLM：它采样 policy 出的两个 rollout，发现"Watch Inception / Watch Titanic"质量没差别，就给原 prompt 追加一条约束"和 5 岁外甥一起看、2 小时以内"，新 rollout 分裂为"Watch Toy Story (81 mins) / Watch Fight Club"。右侧是被训练的 policy LLM。*

问题在于：训练 prompt 是从 WildChat 之类的固定语料里采的，对训练开始时的 policy 刚刚好，**policy 涨上去之后这些 prompt 就集体"过期"了**——所有 rollout 都拿满分，judge 再细致也分不出高下，reward 信号就退化成一个常数。这是 policy-prompt pair 的属性，跟 prompt 本身是不是"难"没关系，跟 policy 当前能力有关系。

更糟的是，这个事实在 RL 训练过程中只会越来越严重。policy 越来越强 → prompt 越来越显得简单 → rollout 越来越同质化 → 优势函数方差塌缩 → 训练信号消失。**标准的 policy-unaware rubric pipeline 会撞上这堵墙，只是没人系统地拆过它。**

作者把这个问题正式命名了：**policy-prompt mismatch**。识别这个 mismatch，并提出修复方法，是这篇论文的第一个贡献。

---

## 方法：把 LLM judge 升级成 tutor

### 一句话核心思路

让 LLM 同时担任两个角色——**examiner**（用成对比较判断 prompt 是否"过易"）和 **generator**（追加一条原子约束，配套加一条 rubric 准则）。追加而非改写，保证难度单调上升。

### 形式化

沿用 GRPO 训练框架。给定 prompt $x$，生成 G 个 rollout $\{y^{(i)}\}$，rubric-based reward：

$$
s(x, y) = \sum_{k=1}^{K} w_k \cdot \mathcal{J}(y \mid x, r_k)
$$

advantage 用 group-relative 方式：

$$
A^{(i)} = \frac{s(x, y^{(i)}) - \mu(x)}{\sigma(x)}
$$

**当 $\sigma(x) \to 0$ 时，advantage 退化为 0，policy gradient 没梯度。** 这就是问题。

修复方式是改 prompt，而不是改 rubric。每个 seed prompt 配一份 base rubric $R(x)$，训练起点 $x^{(0)} = x$。每个训练 iteration 开头，tutor 处理所有 prompt：

1. 让 policy 在 $x$ 上采两个 rollout $y^{(1)}, y^{(2)}$
2. Tutor 成对比较这两个 rollout，判断"质量是否无可区分"
3. 如果是"无可区分"（即 prompt 对当前 policy 已过易），tutor 生成一个原子约束 $c$ 和对应 rubric 准则 $R_c$，把 prompt 改成 $\tilde{x} = x \oplus c$，rubric 改成 $\tilde{R}(x) = R(x) \cup R_c$，权重联合归一化
4. 如果是"可区分"，原样保留

GRPO update 沿用，只是把 $(x, R(x))$ 换成 $(\tilde{x}, \tilde{R}(x))$。

### 为什么是"成对比较"而不是"独立打分"？

判别"prompt 是否过易"本质是个**成对问题**——同一个 prompt 上两个 rollout 有没有质量差。LLM 在成对比较上的一致性和人类对齐度都明显好于独立打分（Zheng et al. 2023, Liu et al. 2024）。这跟 RLHF 偏好数据用 pairwise 而非 pointwise 是同一道理。

### 为什么是"追加约束"而不是"重写 prompt"？

**单调性保证**。任何满足 $\tilde{x} = x \oplus c$ 的回答，必然也满足 $x$——所以 $\tilde{x}$ 不会比 $x$ 更简单。改写操作（让 LLM 重新生成一个"更难"的 prompt）没有这个保证，新 prompt 可能跟原 prompt 难度持平甚至更简单。

而且改写会让 prompt 漂离原任务分布，rubric 也要重做，引入噪声。消融里 Rewrite 比 Append 平均低 0.31 个点。

### 训练流程

以"一个 epoch"作为 adaptation interval：每个 epoch 开头跑一遍 tutor，对所有"过易"prompt 加一条约束。下一个 epoch policy 就在更难的 prompt 上继续训练。当 policy 能力又涨上去、这些 prompt 又变成"过易"时，tutor 再加一条。**约束是累积的，所以单条 prompt 的难度随训练轮次单调上升，但触发条件是 policy 自己的行为决定的。**

这个设计真正的妙处是**没有外部难度调度表**。传统 curriculum learning 需要人工定义"什么难度该用什么 prompt"，这里全部由 policy 自己的 rollout 行为决定——policy 强了，tutor 自动多加点；policy 弱了，tutor 就少加。

### 一个具体例子

原 prompt："写一封友好的工作邮件，关心员工女儿 Lila 的康复，欢迎她回归工作，告知公司政策要求超过两天的病假需要医生证明，结尾寒暄。"

Policy 在这个 prompt 上的所有 rollout 都在 base rubric 上拿满分（$\sigma_{base} = 0$）——过易。

Tutor 追加一条约束："**另外，要明确写出要求医生证明的那条公司政策的具体名称。**"

新 rollout 分裂成两派：
- Rollout A 显式引用了 `[Company Policy Name]`，新 rubric 打 100 分
- Rollout B 含糊地说"请咨询你的主管"，新 rubric 打 60 分

Base rubric 上的均值从 100 掉到 92.5，标准差从 0 升到 14.9。**一条原子约束就把已退化的 prompt 救回来了。**

| 指标 | 改写前 ($x$) | 改写后 ($\tilde{x}$) |
|------|------------|-------------------|
| Base rubric 均值 | 100.0 | 92.5 |
| Base rubric 标准差 | 0.0 | 14.9 |
| Rollout A 分数 | — | 100.0 |
| Rollout B 分数 | — | 60.0 |

*表 1：追加一条约束后，rubric reward 重新获得区分度*

---

## 实验：是否真的有效？

### 设置

- **Policy**：Qwen3-1.7B-Thinking
- **Tutor + Judge**：Qwen3-8B-Thinking
- **训练数据**：WildChat 随机抽 4K prompt，3 个 epoch，adaptation interval = 1 epoch
- **评测基准**：FollowBench（HSR, SSR）、AdvancedIF（Overall, Micro）、InfoBench（DRFR），共 5 个指标 + 1 个平均

### Baseline 分三类

| 类别 | 方法 | 做什么 |
|------|------|------|
| 不改 prompt | Distillation | SFT，跟 8B 老师学 |
| 不改 prompt | Base rubrics | 静态 rubric |
| 不改 prompt | WildChecklists | 离线高质量 rubric |
| 不改 prompt | Policy-adaptive rubrics | rubric 跟着 policy 改（DR Tulu 思路），prompt 不动 |
| 不感知 policy | Evol-Instruct | 一次性把 prompt 加复杂（WizardLM 思路） |
| 感知 policy | EVA | 用 scalar reward gap 选 prompt + 重写 |
| **感知 policy** | **LLM-as-a-Tutor** | **成对判别 + 追加约束** |

### 主结果

| Method | FB-HSR | FB-SSR | AdvIF-Overall | AdvIF-Micro | IB-DRFR | **Avg** |
|--------|--------|--------|----------------|-------------|---------|---------|
| Qwen3-1.7B（未训） | 35.34 | 58.43 | 13.08 | 66.48 | 72.28 | 49.12 |
| Distillation | 34.67 | 57.15 | 11.59 | 58.27 | 68.58 | 46.05 |
| Base rubrics | 38.60 | 60.17 | 13.98 | 66.32 | 73.48 | 50.51 |
| WildChecklists | 38.71 | 60.71 | 14.03 | 66.35 | 73.80 | 50.72 |
| Policy-adaptive rubrics | 39.63 | 61.30 | 13.93 | 66.15 | **74.19** | 51.04 |
| Evol-Instruct | 37.98 | 59.99 | 13.88 | 65.99 | 73.35 | 50.24 |
| EVA | 39.14 | 61.04 | 14.93 | 66.66 | 73.43 | 51.04 |
| **LLM-as-a-Tutor** | **40.91** | **62.28** | **15.07** | **67.97** | 73.59 | **51.96** |

*表 2：主结果。LLM-as-a-Tutor 在 5 个指标里 4 个第一（IB-DRFR 第二），平均分第一。*

几个关键观察：

**1. Policy-adaptive 比 policy-unaware 强。** Policy-adaptive rubrics 和 EVA 平均都拿到 51.04，比静态 rubric baseline（50.51 / 50.72）高 0.3-0.5 个点。说明跟着 policy 调，确实能挖出更多训练信号。

**2. Policy-unaware 改 prompt 会掉点。** Evol-Instruct 把 prompt 一刀切改复杂，结果平均 50.24，比 Base rubrics 50.51 还低 0.27 个点。**难度 ≠ 有用**，没考虑 policy 当前能力的"加压"反而是负优化。

**3. Prompt 改 vs Rubric 改：prompt 改更有效。** LLM-as-a-Tutor 比 Policy-adaptive rubrics 高 0.92 个点，比 EVA 高 0.92 个点。**Rubric 再怎么进化，也只能放大 rollout 里已经存在的信号；当 prompt 让 rollout 没差异时，rubric 无能为力。必须改 prompt。**

**4. Distillation 反而退步。** 平均 46.05，比未训的 Qwen3-1.7B（49.12）还低 3 个点。作者在 Discussion 里把这件事跟 tutor 框架联系起来：SFT distillation 顶到天花板是老师的能力，而 LLM-as-a-Tutor 的天花板是老师"判别两个回答的能力"——后者明显比前者宽，因为 LLM 在 pairwise 上一致性更好。

### 消融 1：策略感知真的必要吗？

都换成"追加约束"操作，只改触发条件：

| 触发方式 | Adaptive? | FB | AdvIF | IB | **Avg** |
|---------|-----------|-----|-------|-----|---------|
| Always（100% 全加） | ✗ | 39.68 | 14.03 | 72.68 | 42.13 |
| Random（28%，与本文对齐） | ✗ | 40.57 | 14.58 | 73.32 | 42.82 |
| Wrong（用 8B 模型判断，47% 触发） | ✓(对错模型) | 40.18 | 14.53 | **73.72** | 42.79 |
| **Adaptive（用 1.7B 自己的 rollout 判断）** | ✓ | **40.91** | **15.07** | 73.59 | **43.19** |

*表 3：策略感知消融。* 注意这个表只对比 3 个 benchmark 的均分（不是 5 个指标的均分），跟主表口径不同。

观察：

- **Always 全加**最差——给"已经有挑战性"的 prompt 强行加约束，反而把信号搞坏。
- **Random 28%** 跟 Always 拉开 0.7 个点，说明"加少一点"比"全加"好。
- **Wrong**（用 8B 强模型判断）触发率高达 47%，比 Adaptive 的 28% 高不少，但效果反而差。**判别信号不能来自一个比 policy 强太多的模型**，因为它看 1.7B 的两个 rollout 都会觉得"差不多"，结果把不该加的全加了。
- **Adaptive（本文）**：用 policy 自己的 rollout 做判别，28% 触发率，最优。

**所以关键不是"加不加约束"，也不是"加多少"，是"对哪些 prompt 加"。** 判别信号必须来自 policy 自己的行为。

### 消融 2：追加 vs 改写 vs 替换

| 策略 | FB | AdvIF | IB | **Avg** |
|------|-----|-------|-----|---------|
| Reset（每轮替换旧约束） | 39.57 | 14.33 | 73.28 | 42.40 |
| Rewrite（重写整个 prompt） | 40.43 | 14.63 | 73.58 | 42.88 |
| **Append（追加，本文）** | **40.91** | **15.07** | **73.59** | **43.19** |

*表 4：修改策略消融。*

Reset 比 Append 低 0.79 个点——丢掉之前的约束会让之前几轮的难度累积功亏一篑，prompt 难度原地踏步。Rewrite 比 Append 低 0.31 个点——改写让 prompt 漂离原任务分布，rubric 也要重做。

**Append 在三个 benchmark 上全部最优。**

---

## 分析：tutor 的判别准不准？

### 1. tutor 判别跟经验难度强相关

![图 2：tutor 判别为"无挑战性"的 prompt，其 base rubric 均分更高、标准差更小](https://mulanai-1253870827.cos.ap-shanghai.myqcloud.com/llm-tutor-260704412/constraint_added_reward_mean_std.jpg)

*图 2：tutor 判别 vs 经验难度。被判为"无挑战性"的 prompt（红色，n=多），base rubric 均分 90.76、标准差 12.96；被判为"有挑战性"的 prompt（绿色），均分 78.24、标准差 27.07。tutor 的二元判别跟 rubric 经验统计量强相关，不是瞎猜。*

### 2. policy 越强，tutor 加的约束越多

![图 3：policy 模型越大，被追加约束的 prompt 比例越高](https://mulanai-1253870827.cos.ap-shanghai.myqcloud.com/llm-tutor-260704412/constraint_added_rate_by_size.jpg)

*图 3：固定 tutor = 8B，policy 从 0.6B 涨到 4B，"被追加约束"的比例从 8.1% 升到 25.8% 再到 40.5%。policy 越强，对当前 prompt 越显得简单，tutor 自动多加点约束。*

这是个挺漂亮的结果——它说明 tutor **不是按固定 schedule 在工作**，是跟着 policy 的实际能力动态调整难度曲线。

### 3. 追加约束让 base rubric 也变得更有判别力

![图 4：追加的约束越多，base rubric 的均值越低、标准差越大](https://mulanai-1253870827.cos.ap-shanghai.myqcloud.com/llm-tutor-260704412/cumulative_difficulty.jpg)

*图 4：x 轴是追加的约束数（0 到 3），y 轴是同一个 base rubric 上的均值和标准差。约束越多，base rubric 均值越掉、标准差越大。说明追加的约束把原来已经"饱和"的 base rubric 重新激活了——同一个 base criterion 评分时，因为多约束必须同时满足，反而能区分出 rollout 高下。*

这点挺反直觉——你不动 base rubric，只是往 prompt 里加约束，**base rubric 上的分数分布居然跟着变**。原因是新约束让 policy 必须"同时满足更多东西"，base criterion 的相对完成度被拉出差距，rubric 的判别力随之回归。

---

## 我的判断：值不值得花时间读？

### 亮点

1. **问题命名做得到位**。"policy-prompt mismatch" 这个概念在 rubric-based non-verifiable RL 里其实一直存在，但没人系统提过。把痛点命名清楚，论文就有了立得住的位置。
2. **核心 idea 简洁可落地**。"用成对比较判别过易 prompt + 追加原子约束" 是个工程上很容易复现的改动，**不需要新算法、不需要新训练目标**，只是在标准 GRPO pipeline 的数据准备阶段加一个 LLM 调用。
3. **消融逻辑闭环**。从"是否策略感知"到"用什么修改策略"再到"约束累积有没有用"，三个维度都消了，结论互相支撑。
4. **分析切中要害**。三张分析图（判别相关性、policy scale 适应性、累积难度）直接回答了"这玩意到底有没有 work"的核心疑问，不用读者自己脑补。

### 不足

1. **绝对提升不算大**。平均 51.96 vs 51.04，0.9 个点的差距。在 instruction following 这种"已经接近饱和"的任务上，0.9 个点是有意义的，但论文没有 effect size / 显著性检验的细节（虽然写的是 5 次独立运行取 mean±SEM），需要读者自己判断是否 robust。
2. **Distillation baseline 反常**：平均 46.05 比未训 policy 还低 3 个点。这要么是 SFT 配置有问题（用同一个 prompt 让 8B 出答案，policy 直接学老师的 thinking trace，但 thinking 模型蒸馏到非 thinking 推理路径上会出问题），要么是实验设置有坑。**论文没展开讨论这点，有点遗憾。**
3. **"0.92 个点改 prompt 比改 rubric 强"的论断略显激进**。Policy-adaptive rubrics（51.04）和 LLM-as-a-Tutor（51.96）差距主要来自 FollowBench 和 AdvIF-Micro。InfoBench 上 Policy-adaptive rubrics 反而高 0.6 个点（74.19 vs 73.59）。说明"prompt 维度优于 rubric 维度"不是普适结论，跟 benchmark 的难度结构有关。
4. **没讨论失败模式**。如果 tutor 自己判断错，把"还有挑战性"的 prompt 也加了约束，结果会怎样？如果原子约束加得跑题（违反 $x$ 的语义），rubric 评分会怎么崩坏？这些 case 论文都没展开。
5. **追加 vs 改写的边界条件没摸清**。作者承认"改写在改写质量差时吃亏"，并猜测"tutor 越强改写越有优势"，但没给实验支撑。Discussion 里说"task-agnostic，可推广到 reasoning steps、edge cases、sourcing"——这个推广是 claim，没验证。

### 跟同期工作比处于什么位置？

- **vs DR Tulu（Shao et al. 2025）**：DR Tulu 在 deep research 长篇任务上做 rubric 进化，本文是 complex instruction following 上做 prompt 进化。两者是"rubric 维度 vs prompt 维度"的互补关系，不是直接竞品。但本文指出了一个 DR Tulu 没解决的痛点——**当 prompt 让 rollout 无差异时，rubric 进化救不回来**。
- **vs EVA（Ye et al. 2024/2025）**：EVA 用 scalar reward gap 选 prompt + 重写，本文用成对比较 + 追加。本文胜在"判别更准 + 难度单调有保证"，EVA 胜在"更通用、不依赖 pairwise 模板"。两者都是"prompt 维度 policy-adaptive"路线，本文更精细。
- **vs Rubrics as Rewards / WildChecklists / OpenRubrics**：都是 rubric 这一维度的优化，**没碰 prompt**。本文是对这类工作的必要补充。

**总评**：这是个**问题定义清楚、解法简洁工程友好、消融扎实的"小而美"工作**。平均 0.9 个点的提升不算炸裂，但论文最大的贡献是把"prompt 也需要 policy-aware"这件事提到了一个独立维度上。后续在 rubric-based RL 上做研究的人，应该会反复引用这篇。

### 对工程实践的启发

如果你也在做 rubric-based non-verifiable RL 训练，碰到 reward 涨不动、rollout 同质化的问题，可以考虑：

1. **加一个 pairwise "饱和检测"pass**。让 LLM judge 在每个 iteration 开头判别哪些 prompt 已经对当前 policy 无挑战性。判别模板论文附录里有。
2. **追加原子约束**，让 prompt 难度跟着 policy 长。不要做 full rewrite，会漂离原任务分布。
3. **如果 tutor 跟 policy 是同一个模型**（强 model 当 tutor+judge 训弱 policy 是常用配置），记得 tutor 要比 policy 强——否则 pairwise 判别会失真，论文里的 Wrong 消融就是反例。
4. **保留 base rubric 不动**。追加约束会让 base rubric 重新获得判别力，不需要重新生成 rubric。

---

## 写在最后

这篇论文最让我欣赏的是它对**"为什么静态 prompt 在 RL 后期会失效"**这件事的系统化拆解——不是甩个数据说"我们涨了 X 个点"，而是从 GRPO advantage 公式出发，论证 $\sigma(x) \to 0$ 这个临界条件，再提出修复方法。

这种"找到 RL 训练中的隐藏 bottleneck，再针对性解决"的研究风格，在 LLM RL 这个赛道上挺稀缺。大部分工作都在刷数据、刷 trick，能停下来问"为什么我们的 reward signal 突然不 work 了"的论文不多。

**唯一的小遗憾是绝对提升偏小**。不过在 Qwen3-1.7B 这种已经被刷得很透的小模型上，能稳定涨 0.9 个点（且 5 个指标 4 个第一）已经不容易了。换到更大模型或更难任务上，policy-prompt mismatch 会更严重，这套框架的边际收益可能更明显。

arXiv: 2607.04412，2026-07-05。读完有兴趣的可以翻翻原文，附录里有 tutor 的 prompt 模板和判别逻辑，落地很直接。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
