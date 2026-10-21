# 让 Agent 自己给自己改脚手架：Self-HPO 把 harness 和 policy 绑在一起进化，SWE-Bench 平均分涨 7 个点

做 Agent 的人大概都有过这种体验：同一个模型，换一套脚手架（prompt、工具封装、提交逻辑），跑 SWE-Bench 的分数能差出好几个点。于是大家分成了两派——一派拼命 RL 训模型，一派拼命搜 harness。这篇来自 Microsoft Research 的论文（arXiv:2610.22791）说：**这两件事得一起做，而且可以让 Agent 自己来**。

**核心摘要**：coding Agent 的表现由 policy（模型）和 harness（脚手架）共同决定，但只优化其中一个是次优的——为某个 checkpoint 调好的 harness 未必适配更新后的 policy，而 policy 也只能从固定 harness 能诱导出的轨迹里学东西。Self-HPO 的做法是让二者交替进化：H-step 里当前 policy 自己分析失败轨迹、提出 harness 改进；P-step 里在新 harness 下做 RL；而 RL 中始终解不出来的难题，又成为下一轮 harness 搜索的目标。在 Qwen3.5-35B-A3B 上，三个 coding benchmark 的平均 resolve rate 从 57.8% 提到 64.8%，SWE-Bench Verified 上到 69.5%。说实话，这个数字本身不算炸裂，但"两个组件的联合增益（8.5 个点）超过各自增益之和（6.9 个点）"这个发现，是真的值得琢磨。

## 论文信息

- **标题**：Self-Evolving Coding Agents via Harness-Policy Optimization
- **作者**：Zhepei Wei, Wenlin Yao, Baolin Peng, Danqing Wang, Qianhui Wu, Hao Cheng, Yu Meng, Jianfeng Gao
- **机构**：Microsoft Research、University of Virginia、Carnegie Mellon University
- **链接**：https://arxiv.org/abs/2610.22791 ｜ 代码：https://github.com/MSR-Orchard/Self-HPO ｜ 模型：https://huggingface.co/self-hpo
- **提交日期**：2026 年 10 月 16 日

---

## 🎯 问题：harness 和 policy 是互相咬着的，分开优化必然错配

先把"harness"这个词说清楚。一个 coding Agent 由两部分组成：policy model（LLM 本身）和 harness——就是那个决定模型看到什么、动作怎么执行、收到什么反馈、episode 什么时候结束的外部脚手架。mini-SWE-agent 那种 ReAct 循环就是典型的 harness：模型发 bash 命令，沙箱执行，最后打印 patch 提交。

同一个模型，harness 不同，能不能解出一道题完全两说。这就带来一个结构性问题：

- **只优化 harness**（比如搜 prompt、改 agent 代码）：模型本身没变强，天花板锁死；
- **只优化 policy**（比如 RLVR 训模型）：harness 是固定的，policy 只能从这套 harness 能产生的轨迹里学，harness 的失败模式学不掉；
- 更麻烦的是**错配**：为 checkpoint A 调好的 harness，针对的是 A 的失败模式；policy 更新到 B 之后，失败模式变了，旧 harness 就失灵了。

之前也有联合优化的工作（比如 WHALE），但它们依赖一个冻结的外部 proposer 模型来改 harness，而且搜索任务集是固定的——policy 一直在变，搜索目标却不动，这就是论文里说的 "moving optimization target" 问题。

Self-HPO 的答案是：既然 policy 自己会变强，那就让**变强后的 policy 自己当 harness 的提案者**，让 **RL 里没解出来的难题当下一轮 harness 搜索的靶子**。整个系统不需要外部强模型也能自转。

![Self-HPO 递归自改进框架总览](https://arxiv.org/html/2610.22791v1/teaser_overview.png)

*图 1（左）：Self-HPO 的交替进化结构。上面一行是 harness 进化（H₀→H₁→H₂），下面一行是 policy 进化（P₀→P₁→P₂）。每个 H-step 用当前 policy 当 harness proposer 搜索新脚手架，每个 P-step 在新 harness 下做 RL 更新 policy。两条线互相咬合，谁也离不开谁。*

![三个 benchmark 平均 resolve rate 随优化步骤的曲线](https://arxiv.org/html/2610.22791v1/teaser_curve.png)

*图 1（右）：SWE-Bench Verified、Multilingual、Pro 三个 benchmark 的平均 resolve rate 沿 Base→H₁→P₁→H₂→P₂ 的走势。Self-HPO（橙线）每一步都在涨，最终压过 policy-only（RLVR）和 harness-only（Self-Harness）两条基线。注意这个曲线的形状——每一步都有正收益，没有平台期，这在自进化框架里不常见。*

## 🏗️ 方法：一个 H-step、一个 P-step，交替滚起来

先给一句话版本：**冻结 policy 找 harness 的毛病，冻结 harness 训 policy，训完 policy 再拿没解出来的题去找 harness 的新毛病。**

形式化一点。给定任务 $x$，Agent 的轨迹为 $\tau \sim \pi_{H,P}(\cdot \mid x)$，可执行验证器给出二值奖励 $y(x,\tau) \in \{0,1\}$。优化目标是 harness-policy 对在训练集上的期望 resolve rate：

$$J(H,P) = \mathbb{E}_{x\sim\mathcal{D}}\,\mathbb{E}_{\tau\sim\pi_{H,P}(\cdot\mid x)}\big[y(x,\tau)\big]$$

![Self-HPO 方法总览](https://arxiv.org/html/2610.22791v1/method_overview.png)

*图 2：Self-HPO 的两步循环。上方 H-step：冻结的 policy P 同时扮演两个角色——任务解题者和 harness 提案者，先分析搜索集 S 上的失败轨迹，再提出 K 个候选 harness（H¹…H^K），在 S 上评估选出最优。下方 P-step：在新 harness 下采样 rollout、用可验证奖励做 RL 更新。随后更新后的 policy 成为下一轮 proposer，训练中从未解出的难题组成下一个搜索集。注意图中雪花（冻结）和火焰（更新）的标记，谁冻谁训一目了然。*

**H-step（失败条件化的 harness 搜索）** 有两个关键设计。

一个是 **self-proposer**：提案者就是当前 policy 自己，不引入外部模型。它能看到当前 harness 的代码、搜索集上的失败轨迹、参考修复、历史候选的得分和笔记，然后针对诊断出的失败原因（覆盖率缺口、精度缺口、提交丢失等）设计**一个新机制**加到父 harness 上。提案有硬约束：父 harness 的所有方法必须原样保留，policy 和采样配置冻结，不许往上下文里注入参考答案。

另一个是 **evolving search set（演化搜索集）**。第一轮搜索集 $\mathcal{S}_0$ 从训练集均匀采样；之后每一轮的搜索集，都来自上一个 P-step 里**所有 rollout 都没解出来的难题**。这个设计很妙——harness 搜索永远盯着 policy 当前最新的失败模式，而不是抱着最初的任务集反复优化。

**P-step** 就是在优化后的 harness 下做 RL：二值 outcome reward，优化器是 GRPO 加上 Orchard 提出的 BAR（Balanced Adaptive Rollout）——后者自适应地多采样轨迹来构造奖励平衡的 group，缓解稀疏奖励下全零组的方差坍缩问题。

实验配置：base policy 是 Qwen3.5-35B-A3B（MoE，激活约 3B），种子 harness 是 mini-SWE-agent，训练集 2,484 个任务（来自 SWE-rebench、SWE-rebench V2、Scale-SWE），跑两轮交替优化，每个 H-step 搜 K=6 个候选，每个 P-step 做 50 次 RL 更新。

## 📊 实验：联合优化全面压过单边优化

主实验在三个 benchmark 上：SWE-Bench Verified（500 题，主榜）、SWE-Bench Multilingual（300 题，9 种非 Python 语言，OOD）、SWE-Bench Pro（642 题，OOD）。结果取三次运行均值（avg@3）。

| 方法 | 优化 Harness | 优化 Policy | SWE-Verified | SWE-Multi | SWE-Pro | 平均 |
|---|---|---|---|---|---|---|
| Base | ✗ | ✗ | 61.0 | 54.2 | 58.3 | 57.8 |
| **外部模型设置（Opus 4.8）** ||||||
| SFT | ✗ | ✓ | 65.0 | 49.6 | 49.1 | 54.6 |
| Meta-Harness | ✓ | ✗ | 69.1 | 56.5 | 64.8 | 63.5 |
| **HPO（本文）** | ✓ | ✓ | **71.6** | **61.4** | **66.0** | **66.3** |
| **自进化设置（无外部模型）** ||||||
| RLVR | ✗ | ✓ | 67.5 | 56.3 | 61.8 | 61.9 |
| Self-Harness | ✓ | ✗ | 65.2 | 54.3 | 60.7 | 60.1 |
| **Self-HPO（本文）** | ✓ | ✓ | **69.5** | **59.7** | **65.2** | **64.8** |

*表 1：主实验结果。两个设置下，联合优化 harness 和 policy 都在全部三个 benchmark 上拿到最好成绩。*

几个值得说的点。

**自进化设置下**，Self-HPO 在 SWE-Verified 上比 RLVR 高 2.0 个点、比 Self-Harness 高 4.3 个点；平均分 64.8，比最强的单边基线 RLVR（61.9）高 2.9 个点。提升不算巨大，但胜在稳定——三个 benchmark 全部领先，没有偏科。

**OOD 泛化是真的**。训练和搜索只用 Python 任务，但 SWE-Multi（非 Python）上 Self-HPO 比 Base 涨 5.5 个点，而 Self-Harness 只涨 0.1 个点——纯 harness 优化几乎不迁移，RL 训出来的能力才是可迁移的主体。这个对比挺能说明问题的。

**外部模型设置里有个反直觉发现**：用 Opus 4.8 的知识，SFT 蒸馏（学轨迹）反而被 Meta-Harness（只改 harness 不动权重）甩开 8.9 个平均分，而且 SFT 在两个 OOD 榜单上**跌破 Base**。蒸馏会把教师模型的行为模式硬塞进权重，域外直接崩；harness 优化只是改了交互方式，不伤模型本身。坦白说，这个结论对"遇事不决先蒸馏"的工程惯性是个不错的敲打。

![优化轨迹对比](https://arxiv.org/html/2610.22791v1/optimization_trajectories.png)

*图 3：三个 benchmark 上 Base→H₁→P₁→H₂→P₂ 的优化轨迹。Self-HPO（橙）和 Self-Harness（粉）第一个 H-step 是共享的，之后分道扬镳：Self-HPO 每次 policy 更新后再搜 harness 还能继续涨；而 Self-Harness 继续做 harness-only 更新在 SWE-Multi 上反而从 55.0% 掉到 54.3%——policy 不动，harness 搜索很快就会过拟合到旧失败模式上。*

## 🔬 消融与分析：联合增益超过各自增益之和

消融实验（SWE-Verified，以 HPO 的 71.6% 为参照）：

| 配置 | Resolve rate | 变化 |
|---|---|---|
| HPO（完整） | 71.6 | — |
| P-step 换成 SFT | 67.2 | 降 4.4 个点 |
| P-step 换成 RFT | 70.5 | 降 1.1 个点 |
| H-step 用固定搜索集 | 70.1 | 降 1.5 个点 |
| H-step 换成自提案（=Self-HPO） | 69.5 | 降 2.1 个点 |

*表 2：消融实验。RL 作为 P-step 明显优于 SFT 和 RFT；演化搜索集有实打实的贡献；自提案比外部 proposer 差 2.1 个点，但仍然压过所有单边基线——外部模型是加分项，不是必需品。*

**最值钱的一组分析是交叉评估**：把不同轮的 harness 和 policy 拆开重新配对。

| | $P_0$ | $P_1$ | $P_2$ |
|---|---|---|---|
| $H_0$ | 61.0 | 61.7 | 64.4 |
| $H_1$ | 64.3 | 67.8 | 68.1 |
| $H_2$ | 64.5 | 68.6 | **69.5** |

*表 3：SWE-Verified 上 harness（行）× policy（列）交叉评估。*

$H_2$ 配 base policy $P_0$ 能拿 64.5（harness 单独贡献 3.5 个点），$P_2$ 配种子 harness $H_0$ 能拿 64.4（policy 单独贡献 3.4 个点）。但组合起来是 69.5——**比 Base 高 8.5 个点，超过两者之和 6.9 个点**。1.6 个点的"协同红利"说明 harness 和 policy 的改进不是简单叠加：好 harness 诱导出的轨迹让 policy 学到的东西，反过来又在新 harness 下被更好地激发。这其实就是整个论文立论的直接证据。

**pass@k 和可靠性**：8 次采样下，Self-HPO 的 pass@k 曲线在每个 k 上都压过 Base，pass@8 从 80.0% 提到 84.8%——不只是单次更稳，能力边界本身外扩了。按难度拆开看更有意思：Medium（Base 偶尔成功）涨 17.2 个点，Easy 涨 12.9 个点，Hard（Base 全败）也涨了 6.1 个点；唯一倒退的是 Resolved 组，降 2.2 个点。收益分布不算均匀，但主战场确实是"会做但不稳"的那批任务。

![pass@k 曲线](https://arxiv.org/html/2610.22791v1/passk_curve.png)

*图 4(a)：SWE-Bench Verified 上 pass@k 曲线，Self-HPO 在 k=1 到 8 全程高于 Base，pass@8 从 80.0% 到 84.8%。阴影是 95% bootstrap 置信区间。*

![按任务难度的 pass@1 分解](https://arxiv.org/html/2610.22791v1/pass1_breakdown.png)

*图 4(b)：按 Base 成功率分桶的 pass@1 对比。Hard（+6.1）、Medium（+17.2）、Easy（+12.9）都在涨，已稳定解决的 Resolved 组略降 2.2 个点。*

还有一个训练侧的观察：RLVR 最后 50 次更新里，全零成功组（组内所有 rollout 都失败、GRPO 拿不到组内对比信号）占比一直在 11%–17%；而 Self-HPO 最后一个 P-step 把这个比例压到了 6.5%。harness 变好后，稀疏奖励问题本身被缓解了——这是 policy 训练环境被 harness 改善的又一个佐证。

![harness 搜索前沿曲线](https://arxiv.org/html/2610.22791v1/search_frontier.png)

*图 6（附录）：SWE-Verified 上 harness 进化的 best-so-far 曲线。圆点是各候选得分：共享的第一个 H-step 到 64.3% 后，纯 harness 搜索（Self-Harness 分支）只能磨到 65.2%；Self-HPO 先做一次 P-step 把 policy 拉到 67.8%，第二个 H-step 再到 68.6%，最后 P₂ 收在 69.5%。两条分支的分叉直观展示了 policy 更新给 harness 搜索"换靶子"的价值。*

## 🧩 案例：优化后的 harness 到底改了什么

说实话，这篇文章我最喜欢的部分是 Figure 5 的 case study——它让你看到"harness 优化"不是玄学，改的东西非常具体。HPO 第二轮搜出的 harness 只重写了 `step()` 一个方法，加了四个检查，每个对应一类真实失败模式：

1. **预算阀门**：步数/时间快用完时插入一条"别再探索了，赶紧 best-effort 提交"的消息——原版里预算耗尽直接交空结果；
2. **空 patch 守卫**：提交的 diff 为空时拦下，提醒模型"新建的文件没被 git 跟踪，`git diff` 看不到，先 `git add -N`"——这是一个极其常见又极其蠢的失败；
3. **契约对齐**：第一次通过空守卫的提交会被 hold 住一次，要求模型拿仓库里已有测试核对 API 名字和签名——防止"patch 能 apply 但实现了错误契约"；
4. **patch 兜底**：提交的 patch apply 失败时，自动用沙箱里的 `git diff` 替代。

论文附录里 sphinx-doc__sphinx-9658 的案例很典型：同一个 policy $P_2$，在种子 harness 下最后提交的是一段**散文总结**（"## Summary of Changes……"）而不是 patch 文件，任务直接判失败；在进化后的 $H_2$ 下，harness 拦下这次提交、报告"你的 patch 里没有文件"，policy 回头 `git diff > patch.txt` 重新提交，任务解决。模型能力没变，结果天差地别——这就是 harness 的重量。

顺便一提，用 35B 总参数（3B 激活）的 Qwen3.5-35B-A3B 做到 69.5%，在 SWE-Bench Verified 榜单上已经超过了 Claude Haiku 4.5（66.6%）、Kimi K2 Thinking（63.4%）这些大得多的模型，离 Gemini 3 Pro（69.6%）只差 0.1。小模型 + 好脚手架 + 好训练的组合拳，性价比相当可观。

## 🤔 我的判断

这篇论文的价值不在单个数字，而在把一个工程直觉变成了可复用的框架：**Agent 系统里"环境接口"和"模型能力"是耦合变量，应该交替优化，而且可以用系统自身的产物（失败轨迹、未解难题、变强后的 policy）来驱动下一轮**。

几个让我信服的地方：联合增益超加性（8.5 > 6.9）有交叉评估矩阵背书，不是嘴上说协同；演化搜索集、RL 优于 RFT 这些设计都有消融撑着；自提案只比外部 proposer 差 2.1 个点，说明"自进化"不是噱头，是真能转起来的。

但也有几个我想追问的点。一是只跑了两轮交替优化——论文说停止准则是难题集稳定，但两轮就收敛，到底是真收敛还是预算限制？更长程的递归会不会失效或震荡，没看到证据。二是 harness 搜索空间被限制在"子类化种子 agent + 改 prompt"，提案约束要求"每次只加一个新机制"，这保证了稳定性，但也限制了进化的想象力——更激进的 harness 重构（换工具集、换控制流）不在这个框架的搜索范围内。三是所有实验都绑定 Qwen3.5-35B-A3B 一个 base model，换个模型家族，self-proposal 的质量能不能撑住，不好说。

工程上的启发倒是很直接：如果你团队在维护 coding Agent，**别只调 prompt 也别只训模型，把"提交前检查"这类 harness 机制和 RL 数据闭环打通**。那个"空 patch 守卫"和"契约对齐"的检查，基本可以直接抄。

顺着这个方向再往前想一步：harness 现在是 Python 类 + prompt，说到底还是"程序"。当 proposer 足够强，harness 的进化会不会走向真正的程序合成——每次迭代生成一个全新的 agent runtime？那时候 H-step 和 P-step 的边界可能就没必要存在了。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
