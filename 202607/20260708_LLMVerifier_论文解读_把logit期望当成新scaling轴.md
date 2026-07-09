# 把 logit 期望当成新 scaling 轴：LLM-as-a-Verifier 论文解读

arXiv: 2607.05391

你有没有遇到过这种场景：一个 Agent 跑了 5 条 trajectory，Pass@1 只有 75%，你心里清楚里面至少有一条是好的，但用 LLM-as-a-Judge 让模型打个分选一条出来，结果 5 条轨迹经常打成同一个分——分不出谁好谁坏。于是你只能蒙，或者干脆重跑。这种"分数打平"的痛点，今天这篇论文给了一个特别轻巧的解法：别让 LLM 只吐一个离散的 token，让它把整个评分 token 的 logits 分布吐出来，再取个期望，连续分数自然就出来了。

听起来简单得过分。但你看实验就会发现，**这个改动 + 三个简单的 scaling 维度，直接在 4 个 SOTA 榜上同时登顶了**。这篇来自 Stanford + UC Berkeley 的工作（Ion Stoica、Azalia Mirhoseini、Chelsea Finn、Marco Pavone 都在作者列里），野心不小——它要论证的是：**verification 本身就是一条新的 scaling 轴**，这条轴之前没人认真挖过。

## 核心摘要

**痛点**：现有 LLM-as-a-Judge 只取最大概率的 token 当分数，评价粒度只有 1/G（G 是可选分数等级，比如 5 等级就只有 0.2 的分辨率）。在两个都很"差不多好"的 trajectory 之间经常打出平分 tie，27% 的 tie rate 直接废掉了 ranking；想用训好的 reward model 又跨域泛化不行。

**核心方案**：把"输出单个最大概率 token"换成"对评分 token 集合 $\{v_1, ..., v_G\}$ 上的 logits 分布取期望"：

$$R(x, \tau) = \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K} \sum_{g=1}^{G} p_\theta(v_g \mid x, c, \tau) \cdot \phi(v_g)$$

三个超参直接对应三条独立的 scaling 轴：G（粒度，1→4→16→20）、K（重复评估次数）、C（评价准则数量，从不同侧面拆解任务）。再叠一个 **Probabilistic Pivot Tournament（PPT）** 算法，把 $\mathcal{O}(N^2)$ 的成对比较预算砍到 $\mathcal{O}(Nk)$。

**关键效果**：
- Terminal-Bench V2 **86.5**%（击败 GPT-5.5 的 84.7%、Opus 4.7 的 80.2%）
- SWE-Bench Verified **78.2**%（击败 Opus 4.5 的 76.8%）
- RoboRewardBench **87.4**% 偏好准确率（击败 RoboReward-8B 训了 45k 数据的 81.4%）
- MedAgentBench **73.3**%（击败 Opus 4.8 的 70.2%）
- 顺带：LIBERO 机器人 1.8× sample efficiency、MATH 数学 1.1× sample efficiency

**一针见血的评价**：这是一篇**方法论级别**的工作——它没有训任何新模型，只改动了"如何读 logits"，就同时在 coding / robotics / medical 三个完全不同的领域刷到 SOTA。属于那种"看完之后会觉得：这思路本该早有人做"的、clean & elegant 的 paper。但要注意它的依赖：需要 verifier 模型能暴露 top-k logits（Gemini 2.5 Flash / Qwen 3.6 这种），对只开 API 不给 logit 的 frontier model 不太友好。

---

## 论文信息

- **标题**：LLM-as-a-Verifier: A General-Purpose Verification Framework
- **作者**：Jacky Kwok (Stanford)、Shulu Li (UC Berkeley)、Pranav Atreya (UC Berkeley)、Yuejiang Liu (Stanford)、Yixing Jiang (Stanford)、Chelsea Finn (Stanford)、Marco Pavone (Stanford, NVIDIA Research)、Ion Stoica (UC Berkeley)、Azalia Mirhoseini (Stanford)
- **机构**：Stanford University + UC Berkeley + NVIDIA Research
- **arXiv**：https://arxiv.org/abs/2607.05391 (v1 提交 2026/07/06，v2 修订 2026/07/07)
- **代码**：https://github.com/llm-as-a-verifier/llm-as-a-verifier
- **主页**：https://llm-as-a-verifier.com

---

## 为什么要把 verification 单独提出来当 scaling 轴？

大家都在聊 scaling：pre-training 加数据加参数，post-training 训 SFT/RLHF，test-time 加 inference compute。这三条路把模型的"生成能力"推到了一个新高度。**但 verification 这一侧，一直没有对应的 scaling law**。

生成和验证是对偶的——"能解"和"能判断哪个解对了"在信息论上是同一个问题的两面。但工程上，验证比生成更难。原因有两个：

1. 判别比生成更"**非黑即白**"：生成可以凑个大致合理的答案蒙混过关，验证要么对要么错，没有中间地带。
2. **输出空间被压缩**：标准 LLM-as-a-Judge 强迫模型从 $\{1, 2, 3, 4, 5\}$ 这样的离散 token 里选一个。这一步本身就把模型的内部信心压平了。哪怕模型对"轨迹 A 比轨迹 B 好 0.6 分"很有信心，token 解码后只能给整数 4 和 5，最后还是打成平分。

论文用一个特别直观的数据点开场：在 Terminal-Bench V2 上采样 100 条 trajectory 用 oracle verifier 挑最优，pass rate **是 98.9 个百分点**——也就是说模型本身其实基本能解这些题，问题全出在"挑哪条"上。

![Oracle Pass@K 曲线](https://arxiv.org/html/2607.05391v2/figures/oracle_bon_plot.png)

*图 1：Terminal-Bench 2.0 上随采样 trajectory 数量增加的 oracle Pass@K。绿线是理论上限，能摸到 98.9%；红色虚线是 Claude Opus 4.7 的实际单次 pass（69.4%），紫色虚线是 Claude Mythos（82.0%）。这个 gap 大得离谱——意味着 evaluation 才是瓶颈。*

这给作者一个很硬的理由：与其再去训一个更大的生成模型，不如**在 verifier 这一侧加 compute**。这其实呼应了 AlphaGo 时代的"value network scaling"思路——围棋里最后赢棋靠的也是验证（局面评估），不是搜索。

---

## 框架总览：把 logit 期望当成"细粒度 reward"

下图是论文的核心框架，标题起得很嚣张——"多模态输入、一个验证框架、所有场景通用"。

![方法框架图](https://arxiv.org/html/2607.05391v2/x1.png)

*图 2：LLM-as-a-Verifier 框架。输入可以是 text/image/video，输出三件事：test-time scaling（best-of-N 选优）、progress tracking（实时监测 agent 进度）、reinforcement learning（dense reward）。中间那个公式就是 R(x, τ) = (1/CK) ΣΣΣ p_θ(v_g|x,c,τ)·φ(v_g)——三轴可独立 scaling。*

论文把 LM-as-a-Judge 和 LM-as-a-Verifier 做了一个关键区分：

> **Judge** is one who forms an overall opinion and assigns a decision, whereas **verifier** is one who confirms the truth or correctness of something and requires more detailed evaluations.

也就是说 judge 给你一个总分（"这答案 7 分"），verifier 要告诉你**为什么**、**哪一步对的、哪一步错的**。要支持这个，输出空间必须更细。

作者的具体做法是让 verifier 模型输出"评分 token 集合"的概率分布，然后取数学期望。比如评分集合是 $\{A, B, C, D, E, ..., T\}$（20 个字母，对应 1-20 分），模型对每个字母都有个 logprob：

$$R(x, \tau) = \frac{1}{CK} \sum_{c=1}^{C} \sum_{k=1}^{K} \sum_{g=1}^{G} p_\theta(v_g \mid x, c, \tau) \cdot \phi(v_g)$$

其中 $\phi$ 把字母映射到数值（比如 $\phi(A)=1, \phi(T)=20$）。这一改动的关键之处：**模型不增加任何知识，只是把"内部的高维信心"投影到一个 1 维的连续尺度上**。原来打成 4 的两个不同 trajectory，现在会自然落到 4.2 和 4.7 这种区分度上。

得到连续分数 $R(x, \tau)$ 后，转 pairwise preference 用 Bradley-Terry：

$$P(\tau_i \succ \tau_j \mid x) = \frac{1}{1 + \exp(-(R(x, \tau_i) - R(x, \tau_j)))}$$

这俩公式就是整篇论文的数学骨架，记一下，后面所有实验都围着它转。

---

## Verification Scaling：三个独立可调的轴

公式里的三个超参 C、K、G 不只是用来调参——它们被显式地当成三条**独立的 scaling 轴**，每条轴解决一种误差。

### 轴 1：分数粒度 G

直觉：分数等级越多，分辨率越高。论文在 Terminal-Bench V2 上用 Gemini 2.5 Flash 验证：

| Granularity G | 1 | 4 | 16 | 20 |
|---|---|---|---|---|
| Verification Accuracy | 73.1% | 75.1% | 77.2% | 77.5% |
| SNR (signal-to-noise ratio) | 0.775 | 0.786 | 0.797 | 0.799 |

从 1 档升到 20 档，准确率涨了 4.4 个点。

为什么？作者给了一个很直观的 SNR（信噪比）解释：设正确 trajectory 分数 $s_c$、错误 trajectory 分数 $s_i$，SNR = E[s_c - s_i] / sqrt(Var(s_c - s_i))。SNR 越高，分得越开。结果显示 G 越大，SNR 越高（0.775 → 0.799）。

**关键洞察**：增大 G 不给模型任何新信息（token set 翻倍，但每个 token 的 logprob 没变），但给了"投影空间"更细的网格——原本会被 round 到同分数的细微置信差异现在能区分开。这是个非常 clever 的设计。

### 轴 2：重复评估 K

每次 query 一次 verifier，再 query 一次，再 query 一次，取平均。K=1 vs K=16：

| K | 1 | 4 | 16 |
|---|---|---|---|
| Verification Accuracy | 74.7% | 77.1% | 77.5% |

看起来只涨了 2.8 个点，但**关键改进是 tie rate 暴跌**——这是更有杀伤力的：

| K | 1 | 4 | 16 |
|---|---|---|---|
| Judge 的 Tie Rate | 26.7% | 11.7% | 5.5% |
| Verifier 的 Tie Rate | 0.0% | 0.0% | 0.0% |

**判别为 0% 才是真正的杀手锏**。从 1→16，Judge 的平分率从 26.7% 砍到 5.5%，但 Verifier（用 logits 期望）从一开始就是 0%。这意味着你想用 verifier 在 N 条 trajectory 里挑最优，至少能保证**一定能排出严格的序**——这就是 Pass@K oracle 能摸到 98.9% 的底层保证。

### 轴 3：评价准则分解 C

把"这一条 trajectory 整体打分"拆成多角度：比如代码任务拆成"specification（是否符合任务要求）/ error（是否报错）/ output（最终结果对不对）"，最后 ensemble 起来。

| 准则 | Specification | Error | Output | Ensemble (All) |
|---|---|---|---|---|
| Verification Accuracy | 75.2% | 76.0% | 76.4% | **78.3**% |

单准则最高 76.4%，三个 ensemble 涨到 78.3%，比最好的单准则高 1.9 个点。

这种 decomposition 不是论文独创——FLASK、HDEval 之前都做过——但**和前两个轴叠加在一起就是 multiplier 的效果**。

### 三个轴放在一起

![三轴 scaling 效果](https://arxiv.org/html/2607.05391v2/x3.png)

*图 3：Verification Scaling 三轴效果。Terminal-Bench V2 上：左图是 G 从 1 涨到 20，准确率从 73.1% 涨到 77.5%；中图是 K 从 1 涨到 16，74.7% → 77.5%；右图是 C=3 时 ensemble 比单准则高约 2 个点。三个轴都能独立涨。*

三条曲线在 Terminal-Bench V2 上**单调递增**，没有任何一条出现饱和或反降。这点很重要——意味着没有哪个轴被"用完"。

---

## PPT 算法：把 N 条 trajectory 排序的预算砍到 O(Nk)

光有"连续分数"还不够。生产环境里 N=20 条 trajectory 怎么选？朴素做法是 $\binom{20}{2}=190$ 对全比较，这就把 verifier 算力烧光了。论文设计了一个 **Probabilistic Pivot Tournament（PPT）**：

![PPT 五步流程](https://arxiv.org/html/2607.05391v2/x4.png)

*图 4：Probabilistic Pivot Tournament 五步流程。1) 候选池；2) 随机 Hamilton 环对所有相邻对打分（每个 candidate 既当 A 又当 B，对消位置偏差）；3) 排序后选 top-k 做 pivot；4) 剩下的只跟 pivot 比，pivot 之间再互比；5) 按 $w_i / c_i$ 归一化选最大。*

整个 PPT 的妙处：

1. **Ring pass 消位置偏差**：随机抽个 Hamilton 环，每条 trajectory 恰好在 A 位和 B 位各出现一次。LLM-as-a-Judge 有"倾向选 A"的偏置（Zheng et al. 2023 的经典发现），环结构让这个偏置在期望上对消。
2. **Pivot 选 top-k**：第一轮 ring pass 后按 w/c 排序，置信最高的 k 条做 pivot 集合 $\mathcal{P}$。剩下 N-k 条不再互相比，只跟 $\mathcal{P}$ 里的每条比。
3. **Pivot 之间再互比一次**：避免"全部 N-k 条输给同一条 pivot"的情况。
4. **归一化选 $w_i / c_i$ 最大的**：因为 pivot 之间比了多次、非 pivot 只比了 k 次，要除以参与比较次数。

预算从 $\mathcal{O}(N^2)$ 砍到 $\mathcal{O}(Nk)$。对 N=20、k=4，190 次比较降到 80 次左右。论文 Table 9 显示 PPT 始终优于 V1（Singh 2026），且 pivot 越多越好。

---

## 实验：四榜 SOTA，但 verifier 用的不是最大的模型

下面这张主表是论文最硬的部分。

![四榜 SOTA 对比](https://arxiv.org/html/2607.05391v2/figures/SOTA.png)

*图 5：四个 benchmark 上 LLM-as-a-Verifier（红）与最强 baseline（米色）的对比。全部击败。Verifier 用的是 Gemini 2.5 Flash（在很多 baseline 之下），靠 verifier 框架取胜。*

具体数字整理成表：

| Benchmark | Pass@1 (单 trajectory) | Oracle Pass@K (上限) | LLM-as-a-Verifier | 最佳 baseline |
|---|---|---|---|---|
| Terminal-Bench V2 | GPT-5.5: 83.1% | 92.1% | **86.5**% | GPT-5.5: 84.7% |
| SWE-Bench Verified | mean: 76.1% | 84.4% | **78.2**% | Opus 4.5: 76.8% |
| RoboRewardBench | — | — | **87.4**% | RoboReward-8B: 81.4% |
| MedAgentBench | Opus 4.8: 70.2% | 75.0% | **73.3**% | Opus 4.8: 70.2% |

**几个值得说的细节**：

- **Terminal-Bench V2**：Verifier 是 Gemini 2.5 Flash（远比生成方 GPT-5.5 弱），但通过 N=5 选最优，pass rate 从 GPT-5.5 单条 83.1% 涨到 86.5%，距离 oracle 92.1% 只差 5.6 个点。
- **SWE-Bench Verified**：候选池**异质**——从 Opus 4.5、Gemini 3 Flash、M2.5 各抽 1 条，避免同质 baseline 跑出来的作弊嫌疑。Verifier 还是 Gemini 2.5 Flash。
- **RoboRewardBench**：直接击败**为这个 benchmark 训练过的 reward models**——RoboReward-8B（45k episodes）、Robometer-4B、TOPReward 都被它一个 0-shot 框架干翻。这其实最吓人。
- **MedAgentBench**：医疗 EHR 任务，验证错误的代价大，是个有意义的 stress test。

论文里我特别喜欢 Table 8 的 harness generalization：在 Terminus-2（GPT-5.3-Codex proposals）和 Terminus-Kira（Opus 4.6 proposals）两个**完全不同的 agent harness** 上分别跑，Verifier 增益从 +2.7 到 +8.3 个点都能跑出来。说明这套方法没有绑死在某个 agent 框架上。

---

## Progress Tracking：当 verifier score 变成时间轴的"心电图"

除了"选最优 trajectory"以外，论文发现 verifier 给出的连续分数**沿 trajectory 步骤单调上升**（成功时），失败时则不升或乱跳。

![代码生成任务的 progress tracking](https://arxiv.org/html/2607.05391v2/x6.png)

*图 6：MNIST inference 任务的 progress score 沿步骤变化。绿色 (SUCCESS) 的 trajectory 分数随步骤上升，最后 1.0；红色 (FAILED) 的 trajectory 因装错包、磁盘满、编译错误，分数一直低位徘徊，到末尾还只有 0.3。*

论文把这个性质形式化成 **Value-Order Correlation（VOC）**——步骤序号和 verifier 分数的 Spearman 秩相关。在 Terminal-Bench V2 500 对 trajectory 上：

| Trajectory outcome | Spearman VOC |
|---|---|
| 成功的 | 0.848 ± 0.012 |
| 失败的 | 0.769 ± 0.016 |
| 差距 | +0.079 |

在机器人 RoboRewardBench 500 条 trajectory 上更夸张：

| 方法 | Spearman VOC |
|---|---|
| **LLM-as-a-Verifier（Qwen 3.6 35B, K=5, G=20）** | **0.966** |
| RoboReward-8B | 0.877 |
| Robometer-4B | 0.780 |
| TOPReward (P(true)) | 0.565 |

TOPReward 在末尾饱和到 P(true)=1.0，所有"看起来像在做事"的步骤都打高分，VOC 反而被冲低。LLM-as-a-Verifier 没有这个毛病。

**最有意思的是这个性质的工程化用法**：作者做了个 **TurboAgent**——一个 inference-time proxy 塞在 Claude Code 和后端 LLM 之间，对用户透明。它除了用 PPT 选最优回复以外，还能**实时把 verifier score 暴露给用户**，让长跑的 agent 任务能被监测、暂停、回滚。等于把"日志心电图"装到 coding agent 上了。

---

## 把 verifier 当 dense reward 用：RL 也能蹭

Verifier 既然能给"沿步骤的连续分数"，自然就能塞进 RL 当 dense reward。论文做了两个 setting：

### Off-policy：DSRL-SAC 训 $\pi_0$ 玩 LIBERO

每个 rollout 结束后，对采样出的帧序列 $\tau_{1:t}$ 计算 $\rho_t = R(x, \tau_{1:t}) \in [0, 1]$，然后用 shaped reward：

$$r_t = r_t^{env} + \lambda \cdot \rho_t$$

存储 relabeled transition 到 replay buffer 训 SAC。这里 shaping 离线做，不动 SAC 目标函数本身，相当于"白嫖"了一层 dense signal。

![LIBERO 训练曲线](https://arxiv.org/html/2607.05391v2/x7.png)

*图 7：左图是 LIBERO ketchup 任务上 $\pi_0$ + DSRL-SAC 的 success rate vs steps。Verifier 持续 reward 1.8× 加速，最终成功率 0.76 vs 0.69；右图是 MATH 上 Qwen3-8B + GRPO，加速 1.1×，成功率也更高。*

### On-policy：GRPO 训 Qwen3-8B 解 MATH

GRPO 早期容易"全军覆没"——一组 G 条 response 全错，group-relative advantage 全是 0，没梯度。这时候 verifier 出来的"reasoning trace 分数"就能当 dense signal 救场：

$$r_i = r_{correct, i} + r_{format, i} + \beta \cdot r_{reasoning, i}$$

论文在 0.2 到 0.6 的 success rate target 上测了，dense reward 平均 **1.1× 加速**。

说实话，1.1× 在 RL 里已经算不错的加速了——RL 训练本身方差大、随机性强，能稳定拿到 10% 提速不是个小数字。

---

## 我的判断：硬核 + 干净，但别被"无需训练"忽悠

读完整篇，我心里的感觉是：**这是那种"工程师立刻想抄回自己项目里"的 paper**。方法不复杂、实验覆盖广、数据真材实料、跨域 SOTA。

**但我得说几点等等**：

**第一，"无需训练"是个 trade-off 包装**。论文反复强调"no additional training"，但实际依赖一个**能暴露 top-k logits 的 verifier 模型**。Gemini 2.5 Flash / Qwen 3.6 这种模型刚好满足。**只开 API 不给 logit 的 frontier model（GPT-5、Opus 4.7）在这套方法下完全用不了**。作者在 Appendix B.6 提了个 two-stage workaround（用闭源模型生成 reasoning 文本，再用一个开放 verifier 去 log），但代价是脱离了"单模型"的简洁性。这个 trade-off 你在做工程选型时一定要想清楚。

**第二，verifier 选什么模型本身就是个 meta 问题**。论文里 Terminal-Bench 用 Gemini 2.5 Flash、RoboRewardBench 用 Qwen 3.6 35B。**两个都不是 SOTA 级别的生成模型**。这套方法在"用弱 verifier 训强生成方"时表现得很好，但**反过来**——用一个强 verifier 去验证一个比它弱的生成方，能带来多少增益？论文没有正面回答。从经验看，强 verifier 对弱生成方应该也有用，但**边际收益曲线**是什么形状？论文没说。

**第三，evaluation 准则 C 是手设计的**。代码任务拆成 spec/error/output 三块是合理的，但**怎么知道哪个拆法对**？论文没给 systematic 的方法。如果 C 是自动的、可学习的（用 LLM 自己去生成评价维度），这套框架的普适性会上一个台阶。Appendix 里说"future work"会做，但当前没有。

**第四，跟 PRM/ORM 的关系没讲清楚**。Process Reward Model（Lightman 2023）和 Outcome Reward Model（Cobbe 2021）已经在 reasoning 领域被广泛研究。LLM-as-a-Verifier 实际上可以看作"不需要训练、能输出连续分数的 PRM 替代品"。但**论文没在 MATH step-level 这种典型 PRM 场景上做 head-to-head 对比**——只在 GRPO 的 reasoning reward 上做了一次端到端实验，1.1× 加速。如果直接和成熟的 PRM（比如 math-shepherd、rStar-Math 用的那种）比，结果会怎样？我有点好奇。

**第五，N 和 K 的选择有玄学**。所有实验都是 N=3 到 N=20，K=1 到 K=16。**没有一个"自动决定 N 和 K"的规则**。生产环境里你得自己拍。给个 heuristic：先用 oracle pass rate 估算 headroom，再选 K 直到 tie rate 接近 0。

---

## 同期工作位置

把"把 logit 期望当 reward"这件事拎出来看，**和 Generative Verifier（Zhang 2025）一脉相承**——后者把 reward modeling 重新 cast 成 next-token prediction，用 SFT 训一个新的 verifier 模型。LLM-as-a-Verifier 更进一步：**根本不去训 SFT 后的 verifier，直接在原始的 LLM 上读 logits**。这是它的核心省力点。

和 **V1（Singh 2026，agentic RM）** 比：V1 是 round-robin pairwise 排序，PPT 是更聪明的 budget-aware 排序，胜出。

和 **TOPReward（Chen 2026）** 比：TOPReward 用 P(true) 作为 progress signal，结果饱和；LLM-as-a-Verifier 的连续分数没这个问题，VOC 0.966 vs 0.565。

和 **Robometer / RoboReward** 比：这些都是**专门为机器人任务训练**的 reward model，LLM-as-a-Verifier 是 0-shot 通用框架。结果是 0-shot 通用打败了专项训练，这其实是个挺强的工作——但也意味着**一旦领域有足够的训练数据**（比如工业级机器人量产），专项 RM 还是会有它的位置。

---

## 收尾：把 logit 读出来是件大事

整篇论文的真正贡献，我个人认为是"**把 logit 分布当成 reward 信号**"——这一观察。

在 pre-LLM 时代，logit 早就是 confidence score 的标准接口。但 LLM 时代，**大家都习惯让模型解码出"字面 token"再做事**，logit 反而被忽视了。这篇 paper 重新提醒：logit 本身是有信息量的。

这个观察能立刻 unlock 一堆事：
- 你训的任何 reward model，**用 logprob 的 expectation 当 scalar target**，会比 cross-entropy 训出来的 discrete target 更好。
- **任何 LLM-as-a-Judge 场景**（评测代码、写作质量、对话流畅度），把 prompt 改成"用一个评分 token 集合打分"再读 logprob 期望，会比"输出一个整数"分得更开。
- **多模态场景**（视频、图像、音频），只要底座模型给 logit，同一套 PPT + 三个轴直接复用——不需要重新设计架构。

这也是为什么这篇 paper 选了**横跨 coding / robotics / medical** 三个完全不同的领域做实验——它要证明的不是某个具体 benchmark 上的胜利，而是"这个方法论**有 scaling 能力**"。

如果你在做的事需要 fine-grained quality signal（agent 评测、RL dense reward、agent progress monitoring、模型 selection），这个思路**应该现在就抄到你的代码里**——成本就是 prompt 里加一行"please use a 1-20 letter scale"，再加 5 行读 logprob。

剩下的就是把 C、G、K 调到合适。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
