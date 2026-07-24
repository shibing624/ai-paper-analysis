---
title: 用 surrogate 策略给潜空间推理装上 outcome-RL：拆解 SLPO
date: 2026-07-24
arxiv: 2607.19691
paper: SLPO: Scaling Latent Reasoning via a Surrogate Policy
authors: Runyang You, Zhiyuan Liu, Yongqi Li, Wenjie Li
affiliations: The Hong Kong Polytechnic University; Sichuan University
---

# 用 surrogate 策略给潜空间推理装上 outcome-RL：拆解 SLPO

你有没有这种感觉：明明 CoT 推理已经让 RL 起飞了，DeepSeek-R1、o1 这些"会思考的模型"靠着 outcome reward 把 test-time scaling 推到新高度，但隔壁搞 latent reasoning（潜空间推理）的那帮人还停留在蒸馏阶段——COCONUT、CODI、CoLaR 一水的 SFT，像一群在 IM 圈外打转的陪练。

问题出在哪？把 outcome-reward RL 搬到潜空间这件事，听起来就一个字：**难**。

显式 CoT 那条路之所以跑得通，是因为每一步推理都是 token，token 自带 vocab 分布，policy gradient 拿到 $\log \pi_\theta(a_t \mid s_t)$ 就能算优势、做信用分配。中间步骤想长就长想短就短，可变 horizon 这件事也是 RL 自己学出来的。

但 latent reasoning 呢？中间步骤是连续向量，不走 vocab head，没有 action probability 这条线，policy gradient 的扳手直接卡壳。雪上加霜的是，COCONUT 这些老前辈还把 thinking budget 写死成超参，模型连"什么时候该停"都决定不了——这种情况下就算有奖励信号，RL 也无从下手。

所以过去几年，latent reasoning 一直有个尴尬的局面：在短 horizon 任务上能压 CoT 一头，但只要涉及"想得再深一点"，就熄火。

arXiv:2607.19691 这篇 SLPO（Surrogate Latent Policy Optimization）想干的事，就是把 outcome-reward RL 原原本本搬进潜空间。搬的方式很巧妙——它不强行给 latent 转移造一个真密度，而是用 MC-dropout 采样出一个**经验高斯分布**，当作 policy 的 surrogate；然后再加一个**带冷启动的 stop gate**，让模型自己决定在第几步停。

两件套组合起来，SLPO 第一次让 latent reasoner 真正具备了两条腿：**Pass@k 的并行 scaling**和**按难度分配的串行 scaling**。在 Llama-3.2-1B 上，CODI+SLPO 的 GSM8K Pass@16 从 67.48 抬到 70.28；在 3B 的软 token 设置里，AIME 2025 的 Pass@1 从 LEPO 的 0.96 翻到 3.33。

---

## 论文基本信息

- **标题**：SLPO: Scaling Latent Reasoning via a Surrogate Policy
- **作者**：Runyang You、Zhiyuan Liu、Yongqi Li、Wenjie Li
- **机构**：The Hong Kong Polytechnic University；Sichuan University
- **链接**：https://arxiv.org/abs/2607.19691
- **发表日期**：2026-07-22

---

## 为什么 latent reasoning 跑不动 outcome-RL？

先把病灶看清楚。论文把这件事归结为两个独立但同时存在的瓶颈：

**瓶颈一：没有 per-step likelihood。** 显式 CoT 里每一步 $y_t$ 是从 vocab 采的，能直接算 $\log \pi_\theta(y_t \mid x, y_{<t})$，优势函数 $A_i = R_i - b_i$ 直接乘上去就完事。但 latent state $h_t$ 是连续向量，从 dropout mask 出来的，没法写出一个简洁的 $\log p(h_t \mid h_{<t})$。结果你拿不到 policy gradient 的抓手。

**瓶颈二：没有可变 horizon。** COCONUT、CODI 这些方法把潜步数 $T_{\max}$ 当成固定超参，模型不会"这一步我想清楚了可以停"。RL 本来可以通过 advantage 重新分配 compute，但既然 horizon 写死了，奖励信号连"算得再久点"这件事都改不动。

没有 likelihood，信用分配做不了；没有可变 horizon，test-time scaling 这条路也封死。两堵墙一立，outcome reward 在 latent 那边就成聋子的耳朵。

SLPO 的解法思路是：一堵墙一把钥匙。

---

## SLPO 的两把钥匙

SLPO 的整体架构长这样：

![SLPO 概览：三条 policy 共同决定 loss，verifier 提供 advantage](https://arxiv.org/html/2607.19691v1/x1.png)

*图 1：SLPO 总览。Prompt 进入潜空间推理模型，生成 latent 序列，stopping gate 决定在第几步停下，再由 verifier 给出 advantage $\hat{A}_i$。最终的策略梯度目标把"潜空间转移的 surrogate likelihood + 停止时间 likelihood + 最终答案 token likelihood"三段 log-likelihood 加权求和。*

### 钥匙一：经验 surrogate likelihood

既然 dropout 采样没法写出解析密度，那就**用采样本身当密度**。

具体做法：对每个 latent 转移 $h_t$，跑 K 次带独立 dropout mask 的前向，得到 $z_t^{(1)}, \ldots, z_t^{(K)}$，算一个经验均值 $\mu_t$ 和对角方差 $\sigma_t^2$，再把真实的 $h_t$（rollout 采样的那一个）放到这个对角高斯里取 log-density：

$$
\log \hat{\pi}_\theta(h_t \mid x, h_{<t}) = -\frac{1}{2}\sum_{j=1}^{d}\left[\frac{(h_{t,j} - \mu_{t,j})^2}{\sigma_{t,j}^2} + \log(2\pi\sigma_{t,j}^2)\right]
$$

这一项是 surrogate，不是 MC-dropout 的真实密度——作者特意强调这一点。它的作用仅仅是**给每个 latent 转移打一个可微的标量分数**，让 policy gradient 能乘 advantage 倒着推梯度。

实现上有个关键 trick：rollout 采到的 $h_t$ 当 stop-gradient target，梯度只走当前的"重算"那条路，避免把 rollout 路径也拉进来。这是和 PPO / off-policy 那套通用的 stop-gradient 范式同源的工程细节。

直觉上，这就像**给一个没标价的商品贴个临时的价格标签**——不要求它真的等于市场价，只要它能稳定反映"这步走得对不对"，RL 就有信号可用。

### 钥匙二：可学习 stop gate + 冷启动

光有 surrogate 还不够，horizon 写死的老问题没解决。SLPO 在 latent reasoner 上挂一个 stop head $g_\theta$：对每个 latent state $h_t$ 输出一个停下的概率 $\rho_t = \sigma(g_\theta(h_t))$。整个序列的 stop 时间 $\tau$ 服从几何分布：

$$
P_\theta(\tau = t) = \rho_t \prod_{k<t}(1 - \rho_k)
$$

问题来了，gate 一开始是随机权重，没人教它什么时候该停。SLPO 设计了一个 **correctness-supervised 冷启动**：

- 拿当前 backbone 在每个训练样本上 rollout N=4 条潜空间轨迹
- 对每条轨迹，在 $[T_{\min}, T_{\max}]$ 范围内枚举所有候选停步 $t$，在 $h_{1:t}$ 之后强制 decode 最终答案
- 答案对就标记 $t$ 属于"正确停步集合" $\mathcal{V}^{(n)}_i$
- 训练目标：把 stop 概率质量往 $\mathcal{V}^{(n)}_i$ 里推
  $$\mathcal{L}_{\text{stop}}^{(i)} = -\frac{1}{N}\sum_n \log \sum_{t \in \mathcal{V}^{(n)}_i} P_\theta(\tau^{(n)}_i = t)$$

这一步是 15 个 epoch 的纯监督训练（用 Adafactor，lr=1e-4），不用 RL 信号，只是给 gate 一个"我大致知道什么时候能停"的先验。

冷启动之后，再用 RLOO 风格的 outcome-reward RL 联合优化三段 log-likelihood：潜空间 surrogate + 答案 token + 停止时间。整体 loss 就是标准 REINFORCE 的样子：

$$\mathcal{L}_{\text{RL}} = -\hat{A}_i \log \pi_\theta(\xi_i \mid x_i)$$

其中 $\log \pi_\theta(\xi_i \mid x_i)$ 是把三段 log-likelihood 拼起来。这一步把"答对"和"答得早"两条 credit 一起灌进模型。

---

## 实验在打什么仗

### 设定

- **数据集**：GSM8K-Aug 训练，GSM8K-Test / GSM-Hard / MultiArith 三个 OOD 评测；软 token 那一组还加了 MATH500、AIME 2025、AMC23
- **基座**：GPT-2 (124M) 和 Llama-3.2-1B-Instruct（主表），3B（软 token 转移）
- **基线**：COCONUT、CODI、CoLaR、ReGuLaR、DART、Latent-SFT、CoT-SFT、iCoT、LEPO、HRPO、Soft Tokens
- **评估**：Acc 是一次确定性推理，Pass@k 是 k 条独立 MC-dropout rollout 至少有一条对

### 主表：Pass@k 全面涨，Acc 涨得克制

主结果在 Table 1。最值得看的不是绝对数，而是 **Pass@k 的提升幅度普遍比 Acc 大**——这点很合理，因为 surrogate likelihood 加 outcome reward 其实就是在塑形 latent rollout 的分布，多采样一次就多一次"撞对"的机会。

GPT-2 124M 上的 COCONUT backbone（最弱的一个）：

| 方法 | GSM8K Acc | P@8 | P@16 | GSM-Hard Acc | P@8 | P@16 | MultiArith Acc | P@8 | P@16 |
|------|-----------|-----|------|--------------|-----|------|----------------|-----|------|
| COCONUT | 34.12 | 45.79 | 49.13 | 7.66 | 10.70 | 12.06 | 80.86 | 88.79 | 91.38 |
| + SLPO | **35.63** | **49.13** | **51.55** | 7.66 | **10.85** | **12.52** | **83.10** | **91.38** | **92.59** |

Llama-3.2-1B 上的 CODI backbone（最强 baseline）：

| 方法 | GSM8K Acc | P@8 | P@16 | GSM-Hard Acc | P@8 | P@16 | MultiArith Acc | P@8 | P@16 |
|------|-----------|-----|------|--------------|-----|------|----------------|-----|------|
| CODI | 55.22 | 63.91 | 67.48 | 12.82 | 15.02 | 15.63 | 95.52 | 98.28 | 98.79 |
| + SLPO | **55.27** | **65.13** | **70.28** | **13.20** | **15.86** | **16.77** | **96.38** | **98.79** | **99.48** |

注意 Llama-3.2-1B + CODI + SLPO 在 GSM8K 上 Acc 几乎没动（55.22 vs 55.27），但 Pass@16 涨了 2.8 个点。说明单次推理基本饱和，但 RL 拓宽了"多次采样时的覆盖"。

弱 backbone 上提升更大（GPT-2 + COCONUT 的 P@16 +2.4，提升幅度 5% 左右），强 backbone 上提升小但稳定。这其实是个常见模式——基础越好天花板越近，SLPO 的上限不会改变，但能把"采样效率"往上推一推。

### 软 token 设置：AIME 大涨

Table 4 那个 AIME 2025 Pass@1 值得单独提一下：

| 方法 (3B) | GSM8K P@1 | MATH500 P@1 | AIME 2025 P@1 | AMC23 P@1 |
|-----------|-----------|-------------|---------------|-----------|
| CoT | 73.22 | 43.75 | 0.52 | 21.48 |
| GRPO | 74.69 | 43.89 | 0.42 | 24.14 |
| LEPO | 77.29 | **46.51** | 0.96 | 27.03 |
| **SLPO ours** | **77.63** | 45.00 | **3.33** | **32.50** |

AIME 2025 的 Pass@1 从 0.96 翻到 3.33，AMC23 从 27.03 抬到 32.50，这是几倍的相对提升。背后的物理意义是：在采样本身就很难的高难度题上，**outcome-reward RL 真的把潜空间 rollout 的成功率抬起来了**——这是 CoT-SFT 单纯对齐 teacher 永远拿不到的能力。

---

## 分析一：超参 G 比 K 更敏感

作者扫了 rollout 超参 $K$（surrogate 的 forward 样本数）和 $G$（每个问题的 rollout 数）。结论很直白：

- $G$（rollout 数量）影响大，从 2 涨到 8，Pass@2 在 GSM-Hard 和 MultiArith 上稳定提升
- $K$（surrogate 样本数）影响小，$K=4$ 基本够用，再大边际收益消失

这其实是符合直觉的——$G$ 直接影响 advantage 估计的方差（REINFORCE 的老问题），$K$ 只影响 surrogate 密度估计的稳定性，后者只要别太离谱就能用。

![RLOO 与 GRPO 在 SLPO 框架下的对比](https://arxiv.org/html/2607.19691v1/x6.png)

*图 2：SLPO 框架下 RLOO 与 GRPO 的 Pass@k 对比。蓝色是 COCONUT，红色是 CODI；实线是 RLOO，虚线是 GRPO。两条线在每个面板都基本贴在一起。*

这点很重要：**SLPO 的 surrogate 是一层独立的"信用分配器"，它上面可以接 RLOO 也可以接 GRPO，两者效果几乎一致**。这意味着 latent 转移的 likelihood 工程是 SLPO 真正的贡献，而具体用哪个 REINFORCE 变体属于次要选择。

---

## 分析二：软 token 上 SLPO 一样能跑

SLPO 不挑 latent 表示。Table 3 / Table 4 是在软 token（softmax-weighted embedding，不是 backbone hidden state）上做的实验。SLPO 的处理是：把 surrogate 直接套在 $z_t = p_t^\top E$ 上，用各向同性高斯代替对角高斯，其他一切不变。

训练过程中的 rollout 长度变化是这张图：

![软 token 训练时的序列长度变化](https://arxiv.org/html/2607.19691v1/x7.png)

*图 3：Llama-3.2-1B 上软 token outcome-RL 训练时的平均生成长度。SLPO（红）的 rollout 长度在 step 60 之后稳定在 1000 以上，GRPO（浅蓝）大致稳定在 850-900，LEPO（深蓝）反而从 ~800 缩到 ~650。*

这张图很能说明问题。SLPO 训练过程中，**rollout 长度在主动增长**——意味着模型自己学会了"用更长的推理换更对的答案"，而 GRPO 和 LEPO 都没做到，LEPO 甚至在收缩。这是 latent test-time scaling 第一次显式地从 outcome RL 中涌现出来，而不是被超参写死。

---

## 分析三：难度-长度正相关

这是 stop gate 的存在性证明。如果 gate 真学会了自适应，理论上难题应该分到更多 latent 步。Fig. 7 把 GSM8K 验证集和测试集按 32 次采样下的错误率（$1 - \text{acc@32}$）分箱，看每个难度 bin 的平均 latent 长度：

![GSM8K 验证集上难度与潜空间长度的关系](https://arxiv.org/html/2607.19691v1/x10.png)

*图 4：GSM8K 验证集上，按难度分箱后的平均 latent 长度。横轴是 $1 - \text{acc@32}$ 分箱，纵轴是平均潜空间步数。Pearson r = 0.297，线性拟合正斜率。*

Pearson r = 0.297（验证集）和 0.26（测试集）——相关性不算极强，但**单调递增**这件事在两个 split 上都成立。简单说，gate 不是在按某个固定 schedule 走，它确实在按题目的难度分配 compute。

---

## 分析四：潜空间几何怎么变

还有个偏"科学好奇"的分析：SLPO 训练完之后，latent 序列的几何性质有什么变化？

- **Inter-step cosine distance**（相邻 latent state 之间的余弦距离）整体涨了。意味着**每一步比上一步走得更远**，推理过程的阶段化更清晰
- **Prefix effective rank**（前 t 步的稳定秩）整体降了。意味着潜空间序列**收敛到一个更低维的子空间**，模型在"用更少的独立方向表达更多内容"

这两条信号叠在一起的物理图像是：**RL 把潜空间推理从漫无目的漂移状态压成沿少数任务相关方向定向推进**。这其实和显式 CoT 训练后期 reasoning 路径变得更"专"是同一种现象。

---

## 我的判断

先说亮点。

**亮点一：问题定义精准。** 论文把"latent RL 跑不动"这件事拆成"没 likelihood + 没可变 horizon"两个独立瓶颈，每个瓶颈给一个解法，整体拼成一个端到端的目标函数。这种"先诊断后开方"的写法在 latent reasoning 圈里比较少见——很多工作上来就是一套新架构，但说不清为什么 CoDI 那种架构不够用。

**亮点二：surrogate 的设计很巧。** 不强求写出真实密度，用 MC-dropout 采样本身当密度的经验估计——这等于把"潜空间 RL"这件事从"建模分布"降维成"用一个合理的标量打分"。工程上实现成本低、迁移到任意 latent 架构（hidden state、soft token、可能是 multimodal）都顺。

**亮点三：消融和分析做得全。** 不光有主表，还给了 G/K sweep、RLOO/GRPO 对比、软 token 转移、潜空间几何变化、难度-长度分配。**这套分析组合起来让结果可信度比单看主表高一个量级**。

再说几个让我皱眉的地方。

**皱眉一：Acc 的提升其实有限。** Llama-3.2-1B + CODI 这个最强组合上，GSM8K Acc 只动了 0.05 个点（55.22 → 55.27）。Pass@k 是涨了，但单次推理几乎没变。论文的叙事把"Pass@k 涨"和"latent test-time scaling"绑在一起，但严格说 Pass@k 涨意味着**多次采样下的覆盖更好**，这在部署成本敏感的场景里未必划算。

**皱眉二：surrogate 的"诚实性"问题。** 作者自己强调这是 surrogate，不是真实密度，但既然梯度流过它，surrogate 偏差大不大？$K=4$ 真的够估对角高斯吗？论文没有正面回答这个——$K$ sweep 显示 $K=4$ 之后几乎没提升，但这只能说明 $K=4$ 进入了"稳定 plateau"，不能说明 surrogate 本身和真实密度的接近程度。严格来说这是一个未关闭的经验问题。

**皱眉三：stop gate 的冷启动数据从哪里来。** 冷启动那一步需要枚举 $[T_{\min}, T_{\max}]$ 上每个候选 prefix 的答案正确性。这意味着冷启动本身需要 backbone 已经能"在某些 prefix 上答对某些题"——这件事对 COCONUT 那种弱 backbone 是不是成立？论文没单独报告冷启动的成功率，只展示了冷启动之后的 RL 阶段。如果冷启动本身就训不好，整个方法就立不住。

**皱眉四：评测只覆盖小模型和数学。** 主实验是 GPT-2 124M / Llama-3.2-1B，最大到 3B 软 token 转移。所有评测在 GSM8K / GSM-Hard / MultiArith / MATH500 / AIME / AMC23——全是数学。**代码、开放问答、agent 任务这些典型的 latent reasoning 应用场景完全没碰**。这是 latent reasoning 圈的通病，但 SLPO 没解决。

**皱眉五：跟同期工作比，"首个"这件事要打折扣。** 论文说"outcome-reward policy optimization over unconstrained hidden-state recurrence, without vocabulary-level probabilities, remains unexplored"——这个 claim 我得拆开看。LEPO 和 Latent-GRPO 都在做 latent RL，但确实它们依赖 vocab 路由（soft token）。所以 SLPO 可能是**首个不依赖 vocab 路由的纯 hidden-state outcome-RL**。但这个"首个"的边界很细，而且 LEPO 的 AIME 数据其实不在 SLPO 的 Table 3 那个 1B 设定下——3B 设定下 LEPO 的 AIME 是 0.96，SLPO 是 3.33，差距 3.5 倍，这是 SLPO 真正有说服力的点。

---

## 工程视角：能不能直接用？

如果想在已有 latent reasoner 上加 outcome-RL，SLPO 提供的工程模板很清爽：

1. **不动 backbone 的循环结构**——COCONUT、CODI 原样的向量递归直接保留
2. **只加一个 stop head**——一个 MLP 头接在 latent state 上
3. **冷启动 15 个 epoch**——先学会"在能答对的 prefix 停下"
4. **RLOO 联合训练 4 万步**——Adafactor，lr=1e-6，无 KL 惩罚，无 PPO clipping
5. **推理时 gate 阈值在 $\{0.5, 0.6, 0.7, 0.8, 0.9\}$ 里扫**——按验证集 Acc 选

硬件是 4×RTX 5880 Ada，对绝大多数 lab 来说负担不大。

唯一需要警惕的是 **surrogate 的稳定性**。如果你的 latent reasoner 用的是非 MC-dropout 形式的随机性（比如加噪），surrogate 那一段可能要重写——论文的实现绑定在 dropout mask 上。

---

## 一句话收尾

SLPO 做了一件听起来很 boring、实际很关键的事——**它把 outcome-reward RL 这条显式 CoT 上跑通的训练范式，原样搬进了 latent 空间**。搬的过程中没有强行给连续向量写解析密度，而是用"采样本身当密度"的工程化近似绕开了那道数学墙；再加一个 correctness-supervised 的 stop gate，horizon 可变这件事也顺带解决了。

结果未必是颠覆性的——Acc 在强 backbone 上几乎不动——但 Pass@k 和训练时的串行 scaling 是真实存在的。在 latent reasoning 还在啃 SFT 这根骨头的今天，这一步相当于给它装上了 RL 这条腿。至于这条腿能不能走远，得看下一波工作能不能把 surrogate 的偏差控制、stop gate 的冷启动数据来源、跨任务迁移这几件事讲清楚。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
