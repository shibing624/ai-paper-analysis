# RL 训练时模型已经走错了，还非要让它写完 8192 个 token？阿里 Tongyi Lab 这篇 ESPO 把失败 rollout 提前掐了

## 🎯 核心摘要

做过 LLM RL 训练的人应该都有这种体验：模型在第 200 个 token 就已经把数学题理解错了，但 PPO 算法依然得让它生成到 8192 才肯停下，这中间几千个 token 全是"错上加错"的废话——既不会拿到正奖励，又会污染整条 trajectory 的优势估计。这种 **rollout continuation 浪费** 是个公开的痛点，但之前的方法要么靠人工标注的 PRM（贵），要么靠学一个独立的 termination 模块（复杂）。

阿里 Tongyi Lab 和北大的这篇 **ESPO（Early-Stopping Proximal Policy Optimization）** 给出了一个挺工程化的方案：**完全不引入新模型、不要新标注**，复用 actor 自己计算 logit 时的副产品——一个 token 的 "regret"（采样 token 与贪婪 token 的对数概率差），再结合 critic 的 value 做一个动态阈值判断。一旦累计 regret 超过 value 给的"容忍上限"，就当场截断这条轨迹，把它当吸收失败态打一个终端负奖励。

效果在 DeepSeek-R1-Distill-Qwen-7B 上：AIME 2024 从 PPO 的 45.25% 涨到 46.28%，AMC 2023 从 82.94% 涨到 85.83%，MATH-500 从 85.43% 涨到 87.42%——同时 **rollout token 累计省了 22% 以上**。换句话说，又快又准。是一篇典型的"工程结合 RL 直觉"型论文，思路漂亮、改动量小、值得在自己的 RL 训练 pipeline 上试一试。

---

## 📖 论文信息

- **标题**：ESPO: Early-Stopping Proximal Policy Optimization
- **作者**：Zihang Li, Rui Zhou, Yingcheng Shi, Wenhan Yu, Zhewen Tan, Zixiang Liu, Zeming Li, Binhua Li, Yongbin Li, Tong Yang, Jieping Ye
- **机构**：Tongyi Lab, Alibaba Group；Peking University
- **arXiv ID**：2605.29860（2026/05/28）
- **链接**：https://arxiv.org/abs/2605.29860

---

## 🤔 一个让人挺难受的现象：错了之后还得写 5000 个 token

先讲个我自己很有共鸣的场景。

之前调一个数学推理的 RL 任务，rollout 长度上限设到 8192。某些 prompt 模型基本就是开局放飞——第一步看错了符号，整条链路就算彻底废了。但 PPO 不管这套，必须老老实实生成到结束 token 或者撞上 8192 的硬上限。

这有两个直接后果：

第一，**算力白扔**。一条已经死透的 trajectory，平均还要再吐 4000+ 个 token，这部分既拿不到正奖励，又消耗显存和时间。在 8 卡 H20 上每多 100M token 就多几个小时。

第二，更阴险的——**梯度被污染**。失败之后那些 token 也会进 advantage 估计。你想想，模型本来错在第 t* 步，结果 PPO 把后面 5000 个 "错上加错的 token" 也算进负优势里，等于在告诉模型"你后面这 5000 步走得也很烂"。可这些 token 走得烂，根源是 t* 错了，跟这些 token 自己的概率分布没多大关系。这种**误归因压力**会逼着策略去压低那些其实没毛病的 token 概率，最后表现就是熵塌缩、探索空间变窄。

这就是论文反复强调的 "rollout continuation problem"：**失败 trajectory 的延续不仅烧钱，还在反过来教坏模型**。

那为什么之前没人解决？不是没人想，是想了的方案都各有各的麻烦：

- **Process Reward Model（PRM）**：靠人工标每一步对错，能给 step-level 信号，但贵，扩不动；
- **Option-Critic 这类学习式终止**：再训一个 termination 网络，目标函数变复杂，工程上不好上；
- **GRPO / DAPO**：在 advantage 估计上做文章（group normalization、dynamic sampling），但**还是要把每条 trajectory 跑满 $T_{\max}$**，并没有解决"失败之后还得继续生成"这件事本身。

ESPO 想做的就是：**不加新模型、不加新标注、不改 PPO 目标函数，纯粹换一个更聪明的 rollout 收集方式**。

![图 1：ESPO 在 AIME 2024 上以更低 token 成本超过 PPO（DeepSeek-R1-Distill-Qwen-7B）。左图：avg@32 准确率随训练步数的变化，ESPO 更早超过 PPO 并保持领先；右图：相对 PPO 的累计 rollout token 节省曲线。](https://arxiv.org/html/2605.29860v1/x1.png)

*图 1：ESPO 与 PPO 的训练动态对比。左侧准确率曲线显示 ESPO 用更少步数就能反超并稳定领先；右侧 token 节省曲线显示训练越往后省得越多，最终累计省下 20%+ 的 rollout token。*

---

## 🏗️ 方法核心：用 actor 自己的 logit 当"反悔信号"

ESPO 的核心 insight 一句话讲完——

> **一个处于"高 regret + 低 value"状态的策略，恢复概率极低**。

什么叫高 regret？就是当前 sample 出来的 token 与策略本来最想 sample 的 token（argmax）之间，对数概率差异越来越大。模型自己都"不太相信"刚才生成的内容了。再叠加 critic 给出的剩余价值估计很低——这时候继续写下去基本就是浪费。

下面把 4 个关键组件一一拆开。

### 1. 单步 surrogate regret：白嫖 logit 的副产品

定义 step-wise 偏离信号（也叫 regret 值）：

$$g_t = \max_{a \in \mathcal{V}} \log \pi_\theta(a \mid s_t) - \log \pi_\theta(a_t \mid s_t)$$

直观理解：**贪婪 token 的对数概率 减去 实际采样 token 的对数概率**。$g_t$ 一定 $\geq 0$，等于零意味着采样到了 mode 上，越大说明采样越偏离策略最自信的选择。

最关键的是——这个值在采样的时候 **已经算过了**，logit 现成的，零额外计算开销。这是 ESPO 工程上能做得这么轻的根本原因。

### 2. EMA 归一化：训练全程让信号"可比"

$g_t$ 的绝对值在训练过程中会变（前期策略乱，后期策略稳），所以直接拿原始值做阈值不靠谱。论文用 batch 级别的 EMA 算均值方差：

$$\mu_g \leftarrow \alpha_{\text{ema}} \mu_g + (1-\alpha_{\text{ema}}) \overline{g}_{\mathcal{B}}$$

$$\sigma_g^2 \leftarrow \alpha_{\text{ema}} \sigma_g^2 + (1-\alpha_{\text{ema}}) \text{Var}(g_{\mathcal{B}})$$

然后做 clipped 归一化：

$$\tilde{g}_t = \text{clip}\left(\frac{g_t - \mu_g}{\sqrt{\sigma_g^2 + \delta}}, -c, c\right)$$

这里有个设计细节挺重要——**EMA 只在每个训练 batch 边界更新**。也就是说在生成当前 rollout 时用的是上一个 batch 冻住的统计量，避免"未来信息"泄漏到当前的终止决策里。这是个挺干净的因果性处理。

最后再做一次轨迹内的指数平滑：

$$z_t = \alpha_s z_{t-1} + (1-\alpha_s) \tilde{g}_t, \quad z_0 = 0$$

$z_t$ 就是最终的 **累计 regret 值**，作为停止判据。

### 3. Value-gated 终止条件：让 critic 给出"耐心额度"

最关键的判定来了：

$$z_t > \beta \cdot \max(V_\phi(s_t), \varepsilon)$$

这一步设计真的挺漂亮。**critic 给出的 value 越高，容忍的累计 regret 越多**——也就是说模型"前途光明"的时候多给它点犯错额度；模型 value 已经塌了的时候稍微一晃就直接掐。

这同时也回答了一个潜在质疑："那如果模型偶尔 sample 了一个低概率但其实正确的 token 呢？"——只要它处在高 value 状态，单步的高 regret 不会触发终止，因为 $\beta \cdot V_\phi(s_t)$ 足够大。

为了控制整体的截断率，$\beta$ 还有个简单的 PI 控制器：

$$\beta \leftarrow \text{clip}(\beta + \eta_\beta(\hat{\rho}_{\text{stop}} - \tau), \beta_{\min}, \beta_{\max})$$

实际截断率高于目标值就把 $\beta$ 拉大（更难触发），反之拉小。论文的目标截断率设的是 0.25。

### 4. 终端失败惩罚：把截断当吸收态

截断点 $T_{\text{stop}}$ 之后不再生成，并给一个终端失败奖励：

$$r_t = 0 \quad (t \lt T_{\text{stop}}), \qquad r_{T_{\text{stop}}} = r_{\text{fail}}$$

论文里 $r_{\text{fail}} = -1.0$。

这里有个 RL 上的**关键判断**值得说一下：为什么是终端惩罚而不是 per-step 惩罚？

如果你给每一步都加一个负奖励（用来"逼"模型早点结束），实际上会引入一个非平稳的 reward 函数，critic 估值会被搞乱，更糟的是模型可能会学到一个奇怪的捷径——**直接把 logit 分布拉得更尖**（让自己 sample 出来的 token 永远很接近 argmax，从而把 $g_t$ 压低），这反而会让熵塌缩、探索消失。

终端惩罚就完全没这个问题。它在 GAE 里只产生**一个集中的负 TD-error**，通过 $\delta_t + \gamma \lambda \delta_{t+1} + \dots$ 反向传播到失败点之前的几步，定位精准、不污染 reward 函数本身。

### 关键的工程细节：critic warmup

随机初始化的 critic 给出的 value 估计是不可信的——这时候如果直接启用 ESPO 的截断，会因为 value baseline 抖动而触发大量误伤。

论文用了一个 adaptive warmup 阶段：先正常跑 PPO，等 critic loss 收敛（连续 3 步 loss \lt 0.5 或相邻步差异 \lt 0.1）才启用截断；如果训完 10% 总步数 critic 还没收敛，就强制启用，免得无限期 defer ESPO 机制。

整个 rollout 收集的伪代码长这样（论文 Algorithm 1）：

```
输入：策略 π_θ，critic V_φ，终端惩罚 r_fail
1.  z ← 0; t ← 0; done ← False
2.  while t < T_max and not done:
3.    计算 logits ℓ；采样 a_t ~ π_θ(·|s_t)
4.    g_t ← max_a ℓ_a - ℓ_{a_t}                  # 单步 regret
5.    g̃_t ← clip((g_t - μ_g)/√(σ_g² + δ), -c, c)  # 用冻结的 EMA 归一化
6.    z ← α_s · z + (1-α_s) · g̃_t                # 指数平滑
7.    if warmup 完成 and z > β · max(V_φ(s_t), ε):
8.      r_t ← r_fail; done ← True               # 进入吸收失败态
9.    else:
10.     r_t ← 0; t ← t+1                        # 继续解码
11.   end if
12. end while
13. return τ = (s_0, a_0, r_0, …, s_{T_stop}, a_{T_stop}, r_fail)
```

整套机制相当于在原有 PPO 上方包了一层"在线终止规则"。**PPO 的目标函数没动一行**，改的只是 trajectory 怎么收集。这就是论文反复强调的 **ESPO 与 GRPO/DAPO 等 advantage 估计改进正交可叠** 的来源——你完全可以在 DAPO 上再叠一层 ESPO。

---

## 🧪 实验结果：双尺度都在涨，token 全在省

主实验在 DAPO-Math-17k 上训练，benchmark 是 AIME24 / AMC23 / MATH500。模型选 DeepSeek-R1-Distill-Qwen-1.5B 和 7B 两个尺度。8 张 H20 训练，rollout 数 8、global batch 64、推理时 32 次取 Pass@1。

### 主实验：1.5B 和 7B 都比 PPO/DAPO 强

| 模型 | 方法 | AMC23 | AIME24 | MATH500 | Avg Acc | Cumulative Tokens(M) | Avg Tokens |
|------|------|-------|--------|---------|---------|----------------------|------------|
| **1.5B** | Base | 58.28 | 20.31 | 74.81 | 51.13 | - | 5808 |
|         | PPO  | 68.43 | 23.02 | 79.65 | 57.03 | 1069.66 | 4178 |
|         | DAPO | 70.23 | **24.37** | 80.28 | 58.29 | 1223.96 | 4781 |
|         | **ESPO** | **71.87** | 23.87 | **81.53** | **59.09** | **927.96** | **3625** |
| **7B**   | Base | 78.64 | 40.13 | 83.36 | 62.04 | - | 5357 |
|         | PPO  | 82.94 | 45.25 | 85.43 | 71.20 | 1072.40 | 4189 |
|         | DAPO | 83.76 | 45.57 | 85.95 | 71.76 | 1035.01 | 4043 |
|         | **ESPO** | **85.83** | **46.28** | **87.42** | **73.17** | **839.24** | **3278** |

几个值得停一下的点：

**7B 上是真涨。** AMC23 + 2.89 个点，AIME24 + 1.03 个点，MATH500 + 1.99 个点，平均涨 1.97 个点。**这种幅度在数学 RL 后训练上算挺能打了**——你要知道 DAPO 相比 PPO 也只多涨了 0.56 个点。

**token 节省更狠**。7B 上累计 token 从 PPO 的 1072M 降到 839M，省了 22%；DAPO 是 1035M，ESPO 比它再省 19%。**单条 trajectory 平均长度从 4189 降到 3278**，相当于平均每条少写 900 多个 token。

**1.5B 上有个小细节值得说**：AIME24 上 ESPO 23.87% 略低于 DAPO 的 24.37%——但要注意 ESPO 比 DAPO 少烧了 24% 的 token。说实话这个 trade-off 是划算的，毕竟 AMC23 + MATH500 上 ESPO 都赢了 DAPO，平均分还是 ESPO 最高。

### 消融：每个组件都不能省

消融实验在 7B + AIME24 上做，用 avg@32 评估：

| Variant | AIME24 ↑ | Cum Tokens(M) | Avg Tokens ↓ |
|---------|----------|---------------|--------------|
| **(A) Full ESPO** | **46.3** | **839.24** | **3278** |
| (B) w/o warmup | 44.2 | 858.37 | 3353 |
| (C) w/o terminal failure penalty | 43.7 | 901.65 | 3522 |
| (D) Value-only stop（仅靠 V_φ \lt τ） | 44.0 | 1090.05 | 4258 |
| (E) Regret-only stop（仅靠 z_t \gt τ） | 44.8 | 1086.51 | 4244 |
| (F) Random stop（同截断率随机截） | 42.4 | 855.59 | 3342 |

我重点关注三组对比。

**A vs F 这组随机截断对照真的是关键**。把 ESPO 的"哪里截"换成同截断率的随机截断——平均长度几乎一样（3342 vs 3278），但 AIME24 掉了 3.9 个点。**这直接证明了 ESPO 的收益不来自训练更短的序列这个 trivial 解释，而是来自截在哪里**。如果只是随机砍 token 也能提升性能，那论文的故事就站不住了——但事实是不行。

**变体 C 去掉终端惩罚，掉了 2.6 个点而且 token 还涨了**。这印证了前面说的 RL 直觉：终端惩罚提供了一个集中的负 TD 信号，能精准反向传到失败点。把它去掉后，截断变成了"无奖励的中断"，credit assignment 失去精度。

**D 与 E 对比看互补性**：单独用 value 或单独用 regret 都不如组合。Value-only 依赖 critic 的绝对尺度（不同任务/阶段差很多），regret-only 缺少 value 提供的"恢复余地"。这俩信号是真的互补。

### 训练动态：熵不塌缩、误伤率 2.7%

![图 2：ESPO 与 PPO 的训练动态对比（DeepSeek-R1-Distill-Qwen-1.5B）。三张子图分别展示响应长度、actor 熵、误伤率随训练步数的变化。](https://arxiv.org/html/2605.29860v1/x3.png)

*图 2(a)：响应长度变化。ESPO/Original 表示如果不执行截断、模型本应生成的 trajectory 长度；ESPO/Actual 表示实际截断后的长度；PPO/Original 是 PPO 训练中的平均长度。可以看到 ESPO 的"原生长度分布"基本贴着 PPO 走，说明截断没有破坏策略本身的长度分布。*

![图 2(b)：actor entropy 随训练步数的变化。ESPO 不仅没有引发熵塌缩，反而比 PPO 衰减得更慢，相当于探索空间被打开了。](https://arxiv.org/html/2605.29860v1/x4.png)

*图 2(b)：策略熵曲线。这个结果挺反直觉但合理——你可能担心 ESPO 的 logit-gap 信号会"奖励"模型生成更接近 mode 的 token，导致熵塌缩。实际上正好相反：ESPO 移除了失败 trajectory 的误归因梯度（那些 post-failure token 在 PPO 里会被打负优势，逼着策略压低非 mode token 的概率），熵反而衰减得更慢。*

![图 2(c)：False Positive Rate 随训练步数的变化，平均 2.7%。](https://arxiv.org/html/2605.29860v1/x5.png)

*图 2(c)：误伤率（被截断的轨迹中其实本来能答对的占比）。论文报告平均 2.7%——这个数我觉得是可以接受的。每个 batch 平均 97.3% 的截断都是正确的"杀死失败 trajectory"，2.7% 是误杀。考虑到截断带来的 token 节省 + 准确率提升的整体收益，这个错杀率是被收益完全覆盖的。*

### 工程细节：训练超参一眼

| 超参数 | 值 |
|--------|----|
| Failure reward $r_{\text{fail}}$ | -1.0 |
| EMA $\alpha_{\text{EMA}}$ | 0.99 |
| Normalization $\alpha_s$ | 0.9 |
| Initial $\beta$ | 7.0 |
| $\beta$ 调整速率 | 0.1 |
| 目标截断率 | 0.25 |
| 最大 rollout 长度 $T_{\max}$ | 8192 |
| Learning rate | 1e-6 |
| Value $\varepsilon$ | 0.2 |

整个机制只多了 6 个超参（$r_{\text{fail}}$、两个 $\alpha$、$\beta$ 初值与调整、目标截断率、$\varepsilon$），相比 PRM 那种"再训一个模型"的代价低太多了。

---

## 💡 我的判断：思路漂亮，但有几个值得追问的地方

### 真正的亮点

**第一，问题诊断准确**。"rollout continuation problem" 不是新的痛点，但 ESPO 把它具体化了——不是 token 浪费这一面，更阴险的是**梯度污染**。失败之后的 token 在 PPO 里被打上负优势，逼着策略压低那些其实没毛病的 token 的概率，这是论文里反复强调的 misattributed pressure，也是熵塌缩的一个隐藏推手。

**第二，方案极简**。单步 regret = $\max \log \pi - \log \pi_{\text{sample}}$，logit 现成的，零计算开销。$z_t > \beta \cdot V_\phi(s_t)$，就这一个判据。critic warmup 用一个简单的 loss 收敛检测搞定。整个 ESPO 模块加进 verl 这种 RL 框架估计就几百行代码的事。

**第三，正交性强**。ESPO 改的是 rollout 收集方式，PPO 目标函数没动。理论上可以叠在 DAPO/GRPO/GSPO 这些 advantage 估计改进之上——论文也确实展示了相比 DAPO 的进一步提升。

**第四，消融做得扎实**。Variant F 那个随机截断对照实验非常关键——它直接堵死了 reviewer 最可能问的"是不是只是因为训练序列短了所以效果好"的质疑。

### 需要保持警惕的地方

**第一，对"自信但错误"的策略无能为力**。论文 Section 8 自己也承认了：如果模型在错误的推理分支上极其自信（比如 $\pi_\theta(a^* \mid s_t) \approx 1$），那 $g_t = \max \log \pi - \log \pi(a^*) \approx 0$，根本触发不了截断条件。这是 logit-gap 这套信号的根本局限——它检测的是"模型自己不确定"，不是"事实错误"。换句话说，**自信地胡说八道，ESPO 抓不住**。

**第二，目标截断率是个手调超参**。论文是固定到 0.25，并通过 PI 控制 $\beta$ 来维持。但不同任务、不同模型规模、不同训练阶段，最优的截断率显然是不同的——这玩意儿能不能学出来？论文没给。

**第三，target benchmark 还是数学**。AIME24/AMC23/MATH500 都是数学题，奖励是 binary correctness。在 RLHF 偏好对齐、long-horizon agent、tool-use 这些更常见的工业场景下，"失败"的定义会复杂得多——比如 agent 调错了一个 API，但后面可能 retry 救回来。这些场景下 logit-gap 是不是还有效，论文没回答。这也是作者自己点出的 future work。

**第四，2.7% 的误伤率值得再想想**。如果你是在做一个对正确率非常敏感的任务（比如生产环境的代码生成），那 2.7% 的"本可以做对的 trajectory 被错杀"会不会反过来限制能力上限？数学题上由于 reward 稀疏，错杀的代价被失败 trajectory 的清理收益盖过去了。但在更稠密、更结构化的奖励场景下，这笔账要重新算。

### 工程落地建议

如果你正在做 RL 后训练，特别是数学/代码这类 long-horizon reasoning 任务，**ESPO 是非常值得试一试的——成本极低、风险可控、在 verl 这种框架上加一段 rollout 收集的 callback 就行**。

要试的话有几个建议：
1. **critic warmup 不能省**。这是论文里讲到的最大稳定性 trick，连续 3 步 loss \lt 0.5 是个挺保守的判据，可以照搬。
2. **目标截断率从 0.25 起，往下试 0.15 / 0.10**。如果你的 base model 已经比较强（比如 R1 蒸馏过的 7B），失败率本身就低，0.25 可能太激进。
3. **$r_{\text{fail}} = -1.0$ 这个 magic number** 我觉得是能调的。它本质上和 reward scale 强相关——如果你的成功奖励是 +1，那 $r_{\text{fail}} = -1$ 是对称的；但如果成功奖励是 +0.5（部分任务会做归一化），$r_{\text{fail}}$ 也要跟着缩。
4. **叠在 DAPO 上**。论文已经验证了和 DAPO 兼容，DAPO 自身的 dynamic sampling + advantage clipping 在工业场景下也是默认选项。

---

## 📝 写在后面

回到最开始那个问题——"模型已经走错了，还非要让它写完 8192 个 token？"

ESPO 的回答是：**不需要**。而且不仅是省算力的问题，让一条已经死透的 trajectory 继续生成，本质上是在**让错误的归因继续污染你的策略梯度**。把它早点掐了，省下的算力是次要的，**清理出来的更干净的学习信号才是关键**。

整篇论文最打动我的地方是**方法的工程感**——它不是堆叠一个 fancy 的新损失函数或新模型架构，而是诚实地从 RL 训练的 actor-critic 里**已有的信号**（logit、value）里挖出一个截断判据。"我不需要给你新东西，我只是把你已经算过的东西用得更好" ——这种思路在工业界往往比"我又训了一个新模型"要走得远。

接下来值得追的方向：
- 截断率自适应，不依赖人工设定 target；
- 扩展到 multi-turn agent 和 tool-use 场景，那里的"失败定义"是跨步的、非 token 级的；
- 与 DAPO/GSPO 等 advantage 估计改进的更深度融合；
- 对"自信但错误"这一 failure mode 的补充检测机制。

如果你也在做 RL 后训练，特别是被 rollout 长度的算力账压得头大，这篇论文值得花半小时仔细读一遍，方法部分（Section 4）尤其推荐。

---

## 🔗 参考链接

- 论文：[ESPO: Early-Stopping Proximal Policy Optimization (arXiv:2605.29860)](https://arxiv.org/abs/2605.29860)
- 相关工作：DAPO (arXiv:2503.14476)、DeepSeekMath/GRPO (arXiv:2402.03300)、PPO (arXiv:1707.06347)、verl 框架 (arXiv:2409.19256)

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
