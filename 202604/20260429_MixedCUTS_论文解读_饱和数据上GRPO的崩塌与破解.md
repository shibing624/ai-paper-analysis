# Too Correct to Learn：当模型"答得太对"时，GRPO反而学不动了

## 核心摘要

你有没有碰到过这种情况——拿一个能打的 base 模型，在 MATH 这种"经典"数学集上跑 GRPO，结果训了半天，MATH 涨一两个点意思一下，AIME 这种真正有难度的题完全推不动，policy entropy 跌到地板上不再变化？

这篇来自 Tencent AI Lab 和 University of Notre Dame 的工作给了一个挺干净的诊断：**问题不在模型不会做，而在它"做得太对了"**。当一个组里 16 个 rollout 全是正确答案，组内 reward 方差 → 0，标准化之后 advantage 也 → 0，梯度信号直接消失——这就是作者起的名字"saturation-induced collapse"。

解法叫 **Mixed-CUTS**（Mixed Constrained Uniform Top-K Sampling），思路特别朴素：**别动训练目标，动 decoding**。一半 rollout 走标准采样（exploitation），另一半走 CUTS——在 Top-K 候选里做 δ 概率过滤后**均匀采样**（exploration）。两条流合并喂给 GRPO 算 advantage，强行把组内方差撑住。

效果说实话挺打脸 baseline 的：Qwen3-4B 在 AIME25 上 Pass@1 比标准 GRPO 涨 **+15.1%**，maj@16 直接干到 **+23.2%**，甚至超过了 base 模型自己的 Thinking Mode。这告诉我一件事——**饱和的"简单数据"里其实还藏着大量 RL 没榨出来的信号**，前提是你得让模型在它的高置信区域里认真"分头走走"。

---

## 论文信息

- **标题**：Too Correct to Learn: Reinforcement Learning on Saturated Reasoning Data
- **作者**：Zhenwen Liang¹†, Yujun Zhou¹²†, Sidi Lu¹, Xiangliang Zhang², Haitao Mi¹, Dong Yu¹（†Equal contribution）
- **机构**：¹Tencent AI Lab, ²University of Notre Dame
- **arXiv**：[2604.18493](https://arxiv.org/abs/2604.18493)

---

## 问题动机：什么叫"饱和数据上的崩塌"

先把这个现象拆开讲。

GRPO 这套东西大家都熟了——给一个 prompt，sample 出 G 条 trajectory，每条得到一个 reward，然后用**组内均值标准差**做 baseline，算 advantage：

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G) + \epsilon}$$

它最大的优点是不用 critic 网络，省了一大堆参数和工程坑。但代价是——**它强烈依赖组内 reward 有方差**。

问题来了。当你拿 Qwen3-1.7B、Qwen3-4B 这种已经很能打的 base 模型，在 MATH 这种现在已经被打烂的数据集上做 RL，会发生什么？base 模型 Pass@1 已经 70%-82% 了。一个 prompt 采 16 个 rollout，可能 14 个都对，2 个错。组内 reward 方差还行。但训不到 30 个 step，模型就把这种 prompt 全扫平了——16 个全对。

这一刻，分子 $r_i - \text{mean}$ → 0，分母 $\text{std} + \epsilon$ → ε，整个 advantage → 0。**梯度死了**。

但模型并不是不会做更难的题。它只是在 MATH 这种数据上**没动力学了**——因为它"太对了"。

更微妙的是，这种"假训练"还有一个隐藏的坏作用——**正在悄悄把策略推向 mode collapse**。GRPO 对每个 trajectory 用同一个 token-level advantage（$\hat{A}_{i,t} = \hat{A}_i$），在饱和阶段 advantage 虽然趋零但不是严格的零，clipping 之外那点残余信号会持续把 policy 往**已经成功的那条 dominant 路径**上挤。结果就是模型不仅没学到新东西，还把它原本会的"非典型解题路径"逐步剪掉了——这就是为什么后续在 AIME 这种 OOD 上反而会表现不如 base 模型的预期。


> "Mode collapse 不是因为它错，而是因为它对得太彻底（too correct to learn）。"

这跟传统理解里的 mode collapse 不一样。传统的 mode collapse 是 reward 太稀疏（基本全错），策略没正反馈所以塌缩。这里是反过来的——**reward 太稠密（基本全对），策略没负反馈，所以塌缩**。两端都坍，中间才有学习。

我看到这里第一反应是——这个观察其实跟之前 DAPO、SimpleRL 那拨工作里提到的"hard-only 训练"动机是一脉相承的。大家都意识到 saturated easy data 是个问题，但通用做法是**换数据**：要么去 curate AIME/IMO 级别的硬题（DAPO-17K 就是这个路子），要么用 reward shaping 引入难度采样（DIVA-GRPO、DRA-GRPO 都是变种）。

但这条路有个根本的可持续性问题——**当模型越来越强，你能找到的"够难"的数据越来越少，curate 成本指数级上升**。这篇论文换了一个角度：**别去找硬数据，把简单数据里的信号挤出来**。

挤的方式是——动 decoding。

---

## 方法核心：CUTS 和 Mixed-CUTS

![图1：Mixed-CUTS 框架。上半路是 Standard Sampling 的 G/2 条 exploitative trajectory，下半路是 CUTS 的 G/2 条 exploratory trajectory，合并成一个 G 大小的 Combined Group 算 GRPO advantage，最后做 policy update。](https://arxiv.org/html/2604.18493v1/x1.png)

*图1：Mixed-CUTS 整体框架——左侧 Input Query 同时进入两个 batch：Exploitation Batch 走标准采样（Standard Sampling, $\pi_\theta$），Exploration Batch 走 CUTS（先 Top-k 加 δ 阈值过滤，再做均匀采样）。两条流合并到 Combined Group 后，按标准 GRPO 公式 $A_i = (r_i - \text{mean}(r))/\text{std}(r)$ 算 advantage，再做 policy update。结构上对原 GRPO 几乎零侵入——仅仅是 rollout 阶段把一半样本切到 CUTS decoding。*

### CUTS 是什么

CUTS（Constrained Uniform Top-K Sampling）本质是一个 inference-time 的 token-level 操作子。每一步 decode：

1. 拿到模型原始概率分布
2. 取 Top-K 候选（论文用 $K=5$）
3. 砍掉概率小于 $\delta$ 的（$\delta=0.03$，相当于"不太可能"的尾部直接淘汰）
4. 剩下的候选**全部赋予均匀概率**，然后采样
5. 前 $T_{\text{warmup}}=5$ 个 token 不用 CUTS，先用标准 decoding 稳住开头

整个过程**parameter-free**——不动模型权重，不动训练目标，纯改 decoding。

为什么要这么设计？回答两个问题就清楚了。

**第一个问题：为什么不直接用 entropy bonus？**

加 entropy 项是 RL 里最朴素的"鼓励探索"做法，但它有个致命缺陷——**它无差别地惩罚高置信度**。模型在某些 token 上确实就该 99% 输出"="，结果 entropy bonus 把它拍平，整段 reasoning 就开始胡言乱语。这种"diversity"是 incoherent 的——多样性是有了，但语义全废了。

CUTS 的做法是反过来——**多样性只在高置信度区域内部展开**。Top-K 已经过滤掉所有"不太可能"的 token，剩下的 K 个都是模型认为"语义合理"的。在这 K 个里做均匀采样，相当于"模型你别老选概率最大那条路，把第二、第三选择也试试"——但绝不会跳出"语义合理"的边界。

这个设计真的挺巧的。它把"global exploration"（容易破坏语义连贯）替换成了"**local width-first exploration**"（保持局部语义有效），在 GRPO 这种 group-relative 的 setting 里特别贴。

**第二个问题：为什么要 δ 阈值过滤？**

光取 Top-K 还不够。如果某一步分布是 $[0.95, 0.03, 0.005, 0.005, 0.01]$，那 K=5 之后均匀采样，相当于 80% 的概率从尾部低质量 token 里选。这种"采样"会把 reasoning chain 直接采崩。

δ 这个阈值就是把那些"虽然在 Top-K 里但概率太低"的尾部 token 砍掉。实测里 $\delta=0.01$ 太松，AIME25 Pass@1 直接掉到 22.0%（4B 模型 35.0%）；$\delta=0.03$ 是甜点；$\delta=0.05$ 又过滤太多，探索空间不足。

### 一段简化的伪代码

把 CUTS 写出来其实就这么几行：

```python
def cuts_sample(logits, K=5, delta=0.03, step_idx=0, T_warmup=5):
    if step_idx < T_warmup:
        return standard_sample(logits)

    probs = softmax(logits)
    topk_vals, topk_idx = topk(probs, K)
    valid_mask = topk_vals >= delta
    valid_idx = topk_idx[valid_mask]

    if len(valid_idx) == 0:
        return topk_idx[0]
    if len(valid_idx) == 1:
        return valid_idx[0]

    return uniform_sample(valid_idx)
```

整个改动量就是这么大。在原来的 GRPO rollout pipeline 里，把 `decoder.step()` 替换成 `cuts_sample()` 即可。Mixed-CUTS 在 batch 维度上一半走 `standard_sample`、一半走 `cuts_sample`，合并后送进原 GRPO loss 算。**几乎为零的工程成本**。

`T_warmup=5` 这个细节别忽略——前 5 个 token 强制走标准 decoding，是为了"prefix protection"。一段 reasoning 的开头几个 token 决定了整个 chain 的方向（"Let's think step by step"、"First, we have..."这种），如果在开头就引入 uniform 探索，模型可能直接被带偏到不合语义的开局。warmup 把 CUTS 的扰动限制在中后段——前缀稳，后续放开探索。

这种细节在论文里只是顺手提了一句，但工程实现的时候很容易踩坑。如果你直接从第一个 token 就用 CUTS，AIME25 大概率会掉点。

### Mixed-CUTS：为什么要混

如果只用 CUTS（全部 rollout 都走 CUTS），会有什么问题？

理论上——baseline 跑歪了。policy 不再 anchor 在 $\pi_\theta$ 的高概率路径上，advantage 信号全变成"和 uniform 的偏差"，optimization 失去稳定性。这就回到了类似强 entropy regularization 的老坑。

所以作者干脆**两条流都跑**：

- **Exploitation Batch**（$G_{\text{std}} = 8$）：走标准采样，作为 anchor
- **Exploration Batch**（$G_{\text{CUTS}} = 8$）：走 CUTS，注入多样性
- **合并到 Combined Group**（$G = 16$）做 GRPO

这一步特别关键——**advantage 是在合并组上算的，不是分别算两个组各自的 advantage**。这意味着：标准采样的 trajectory 和 CUTS 的 trajectory 之间会被强行拉开 reward 差异（如果 CUTS 找到了被标准采样漏掉的正确解，或者反过来 CUTS 因为探索踩了坑），这种**结构性差异直接撑起了组内方差**。

### 为什么方差不会塌——一个简洁的推导

这是论文里最数学的部分，但其实就是初等概率。两个等大子组合并，组内方差有 law of total variance：

$$\sigma^2_{\text{mixed}} = \frac{1}{2}(\sigma^2_{\text{std}} + \sigma^2_{\text{CUTS}}) + \frac{1}{4}(\mu_{\text{std}} - \mu_{\text{CUTS}})^2$$

第二项是关键——**只要两个子组的均值不同，整体方差就有下界**。

考虑两个最坏情形：

**情形 A：题目"太简单"** —— 标准采样 16 条全对，$\mu_{\text{std}} \to 1$, $\sigma^2_{\text{std}} \to 0$。这是经典的 saturation 死地。但 CUTS 因为强制走"次优 token"，偶尔会拐到一条非典型的解题路径，可能某些步推错了，所以 $\mu_{\text{CUTS}} \lt 1$。代入：

$$\sigma^2_{\text{mixed}} \approx \tfrac{1}{2}\sigma^2_{\text{CUTS}} + \tfrac{1}{4}(1 - \mu_{\text{CUTS}})^2 > 0$$

**情形 B：题目"太难"** —— 标准采样 16 条全错，$\mu_{\text{std}} \to 0$, $\sigma^2_{\text{std}} \to 0$。模型 greedy 死磕同一条错路。CUTS 强制让它考虑 Top-K 里的其他选项，偶尔能"误打误撞"走到正确路径，$\mu_{\text{CUTS}} > 0$。代入：

$$\sigma^2_{\text{mixed}} \approx \tfrac{1}{2}\sigma^2_{\text{CUTS}} + \tfrac{1}{4}\mu_{\text{CUTS}}^2 > 0$$

两端都不会塌。这就是"structural variance preservation"——不是靠噪声给方差，而是靠**两条 decoding 策略行为差异本身**给方差。

我读到这里挺被打动的。它把"为什么这套机制 work"讲得特别干净——不用复杂收敛性证明，就用一个二阶矩分解，把"vanishing advantage"这个核心痛点的解决路径展示得清清楚楚。

---

## 实验结果：硬核数据

主实验在 Qwen3-1.7B 和 Qwen3-4B 上跑（都用 non-thinking mode），训练数据是 MATH，评测覆盖五个 benchmark：MATH、AIME24、AIME25、AMC、GPQA-Diamond。

### Pass@1 / Pass@16 主表

| 模型 | 方法 | MATH P@1 | AIME24 P@1 | AIME25 P@1 | AMC P@1 | GPQA P@1 |
|------|------|---------|-----------|-----------|---------|---------|
| **Qwen3-1.7B** | Base | 70.2 | 12.9 | 11.7 | 39.8 | 32.1 |
|  | GRPO | 83.6 | 29.5 | 22.8 | 59.8 | 34.2 |
|  | Mixed-CUTS | **85.1** | **32.3** | **28.1** | **62.7** | **36.0** |
|  | Δ vs GRPO | +1.5 | +2.8 | **+5.3** | +2.9 | +1.8 |
|  | Thinking Mode | 82.6 | 28.9 | 24.9 | 57.5 | 34.9 |
| **Qwen3-4B** | Base | 82.5 | 24.2 | 21.5 | 61.3 | 45.3 |
|  | GRPO | 86.4 | 32.5 | 26.6 | 68.9 | 48.1 |
|  | Mixed-CUTS | **90.8** | **46.0** | **41.7** | **76.7** | **50.1** |
|  | Δ vs GRPO | +4.4 | **+13.5** | **+15.1** | +7.8 | +2.0 |
|  | Thinking Mode | 89.9 | 54.1 | 42.1 | 73.6 | 52.0 |

几个值得反复看的数字：

**+15.1% AIME25 Pass@1（Qwen3-4B）。** 这是真的硬。AIME25 是当下数学评测里最有区分度的几个 benchmark 之一——base 模型 21.5%，GRPO 只能拉到 26.6%（+5.1），Mixed-CUTS 直接干到 41.7%（+20.2）。**纯改 decoding 的一个改动，把训了 100 step 的 GRPO 多榨出一倍以上的提升**。

**+13.5% AIME24 Pass@1。** 这块比 AIME25 略低（毕竟 AIME24 没那么"新"），但 13.5 个绝对点的提升也是相当反常的。

**Mixed-CUTS 在 4B 上甚至贴近了 Thinking Mode（41.7 vs 42.1）。** 在 1.7B 上更夸张——直接超过 Thinking Mode（28.1 vs 24.9）。这意味着 Mixed-CUTS 在 standard decoding 模式下，几乎拿到了"开 Thinking Mode + 长 inference"的效果，但 inference cost 完全不变。

注意一个细节：**Pass@1 涨幅远大于 Pass@16 涨幅**（4B 上 MATH P@1 +4.4 vs P@16 +0.7）。这是个挺重要的信号——它说明 Mixed-CUTS 不是靠"采样多了总有一条对的"这种随机覆盖，而是**真的把概率质量往正确路径上推**。这一点对工程上落地特别有意义——你部署的时候 inference 成本不需要变，单次采样就能享受这个提升。

### Scalability：模型越大，收益越大

AIME25 的提升从 1.7B 的 +5.3% 跳到 4B 的 +15.1%。这个 scaling 趋势挺有意思——更大的模型，Top-K 里"次优 token"的语义合理性更高（毕竟模型更强），所以 CUTS 的 width-first exploration 能挖出来的 latent reasoning branch 也更多。

我觉得这背后其实是一个挺普适的洞察：**模型越强，它的高置信度区域里"被压抑的合理路径"就越丰富**。greedy decoding 是把这部分财富一刀切掉了，CUTS 把它拿回来。这意味着随着 base 模型越来越强（Qwen3 → Qwen4 → ...），这条路只会越走越宽。

### 训练动态：entropy 不再塌缩

![图2：Qwen3-4B 训练动态对比。左：response length（GRPO 灰色 vs Mixed-CUTS 橙色），Mixed-CUTS 把响应长度从 ~1200 token 拉到 ~1800+ token；中：policy entropy，GRPO 死死卡在 0.20-0.25，Mixed-CUTS 一路涨到 0.40；右：AIME25 reward，GRPO 训了一百多 step 还在 0.25 徘徊，Mixed-CUTS 在 step 30 之后陡然分叉，最终拉到 0.40+。](https://arxiv.org/html/2604.18493v1/x2.png)

*图2：训练动态——三条曲线把"为什么 Mixed-CUTS work"讲得明明白白。GRPO 走的是典型的"急速饱和→停滞"路径：response length 卡在 1200 token，policy entropy 跌到 0.20-0.25 不动，AIME25 reward 100 step 还在 0.25。Mixed-CUTS 三个指标全反过来——entropy 持续涨（说明 advantage 信号没死），response length 变长到 1800+ token（说明模型开始自发地展开更深的推理链），AIME25 reward 在 step 30 之后陡分叉。*

我在做 GRPO 类训练的时候经常遇到 entropy 塌缩的问题，常规打法是加 entropy bonus、调 KL coefficient、或者干脆 early stop。这篇论文给了一个很不一样的做法——**entropy 涨不是目的，advantage 信号活着才是目的**。

CUTS 的妙处在于，它通过"行为差异"维持 advantage 信号活力，**作为副产物**让 entropy 自然涨上去；而不是反过来——直接加 entropy bonus，强行把 entropy 顶起来，结果把 reasoning 搞乱。

response length 从 1200 涨到 1800+ 这个细节也挺值得说。CUTS 强迫模型在某些"次优分支"上展开，这些分支天然会触发 self-correction、回溯、二次验证等"System 2"行为——模型本来内置了这些能力，但 greedy decoding 把它们路径剪掉了。Mixed-CUTS 等于把 RL 当成一个"latent capability awakening"工具用。

### maj@16：consistency 才是真大头

这一组数据我觉得比 Pass@1 更值得讨论。

| 模型 | 方法 | MATH | AIME24 | AIME25 | AMC | GPQA |
|------|------|------|--------|--------|-----|------|
| **Qwen3-4B** | Base | 88.3 | 33.1 | 24.8 | 70.5 | 48.1 |
|  | GRPO | 90.1 | 43.0 | 31.9 | 78.1 | 51.4 |
|  | Mixed-CUTS | **94.0** | **54.9** | **55.1** | **83.0** | **53.1** |
|  | Δ | +3.9 | +11.9 | **+23.2** | +4.9 | +1.7 |
|  | Thinking Mode | 92.4 | 67.5 | 54.0 | 81.2 | 54.8 |

**AIME25 maj@16 +23.2%，从 31.9% 直接到 55.1%，干过了 Thinking Mode 的 54.0%。**

maj@16 的语义跟 Pass@N 不一样——Pass@N 是"采样空间里只要有一个对，就算对"，maj@16 是"采样 16 次，多数投票算一个答案"。一个方法 Pass@16 高但 maj@16 低，说明它**会"偶尔猜对"但不稳定**；maj@16 高才说明**模型对正确答案有真实的高置信度**。

Mixed-CUTS 的 maj@16 提升远大于 Pass@1 提升，这进一步坐实了"它在重塑 policy 概率分布，把概率质量真的搬到了正确路径上"。这一点在生产环境里太重要了——你部署一个 reasoning model，你要的不是"多采几次能蒙到正确答案"，你要的是"它就是稳定地输出正确答案"。

### 跨域泛化：训练的是 math，提升的是 general reasoning

| 训练方法 | MMLU-Pro | SuperGPQA |
|---------|---------|-----------|
| Base Model | 63.80% | 33.05% |
| Standard GRPO | 68.59% | 40.03% |
| Mixed-CUTS | **69.65%** | **41.28%** |
| Δ vs GRPO | +1.06 | +1.25 |

只在 MATH 上训，不做任何额外微调，在 MMLU-Pro 和 SuperGPQA 上 Mixed-CUTS 一致超过 GRPO。这进一步说明 Mixed-CUTS 学到的不是 math overfit，而是更底层的 reasoning skill。

提升幅度不大（1-2 个点），但方向明确，符号一致——足以证明"打破饱和"获得的能力是可以跨域迁移的。

### 边界条件：在难数据上还 work 吗？

这是我读到一半就在想的问题——如果 Mixed-CUTS 是为"saturated easy data"设计的，那在 AIME 级别的 hard data 上是不是就没用了？

作者给了 DAPO-17K（一个 AIME 难度级别的训练集）的对照实验：

| Qwen3-4B 训练在 DAPO-17K | AIME24 | MATH-500 | AIME25 | AMC | GPQA |
|--------------------------|--------|----------|--------|-----|------|
| Standard GRPO | 65.6 / 82.6 | 92.6 / 96.3 | 54.1 / 68.7 | 84.6 / 95.6 | 54.9 / 81.9 |
| Mixed-CUTS | **67.5 / 84.1** | **93.8 / 97.8** | **56.0 / 73.6** | **87.7 / 97.7** | **56.6 / 88.9** |

在 DAPO 上 Mixed-CUTS 仍然全面领先，但**绝对幅度小很多**——AIME25 Pass@1 只 +1.9（vs MATH 训练时的 +15.1）。

这正好印证了核心 thesis：**当数据难度本身已经足够，模型自然会被强迫探索（reward 方差天然存在），CUTS 的边际贡献就小**。但当数据已经被模型刷穿（saturated easy），CUTS 是把模型从"too correct to learn"的死胡同里拉出来的关键。

更工程的意义——**当你不想（也很难）继续 curate AIME 级别的硬数据时，Mixed-CUTS 让你能继续从已有的"廉价数据"里榨出 RL 信号**。这个生态意义可能比单一指标提升更大。

### 超参敏感性

这一块论文做得挺扎实，值得专门看一眼。

**$\delta$（最小概率阈值）的影响（K=5 固定）：**

| $\delta$ | Qwen3-1.7B AIME25 | Qwen3-4B AIME25 |
|----------|---------|---------|
| 0.01 | 22.0 / 44.1 | 35.0 / 64.2 |
| 0.02 | 26.5 / 50.2 | 40.0 / 69.5 |
| **0.03（默认）** | **28.1 / 52.5** | **41.7 / 71.9** |
| 0.04 | 27.2 / 51.4 | 40.6 / 70.8 |
| 0.05 | 26.0 / 49.8 | 39.8 / 70.1 |

$\delta=0.01$ 直接掉了 6-7 个点，因为低质量尾部 token 进了候选集；$\delta \in [0.02, 0.05]$ 都明显比 GRPO 强。

**Top-K 的影响（δ=0.03 固定）：**

| $K$ | Qwen3-1.7B AIME25 | Qwen3-4B AIME25 |
|-----|---------|---------|
| 3 | 25.5 / 48.9 | 38.9 / 68.5 |
| **5（默认）** | **28.1 / 52.5** | **41.7 / 71.9** |
| 7 | 26.8 / 53.4 | 40.2 / 72.8 |
| 9 | 21.5 / 48.2 | 35.5 / 68.2 |

K=7 在 Pass@16 上略高（更宽搜索空间多样性更好），但 K=9 Pass@1 直接崩了 6.2 个点——**过度宽的候选集会引入噪声**。这个 trade-off 跟我做 nucleus sampling 调 top-p 时的体感很像。

整体看，超参 robust 范围还挺宽的（$\delta \in [0.02, 0.05]$, $K \in [3, 7]$ 全比 GRPO 强），不需要逐模型精调。这对落地很友好。

---

## 我的判断

### 这套方法到底厉害在哪

**第一，问题诊断特别到位。** "saturation-induced collapse" 这个概念其实业内很多做 RL 的人都隐约感觉到了，但论文是第一次把它和 vanishing gradient 之间的因果链讲清楚——**不是模型不会做，是 GRPO 没信号了**。这种"重新框架问题"的能力比单纯提一个新方法更值钱。

**第二，方案极简但不简单。** parameter-free decoding 改动，对 GRPO 几乎零侵入——你现有的 RL pipeline 可能就改 10 行代码（rollout 阶段一半样本切到 CUTS decoding）。但理论上有 variance preservation 的硬保证，工程上有 +15.1% 的硬数字。简洁度和有效性这种平衡其实挺难做到的。

**第三，"local uniform exploration" 这个概念可能会被更广泛地用起来。** 加 entropy bonus 这种 global exploration 一直有"破坏语义"的副作用，CUTS 这种把多样性约束在 Top-K 高置信区域内的做法，提供了一个更安全的探索接口。我猜未来在 code generation、agentic planning 这些 trajectory 更长、错一步全错的场景里，这种"width-first exploration"会更受欢迎。

### 哪些地方让我皱眉

**第一，理论分析停在 variance preservation 层面，没有收敛性证明。** Limitation 里作者自己也承认了——"the mixed behavior policy induced by CUTS introduces a controlled deviation from strict on-policy sampling"，这个 off-policy 程度对长 horizon 收敛性有什么影响，论文没给。Empirically 看 KL coefficient $1\times 10^{-3}$ 加上 PPO clipping 是稳的，但理论上它仍然是个 open question。

**第二，实验只在 Qwen3 1.7B/4B 上做了。** 没有 7B、14B、72B 这种更大模型的验证。基于 scaling 趋势（1.7B +5.3 → 4B +15.1）我倾向于相信更大模型上收益会继续增长，但这毕竟是猜测。

**第三，跟同期工业界其他"打破 saturation"方案的对比缺失。** 比如 DAPO 本身有个 dynamic sampling 机制丢弃零方差组，DIVA-GRPO 用难度加权，COPO 用 batch-level global advantage——这些方案跟 Mixed-CUTS 的对比实验没做。我猜部分原因是这几个方法实现细节不一致，但这块如果能补上会更有说服力。

**第四，Figure 2 的 maj@16 panel 在我下载的图里没显示完整**（论文 caption 提到右侧应该是 maj@16 consistency 曲线，但实际图我看到的是三个 panel 而不是四个）——这可能是 HTML 渲染的小 bug，不影响结论。

### 工程启发

如果你也在做基于 GRPO/类似算法的 reasoning RL，这个工作给了几个直接可借鉴的点：

**一是诊断你是不是踩到了 saturation 死地**——盯着 policy entropy 和 intra-group reward variance。如果 entropy 在前 30 step 就跌到 0.2 以下不动，且 reward variance → 0，你的 advantage 信号已经死了。继续训没意义。

**二是 CUTS 这种"local uniform exploration"几乎零成本可复用**——不需要重训模型、不需要改训练目标，rollout 阶段加个 decoding wrapper 就行。论文超参（K=5, δ=0.03, T_warmup=5, G_std/G_cuts=8/8）可以直接当合理初值。

**三是别迷信"换硬数据"是唯一出路**——saturated easy data 里仍然有大量 RL 没榨出来的信号，前提是有合适的探索机制。这点对资源受限的团队特别友好——AIME 级别的高质量数据极贵且少，但 MATH 这种"经典数据"反而满地都是。

**四是 8/8 的混合比例不是拍脑袋的**——它本质上是 anchor 和 explorer 的最优 trade-off。如果 CUTS 比例太高（比如 12/4），会丢掉 baseline 的稳定性，policy 跑歪；如果太低（比如 4/12），exploration batch 不够多，方差恢复力不足。8/8 在 G=16 这个量级上是个比较 robust 的甜点。我猜如果你 G 取更大（32 或 64），这个比例可能可以适当往 explorer 倾斜。

### 跟同类工作的关系

这块顺便聊一下定位。最近几个月围绕"GRPO vanishing advantage"出了一波工作：

- **DAPO**（开源 RL 系统）通过 dynamic sampling 在 batch 层面**丢弃零方差的组**，相当于"治标"——不让坏样本浪费训练 step；
- **DRA-GRPO** 用 Submodular Mutual Information 给 reward 加"diversity 调整项"，鼓励 semantically 不同的 trajectory；
- **DIVA-GRPO** 通过 difficulty-adaptive variant 主动制造组内难度差异；
- **COPO**（Consistency-Aware Policy Optimization）引入 batch-level global advantage 保证信号不全消失；

Mixed-CUTS 跟这几个方向都不一样——**它不动 advantage 计算，不动 batch 选择，不动 reward shaping**，纯在 decoding 这一层注入"行为差异"。优点是工程改动最小；缺点是它绑定了一个特定的 decoding strategy（Top-K uniform），对其他 RL 算法（不是 group-relative 的）可能不直接适用。

我觉得这几条路本质上是互补的——**Mixed-CUTS 解决 sampling-level 的 diversity，DAPO 解决 batch-level 的 sample selection，COPO 解决 advantage-level 的 信号融合**。未来很可能会看到把这几条路缝在一起的"集大成"系统。

### 一句话定位

**这是一篇"小改动大收益"的工程论文，问题诊断比方法本身更值钱。**

如果你只看一个创新点——把"diversity 必须从结构性差异中来，不能从噪声中来"这件事讲清楚，已经值回票价了。这个原则我猜未来会被更多 RL post-training 工作借鉴。

如果你在做 reasoning model 的 RL 训练，强烈建议把 Mixed-CUTS 加进你的 baseline 对比里——它可能让你重新评估"我到底需不需要花那么多钱去 curate hard data"。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
