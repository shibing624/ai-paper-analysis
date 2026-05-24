# 用策略提示把模型推出舒适区：NudgeRL 让 8 个 rollout 跑赢 64 个

> arXiv: [2605.15726](https://arxiv.org/abs/2605.15726) ｜ Code: [tally0818/NudgeRL](https://github.com/tally0818/NudgeRL)
> 作者：Chanuk Lee, Sangwoo Park, Minki Kang, Sung Ju Hwang
> 提交：2026-05-15（NeurIPS 2026 投稿模板）

---

## 一、先说结论：为什么这篇值得看

做 RLVR 的人最近应该都有这种感觉——GRPO 已经被卷成一个"加 rollout 就能涨点"的力大砖飞游戏。8 不行就 16，16 不行就 32，64，128。算力账单蹭蹭往上窜，结果还经常出现一个让人有点尴尬的现象：

**rollout 加到 64，反而比 32 还低**。

这篇 NudgeRL 给出的实验数据，把这件事摆得非常明白：在 Qwen3-4B-Instruct 上，GRPO 在 N=32 时平均 pass@1 是 0.487，加到 N=64 直接掉到 0.451。Olmo3-7B-Instruct-SFT 上更夸张，从 0.281 直接崩到 0.160——基本等于训练失败。

NudgeRL 怎么干的？给每个 rollout **塞一段轻量级的策略提示**——不是答案，不是中间步骤，就是一句话级别的解题方向（比如"用柯西函数方程"、"用 shoelace 公式"），然后用 8 个 rollout 跑赢 GRPO 的 32/64 rollout。

8 倍 rollout 预算的差距，被一段几十 token 的策略提示给抹平了。

说实话第一次看到这个对比表的时候我的反应是："不会是 GRPO baseline 没调好吧？" 但仔细看了消融、训练曲线和 case study 之后，我觉得这篇工作真正打在了 RLVR 的一个软肋上——**所谓"探索"，在 GRPO 这种范式里从来就不是真的探索，只是在同一片小山头上反复采样**。

下面慢慢拆。

---

## 二、问题：RLVR 的探索瓶颈到底是什么

### "你只能在已经采到的轨迹上变强"

RLVR 的一个朴素事实是：**policy 只能从被采样到的 trajectory 里学**。没采到的，再正确也学不到。

那为什么不能直接多采几条？这就是 BroRL 那篇文章给出的一个挺漂亮的分析。它把 RLVR 一步更新的期望性能增量拆成了这样一个表达式：

$$\Delta Q_{\text{pos}} = \frac{\eta}{N}\big[(1-S_R)Q_{\text{neg}}A_2 + S_R Q_{\text{pos}}B_2 + S_R(Q_{\text{pos}}U_{\text{neg},2} - Q_{\text{neg}}U_{\text{pos},2})\big]$$

公式不用细抠，看最后一项 $-S_R Q_{\text{neg}} U_{\text{pos},2}$ 就够了：

- $Q_{\text{neg}}$ 是错误 token 的总概率质量，在大多数推理任务里**远远大于** $Q_{\text{pos}}$
- $U_{\text{pos},2}$ 是**没被采到的"正确" token** 的二阶矩

把这两项乘起来再加个负号，结果就是：**模型本来有能力生成、但这次 rollout 没采到的那些正确轨迹，会反过来当作惩罚项压制学习**。

说人话就是——只要模型脑子里有"另一条正确的路"但这次没采到，每次更新都在朝错的方向偏。

### 那加 N 不就行了吗？

理论上，N 加大确实能减少"没采到的正确轨迹"。BroRL 给出的衰减速率是：

$$\sum_i p_i^2 (1 - p_i)^N$$

对每个 token，随 N 单调下降。听起来很美好。但有个坑：**长尾正确轨迹的 $p_i$ 太小，$(1-p_i)^N$ 衰减得极其慢**。你想覆盖一个 $p_i = 10^{-3}$ 的轨迹，期望需要 1000 个 rollout 才能采到一次。

这就是为什么 GRPO 加到 32、64 之后开始崩——你并没有在"探索新东西"，你只是在主导模式上反复采样，让 group 内方差不断收缩，梯度信号越来越同质化。

> 这块我看完之后挺有共鸣的。之前在 RLVR 项目里调过 group size，经验上 32 几乎就是天花板，64 一上来训练曲线就开始抖。当时一直归结为"超参没调好"，现在回头看，可能本质问题就在这——**N 加大只能压低你已经熟悉的模式的方差，没法把你不熟悉的模式拉出来**。

### 现有解法的两条路，都不太行

业界已有的尝试基本分两派：

**路径一：改优化目标**。DAPO 的 decoupled clipping、各种熵正则、KL warm-up。这些方法让分布层面更"散"，但有个致命问题——**它们不知道往哪散**。你说不清模型熵高是因为发现了一个新策略，还是单纯在乱说话。

**路径二：注入特权信息（oracle）**。POPE、BREAD、SAGE、RLTF 这一类，往 prompt 里塞 oracle 解的前缀，等于直接告诉模型"前面这几步对了"。问题也很明显：

1. **oracle 解很贵**——要么人工标，要么用更强的模型蒸馏，规模上不去
2. **思路被锁死**——模型沿着 oracle 的方向走，反而探索不到别的解法

NudgeRL 想做的事情，是在这两条路之间挖出第三条：**不给答案，只给方向；不依赖 oracle，但又能引导探索**。

---

## 三、NudgeRL 的核心：Strategy Nudging

![图1：Strategy Nudging 概念图。(a) Naive Sampling（GRPO）容易塌缩到主导模式——三个 rollout 中两个都用了策略 A；(b) Strategy Nudging 给每个 rollout 分配不同的策略提示（A/B/C），强迫模型走不同的推理路径；(c) 统计上，使用 Strategy Nudging 后，单题产生的不同推理结构数量（用 GPT-4o-mini 聚类）显著高于 Naive Sampling——后者超过 75% 的题目只产生 1 种推理模式，而前者会均匀分布到 2-5 种。](https://arxiv.org/html/2605.15726v1/x1.png)

### 一句话总结

> 不要让模型自己决定走哪条路。**给它一段几十 token 的策略关键词，强迫它走那条路**。然后多采几次，覆盖不同的策略。

具体怎么做：

对每个题目 $x_0$，离线用一个轻量 LLM（gpt-4o-mini）生成一个**策略池** $\mathcal{C}(x_0) = \{c_1, ..., c_M\}$，每个 $c_i$ 就是一个简短的"解题方向"提示。论文里 M=2，一题就两个候选策略。

每次 rollout 时，**独立**从池子里随机抽一个策略 $c^{(i)}$，再以概率 $p_{\text{drop}}$ 把它丢掉（用空 context），拼到原 prompt 后面：

$$z^{(i)} = \begin{cases} c^{(i)}, & b^{(i)} = 1 \\ \emptyset, & b^{(i)} = 0 \end{cases}, \quad b^{(i)} \sim \text{Bernoulli}(1 - p_{\text{drop}})$$

然后从 $\pi_\theta(\cdot | x_0, z^{(i)})$ 采样。

### 三个关键设计

**1. 每个 rollout 单独分配一个策略，而不是把所有策略堆在一个 prompt 里**

这点挺重要的。如果你把"可以用方法 A、B、C"全塞到 prompt 里，模型大概率会选它最熟悉的那个，塌缩回主导模式。但如果**强制**每个 rollout 只看到一个策略，那它就没得选——必须走那条。

我自己之前做工具调用 Agent 的时候踩过类似坑，"提供多个工具选项"和"一次只暴露一个工具"是完全不同的 prompt 工程效果。这篇的设计直觉跟那个一脉相承。

**2. 策略 dropout——保留与基础 policy 的兼容性**

如果训练时全程喂带策略提示的 prompt，推理时没提示直接给原题，模型可能就懵了。所以训练时以 $p_{\text{drop}} = 0.5$ 的概率丢掉策略，让一半 rollout 还是用原 prompt——既能探索，又能保持"无提示"场景下的能力。

**3. 策略池不需要正确性验证**

这是 NudgeRL 和 POPE 这类 oracle 方法最大的区别。gpt-4o-mini 生成的策略可能根本就是错的——"用对称性"对一道不对称的题——但**没关系**，反正下游 GRPO 会通过 reward 信号自己淘汰。

> 我特别喜欢这个设计哲学。oracle 方法骨子里是"我比模型聪明，我告诉它怎么做"，而 NudgeRL 是"我只负责把它从舒适区推出去，对错让 reward 自己判"。前者依赖一个比目标模型更强的 teacher，后者只需要一个更便宜的 helper。

### 数学动机：把 rare trajectory 的采样代价拉下来

论文里给了一个挺直白的动机表达式。要从 $\pi$ 中至少采到一次某条 rare trajectory $y$，期望需要的 rollout 数是：

$$\mathbb{E}[\#\text{rollouts}] = \frac{1}{\pi(y | x_0)}$$

如果加了 context $c$ 之后，$\pi(y | x_0, c) \gg \pi(y | x_0)$，那期望采样数就直接除了一个倍数。

也就是说——**context 不需要直接给答案，只要能把分布"挪一挪"，让稀有轨迹的概率从 $10^{-3}$ 提到 $10^{-1}$，期望采样数就从 1000 降到 10**。

这是 NudgeRL 能用 8 个 rollout 跑赢 GRPO-64 的核心数学解释。

---

## 四、Inter-Intra Group Advantage：Strategy Nudging 带来的新麻烦

![图2：(a) Inter-Intra Group Advantage——把 rollout 按策略分组，既比较组内（intra）轨迹质量，也比较组间（inter）策略可靠性；(b) Self-distillation——RL loss 在 context-conditioned policy 上做，distillation loss 把发现的好轨迹蒸回 base policy，保证推理时无 context 也能用。](https://arxiv.org/html/2605.15726v1/x2.png)

Strategy Nudging 听起来很美，但落地的时候会撞上一个尴尬的问题：

**GRPO 是按 group 算 advantage 的——同一个 prompt 下采的 N 个 rollout，谁高于均值谁就是正样本**。但 Strategy Nudging 一搞，N 个 rollout 来自不同的 context，相当于来自不同的 prompt 分布。你直接用一个 group baseline 去算，**reward 的变化就既混杂了轨迹本身的质量，也混杂了策略本身的得分倾向**。

举个例子：策略 A 是"用 shoelace 公式"，策略 B 是"用对称假设"。如果这道题只有 shoelace 能解，那 group 里所有用 A 的 rollout 都会得到正 reward，所有用 B 的都得 0。这时候你给 A 的所有轨迹都打高 advantage、给 B 的都打低 advantage——但**这其实没传递任何细粒度信号**，A 组里有的解得漂亮有的解得拖沓，B 组里也可能有的逻辑很清楚只是策略选错了。

### 解法：拆成 intra + inter 两个信号

NudgeRL 引入了一个叫 **Inter-Intra Group Advantage** 的设计。先按 context 分组：

$$\mathcal{G}(x_0) = \text{Unique}(\{z^{(i)}\}), \quad I_g = \{i : z^{(i)} = g\}$$

然后算两个 baseline——每个 context 内的均值 $\bar{r}_g$，以及全局均值 $\bar{r}$。最终 advantage：

$$A_i = (r_i - \bar{r}_{z^{(i)}}) + \lambda \cdot (\bar{r}_{z^{(i)}} - \bar{r})$$

拆开看就是：

- **Intra**: $r_i - \bar{r}_{z^{(i)}}$ —— "在用同一个策略的兄弟里，你算靠谱的吗？"
- **Inter**: $\bar{r}_{z^{(i)}} - \bar{r}$ —— "你用的这个策略，比平均策略更好吗？"

$\lambda$ 是平衡两者的超参。论文证了一个挺有意思的性质：当 $\lambda \in [0, 2]$ 时，**reward 高的轨迹一定 advantage 高**——也就是说不管怎么调 $\lambda$，都不会出现"reward 高反而被压"的反向激励。

更细的：

- $\lambda < 1$：偏 exploration，鼓励"reward 不那么高的策略组里冒尖的"
- $\lambda > 1$：偏 exploitation，让"已经被验证有效的策略组里的好样本"得到更强信号
- $\lambda = 1$：中性

消融里 $\lambda = 1.1$ 最佳——也就是稍微偏 exploitation 一点。这其实蛮符合直觉的：Strategy Nudging 在采样阶段已经把 exploration 做掉了，**学习阶段反而该更注重把已经发现的好策略压实**。

---

## 五、Distillation：让训练时的 context 不污染推理时的 policy

还有个坑：训练时 policy 看的是 $x_1 = (x_0, z)$，推理时只能看 $x_0$。如果直接用 GRPO 那套，policy $\pi_\theta(\cdot | x_1)$ 学会了"看到提示就用 shoelace"，但 $\pi_\theta(\cdot | x_0)$ 可能根本没动。

NudgeRL 加了一个 advantage-weighted distillation 项：

$$\mathcal{L}_{\text{Distill}}(\theta) = -\mathbb{E}_{y \sim \pi_\theta(\cdot | x_1)} \big[\hat{A} \log \pi_\theta(y | x_0)\big]$$

意思是：**用 context 采到的轨迹 $y$，按它的 advantage 加权，去更新 base policy $\pi_\theta(\cdot | x_0)$ 上对 $y$ 的对数似然**。说人话——"在 context 帮助下发现的好套路，把它蒸回到没有 context 的版本里"。

注意这跟 behavior cloning 不一样——它**按 advantage 加权**，所以只有真正得高分的轨迹才会被强化进 base policy。Advantage 是负的（reward 低）的轨迹，其实是在被推开。

总目标：

$$\mathcal{L}_{\text{NudgeRL}} = \mathcal{L}_{\text{RL}} + \lambda_{\text{distill}} \cdot \mathcal{L}_{\text{Distill}}$$

RL 项在 context-conditioned policy 上跑（用 Inter-Intra advantage），distill 项把发现的好轨迹拉回 base policy。两条腿走路。

---

## 六、实验：8 个 rollout 跑赢 64 个

### 主实验

设置：

- **Base 模型**：Qwen3-4B-Instruct-2507 和 Olmo3-7B-Instruct-SFT
- **训练集**：DAPO-17k-Processed
- **评测**：AIME24 / AIME25 / AMC23 / MATH500 (level-5) / Apex Shortlist
- **指标**：pass@1，128 rollout 估计
- **Baseline**：GRPO with N ∈ {8, 16, 32, 64}，以及 POPE（oracle 前缀）

| 模型 | 方法 | N | AIME24 | AIME25 | AMC23 | MATH500 | APEX | **均值** |
|------|------|---|--------|--------|-------|---------|------|---------|
| **Qwen3-4B** | Base | -- | 0.374 | 0.352 | 0.653 | 0.592 | 0.036 | 0.402 |
| | GRPO | 8 | 0.444 | 0.367 | 0.749 | 0.668 | 0.040 | 0.454 |
| | GRPO | 16 | 0.454 | 0.355 | 0.840 | 0.655 | 0.045 | 0.470 |
| | GRPO | 32 | 0.451 | 0.370 | **0.881** | **0.674** | **0.058** | 0.487 |
| | GRPO | 64 | 0.415 | 0.324 | 0.848 | 0.641 | 0.027 | 0.451 |
| | POPE | 8 | 0.460 | 0.337 | 0.838 | 0.652 | 0.048 | 0.467 |
| | **NudgeRL** | **8** | **0.482** | **0.393** | 0.857 | 0.660 | 0.053 | **0.489** |
| **Olmo3-7B** | Base | -- | 0.134 | 0.118 | 0.467 | 0.384 | 0.021 | 0.225 |
| | GRPO | 8 | 0.187 | 0.159 | 0.537 | 0.434 | 0.025 | 0.268 |
| | GRPO | 32 | **0.195** | 0.176 | 0.553 | 0.459 | 0.024 | 0.281 |
| | GRPO | 64 | 0.081 | 0.053 | 0.349 | 0.291 | **0.027** | 0.160 |
| | POPE | 8 | 0.186 | 0.169 | 0.558 | 0.460 | 0.023 | 0.279 |
| | **NudgeRL** | **8** | 0.190 | **0.179** | **0.563** | **0.468** | 0.025 | **0.285** |

几个观察值得展开说说：

**1. GRPO 在 N=64 时崩了**。Qwen3-4B 从 0.487 掉到 0.451，Olmo3-7B 直接腰斩到 0.160。这其实是个挺关键的反证——如果"加 rollout 就能涨"是对的，那为什么 64 反而比 32 差？论文的解释是：N 越大，group baseline 估计越精确，正负样本的 advantage 越接近 0，**梯度信号反而被稀释**。Olmo3 这种基础能力较弱的模型尤其敏感。

**2. NudgeRL 8 rollout = GRPO 32 rollout**。Qwen3 上 NudgeRL 平均 0.489，GRPO-32 是 0.487。算力账：GRPO-32 训练时每步采 32×prompt 个 token，NudgeRL 是 8×prompt——**直接 4 倍效率**。如果你考虑到 GRPO 在 32 已经是它的天花板（再加就崩），那 NudgeRL 实际上是用更少算力突破了 GRPO 的上限。

**3. NudgeRL > POPE（oracle 引导）**。POPE 用 DeepSeek Reasoner v3.2 生成 oracle 解前缀，理论上应该是个非常强的 baseline。但 NudgeRL 不依赖 oracle 解，**只用 gpt-4o-mini 生成几十个 token 的策略关键词**，反而在两个 base 模型上都超过了。

> 这个结果说实话挺反直觉的。oracle 给的是"正确答案的开头"，NudgeRL 给的是"可能不对的方向词"，后者却更好。论文的解释是：oracle 的方向太窄，所有 rollout 都被锁在一个解法上；而 NudgeRL 的策略多样，反而让模型见过更多解题模式。
>
> 这个解释合理但也有一个潜在的水分——POPE 这一行只跑了 N=8，没跟 NudgeRL 一样做 rollout scaling 的对比。如果 POPE 加到 N=32 是不是也能逼近 NudgeRL？论文里没给。不过考虑到 oracle 生成本身就贵，scaling POPE 的算力账其实更不划算，所以这个比较从工程角度还是站得住的。

### 训练动态：sample efficiency 的真实差距

![图3：AIME24+AIME25 上的训练曲线对比（Qwen3-4B）。NudgeRL（红色）在 200 步左右就突破了 0.42 pass@1，并且后期保持稳定；GRPO-32（青色）训练后期持续下降；GRPO-16（蓝色）和 GRPO-8（紫色）卡在 0.40 附近上不去。NudgeRL 用 8 rollout 跑出了所有 GRPO 变体里最稳定也最高的曲线。](https://arxiv.org/html/2605.15726v1/x4.png)

这张图比主表更直观——**NudgeRL 不光最终性能高，收敛也快**。

我尤其想吐槽 GRPO-32 那条青色曲线，前期上升得最快（毕竟 32 rollout 信号确实丰富），但 200 步后开始持续下降，最后比 GRPO-16 还低。这个现象论文虽然没明说，但我猜跟 reward hacking 和 mode collapse 有关——N 越大，policy 越容易过拟合到一小撮主导模式，长尾能力反而退化。

NudgeRL 那条红线是另一回事。200 步突破 0.42 之后基本就稳了，没有 GRPO 那种"先涨后跌"的曲线。我觉得这是 Strategy Nudging 在每个 step 都强行注入多样性的副产品——不让 policy 有机会塌缩。

---

## 七、案例研究：为什么 NudgeRL 能做对 GRPO 全错的题

![图4：一道 AIME25 题目上的策略分布对比。左边 GRPO 训练后的模型 32/32 全部失败——主要走了 coordinate geometry（坐标几何，13/32），还有 trigonometry、symmetry assumption、area decomposition 等无效策略，所有解都因思路冗长而被截断。右边 NudgeRL 训练后的模型 6/32 成功——也大量用 coordinate geometry，但显著提升了 shoelace formula（鞋带公式）的采样频率（10 次），其中 6 次成功解出。](https://arxiv.org/html/2605.15726v1/x6.png)

这个 case study 我看完印象很深。

同一道 AIME25 题，GRPO 训练的模型采 32 次**全错**。NudgeRL 训练的模型采 32 次对了 6 个。

差别在哪？看那个橙色块——**shoelace formula（鞋带公式）**。GRPO 训练后的模型 32 次里只用了 1 次这个策略，但 NudgeRL 训练后的模型用了 10 次，其中 6 次成功。

这个结果有两层含义：

**1. Strategy Nudging 真的"教会"了模型走原来不走的路**。Shoelace formula 在 base policy 里大概是一个 $p \approx 0.03$ 的长尾策略，GRPO-32 偶尔能采到，但 group 里只有 1 个正例没法形成强梯度信号，下次更新基本就把它忘了。NudgeRL 在策略池里塞了"shoelace formula"，强行让 1/4 的 rollout 走这条，**把长尾路径的采样频率从 3% 拉到了 25%**，gradient 信号足够强，policy 才学到。

**2. Inter-Intra advantage 起到了"识别并放大有效策略"的作用**。注意，NudgeRL 训练后的模型最常用的还是 coordinate geometry（15 次，超过 GRPO 的 13 次）。这说明 NudgeRL 没有**强迫**模型用 shoelace，只是把它**暴露**给了模型，让模型自己通过 reward 信号判断哪个策略好用。这是 NudgeRL 比硬式 oracle 注入更聪明的地方。

> 这个 case 也回答了我一开始的一个疑问——既然策略池只有 2 个策略，会不会把模型的输出锁死在这两个上？事实证明不会。Strategy Nudging 是"打开一扇窗"，不是"砌一堵墙"。Distillation 那一项保证了 context-free 推理时模型还能用它自己想出的策略。

---

## 八、消融：每个组件都有用

论文做了一组挺扎实的消融，挑几个有意思的说：

### Context dropout：0.5 是甜点

| $p_{\text{drop}}$ | 0.0 | 0.25 | **0.5** | 0.75 |
|------|------|------|------|------|
| Avg pass@1 | 0.537 | 0.561 | **0.598** | 0.566 |

$p_{\text{drop}} = 0$（永远带策略）和 $p_{\text{drop}} = 0.75$（绝大多数时候不带）都比 0.5 差。论文的解释是：**dropout 的作用不光是 regularization，更是稳定 group 统计**——如果全程带 context，那 group baseline 估的就是"带 context 的均值"，跟推理时无 context 的分布差得太远；反过来如果几乎不带 context，那就退化成普通 GRPO 了。

### Hint sampling：随机 > top-ranked

| Sampling | Random | Top-ranked |
|----------|--------|-----------|
| Avg pass@1 | **0.598** | 0.552 |

这个结果有点反直觉。"Top-ranked"是先生成 5 个候选策略、用 oracle 评测挑出 pass@16 涨幅最大的两个。听起来应该比随机选两个强吧？

但**随机更好**。论文的解释是：top-ranked 选出来的策略往往集中在某一类思路上，覆盖度反而下降；随机选则保证了策略多样性。这跟"在 RLVR 里 exploration diversity 比 exploration efficacy 更重要"这个核心论点是一致的。

> 这条消融对工程落地非常重要——**你不需要费劲挑最优策略，随机两个 LLM 生成的方向词就行**。也就是说策略池构建几乎零成本。

### Distillation 系数

| $\lambda_{\text{distill}}$ | 0 | **0.1** | 0.5 |
|----|----|----|----|
| Avg pass@1 | 0.560 | **0.598** | 0.536 |

$\lambda_{\text{distill}} = 0$（关掉 distillation）掉到 0.560，证明 distill 那一项是必要的。但开太大（0.5）也不行——会过度约束 policy，让它过拟合 sampled trajectory。0.1 是甜点。

### NudgeRL vs DAPO 的 $\epsilon_{\text{high}}$ scaling

![图5：与 DAPO 的 decoupled clipping 对比。GRPO 在 $\epsilon_{\text{high}} \in [0.20, 0.36]$ 范围内大致稳定在 0.555-0.566，但在 0.40 时崩到 0.541。NudgeRL（$\epsilon_{\text{high}}=0.20$）维持 0.598，全程压制 GRPO 所有 scaling 设置。](https://arxiv.org/html/2605.15726v1/x12.png)

这张图是我个人觉得 NudgeRL 最有杀伤力的实验之一。

DAPO 的 decoupled clipping 是目前 RLVR 圈很流行的"放大正样本梯度"的技巧，核心思路是把 PPO clip 的上界（$\epsilon_{\text{high}}$）调大，让 rare 但 reward 高的 rollout 能产生更强的策略更新。**这其实跟 NudgeRL 是在解决同一个问题——怎么把长尾正样本的信号利用起来**。

但两者的方法论完全不同：

- DAPO：调大 $\epsilon_{\text{high}}$，让"看到的正样本"产生更大更新（**update magnitude**）
- NudgeRL：用 Strategy Nudging 让模型"能看到更多正样本"（**exploration coverage**）

实验结果——**NudgeRL 0.598，GRPO 在所有 $\epsilon_{\text{high}}$ 设置上的最高分 0.566**。而且 GRPO 在 $\epsilon_{\text{high}} = 0.40$ 时崩到 0.541，说明"无脑放大梯度"是有上限的。

这个对比非常清楚地说明了一个 takeaway：**在 RLVR 里，提升 update magnitude 的边际效益远小于提升 exploration coverage**。能看到的东西没变多，把已有信号放大再多，也只是在原地踏步。

---

## 九、批判性审视：哪些地方值得保留怀疑

### 1. 策略池构建的 LLM 依赖

NudgeRL 自己说"不依赖 oracle"，但其实**它依赖一个能生成合理策略关键词的 LLM**（gpt-4o-mini）。对于数学题这种领域知识结构良好的场景，gpt-4o-mini 生成"shoelace formula"、"Cauchy's functional equation"是没问题的。但如果换到代码、工具使用、Agent 多步决策这种场景，策略关键词的语义粒度怎么设计？这是一个开放问题。

论文里其实也承认了——limitations 里写了 "context 是 offline 生成、与训练模型独立的，随着 policy 演进可能不再有效"，并提出 future work 是 "model-adaptive context generation"。

### 2. 主表里 GRPO-64 崩盘可能放大了对比效果

Qwen3 上 GRPO-32 已经做到 0.487，跟 NudgeRL-8 的 0.489 几乎打平。如果不看 GRPO-64 那行（0.451），主表的"NudgeRL 8x 效率"叙事会弱很多。

但反过来想，**GRPO-64 崩盘本身也是 NudgeRL 论证的一部分**——它在说"加 rollout 不是无限有效的，到了某个点就会反噬"。从这个角度看，把 64 列出来确实是必要的负面对照。

### 3. 评测集偏数学，泛化性待验证

5 个 benchmark 全是数学。NudgeRL 论文目前只能说 "在数学推理上有效"，对于 code、tool use、multi-step Agent 任务，效果尚未验证。

> 我个人猜测 code 任务上 NudgeRL 会更难做，因为 code 的"策略"很难用关键词表达——"用动态规划"和"用 BFS" 这种粒度的提示对 LLM 的引导作用有限，需要更结构化的 prompt 设计。但这只是猜测。

### 4. 跟 ProRL、Self-Rewarding 这类 long-context RLVR 的关系

最近 ProRL（NVIDIA）那条线提出"训练 step 加大 + 大 $\epsilon_{\text{high}}$"也能解决 RLVR 探索瓶颈。NudgeRL 在 $\epsilon_{\text{high}}$ scaling 那个实验里和 DAPO 比了，但**没有跟 ProRL 这种"训练时长 + clip 扩展"的组合方案直接对比**。这一块的对照工作算是个缺口。

---

## 十、工程启发：哪些可以直接抄

如果你也在做 RLVR，NudgeRL 里有几个东西我觉得可以直接搬到自己的 pipeline：

**1. 策略池构建几乎零成本**。一次性用便宜 LLM 给训练集每条样本生成 2 个策略关键词。算力大头是训练，策略生成那点钱可忽略。

**2. Context dropout = 0.5 是工程上的安全甜点**。不用调，直接抄。

**3. Random sampling > top-ranked sampling**。别花时间挑"最优策略"，随机两个就行。这个发现非常工程友好。

**4. Inter-Intra advantage 不需要新基础设施**。基于 GRPO 改两行代码——按 context 分组算 baseline，再加权组合。任何 GRPO 实现都能改。

**5. Distillation loss 是必要项但权重要小**（$\lambda_{\text{distill}} = 0.1$）。开太大会过拟合，关掉会掉 4 个点。

**6. 如果你的 GRPO 在 N=32 已经撞墙了，试试用 N=8 跑 NudgeRL**。不光省算力，还可能涨点。

---

## 十一、收尾：探索这件事，方向比力气更重要

读完这篇我自己有个比较深的感受。

过去一年 RLVR 这个方向，主流叙事一直是"更大的 rollout、更狠的 clip、更长的训练步数"。但 NudgeRL 摆出来的数据其实是一个反向论证——**当你在错误的山头反复采样，加再多 rollout 也只是在原地踏步**。

真正决定 RLVR 能不能突破的，从来不是 N 多大、$\epsilon$ 多大，而是**你能不能让 policy 看到它原本看不到的轨迹**。一段 30 token 的策略提示，效果好过 64 倍的 rollout 预算——这不是优化算法的胜利，是**问题定义的胜利**。

GRPO 的探索瓶颈不是算法瓶颈，是**输入分布的瓶颈**。改输入比改算法便宜得多，也有效得多。

我觉得这个 insight 会被很多后续工作继承——下一步大概率会看到：

- Model-adaptive context generation（论文 future work 提的）
- 在 code、tool use 任务上的策略提示设计
- 把 Strategy Nudging 和 long context RL（ProRL 这条线）结合
- 跨任务的策略池迁移（一个 domain 的策略能不能用到另一个）

至少在数学推理这个赛道，NudgeRL 给出了一个让我觉得"对，应该是这个方向"的方案。如果你也在做 RLVR，**这篇值得花时间把代码跑一遍**。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
