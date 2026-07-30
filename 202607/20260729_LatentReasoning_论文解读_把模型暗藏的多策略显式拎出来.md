# 当大模型"会"很多种思路：怎么把隐式策略显式拎出来？

**论文**：Uncovering Latent Reasoning Strategies in Language Models
**作者**：Awni Altabaa, John Lafferty（耶鲁大学）
**arXiv**：https://arxiv.org/abs/2607.17674
**代码**：https://github.com/Awni00/latent-strategies-in-lms

---

## 核心摘要

你有没有这种感受：让大模型解一道题，它其实有多种解法，归纳、反证、构造都行，但默认你问一次，它只吐一种——具体走哪条路你根本控制不了。这就是这篇论文要拆的骨头：模型的回复分布其实是多种高层策略的"缠在一起"的混合体，但没有任何显式变量负责"我现在用的是哪条策略"。

作者想把这种隐式结构挖出来：让一个 router 给每个输入分配一个 latent strategy 变量 $z$，让 generator 在 $z$ 的条件下生成答案，听起来就是个普通条件 VAE。但事情坏在——generator 是从基础模型初始化的，**它本来就会无条件复现 $p_\theta(y|x)$，对 $z$ 爱答不理**。这导致标准 ELBO 训练出来 latent 完全被忽略，但目标函数依然能拿到很漂亮的分布拟合分。这是一类"病态的 posterior collapse"。

论文的解法分两步：把重建损失归一化到"基础模型自己的回复损失"单位上（变成 fractional information gain），再按基础模型在每个 token 上的 surprisal 给不同位置加权，让 $z$ 专门去解释"基础模型也没把握的分支点"。在自建的六任务算法 benchmark 上，对 Qwen2.5 0.5B/1.5B 跑出来 Strategy Alignment 从 ELBO 的 ~0.33 干到 ~0.91，同时 Distributional Fidelity 保持不变——这是真涨，不是 trade-off。

我的判断：这是一篇**问题定义比方法本身更有价值的论文**。它把"在已经训好的模型上挖 latent"和"从零学 latent 模型"两件事的差异点得很清楚，对应给出的 fractional objective 是工程上立刻能用的东西。但理论那一节读起来很硬，结论只覆盖有限类假设，离"什么时候 z 一定能挖出来"还有距离。benchmark 是合成的，所以也别急着把它当 LLM 通用解法。

---

## 论文信息

| 字段 | 内容 |
|------|------|
| 标题 | Uncovering Latent Reasoning Strategies in Language Models |
| 作者 | Awni Altabaa, John Lafferty |
| 机构 | Yale University |
| arXiv | 2607.17674 |
| 提交日期 | 2026-07-20 |
| 类别 | cs.LG / cs.CL |
| 代码 | https://github.com/Awni00/latent-strategies-in-lms |

---

## 1. 为什么要挖 latent 策略？

先说动机，这部分在论文引言里讲得很实在。

一个在推理任务上训过的语言模型 $p_\theta(y|x)$，它的回复分布其实是多种高层策略的混合。比如让模型证一个命题，它可能用归纳、用反证、用构造法；让模型解一道程序题，它可能写动态规划、贪心、或者图搜索。**这些策略可以复用于很多题目，但模型表示这些策略时完全没有显式变量区分它们**。

![图1：隐式策略结构与显式 latent 空间对比](https://www.mulanai.com/fs/files/0730_737ad40e_three_pa.png)

*图 1：左：基础模型 $p_\theta(y|x)$ 在输出空间里把多种策略缠在一起；中：用 latent 模型初始化时 $z$ 不被使用；右：训练让 $z$ 与参考策略对齐，输出空间里出现按策略分开的可解释区域*

作者想要的 latent 满足三个性质：

1. **Informativeness**：$z$ 要编码 $y$ 中超出 $x$ 的信息；
2. **Strategy-alignment**：$z$ 的变化对应"高层、可复用的策略"变化，而不是 token 级细节或表层措辞；
3. **Cross-input semantic consistency**：同一个 $z$ 在不同输入上应该代表同一个策略——这一点是关键的严格性。

前两个是局部要求，第三个是真正"z 是个可复用的策略变量"的关键。

这事儿能干嘛？作者明说他们把下游应用都留给未来，但点了几条线：interpretability（可以干预策略）、controllability（指定 $z$ 选策略）、exploration（在 RL 后训练时显式遍历策略而不是被动采样）。**如果 $z$ 真的能稳定指代策略，那 RL 探索就从 token-level 升级到 strategy-level**——这个 lift 是挺诱人的。

---

## 2. 为什么标准 VAE 训练这里直接失效

这部分是论文最硬核的地方。

### 2.1 标准条件 ELBO 的恒等式分解

论文从标准条件 ELBO 开始。给定输入-回复对 $(x, y)$，训练时需要 posterior $q_\xi(z|x,y)$ 提供 latent 估计，router $r_\phi(z|x)$ 在生成时提供 latent 分布，generator $g_\phi(y|x, z)$ 是策略条件下的生成器。cELBO 是：

$$
\calJ_{\mathrm{cELBO}}(x,y;\phi,\xi) = \mathbb{E}_{z \sim q_\xi(\cdot|x,y)}[-\log g_\phi(y|x,z)] + \mathrm{KL}\!\left(q_\xi(\cdot|x,y) \,\Vert\, r_\phi(\cdot|x)\right)
$$

这玩意儿有个恒等式：

$$
\calJ_{\mathrm{cELBO}} = -\log p_\phi(y|x) + \mathrm{KL}\!\left(q_\xi(\cdot|x,y) \,\Vert\, p_\phi(\cdot|x,y)\right)
$$

它要求**生成器对 $z$ 取边缘后的边缘分布 $p_\phi(y|x)$ 拟合观测分布**。它**不**要求 $z$ 解释 $y$ 中超出 $x$ 的策略级变化。

### 2.2 这就坏了

如果 generator 的表达能力允许，一个"对所有 $z$ 都输出 $p_\theta(y|x)$"的方案在初始化时就是可用的——generator 本来就来自基础模型，**它一开始就会无 $z$ 复现目标分布**。这个方案在 cELBO 意义下已经是全局最优。

对带 LoRA 微调的预训练 Transformer 而言，这个"惰性解"（inert solution）距离初始化非常近。优化器很容易就停留在那：分布拟合漂亮，KL 项压到 0，但 $z$ 死寂不动。

这跟传统 VAE 的"posterior collapse"不一样。传统 VAE 是因为 decoder 太强、prior 退化成退化分布；这里是因为 **目标分布已经被 generator 完美实现**，$z$ 完全没有用武之地。作者管这叫"objective-level under-specification"。

![图2：ELBO 训练过程——分布保真度上去、Strategy Alignment 永远是 0.5](https://www.mulanai.com/fs/files/0730_a92a9e6b_elbo_fai.png)

*图 2：在 Qwen2.5 0.5B 上跑标准 ELBO：Distributional Fidelity（绿）从 0.6 升到接近 1.0；Strategy Alignment（粉）始终贴在 0.5 不动；KL/initial KL（蓝）一路崩塌到 0——一个标准的"惰性 latent"案例*

### 2.3 基线：先确认问题在哪儿

作者没有在 cELBO 上死磕，而是先用一个 linear probe 测了一下：**基础模型里到底有没有策略信息**。

![图3：策略标签在基础模型隐状态上可线性解码](https://www.mulanai.com/fs/files/0730_58568880_entangle.png)

*图 3：六任务（Summation, Addition, Base conv., Linear eqs., Sorting, Grid paths）× 归一化 token 位置的热图。颜色深表示 linear probe 准确率高。Addition 上位置 0.1 之后就完全可分；Sorting 整体可分；Grid paths 早中晚期都分布得不错——基础模型的隐状态里早就写着策略，只是没暴露成 latent 变量*

这说明**目标结构确实存在**——它就在 hidden states 里，只是没作为可采样、可干预、可复用的 latent 暴露出来。

那 ELBO 失败就不是因为"模型里没东西可挖"，而是目标函数压根没让 $z$ 去解释 strategy-level 的变化。

---

## 3. 核心方法：用基础模型来"指挥"重建

作者的思路是：**让固定的基础模型 $p_\theta$ 当参考，告诉训练"哪些位置才是策略分叉点、应该把压力集中到那儿"**。

### 3.1 三个关键想法

1. **用基础模型做归一化**：把重建损失按基础模型自己的回复损失做归一化，使重建改善可以解释为"基础模型损失中被 latent 解释掉的比例"——一个 fractional information gain 的量纲。
2. **用基础模型 surprisal 做 token 级加权**：每个 token 给一个权重 $a_{\theta,t}(x,y)$，正比于 $b_{\theta,t}(x,y)$ 的某个 power。$b_{\theta,t}$ 大说明基础模型在那个位置不确定，多种延续都还可能——那正是策略分叉点。
3. **保留 KL 正则项**：别让 $z$ 退化到单点。

### 3.2 目标函数

$$
\begin{aligned}
\calJ_\theta(x,y;\phi,\xi) &= \calR_\theta(x,y;\phi,\xi) + \beta \, \mathrm{KL}\!\left(q_\xi(\cdot|x,y) \,\Vert\, r_\phi(\cdot|x)\right), \\
\calR_\theta(x,y;\phi,\xi) &= \frac{1}{c_\theta} \mathbb{E}_{z \sim q_\xi(\cdot|x,y)} \left[ \sum_{t=1}^{T_y} a_{\theta,t}(x,y) \cdot \left(-\log g_\phi(y_t|x, z, y_{<t})\right) \right], \\
a_{\theta,t}(x,y) &= \alpha \cdot \frac{1}{T_y} + (1-\alpha) \cdot w_{\theta,t}^{(\gamma)}(x,y), \\
c_\theta &= \mathbb{E}_{x \sim \calD_X, \, y \sim p_\theta(\cdot|x)}[b_\theta(x,y)].
\end{aligned}
$$

几个细节值得拎出来：

- $b_{\theta,t}(x,y) = -\log p_\theta(y_t | x, y_{<t})$：基础模型在位置 $t$ 给的 per-token surprisal。
- $c_\theta$：基础模型在采样分布上的平均回复损失。当 generator 等于基础模型时，$\calR_\theta$ 期望等于 1——这个量纲就是"基础模型损失的分数"。
- $w_{\theta,t}^{(\gamma)}$ 是 surprisal 比例加权，并通过 $\kappa_\theta^{(\gamma)}$ 做了归一化，**保证加权前后总重建量纲不变**。这是个巧妙但容易被忽视的设计。
- $\alpha \in [0,1]$ 在均匀压力和 surprisal 集中压力之间插值；$\gamma \ge 0$ 控制 surprisal 集中度。

直觉上：自回归生成里"策略分叉"通常发生在答案路径的前几个 token 上——选定"先证归纳还是先证反证"之后，后面很多 token 就被前缀决定了。**基础模型知道哪些位置还"犹豫"，所以 surprisal 加权相当于告诉 $z$ 去解释"分叉点的不确定性"**，而不是被前缀已经决定了的部分。

### 3.3 架构：轻量化适配

参数化上作者用了相对克制的设计：

- Router $r_\phi(z|x)$：在基础模型顶层表示上接两个投影头（mean、log-variance），输出对角高斯。
- Generator $g_\phi(y|x, z)$：采样得到的 $z$ 被投影成 embedding $E(z)$，**作为伪 token 插到 embedding 层**——这样 response token 在自回归时能 attend 到 $E(z)$。
- 训练时 posterior $q_\xi(z|x, y)$：读完整 $(x, y)$ 的另一套 Transformer，参数 $\xi$。**只在训练时用**。

router 和 generator 是同一套带 LoRA 微调的 Transformer；posterior 是独立 Transformer。整个 adapter 训练量非常小，避免破坏基础模型的回复分布。

---

## 4. 实验：怎么挖出来的

### 4.1 Benchmark 设计

为了能让 strategy alignment **可测量**，作者建了一个**有 ground-truth 策略标签的算法任务集**——这是这篇论文能站住脚的关键。

6 个任务族：

| 任务 | 策略变化举例 |
|------|------------|
| List summation | 不同的递归/迭代求和顺序 |
| Sorting | 冒泡、插入、选择等不同排序算法 |
| Grid pathfinding | BFS/DFS/不同遍历顺序 |
| Linear equations | 高斯消元的变量消元顺序 |
| Base conversion | 不同进位路径 |
| Multi-digit addition | 不同进位传播方向 |

每个样本有 problem instance $X$、solution trace $Y$、reference strategy $S$。**$S$ 在训练时 withheld，评估时用来打分**。

### 4.2 两个核心评估指标

- **Distributional Fidelity**：factorized 模型生成的 trace 是否仍然是合法解（trace 可解析，答案正确）。
- **Strategy Alignment**：更严格，用**Analogical Consistency** 测的——从源输入 $X$ 抽 $z$，**在另一个相关输入 $X'$ 上 reuse 同一个 $z$**，看 $g_\phi(\cdot|X',z)$ 和 $g_\phi(\cdot|X,z)$ 是不是同一种策略：

$$
\mathrm{AnalogicalConsistency} = \Pr[\mathrm{strat}(Y) = \mathrm{strat}(Y')].
$$

这是关键的 cross-input 测度——只过 in-input strategy separation 是骗不过这一关的。

### 4.3 方法对比主图

![图4：预训练 Qwen2.5 上各目标函数对比](https://www.mulanai.com/fs/files/0730_2f5353e3_pretrain.png)

*图 4：Qwen2.5 0.5B（左）和 1.5B（右），6 种目标函数 × 2 个指标。Distributional Fidelity（绿）所有方法都在 ~0.97–0.99；Strategy Alignment（粉）出现严重分化——ELBO/β-ELBO ~0.32，token inverse ~0.32，本文方法（global+token、token weighting、global scale）达到 ~0.75–0.91。空心点是 per-task 散点*

具体数（从图中读，主表有更细的）：

| 方法 | Fidelity (0.5B) | Alignment (0.5B) | Fidelity (1.5B) | Alignment (1.5B) |
|------|-----|------|-----|------|
| global + token（推荐） | ~0.99 | **~0.91** | ~0.99 | **~0.91** |
| token weighting | ~0.98 | ~0.83 | ~0.98 | ~0.87 |
| global scale | ~0.97 | ~0.75 | ~0.96 | ~0.75 |
| token inverse（ablation） | ~0.99 | ~0.35 | ~0.99 | ~0.32 |
| ELBO（baseline） | ~0.99 | ~0.33 | ~0.99 | ~0.32 |
| β-ELBO（baseline） | ~0.99 | ~0.32 | ~0.99 | ~0.32 |

我得承认这个图挺让人印象深刻的。**不是说"涨了多少个点"，而是"ELBO 一行基线永远是 0.5，论文方法把 alignment 拉到了 0.9，但 fidelity 一点没掉"**——这说明这不是一个 trade-off，是真的有信号被解锁了。

### 4.4 训练过程中的几何变化

![图5：训练过程中 latent 空间按策略和任务分离](https://www.mulanai.com/fs/files/0730_b767a808_posterio.png)

*图 5：后验均值在 PCA 投影上随训练进度（2%→6%→10%→20%→50%→100%）的演化。颜色按任务+策略。初始化阶段一团混杂；训练中后期按任务和策略逐步出现明显的簇。说明 latent 既学会了 task-conditional 也有 strategy-conditional 的结构*

需要注意一点：**多任务场景下，latent 是"task-conditioned"而不是单一全局策略码本**。同一片 latent 区域在不同任务下可以代表不同策略含义。论文里的线性可分性诊断确认了这点：把 task 信息加入后，策略才能从 $z$ 线性分出，单靠 $z$ 是分不出的。

### 4.5 机制验证：phase space 与散点图

这两张图解释了"**为什么 fractional + surprisal weighting 起作用**"。

![图6：训练轨迹 + 重建损失 vs 对齐的散点](https://www.mulanai.com/fs/files/0730_567da722_random_i.png)

*图 6（左半）：Fidelity-Alignment phase space 上不同方法的训练轨迹。空心圆=起点，实心圆=终点。ELBO/β-ELBO 往 Fidelity 高、Alignment 不动的方向走；本文方法（global scale、global+token、token weighting）同时拉高 Alignment，达到 ~0.80–0.88；token inverse 是个反例，停在低 Alignment*

![图6（右半）：Token-Weighted Reconstruction vs Strategy Alignment](https://www.mulanai.com/fs/files/0730_09d50a62_random_i.png)

*图 6（右半）：横轴是 token-weighted 重建损失（log scale），纵轴是 Strategy Alignment。**对所有方法，token-weighted reconstruction 越低、alignment 越高**——这是个非常干净的负相关，意味着"逼 z 去解释 base model 还没解释掉的 token → z 越来越 strategy-aligned"*

第二个图我读了好几遍，挺漂亮的一图。**它把"为什么这个 objective 起作用"这件事讲清楚了**——不是"换了个 loss 就涨了"，而是"loss 和目标 metric 之间的因果链有具体可观测的代理变量对应"。

### 4.6 β 鲁棒性

![图7：不同 β 设置下的方法稳定性](https://www.mulanai.com/fs/files/0730_733f192a_multitas.png)

*图 7：横轴是 KL 权重 β 从 1 → 0.1 → 0.01；左：constant β，右：linear β warmup。ELBO baseline（灰）和 token inverse（红）永远停在 0.32 左右，跟 β 无关；本文方法三个变体在 β 降低时单调涨到 ~0.75；warmup 调度下涨得更稳*

这说明效果**不依赖于一个精调的 β**，对工程友好。

### 4.7 Ablation：inverse-surprisal

作者还做了一个反向的消融：让 token 权重**反着来**——给低 surprisal 位置更多重建压力。结果是 Strategy Alignment 直接掉到 ~0.32，跟 ELBO 一样惨。**这反过来证实了"高 surprisal 位置确实就是策略分叉点"这个核心假设**。

---

## 5. 一点理论味道

论文第 5 节给了一个**复杂度惩罚变分下界**的框架。形式上他们改写 population objective 为：

$$
\calL_{\beta,\lambda}(q,r,g) = H(Y|X) - (1-\beta) I_q(Y;Z|X) + \mathrm{DecGap}(q,g) + \beta \mathrm{PriorGap}(q,r) + \lambda C(r,g)
$$

profile 掉 $(r, g)$ 后最大化的是

$$
\calJ^{\mathrm{prof}}_{\beta,\lambda}(q) := I_q(Y;Z|X) - \frac{A_{\beta,\lambda}(q)}{1-\beta}.
$$

论文给了一个**充分条件**：

$$
\Delta_{\beta,\lambda}^{\mathrm{prof}}(\delta) = \calJ^{\mathrm{prof}}_{\beta,\lambda}(q^\star) - \sup_{q \in \calQ: d(q, S^\star) \ge \delta} \calJ^{\mathrm{prof}}_{\beta,\lambda}(q) > 0
$$

那么所有 population minimizer 满足 $d(\widehat q, S^\star) < \delta$。样本复杂度 $\calO(\log(|\calQ||\calR||\calG|) / ((1-\beta)^2 (\Delta)^2))$。

坦率说，这部分的实际可读性对应用研究者不太友好——核心意思是"如果目标策略编码的信息量是 profile 后最高的，且跟其他候选相比 gap 足够大，recovery 才发生"。这是个**oracle 视角**的结果，跟实际训练动态的差距论文自己也明说没闭合。理论这块的意义更多是给后续工作一个形式化锚点，不是给方法提供新的设计灵感。

---

## 6. 我的判断

**它解决了什么真问题**：在"已经训好的 LM 上挖 latent"和"从零学 latent 模型"之间，长期被混作一类问题——VAE 套上去跑就完事了。这篇论文把这种混用暴露出来：基础模型已经是 $p(y|x)$ 的一个解，标准 ELBO 会卡在惰性解上。**这是一个被忽视的训练目标问题**。

**方法层面的取舍**：fractional + surprisal weighting 是简洁的设计直觉。"loss 归一化到 base-model 单位"和"按 base-model 自身的 surprisal 集中压力"两个动作各自有清晰动机。架构上的 LoRA + pseudo-token 也很克制。**核心洞察是"用 base model 当尺子"，而不是"加一个新 loss"**。

**真正有用的工程结论**：

- 抽 latent 这种事儿，先确认 base model 里**已经**有目标信号——别上来就训 VAE。
- 如果真要训，**别用标准 ELBO**。直接用 base-relative 的重建项，按 base-model 的 per-token 负 log prob 加权。
- 评估时一定要测**cross-input consistency**，不要只测 in-input separation——in-input 容易作弊。

**几个我皱眉的地方**：

1. **Benchmark 是合成的**。六类算法任务虽好，但跟"真实开放世界 reasoning"还差得远。论文自己也说"如何迁移到开放任务是开放问题"。
2. **理论只是充分条件**。"如果信息量 gap 足够大就 recover"——这不是一个易验证的假设，实际中你怎么知道 gap 大不大？
3. **Latent 是 task-conditioned，不是全局策略码本**。这意味着对一个新任务、或者跨任务组合场景，$z$ 的语义可能不直接迁移。论文没充分讨论。
4. **消融里 token inverse 跑到了与 ELBO 一样的惨**——但 ELBO 也是 ~0.32，这说明两个 baseline 在量纲上其实没本质差距？论文没仔细拆这件事。

**与同期工作的位置**：在 latent variable disentanglement 这条线上，bowman2016 sentence VAE、sohn2015 CVAE 那一脉主要解决"从零学 latent"；sparse autoencoder 那条（cunningham2023、galichin2025）解决"事后分解隐藏表示"。这篇打的是**在 generator 已经训好的情况下挖可复用 latent** 这个更窄、但工程更直接的卡点。它跟前作最大的差异是**显式处理了 base-model 已经是 inert 解的退化陷阱**——这一点过去没人系统讲过。

---

## 7. 如果你也在做相关事

几个直接的 takeaway：

- **想做 latent intervention / controllable generation**：把 base-relative + surprisal-weighted 当起点，至少比直接上 ELBO 强一截。
- **想做策略级 RL 探索**：先把这套 latent 挖出来，再在 $z$ 上做 exploration，这比在 token-level 做 self-consistency 或 tree-of-thoughts 信号密度高得多。
- **遇到 latent collapse 但 ELBO loss 漂亮**：先用 linear probe 测 base model 隐状态里有没有目标信号——如果有，再考虑 fraction 这类目标；如果没有，VAE 也救不回来。

**一个更本质的问题没解决**：当策略分叉点是**跨多 token 的组合决策**时（不是单 token 分叉），surprisal 单点加权还够不够？这篇没测。可能下个工作可以试试 surprisal 路径的 attention 集中度。

---

## 参考文献

- Altabaa, A., & Lafferty, J. (2026). *Uncovering Latent Reasoning Strategies in Language Models*. arXiv:2607.17674.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. arXiv:1312.6114.
- Sohn, K., Lee, H., & Yan, X. (2015). Learning Structured Output Representation using Deep Conditional Generative Models. NeurIPS.
- Bowman, S. R., et al. (2016). Generating Sentences from a Continuous Space. CoNLL.
- Bowman, S. R., & Vilnis, L., et al. (2016). *The ELBO Recurrence*, etc. —— posterior collapse 一脉。
- Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Cunningham, H., et al. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600.
- Wang, X., et al. (2023). Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR.
- Yao, S., et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models. NeurIPS.
- Qwen2.5 Technical Report. https://qwenlm.github.io/blog/qwen2.5/

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
