---
title: "只动一层就够：RL后训练里Transformer的"层贡献"集中现象"
date: 2026-07-09
arxiv: 2607.01232
tags: [RL后训练, GRPO, 层贡献, 单层训练, 参数高效, Qwen3, RLVR]
---

# 只动一层就够：RL后训练里Transformer的"层贡献"集中现象

你有没有过这种直觉：GRPO 跑一晚上，几十层 Transformer 一起被刷一遍，最后分数涨了——你下意识会觉得"每一层都贡献了一部分"。但这篇来自明大+北大+Amazon 的合作论文（arXiv:2607.01232）想告诉你，这个直觉大概率是错的。

他们做了一个看起来很朴素、但极少有人系统性跑过的实验：把模型的全部参数冻住，只放开中间某一层，用 GRPO 在数学题上做 RL 后训练。猜猜结果怎么着？在 1.7B、4B、8B 三个 Qwen3 base 上，最强的那一层单独训练，恢复出来的能力等于甚至超过全参数 RL 训练——最高达到全参数 1.14 倍，意思是"只调一层反而比全调所有层还好"。

更夸张的是，这个现象不是偶然。论文在 7 个模型、2 个模型族、3 种 RL 算法、3 类任务上重复了这个实验，发现了一张相当稳定的"层贡献地形图"：高贡献层总是集中在 Transformer 栈的中段，靠近输入和靠近输出的层都贡献有限。Best layer vs. worst layer 的差距能拉到四倍以上。

读完之后我的第一反应是——这等于给 LoRA、Tina 这种参数高效 RL 后训练的思路又添了一个更强的理论依据：与其在所有层上加低秩更新，不如只看深度方向上的一个层。

## 核心摘要

这篇论文在 7 个模型（Qwen3-1.7B/4B/8B-Base、Qwen2.5-Math-1.5B、Qwen2.5-1.5B/3B-Instruct、DeepSeek-Distilled-Qwen-7B）上系统性地测试了"只训练 Transformer 单一层、其他全部冻结"的效果。他们提出了一个叫做 layer contribution 的指标（"该层独立训练能恢复全参数 RL 增益的比例"），并发现：

1. **高贡献层集中于中间**：在所有 7 个模型上，无论模型族、规模、RL 算法、任务（数学 / 代码 / 智能体决策），40%~60% 深度的层始终贡献最大。
2. **单层可打平甚至超越全参数**：最强层在 1.7B 模型上达到 1.14 倍的全参数增益，在 8B 模型上达到 1.07 倍。
3. **层排名跨数据集/任务稳定**：同一个 Qwen3-1.7B，NuminaMath-CoT 和 DeepScaleR 之间的层贡献排名 Spearman 系数很高；连数学和代码之间也高度相关。
4. **可以直接拿来改进训练**：基于层贡献的"boost 高贡献层学习率"或"只训练高贡献层"在所有三个 Qwen3 规模上稳定优于全参数 baseline；甚至**只取中间 5 层训练**这个零 profiling 启发式也跑赢了全参数训练。
5. **不同层训练出来的模型是互补的**：高贡献层之间 Jaccard 相似度仅 0.22，对 7 个层训练模型做多数投票，在 OlympiadBench 上比全参数基线还高。

一句话判断：这是一篇把"RL 后训练在 Transformer 内部到底长什么样"这个问题真正讲清楚的工作，工程上不复杂，但数据扎实、结论稳定，值得一读。批判性地说，论文的可解释性分析（Section 5.2）做得比方法本身更值得琢磨——它直接反驳了"高贡献 = 权重变化大"这个直觉假设。

## 论文信息

- **标题**：Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training
- **arXiv**：2607.01232v2（2026-07-02 更新）
- **作者**：Zijian Zhang（明大）、Rizhen Hu（北大）、Athanasios Glentis（明大）、Dawei Li（明大）、Chung-Yiu Yau（明大）、Hongzhou Lin（Amazon）、Mingyi Hong（明大）
- **许可**：CC BY-NC-ND 4.0

## 为什么需要这篇论文

RL 后训练现在已经是 LLM 标配了。GRPO、Dr. GRPO、GiGPO 一通招呼，再加 rule-based verifier（数学对答案、代码跑测试、智能体看任务完成度），分数蹭蹭就涨了。但问题来了——我们一直在改所有层。

这个假设其实没什么人检验过。所有层一起动，看起来"全面"且安全，但你心里总会有一丝怀疑：会不会有相当一部分参数更新其实是"陪跑"？会不会有 80% 的有效更新其实只发生在 20% 的层上？

这种怀疑其实有直觉支撑。之前 Song et al. (2026)、Zhang et al. (2024) 早就发现，Transformer 各个层扮演的角色差异巨大——有些层是"角柱层"，砍掉直接性能崩塌，有些层砍了几乎没影响。Nepal et al. (2025) 更进一步，发现数学推理的关键层在 pretrain 阶段就被定型，post-training 之后也基本不变。

但这些工作都集中在**推理时剪枝**和**监督微调（SFT）** 上。RL 后训练是不是也有类似的层结构？之前没人系统跑过。

Paper 里反复提的 LISA（Pan et al., 2024）是个有趣的参照——LISA 在 SFT 阶段随机采样一层来做更新，发现只要采样分布合理，效率能高很多。但 LISA 是"随机采样"，这篇论文要回答的是**"按贡献排序采样"到底有多大的提升**——以及这个排序是不是稳定的。

## 核心方法

### Layer Contribution：怎么量化"单层能恢复多少"

论文的实验设计非常干净。对一个 L 层 Transformer，每次只解冻第 k 层（包括 attn、MLP 等该层的所有参数），把 embedding、lm_head、其他所有 decoder 层都 freeze 住，跑完整轮的 GRPO。训练完成之后在 in-domain benchmark 上测分。

定义 $S_0$ 是 base 模型分数，$S_{\text{full}}$ 是全参数 RL 后的分数，$S_k$ 是只训练第 k 层后的分数。那么 layer contribution 就是：

$$
\mathcal{C}(k) = \frac{S_k - S_0}{S_{\text{full}} - S_0}
$$

$\mathcal{C}(k) = 1.0$ 表示这一层单独训练能完全复刻全参数训练的收益；大于 1.0 表示"比全参数还好"；接近 0 表示这层基本没从 RL 里学到东西。

实现上，PyTorch 里就是一行 `requires_grad=False` 的事——简单到没朋友。但实验的算力开销一点都不简单：每个模型要跑 L+1 次 GRPO，1.7B 的 28 层模型相当于 29 次完整训练。说实话这个量级其实还可以接受，但你要去说服产品/老板"为了选层先花 29 倍算力"，还真得掂量掂量。

### 实验公平性

这种实验最容易踩的坑是"baseline 不够强"或"超参有偏"。论文花了相当篇幅论证他们的对照是公平的：

1. **全参数 baseline 学习率扫了 5 个量级**（1e-5 到 1e-7 量级），用最佳结果做对照。
2. **单层训练用相同的最佳学习率**，不存在"全参数欠拟合、单层恰好过拟合"的混淆。
3. **所有其他超参（batch size、KL 系数、clip range、epoch 数）完全相同**。
4. **专门跑了一个 LR ablation**（附录 A.7），证明调整学习率不会改变层贡献的排序。

> 论文用一句话总结："if a single layer matches or surpasses full-parameter training, the comparison is rigorous: the full-parameter baseline is already at its best learning rate, and the single-layer run uses the same settings."（即当单层能打平或超过全参数时，这个比较是严格的）

## 关键实验结果

### 1. Layer contribution 在所有 7 个模型上高度结构化

![图1：7个模型的层贡献全景与策略对比](https://arxiv.org/html/2607.01232v2/x1.png)

图 1 是全文的"门面"。左边子图把 7 个模型的 layer contribution 都画在了相对深度（0 = Layer 0，1 = Layer L-1）上——所有曲线在中间（约 0.4-0.6）都冒到绿色虚线之上，靠近输入和输出的两端都往下掉。7 个模型、3 种 RL 算法、2 个任务域，全是同一个形状。

右边的柱状图更震撼。三个 Qwen3 规模（1.7B、4B、8B）上，base、Full RL、最佳单层、Best Guided 四组柱子排开——**Best Guided 全部高于 Full RL**，涨幅分别为 **+43%、+27%、+32%**（这是相对"Full RL 自身的增益"再额外多出的比例，不是相对 base 的比例）。

> 注意论文里这种"额外增益"的计算是相对全参数 RL 之后的涨分幅度，不是相对 base 模型的涨分。读图时别被表面的"43 个点"唬住——3 个点对应 1.7B 的最佳单层策略在 MATH500/GSM8K/OlympiadBench/AMC 这 4 个题上的平均分。

### 2. 不同模型规模的 layer contribution 曲线

![图2：三个 Qwen3 规模的逐层贡献曲线](https://arxiv.org/html/2607.01232v2/x2.png)

图 2 把 Qwen3-1.7B/4B/8B 三档模型的逐层贡献拆开看。蓝色是数学 in-domain 贡献，黑色是四类任务（数学/代码/推理/语言）平均的整体贡献。两条曲线在所有三个规模上**都紧紧贴在一起**（Pearson 相关系数 0.91、0.67、0.65），说明"在数学题上贡献高的层，在其他任务上贡献也高"——这不是 task-specific 现象。

数据上：1.7B 上 Layer 10 达到 1.14（最高），Layer 24 只有 0.28（最低）；4B 上 Layer 16 达到 1.06；8B 上 Layer 16 达到 1.07，但 Layer 0 居然是**负贡献（$\mathcal{C}(k) < 0$）**——光训练这一层，数学分反而掉到 base 之下。这个"反向贡献"我觉得挺有意思的，论文没细讲但确实是 8B 特有的现象。

### 3. 跨数据集、跨任务的一致性

![图3：Qwen3-1.7B 上跨数据集/任务的层贡献 Spearman 相关性](https://arxiv.org/html/2607.01232v2/x3.png)

论文还测了"用 DeepScaleR 和 DeepCoder 重新跑一遍同一模型"。NuminaMath-CoT vs. DeepScaleR（两个数学数据集）之间层排名 Spearman 系数很高（论文摘要说 0.83，具体数值我重看了一下表，是数学 in-domain 下的高度一致）；NuminaMath-CoT（数学）vs. DeepCoder（代码）之间也高（论文报了一个相对低的数）。换言之——layer contribution 是模型结构决定的属性，跟你用哪个数据集、训什么任务**关系不大**。

> 论文这一段的工程含义很硬：你可以用一个便宜的、容易搞到的小数据集先做 layer profiling，然后把这个 layer ranking 用到大模型、大数据集的训练上。这点对工业界真的省事。

### 4. 不同模型族、RL 算法、任务上同样成立

论文补做了几个 generalization 实验，结果如下：

| 模型 | 模型族 | 规模 | RL 算法 | 任务 | Best $\mathcal{C}(k)$ | Worst $\mathcal{C}(k)$ | $\mathcal{C}(k) \geq 1$ 的层数 |
|------|--------|------|---------|------|---------------------|------------------------|------------------------------|
| Qwen3-1.7B-Base | Qwen3 | 1.7B | GRPO | Math | 1.14 | 0.28 | 5/28 |
| Qwen3-4B-Base | Qwen3 | 4B | GRPO | Math | 1.06 | 0.66 | 4/36 |
| Qwen3-8B-Base | Qwen3 | 8B | GRPO | Math | 1.07 | -0.51 | 4/36 |
| Qwen2.5-Math-1.5B | Qwen2.5 | 1.5B | Dr. GRPO | Math | 1.01 | 0.42 | 2/28 |
| Qwen2.5-1.5B-Instruct | Qwen2.5 | 1.5B | GiGPO | Agentic (ALFWorld) | 1.02 | 0.25 | 1/8 |
| Qwen2.5-3B-Instruct | Qwen2.5 | 3B | GiGPO | Agentic (ALFWorld) | 1.01 | 0.17 | 1/11 |
| DeepSeek-Distilled-Qwen-7B | Qwen2.5 | 7B | GRPO | Math (Skywork) | 1.05 | 0.33 | 2/8 |

> 注：后 3 个模型因为算力原因只训练了代表性层（partial scan），所以 $\mathcal{C}(k) \geq 1$ 的"层数"统计是分母小一些的数。

智能体任务（ALFWorld）那个数据我觉得最有意思。**RL 在 ALFWorld 上能涨 80 多个点**（base 4% → full 87.8%），这种大幅度的能力跃迁，按理说应该是"全网协调改动"才能实现吧？结果一看——还是中间层贡献最大，Layer 14 在 1.5B 上单独训练就能达到 89.1%，比 full 还高。论文这个实验等于在说："哪怕 RL 学到的是完全新的能力，承载这个学习的层还是同一批。"

### 5. 关键消融：Qwen2.5-Math + Dr. GRPO

![图4：Qwen2.5-Math-1.5B + Dr. GRPO 的逐层贡献](https://arxiv.org/html/2607.01232v2/x4.png)

图 4 把模型族从 Qwen3 换成了 Qwen2.5-Math，RL 算法从 GRPO 换成了 Dr. GRPO。曲线形状跟 Qwen3 几乎一模一样：中间两个红圈（Layer 14、Layer 16）达到或超过 full-parameter baseline，靠近输出端的 Layer 23/24/25 贡献骤降到 0.4 上下。

Layer 14 的数学综合分（Avg）做到了 36.3，比官方 Dr. GRPO 的 36.2 报告值还高 0.1 个点。说"工程上可以当真做"，就有点说服力了。

### 6. 把 layer contribution 直接用来指导训练

这是论文的"实践环节"，图 7 摆了三组策略在三个 Qwen3 规模上的对比：

![图7：基于层贡献的训练策略对比](https://arxiv.org/html/2607.01232v2/x7.png)

**蓝色 Boost B_N**：把 layer contribution 最高的 N 层学习率拉到 3 倍基准学习率（其他层用基准 LR）。"B5"、"B10" 是 5 层、10 层两种。

**绿色 Only B_N**：完全冻结其他层，只训练贡献最高的 N 层。

**红色 W_N（对照组）**：跟 B_N 对照，要么 Boost 最差的 N 层（Boost W），要么只训练最差的 N 层（Only W）。这一组是为了排除"是不是改学习率本身就有正向效果"。

**紫色 Mid 5**：零 profiling 启发式——直接选**位置上的中间 5 层**训练（1.7B 是 Layer 11-15，4B/8B 是 Layer 15-19）。

**灰色 Full**：全参数 baseline。

我把论文里几个关键数据整理成表（MATH500+GSM8K+OlympiadBench+AMC 的 4 题平均）：

| 策略 | 1.7B | 4B | 8B |
|------|------|------|------|
| Full | 50.8 | 63.0 | 66.4 |
| Boost B5 | 52.5 | 64.2 | 67.5 |
| **Boost B10** | **53.7** | **64.4** | **67.4** |
| **Only B5** | 51.5 | **65.9** | 69.0 |
| **Only B10** | 51.4 | **65.4** | **69.1** |
| Boost W5（对照） | 49.2 | 62.5 | 65.9 |
| Boost W10（对照） | 49.2 | 62.6 | 65.5 |
| Only W5（对照） | 46.9 | 59.4 | 62.0 |
| Only W10（对照） | 49.4 | 60.5 | 64.1 |
| **Mid 5（零 profiling 启发式）** | **51.4** | **64.8** | **68.2** |

这张表透露的信息很密：

- **Boost B5 / B10 永远高于 Full**：在三个规模上都稳定涨分。
- **Only B5 / B10 在 4B 和 8B 上比 Boost 还猛**：4B 上 Only B5 直接拉到 65.9，比 Full 高 2.9 个点。论文对此的解释我认同：**低贡献层不仅没用，反而可能"稀释"训练信号**——冻结它们反而让优化更干净。
- **W_N 对照组全部跌穿 baseline**：证明正向效果来自"选对了层"，而不是"学习率调整本身"。
- **Mid 5 零 profiling 启发式在所有规模上都高于 Full**：1.7B 上 +0.6，4B 上 +1.8，8B 上 +1.8。这是个相当实用的工程建议——**没时间/算力做 layer profiling 的话，直接选中间 5 层训练就行**。你想想看，连一次 profiling 都不用做，凭经验取中间就能白拿 1-2 个点，这种好事还挑什么。

> 但这里有个隐含的工程问题。论文说"中间层贡献高"，但它是在 Qwen3 上做的启发式。DeepSeek-Distilled-Qwen-7B 的"Layer 16"对应相对深度 0.57；Qwen2.5-Math-1.5B 的"Layer 14"对应相对深度 0.5——确实都在 0.4-0.6 之间。所以这个"取中间 5 层"的启发式至少在 Qwen 系上能复用。换成 Llama 系或别的结构没验证，迁移性不好说。

### 7. 关键反直觉发现：权重变化大小 ≠ 贡献大小

![图9：逐层参数变化量与 layer contribution 的对比](https://arxiv.org/html/2607.01232v2/x9.png)

这是我觉得全文最漂亮的一张图。

蓝色实线是全参数 GRPO 训练后**每一层**的 L2 权重变化量（$||\Delta\theta_k||_2$）。注意这条线**几乎是平的**——所有层在 0.5 到 0.8 之间均匀变化。颜色尖峰是单层训练时那一层的变化量——比全参数下还要大（约 0.8-1.0），但其他所有层变化为 0。

这个对比直击我之前对"为什么中间层贡献大"的猜想——我本来以为是"全参数训练时中间层动得最多"。**结果不是**。中间层在全参数训练中并不比其他层动得更多，但当它**被单独拎出来**时，捕捉 RL 增益的能力远超两端。

换言之：高贡献层不是因为"动得多"而贡献大，而是因为它的参数子空间对 RL 信号有"更高的兼容性"——这层有权重方向可以走、走出来的结果有泛化性。

> 论文一句话总结："layer contribution reflects the effectiveness of a layer's parameter subspace for capturing RL improvement, rather than the magnitude of parameter change."（layer contribution 反映的是参数子空间对 RL 增益的吸收能力，不是参数变化量本身）

### 8. 另一个加分项：单层训练模型是互补的

Section 5.1 做的实验是在 Qwen3-1.7B 上训练 28 个单层模型（每个层一个），拿前 7 个高贡献层在 OlympiadBench 上做多数投票。

![图8：单层训练模型的多数投票 vs. self-consistency](https://arxiv.org/html/2607.01232v2/x8.png)

- 7 个层的平均 pairwise Jaccard 相似度只有 0.22——它们解的题几乎是**不重叠的子集**。
- 多数投票结果（39.4%）比最佳单层（34.5%）和 Full RL（35.2%）都高。
- 跟 self-consistency（同一个 Full RL 模型采 7 个样）投票的 35.6% 比，差距更明显。

这个发现我比较喜欢：它说明**Transformer 深度方向上的不同层是天然的"专家集合"**，每一层都擅长不同子问题。这个观察跟 Gan & Isola (2026) 那篇 "Neural Thickets" 的工作相互呼应——pretrained 权重附近密集地藏着各种 task-specific experts，而单层训练是沿着"深度"这个 interpretable 轴去够到这些 experts 的方式。

> 论文自己也强调：单层训练+投票更多是**分析工具**而不是实战训练策略（要训 28 次模型不现实），但这种分析对理解 RL 后训练在 Transformer 内部到底怎么改的非常有价值。

## 我自己的判断

### 这篇论文解决了一个真问题

之前在工业界做 RL 后训练时，我自己的体感是"大部分时间在调学习率、KL 系数、reward shaping 之类的东西"，**至于哪些层在动、动得有没有用，没人问过**。大家都默认"全参数一起动是 baseline"。

这篇论文的价值是把"层结构"这件事从"经验性的剪枝观察"提升到了"系统性的训练原则"。

### 三个立刻能用的工程洞察

1. **新模型做 RL 后训练前，先跑一次 layer profiling**。Qwen3-1.7B 跑 28 次 GRPO 听着吓人，但 1.7B 模型本身小，单次训练算力不大。整个 profiling 流水线跑完，拿到一个稳定的 layer ranking 之后，后续做大规模 RL 训练就能用上。

2. **算力紧就只用中间 5 层**。论文里 Mid 5 这个零 profiling 启发式在所有规模上都跑赢 Full。如果你的模型是 Qwen 族、你的训练任务是推理/数学/智能体，**不需要 profiling，直接取中间 5 层**就能稳赚 1-2 个点。

3. **如果一定要全参数训，至少把高贡献层学习率调到 3 倍**。论文里 Boost B5/B10 提升明显，但绝对值上比 Only B5/B10（只训练高贡献层）差一点。我的解读是：低贡献层冻住比让它们"低 LR 跟着动"还要干净。

### 几个我存疑的地方

**第一，论文没有解释为什么中间层贡献大**。作者在 limitation 那节自己承认"a deeper theoretical understanding of why middle layers are disproportionately important for RL adaptation remains an open question"。这不怪他们——能解释清楚这件事需要一些可解释性工作。但工程上能跑赢是事实，机制为什么是这样，留给后人。

**第二，layer contribution 的稳定性在不同模型族之间没充分验证**。论文主要在 Qwen3 / Qwen2.5 上做，外加一个 DeepSeek-distilled Qwen。**Llama、Mistral、DeepSeek 原生架构**上没测过。"中间层贡献大"是不是 Qwen 的特定现象？坦白讲我有点拿不准。

**第三，全参数训练 vs. 单层训练对**没说清楚为什么单层能赢。论文 Section 5.2 反驳了"权重变化大 = 贡献大"，但**为什么高贡献层单独训练时能恢复甚至超过全参数**——是因为全参数训练时其他层在"反向干扰"吗？还是因为冻结其他层后优化 landscape 变干净了？论文没讲清楚。我倾向于是后者（feature collapse 或者"非贡献层变化冲淡了贡献层的有效方向"），但需要更多 ablation 验证。

**第四，Only B5/B10 这个"冻住非贡献层"的做法在 1.7B 上**并没有比 Full 好太多（Only B5: 51.5 vs. Full: 50.8，+0.7；Boost B10: 53.7 vs. Full: 50.8，+2.9）。**为什么大模型上"冻结非贡献层"效果放大这么明显**？论文没仔细分析，但直觉上是：小模型本来能承载的容量有限，全参数训练时信号在所有层分配，浪费一些是合理的；大模型容量大，全参数训练时"非贡献层"占用的算力和优化空间反而更明显，所以冻住它们收益更大。这是个猜测，论文没做对照实验。

### 这篇论文值不值得花时间深读

值得。

不是因为方法有多创新——说到底就是"逐层实验"这个老套路的扎实版。值得读是因为**它给出了一个你读了之后会改变对 RL 后训练直觉的发现**：在 1.5B-8B 的现代 Transformer 上，RL 后训练的有效更新高度集中在中段，靠近输入输出的层基本是"陪跑"。

如果你是做 RL 后训练、PEFT、模型压缩的工程师，这个发现值得你回头看看自己项目的 layer 分布。如果你只关心方法本身，这篇论文没什么大改动。但如果你想理解"为什么 GRPO 能 work"，这篇论文提供了一个目前最系统、最扎实的"层结构"视角。

---

如果觉得有启发，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
