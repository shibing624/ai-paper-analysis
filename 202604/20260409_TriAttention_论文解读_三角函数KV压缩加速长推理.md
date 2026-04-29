# 不看Attention分数也能压缩KV Cache？TriAttention用三角函数找到了捷径

你有没有碰到过这个问题：模型推理的时候，生成几千个token还行，一旦到了长链推理（比如数学竞赛题，动辄生成32K token），显存直接爆炸？

这不是个小众问题。现在的推理模型（DeepSeek-R1、QwQ这些）动不动就要"想"上万步，KV Cache的显存占用跟序列长度成正比增长。结果就是：一台24G显卡根本跑不起来，或者batch size只能开1，吞吐量惨不忍睹。

KV Cache压缩这个方向已经有不少工作了，SnapKV、H2O、R-KV这些方法的核心思路都差不多——看attention score，把"不重要"的KV对踢掉。但问题来了：**在长推理场景下，这些方法的效果断崖式下跌**。AIME25上R-KV只剩17.5%的准确率，而Full Attention是40.8%。差距大到离谱。

今天聊的这篇 TriAttention，换了个完全不同的视角来做KV压缩。它不看attention score，而是回到RoPE旋转之前的"原始空间"，发现Q和K向量居然高度聚集在固定中心附近。然后利用这个性质，用三角函数级数来估计每个Key的重要性。

效果？在AIME25上用3072的KV预算（全量是32K），准确率达到40.8%——跟Full Attention持平。吞吐量提升2.5倍，KV显存压缩10.7倍。

这篇论文我觉得最值钱的地方不在于"又提了一个新的KV压缩方法"，而在于它揭示了一个之前大家没怎么注意的现象：**pre-RoPE空间里Q/K的高度集中性**。这个发现本身比具体的方法设计更有价值。

---

## 论文信息

- **标题**：TriAttention: Efficient Long Reasoning with Trigonometric KV Compression
- **作者**：Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen
- **链接**：https://arxiv.org/abs/2604.04921
- **日期**：2026年4月6日

---

## 为什么现有的KV压缩方法在长推理上不work？

要理解TriAttention在做什么，得先搞清楚现有方法为什么会失效。

目前主流的KV Cache压缩方法（SnapKV、R-KV、H2O等）有一个共同的思路：**用当前query的attention score来判断哪些KV对重要**，然后把不重要的踢掉。听起来很合理对吧？你看哪些Key被"关注"得多，就留哪些。

问题出在哪？

这些方法用的是**旋转后（post-RoPE）的query**去算attention score。RoPE旋转会给Q和K引入位置相关的相位旋转，导致同一个attention head在不同位置上的score分布长得完全不一样。

打个比方：你用今天的心情去决定保留过去哪些记忆，但明天你的心情变了，那些被你删掉的记忆可能恰好是明天需要的。

在短序列上这个问题还不严重——反正总共就那么多KV对，压缩也压不掉多少。但到了32K长推理，你只保留2048个KV，相当于删掉了93%的"记忆"。这时候基于post-RoPE attention score的判断就非常不稳定了。

---

## 核心发现：Pre-RoPE空间里的Q/K集中现象

TriAttention的出发点很有意思：**别看旋转后的Q/K了，回到旋转之前看看**。

![图1：Pre-RoPE vs Post-RoPE空间中Q/K向量分布、集中度分布、以及三角级数重建效果](https://arxiv.org/html/2604.04921v1/fig/fig_intro_combined_v2.png)

*图1：四个子图从左到右分别是：(A) Pre-RoPE空间中Q/K向量高度聚集在固定中心附近，R值接近1.0；(B) Post-RoPE空间中同样的Q/K被旋转成散布在圆弧上的分布；(C) 各attention head的集中度R分布，绝大多数head的R>0.9；(D) 三角级数对真实attention logit的重建效果，虚线与实线高度吻合。*

看图1(A)，这是某个attention head在pre-RoPE空间里Q和K向量的分布（投影到2D复平面）。Q向量（蓝色）几乎全部挤在一个小区域里，K向量（橙色）也是。集中度 $R_Q = 1.00$，$R_K = 1.00$。

再看图1(B)，同样的Q和K经过RoPE旋转后，散布在整个圆弧上。这就是为什么post-RoPE的attention score不稳定——旋转把原本稳定的聚集结构打散了。

图1(C)更直观：在所有attention head上统计集中度R，绝大多数head的 $R > 0.9$。这不是个别现象，是普遍规律。

这个发现的物理含义是什么？**每个attention head学到了一组"偏好的方向"——Q和K各自有一个稳定的中心向量**。这个中心向量跟输入内容、位置都基本无关，是模型权重决定的固有属性。

作者把这个性质叫做 **Q/K Concentration**，用 Mean Resultant Length (MRL) 来量化：

$$R = \left\| \frac{1}{N} \sum_{i=1}^{N} e^{j\theta_i} \right\|$$

$R$ 越接近1，说明向量越集中。

---

## 方法：从集中性到三角函数评分

有了Q/K Concentration这个性质，接下来的推导就比较自然了。

### 直觉

如果Q和K都聚集在各自的中心附近，那attention logit（也就是 $q^T k$）就主要取决于两件事：

1. **Q和K中心之间的关系**（这是固定的，可以离线算）
2. **Q和K之间的位置距离** $\Delta$（因为RoPE会根据位置差引入旋转）

换言之，attention logit可以近似为 **位置距离 $\Delta$ 的函数**。而因为RoPE用的是旋转（也就是三角函数），这个函数自然就是三角级数的形式。

### 核心公式

当Q/K高度集中时，attention logit近似为：

$$\text{logit}(\Delta) \approx \sum_f \left[ a_f \cos(\omega_f \Delta) + b_f \sin(\omega_f \Delta) \right]$$

其中 $\omega_f$ 是RoPE各个频率分量的旋转角速度，$a_f$ 和 $b_f$ 是由Q/K中心决定的系数。

图1(D)展示了这个近似的效果：虚线（三角级数重建）和实线（真实attention logit）高度吻合。

### 完整的评分框架

![图2：TriAttention方法总览——离线标定Q/K分布，在线用三角级数评分+Norm评分，保留Top-B的KV对](https://arxiv.org/html/2604.04921v1/x1.png)

*图2：TriAttention流程。左侧离线收集Q/K中心统计量；中间用三角级数计算每个Key的距离偏好分（曲线），加上Norm-based Score（柱状图），得到综合评分；右侧按Top-B保留KV对，生成压缩后的attention map。*

具体操作分三步：

**第一步：离线标定**。跑一小批数据（约10K token就够），收集每个attention head的Q/K中心向量。这一步只需要做一次。

**第二步：三角级数评分** $S_{\text{trig}}(k, \Delta)$。对于每个候选Key位置 $k$，根据它与当前query的距离 $\Delta$，用三角级数算出一个"距离偏好分"。同时用 $1 - R_f$ 作为权重——集中度低的频率分量说明该head在这个维度上不太聚集，应该降权。

**第三步：Norm评分** $S_{\text{norm}}(k)$。除了距离偏好，Key向量的范数（模长）也提供了重要信息。范数大的Key倾向于得到更高的attention score，这一点三角级数没有捕捉到。

最终评分是两者的加权和：

$$\hat{S}(k) = S_{\text{trig}}(k, \Delta) + S_{\text{norm}}(k)$$

然后保留得分最高的Top-B个KV对。

![图3：更详细的方法分解——从Q/K分布到距离偏好评分到KV Cache修剪的完整流程](https://arxiv.org/html/2604.04921v1/x2.png)

*图3：方法的两行对比。上行"Method"展示了从Q/K分布到三角级数评分、Norm评分、最终修剪的计算流程；下行"Effect"展示了每一步对应的attention map变化——修剪后的attention map保留了原始Full Attention的主要结构。*

### 一个细节：未来位置的处理

有个工程上的小巧思值得一提。压缩KV Cache的时候，你不光要考虑当前query跟各Key的距离，还要考虑未来query的需求。TriAttention的做法是评估一组几何间隔的"未来偏移量" $D = \{1, 2, 4, ..., 2^{16}\}$，取最大值作为Key的最终得分。

这比只看当前位置要鲁棒很多，尤其是在长推理中——你不能因为一个Key现在看起来不重要就把它删了，它可能10000步以后突然被用到。

---

## 实验：数据说话

### 主实验：数学推理任务

作者在AIME24、AIME25、MATH500三个数学推理benchmark上测试，覆盖了4个模型：Qwen3-8B、DeepSeek-R1-Distill-Llama-8B（DS-Llama）、DeepSeek-R1-Distill-Qwen-7B（DS-Qwen）、GPT-OSS-20B。

**AIME24/25 主要结果（KV budget = 2048）**：

| 方法 | AIME24 Qwen3-8B | AIME24 DS-Qwen | AIME24 GPT-OSS | AIME25 Qwen3-8B | AIME25 DS-Qwen | AIME25 GPT-OSS |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Full Attention | 57.1 | 43.8 | 69.2 | 40.8 | 34.2 | 60.0 |
| SnapKV | 34.6 | 34.6 | 48.3 | 20.0 | 25.0 | 36.7 |
| R-KV | 25.4 | 34.6 | 49.6 | 17.5 | 23.3 | 39.2 |
| **TriAttention** | **42.1** | **42.5** | **59.2** | **32.9** | **30.0** | **49.2** |

几个关键数据：

- AIME25上Qwen3-8B：TriAttention 32.9% vs R-KV 17.5%，差了15.4个点。这个差距非常大——说明在极端长推理场景下，基于attention score的传统方法几乎不可用了。
- AIME24上GPT-OSS-20B：TriAttention 59.2% vs Full Attention 69.2%，差了10个点。坦率讲，这个差距在竞赛级别的任务上还是有的。但对比SnapKV的48.3%和R-KV的49.6%，TriAttention的领先幅度依然很清楚。

**MATH500（KV budget = 512，更激进的压缩）**：

| 方法 | Qwen3-8B | DS-Llama | DS-Qwen | GPT-OSS |
|------|:---:|:---:|:---:|:---:|
| Full Attention | 69.6 | 82.4 | 87.0 | 91.4 |
| SnapKV | 49.2 | 65.5 | 66.4 | 68.2 |
| R-KV | 46.4 | 76.9 | 71.6 | 77.4 |
| **TriAttention** | **56.0** | **80.6** | **79.6** | **81.2** |

MATH500相对简单一些，但512的KV budget意味着压缩得更狠。TriAttention跟Full Attention的差距在DS-Llama上只有1.8个点（80.6% vs 82.4%），这就很能打了。

### 吞吐量和显存

![图4：Throughput vs Accuracy 和 KV Memory vs Accuracy 的帕累托曲线](https://arxiv.org/html/2604.04921v1/fig/fig_kv_budget_throughput_accuracy_memratio.png)

*图4：左图是吞吐量-准确率曲线，TriAttention在达到Full Attention同等准确率（40.8%）时吞吐量高了2.5倍；右图是KV显存占比-准确率曲线，TriAttention在同等准确率下KV显存只需Full Attention的约1/10（10.7倍压缩）。*

这张图很有冲击力。TriAttention的帕累托曲线（绿色）全程压制R-KV（蓝色），意味着**在任意精度-效率权衡点上，TriAttention都是更优选择**。

具体数字：

| 指标 | MATH500 | AIME24 | AIME25 |
|------|:---:|:---:|:---:|
| Full Attention 吞吐 (tok/s) | 222.8 | 222.8 | 222.8 |
| TriAttention 吞吐 (tok/s) | 1405.2 | 413.9 | 563.5 |
| **加速倍数** | **6.3x** | **1.9x** | **2.5x** |

MATH500上6.3倍的加速——从222.8 tok/s到1405.2 tok/s。这个数对实际部署来说意义很大。

### 不同KV Budget下的表现

![图5：四组实验——MATH500、AIME24、AIME25在不同KV Budget下的准确率曲线，以及Memory Retention Benchmark](https://arxiv.org/html/2604.04921v1/fig/fig_kv_budget_accuracy_combined_4panel.png)

*图5：(A-C) 三个benchmark在512到4096 KV budget范围内的准确率。TriAttention（绿色）在所有budget下都大幅领先R-KV（蓝色），且随budget增加快速接近Full Attention（红色虚线）。(D) Memory Retention Benchmark测试递归DFS任务的记忆保持能力——TriAttention在depth 16之前都能跟上Full Attention，之后逐渐下降但仍优于R-KV。*

图5(D)这个Memory Retention Benchmark我觉得特别有说服力。它用递归DFS模拟的方式，测试模型在"回溯"时能不能还记得之前的中间状态。

![图6：递归DFS中Complete Memory vs Memory Loss的对比示意](https://arxiv.org/html/2604.04921v1/fig/RECURSIVE_SIMULATION.jpg)

*图6：左侧是完整记忆的情况，每层递归的中间状态都被保留，返回值正确传播；右侧是记忆丢失的情况，中间某层状态被"遗忘"，导致错误从底向上传播，最终结果出错。*

这个测试跟实际的长链推理场景非常对应。数学推理经常需要多步回溯，中间任何一步的KV被错误删除，后续的推理链都会崩掉。R-KV在depth 12之后就开始明显掉点，而TriAttention一直撑到了depth 16。

### 消融实验

消融实验回答了几个关键问题：

**三角级数评分到底有多重要？**

去掉 $S_{\text{trig}}$ 后，AIME24准确率从42.1%暴跌到18.8%。23.3个点的差距——这说明三角级数评分是方法的核心，不是锦上添花。

**集中度加权有多重要？**

去掉集中度 $R$ 的加权后，AIME24从42.1%降到41.3%，AIME25从32.9%降到28.7%。AIME24上影响不大，但AIME25上差了4.2个点。说明在更难的任务上，对非集中head的降权处理是有意义的。

**标定数据跨域泛化？**

用Coding数据标定，去做数学推理：AIME24 44.2%，AIME25 29.2%。跟用推理数据标定的结果（42.1%、32.9%）差别不大。这说明Q/K中心确实是模型的固有属性，不太依赖标定数据的领域。

**几何间隔 vs 线性间隔的未来偏移？**

几何间隔45.8% vs 线性间隔28.7%。差距很大，说明"远处也要照顾到"这个设计决策很重要。

### 跨模型泛化

![图7：三角级数重建在不同模型上的Pearson相关系数分布](https://arxiv.org/html/2604.04921v1/fig/fig_freq_reconstruction_multimodel_row1_only.png)

*图7：DS-Qwen3-8B、DS-Qwen-7B、DS-Llama-8B三个模型上，三角级数重建的Pearson相关系数分布。均值分别为0.53、0.56、0.51，说明Q/K集中现象在不同架构上都存在。*

三个模型上的重建相关性都在0.5-0.6左右。说实话这个数字不算特别高——大约一半的head重建得很好（$r > 0.7$），但也有一部分head的重建效果一般。作者用Norm评分来补偿这些集中度不高的head，这个设计是合理的。

---

## 我的判断：亮点和疑问

### 亮点

**1. Pre-RoPE空间的Q/K Concentration是个有价值的发现。** 之前的KV压缩工作基本都在post-RoPE空间做文章，TriAttention跳出来看了一层更底层的东西。这个发现不仅能用于KV压缩，我觉得对理解Transformer的attention机制本身也有启发。为什么Q和K会如此聚集？这背后的训练动力学是什么？论文没有深入讨论，但这个问题值得进一步挖掘。

**2. 方法的数学推导干净利落。** 从集中性假设出发，经过三角恒等变换，自然地推导出三角级数评分。不像有些工作靠empirical trick堆出来的，这里的motivation-method链条很清楚。

**3. 离线标定的工程友好度。** 只需要一次前向pass收集统计量，不需要额外训练，不需要修改模型结构。这对实际部署来说门槛很低。

### 疑问和局限

**1. 重建相关性只有0.5左右，够用吗？** 虽然最终效果不错，但平均Pearson $\bar{r}$ 在0.5左右意味着三角级数只解释了大约25%的attention logit方差。实际效果好可能是因为我们只需要找到Top-B个最重要的Key，不需要精确重建每个logit值——排序正确就够了。但这也意味着，如果KV budget压得更低（比如256），重建精度不够可能就开始成问题了。

**2. 集中度假设在所有模型上都成立吗？** 论文测的都是7B-20B的模型。更大的模型（70B+）或者不同训练范式的模型（比如纯SFT没有RLHF的）是否也有这么强的Q/K集中性，还需要验证。

**3. 跟Full Attention的差距在高难度任务上依然存在。** AIME24上59.2% vs 69.2%，差了10个点。AIME25上32.9% vs 40.8%，差了约8个点。对于数学竞赛这种"差一步就全错"的任务，这个差距可能意味着很多本来能做对的题做错了。不能因为大幅领先baseline就忽视跟Full Attention的差距。

**4. 与MLA架构的兼容性。** 论文提到在GLM-4.7-Flash（MLA架构）上也做了验证，但细节不多。DeepSeek-V3/R1用的也是MLA，如果TriAttention在MLA上效果有打折，那实际应用范围会受限。

**5. 标定数据量和分布的鲁棒性。** 虽然消融显示跨域标定的损失不大，但只测了Coding到Math这一个跨域方向。如果是从纯代码标定去做长文档摘要，效果会不会不一样？

---

## 技术背景补充：KV Cache压缩这条线

KV Cache压缩这个方向，可以粗略分成几个流派：

| 流派 | 代表方法 | 核心思路 | 典型问题 |
|------|---------|---------|---------|
| 基于Attention Score | SnapKV, H2O | 用最近query的attention score判断Key重要性 | 长序列上score分布不稳定 |
| 基于统计特征 | R-KV | 用Key的统计特征（如累计attention）做筛选 | 长推理中仍有较大精度损失 |
| 量化压缩 | KIVI, KVQuant | 把KV对量化到低精度 | 只能压4倍，跟长度无关 |
| 架构级 | MQA/GQA/MLA | 从模型设计上减少KV的head数 | 需要重新训练 |
| **模型固有属性** | **TriAttention** | **利用pre-RoPE的Q/K集中性做三角级数评分** | **新方向，待更多验证** |

TriAttention开辟的其实是第五条路：不看运行时的attention分布，而是利用模型权重决定的固有属性来做压缩决策。这个思路的好处是评分信号更稳定（因为不依赖具体输入），缺点是丢失了一些上下文相关的信息（Norm评分作为补偿）。

顺便说一句，RoPE引入位置编码的方式——在复平面上做旋转——从信号处理的视角来看，就是把位置信息编码到了不同频率的正弦/余弦分量上。TriAttention利用的三角级数分解，其实就是在做RoPE机制的"逆工程"。从这个角度看，这个方法的理论基础是挺solid的。

---

## 总结

TriAttention这篇论文，我觉得定位在"有理论发现支撑的工程方法"这个位置上。Q/K Concentration这个发现有一定的学术价值，三角级数评分框架在工程上也比较优雅。实验结果在长推理场景下确实有亮眼的表现，尤其是跟SnapKV、R-KV比起来优势明显。

但也别过度乐观——跟Full Attention比还是有gap的，特别是在竞赛级的硬核数学题上。这类方法更适合的场景可能是：你的显存不够跑Full Attention，或者你想在相同硬件上跑更大batch。它是一个"在资源受限场景下的高性价比选择"，而不是"可以无损替代Full Attention"。

如果你在做推理模型的部署优化，这篇论文的方法值得试试——离线标定的门槛很低，代码也开源了。如果你在做Transformer机制研究，Q/K Concentration这个现象值得深入探究。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
