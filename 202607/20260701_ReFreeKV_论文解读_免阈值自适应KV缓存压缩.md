---
title: "ReFreeKV：别再为 KV 缓存压缩挑阈值了"
date: 2026-07-01
paper: "ReFreeKV: Towards Threshold-Free KV Cache Compression"
arxiv: 2502.16886
---

# ReFreeKV：别再为 KV 缓存压缩挑阈值了

你有没有这种感觉——同一个 KV 缓存压缩工具，在论文里刷 LongBench 把把 95% 的成绩单扔出来很威风，但真上线服务一接用户的 query，吞吐量、显存都按设计省下来了，结果不是答非所问就是幻觉。**问题到底出在哪？**

直白点：所有这些方法在论文里都需要先给你一把尺子，叫"保留 20% 的缓存"、"保留 1024 个 token"或者"top-p 留 0.95"。这把尺子在它刷的 benchmark 上挑得很准，可一旦换成你 open-domain 的真实流量，就抓瞎了。

ReFreeKV 这篇论文就盯死了这件事：它不想让你再调阈值。它用一个**通用的"归一化注意力范数差"**作为停手信号，让每个输入自己决定要保留多少 KV——简单题少留，难题多留，全程不需要你选一个全局的 budget。

## 核心摘要

KV 缓存压缩不是新东西，从 H2O、StreamingLLM、SnapKV 到 CAKE、Ada-KV、Twilight 一路下来，工作不少。但所有方法都被一个隐含的诅咒困住：**必须先预设一个"保留多少"的阈值**。同一个阈值，在 GSM8K 上能压到 20% 还能保持精度，在 NarrativeQA 上可能需要 80%，在 needle-in-a-haystack 上又是另一回事。真实场景的输入是开放域混合的，**你根本没法为每一类输入提前挑好阈值**。

ReFreeKV 提出一个叫 "Threshold-Free" 的新目标：在不调任何输入相关阈值的前提下，让压缩后的推理质量始终接近甚至超过 full-cache。它的实现 ReFreeKV 是一个**两阶段并行**的方法：先按位置把 KV 排个序（首部 + 尾部反序），再用一个跨模型、跨任务都稳定的停止准则——**注意力矩阵的 Frobenius 范数差**——来决定每一层砍到哪为止。一个固定的 1% 阈值就能在 13 个数据集、5 个模型规模（8B 到 72B）上跑出接近 full-cache 的成绩：Llama3-8B 平均 +0.12%，Qwen2.5-7B 平均 **+2.63%**，Mistral-7B 平均 -1.5%，而实际 KV 预算自动落在 63%~87% 区间。

论文：https://arxiv.org/abs/2502.16886

> **我的判断**：这是一个**目标重定义**的论文。它的方法本身不算惊艳（位置排序 + 范数停手，听起来都很朴素），但它把"压缩方法必须给个阈值"这个被业内默认的约束**摆到台面上质疑**，并用一个简单方案填了这个坑。**对做 KV 缓存推理加速的工程同学特别值得读**——很多产品里 SnapKV、StreamingLLM 跑炸的真实原因就是阈值没选对。

---

## 一、问题动机：为什么"挑阈值"是个真问题

论文的 Table 1 一上来就把这个痛点摆得很直接：H2O、StreamingLLM、SnapKV 在 50% budget 下，GSM8K 上可能掉 60%，NarrativeQA 上可能只掉 1%——**阈值合适就没事，阈值不合适就崩**。所谓"在某数据集上达到 full-cache 等价"，**说到底就是在过拟合这个数据集**。

这事我之前做长上下文推理服务时也踩过坑：用一个静态的 budget 在文档 QA 上很省，跑到多轮对话里直接被用户报告"答非所问"。当时我们的临时方案是按任务类型分流——QA 服务用一个 budget，代码生成用另一个，闲聊用第三个。但这是**任务分桶+每桶挑阈值**，而不是从根本上解决"挑阈值"这个动作本身。

更糟糕的是，近两年的工作把"自适应"当主旋律，但**自适应不等于免阈值**：

- **Ada-KV**：head-wise 分配预算，但全局预算还是预设的；
- **DuoAttention**：把 head 分成 retrieval/streaming 两类，streaming head 用固定窗口，这个窗口还是人为拍的；
- **Twilight**：借鉴 top-p 思路干掉显式 budget，但**p 值还是要按模型调**（Llama3 要 0.95，Mistral 要 0.85）。

**ReFreeKV 第一个说：别挑了。我给你一个不挑阈值也能逼近 full-cache 的方法。**

---

## 二、方法核心：两阶段，免阈值

ReFreeKV 的工作流就是两个阶段，PyTorch 一行累计和就能搞定，没有循环，延迟可以忽略。

![ReFreeKV 工作流](https://arxiv.org/html/2502.16886v4/x1.png)

*图 1：ReFreeKV 工作流。Prefill 完成后，先按位置给 token 排序（首部 + 尾部反序），再对每层、每个 head 用范数差信号砍掉最不重要的 token，把剩下的 KV 留到后续 decode。*

### 阶段一：Initial Ranking（位置排序）

这步简单得有点反直觉：**不靠注意力分数，纯靠位置**。

具体做法：把 KV 序列重排成 `[x_1, x_2, ..., x_m, x_n, x_{n-1}, ..., x_{m+1}]`。前 m 个位置是"开头锚点"（attention sinks），后面是反序的"最新位置"（recency bias）。m 在论文里固定为 4。

为啥不用 attention score 排？论文 Table 5 给了数据：用 attention score 当初始排序（类似 H2O），在 0.01% 阈值下 NarrativeQA 14.79、QMSum 20.90，但同样的输入在 Pos 排序下 15.96、21.95——**位置排序又快又稳**。直觉上也讲得通：attention 矩阵的稀疏模式在 prefill 阶段还是带噪声的，但**开头 token 是结构性必要、尾部 token 是滚动必要**，这两个东西是普适的，不挑模型不挑任务。

### 阶段二：Eviction by Uni-Metric（用统一指标砍）

这是 ReFreeKV 的灵魂。

定义：

$$\text{Uni-Metric}(i) = \frac{\|A\|_F - \|\widetilde{A_i}\|_F}{\|A\|_F}$$

其中 $A$ 是这一层、这个 head 的 attention 矩阵（n×n），$\widetilde{A_i}$ 是把 $A$ 中第 i 列之后（按排序后位置）所有位置都 mask 掉之后的矩阵。

**直觉**：如果砍掉 i 之后的所有位置，范数变化不大，说明这些位置的注意力贡献本来就很小，可以放心砍。当 Uni-Metric 达到阈值 T 时，停手。

**最关键的发现**：阈值 T 在**所有模型、所有任务上都可以是 1%**。

论文 Figure 2 给了 LLaMA3-8B、Mistral-7B、Qwen2.5-7B 在 0.1% 到 5% 阈值下，两个数据集（GSM8K 数学推理、NQA 阅读理解）的归一化分数：

- 在 T ≤ 1% 区间，所有曲线的归一化分数都 ≥ 0.9
- T 一旦超过 1%，Mistral 在 GSM8K 上直接掉到 0.42（不可用）

**T = 1% 是一个跨模型稳定的"安全档位"**。

![阈值 vs 性能](https://arxiv.org/html/2502.16886v4/x2.png)

*图 2：不同阈值下三个模型的归一化分数。T ≤ 1% 时所有曲线都稳在 0.9 以上，T > 1% 之后 Mistral 急剧掉到 0.4。1% 是 Universal Threshold 的实证依据。*

### 计算复杂度：怎么把 O(n²) 压到 O(1)

Uni-Metric 的原始计算需要对 n×n 矩阵做 Frobenius 范数，单层就是 O(n²)，32 层堆起来 32×n²，撑不住。

论文的招数很巧：

1. **只取 A 的最后一行**（k=1）做平均，得到一个 1×n 向量 A'。这一步的复杂度是 **O(1)**，与序列长度无关。
2. 对 A' 做 **cumulative-sum-of-squares**，直接用 PyTorch 的 `torch.cumsum` + `torch.where` 算到每一列的范数差，**并行算所有层**。

论文 Table 5 给了 k 取值的消融：k=1、k=1%n、k=5%n、k=10%n，**k=1 的效果跟 k=1%n 几乎一样，但省了大量计算**。k=10%n 反而崩了——NQA 从 23.44 掉到 9.74，证实了**取太多行会带进噪声**。

### 实现细节：保留前两层

作者沿用 Wan et al. (2024) 和 Xiao et al. (2024) 的发现：LLM 的最下面几层 attention 分布很均匀，不能砍。**ReFreeKV 直接把前 2 层的 KV 完整保留**，后续 30 层才参与范数剪枝。Appendix B 进一步验证了 frozen layers 的层数选择是稳的。

这个设计让"砍哪些 token"这件事非常本地化——只在每层内部做，不跨层共享决策。

---

## 三、实验：13 个数据集，5 个模型规模

### 3.1 主实验：自动逼近 full-cache

Table 2 覆盖了 13 个数据集、6 大类任务：

- **Math & Science**：GSM8K、GPQA、TheoremQA、TheoremQA-Hard
- **Commonsense Reasoning (CR)**：CoQA、NarrativeQA
- **Single-Doc QA**：Qasper、2WikiMQA
- **Multi-Doc QA**：Musique
- **Summarization**：QMSum、Multi-News
- **Few-Shot Learning (FSL)**：TriviaQA
- **Code**：Lcc

跨 3 个 7B/8B 模型的结果（取自 Table 2，归一化平均）：

| 模型 | Avg.（相对 full-cache） | 实际 KV budget |
|------|-----------------------|----------------|
| **Llama3-8B-Instruct** | **+0.12%** | 63.68% |
| Qwen2.5-7B-Instruct | +2.63% | 76.02% |
| Mistral-7B-Instruct | -1.50% | 86.75% |

这个数据特别有意思——**ReFreeKV 居然能超过 full-cache**。在 Llama3-8B 和 Qwen2.5-7B 上，平均分都比完整 KV 缓存高 0.12 和 2.63 个点。作者的解读是：完整 KV 缓存里其实混了"中等重要但其实可砍"的 token，**这些 token 会成为 attention 计算的噪声**——范数剪枝顺带把它们清掉了，结果性能反而比 full-cache 高。

对比基线更扎心。基线统一用 90% budget（H2O 0.9、SLM 0.9、SnapKV 0.9 等），在 Llama3-8B 上的归一化平均：

| 方法 | Llama3-8B Avg. | Qwen2.5-7B Avg. |
|------|----------------|-----------------|
| Full-cache | 100% | 100% |
| **Ours (k=1)** | **+0.12%** | **+2.63%** |
| H2O @ 0.9 | -0.39% | -1.46% |
| SLM @ 0.9 | -0.59% | -0.41% |
| SnapKV @ 0.9 | -1.07% | -1.01% |
| PyramidKV @ 0.9 | -1.59% | -2.76% |
| CAKE @ 0.9 | **-4.17%** | -0.07% |
| H2O @ 0.5 | -16.17% | -13.47% |
| SLM @ 0.5 | -24.73% | -23.56% |
| SnapKV @ 0.5 | -15.57% | -14.39% |
| SLM @ 0.2 | -28.21% | -37.22% |

看到 50% budget 下 SLM 在 Qwen2.5 上掉了 23 个点吗？这就是"挑阈值"在工程里炸的样子。**ReFreeKV 不用挑，全程稳定在 full-cache 附近**。

### 3.2 通用化：70B、32B、72B 都不需要重调

Table 6 直接拿同一套配置（k=1, m=4, T=1%）打到 Llama3-70B、Qwen2.5-32B、Qwen2.5-72B 上：

| 模型 | GSM8K | GPQA | NQA | 2WkMQA | Musique | Avg. Budget |
|------|-------|------|-----|--------|---------|-------------|
| Llama3-70B Full | 90.40 | 52.94 | 30.43 | 36.40 | 21.00 | 100% |
| Llama3-70B Ours | 89.94 | 51.68 | 30.66 | 35.95 | 19.50 | ~50% |
| Qwen2.5-72B Full | 90.22 | 54.14 | 24.36 | 42.13 | 23.93 | 100% |
| Qwen2.5-72B Ours | 90.30 | 54.19 | 24.10 | 41.70 | 23.31 | ~75% |

**同一套超参，模型从 7B 涨到 72B 都不用动**。这是 ReFreeKV 最实用的一点——线上服务多模型并存时，你不用为每个模型维护一套 budget。

### 3.3 效率：剪枝开销可忽略，端到端还更快

论文 Table 4 比较了 Llama3-8B/70B 上 ReFreeKV 与 H2O、SLM、SnapKV 在 50% budget 下的剪枝时间（Prune）和端到端生成时间（Overall）：

- **剪枝开销**：跟 H2O、SnapKV 一个量级
- **端到端**：ReFreeKV 自动适配更高压缩比，**12 个对比里 8 个 Overall 最快**

Table 11 把 batched throughput 拿出来对比：单 A100 上，HF Accelerate vs ReFreeKV 在不同 batch size 和序列长度下，**ReFreeKV 吞吐量提升 10-20%**，且 batch 越大优势越稳。

注意 ReFreeKV 的剪枝位置是**用 cumsum + where 算出来的，不是 Python 循环**——这一点很关键，否则就是另一个"Demo 数据集很美，真实 batch 撑不住"的玩具。

### 3.4 消融：k=1、位置排序、T=1% 三个决策都站得住

**k 的选择**（Table 5 上半部分）：

| k | GSM8K | CoQA | NQA | Musique | QMSum | Budget (avg) |
|---|-------|------|-----|---------|-------|--------------|
| **k=1** | 76.50 | 52.86 | 23.44 | 15.96 | 21.95 | ~57% |
| k=1%n | 76.19 | 52.87 | 23.17 | 15.40 | 21.94 | ~81% |
| k=5%n | 75.59 | 52.75 | 21.62 | 13.71 | 21.43 | ~61% |
| k=10%n | 76.72 | 52.85 | **9.74** | 13.67 | 21.11 | ~49% |

k=1 是性能/预算/计算复杂度的甜点。

**初始排序策略**（Table 5 下半部分）：

- Attention-based 排序 @ T=1%：GSM8K 直接崩到 2.35，CoQA 43.68——**完全不能用**
- Attention-based 排序 @ T=0.01%（更激进）：54.66、51.58、17.45、14.79、20.90——平均掉了 5 个点

**位置排序稳赢**。这跟我之前的工程经验吻合：prefill 阶段的 attention 矩阵本身就是带噪声的，用它做排序等于"信噪比不到 1 的信号当真理"。

**通用阈值 T**（Figure 3）：

![性能 vs 预算的权衡](https://arxiv.org/html/2502.16886v4/x3.png)

*图 3：左图为 5 个数据集在 T=0.1%~10% 范围内的性能保持率，右图为对应的 KV 预算消耗占比。T=1% 是性能-预算的甜点。*

右图特别值得看：在 T=1% 时，5 个数据集的预算已经被砍到 15%~50% 区间（QMSum 这种简单摘要甚至能压到 15%），但左图对应的性能保持率都在 95% 以上。**T 再小收益微乎其微，T 变大风险陡增**。

---

## 四、ReFreeKV 与 Twilight 的对比

值得单独说。Twilight 也在向"无显式 budget"靠拢——它借鉴 top-p 的思路动态决定保留哪些 token。但 Twilight 论文里**Llama3 用 p=0.95，Mistral 用 p=0.85**——这就是"换了个名字的阈值"。

Table 3 把两者放一起比：

- GSM8K 上 ReFreeKV 4.638 vs Twilight 5.330（Llama3-8B）
- TriviaQA 上 ReFreeKV 3.300 vs Twilight 3.318（Llama3-8B）
- 70B 上 ReFreeKV 8.290 vs Twilight 8.487

**性能基本打平**，但 ReFreeKV 不用调 p，Twilight 每个模型要调。**ReFreeKV 的"免阈值"是真正的免**。

---

## 五、我的判断：值不值得读、值不值得用

### 优点

1. **目标重定义的价值比方法本身大**。它把"挑阈值"这件事摆到台面上，迫使整个领域反思。在 Llama3-8B 和 Qwen2.5-7B 上**反超 full-cache** 是个意外的好结果，提示了"全量 KV 不一定是最优 KV"这个反直觉的事实。
2. **实现极简**。cumulative-sum + where + frozen first 2 layers，三件事就完了。不需要重训，不需要额外模型，不需要复杂调度。
3. **跨模型、跨任务、跨 batch 稳定**。Table 6 的通用化数据是真的省心，线上服务多模型并存的话，能少维护一堆超参。
4. **延迟可控**。Table 11 的 throughput 提升是 10-20%，batch 越大越好，这跟 vLLM 的 page attention 还能继续叠。

### 不足

1. **压缩比不够激进**。Mistral-7B 在 QMSum 上保留了 84.3% 的 KV——而 50% budget 跑出来分也不掉。**它把"免阈值"摆在了"最激进压缩比"前面**，所以压得不狠。如果你的服务卡在显存而不是吞吐量，ReFreeKV 不是最优选。
2. **Mistral 上掉了 1.5%**。不是 full-cache 等价。论文里也老实承认了。Mistral 系列推理的 attention 分布可能跟 Llama/Qwen 不一样，导致位置排序假设不那么稳。
3. **没有形式化保证**。T=1% 是经验值，没有理论推导说"为什么是 1% 而不是 2% 或 0.5%"。这个工作更像工程经验主义而不是理论突破。
4. **Prefill 阶段才做剪枝**，对 streaming decode 的支持没说清楚，对 prefill 已经成瓶颈的长上下文场景可能不适用。

### 工程上的建议

- **如果你正在做长上下文 LLM 推理服务，且多模型、多任务并存**，ReFreeKV 值得直接试试。它大概率比 SnapKV + 调 budget 来得省心。
- **如果你卡显存、对压缩比敏感**，建议先看 H2O/SnapKV 的更激进配置，ReFreeKV 偏保守。
- **如果你做研究**——这篇论文的"Threshold-Free"框架才是真正值得 follow 的点。**"自适应"的方向已经做到头，下一步应该问"能不能让系统自己选超参"**。ReFreeKV 是这个方向的第一步，但肯定不是最后一步。

### 一句话总结

ReFreeKV 的核心方法朴素（位置排序 + 范数差停手），但它**重新定义了"好的 KV 压缩"应该满足的条件**——不再需要调阈值。这件事的意义远超"又一个 KV 压缩 SOTA"。

---

**参考**

- 论文：https://arxiv.org/abs/2502.16886
- 代码：https://github.com/Patrick-Ni/ReFreeKV
- 同方向对比工作：H2O (Zhang 2023)、StreamingLLM (Xiao 2024)、SnapKV (Li 2024)、CAKE (Qin 2025)、Ada-KV (Feng 2024)、DuoAttention (Xiao 2025)、Twilight (Lin 2025)

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
