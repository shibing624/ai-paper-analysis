# KVarN：方差归一化让 2-bit KV 缓存扛住长链推理

> 一句话先说结论：长程推理时代，KV-Cache 量化的真正杀手不是「方向偏」，而是「token 之间幅度差太大」。KVarN 把方向用 Hadamard 抹平、再用一个像 Sinkhorn 一样的双轴方差归一化把幅度也压平，就这么两步，把 2-bit KV 在 Phi-4-14B 上的 AIME24 从 KIVI 的 65.0 / QuaRot 的 56.7 拉到了 73.3，几乎追平 FP16 的 75.0，而额外开销只有 0.18%。

## 论文信息

- **标题**：KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks
- **arXiv ID**：2606.03458（[arXiv 链接](https://arxiv.org/abs/2606.03458)）
- **机构**：华为（Huawei）
- **代码**：[https://github.com/huawei-csl/KVarN](https://github.com/huawei-csl/KVarN)
- **发表时间**：2026 年 6 月

---

## 一、问题：长程推理是 KV 量化的"放大器"

如果你最近在跑 o1/r1 风格的长链推理模型，你会很快撞上一个非常具体、非常贵的痛点：**KV-Cache 把显存吃完了**。一个 8B 量级的模型，开 32K 上下文 + 多个并发请求，KV-Cache 体积可以远超模型权重本身。把 KV 从 FP16 量到 INT4 是常规操作，已经有不少成熟方法（KIVI、QuaRot、KVQuant…）在 4-bit 上表现得几乎无损。

**但如果再激进一步，压到 2-bit 呢？**

之前的工作里，2-bit KV 量化在常规任务（比如 perplexity、问答）上看起来还能对付——毕竟那些任务的解码长度都不长，几百个 token 就出结果。可一旦切到推理任务（MATH、AIME、HumanEval），需要模型连续生成几千上万个 token，量化误差会沿着自回归链一路滚雪球。论文里 KIVI 在 Phi-4-14B 上的 AIME24 直接从 75.0 掉到 65.0，HumanEval 更是从 91.5 崩到 74.6。

KVarN 这篇论文的切入点就特别直白——**专门治长程推理下的 2-bit KV 量化**。它的两个核心断言是：

1. **decode 阶段误差远比 prefill 严重**，因为新生成 token 的 attention 完全依赖之前所有量化过的 K、V，错误会跨步累计。
2. **token 之间幅度差异（magnitude variance）才是误差的主导来源**，比常被讨论的"channel 维度异常值"更值得治理。

第二点是这篇文章我个人觉得最有意思的地方，因为它直接挑战了 KIVI 那一脉"per-channel 处理 K、per-token 处理 V"的隐含假设。

---

## 二、误差从哪来：分解成幅度与方向

要理解 KVarN 为什么这么设计，得先看作者怎么把"量化误差"拆开。

对一个 token 的向量 $x$，它的量化版本 $\hat{x} = x + \epsilon$。作者把误差分解为两个正交分量：

$$
\epsilon = \underbrace{(\|\hat{x}\| - \|x\|)\cdot \frac{x}{\|x\|}}_{E_M\ \text{幅度误差}} + \underbrace{\|\hat{x}\|\cdot \left(\frac{\hat{x}}{\|\hat{x}\|} - \frac{x}{\|x\|}\right)}_{E_D\ \text{方向误差}}
$$

通俗讲：$E_M$ 关心「向量长度有没有量错」，$E_D$ 关心「向量指的方向有没有偏」。

![Figure 1：KV 误差分解，以及不同 token 幅度差异下的可视化](https://arxiv.org/html/2606.03458v1/x1.png)

作者真正做的实验是：**在同一个 head 内，逐对比较两个 token 的量化前后 attention 分数变化**，看看是 $E_M$ 还是 $E_D$ 起主导作用。结论非常清晰——**当两个 token 的范数差距足够大（比如一个是 outlier token、一个是普通 token）时，$E_M$ 完全主导误差**。换句话说，吃亏的不是"方向偏一点"，而是"小的被压成 0、大的还在原位"，softmax 一过就严重偏移。

这个观察其实把社区的注意力从 channel-wise outlier 拉回到了 **token-wise variance**，是一件比较底层的认知校正。

---

## 三、KVarN 方法：Hadamard 抹方向 + 双轴方差归一化压幅度

![Figure 2：KVarN 整体流程，Hadamard 旋转 + 双轴方差归一化嵌入到 KV 量化路径](https://arxiv.org/html/2606.03458v1/x2.png)

KVarN 的整体流程其实非常简单，简单到看完会忍不住怀疑"就这？"——但简单往往就是它的优点。

### 3.1 第一步：Hadamard 旋转把 channel-wise outlier 揉散

这部分基本沿用 QuaRot 的做法：在 K、V 进入量化前，先乘一个随机化 Hadamard 矩阵 $H$，把通道维度上的尖锐 outlier 平均到所有 channel 上。Hadamard 矩阵的好处是 $H H^\top = I$，attention 分数 $QK^\top$ 在 Q 那侧也乘一次就抵消了，数学上完全等价。

这一步主要是处理**方向（channel-wise）误差**，但它对幅度问题作用有限——因为旋转不改变向量长度。

### 3.2 第二步：双轴方差归一化（VarN）

这是 KVarN 的核心。Hadamard 抹掉 channel outlier 之后，token 之间的幅度差异依然在，作者引入一个**类 Sinkhorn-Knopp 的迭代**：

1. **沿 token 轴归一化**：让每个 token 的方差对齐到 1。
2. **沿 channel 轴归一化**：再让每个 channel 的方差对齐到 1。
3. **反复 2~3 次**，让两个轴同时接近 1。

这一步有两个关键点要强调：

- **缩放因子是分组的**：每个 group（一般 32 或 64 个 channel/token）共享一个 scale，这样就可以在反量化时还原，没有信息损失，只是让 quantization grid 用得更"满"。
- **scale 本身需要在解码时计算**：每来一个新 token，要把它沿当前 group 的统计更新。论文里给出的 latency overhead 是 **0.18%**，因为只是几个标量运算，且能 fuse 进 KV 写入 kernel 里。

直觉上，这相当于把 K、V 的协方差从"长椭球"压成"近似各向同性球"，让每一个量化 bin 都被均匀使用。对 2-bit 这种只有 4 个值的极端场景，这个差异是决定性的。

![Figure 3：异常值不成比例占用量化区间，普通 token 被挤压](https://arxiv.org/html/2606.03458v1/x3.png)

---

## 四、伪解码评估：方法学层面的另一份贡献

如果说 VarN 是工程上的核心，那这一节是方法学上让我反复看了几次的部分。

**问题**：评估 KV 量化方法的标准做法，是把 prompt 跑 prefill、然后并行 decode 几个 token，再算 perplexity。但 prefill 时所有 KV 都是当前这一步算出来的，量化误差**不会沿时间累计**。这跟真实 long chain-of-thought 的解码完全不是一回事。

**KVarN 提出的方案**：**伪解码（pseudo-decode）评估**。具体来说，把已经写好的长答案当作 ground-truth decode 序列，逐 token 让模型在量化 KV 上"复述"——每一步都强制使用前面已经被量化、已经反量化、已经累积误差的 KV，但不真生成新 token，只算这个 step 的 logit 分布与 FP16 的偏差。

![Figure 4：伪解码评估 vs 传统并行 prefill 评估，误差曲线差异巨大](https://arxiv.org/html/2606.03458v1/x5.png)

这套评估方法直接揭示了：**很多方法在并行 prefill 评估下看起来差不多，伪解码下立刻拉开差距**。我觉得这个工具未来会成为 KV 量化论文的标配，因为它便宜、可控、且与下游推理基准的相关性很高。

---

## 五、实验结果

实验部分非常工整，覆盖三个模型（Qwen3-4B、Phi-4-14B、Llama-3.1-8B）× 五个推理基准（MATH500、AIME24、HumanEval、IFEval、Line Retrieval）× 一堆 2-bit 基线（KIVI、QuaRot、KVQuant、PolarQuant、TurboQuant、Kitty）。

下面挑两个最有代表性的数字：

**Phi-4-14B（FP16 baseline = 75.0 AIME24 / 91.5 HumanEval）**

| 方法 | AIME24 | HumanEval | IFEval | Line Retrieval |
|------|--------|-----------|--------|----------------|
| FP16 | 75.0 | 91.5 | 82.4 | 100.0 |
| KIVI (2-bit) | 65.0 | 74.6 | 76.1 | 91.4 |
| QuaRot (2-bit) | 56.7 | 78.4 | 74.5 | 95.1 |
| KVQuant (2-bit) | 60.0 | 80.5 | 75.8 | 93.6 |
| **KVarN (2-bit)** | **73.3** | **89.6** | **81.7** | **99.4** |

**Qwen3-4B（FP16 = 80.4 AIME24）**

| 方法 | AIME24 | MATH500 | HumanEval |
|------|--------|---------|-----------|
| FP16 | 80.4 | 96.0 | 89.0 |
| KIVI (2-bit) | 70.4 | 92.6 | 84.1 |
| QuaRot (2-bit) | 72.1 | 93.2 | 84.8 |
| **KVarN (2-bit)** | **78.8** | **95.6** | **88.4** |

几个值得注意的点：

1. **KVarN 在 AIME24 上拉开的差距最大**——AIME24 是 30 题级、长解题链、高方差，对量化误差最敏感，恰好对应论文第二节里"长程解码累积"的判断。
2. **Line Retrieval 上 99.4 vs FP16 100.0**，几乎无损。这个任务高度依赖 KV 检索精度，能保住说明 attention 分布没有被搞坏。
3. **HumanEval 上 KIVI 在 Phi-4 突然崩到 74.6**，KVarN 拉回 89.6。这一组数字最戏剧，但也是我下一节想稍微 push back 的地方。

### 消融

- **只 Hadamard、不 VarN**：等价于 QuaRot，AIME24 56.7。
- **只 VarN、不 Hadamard**：64.x，证明两个组件都需要。
- **VarN 只做 token 轴归一化**：71.x。
- **VarN 只做 channel 轴归一化**：68.x。
- **完整 KVarN**：73.3。

这组消融实证了「双轴 Sinkhorn」确实优于单轴归一化，差距在 2-5 个点之间，并不算戏剧性，但稳定。

---

## 六、批判性点评：哪些地方需要保留怀疑

读完之后我有几点克制的疑虑，列出来供大家自己判断：

1. **Phi-4 上 KIVI 的 HumanEval 74.6 是否被 cherry-pick？** Phi-4 是 14B 模型，KIVI 在它上面比在 Qwen3-4B 上掉得多得多（4B 上 KIVI 还有 84.1）。这种"被打靶"的对比，很容易让 KVarN 显得过分突出。我希望看到论文给出 KIVI 在 Phi-4 上的更多 seed 平均，而不是单跑。
2. **"延迟开销 0.18%"是否真实？** VarN 需要在解码时维护 token 轴的 running variance，理论上要在每个 attention layer 之前同步。0.18% 是端到端打点，但分布式解码（vLLM、SGLang）下这个开销是否还能保持，论文没正面回答。给的 vLLM 实现链接是个加分项，需要后续社区复现。
3. **2-bit 之外的扩展性**：所有数字都是 2-bit。如果到 1-bit、ternary 这种更极端场景，VarN 还成立吗？方差归一化是线性变换，对极低 bit 下的离散误差能否继续帮上忙是开放问题。
4. **与 Kitty / TurboQuant 的差距并不大**。这两个 2025 年底的方法在 KVarN 表里也还有竞争力（很多任务上差 1-3 个点），所以 KVarN 应该被理解为"在前沿基础上又前进了一步"，而不是"碾压前作"。

---

## 七、对工程的启发

如果你正在做长程推理推理引擎的 KV 量化，下面几点我觉得可以直接借鉴：

- **2-bit KV 不再是"理论上可以"**：在 KVarN 这个工作之后，2-bit KV-Cache 在推理任务上几乎追平 FP16 已经是事实。如果你们的服务在 long-context reasoning 上吃显存，这是值得严肃投入工程化的方向。
- **评估方法要换**：标准 perplexity / 并行 prefill 已经测不出 KV 量化方法在长链解码下的实际表现。把伪解码评估纳入回归测试。
- **Sinkhorn-Knopp 这种古典工具在 LLM 系统里还能挖到东西**。VarN 本质就是一个轻量级 Sinkhorn 迭代，但放对了地方就有用。
- **KV 量化和 weight 量化的方法学已经分叉**：weight 量化关心的是 channel-wise outlier、calibration set；KV 量化越来越关心 token-wise variance、time-累积误差。两者要分别建模，不要复用同一套 intuition。

---

## 八、总结

KVarN 的贡献我给三段式总结：

- **观察**：在长程推理解码下，KV 量化误差被自回归放大，主导因素是 token 之间的幅度方差，而不是 channel-wise 方向误差。
- **方法**：在 QuaRot 的 Hadamard 旋转之上加一层双轴方差归一化（VarN），相当于做一次轻量 Sinkhorn 迭代，让 quantization grid 在两个维度上都被充分利用。
- **结果**：2-bit KV 在 AIME24/HumanEval 上几乎无损，端到端延迟开销 0.18%，且开源 vLLM 实现。

它不是一篇追求理论新颖性的论文——Hadamard 是 QuaRot 的、Sinkhorn 是数十年前的、误差分解也很经典。但它把这些已知工具组合在一个被严重低估的问题上（长程解码下的 KV 累积误差），并配套提出了**伪解码评估**这个非常实用的方法学工具。在这个长程推理已经成为标配的时代，这种把"工程上有用的简单组合"打磨清楚的工作，价值是被低估的。

如果你只能从这篇文章带走一句话，我会选这句：**"KV 量化在长链推理下别再只盯 channel outlier，token-wise variance 才是真正的隐形杀手。"**

