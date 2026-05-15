# UniPrefill：当稀疏注意力遇到混合架构，长上下文 Prefill 该怎么加速？

## 核心摘要

上周在调一个长上下文推理服务的时候，我又被 prefill 的 TTFT 折磨了一遍。128K 的输入，光首 token 出来就要等好几秒。再翻最近的论文，我有个挺直接的感受：MInference、FlexPrefill 这类稀疏注意力方法在**纯 Full Attention** 的 LLaMA 上吊打 baseline，可一旦把它们扔到 Qwen3-Next、Gemma-3 这种**混合架构**上，速度立马打回原形——有的甚至比 baseline 还慢一点。

这篇 UniPrefill 就是冲着这个尴尬来的。它的核心思路非常朴素：**不在注意力层里"少算几行"，而是在 token 这个维度直接"扔掉一批"**——一旦某个 token 被判定为冗余，后面所有层（Attention、Linear Attention、Sliding Window、FFN）通通不用算它。再加上把这套机制硬塞进 vLLM 的 continuous batching 调度器里，让它真的能上线服务。最终在 LLaMA-3.1-8B、Qwen3-Next-80B-A3B、Gemma-3-12B 三种架构上，128K 上下文实现最高 **2.26× TTFT 加速**，精度几乎不掉。

说实话这篇文章的算法部分并没有特别惊艳，关键价值在两块：**架构无关 + vLLM 工程落地**。前者解决了稀疏注意力方法的最大软肋，后者把"实验室加速"做成了"生产可用"。

---

## 论文信息

- **标题**：UniPrefill: Universal Long-Context Prefill Acceleration via Block-wise Dynamic Sparsification
- **作者**：Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, Ran He
- **机构**：MAIS&NLPR CASIA、UCAS、微信（Tencent）
- **arXiv**：https://arxiv.org/abs/2605.06221
- **代码**：https://github.com/qhfan/UniPrefill

---

## 为什么 Prefill 加速这块还没收敛？

LLM 服务有两段：prefill（处理输入提示词）和 decode（自回归生成）。长上下文场景下，**prefill 的 cost 完全压过 decode**——你给模型扔 100K 的文档，prefill 阶段要把所有 token 跑一遍 Self-Attention，标准 Softmax 是 $O(N^2)$ 复杂度，128K 输入的 attention map 就是 16B 个元素，光算这个就够受的。

业界这两年解这个问题，走的是两条路：

**路一：换架构**。把部分 Attention 层换成 Linear Attention 或 Sliding Window Attention。Qwen3-Next、Gemma-3、MiniMax-01 都是这个思路，复杂度从 $O(N^2)$ 降到 $O(N)$ 或常数。

**路二：改算法**。保留架构不动，在 Attention 里识别稀疏 pattern，跳过那些 attention score 接近零的位置。MInference、FlexPrefill、XAttention 这一类。

问题来了：这两条路其实**互相打架**。

你想想看，稀疏注意力的加速比，来自于**"省下来的 attention 计算 / 总 prefill 计算"**。Full Attention 模型里，attention 占大头，所以 MInference 能在 128K 上跑出 10× 加速。但混合架构呢？以 Qwen3-Next 为例，**Linear / Full = 3:1**，也就是说每 4 层里只有 1 层是 Full Attention，那稀疏注意力最多只能加速这 1/4——剩下 3/4 的 Linear Attention + FFN 你动不了，整体加速比立刻被 Amdahl 定律拉爆。

论文里给的 Table 1 把这点讲得非常清楚：

![图1：UniPrefill vs Standard Prefill 在三种架构下的 prefill 吞吐量对比](https://arxiv.org/html/2605.06221v1/x1.png)

*图1：三种架构、四种 batch size、六种上下文长度下，UniPrefill（斜纹）与 Standard Prefill（实色）的吞吐对比。可以看到上下文越长、并发越高，UniPrefill 的优势越大——这正是生产场景的痛点所在。*

Qwen3-Next 上，MInference 在 128K 的加速比只有 **1.05×**——几乎等于没加速。但 Full Attention 的 LLaMA-3.1-8B 上，sliminfer/lazyllm 这类 token-level 方法能跑到 2× 以上，可这些方法又有**精度暴跌**的问题（LazyLLM 在 128K 上 RULER 从 76.89 掉到 49.71，掉了 27 个点）。

所以现状就是：**精度能保留的（稀疏注意力）在混合架构上不加速，加速猛的（lazy token drop）精度崩溃**。UniPrefill 想做的是兼顾两边。

还有一个被很多论文忽略的问题——**continuous batching 兼容性**。vLLM 的核心调度策略是把不同请求的 token 打包在一起跑（packed tensor），请求随时进、随时出。FlexPrefill 这类方法是按 per-request 的方式设计的，假设 batch 静态，根本塞不进 vLLM。这就是为什么很多 prefill 加速论文最后只能停留在 "我们在 HuggingFace Transformers 上跑了一下"，离生产环境差着十万八千里。

---

## 核心思路：扔 token，不要省 attention

UniPrefill 的核心 idea 一句话能讲清楚：**在每个 Full Attention 层上判断哪些 token 不重要，扔掉，然后这些 token 在后续所有层（包括 Linear、FFN、Sliding Window）都不再参与计算。**

这个 idea 听上去和 LazyLLM、SlimInfer 很像——它们也是扔 token 啊？区别在哪？

区别在**"扔的依据"和"扔的范围"**。LazyLLM 是基于 attention score 的简单阈值，扔得粗糙，精度掉。UniPrefill 用了一个更克制的机制：**Top-p 选择 + block-wise 粒度 + attention sinks 保护**。

下面这张总览图把整个流程串了起来：

![图2：UniPrefill 总览。左侧是算法部分，右侧是 vLLM 工程集成](https://arxiv.org/html/2605.06221v1/figs/main.png)

*图2：左半部分（1→2→3）是算法核心——用最后 n 个 query 做 block-level importance 估计，按 Top-p 累积阈值选择保留的 block，把这个 mask 传播给后续所有 sublayer。右半部分（4→5→6→7）是工程集成——融合 kernel、KV cache 的 block table 更新、per-layer 序列长度跟踪、tensor-parallel 元数据同步。这张图基本上就是这篇论文的全部信息。*

### Token 重要性怎么估？

直觉是这样的：**对 next-token prediction 真正有用的，是最后那几个 query 位置看到的 attention**。下一个 token 的预测只依赖 $\mathbf{h}_N^{(L)}$，所以位置 $i$ 的重要性可以用 $\mathbf{A}^{(b)}_{N,i}$ 来衡量——也就是最后一个 query 对 key $i$ 的 attention 权重。如果这个值接近 0，那这个 token 就是冗余的。

为了减少估计方差，作者不只看最后 1 个 query，而是聚合最后 $n=128$ 个 query：

$$s_i^{(b)} = \frac{1}{n}\sum_{j=N-n+1}^{N} \mathbf{A}^{(b)}_{j,i}$$

注意这里只算 $n \times N$ 的 attention（128 × 128K），不算完整的 $N \times N$，开销可以忽略不计。

然后把 token 分成大小为 $G=64$ 的 block，按 block 聚合 score：

$$\bar{s}_g^{(b)} = \frac{1}{G}\sum_{i \in \mathcal{B}_g} s_i^{(b)}$$

这样选择决策的数量从 $N$ 个降到 $\lceil N/G \rceil$ 个，效率提升 64 倍。

### 为什么是 Top-p 而不是 Top-k？

这个细节挺关键。Top-k 是固定数量——不管 attention 分布是集中还是发散，都保留 k 个 block。但 attention 分布的形状是动态的：retrieval 任务可能只有几个关键位置 attention 集中爆表，summarization 任务可能 attention 比较弥散。Top-k 在这两种场景下都不合适。

Top-p 是按累积概率切：

$$\mathcal{S}^{(b)} = \{\pi(1), \ldots, \pi(k^*)\}, \quad k^* = \min k \text{ s.t. } \frac{\sum_{j=1}^k \bar{s}_{\pi(j)}^{(b)}}{\sum_g \bar{s}_g^{(b)}} \geq p$$

设 $p = 0.99$，意思是**保留累积 attention mass 占 99% 的 token，扔掉剩下 1%**。这给了一个明确的误差界：

$$\left\|\Delta \mathbf{h}_j^{(b,1)}\right\| \leq (1-p) \cdot V_{\max}^{(b)}$$

简单说就是：**扔掉的 attention 质量上限是 1%，扰动有保证**。

还有两个永远保留的 token 集合：**前 128 个 token（attention sinks）**和**最后 n 个 query 自己**。Attention sinks 是 StreamingLLM 那篇论文发现的现象——很多模型在前几个 token 上堆积大量 attention，扔了它们整个 attention pattern 就崩了。这是个工程上必须处理的细节。

### 稀疏性怎么传播？

这是 UniPrefill 区别于稀疏注意力方法的关键。一旦在 Full Attention 层决定扔掉 token 集合 $\bar{\mathcal{S}}^{(b)}$，**这些 token 在后续所有 sublayer（FFN、Linear Attention、Sliding Window）都不再参与计算**：

$$\mathbf{H}_\mathcal{S}^{(b,m+1)} = f_m(\mathbf{H}_\mathcal{S}^{(b,m)}), \quad m=1,\ldots,M_b$$

到下一个 block 开始时，被扔掉的 token 状态被原样"复活"，重新做重要性估计。

这个机制带来的 FLOPs 节省非常可观。论文里给了一个对比公式：

$$\frac{\Delta \text{FLOPs}_{\text{UniPrefill}}}{\Delta \text{FLOPs}_{\text{SparseAttn}}} = \frac{(L-\ell_1)\cdot Nd^2}{N^2 d_k} \xrightarrow{N\to\infty} \infty$$

注意这个比值随 $N$ 增大趋于无穷——意思是**上下文越长，UniPrefill 相对稀疏注意力的优势越大**。原因很直观：稀疏注意力省的是 attention 那部分 FLOPs，而 UniPrefill 省的是后续所有 layer 的 GEMM FLOPs，token 数减少多少，整个网络的计算量就降多少。

---

## 工程实现：怎么塞进 vLLM？

这一块我觉得是这篇论文真正的硬骨头。算法 idea 不复杂，但要让它在 vLLM 里跑起来，要解决一堆细节。

### Fused Kernel

整个 importance estimation + top-p 选择被实现成 4 个融合 Triton kernel：

1. **Partial GEMM**：算 $\mathbf{S} = \mathbf{Q}_{[N-n:N]}\mathbf{K}^\top$，只算最后 n 个 query 行
2. **Online Softmax**：在 N 维度上做两遍 stable softmax，得到每个 token 的 importance score
3. **Block Reduce**：把 token-level score 按 block 聚合
4. **Top-p Selection**：这个是最骚的——直接在 GPU 上排序+阈值，不走 CPU

Top-p kernel 用了一个 IEEE-754 bitcast 的技巧：把 (score, index) 对编码成一个 int64 word，对 word 排序就等价于按 score 排序。这样可以用 GPU 的 radix sort 高效完成，避免了 host-device 来回。

### vLLM 集成的核心问题

vLLM 的 continuous batching 把多个请求的 token 打包成一个 packed tensor，用 `cu_seqlens` 数组标记每个请求的边界。UniPrefill 要在这种 packed 表示上工作，需要：

1. **Per-request 的 token drop**：每个请求的保留 token 数不一样，要更新 `query_start_loc`、`seq_lens`、`num_actual_tokens`
2. **KV cache slot mapping**：drop 之后，每层的 KV cache 写入位置要重新计算。Gemma-3 这种 Sliding Window/Full 混合架构，每层的 block table 还不一样
3. **Decode 阶段的 per-layer seq length 校正**：每个 layer 看到的 KV 长度是它自己写入时的长度（可能因为之前的 drop 而短一些）加上后续 decode 增量

最后这一点是 sparse attention 方法过不去的坎——它们在 prefill 时算完就完了，KV cache 写入是按原长度写。但 UniPrefill 是真的把 token 从计算流里拿掉，KV 也没写，decode 时必须知道每层"实际有多长"。

论文给了一个清晰的公式：

$$\text{seqused}_r^{(\ell')} = s_r^{(\ell^-)} + \Delta_r$$

其中 $\ell^- = \max\{\ell_k \in \mathcal{L}_{\text{drop}}: \ell_k < \ell'\}$ 是 layer $\ell'$ 之前最近的一次 drop，$\Delta_r$ 是 decode 时增量。这个机制保证了每层 attention 操作时看到的 KV 长度和它真实写入的一致——**不改 model weight，不改 PagedAttention**。

工程上这套东西能跑通，我觉得是这篇论文最值钱的部分。

---

## 实验结果：精度几乎不掉，速度可观提升

### 主表：精度 vs 速度

| 模型 | 方法 | RULER Avg | 128K TTFT 加速 |
|------|------|-----------|----------------|
| LLaMA-3.1-8B | Baseline | 90.36 | 1.00× |
| LLaMA-3.1-8B | LazyLLM | 68.50 | 2.51× |
| LLaMA-3.1-8B | MInference | 90.68 | 1.34× |
| LLaMA-3.1-8B | FlexPrefill | 89.62 | 1.46× |
| LLaMA-3.1-8B | **UniPrefill** | **90.45** | **2.26×** |
| Qwen3-Next-80B-A3B | Baseline | 94.76 | 1.00× |
| Qwen3-Next-80B-A3B | MInference | 94.31 | 1.05× |
| Qwen3-Next-80B-A3B | **UniPrefill** | **93.94** | **1.68×** |
| Gemma-3-12B | Baseline | 79.99 | 1.00× |
| Gemma-3-12B | MInference | 79.25 | 1.03× |
| Gemma-3-12B | **UniPrefill** | **78.87** | **1.49×** |

读这张表的几个观察：

**第一**，LazyLLM/SlimInfer 这类朴素 token drop 方法，速度确实快（2.51×），但精度暴跌——LLaMA 上 RULER 从 90 掉到 68。说明粗暴扔 token 真的会出问题，UniPrefill 的 Top-p + attention sink 保护是有意义的。

**第二**，MInference 等稀疏注意力方法在 Full Attention 模型上还能打（1.34× / 1.46×），但到 Qwen3-Next 这种 3:1 混合架构上，加速比塌成 1.05×。这恰好印证了论文动机里讲的 Amdahl 限制——稀疏注意力只能加速少数 Full Attention 层。

**第三**，UniPrefill 在三种架构上都达到 1.49× ~ 2.26× 的加速，精度损失控制在 1 点以内（Qwen3-Next 从 94.76 → 93.94 掉了 0.82）。这个 tradeoff 是这一类方法里目前最好的。

### vLLM 吞吐：concurrent 越多优势越大

| 模型 | BSZ=1, 128K | BSZ=16, 128K |
|------|-------------|---------------|
| LLaMA-3.1-8B | +107% | +109% |
| Qwen3-Next-80B-A3B | +48% | +68% |
| Gemma-3-12B | +42% | +42% |

LLaMA 上 1 个请求就能跑到 +107%（基本 2.07×），到 16 并发还是 +109%。Qwen3-Next 上单请求 +48%，并发上去后涨到 +68%——并发越高，调度收益越大。这个 scaling 性质对生产场景很关键。

### 消融实验

**Block size G**：默认 64 是个 sweet spot。G=128 在短上下文（4K-16K）下加速更好（选择 overhead 低），但长上下文下 G=32 加速更猛（粒度细，能扔的 token 更多）。LLaMA 128K 上，G=32 跑到 +121%，但短上下文劣势明显。

**Last n**：默认 128。n=32 时方差太大，RULER 掉 2.7 点；n=512 精度略升但 overhead 增加。

---

## 我的判断：这篇论文值不值得读？

先说结论：**算法没有惊艳，但工程价值很高，值得花时间读**。

**亮点**：
1. **架构无关**是真的——三种风格迥异的架构（纯 Full、Linear/Full、SW/Full）上都跑出了显著加速。这在 prefill 加速这块挺难得，大部分论文都只在 LLaMA 上 demo
2. **vLLM 集成**是真的能跑——大部分稀疏注意力论文只在 HuggingFace Transformers 上做实验，离生产服务差着大半截。这篇直接把 continuous batching、TP=8、KV cache 的细节都处理了
3. **Top-p + block-wise + attention sink** 是个干净的组合，提供了明确的误差界

**问题**：
1. **算法 novelty 偏弱**。Top-p selection、attention sink 都是已有的 idea，"propagate sparsity 给后续所有 layer" 这个机制确实和 SnapKV 之类不同，但更多是"工程整合"层面的创新
2. **缺更强 baseline 的对比**。LazyLLM、SlimInfer 这类粗暴 token drop 方法被吊打很正常，但 SnapKV、H2O 这些更精细的 KV 压缩方法，作者只口头提了，没做完整的实验对比
3. **混合架构的"代表性"略弱**。Qwen3-Next 是 3:1，Gemma-3 是 5:1。如果是更激进的 7:1 或 9:1 混合呢？UniPrefill 的加速比应该会更猛，但 baseline 也会更难加速，作者没探这个边界

**特别想吐槽一点**：Table 1 的 TTFT 加速是在 HuggingFace 上测的，而 Table 2 的 throughput 是在 vLLM 上测的——这两套数对应不上。比如 Llama 128K BSZ=1 在 HF 上是 2.26×，在 vLLM 上是 +107%（约 2.07×）。差距不大但不一致，论文里没解释为什么要分两套测。我猜是因为部分 baseline（FlexPrefill 等）没法塞进 vLLM 才走了 HF。

**对工程实践的启发**：
- 如果你在做长上下文 serving，且用的是 Qwen3-Next 这类混合架构，那么稀疏注意力方法基本无用，token-level drop 是更靠谱的方向
- Continuous batching 兼容性是核心。下次评估一个 prefill 加速方法时，**第一个问题不应该是"加速多少"，而是"能不能塞进 vLLM"**
- Top-p + attention sink + block granularity 这套组合应该会成为 prefill 加速的标准范式

---

## 收尾

如果你也在做长上下文的 inference serving，这篇文章的工程经验值得参考。但如果你期待的是"颠覆性的算法创新"，那它给不了——它更像是一篇**把已有 idea 工程化做扎实**的论文，价值在落地能力，不在 novelty。

回头看长上下文加速这个赛道，从最早的 StreamingLLM 给出 attention sink 现象，到 SnapKV、H2O 做 KV 压缩，到 MInference 做稀疏 pattern，再到 UniPrefill 把 token-level 加速和 continuous batching 打通——整个方向其实在收敛到一个共识：**真正能上线的 prefill 加速，必须是架构无关 + 调度兼容的**。这篇论文在这个方向上又往前推了半步。

下一个值得追的问题大概是：**能不能把 prefill 阶段的 drop 决策迁移到 decode？** 一个 token 在 prefill 时被认为冗余，decode 阶段是不是也不用它的 KV？这个方向论文也提到了，作为 future work。我估计明年这块还会出一波新工作。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
