# Attention 自己就是检索器：NVIDIA 把外挂 retriever 拆了，多跳 QA 反而更强

> 论文：Retrieval from Within: An Intrinsic Capability of Attention-Based Models
> 作者：Elad Hoffer, Yochai Blau, Edan Kinderman, Ron Banner, Daniel Soudry, Boris Ginsburg
> 机构：NVIDIA；Technion 电子工程系
> 链接：https://arxiv.org/abs/2605.05806

---

## 先说一个我自己踩过的坑

之前接手过一个 RAG 项目，问题非常典型：embedding 检索器（用的是 BGE-large）召回的 top-5，离线 NDCG 看着挺漂亮，但接到下游 LLM 里答案质量就是上不去。换 Qwen3-Embedding-4B、上 reranker、调 chunk size 都试过，HotPotQA 这种多跳任务始终卡在 41 EM 左右。

后来做了个很笨的实验：把 oracle 证据（人工标注的支持段落）直接喂给 LLM，EM 立刻飙到接近上限。也就是说，**模型本身能答对，是检索器没把对的东西捞回来**。

这件事让我一直怀疑——**检索器和生成器分开训练、各用各的表示空间，这个范式真的合理吗**？检索器认为"相关"的，生成器未必觉得"有用"。两边的损失函数不对齐，中间用 top-k 这个硬切口连接，本来就是一种妥协。

NVIDIA 这篇论文给我的感觉是，他们把这个怀疑用一个干净的方式落地了。结论非常直接：**别再外挂 retriever 了，decoder 的 cross-attention 自己就在做检索，把它显式拽出来用就行**。

---

## 核心摘要

这篇论文提出 **INTRA**（INTrinsic Retrieval via Attention），一个让 encoder-decoder 模型用自身的 cross-attention 直接做检索的框架。整个思路就一句话：decoder 在生成时本来就要对 encoded chunks 做注意力打分，那干脆把这个分数显式提出来当检索分用，根本不需要外挂的 embedding 模型。

工程上很轻：encoder 和 decoder 整个冻住，只训练 R 个可学习的 retrieval token（约 16.4 万参数）和 L 个层聚合权重 alpha（272 个标量），加起来连 17 万都不到。在 T5Gemma2 4B 上跑出来，多跳 QA 平均 EM 达到 40.2，超过 Qwen3-Emb-4B 加 reranker 的 39.2；HotPotQA 上 EM 46.4，把所有 baseline 都比下去了。更关键的是，它跑出了一个真正成立的论点：**检索和生成共享同一套表示，比堆参数更重要**——同一个 4B 模型既检索又生成，gap closure 比 INTRA 检索结果丢给 27B 模型生成还高。

但有个前提，这套东西目前只在 encoder-decoder 架构上验证过，主流 decoder-only 大模型怎么迁移，论文没给答案。

---

## 问题动机：RAG 的"两张皮"到底有多伤

做 RAG 系统的人多多少少都体会过这种割裂感。我把它拆成三层：

**第一层：表示空间割裂。** 检索器（BGE、E5、Qwen3-Embedding 这类）训练目标是"query 和 document 的相似度"，而 LLM 生成的目标是"给定上下文输出答案"。两者的最优表示空间不一样。检索器学会的"相关性"和生成器需要的"可用性"经常错位。

**第二层：重复计算。** 检索回来的文档要再塞进 LLM context，做一次完整的 prefill。chunk 越长、检索回来越多，prefill 越贵。这是工程上最直观的浪费——同一段文本，检索阶段编码过一次，生成阶段又编码一次。

**第三层：多跳推理失灵。** 多跳问题需要把分散在不同段落的证据"拼"起来。但 query-document 相似度说到底是单跳信号——它衡量"这段文本和问题有多像"，而多跳推理需要的是"这段文本能不能作为推理链的一环"。这俩根本不是一个东西。

论文的切入点非常优雅：**cross-attention 本来就是 query 条件下对候选状态加权选择的机制，这跟"检索"在数学结构上是同一件事**。既然 decoder 在生成时已经在做这件事了，为什么不让这个内在能力直接对外服务？

---

## 方法：INTRA 是怎么把内在检索能力激发出来的

### 一图看懂 INTRA 和传统 RAG 的区别

![INTRA 架构对比图](https://arxiv.org/html/2605.05806v2/x1.png)

*图1：左边是传统 RAG，外置 retriever 负责选文档，decoder 拿到选出来的文本后再编码一次。两者表示空间不同。右边是 INTRA：encoder 一次性把 corpus 编码好缓存，decoder 用自己的 cross-attention queries 直接给 chunks 打分，选出来的预编码状态又直接作为生成上下文。检索和生成共享一个 representation space，没有外置 retriever。*

这张图把核心差异讲得很清楚。注意右边 decoder 旁边那一列 retrieval tokens（rho_1 到 rho_R），这是 INTRA 唯一可训练的部分——它们是被插在问题后面的"探针 token"，专门用来在 cross-attention 里激发出对检索有用的 query 状态。

### Step 1: 离线把 corpus 全部预编码

把整个文档语料切成固定大小的 chunk，用 encoder 提前编码好，缓存为 K = encoded chunks 的集合。这个步骤一次完成，之后所有 query 共用。

实验里他们用了大约 1 亿 token、75.9 万个 chunk 的池子。这部分跟传统 RAG 的离线索引阶段对应，只是这里存的是 encoder hidden states，不是单独训练的 embedding 模型的输出。

### Step 2: 检索阶段——让 decoder 自己挑证据

这是 INTRA 的灵魂部分。给定问题 x，构造一个增强输入：

$$x_{\text{retrieval}} = [x_1, \dots, x_{L_q}, \rho_1, \dots, \rho_R]$$

也就是把 R 个可学习的 retrieval token rho 接在问题后面。然后让这个增强输入过一遍 decoder，**但不生成答案**，而是把每一层 cross-attention 的 query 状态 q_l 都暴露出来。

对每个候选 chunk k_i，打分公式是：

$$s_i = \sum_\ell \alpha_\ell \cdot \mathrm{MaxSim}(q_\ell, k_i)$$

MaxSim 是从 ColBERT 那边借来的 late interaction 操作：对 query 序列中每一个 token，找到与 chunk 中**最匹配的那个 token** 的点积，全部加起来。这比单向量内积更细粒度，能捕捉到"问题里某个关键词刚好和文档里某个关键词对上"的局部信号。

层聚合权重 alpha_l 是可学习的——不同 decoder 层捕获的语义粒度不一样，底层偏向词汇匹配，高层偏向语义推理，**让模型自己学每一层应该贡献多少检索信号**。这个细节比我预期得更聪明，比简单选某一层或者无脑均权要强。

### Step 3: 生成阶段——直接复用预编码状态

选出 top-n 的 chunks 后，把它们的预编码 K 直接作为 cross-attention 的 memory，让 decoder 正常生成答案。注意——**这里完全不需要重新编码**，因为检索阶段和生成阶段用的是同一套 encoded 表示，天然兼容。

到这一步已经能看出来这个设计的优雅：传统 RAG 里"检索→喂文本→重新 prefill"这个流程，在 INTRA 里压缩成"检索→直接用 K"。Prefill 那部分的大头省掉了。

### 一个魔鬼细节：Reverse-QWK 投影

但还有个工程上的硬骨头：标准 Transformer 里，每一层 cross-attention 都有自己的 key 投影矩阵 W_K_l 和归一化 scale gamma_K_l。同一段文本在不同层会被投影成不同的 K。如果直接照搬，要么每层存一份 encoded K（存储爆炸），要么每次检索都要算一遍每层的 K（计算爆炸）。

作者提了个叫 **Reverse-QWK** 的 trick：把 layer-specific 的变换从 key 侧搬到 query 侧。

$$\widetilde{q}_\ell = (q_\ell W_{K,\ell}^\top) \odot \gamma_{K,\ell}$$

$$\bar{K}(\mathcal{S}) = \mathrm{RMSNorm}(K(\mathcal{S}))$$

数学上完全等价（内积是对称的，把变换放在哪边都行），工程上只需要存**一份归一化后的 encoder 表示** K_bar，所有层共享。检索的存储和计算开销都成可控的了。这种把"层异质性"变成"query 端处理"的思路，我觉得是这篇论文最有工程价值的发现之一。

### 训练目标：轻得离谱

冻住 encoder 和 decoder，只训练 retrieval token rho 和 层权重 alpha。损失函数是 soft cross-entropy，把概率质量均匀分给所有 oracle chunks：

$$\mathcal{L}_{\text{retrieval}} = -\frac{1}{|\mathcal{O}(x)|} \sum_{j \in \mathcal{O}(x)} \log(\mathrm{softmax}(s)_j)$$

总共训练参数 16.4 万 + 272 ≈ 16.4 万。对比 Qwen3-Embedding-4B 的 40 亿参数，这个 ROI 高得有点离谱。

---

## 实验：真打得过专业检索器吗

### 主结果：四个 QA 基准上的端到端 EM 和 F1

实验用 T5Gemma2 4B-4B 作为基模，在 HotPotQA、2WikiMultihopQA、MuSiQue 和 Natural Questions 上评测。对照组很扎实：从最朴素的 TF-IDF、BM25，到 BGE、Qwen3-Embedding 不同尺寸，再到 Hybrid RAG 和带 reranker 的强 pipeline，一共九个 baseline。

| 检索方法 | HotPotQA EM | 2Wiki EM | MuSiQue EM | NQ EM | 平均 EM | 平均 F1 |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF | 34.2 | 39.0 | 5.3 | 34.9 | 28.4 | 36.2 |
| BM25 | 40.5 | 41.7 | 7.7 | 43.4 | 33.3 | 41.5 |
| MaxSim 单独 | 40.7 | 41.6 | 10.1 | 48.4 | 35.2 | 43.9 |
| Hybrid RAG | 43.4 | 46.0 | 10.6 | 50.5 | 37.6 | 45.8 |
| BGE-large | 41.9 | 46.1 | 10.8 | 52.2 | 37.8 | 45.9 |
| Qwen3-Emb-0.6B | 37.0 | 45.7 | 11.1 | 36.4 | 32.6 | 40.5 |
| Qwen3-Emb-4B | 40.3 | 46.0 | 12.7 | 54.5 | 38.4 | 46.7 |
| Qwen3-Emb-4B + Reranker | 41.6 | 46.8 | 13.3 | **55.1** | 39.2 | 47.5 |
| **INTRA** | **46.4** | **49.2** | **14.0** | 51.2 | **40.2** | **48.6** |

我读这个表的时候有几个想说的：

**第一，多跳任务上的优势是真的能打**。HotPotQA 相比 Qwen3-Emb-4B + Reranker 涨了 4.8 个 EM，2Wiki 涨 2.4，MuSiQue 涨 0.7。MuSiQue 这种本来基线就只有 13 分的硬骨头任务，加 0.7 看着不多但相对幅度很可观。多跳推理需要把分散证据拼起来，decoder 的 cross-attention 本来就在做"综合多段信息"这件事，让它去做检索，方向天然对路。

**第二，单跳上 INTRA 输给了 Qwen3-Emb-4B + Reranker**。NQ 上 51.2 vs 55.1，差了快 4 个点。这个我觉得很正常——NQ 大部分问题是"找到那一段就行"，专门为检索训练的大规模 embedding 模型在这种场景下有它的优势，而且 Qwen3-Embedding 的训练数据本来就包含 NQ 的监督。INTRA 的多跳优势来自"decoder 知道生成需要什么"，单跳问题里这种 advantage 弱化了，反而 Qwen3-Embedding 那种针对单跳大量监督的检索器更对路。

**第三个有点意思的对比**：MaxSim 单独这一行（不带 INTRA 的可学习参数，纯用 encoder 输出做 late interaction）平均 EM 是 35.2，INTRA 是 40.2，差了 5 个点。也就是说，可学习的 retrieval token rho 和 层权重 alpha 加起来贡献了大约 5 个点的提升。区区 16 万参数能撬动 5 个点，确实漂亮。

### Complete-Evidence Recall：多跳上的碾压更明显

光看 EM 不够，因为 EM 受生成器影响。论文专门用了一个叫 **complete-evidence recall** 的指标——所有 oracle 证据 chunk 是否**全部**被检索到，这是个非常严格的指标，漏一个就算失败，特别适合衡量多跳检索的真实能力。

![Complete-evidence recall 对比](https://arxiv.org/html/2605.05806v2/x2.png)

*图2：在 k=5/10/20 三个不同召回数量下的 complete-evidence recall。蓝色 INTRA 在 HotPotQA、2Wiki、MuSiQue 三个多跳数据集上几乎是断崖式领先；NQ 这种单跳任务上和 BGE、Qwen3-Embedding 拉开的差距就小很多。*

这张图比 EM 表更说明问题。HotPotQA k=20 时 INTRA 是 76% 左右，Qwen3-Emb-4B + Reranker 大概 66%，差距 10 个点。**复杂多跳任务上 recall 一旦拉开，下游生成 EM 自然跟着涨**——所以前面那个 EM 表的优势其实根源就在这。

### 一个我特别想拿出来讲的实验：Gap Closure

这个实验设计我觉得是全文最有说服力的一部分，专门用来回答"检索和生成共享表示空间到底有没有用"。

定义：Gap closure = INTRA 检索给到该生成器的 EM，相对于"随机检索→该生成器"和"oracle 完美检索→该生成器"两个端点之间，填了多少 gap。这个比例越高，说明检索方法越好地服务了这个生成器。

| 生成器 | HotPotQA | 2Wiki | MuSiQue | NQ | 平均 |
|---|---:|---:|---:|---:|---:|
| Mistral0.3-7B | 50.0 | 21.6 | 24.0 | 56.8 | 38.1 |
| Phi4-3.8B | 55.9 | 18.4 | 18.9 | 68.7 | 40.5 |
| Llama3.1-8B | 61.9 | 38.8 | 23.1 | 71.1 | 48.7 |
| Gemma4-E2B | 63.0 | 47.5 | 25.7 | 74.2 | 52.6 |
| Qwen3.5-9B | 64.0 | 48.0 | 26.1 | **78.5** | 54.1 |
| Qwen3.5-27B | 63.2 | 51.8 | 23.9 | 76.5 | 53.8 |
| **T5Gemma2-4B（同一个模型既检索又生成）** | **66.4** | **56.8** | **38.5** | 75.9 | **59.4** |

划重点：**INTRA 检索 + T5Gemma2 4B 生成（同一个模型）的 gap closure 平均 59.4，比 INTRA 检索 + Qwen3.5-27B 生成（53.8）还高**。Qwen3.5-27B 是 27B 参数，比 4B 大了快 7 倍，但 gap closure 反而输了。

这个结论很重要。它说明：**当检索器和生成器共享表示空间时，检索回来的证据天然对齐生成器的"注意力偏好"，这种对齐价值超过单纯堆生成器参数**。你想想看，与其追求一个超大的生成模型来"消化"乱七八糟的检索结果，不如让检索器知道生成器到底想看什么。

我觉得这是这篇论文最值得带走的洞察。它给的不只是一个新方法，而是一个范式判断。

### 消融：全语料打分 vs 初始池重排

![三种检索策略对比](https://arxiv.org/html/2605.05806v2/figures/recall_s0_sintra_6.png)

*图3：三种策略的 complete-evidence recall@5 对比。S_0 是 encoder-only 的初始检索集（取 top-20），S_0 reranked 是用 decoder 分数对这 20 个重排，S_INTRA 是对全语料做 INTRA 打分。可以看到三个多跳数据集上 S_INTRA 都明显超过 reranked 版本——重排的天花板就是初始池的召回率，而 INTRA 能捞回初始池根本没看到的证据。*

这个消融做得很扎实，专门回答了"INTRA 到底是 reranker 还是检索器"。答案是后者：HotPotQA 上 S_0 召回 36.1%，重排到 47.5%，全语料 INTRA 直接跳到 59.9%——多出来的 12 个点全是"初始池没捞到"的证据。如果只做 reranker，无论排得多准都到不了这里。

---

## 效率分析：把 prefill 那块成本省了

INTRA 在计算成本上有一个结构性的优势：**检索回来的 chunks 不需要再 prefill**。

| 模式 | Pre-Query | Retrieval | Prefilling | Generation |
|---|---|---|---|---|
| 全上下文 | O(1) | O(1) | O((L_q + ML_c)^2) | O(L_g(L_q + ML_c + L_g)) |
| 标准 RAG | O(ML_c^2) | O(sqrt(M) L_q L_c) | O((L_q + kL_c)^2) | O(L_g(L_q + kL_c + L_g)) |
| **INTRA** | O(ML_c^2) | O(sqrt(M) L_q L_c) | O(L_q(L_q + kL_c)) 这一项 | O(L_g(L_q + kL_c + L_g)) |

注意 INTRA 那一行的 prefill 是 O(L_q(L_q + kL_c))，标准 RAG 是 O((L_q + kL_c)^2)。区别就在于 INTRA 不需要把 kL_c 这部分文本重新跑一遍前向。当 k 增大或者 chunk 变长，prefill 就是省得越来越多。

存储成本：把 10 亿 token 的语料用 8-bit 量化存预编码状态，大约 2.5 TB。论文说这"在现代基础设施下完全可接受"。我个人觉得这个 trade-off 要看场景——对企业级知识库这个数字没问题，但跟传统 embedding 检索（每个 chunk 一个 768 维向量）相比，存储开销大了至少两个数量级。论文对这个点的讨论有点轻描淡写。

---

## 我的判断：这篇论文到底值不值得追

### 它做对了什么

**视角自洽且有解释力**。"attention 即检索"不是强行 novelty，而是有清晰的数学结构支撑。cross-attention 本来就是 query 条件下对 key-value 加权选择，跟检索的形式化定义完全一致。论文把这个观察落地成一个具体方法，在 multi-hop QA 上拿到了真实提升。

**工程上精打细算**。冻住主干、只学 16 万参数，Reverse-QWK 解决多层 K 异构的存储问题，pooled chunk embedding 压低 MaxSim 计算成本——这些细节连起来看，能感觉到作者是真的在意"这个方法能不能跑得起来"。

**Gap closure 实验的洞察很重**。"同模型检索+生成 > 异构大模型检索+生成"这个结论，对 RAG 系统设计的指导意义比那张 EM 主表更深远。它意味着检索和生成一体化不只是工程便利，而是有性能上的根本理由。

### 值得追问的几个点

**第一，encoder-decoder 这个前提是硬约束**。当前主流 LLM 几乎全是 decoder-only（GPT、Llama、Qwen、DeepSeek 全是）。论文核心 insight 在 decoder-only 架构上怎么落地，作者明确说留给 future work。这意味着这套东西要进入主流生产链路，还差一个非平凡的扩展。

**第二，2.5 TB 是个不小的数字**。对比传统 embedding 检索（768 维向量每 chunk，大约一两个 GB 量级）大了 3 个数量级。论文说"完全可接受"，但实际工程上需要做 8-bit 量化、特定的 IVF 索引、特殊的存储格式，迁移不是无痛的。

**第三，语料规模受限**。实验用的是 1 亿 token、76 万 chunk 的池子。Web 规模能不能 scale，论文明确说"不主张 INTRA 替代 web-scale RAG"。目前 INTRA 更像企业知识库 / 文档 QA 这类**封闭语料库**场景的方案。

**第四，单跳任务并不占优**。如果你的业务大头是单跳（FAQ、客服检索），那 INTRA 相比成熟的 Qwen3-Embedding 系列没有明显优势，反而带来 2.5 TB 存储负担。

**第五，跟 KV cache 复用类工作的关系没讨论**。最近 decoder-only 圈子里有不少工作在搞 KV cache 跨 query 复用、prefix cache 优化，思路跟"预编码 + 共享表示"是同一方向。论文对这些工作的对比缺失。

### 给工程实践的启示

如果你正在做企业知识库类型的 RAG，特别是涉及多跳推理（财报分析、技术文档 QA、复杂客服）的场景，INTRA 这个方向值得严肃跟。即使你用的是 decoder-only 模型不能直接套，它揭示的"检索-生成一体化"原则也是有用的——比如可以尝试用 LLM 自己生成 query 嵌入（而不是用独立的 embedding 模型），或者用 LLM 中间层做证据打分。

如果你做的是单跳 FAQ 检索之类的简单场景，老老实实用 BGE 或 Qwen3-Embedding 就够了，没必要为了"统一架构"的优雅去多扛 2.5 TB 存储。

---

## 收尾

INTRA 这篇论文最打动我的不是某个具体数字，而是它逼着你重新审视一个被默认了好多年的范式假设：**检索和生成是两件事，所以需要两个模型**。论文用一个具体的实现告诉你——不必如此，attention 本身就是检索的形式化，把它显式拽出来用就行。

当然这把钥匙目前只匹配 encoder-decoder 这把锁。decoder-only 主流模型上能不能复刻这个 insight，是后续最值得关注的方向。如果谁能在 Qwen3 或者 Llama4 这种规模的 decoder-only 模型上证明同样的"内在检索"能力，那才是真正颠覆 RAG 工业格局的时刻。

在那之前，这篇论文至少给我们一个值得放在工程脑子里的判断标准——**做 RAG 系统时，不要默认检索器和生成器应该是分开的两个东西**。它们越对齐，你的系统就越能打。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
