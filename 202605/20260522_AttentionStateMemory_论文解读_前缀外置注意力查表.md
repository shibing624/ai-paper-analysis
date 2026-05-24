# 长前缀的第三条路：把注意力"提前算好存起来"，推理时直接查表

> 在 KV Cache 压缩和 Prompt Distillation 之间，居然还藏着一条没人走过的路。

---

## 核心摘要

如果你做过 in-context learning、RAG 或者 Agent prompt 工程，大概率被同一个问题折磨过——**前缀越来越长**。几千 token 的 in-context 示例、几万 token 的规则手册、十几万 token 的工具描述，每次生成都得让模型重新 attend 一遍，延迟和显存都顶不住。更糟的是，**生成到后面，前缀的影响力会肉眼可见地衰减**——模型一开始还乖乖按指令走，写着写着就开始"放飞自我"。

这篇来自 Institute of Science Tokyo 等机构的论文给了一个我觉得挺漂亮的新解法：**Attention-State Memory（ASM）**。思路简单粗暴——既然 attention 是个数学操作，那我能不能**离线把"前缀对各种 query 的 attention 输出"算好存起来**，推理时直接查表，跳过所有跟前缀相关的 attention 计算？

听起来像 KV Cache 压缩？不对。KV Cache 压缩存的是 key 和 value，推理时还得做点积；ASM 存的是 **attention output 本身**（确切说是 $(a, Z)$ 对），查到就直接用，**完全不再 attend 前缀**。听起来像 Context Distillation？也不对。Distillation 要做梯度反传、改模型权重；ASM 是 **training-free**，只需要前向跑一遍构造字典就行。

效果上：在 LLaMA-3.1-8B 上的 ManyICLBench 评测里，**1K-8K 内存预算下准确率超过 ICL 基线**，8K 时注意力延迟降低 **1.36 倍**；在 NBA 规则检索任务上，**只用 20% 的内存就超过了全注意力 RAG**。

我的判断是：这篇的贡献不是性能上吊打谁，而是**指出了一个长期被忽略的设计空间**——在线 softmax 恒等式其实可以跨 query 复用，FlashAttention 在用、MAC-Attention 在用，但没人把它当成"前缀外置存储"来用。这个角度本身就值。

---

## 论文信息

- **标题**：Context Memorization for Efficient Long Context Generation
- **作者**：Yasuyuki Okoshi, Hao Mark Chen, Guanxi Lu, Hongxiang Fan, Masato Motomura, Daichi Fujiki
- **arXiv**：[2605.18226](https://arxiv.org/abs/2605.18226)（2026 年 5 月 18 日）
- **代码**：https://github.com/yasu0001/AttentionMemory

---

## 一、为什么我们需要"第三条路"

先聊聊长前缀这个老大难问题为什么一直没有完美解。

你想想看，**现代 LLM 应用里，"长前缀"已经是默认配置**：

- ICL 把几十个 few-shot 例子塞进去
- RAG 把检索出来的文档拼在 query 前面
- Agent 把工具说明、历史对话、系统 prompt 全堆在最前面
- Claude Code 这种 coding agent 直接靠 prompt caching 续命，Anthropic 自己都明说了

但前缀长了，就有两个绕不开的代价：

**第一个是 prefix decay（前缀衰减）**。生成进行下去之后，模型的 attention 会逐渐被新生成的 token "稀释"，最早的那些指令、示例、规则就被"遗忘"了。你让 Agent 严格按照某个格式输出，前 100 个 token 还守规矩，到 500 个 token 就开始飘——这就是 prefix decay 在搞事。

**第二个是推理低效**。每个 decode step 都要对整个前缀做 attention，前缀有 32K，就有 32K 的 KV cache 要扫一遍。Prefix caching（比如 vLLM/SGLang 那套）只能 amortize prefill，**decode 阶段每一步还是要 load 整个 KV cache**——这个 memory bandwidth 才是真瓶颈。

现有方案大致分两派，作者用一张图讲得特别清楚：

![图1：长前缀处理三种范式对比 —— 上半部分是典型场景（共享前缀+多用户查询），下半部分对比三种方法：Prefix Compression（保留 attention 但压短）、Prefix Internalization（吃进权重里）、Attention-State Memory（外置成查表结构）](https://arxiv.org/html/2605.18226v1/x1.png)

*图1：三种长前缀处理范式的对比。Prefix Compression 仍要对压缩后的前缀做 attention，复杂度 O(K')；Prefix Internalization 把前缀吃进权重，没有推理开销但要训练；ASM 把前缀外置成 K 个查表项，复杂度 O(log K)，且训练无关。*

### Prefix Compression：压缩了但还得 attend

这一派的代表是 LLMLingua（hard compression，删低信息量 token）、AutoCompressor/Gist Tokens（soft compression，编码成几个连续 token）、KVZip（KV cache 级别的压缩/淘汰）。

思路都是 **让前缀变短，但仍然作为 attention 的输入**。问题在于：

- attention 计算还是 $O(K')$，只是 $K'$ 比原来小
- prefix decay 问题仍然存在——前缀虽然短了，但还是要跟生成的 token 抢 attention 权重
- 压缩本身可能丢关键信息（比如 ICL 里的 label token 一旦被 evict 就完蛋）

### Prefix Internalization：吃进权重但要训练

这一派包括 Context Distillation（Snell 2022、Asawa 2026 Sieve、Zhang 2026 Tsubasa 等）和 Hypernetwork 路线（Charakorn 2025/2026）。思路是 **把前缀的行为 fine-tune 进模型权重里**，推理时就不需要前缀了。

- Context Distillation：对每个前缀微调一次模型，让"不给前缀也能输出和给前缀一样的结果"
- Hypernetwork：训一个超网络，输入是前缀，输出是 LoRA 权重，一次前向就生成参数

听起来很美，但代价巨大：

- 梯度训练贵，对超参敏感
- **前缀一改就得重训**，不适合频繁更新前缀的场景（RAG 检索到的文档每次都不一样啊）
- Hypernetwork 本身的训练就要烧上亿 token

说实话，工业界 RAG 系统每次检索召回的文档都不一样，根本不可能每次都跑一次 distillation。这就是 Internalization 的硬伤。

### ASM 想干的事

作者在表格里把三条路的优劣写得非常清晰：

| 方案 | 前缀 attention 开销 | 训练开销 | 前缀更新 | Prefix Decay |
|------|--------------------|---------|---------|-------------|
| Prefix Compression | $O(K')$，**还是有** | 无（部分方法要训 encoder） | 灵活 | **仍存在** |
| Prefix Internalization | **无** | 高（梯度训练） | 不灵活（需重训） | 无 |
| **ASM（本文）** | $O(\log K)$，**几乎无** | **无**（仅前向） | 灵活（前向重建） | **无**（外置存储不被自注意力稀释） |

三个维度全占。**这就是这篇论文的 selling point——同时解决衰减、效率和训练成本三个问题**。

---

## 二、技术核心：online-softmax 恒等式的跨 query 复用

ASM 的整个数学基础就一个东西——**online-softmax 恒等式**。FlashAttention 用它在单次 attention 计算里分块求和，作者把它推广到**跨 query 的前缀复用**上。

### 一句话讲清楚 online-softmax 恒等式

给你一个 query $q$，一段长度为 $L$ 的 KV（key-value），你想算 attention 输出 $\text{Attn}(q, K, V)$。

如果我把 KV 切成 B 块 $K_1, V_1, K_2, V_2, \dots$，能不能**先在每一块上独立算 attention，最后合起来**？

答案是**能，而且无损**：

$$\text{Attn}(q, K, V) = \sum_{b=1}^{B} \alpha_b \cdot \text{Attn}(q, K_b, V_b), \quad \alpha_b = \frac{Z_b(q)}{\sum_{b'} Z_{b'}(q)}$$

其中 $Z_b(q) = \sum_{k \in K_b} \exp(qk/\sqrt{d_h})$ 是该块的归一化常数（softmax 的分母那一项）。

作者管 $(a_b(q), Z_b(q))$ 这一对叫 **attention state**。这个对子有两个关键性质：

- **Sufficiency（充分性）**：对于固定 query $q$，只要存了 attention state $(a_b(q), Z_b(q))$，**你完全不需要原始的 $K_b, V_b$**，就能恢复这一块对最终 attention 的贡献。
- **Composability（可组合性）**：两个 attention state 可以合并成一个：

$$\text{Merge}((a_A, Z_A), (a_B, Z_B)) = \left(\frac{Z_A a_A + Z_B a_B}{Z_A + Z_B},\; Z_A + Z_B\right)$$

合并后的结果恰好等于在拼接块 $[A, B]$ 上算 attention 的结果。

### 关键洞察：跨 query 复用

到这一步，FlashAttention 也是这么做的——但 FlashAttention 是在 **同一个 query 上跨 block** 应用这个恒等式。

作者的灵感在于：**如果我能预先针对一组"代表性 query"算好 attention state，存起来，那新的 query 来的时候，我找一个最像的代表性 query 的 attention state 拿来用，是不是就近似无损地省掉了前缀 attention？**

这就是 ASM 的核心 idea。把它拆开看：

1. **离线**跑一遍前向，针对一堆"训练 query"算出它们对前缀的 attention state $(a, Z)$
2. 用 k-means 把这些 attention state **聚类**成 $K$ 个 centroid，存成一个"字典"
3. **推理**时，新的 query 进来，先找最近的 centroid，retrieve 它的 $(a, Z)$，然后通过 Merge 操作和"非前缀部分"的自注意力融合

注意第 3 步——**模型在推理时根本不再 attend 前缀**，它只对当前 user query + 已生成 token 做正常的 self-attention，然后把查表得到的"前缀贡献"merge 进去。这就是为什么前缀 attention 开销变成 $O(\log K)$ 的：成本只来自查表，跟前缀长度完全无关。

---

## 三、方法细节：怎么构造，怎么查

![图2：ASM 整体框架。左侧 Offline Calibration（i 存储结构、ii 收集 attention 状态、iii K-means 聚类），右侧 Online Inference（iv 检索最近的 centroid、v 与 self-attention 合并）](https://arxiv.org/html/2605.18226v1/x2.png)

*图2：ASM 的离线构造 + 在线推理流程。离线阶段对一批 (prefix, response trace) 对跑前向，每个 response token 都会产生一个 (query, attention output, normalizer) 三元组；这些三元组按 query 做 k-means 聚类，每层、每 head 独立。在线阶段，新 query 进来按 cosine 相似度找最近的 centroid，retrieve 出来的 attention state 通过 online-softmax merge 进 self-attention。*

### 离线构造（Offline Calibration）

输入：
- **前缀集合** $\mathcal{P} = \{p\}$：可以是 in-context 示例、规则手册、系统 prompt 等
- **Response trace 集合** $\mathcal{T} = \{t\}$：每个 trace 是一对 (user prompt, response)，用来"模拟"将来会问什么样的 query

流程：

**Step 1 - Collection**：对每个 $(p, t)$ 对，把 $[p, t]$ 拼起来跑一次前向。对 response trace 里的每个 token，记录它的 query vector $q$，以及它**对前缀部分** 的 attention state $(a_p(q), Z_p(q))$。这一层每个 token 都会产生一个三元组 $(q, a, Z)$。

**Step 2 - Clustering**：对每层的三元组集合做 k-means，按 query vector 聚成 $K$ 类。每个 cluster 内部用 Merge 操作合并出一个 centroid：

$$\bar{q}_k = \frac{1}{|C_k|}\sum_{q \in C_k} q, \quad \bar{Z}_k = \frac{1}{|C_k|}\sum Z, \quad \bar{a}_k = \frac{\sum Z \cdot a}{\sum Z}$$

注意 $\bar{a}_k$ 是按 $Z$ 加权平均的——这正是 online-softmax 的合并公式（除了 $\bar{Z}_k$ 做了 normalization）。

这里有个工程细节值得提一下：**为啥要把 $\bar{Z}_k$ 除以 $|C_k|$ 做平均，而不是无脑求和？** 作者引用 APE 的工作说，把多个独立编码的 context 不归一化合并，会导致 attention scale mismatch 让性能下降。说人话就是——如果一个 centroid 代表 1000 个 token 的"求和效应"，到推理时和 self-attention merge，新 query 的 self-attention 只代表当前几个 token，**比例完全不对**，merge 出来的结果会被 centroid "压死"。归一化让 centroid 表现得像一个"代表性 token"而不是一个"求和块"。

**Step 3 - Efficient calibration via chunking**：长前缀（比如 32K）一次性跑前向会爆显存，怎么办？利用 Merge 的可组合性，**把 32K 切成 8 个 4K，独立编码后再 Merge**。论文里实测 4 个 4K chunk 的效果跟一次性 16K 几乎一样（见后面消融）。

这里我得说一句——**这个 chunked construction 才是 ASM 真正能落地的关键**。如果非得一次性吃下 32K 才能构造字典，那对 8B 模型来说显存就直接爆了。chunking + Merge 让构造过程的 peak memory 完全可控。

### 在线推理（Online Inference）

新 query 来了：

**Step 1 - Retrieve**：当前 query vector $q$（在某一层、某个 head 上），用 cosine similarity 找最近的 centroid：

$$c^*(q) = \arg\max_k \frac{\langle q, \bar{q}_k \rangle}{\|q\|\|\bar{q}_k\|}$$

**Step 2 - Merge**：拿到 centroid 的 $(\bar{a}, \bar{Z})$ 后，跟 query 对"非前缀部分"的 self-attention $(a(q), Z(q))$ 做 online-softmax merge：

$$a_{\text{merge}}(q) = \frac{Z(q)}{Z(q) + \bar{Z}} a(q) + \frac{\bar{Z}}{Z(q) + \bar{Z}} \bar{a}$$

这个 merged output 就当成最终 attention 输出，进 MLP。

整个过程里，**模型完全没有对前缀做任何 attention 计算**。前缀的影响完全靠查表+merge 体现。

### 关键设计选择：lookup key 怎么选

作者花了不少篇幅讨论 lookup key 用什么 query representation：

- **Pre-RoPE vs RoPE-unified**：是用 RoPE 之前的纯语义 query，还是把所有 query 都"虚拟"到同一个位置上加 RoPE？前者捕捉纯语义相似，后者兼顾语义和位置
- **是否做 whitening**：对 query 做 $q \leftarrow \Sigma^{-1/2} q$ 的白化变换。原因是 query projection 各维度方差不均，cosine similarity 会被高方差维度主导

这 4 种组合在不同任务上的最佳配置不同，作者最后是按任务在验证集上选最优——这块算是工程调参，但不离谱。

### GQA 适配

LLaMA-3.1 是 GQA 模型，$G = H_q / H_{kv}$ 个 query head 共享一个 KV head。作者的处理是：**同一个 GQA group 内的 query head 共享一个 centroid**，因为它们都 attend 同一个 KV，存多份是冗余。

每个 group 形成一个 aggregated query（把组内 head 的 query 拼起来），然后聚类。这样每层每个 KV head 一个字典，存储成本和标准 GQA decode 的 prefix traffic 对得上。

---

## 四、实验：从 ICL 到 RAG 全面跑一遍

实验主战场是两个：**ManyICLBench**（in-context learning，5 个任务）+ **RuleArena NBA bench**（RAG，20K token 规则手册）。模型固定 LLaMA-3.1-8B-Instruct。

Baseline 选得也合理：
- **Full-context** ICL：完整前缀塞进去，性能上界
- **No-context** Zero-shot：只给任务描述，性能下界
- **KVZip**：训练无关的 KV cache 压缩方法，最公平的对照组

### ManyICLBench 主结果

![图3：5 个 ICL 任务上的精度对比 —— ASM（绿）vs ICL（灰）vs KVZip（橙），横轴是 memory entries 数量（也就是字典大小，等价于压缩后的内存占用）](https://arxiv.org/html/2605.18226v1/x3.png)

*图3：ManyICLBench 上的精度对比。ASM 在 1K-8K 内存预算下全面领先 ICL，KVZip 几乎在所有设置都垫底。16K 上 ICL 反超 ASM，主要原因是部分小数据集（比如 bbh_geometric_shapes 只有 150 个训练样本）撑不起 16K 个 cluster。*

几个看点：

**1K-8K 范围 ASM 全线领先**。平均上：ASM 在 1K 是 42.x%，ICL 是 41.x%；ASM 在 4K 是 65%，ICL 是 58%；ASM 在 8K 是 72%，ICL 是 70%。差距虽不算夸张，但**注意 ASM 在 4K 就达到了 ICL 8K 才能到的水准**——这就是关键。前缀长度 4K 的 ASM 字典，等价于一个 ICL 跑 8K 的精度，但 attention 开销只有 $O(\log 4K)$。

**KVZip 全线垫底**。作者的分析挺扎眼："KV cache 压缩会把 ICL 里的 label token 这种关键信号给压没了"。ASM 存的是 attention output——已经聚合过整个前缀的信息——所以**压缩比再高也不会丢关键 token**。

**16K 上 ICL 反超**。作者老实承认这是个问题：小数据集训练样本不足，撑不起 16K 个 k-means cluster。bbh_geometric_shapes 只有 150 个训练样本，硬要分成 16K 类，每个 cluster 平均 0.01 个样本——根本聚不起来。这部分作者写得比较诚实，没有粉饰。

### Reasoning 任务：增益变小

| Entries | math_counting (ICL / ASM) | gpqa_cot (ICL / ASM) |
|---------|--------------------------|---------------------|
| 1K | 23.9 / **24.0** | **28.3** / **28.3** |
| 2K | 22.0 / **23.0** | **28.3** / 24.2 |
| 4K | **25.0** / 22.5 | 27.3 / **27.8** |
| 8K | **26.0** / 24.5 | 24.2 / 23.7 |
| 16K | 22.5 / **26.0** | 22.7 / **23.7** |

两个推理基准上 ASM 和 ICL 基本打平，**没有明显增益**。作者的解释很坦诚：ICL 本身在这俩任务上加 shot 也没怎么涨，ASM 既然是基于 ICL 同样的 attention 信号建出来的字典，**ICL 涨不动它也涨不动**。这个解释我觉得合理——ASM 不是魔法，它是 ICL 的"高效再实现"，ICL 本身不 work 的场景它也救不了。

### RAG（NBA bench）：20% 内存超过全注意力

| Method | # Entries | Accuracy |
|--------|-----------|----------|
| Zero-shot | 0K | 21.2 |
| ICL (full 20K rulebook) | 20K | **24.1** |
| **ASM** | 1K | 19.4 |
| **ASM** | 2K | 19.9 |
| **ASM** | 4K | **25.5** |
| **ASM** | 8K | 23.2 |
| **ASM** | 16K | 19.4 |

这张表是我觉得最 impressive 的一个结果——**ASM 在 4K entries 就达到了 25.5%，比 ICL 用全 20K 规则书还高 1.4 个百分点**。内存只用了 20%。

但有个反直觉的观察：**精度不是 entries 越多越好**。4K 是 sweet spot，再往上加反而下降。作者推测：太多 entries 会导致每个 cluster 太"专一"，新 query 难以匹配到合适的；entries 数应该作为**任务相关超参**，而不是无脑加大。

这个发现挺有意思的。说实话我看到这里第一反应是"会不会是过拟合训练 query 的分布了"——entries 多了之后，每个 centroid 就是某种"非常具体的查询模式"的代表，泛化到新 query 上反而不如更宽泛的 cluster。但论文没深入挖这个角度，算是个 open question。

### 延迟分析

![图4：注意力延迟对比 —— ICL（蓝）vs ASM linear lookup（橙）vs ASM hierarchical lookup（绿斜线）](https://arxiv.org/html/2605.18226v1/x5.png)

*图4：归一化注意力延迟。ICL 随前缀长度线性增长，16K 时延迟最高；ASM linear lookup 在 4K 之后才反超 ICL（小预算时查表本身有开销）；ASM hierarchical lookup 增长更平缓。1.36× speedup at 8K，1.8× at 16K（按作者文本说法）。*

这张图揭示了一个关键事实：**ASM 在小内存预算（1K-2K）下其实比 ICL 更慢**。原因是 linear lookup 本身有 $O(K)$ 的扫描成本，当 K 很小时，prefix attention 也很便宜，两边没差距，反而 lookup 的固定开销占了便宜。

**真正发挥 ASM 优势的地方是大内存预算**：
- 4K entries：ASM linear 开始反超 ICL
- 8K entries：1.36× speedup
- 16K entries：1.8× speedup（hierarchical lookup）

Hierarchical lookup 把 $O(K)$ 降到 $O(\log K)$，做法是**两级聚类**：第一级 centroid 数量少，先扫一遍找 top-m；第二级在 top-m 关联的子 centroid 里 linear search。Heatmap（论文 Figure 5）显示 top-m=16 时 hierarchical 和 linear 精度持平。

### Chunked Calibration 消融

| Chunk size | # Chunks | Total iter | Accuracy |
|-----------|----------|-----------|----------|
| 4K | 1 | 5,000 | 61.5 |
| 16K | 1 | 1,250 | 79.0 |
| **4K** | **4** | **5,000** | **78.5** |

这张表证明了 chunked construction 的有效性：4 个 4K chunk merge 起来（峰值显存只用 4K），效果接近一次性 16K（79.0 vs 78.5，差距可忽略）。这意味着 **8B 模型在消费级 GPU 上就能构造 32K-128K 前缀的字典**——这是 distillation 路线完全做不到的。

---

## 五、我的判断：这篇论文到底怎么样

聊聊我的几点感受。

### 真正的亮点：把 online-softmax 拓展到跨 query

说实话，online-softmax identity 这东西从 2021 年 Rabe 那篇论文出来就在了，FlashAttention 是它最有名的应用，MAC-Attention 也用过。但**所有现有工作都是在"单 query 内、不同 KV block 之间"用它来分块计算**。

把它推广到 **不同 query、共享 KV** 这个方向，让我有点意外。具体说就是——既然 attention state 对固定 query 是 sufficient 的，那我能不能反过来，**把 query 当索引、把 attention state 当 value，建一个查找表**？

这个 framing 是真的漂亮。它把"前缀重复使用"这个问题从"压缩 KV"的视角彻底切换到了"缓存计算结果"的视角。一字之差，思路完全不同：

- 压缩 KV：还是要重新计算，只是计算量小了
- 缓存 attention output：**计算结果直接复用**，不需要重算

这种"换个角度看问题"的思路在论文里其实很难拿到——大部分工作都是在已有 framework 里继续优化。

### 但有几个地方我不那么满意

**第一，泛化性的边界没说清楚**。ASM 本质上是在"训练 query 分布"上做 k-means，然后假设"测试 query 也落在同样分布内"。这在 ICL 任务上还好——训练集和测试集都是同一个 task；在 RAG 任务上就有点悬——用户问题千变万化，字典里的"代表性 query"真能覆盖吗？NBA bench 上 4K entries 是 sweet spot 这个结果，多少暗示了**字典容量和泛化能力之间有 trade-off**，但论文没在更多 RAG 任务上验证。

**第二，"无损"这个词用得有点过**。理论上 online-softmax merge 是无损的，**但前提是 query 一样**。ASM 实际是用 centroid 的 $(a, Z)$ 近似新 query 的 $(a(q), Z(q))$——这一步是 lossy 的。论文里也没仔细分析这种近似带来的误差有多大，只是从下游精度上看"差不多"。我希望看到更细的分析，比如 cluster 大小 vs 误差的曲线。

**第三，对比的 baseline 还可以更狠**。KVZip 是个合理对比，但 prompt compression 那一路（LLMLingua-2、AutoCompressor）至少做个对照吧？作者解释说"prompt compression 是 token 级，KVZip 是 KV 级，跟 ASM 不在一个粒度上"——这个解释勉强能接受，但完整的工作应该把所有可比对象都跑一遍。**和 Context Distillation 路线（比如 Sieve）的对比也只是口头提了一下，没有实测**。

**第四，扩展到更长前缀的可行性**。论文最长试到 32K 前缀。但实际工业场景里，Claude 200K context、GPT-4 128K，前缀动辄过 100K。chunked construction 理论上能 scale，但还没有实测证明 128K 前缀+8K entries 还能 work。

### 工程上能直接用吗

短期内说实话有点远，几个原因：

1. **需要额外的训练数据**（response traces 集合）来做 calibration，对于"一次性查询"场景不划算
2. **对每个固定前缀都要构造一次字典**——如果你的 RAG 每次检索都召回不同文档，这个开销就吃不消
3. **多 turn 对话怎么办？** 论文没讨论。如果对话历史每轮都变，字典每轮都得重建，开销直接拉满

但下面这些场景我觉得 ASM 是真的能落地的：

- **Few-shot Coding Agent**：system prompt + tool description 是固定的，几万 token，正好可以预先构造字典
- **客服机器人/规则引擎**：业务规则相对稳定，更新频率低
- **教科书/法规检索**：内容固定，可以离线构建一次字典反复用

对应的工程切入点是——**先把 ASM 和现有的 prefix caching（vLLM/SGLang）混合部署**。当前缀热度高时，从 prefix cache 升级到 ASM 字典；当前缀冷时，走普通 attention。这种 tier 化的存储其实和 CPU cache hierarchy 思路一脉相承。

### 跟同期工作的位置

把 ASM 放回到整个长上下文加速的大图里看：

| 类别 | 代表 | 是否训练 | 是否 attend 前缀 | 前缀更新 |
|------|------|---------|----------------|---------|
| Prefix Caching | vLLM PrefixCache | 无 | 是 | 灵活 |
| KV Compression | KVZip, H2O, SnapKV | 无 | 是 | 灵活 |
| Prompt Compression | LLMLingua, AutoCompressor | 部分需训 encoder | 是 | 灵活 |
| Context Distillation | Sieve, Tsubasa | 是（梯度） | 否 | 不灵活 |
| Hypernetwork | DOC, TextDoc | 是（大规模训） | 否 | 部分灵活 |
| **ASM（本文）** | - | **无** | **否** | **灵活** |

ASM 占据了"不训练 + 不 attend 前缀 + 前缀可更新"这个**之前没人占的角落**。从这个意义上说，它的定位是清晰的——一个新的设计点，而不是某条已有路线上的微创新。

---

## 六、写在最后

这是一篇我看完之后会去翻代码的论文。倒不是它的精度有多炸——4K entries 在 NBA 上比 ICL 高 1.4 个点这种数字，放任何顶会都不算 striking。**真正打动我的是它揭示了一个新的设计空间**：在线 softmax 恒等式可以用来构建**跨 query 的 attention 缓存**，这件事之前是没人系统讨论过的。

如果你也在做长上下文相关的工程优化，这篇论文我推荐你认真读一遍。即使最终你不用 ASM 本身，**把 attention output 作为可缓存对象** 这个视角对你的系统设计会有启发。比如——在多模态系统里，视觉 token 的 KV 是不是也可以 ASM 化？工具调用的 schema 描述能不能 ASM 化？甚至在 long-CoT 推理里，每一段 thought 的 attention state 能不能在 task 间复用？

我觉得这是一个值得 follow 的方向。

---

**参考文献**

- Okoshi et al. "Context Memorization for Efficient Long Context Generation." arXiv:2605.18226, 2026.
- Rabe & Staats. "Self-attention Does Not Need O(n²) Memory." 2021.
- Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS 2022.
- Ratner et al. "Parallel Context Windows for Large Language Models." 2023.
- Yang et al. "APE: Adaptive Position Encoding for Long Context." 2025.
- Asawa et al. "Sieve: Context Distillation via Selective Forgetting." 2026.
- Zou et al. "ManyICLBench: Benchmarking Long-Context In-Context Learning." 2025.
- Zhou et al. "RuleArena: Benchmark for Rule-Following LLMs." 2025.
- Kim et al. "KVZip: Query-Agnostic KV Cache Compression." 2025.

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
