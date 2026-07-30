# Coding Agent 上下文为啥越用越慢？CodeNib 用一份 Manifest 把"碎一地"的仓库视图焊死

> arXiv 2607.25431 | 2026-07-28 提交

---

## 🎯 核心摘要

如果你搭过 Claude Code / Cursor / Cline 这类 Coding Agent，一定踩过这个坑：每次新会话都要从零开始 grep、读文件、问 LSP、跑 BM25、做嵌入……同一份代码库的"上下文"，每个任务都要重新发现一遍。更麻烦的是，词法索引、稠密索引、LSP 响应、任务历史这几样东西分属不同存储、不同布局、不同失效模式，根本没有一份统一的"仓库视图"。

这篇 CodeNib 的论文（arXiv 2607.25431）把仓库上下文当成一个**数据系统问题**来处理。它做了三件事：

1. **每 commit 编译三份视图**——词法（BM25/Zoekt）、稠密（FAISS L0/L2）、结构（symbol graph），并写一份 manifest 描述这些视图的 commit、状态、能力
2. **LSP 辅助的增量维护**——图用符号级修复，向量用内容寻址嵌入重用，不再每次全量重建。在 8 个仓库 33 个源变更上，匹配的 transition 上图更新中位加速 **8.67×**，向量更新中位加速 **25.44×**
3. **有界上下文服务**——会话只读 manifest、只加载需要的视图，不在线构建；可选的 eager 注入 + 一次性 history 压缩让 trajectory tokens 比配对 grep/read **少 50 到 87 个百分点**

**我的判断**：这不是底层方法突破，是工程范式上的一次认真的"焊接"。论文最值钱的地方不是某个新算法，而是把"视图、查询、agent 接口"这条链路在数据系统层面打通了，并诚实标注了哪些场景下静态视图不能替代实时 LSP、哪些加速比的成立是有前提的。看完整篇，我对"仓库上下文是个数据系统"这个视角的好感又涨了几分。

---

## 📄 论文信息

| 项 | 内容 |
|---|---|
| **标题** | CodeNib: A Multi-View Data System for Serving Repository Context to Coding Agents |
| **作者** | Zhongming Yu, Hengjia Yu, Boqin Yuan, Shuting Zhao, Yizhao Chen, Aryan Dokania, Mihir Jagtap, Jiayu Chang, Yitong Ma, Yash Jayswal, Wentao Ni, Hejia Zhang, Zhaoling Chen, Gangda Deng, Jishen Zhao |
| **机构** | 来自 arXiv 摘要（cs.SE），未在摘要页明确标注单一机构 |
| **日期** | 2026-07-28 |
| **链接** | https://arxiv.org/abs/2607.25431 |
| **分类** | Software Engineering (cs.SE) |

---

## 🤔 问题动机：为什么 Coding Agent 的"上下文"这么碎

打开一份中型代码库（几十万行 Go + Python + Rust），让一个 agent 修个 bug，常见的执行路径是这样的：

1. 模型说："我要找 `Logger.write` 的定义" → 调一次 grep，搜出 30 个候选
2. 模型说："定义应该在哪" → 调一次 LSP 跳转
3. 模型说："我得先理解 import" → 再读 5 个文件
4. 模型说："这个文件变了吗" → 再问 git diff
5. 模型说："我看看相似代码" → 再调一次 embedding search

这些操作背后的事实真相是：

- **词法索引**（BM25/trigram）建一次要几分钟，跟 Git 一起失效
- **稠密索引**（embedding + FAISS）建一次要几小时甚至更久
- **结构图**（symbol graph）需要 SCIP/clangd 编译产物，更新要么全量要么瞎糊弄
- **任务历史**只活在一个进程里，下个任务清零
- **LSP 响应**是 live 的，但需要起一个语言服务进程，且要逐文件 sync

这五样东西各有自己的存储格式、自己的更新路径、自己的输出契约，**没有任何"事务"或者"一致性"机制**。所以"哪些索引的哪些版本对应哪个 commit 的哪个文件"——这个事实在大多数 agent 系统里其实**没人真的管**。每个 issue 都重复一次 model-guided 探索，每个任务特定的观察结果被"污染"到下一轮。

CodeNib 的核心主张是：**别让 agent 每次都重新发现仓库。把仓库上下文当作可复用的派生状态，用数据系统的视角去管它**。

---

## 🏗️ 方法核心：三个视图 + 一份 Manifest + 一条增量维护链

### Figure 1：系统总览

![Figure 1: CodeNib 编译并维护异构仓库视图](https://arxiv.org/html/2607.25431v1/x2.png)

*图 1：CodeNib 的总体框架。顶部是三大挑战对应模块——异构视图（C1）、增量维护（C2）、Agent 交付（C3）；中间是物化仓库视图（按 commit 编译 lexical/dense/structural 三份产物，写 manifest）；底部是 Agent-native 查询执行（Context Serve 路由请求，Context Delivery 提供有界上下文，MCP 工具暴露给 agent）。*

注意右边那一列：**Compile 路径走 Materialize，Diff 路径走 Maintain**。这是 CodeNib 最关键的设计选择——视图构建和视图维护用两套完全不同的代码路径，因为它们的优化目标根本不一样（构建求稳定，维护求精确）。

### 三种视图

每个仓库的每个 commit，会被编译成三份视图：

| 视图 | 内容 | 物理实现 | 何时用 |
|---|---|---|---|
| **词法 V_c^lex** | 标识符、路径、注释、源文本的 posting/trigram 记录 | BM25 + 可选 Zoekt | 路径/标识符检索、正则 |
| **稠密 V_c^dense** | L0 文件或 L2 可调用单元的嵌入，映射回源范围 | FAISS（默认 Flat IP，可配置 IVF/Q2 中消融 HNSW） | 语义相似度搜索 |
| **结构 G_c** | 类型化包含和关系边，连接源链接的文件、作用域、定义，加上持久化的 occurrence 记录 | igraph 后端 pickle | 符号导航、依赖查询、图扩展 |

源单元的统一格式是：

$$u = \langle p, r_s, r_e, \ell, \tau, x, s \rangle$$

- p = 仓库相对路径
- [r_s, r_e] = 源范围（行号区间）
- ℓ ∈ {L0, L1, L2} = 粒度（L0 = 文件，L2 = 可调用定义）
- τ = 节点类型
- x = 源文本
- s = 可选已解析符号

这个格式看起来不起眼，但它是后面所有"统一寻址"的根基——**任何视图的查询结果都能映射回 ⟨p, r_s, r_e, τ⟩ 这个四元组**，所以 agent 拿到的"代码块"在三个视图间是可拼接的。

### Manifest：把"什么视图对应什么 commit"焊死

```
M_c = ⟨c, V_c^lex, V_c^dense, G_c, K_c⟩
```

每个 commit c 对应一份 manifest M_c，它链接三类产物的 profile、状态、能力 K_c。论文特别强调：**manifest 不是视图载荷的容器，是运行时的查找边界**。所以即使 BM25 builder 失败了、向量 builder 成功了，manifest 仍然可以被发出来，BM25 只是不出现。

这个设计有一个很值得说的工程后果：**失败的 optional 视图不会让成功的 sibling 视图失效**。这是很多"全栈索引"系统都做不到的事——它们倾向于"要么全有要么全无"。

### Figure 2：数据流图

![Figure 2: CodeNib 仓库到 agent 的数据流](https://arxiv.org/html/2607.25431v1/x3.png)

*图 2：从 Repo（Code Chunker、Tree-sitter 增量提取）→ Agent Index Compiler（Symbol Index, Vector Index, Trigram Index；Vector Patch, Graph Patch）→ Repo Manifest（graph/vector/lexical index 状态表）→ Agent Index（Indexer skill 注入）→ Agent Context（Context policy: grep/read、eager、Compact）。左侧列出 Libraries（AST、LSP、Embeddings、Reranker、LLM、Graph/Vector/Lexical Index）和 Storage 层；右上角是 Query 入口（Ranked Retrieval, Symbol Navigation）；右下角是 Context Policy（grep/read vs eager+Compact）和 Agent Loop。*

这张图把"谁产什么、谁消费什么"画得特别清楚。注意 **Agent Loop 那个分支**——它明确写了"不开 Indexer skill"，也就是说**运行时不在线构建视图**，只消费已存在的 manifest 链接的视图。

### Figure 3：LSP 辅助的增量图维护

![Figure 3: LSP 辅助增量图维护](https://arxiv.org/html/2607.25431v1/x4.png)

*图 3：六个面板（a-f）展示了符号级增量图维护的全过程。(a) Old Graph：Foo.py 有 print, Logger, bar, main, mypy 等节点；(b) New Graph：相同节点但内部引用边调整；(c) Unified Text (5 hunks)：左 5 个变更片段（`def print(self):` → `def __str__(self):`、删除 `def ` 标签、新增 `def new(self):` 等）；(d) File-level Update：旧/新文本 diff；(e) Symbol-based Update：分类 old(7) → reuse(5)、delete 3、new 4；(f) Reconnect：每条边都按新的 source location 重新连接。文件级需要 9 个 LSP 请求（4 references + 5 definitions），符号级只需 5 个（1 reference + 4 definitions），**请求数从 9 降到 5、减少 44.4 个百分点**。*

这是 CodeNib 最精彩的一张图。它展示了**符号分类器**怎么把变更分类成 deleted / affected / shifted / unchanged / added 五类——其中 **unchanged 的符号连同其后端不可见符号一起保留**（前提是声明行没变），从而省下大量的 LSP 重新解析。

具体流程：
1. 同步变更文本（unified diff hunks）
2. 用 zero-context hunks + 新的 documentSymbol tree 做分类
3. deleted → 删节点和边
4. shifted → 用 position-based definition 重新解析入边
5. added → 提前创建顶点（**关键**：在边修复前就创建所有新顶点，避免重连时跨文件目标不存在）
6. unchanged → 完全保留

这套设计带来的实测收益，后面 Q4 那块会说。

### 排序查询计划（Figure 5）

![Figure 5: 确定性排序查询编译](https://arxiv.org/html/2607.25431v1/x6.png)

*图 5a：输入 signals (lex/sem/struct) + budget b + capabilities c → 确定性 planner → 可执行计划 z=⟨r, k, ρ, h⟩。r 是路由（sparse/dense/hybrid/graph），k 是宽度，ρ 是 reranker，h 是图扩展。*

四种物理路由：

| 路由 | 类型 | 特性 |
|---|---|---|
| A | 词法 | BM25 / 正则 / Zoekt |
| B | 语义 | 稠密向量搜索 |
| C | 混合 | **唯一拥有 RRF（倒数排名融合）的路由** |
| D | 结构 | 图扩展 + 可选 reranking（**无 RRF**） |

**D 路由上禁用 RRF** 是论文的一个有意思的小决定——因为图扩展的输出已经是有结构的，加 RRF 反而会把结构信息糊掉。

实验中用到的稠密-图融合公式（Q1 消融）：

$$\text{score}(u) = \frac{w_d}{\kappa + r_d(u)} + \frac{w_g}{\kappa + r_g(u)}$$

固定 $w_d=1, \kappa=60$，在仓库不相交的调优分区上选择 $w_g$，最终选 $w_g=0.5$。

### Figure 4：Agent 循环和三种上下文策略

![Figure 4: Agent 循环、视图加载和上下文策略](https://arxiv.org/html/2607.25431v1/x5.png)

*图 4：左侧 Session Setup 阶段，从 User Query 到 Harness + Manifest 注册 Session Tool Set（load views、preflight resources、bind skills）；下方是 Agent Loop（Tool Dispatch → LLM → History → Agent result / Trace）；右侧三种策略的 History 对比：Grep/read 一直把观察塞进 history；Eager 把候选 C 注入但不压缩；Compact 一次性把 history 重写成保留 seed d。*

三种上下文策略共享 issue、工具、预算、turn 上限、模型参数、仓库 commit：

1. **Grep/read**：无候选起始，模型引导探索
2. **Eager**：注入冻结的 top-10 L2 代码块 $C_{10}^{ctx}$
3. **Eager+Compact**：相同注入 + 一次性 history 重写

Compact 的重写公式：

$$H_j = [s, q\|C_{10}^{ctx}, e_{1:j}] \rightarrow \hat{H}_j = [s, q\|d_j]$$

其中 $d_j$（"方向种子"）只包含：
- 去重的读取路径列表
- 最新成功读取的全文
- 最新非空助手消息的有界前缀（**只截到 600 字符**）

这个重写是**确定性的**——不调摘要模型，不依赖 LLM 本身的状态。所以同一份 history 无论谁来重写，得到的 $d_j$ 都一样。这一段是论文中我最喜欢的部分，因为它把"上下文压缩"这件事**去神经化**了，避免了"摘要模型可能引入新错误"的风险。

---

## 🧪 实验结果：数据说话

实验覆盖 5 个研究问题，跨 100 个仓库快照、5 个 agent 模型、7,500 条轨迹。规模不算巨大，但**数据点全部冻结**（frozen records），意味着每一项报告都有可重放的来源。

### Q1：稠密检索 + reranker 是一笔划算的买卖吗

![Figure 6: Embedding 和 pointwise reranker 的 operating point](https://arxiv.org/html/2607.25431v1/x8.png)

*图 6：5 个 embedder 族（SR-Small 137M、Qwen3-0.6B、Jina-1.5B、Qwen3-4B、SR-Large 7B）在 L2 可调用粒度上的对比。(a) File-level build time vs Recall@10：L2 构建时间从 26s 到 295s 不等；(b) File-level query time：增加 reranker 后召回升到 0.858-0.886，但查询延迟进入秒级；(c) Symbol-level query time：4B+8B reranker 取得最高符号召回 0.742，代价是 14.1s 查询时间。虚线是 Pareto frontier。*

| Embedder | File Recall@10 (k=10) | Symbol Recall@10 (k=10) | 纯稠密查询时延 |
|---|---|---|---|
| SR-Small (137M) | 0.705 | 0.422 | 26 ms |
| Qwen3-0.6B | 0.71 (估) | 0.44 (估) | ~50 ms |
| Jina-1.5B | 0.78 (估) | 0.49 (估) | ~75 ms |
| Qwen3-4B | 0.81 | 0.554 | 92 ms |
| SR-Large (7B) | 0.82 | 0.638 | 295 ms |

加了 **Jina + Qwen3-4B reranker（k'=50）** 之后，File Recall@10 冲到 **0.858**，但查询延迟变成 **4.29 秒**——是纯稠密的 **46.6 倍**。

**判断**：这 4.6 个百分点的提升换 46 倍延迟值不值？论文态度很诚实——reranker 是"可控的秒级权衡"，可以作为可选 arm 提供，但默认 arm 应当只用稠密。

### Q1.2：图扩展是个真有效的信号吗

![Figure 7: 任务级图和物理 ANN 消融](https://arxiv.org/html/2607.25431v1/x9.png)

*图 7：(a) Graph expansion effect：5 个 embedder 加图扩展（带/不带 reranker）的 File Success@10 变化，效应范围 -4.8 到 +7.1 个百分点，**所有置信区间都跨 0**；(b) ANN fidelity-latency frontier：Flat、HNSW (ve=10)、IVF (25%) 三个物理组织在 top-10 overlap ≥ 0.95 下的延迟曲线；(c) Speedup 随仓库规模变化，HNSW 在大仓库上逐渐胜过 IVF。*

(a) 的结果**很打脸**——加了图扩展，召回的变化范围是 -4.8 到 +7.1 个点，但**所有 embedder 族的置信区间都跨 0**。也就是说**图扩展带来的收益在统计上不显著**。

**但**这个消融本身有价值——它告诉读者：不要把"加了图扩展就能涨点"当成默认假设。在某些 embedding 族上它可能还掉点。

图扩展的代价是**额外 15-39 ms 中位延迟**到非 reranking 查询上，reranker arms 总成本 2.7-6.8s。

### Q2：索引构建和 ANN 物理组织

![Figure 8: 5 个 embedder 族在 100 个快照上的构建时间 vs LOC 和 warm query 延迟](https://arxiv.org/html/2607.25431v1/x10.png)

*图 8：(a) L0 file-level embedding：5 个 embedder 族的 build time vs Total LOC。SR-Large 7B 在 500K LOC 上需要 350s，Jina-1.5B 同样规模只要 ~50s；(b) L2 callable-level embedding：L2 粒度成本高出文件级一个数量级，SR-Large 在 500K LOC 上达 2000s；(c) Warm query latency：violin plot 展示各 embedder 在 SR-Small(13M) 137M、Qwen3-0.6B 768d、Jina-1.5B 1536d、Qwen3-4B 2560d、SR-Large 7B 3584d 维度下的延迟分布。*

从这张图能直接看出：**embedder 维度和 LOC 是两个独立的成本轴**。SR-Large 7B 在 3584 维上查询，p95 已经接近 1 秒。论文中（Q2 消融里）挑选"每族在 mean overlap ≥ 0.95 时最快"配置——这个选择规则比"统一一种 ANN"更稳妥。

### Q3：静态索引能替代实时 LSP 吗

![Figure 9: 100 个快照上静态索引 vs 实时 JSON-RPC 重放](https://arxiv.org/html/2607.25431v1/x11.png)

*图 9：4 种语言（C/C++, Go, Python, Rust, TS/JS）× 2 种请求类型（def / refs）的延迟分布。紫色圆点 CodeNib static，橙色三角 Live JSON-RPC。右列是 median live/static 比：C/C++ def 54.0×、Python refs 57.6×、TS/JS refs 22.1×；最友善的是 Python def 1.6×、Rust def 2.4×。每行还有匹配率（def 80-99%，refs 22-73%）。*

1000 个请求里，静态索引在 **632 个（63%）** 上能复现实时 LSP 的归一化 path/start-line 集合。**在匹配的子集上，中位 live/static 延迟比是 4.7×**（精确 4.72×）。

但这个数字**很依赖语言和请求类型**：

- Python def：1.6×（几乎没差）
- Python refs：57.6×（差得离谱）
- C/C++ def：54.0×（差得离谱）
- TS/JS refs：22.1×

**判断**：这就是为什么论文坚持"接口必须保留实时 JSON-RPC"。静态视图对"按 def 找位置"这种高匹配率请求（85-99%）有 1.6-4.7× 的加速，**值**；但对 refs 类型（C/C++ 只有 22% 匹配），加速没有意义，匹配率本身就不高。论文把这种"不通过全请求相等性测试就不路由"的设计选择写得很明白——它没有被 4.7× 这个平均值冲昏头脑。

### Q4：增量维护的加速比

这是最值钱的两个数字。论文把它们条件化在"匹配独立重建目标的源变更 transitions"上——这意味着**没有和全量重建产出对齐的 transition 不计入加速比**。

| 类型 | 源变更 transitions | 匹配独立重建 | 中位加速 |
|---|---|---|---|
| **图** | 33 | 15/33 (45%) | **8.67×** |
| **向量** | 31 | 28/31 (90%) | **25.44×** |

加速公式：

$$\Gamma_{G,a} = T_f^G / (T_u^{G,a} + T_s^{G,a} / n_{share})$$

$$\Gamma_V = T_f^V / T_u^V$$

其中 $n_{share}=5$（共享一个服务器设置的 transitions 数）。

**判断**：图更新 45% 匹配率其实**不算高**——一半多的源变更，符号级修复出来的图和全量重建不一致，所以加速比不计入。这意味着**符号级修复的鲁棒性还不够**，很多变更下回退到全量重建更安全。

向量这边 90% 匹配率就好多了——内容寻址嵌入重用对"没改的文件"是天然安全的，加速比直接堆到 25×。

**限制声明**（论文写得很清楚）：独立重建和比较**不在维护路径上执行**，所以 8.7× 和 25.4× 是**离线计时**的结果，不能直接当 online latency 来看。

### Q5：Agent 上下文交付的 token 节省

![Figure 11: 5 个 agent 模型在三种策略下的效果](https://arxiv.org/html/2607.25431v1/x13.png)

*图 11：(a) Token Usage：5 个模型（Claude Haiku 4.5、Qwen3.5-9B、Qwen3.5-27B、Gemma 4-12B-IT、Gemini 2.5 Flash）上 Eager 和 Eager+Compact 相对 grep/read 的 token 比例；(b) Localization Quality：AnswerRecall@5 差值 ΔAR@5 vs grep/read；(c) Workload Effects：按问题类型（Bug Fix、Feature、Test Failure、Module Int、Refactor 等）切片。橙色是 Eager，蓝色是 Eager+Compact。*

**核心数字**：5 个 agent 模型中每一个，**满足共同定位边际的最低 token 核心策略**比配对 grep/read **少用 50-87% 的 trajectory tokens**。

注意"满足共同定位边际"这个条件——**质量保持规则是** $LB_{.95}[\Delta AR@5(\pi)] \geq -\epsilon$，$\epsilon=0.05$。也就是说，token 省了，但定位质量下界 95% 置信区间不能比 grep/read 低超过 5 个点。

**判断**：这个 token 节省数字看起来很爽，但要看条件——Eager 把 10 个 L2 候选注入 history 是一笔预付，**只有任务确实需要这些候选时才算赚**。Compact 进一步做 history rewrite 的收益更小但也稳。论文没有把"任务难度"、"候选相关性"做更细的切分，这部分留白。

---

## 🧠 我的判断

### 这篇论文真正卖的是什么

不是新算法，**是工程焊点**。具体来说：

1. **多视图统一寻址**——把 lexical/dense/structural 三种产物的结果都映射到 ⟨p, r_s, r_e, τ⟩，让 agent 拿到的"代码片段"在三个视图间是兼容的
2. **manifest 替代了"全栈索引"假设**——失败的 optional 视图不会让 sibling 失效，这是个朴素但少有人做对的事
3. **确定性上下文压缩**——Compact 不调 LLM 摘要，只用规则做 history rewrite，可重放、可解释

这三件事单看每一件都不新鲜，但合起来形成了一个完整的仓库上下文供应链。这是大部分 agent 系统都还没做到的事。

### 几个我看完觉得"嗯？这里我不太信"的地方

**第一个**：Q1 的图扩展。-4.8 到 +7.1 个点的范围、所有 CI 跨 0——论文作者自己说了 ρ 和 h "需要单独的计划策略"，也就是说图扩展目前还**不是一个明确的赢家**。作为 reader 我会觉得"那为什么还要保留它"——但论文的态度是"先建机制，再证效果"。

**第二个**：Q4 的图更新 45% 匹配率。15/33 的源变更上符号级修复产出的图**和全量重建不等价**，所以加速比不计入。剩下 18 个 transition 上发生了啥？论文没明说，但合理推测是 LSP 重新解析失败、回退到全量重建。这其实是个挺重要的工程回退点，论文没展开有点可惜。

**第三个**：Q5 的"满足共同定位边际"是 95% 置信区间的下界 ≥ -0.05。**这是个操作性的容差，不是严格的 quality preservation**。如果某类 query 的 AR@5 本身就在 0.3-0.4 范围，那 0.05 的下降其实是 15% 的相对退化。论文这里没说 baseline 上不同 query 类型的 AR@5 分布。

### 跟同期工作的位置

论文 Table 1 自己列了四族系统：

| 系统族 | 复用状态 | Agent 面向结果 |
|---|---|---|
| Code stores（Glean, Sourcegraph, Zoekt, CocoIndex） | 物化事实/增量转换 | 搜索、导航、开发者/agent 数据 |
| Retrieval graphs（CodexGraph, RepoGraph, Codebase-Memory, RIG） | 仓库/任务范围结构 | 排序和结构上下文 |
| Live analysis（Serena, LSPRAG, OpenCode） | 工作区状态 + 实时 LSP/工具 | 按需语义、文件/搜索、编辑命令 |
| **CodeNib** | **manifest + lexical/dense/structural + 增量维护** | **检索、静态导航、有界上下文** |

CodeNib 想吃的是**集成但分清边界**这个卡位——词法/稠密/结构三视图都做、增量维护都做、agent 工具也做，**但每个组件的边界都写清楚**。这是个聪明的卡位，因为：

- 它不和 Sourcegraph/Zoekt 抢全代码搜索市场份额
- 它不和 Serena 抢实时 LSP 替代市场
- 它不和 CodexGraph 抢纯结构图市场
- 它提供的是**带成本可见性的 retrieval/navigation/context 三件套**，这恰好是搭 coding agent 时最缺的一块

### 工程启发

如果我自己搭 coding agent，这篇论文给我最大的启发是**视图保鲜成本**这个维度。

之前做 PoC 的时候，我的常见写法是：
- 起一个 BM25 服务（rogettf/ripgrep）
- 起一个 FAISS 服务
- 起一个 LSP 服务
- 写个简单的 wrapper 把它们暴露成 MCP 工具

**问题在于**：这三个服务的失效模式、更新路径、版本一致性没人管。代码改完之后 BM25 索引要不要重 build？LSP 要不要重启？嵌入要不要重算？我在 PoC 里都是"再调一次工具就发现了"。CodeNib 的 manifest 机制把这事**显式化**了——每个 commit 写一份"哪些视图可用、能力是什么"，**让 manifest 本身成为 fresh truth**。

另外一个比较细微但好用的点是**失败的 optional 视图不让 sibling 失效**。这意味着我们可以让用户**部分接入**——比如先只跑 BM25，跑稳了再加 dense，再加 graph。每个阶段都能独立 ship，**不需要等全栈索引可用**才上线。

---

## 📌 收尾

CodeNib 这篇论文给我留下的最深的印象是**它的边界感**。

它没说自己要替代 LSP，没说自己要替代 Sourcegraph，没说自己能做 patch 正确性——**它清晰地标出了 6 条它不覆盖的边界**。在一个 agent 研究普遍"我什么都能做"的氛围里，这种克制反而让人更愿意相信它做的那部分。

如果你也在搭 coding agent，这套"多视图 + manifest + 增量维护"的思路值得认真读一遍原文。**单看任何一个点都不算新颖，但合起来形成的"仓库上下文供应链"，是大部分现有 agent 框架都还缺的那一环**。

工程上最容易上手的，应该是它的 **Compact 上下文压缩**——不调 LLM、用确定性规则做 history rewrite。这一招对成本敏感的小模型 agent 收益可能比想象中大（50-87% token 节省可不是小数）。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
