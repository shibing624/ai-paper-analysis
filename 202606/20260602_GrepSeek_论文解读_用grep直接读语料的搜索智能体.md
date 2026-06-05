# GrepSeek：让搜索智能体扔掉向量索引，直接用 grep 翻语料

## 核心摘要

聊一篇挺反潮流的论文。

过去两年大家做 LLM 搜索智能体几乎都是一个套路——把语料先离线编码成 embedding，存到向量数据库，agent 通过 retriever 拿 top-k 文档做 RAG。这套范式很 Web 2.0：先建索引，再查询。

GrepSeek 这篇论文反着来——**让 agent 直接对着原始语料库（一个 14GB 的 Wikipedia 文本文件）跑 shell 命令**，`rg`、`grep`、`awk`、`sort | uniq | head`，全部走 Unix 管道，没有任何预编码的 embedding 索引。然后用一套两阶段训练（冷启动 SFT + GRPO 强化学习）让一个小模型学会"怎么 grep"。

效果：在 7 个开放域问答 benchmark 上微平均 F1 拿到 **0.5691**，比最强的 Search-R1（GRPO + Qwen3-Embedding-4B 稠密检索）的 0.5441 高了将近 2.5 个点；多跳推理上提升尤其明显（2WikiMultihopQA 涨了 8.79 个点）。代价：单次推理慢一点点（8.67s vs 4.77s），但**索引内存从 221GB 降到 14GB**，离线建索引的 62.4 个 A100 小时直接清零。

我的评价：这不是一个面向所有人的"杀手方案"，长尾实体和模糊查询（PopQA）上它是输给稠密检索的。但作为一种**反向重新审视检索范式**的工作，它挺值得一读——尤其是当你被向量数据库的运维成本搞得心累的时候。

---

## 论文信息

- **标题**：GrepSeek: Training Search Agents for Direct Corpus Interaction
- **作者**：Alireza Salemi, Chang Zeng, Atharva Nijasure, Jui-Hui Chung, Razieh Rahimi, Fernando Diaz, Hamed Zamani
- **arXiv**：[2605.29307](https://arxiv.org/abs/2605.29307)（2026 年 5 月 28 日提交）

---

## 一、为什么有人会想"不用 embedding 直接 grep 语料"？

先说说我读到这篇论文时的第一反应。

**说实话有点错愕。**2024 年开始，搜索智能体几乎成了 RAG 的代名词：embedding → 向量数据库 → top-k → ReAct。Search-R1、Search-O1、IRCoT 这些工作的差异基本都在"怎么 prompt 这个 retriever、怎么 RL 训练这个 agent"，retrieval 这一环大家是默认接受 dense embedding 的。

但你想想稠密检索这套范式的代价——

第一，**离线建索引的成本**。Wikipedia 那 21M 个 passage 要全部用 Qwen3-Embedding-4B 编一遍，论文给的数据是 **62.4 个 A100 小时**。如果你的语料是动态更新的（新闻、内部文档、代码库），这套预计算就成了运维负担。

第二，**runtime 内存膨胀**。原始语料 14GB，编码后 Qwen3-4B 的索引要 **221GB**，膨胀了将近 16 倍。E5-base 也要 70GB。这是单纯为了一个查询入口付的常驻内存税。

第三，也是更隐蔽的——**embedding 是 lossy 的**。你的查询语义被压到一个 4096 维的向量里，跟语料里那条 passage 的向量做余弦相似度。当查询带稀有专有名词、化学式、精确人名时，dense retriever 经常会把"看起来差不多"的实体糊在一起。

GrepSeek 的提法是：既然 LLM 本身已经能写代码了，那为什么不让它**直接生成 shell 命令在原始文本上做检索**？

```bash
# Agent 自己生成的查询例子（论文 Figure 2 风格）
rg -F "Olav Aukrust" corpus.txt | head -5
rg -F "Aukrust" corpus.txt | rg "born" | head -10
```

你看，这非常符合一个工程师的工作流——拿到一个新语料先 `rg` 几下找关键词，不会的字段再 `awk` 提一下，需要去重 `sort | uniq`。LLM 既然能写代码，做这种事天然合适。

![图1：检索增强 agent 与 GrepSeek 的范式对比 — 左侧是传统 RAG，要预先把语料编成 embedding 索引；右侧是 DCI（Direct Corpus Interaction），agent 直接通过 shell 命令访问原始语料，由分片并行引擎执行](https://arxiv.org/html/2605.29307v1/x1.png)

*图 1：左边是大家熟悉的"RAG + retriever"范式，右边是 GrepSeek 主张的 DCI（Direct Corpus Interaction）。差别一句话——DCI 没有"index"这个组件。*

---

## 二、技术上的关键挑战

但你别以为这就是"换个 tool 接口"那么简单。论文坦诚地讲，直接训一个小模型做 DCI 有三个真实的坑：

**坑一：corpus 太大，RL 优化极不稳定。** 14GB 文本意味着每一次 `rg` 命令都要扫一遍十几个 G 的数据。如果 agent 一开始没学好，频繁生成低质量的全语料扫描命令，训练会被 I/O 卡到几乎不动。

**坑二：命令空间过大，模型容易瞎试。** Unix 工具有几十个，再加上 pipe 组合，搜索空间是爆炸的。从零 RL 学，模型很容易陷入"乱组合 → reward 全 0 → 学不到东西"的死循环。

**坑三：直接给答案做 RL 容易"作弊"。** 如果你冷启动数据是用一个能看到答案的强模型生成的，那它生成的 trajectory 经常会在 query 里直接出现答案的关键词，agent 会学到"先猜答案再去 grep 答案"这种 inference 时不存在的捷径。

GrepSeek 怎么解决？两条线——**训练管线**（解决稳定性和 trajectory 质量）+ **执行引擎**（解决 I/O 慢的问题）。

---

## 三、方法核心：两阶段训练 + 分片并行执行

![图2：GrepSeek 的工作流程 — agent 在 ReAct 框架下迭代生成 think → tool_call (shell command) → tool_response 的多轮交互，每一轮都直接对原始语料执行命令拿回结果，最后给出 answer](https://arxiv.org/html/2605.29307v1/x2.png)

*图 2：一个真实的 trajectory。agent 先 think，然后写一个 `rg -F "..." corpus | head` 的 shell 命令，执行结果作为 tool_response 反馈回来，然后再 think → 再写命令 → 直到能回答问题。*

### 3.1 冷启动数据生成：Tutor + Planner 的双角色设计

这部分是我觉得论文最巧妙的一块，值得多说几句。

冷启动数据的目标是什么？你要给 agent 提供一批"好的 trajectory"——每条都包含：合理的 think、可执行的 shell 命令、能返回有用结果的命令、最终能从这些结果导出正确答案。**而且，trajectory 必须是 inference-time-realistic 的——agent 在每一步只能看到截至当前的历史，不能偷看后面的答案**。

朴素做法是用一个强 LLM（如 Claude）直接给 (question, answer, corpus) 让它生成 trajectory。问题来了——它会作弊。如果它知道答案是 "Olav Aukrust"，它的第一条命令很可能就是 `rg "Aukrust"`，但 inference 时 agent 是不知道答案的，这种轨迹学了反而有害。

GrepSeek 的方案是把数据生成拆成两个角色：

| 角色 | 能力 | 限制 |
|---|---|---|
| **Tutor**（answer-aware） | 知道 gold answer，负责构造 verified evidence chain | 生成命令时禁止使用 answer 实体本身或它的 alias 作为关键词（target-masking） |
| **Planner**（answer-blind） | 完全看不到答案，模拟真实 agent 行为生成 trace | 只能基于截至当前的历史做推理 |

然后用一个**反向构造 + 正向回放**的设计：

- **Backward Phase（后向）**：Tutor 把 question 拆成 N 个 sub-query。然后从最后一跳开始**倒着**构造：第 i 跳要找一个 evidence 文档 $d_i$ 能回答 $q_i$，命令 $c_i$ 必须满足 target-masking（不许直接搜答案）。每一跳成功后，提取一个 bridge entity 当作前一跳 $q_{i-1}$ 的目标。这样反向走 N 步，能构造出一条**不靠"知道答案"但仍然能找到 evidence 的命令链**。

- **Forward Phase（前向）**：把上面这条链反过来，按时间顺序回放。关键操作是用 answer-blind Planner 在每一步生成 reasoning（它不知道答案，所以推理是真实的），再用 Tutor 做一次 alignment——把 Planner 的 reasoning 微调一下，让它在逻辑上能 motivate 那条已经验证过的命令 $c_i$，但**不能引入未来才能观察到的信息**。

- **Quality Filtering**：最后再过一道滤——Planner 用最终历史生成 $\hat{y}$，要求 $F_1(\hat{y}, y) > 0$；Tutor 再做一次 causal 一致性 judge，把任何"隐含偷看了未来"的轨迹丢掉。

我觉得这个设计是真的精巧。**Backward 解决了"我怎么知道哪条命令能找到 evidence"的可行性问题，Forward 解决了"agent 在 inference 时根本不知道答案"的真实性问题。** 两者用 Tutor + Planner 的角色分离来 enforce，比单 LLM 一把梭要严谨得多。

### 3.2 SFT + GRPO 两阶段优化

冷启动数据生成完之后是标准动作：

**SFT 阶段**：在合成 trajectory 上做监督微调，模型学会"输出 think + tool_call + tool_response + answer"的结构化格式，以及合理的 shell 命令风格。论文反复强调一个点：**这一步主要是在固化"低层语法"——pipe 深度、`-F` 固定字符串模式、`| head -n` 截断这些东西在 SFT 后基本就稳定了**。

**GRPO 阶段**：在 NQ 和 HotpotQA 上做 RL，组大小 $n=5$，跑 200 步。reward 设计很直接：

$$R(\tau^{(i)}) = \phi(\tau^{(i)}) \cdot R_{\mathrm{ans}}(\tau^{(i)})$$

$\phi$ 是格式 indicator，要求 trajectory 包含合法的 `<think>`、`<tool_call>`、`<tool_response>`、`<answer>` 标签；$R_{\mathrm{ans}}$ 是 token-level F1。GRPO 的标准操作——组内归一化算 advantage：

$$A^{(i)}=\frac{R(\tau^{(i)})-\mathrm{mean}(\{R(\tau^{(j)})\}_{j=1}^{n})}{\mathrm{std}(\{R(\tau^{(j)})\}_{j=1}^{n})+\epsilon}$$

这套 reward 没有用 process supervision 也没有用 PRM，纯 outcome reward + 格式校验，模型靠 GRPO 的相对优势自己摸出来什么样的命令更有用。

论文有一个观察我觉得挺有意思——**SFT 决定低层语法，RL 主要 shape 高层搜索行为**。后面分析部分会详细聊这个发现。

### 3.3 分片并行执行引擎：把 grep 加速 7.6 倍

说完训练说工程。如果你真把 21M 条 Wikipedia 当一个 14GB 的文本文件给 `rg`，单线程 grep 跑一次大概 5 秒。一条 trajectory 平均要 6 个回合（最多 T=6），那一次推理光 grep 就 30 秒，做 RL 训练的 rollout 成本会爆炸。

GrepSeek 的解法是一个**语义保持的分片并行执行引擎**：

- 把 corpus 切成 $S$ 个 shard（实验里用到 32 shard）
- 每个 shell pipeline 在所有 shard 上并行执行
- 结果在最后做一次合并（merge 策略要保持和单文件顺序执行的字节级等价）

加速效果：

| Shard 数 | 命令延迟 |
|---|---|
| 1 | 5.39 s |
| 8 | 1.22 s |
| 32 | 0.71 s |

最大加速 7.6×。论文还做了一个细节——把 corpus 常驻 RAM、用 deterministic execution flag 保证 byte-exact 等价，再加一个 persistent search daemon 避免每次 spawn 进程的开销。

我的工程经验是这种 shard-then-merge 的并行 grep 不是新东西，类似 `ripgrep` 自己用 `--threads` 多线程也能做到一些。论文这里的价值在于把它做成了一个**对 agent 透明的工具层**——agent 写一个完整 pipeline，引擎负责并行化，结果跟串行一致。

---

## 四、实验：DCI 真的能打过稠密检索吗？

实验设置一句话——backbone 是 Qwen2.5-7B（论文里写的是 Qwen3.5-27B 做 Tutor，但 agent 本身没那么大）；7 个 benchmark：单跳的 NQ、TriviaQA、PopQA，多跳的 HotpotQA、2Wiki、MuSiQue、Bamboogle；训练只用 NQ + HotpotQA，其他 5 个全是 OOD。

baselines 覆盖了 Direct（不带检索）、RAG（直接拼 top-3）、IRCoT（迭代 CoT）、Search-O1、Rejection Sampling、**Search-R1**（GRPO 训出来的当前 SOTA）。每个 retrieval 类 baseline 都跑了 BM25、E5-110M、Qwen3-Embedding-4B 三种 retriever。

### 4.1 主实验：F1 微平均 0.5691，赢得有点意外的多

| 方法 | Retriever | NQ* | TriviaQA | PopQA | HotpotQA* | 2Wiki | MuSiQue | Bamboogle | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Direct | — | 0.2733 | 0.5565 | 0.2364 | 0.2837 | 0.3353 | 0.1151 | 0.1648 | 0.3340 |
| RAG | Qwen3-4B | 0.5002 | 0.7212 | 0.5046 | 0.4548 | 0.3498 | 0.1609 | 0.3484 | 0.4905 |
| Search-O1 | Qwen3-4B | 0.4622 | 0.7290 | 0.4731 | 0.4828 | 0.4009 | 0.2517 | 0.6103 | 0.5021 |
| Rej. Sampling | Qwen3-4B | 0.4294 | 0.7258 | 0.4630 | 0.5442 | 0.4255 | 0.2697 | 0.6569 | 0.5133 |
| **Search-R1** | Qwen3-4B | 0.5067 | 0.7693 | **0.5101** | 0.5591 | 0.4299 | 0.2878 | **0.6989** | 0.5441 |
| **GrepSeek** | — | **0.5223↑** | 0.7673 | 0.4861↓ | **0.6231↑** | **0.5178↑** | **0.3006** | 0.6212 | **0.5691↑** |

\* 表示训练集，其他都是 OOD。↑ 表示统计显著优于最强 baseline，↓ 表示显著低于。

我看到这个表的第一反应——**多跳那一栏的提升真的是离谱**。

- HotpotQA：从 Search-R1 最高的 0.5591 涨到 0.6231，**涨了 6.4 个点**
- 2WikiMultihopQA：0.4299 → 0.5178，**涨了 8.79 个点**
- MuSiQue：0.2878 → 0.3006，涨 1.28 个点

这个结论挺直觉的——多跳问题需要在多个文档之间精确传递实体名。dense retriever 的 embedding 平滑性在这种场景反而是劣势，它会把 "Goldman Sachs Asset Management" 和 "Goldman Sachs Group" 糊在一个邻域里。GrepSeek 用 `rg -F "Goldman Sachs Asset Management"` 直接 lexical match，第一跳就能精确锁定，再用第二跳的 grep 找 evidence。**lexical 精度在多跳推理场景反而成了优势。**

但 PopQA 上输了。这也合理——PopQA 是长尾实体，很多查询里的实体名跟语料里的写法不完全一致（缺音标、拼写差异），exact-string match 在这种场景下会直接 miss。论文坦诚这是 lexical interaction 的固有缺陷。

### 4.2 消融：SFT 和 RL 缺一不可

| Variant | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | MuSiQue | Bamboogle | Avg |
|---|---|---|---|---|---|---|---|---|
| GrepSeek | 0.5223 | 0.7673 | 0.4861 | 0.6231 | 0.5178 | 0.3006 | 0.6212 | **0.5691** |
| − w/o GRPO | 0.3879 | 0.6389 | 0.3903 | 0.4737 | 0.2069 | 0.4231 | 0.2956 | 0.4249 |
| − w/o SFT | 0.2896 | 0.5451 | 0.3163 | 0.3705 | 0.1838 | 0.1291 | 0.3544 | 0.3314 |

两个观察：

1. **去掉 SFT 直接崩**——平均掉到 0.3314，比基础 RAG 都低。论文说 w/o SFT 跑到中后段会出现 training collapse，他们 report 的是崩之前的 checkpoint。这印证了"没有结构化的轨迹初始化，直接 RL 在大语料上根本学不动"。

2. **去掉 GRPO 也掉得不少**——0.4249，主要是多跳上崩盘（2Wiki 直接砍半到 0.2069）。但 MuSiQue 反而**涨了**（0.3006 → 0.4231），这个反直觉的现象论文没解释，我猜测是 SFT 模型在 MuSiQue 上是 over-conservative 的，RL 把它带向了更大胆但平均更优的策略，单点上反而退步。

### 4.3 效率：贵在 LLM decode，不在 grep

![图3：GrepSeek 与 dense retrieval baselines 的效率对比 — (a) 每查询推理延迟拆分（LLM 生成时间和工具执行时间），(b) 检索索引常驻内存占用，(c) 离线建索引的 A100 小时成本，(d) 搜索工具延迟随分片数的变化](https://arxiv.org/html/2605.29307v1/figs/efficiency_1x4.png)

*图 3：四个维度的效率对比。GrepSeek 的端到端延迟更高（8.67s）但内存和建索引成本是数量级优势。*

数据展开看：

| 维度 | GrepSeek | E5-110M | Qwen3-4B |
|---|---|---|---|
| 推理延迟（s/query） | 8.67 | 4.77 | 6.07 |
| └ LLM decoding | 7.86 | — | — |
| └ tool execution | 0.81 | — | — |
| 索引内存（GB） | 14 | 70 | 221 |
| 离线建索引（A100·h） | ≈ 1 min | 3.2 | 62.4 |

我看完这组数据有几个感受：

第一，**慢主要慢在 LLM 多写了几个 token**。GrepSeek 一条 trajectory 平均要 6000+ token（因为带 raw retrieved context），decoding 时间占了 7.86s，真正 grep 加上分片并行只要 0.81s。所以这个 8.67s 不是 grep 的问题，是 agent 本身需要更多的 think token 才能写好 shell 命令。

第二，**省内存是真省**。Wikipedia 14GB 原文，GrepSeek 就用 14GB（语料常驻 RAM）。Qwen3-4B 的 dense 索引要 221GB——这意味着如果你在 64GB 内存的服务器上跑，Qwen3 索引你**装都装不下**，必须用磁盘 mmap，那查询又会变慢。GrepSeek 在内存受限的真实环境里有结构性优势。

第三，**离线 cost 几乎归零**。62.4 个 A100 小时按云上价格大概 $200+，每次语料更新都要跑一次。GrepSeek 一分钟搞定。**对动态语料场景这是很实在的好处。**

### 4.4 训练动态：模型从"频繁 grep"学到"复杂 pipe"

![图5：训练动态曲线 — (a) 平均奖励，GrepSeek 高于所有 dense/sparse retriever 的 Search-R1 变种；(b) 平均响应长度，GrepSeek 更长；(c) 每条样本的平均搜索查询数，GrepSeek 在训练中下降，但 baseline 在上升](https://arxiv.org/html/2605.29307v1/figs/training_dynamics.png)

*图 5：训练 200 步内 GrepSeek（红线）vs Search-R1 系列（其他线）的对比。最反直觉的是 (c)——GrepSeek 的 query 数随训练下降，而 baseline 都在上升。*

关于 retrieval behavior，论文给了一张表：

| RL Step | Cmds/Traj. | Lines Scanned (head -n) | Resp. Tokens | Filter Chain | Pipes/Cmd |
|---|---|---|---|---|---|
| 10 | 3.06 | 4.9 | 4251 | 79% | 1.98 |
| 50 | 2.59 | 6.9 | 4554 | 80% | 2.04 |
| 100 | 2.56 | 11.8 | 5979 | 79% | 2.04 |
| 200 | 2.56 | 10.4 | 6409 | 78% | 2.01 |

读下来的结论是：

- **命令数变少**（3.06 → 2.56）：早期是"多发几个简单命令"，后期变成"少发几个复杂命令"
- **每命令 head 行数变多**（5 → 10）：从"小心翼翼只看 5 行"到"我相信我命令写得准，多看一点"
- **总 reasoning token 涨了 50%**（4251 → 6409）：think 写得更详细
- **pipe 深度、`-F` 用法、级联过滤这些低层模式不变**

这印证了我前面提到的那个论断——**SFT 把"怎么写一条合规命令"教会了，RL 在改的是"什么时候写、写多复杂、看多少结果"这种 meta-level 的策略**。这个分析挺干货的，值得做 agent RL 的人看。

### 4.5 SFT 数据规模的影响

![图4：SFT trajectories 数量对最终 RL 后 F1 的影响 — 0（base 模型）、2.5k、5k、10k 四档的对比，2.5k 已经能大幅拉起性能，5k 之后增益显著放缓](https://arxiv.org/html/2605.29307v1/figs/sft_size_effect_f1.png)

*图 4：冷启动数据规模 ablation。2.5k → 5k 涨幅明显，5k → 10k 增益放缓。*

简单读：**冷启动 2.5k 条就已经能把性能从 base 模型拉到接近最优的 80%+**，5k 条几乎打满，再加到 10k 收益边际下降。这对工程实践是个好消息——不需要造海量合成数据，那个 Backward + Forward 的 pipeline 跑个几千条就够了。

---

## 五、批判性视角：哪些地方我有保留？

写到这里，我得说几个让我皱眉的地方，免得听起来像无脑吹捧。

**第一，"DCI 是新范式"这话有水分。** 论文自己也承认（Section 2 开头），同期 Li et al. 2026 已经独立提了 DCI 的概念，只不过他们是 inference-time prompting Claude 这种大模型来做。GrepSeek 真正的创新点其实是"**怎么把 DCI 训到一个 7B 小模型上**"，而不是 DCI 本身。如果只看摘要可能会以为这是个"首创范式"的工作，实际是工程上的精进。

**第二，跟 Search-R1 的对比有点"挑战赛"味道。** Search-R1 + Qwen3-4B 的索引要 62.4 A100 小时建，那 4B 的 embedding 模型跟 7B 的 agent 加起来也是不小的算力。但如果让 dense retriever 配一个更弱的 retriever（比如 BM25）+ 更强的 reasoning agent（比如 14B 的 GRPO），鹿死谁手不一定。论文比的是同 backbone（同 7B agent），但 retriever 的算力差异其实没拉平。

**第三，"7.6× 加速"这个数字有点漂亮的可疑。** 这是 32 shard、最优 case 下的数字，从 5.39s → 0.71s。但这只是 grep 命令本身的延迟，端到端来看 grep 只占 0.81s，其他 7.86s 是 LLM decoding，加速 grep 对端到端意义有限。这个 headline 数字放摘要里有点抢戏。

**第四，PopQA 的 trade-off 没有方案。** 论文坦诚地说 lexical interaction 在长尾实体上吃亏，但没提出 hybrid 方案——比如什么时候 fallback 到 dense retriever。这其实是部署中真正要解决的问题，留作 future work 多少有点遗憾。

但说回来，这些都是合理的局限，不影响这篇论文的核心贡献。**它首次系统地把"训一个小模型做 DCI"这个事跑通了，给了完整的 SFT + RL 配方和效率工程**，这就够有价值了。

---

## 六、对工程实践的启发

如果你也在做 RAG / search agent，这篇论文有几个点可以借鉴：

**启发一：lexical 工具不只是 BM25 的角色。** 我们做 RAG 默认 retriever 是个"句向量计算器"，但其实 LLM 自己能写代码这件事意味着——你给它 `grep`、`jq`、`SQL` 这种精确工具，配合 reasoning 它能做出比 dense retrieval 更精准的事。**对于动态语料、内部文档、代码库这种"建索引贵但用得不那么频"的场景**，DCI 这条路值得评估。

**启发二：Tutor + Planner 的 trajectory 合成范式。** 这套"answer-aware 倒着造 + answer-blind 正着回放"的设计可以泛化——任何需要"知道答案才能造好 trajectory，但又不能让 trajectory 暴露答案"的场景都能用。我能想到的应用是 long-horizon agent task 的合成训练数据，或者 tool-use trajectory 的引导生成。

**启发三：SFT 决定语法，RL 决定策略。** 这个观察对 agent RL 工程很有用——如果你的 RL 训练发现 agent 一直在出 broken 命令或 broken format，那是 SFT 数据没做好；如果 RL 训练 reward 涨了但具体行为变化不大，那说明 SFT 已经把骨架定了，RL 只能微调 hyper-parameter 一样的东西。**别指望 RL 教会模型一个全新的工具语法。**

**启发四：分片并行 + RAM 常驻 + persistent daemon 这套工程组合**。如果你要在生产环境跑一个 LLM agent + shell tool 的系统，这三件套基本是标配，论文给了一个干净的实现范本可以参考。

---

## 七、收个尾

GrepSeek 给我最大的感受不是"DCI 多牛"，而是**它逼我重新审视一个被默认接受的范式**——为什么搜索就一定要先建 embedding 索引？这个假设过去两年大家都没怎么质疑。当大模型已经能写代码、能调用工具的时候，**让它直接对原始数据用专业工具操作**这条路，确实可能比"先把数据压成向量再让模型查"更直接、更精确。

当然这条路有它自己的问题——长尾实体、模糊查询、I/O 成本——但它给出了一种新的可能性。**未来真正好用的搜索智能体大概率是 hybrid 的**：lexical 工具走精确侧（多跳实体定位），dense retriever 走语义侧（PopQA 那种长尾），由 agent 自己根据 query 性质选路。

GrepSeek 把 lexical 这一边用 RL 训到了能打的状态，这就够它在这场范式之争里占一席之地了。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*

