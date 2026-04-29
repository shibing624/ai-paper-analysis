# AgentSearchBench：在 1 万个真实 Agent 里挑一个能干活的，到底有多难？

## 核心摘要

你做过一件挺费劲的事吗——在 GPT Store 里翻找一个"能帮你写 Java 代码"的 GPT。搜索框里输入"java code"，跳出来几十个看上去都一样的助手：JAVA Code Guide、Java Helper、Code Master Java……描述都是"擅长 Java 编程，遵循最佳实践"，截图都漂亮，评分也都 4.5+。你随便点进去几个，发现真正能用的可能就一个，剩下的要么答非所问，要么生成一堆看似规范实则报错的代码。

UCL 这篇论文就是死磕这个问题：**Agent 多到泛滥的时候，怎么从近 1 万个真实 Agent 里挑一个真的能完成任务的？** 他们做了一个叫 AgentSearchBench 的 benchmark——9,759 个从 GPT Store、Google Cloud Marketplace、AgentAI Platform 抓来的真实 Agent，3,211 个任务，66,740 次真实执行评测。最扎心的发现是：**语义相似度和实际性能之间存在持续的、显著的 gap**——top 1 的 reranker 给出的排名和"上帝视角"的最优排名相比，依然差着十万八千里。BM25 在他们合成 query 上 NDCG@5 是 0.236，搬到真实 HLE benchmark 上直接掉到 0.022——93%的衰减。这不是某个方法的问题，是整个"靠描述匹配 Agent"范式的问题。论文给了一个轻量级补救方案——执行感知 probing，让 Agent 先跑几个探针 query，再用执行结果辅助排序。效果有但不大，更像是在告诉社区：这个方向才是对的，descriptive matching 已经撞墙了。

值不值得看？如果你正在做 Agent 路由、Multi-Agent Orchestration、或者类似 Agent Hub 之类的产品，**这篇论文应该当作一个警告而不是一个解法来读**。

---

## 论文信息

- **标题**：AgentSearchBench: A Benchmark for AI Agent Search in the Wild
- **作者**：Bin Wu, Arastun Mammadli, Xiaoyu Zhang, Emine Yilmaz
- **机构**：Centre for Artificial Intelligence, University College London (UCL)
- **arXiv**：[2604.22436](https://arxiv.org/abs/2604.22436)
- **代码**：[github.com/Bingo-W/AgentSearchBench](https://github.com/Bingo-W/AgentSearchBench)

---

## 为什么 Agent 搜索是一个全新的问题

先承认一件事——我第一次看到"Agent 搜索"这个 framing 的时候，第一反应是"这不就是 Tool Retrieval 换了个壳吗？" ToolBench、ToolRet 这条线已经卷了很久了，从 BM25 一直卷到 7B 的 Decoder-Only retriever，难道 Agent 不就是 Tool 加了个 LLM 包装吗？

但坐下来读完前两节，我承认作者把问题厘清楚了。Tool 和 Agent 在搜索这件事上，差别其实挺本质：

| 维度 | Tool | Agent |
|------|------|-------|
| 功能边界 | 明确（API schema 定义清楚） | 模糊（一个 GPT 可能既能写代码又能讲笑话） |
| 描述与能力关系 | 文档基本反映了真实能力 | 描述夸大、能力重叠是常态 |
| 评估方式 | 输入输出可枚举 | 必须执行才知道行不行 |
| 候选规模 | ToolBench 1.6w，ToolRet 4.3w | GPT Store 一家就 1w+ |

更关键的是查询本身的形态。Tool retrieval 大多是"给定一个具体可执行的 query，找能完成它的工具"——这是一种**有标准答案的搜索**。但用户找 Agent 的时候，往往说的是"我想做个理财助手"、"帮我处理数据分析"——**这是高层级、不可直接执行的目标**。这种 query 在 ToolBench 那套 setup 里根本不存在。

论文用一张表把这个差异讲得很直接：

| Benchmark | 类型 | 候选数 | 真实场景 | Query 类型 |
|-----------|------|--------|---------|-----------|
| ToolBench | Tool | 16,464 | ✓ | 仅可执行 |
| ToolRet | Tool | 43,215 | ✓ | 仅可执行 |
| TREC 2025 | LLM | 1,131 | ✓ | 仅可执行 |
| AgentSquare | Agent | 16 | ✗ | 仅可执行 |
| OKC Bench | Agent | 127 | ✗ | 仅可执行 |
| **AgentSearchBench** | **Agent** | **9,759** | **✓** | **可执行+不可执行** |

看到 AgentSquare 候选 16 个、OKC Bench 候选 127 个我有点想笑——这种规模哪叫"搜索"，叫"模型选择"差不多。在真实开放生态里，候选池是 4 位数起步的，AgentSearchBench 把这个数量级补齐了。

## 怎么造出这 1 万个 Agent + 3000 个任务

这部分是工程量最重的活，我重点讲讲他们怎么避免几个常见的坑。

### Agent 池：直接爬真货

9,759 个 Agent 来自三个真实平台：GPT Store（chatgpt.com/gpts）、Google Cloud Marketplace、AgentAI Platform。其中 7,867 个有可执行接口——可以真的调用、真的测能力。

为了把不同平台的异构 Agent 喂给同一个 retriever，作者设计了一个统一 schema，把每个 Agent 拆成四组语义信息：

| Schema 组 | 字段示例 |
|----------|---------|
| Agent 元信息 | 版本化 ID、名称、官方 URL、平台来源 |
| 能力描述 | 功能描述、分类标签、支持模态 |
| 使用指引 | quick-start 指令、示例输入输出 |
| 可用性约束 | 价格、可访问性、底层模型、更新时间 |

举个例子：GPT Store 上的 "JAVA Code Guide" 在他们的 schema 里是这样：

```
ID:          agt:openaiagents:2518c1@v1.1
Source:      openaiagents
Name:        JAVA Code Guide
Description: A JAVA development assistant focusing on coding standards and quality.
Tools:       developer, browser, dalle, python
Model:       GPT-5.2
Quick start: Explain JAVA exception handling standards.
             How can I improve this MySQL query?
             Review my JAVA code snippet.
Access:      free
URL:         https://chatgpt.com/g/g-EYiFThMtQ
```

注意 quick-start 这一栏——这是 GPT Store 上每个 GPT 都会自带的"几个使用建议"。这玩意儿不是普通描述，是**开发者实际验证过 GPT 能完成的事**。后面的实验里，把 quick-start 一起喂给 retriever（叫 full-document indexing），效果有显著提升。这点我后面会展开。

### 任务和 relevance 的生成 pipeline

![图1：AgentSearchBench 的任务与相关性标签生成 pipeline](https://arxiv.org/html/2604.22436v1/x1.png)

*图1：AgentSearchBench 的核心 pipeline。绿色路径是单 Agent 任务查询的构造，蓝色路径是高层任务描述的构造，红色路径是基于执行的 relevance 标签生成。整个 pipeline 的核心思想是 "任务和标签都从 Agent 出发，再用真实执行去验证"。*

这张图基本浓缩了整篇论文的工程亮点，我拆开讲讲：

**第一阶段：Task Query 构造（绿色路径）**

不是凭空写 query，而是 grounded in agent documentation——从随机抽样的 Agent 出发，让 LLM 看着 Agent 的描述生成一个该 Agent 应该能完成的具体任务。然后过滤掉"闭卷任务"（不需要执行就能答的、纯知识问答类）。这一路下来产出了 2,452 个单 Agent 任务 + 500 个多 Agent 任务。

多 Agent 任务怎么造？把语义相关但不冗余的子任务用 LLM 拼成一个长 query，然后用 NLI（自然语言推理）验证：拼出来的复合 query 是否真的语义包含每一个子任务。论文里的例子很有代表性：

> "I'm trying to reset my health and routine, so first ask me 5 questions about my schedule... then build a 2-week productivity plan... plus a 7-day Mediterranean meal plan at 1,800 kcal/day... and a 7-day English study plan..."

这种 query 显然不是一个 Agent 能搞定的——你需要规划助手、健康饮食助手、英语学习助手。多 Agent 检索就是要找出这些能配合完成 query 的 Agent 组合。

**第二阶段：Task Description 构造（蓝色路径）**

这是论文的真正贡献之一。task description 不是从某个 Agent 描述衍生出来的，而是从一**簇**语义相关的 query 抽象出来的高层目标。比如把"总结新闻"、"对比新闻源"、"监控网站更新"这些 query 聚类，抽象出一个 description：**"Monitor, summarize, and compare news and web sources."**

然后用 rubric-based judge（5 个维度评分），每个维度选 top-2 query 和这个 description 关联起来，最终每个 description 配 10 个具体 query。**Agent 对 description 的相关性，等于它在这 10 个 query 上的平均执行表现**。这个设计挺漂亮——它把"我能不能完成抽象目标"这件无法直接评估的事，转化成"我在这 10 个具体子目标上的命中率"。

**第三阶段：Relevance 标签生成（红色路径）**

每个 Agent 在 query 上的执行结果用 GPT-5.2 当 judge 打 1-5 分。≥4 分算 relevant。为什么用 LLM 当 judge？因为这个量级的标注（66,740 次执行）人工根本搞不定。

但 LLM judge 靠不靠谱？作者拉了 3 个 PhD 级别的标注者，对 500 个执行实例做人工评估。**Cohen's kappa = 0.93，准确率 96.67%**。这个一致性已经接近"两个人工标注者之间的一致性"了，说明在这个相对简单的 binary 判断（4-5 vs 1-3）上，LLM judge 是可信的。

还有个细节我得夸一下——**doc-performance alignment 折扣**。如果一个 Agent 文档里没声明能做某事，但执行起来居然能做，作者会给它打折（relevance × 0.5）。这是为了防止"瞎猫碰死耗子"型的 Agent 拉高指标——你要是文档写得清楚、执行能力也匹配，才算真正"相关"。

## Benchmark 长什么样

数字层面：

| 统计项 | 数值 |
|--------|------|
| Agent 总数 | 9,759 |
| 任务总数 | 3,211 |
| - 单 Agent 任务 | 2,452 |
| - 多 Agent 任务 | 500 |
| - 任务描述 | 259 |
| 每个描述对应的平均 query 数 | 10 |
| 每个 query 评估的平均 Agent 数 | 20 |
| 总执行次数 | 66,740 |

6.7 万次真实 Agent 执行——这个工程量我得说一句，是真烧钱。GPT-5.2 当 judge 又跑这么多 query，光 API 费用就是天文数字。

Agent 的能力分布是典型的长尾：

![图2：AgentSearchBench 的 Agent 类别分布](https://arxiv.org/html/2604.22436v1/x2.png)

*图2：Agent 多样性。Data Analytics（11.3%）、Customer Support（10.1%）、Visual Content（9.9%）、Professional Advisory（9.6%）这四类占了 40% 以上。但长尾里也有 Career Services、Social Platforms、Creative Writing 这种小众但真实存在的类别。这种分布反映了一个真实问题——你想找 Customer Support 类 Agent，候选池里就有近千个，怎么选？*

更关键的是 **每个 query 对应多少个 relevant Agent**：

![图3：每个 query 对应的相关 Agent 数量分布](https://arxiv.org/html/2604.22436v1/x3.png)

*图3：上图——单 Agent 任务的 relevant agent 数集中在 1-20 个，多 Agent 任务集中在 10-40 个；下图——任务描述的 relevant agent 数中位数在 80 左右，分布更宽。这意味着仅靠"召回"是远远不够的，必须能 rank——在同样相关的几十个 Agent 里挑出真正最强的那几个。*

这个分布是论文的核心 motivation——**单纯的 retrieval 不解决问题，必须做精排**。Task description 一查就能查出 80+ 个相关的 Agent，你怎么从里面挑出最能干的 5 个？这是 AgentSearchBench 真正要 benchmark 的事。

## 实验结果：所有方法都和 oracle 差出几个身位

### Retrieval：在 1 万 Agent 里召回 top-K

下面是 retrieval 阶段的核心结果（NDCG @5/10/20，越高越好）：

| 类型 | 模型 | Task Query NDCG@5 | Task Description NDCG@5 |
|------|------|-------------------|--------------------------|
| Sparse | BM25 | 32.41 | 16.35 |
| Sparse | SPLADE v2 | 4.09 | 12.02 |
| Dense | BGE-Large v1.5 | 31.78 | **23.08** |
| Dense | MiniLM-L6 v2 | 29.02 | 20.67 |
| Tool | **ToolRet** | **37.52** | 21.15 |
| Tool | Tool-Embed | 34.02 | 21.15 |
| Decoder | Qwen-Embed 8B | 25.25 | 20.67 |
| Decoder | E5-Mistral 7B | 19.57 | 15.87 |

几个让我皱眉的事：

**第一**，**Decoder-Only 的 7B/8B retriever 全面落后**。E5-Mistral 7B 在 Task Query 上 NDCG@5 只有 19.57，比 BM25（32.41）还差一大截。这不是 Decoder retriever 不能用，而是它们**没有在 Agent/Tool 这种异构、短描述、长尾分布的语料上训练过**。这是个挺重要的工程信号——**通用 LLM-based retriever 不会自动迁移到 Agent 检索任务上**。

**第二**，**Task Description 的绝对分数明显低于 Task Query**。ToolRet 在 Task Query 上 NDCG@5=37.52，到 Task Description 直接掉到 21.15。原因不难理解——抽象目标里能力需求是隐含的，靠语义匹配自然吃亏。

**第三**，**Completeness 极低**。在 Task Description 上 @20 的 Completeness（即 top-20 至少覆盖每个子任务有一个相关 Agent 的比例）大多数方法都不到 5%。意思是说：**就算你把 top-20 都召回出来，也极少能凑齐覆盖所有子任务的 Agent 组合**。这对多 Agent 编排是个噩耗。

### Reranking：top-20 已经给你了，请精排

reranking 假设 retrieval 已经把 top-20 召回出来（用执行结果做的 ground truth），现在你要在这 20 个里排序：

| 类型 | 模型 | Task Query NDCG@1 | Task Description NDCG@1 |
|------|------|-------------------|--------------------------|
| - | Random Shuffle | 51.43 | 53.00 |
| Cross-Encoder | BGE Reranker v2 | 63.09 | 54.31 |
| Cross-Encoder | MXBAI Reranker Large | 52.09 | 52.84 |
| Tool | Tool-Rank 4B | 63.84 | 52.24 |
| Tool | **Tool-Rank 8B** | **66.67** | 53.92 |
| Decoder | Qwen Reranker 4B | 64.67 | 58.00 |
| LLM | RankGPT GPT-5.2 | 63.59 | **66.00** |
| LLM | RankGPT Qwen-3 32B | 61.00 | 53.95 |

注意一个**特别有意思**的细节：**Random Shuffle 的 NDCG@1 居然有 51.43**。这说明什么？top-20 里大部分都是相关 Agent（毕竟 retrieval 已经过滤过一遍），随便挑一个都不至于太差。这才是 reranking 的真实场景——**不是从噪音里挑金子，而是从一堆金子里挑最大那块**。

在这个语境下，看具体的 lift：

- Task Query 上 Tool-Rank 8B 比 Random 高 15.24 个点
- Task Description 上 RankGPT GPT-5.2 比 Random 高 13.00 个点

是有提升，但**没有那种"碾压"的感觉**。说实话，看到 GPT-5.2 在 Task Description 上能拿到 66.00 而其他模型基本卡在 53-58，我反而担心一件事——**这是不是 GPT-5.2 当 judge 自带的偏向？** GPT-5.2 既是评分员，又是 RankGPT 的 backbone，多少有点既当裁判又当运动员的味道。论文没怎么讨论这个潜在 bias，是个小遗憾。

### 最扎心的图：Oracle Gap

![图4：Single-Agent 任务上的 Oracle Gap](https://arxiv.org/html/2604.22436v1/x6.png)

*图4：单 Agent 任务上各类方法的 accumulated golden score vs Oracle。蓝色虚线（Oracle）就是上帝视角——top-K 完美按真实性能排序的累计得分。下面四条线（Dense / Sparse / Decoder-Only / Tool-Specific）是各类方法的实际表现，互相之间贴得很近，但跟 Oracle 的距离始终在 2-3 倍——top-20 里加起来才 4.x，而 Oracle 已经 10+。这是整个领域的天花板距离。*

这张图比所有数字都有说服力。**所有方法离 Oracle 都有持续 2-3 倍的 gap**，并且这个 gap **不集中在 top-K，而是均匀分布在整个排名上**。意思是：**好 Agent 不是被排到了第 6 位（你可以再多召回几个挽救一下），而是被埋在了第 15 位、第 18 位**。这种 misalignment 不是召回数量能补救的，是**排序能力的根本缺失**。

为什么会这样？作者的判断是 **semantic-performance gap**——Agent 的描述质量参差不齐，有的 Agent 写得花里胡哨但实际能力一般，有的低调内敛但实际很能打。基于文本相似度的 retrieval/reranking 很难穿透这层包装。

## 论文的两个我特别欣赏的细节

### 细节一：合成 query 与真实 query 的差距

作者很坦诚地承认：他们的 query 是合成的（LLM 从 Agent 描述生成的），不是真实用户提的。会不会因此高估了所有方法的性能？

他们做了一个验证实验——把同样的 retriever 拉到 HLE（Humanity's Last Exam）和 Finance Agent Benchmark 这两个真实 query 上跑：

![图5：Synthetic vs Realistic 对比](https://arxiv.org/html/2604.22436v1/x9.png)

*图5：同样的 BM25/BGE/ToolRet 在合成 query 和真实 query 上的 NDCG@5 对比。BM25 衰减最严重（0.236 → 0.022 / 0.047），跌了 80-93%；BGE 衰减相对温和（0.251 → 0.191 / 0.166）；ToolRet 最稳定（0.285 → 0.194 / 0.147）。绝对值都掉了不少，但**相对排序保持一致**——这才是 benchmark 设计者最关心的事。*

这个图我必须给个掌声。它说明两件事：

1. **合成 benchmark 是高估的**——真实场景的难度远高于 LLM 生成的 query
2. **方法之间的相对优劣关系是保留的**——你在合成 benchmark 上比赢了，在真实场景下大概率还是赢

这种"先承认局限，再证明 benchmark 至少能区分出方法优劣"的研究态度，是一篇严谨论文的标志。

### 细节二：Full-document indexing 的免费午餐

![图6：Full-document indexing 几乎全方位胜过 Description-only](https://arxiv.org/html/2604.22436v1/x10.png)

*图6：x 轴是只用 description 做索引的 NDCG@5，y 轴是把 quick-start usage examples 也加进索引的 NDCG@5。基本上所有点都在 y=x 上方——加入使用示例之后效果普遍更好。其中 Tool-Rank、Qwen Reranker、RankGPT GPT-5.2 提升最明显。*

这其实是一个挺反直觉的发现——**与其换一个更强的 retriever，不如把 indexing 内容拓宽**。为什么 quick-start 这种使用示例能帮上忙？回顾前面提到的——quick-start 是开发者**实际验证过 Agent 能干的事**，相当于 Agent 自带的"行为证据"。把这些行为证据并入索引，等于把 retriever 从"只能看简介"升级成"还能看履历"。

工程上这是个**几乎免费的优化**——你不需要换模型，不需要 fine-tune，只要在索引时多塞几行 usage example 就能拿到几个点。我觉得任何在做 Tool/Agent 检索的同学都应该立刻试一下。

### 细节三：Execution-Aware Probing 的小尝试

还有一个实验是 probing——让 LLM 生成一些 probing query，让候选 Agent 真的去跑，再用执行结果做额外排序信号。

![图7：Probing 效果与方差的关系](https://arxiv.org/html/2604.22436v1/x12.png)

*图7：横轴是不同 reranker，纵轴是 win rate（probing 比不 probing 好的比例）。绿色（高方差 probe）的 win rate 普遍最高，bge 在高方差下达到 50%、tool-8B 达到 53.1%；蓝色（低方差 probe）在大多数模型上效果一般。一个反例是 rankgpt-gpt 在高方差下反而下降到 34.4%。*

具体数字：

| Model | NDCG@5（无 probing） | NDCG@5（有 probing） | 变化 |
|-------|----------------------|------------------------|------|
| BGE Reranker v2 | 57.93 | 58.16 | +0.40% |
| Tool-Rank 8B | 60.82 | 61.71 | +1.46% |
| Qwen Reranker 4B | 60.96 | 61.91 | +1.56% |
| RankGPT GPT-5.2 | 61.25 | 59.60 | **-2.69%** |

我得说实话——**这个提升幅度其实不大**。1.46%、1.56%，在 paper 里能写一段，但拿到工业落地是值不值得多做一轮额外执行（每次 probing 都要真的调 Agent，成本不低）的成本，是个开放问题。

更让我皱眉的是 **RankGPT GPT-5.2 反而下降了 2.69%**。这个负向效果作者没有给出充分解释。我的猜测是 GPT-5.2 自己已经"懂"得足够多，再喂给它一些 probing response 反而是干扰信号——大模型不需要这种额外的行为证据，它的 prior 已经够强。如果是这样，那 probing 这条路对小模型才有意义，对前沿大模型可能用处不大。

不过 probing 这个方向我是认的——**未来 Agent 检索一定会从"描述匹配"走向"执行验证"**，只是怎么做得更轻量、更便宜，是要继续探索的。

---

## 我的判断：这篇论文值得读吗

**值得，但要带着批判去读**。

### 它的真正贡献是什么

不是某个新算法，是**第一次在真实开放生态规模上把 Agent 搜索这个问题严肃地提出来**。9,759 个真实 Agent、66,740 次真实执行、3 个真实平台、合成 query + 真实 query 双重验证——这种 setup 之前没人做过。AgentSquare 那种 16 个候选的"benchmark"在这种数据面前根本不算 benchmark。

更重要的是它**揭示了一个集体的失败**——目前所有 retrieval/reranking 方法（包括最新的 LLM-based RankGPT）在执行驱动的 ground truth 面前，离 oracle 都差着 2-3 倍的 gap。这不是某个 baseline 没做对的问题，是整个范式的 ceiling。这种"全员卡顶"的论文比"我提出一个新方法 SOTA"的论文有价值得多——它告诉社区：**这条路要换新方向了**。

### 它的局限和我的疑虑

**第一**，**LLM-as-judge 的循环问题**。GPT-5.2 既是相关性 judge、又是 RankGPT 的 backbone、又是 task generation 的 backbone。这种"裁判 = 选手"的 setup 多少有 self-bias。κ=0.93 的人工一致性证明了 judge 在 binary 任务上是可靠的，但在更精细的 ranking score 上呢？没有验证。

**第二**，**合成 query 的局限**。虽然作者用 HLE 和 Finance Agent 验证了相对趋势保留，但绝对分数的差距太大了——BM25 从 0.236 掉到 0.022，掉了 90%。这说明真实场景下的难度被严重低估，论文的乐观结论（"reranker 可以达到 60+ NDCG"）在真实业务里可能要打骨折。

**第三**，**Probing 的成本被淡化**。论文展示了 probing 有 1-1.5% 的提升，但没有讨论 probing 的成本——每次 probing 要真的调 Agent，对 GPT Store 这种闭源 Agent 还可能有 API 限流。如果你要给一个 query 里 top-20 候选 Agent 都跑 probing，那成本可能比直接让用户挨个试还高。这个**成本-收益比的真实账**论文没有算清楚。

**第四**，**没有覆盖动态能力变化**。Agent 是会更新的，今天的 GPT 明天就可能换底层模型、改 prompt。AgentSearchBench 是一个 snapshot benchmark，**动态语义漂移**这个问题完全没碰。但在真实场景里，这恰恰是 Agent 比 Tool 更难管理的地方。

### 工程启发

如果你正在做 Agent 路由、Agent Marketplace、Multi-Agent Orchestration 之类的产品，我从这篇论文里能拿走的几个具体动作：

1. **不要相信 Agent 的自我描述**。准备一个内部的执行记录池，定期跑一些 probing query 验证候选 Agent 的真实能力。
2. **Indexing 时把 usage examples / 历史调用样本一起塞进去**。这是免费的提升。
3. **不要指望 Decoder-Only retriever 直接迁移**。如果你要做 Agent retrieval，专门 fine-tune 一个领域内的 retriever（比如 ToolRet 风格的），效果远好于通用大模型。
4. **多 Agent 编排不要只看召回**。Completeness 这个指标比 Recall 更重要——你需要的是覆盖所有子任务的 Agent 组合，不是 top-K 里有几个相关的。
5. **执行成本是真实约束**。execution-aware probing 这条路要走，必须配合 caching、subset sampling、cheap-model probing 等手段才划算。

### 一句话评价

这是一篇**Benchmark > 方法**的论文，价值在于把"从 1 万真实 Agent 里挑一个能干活的"这个问题扎实地 setup 起来，并且诚实地告诉社区："你们现在都在 30-60 分挣扎，oracle 是 90 分以上，你们的路还远着呢。" 这种 honest paper 在卷 SOTA 的浪潮里挺珍贵的。

Agent 时代的搜索问题，远比我们想象的复杂。这篇论文是个开始，不是终点。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
