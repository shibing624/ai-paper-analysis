---
title: 让相关性重新做主角：RARG 如何把 ripgrep 从无头苍蝇变成精准搜索
date: 2026-07-29
arxiv: 2607.24223
authors: Jiangnan Li, Yuqing Li, Mo Yu, Jinchao Zhang, Jie Zhou
institutions: Tencent, IIE-CAS
github: https://github.com/LeqsNaN/RARG
---

# 让相关性重新做主角：RARG 如何把 ripgrep 从无头苍蝇变成精准搜索

你有没有过这种体验——给一个 AI 智能体配了全套命令行工具（grep、ripgrep、cat、read），它确实能搜到东西，但每跑一轮就像一只没头苍蝇，在整个语料库里到处乱撞，明明相关答案就藏在某几个文档里，它却绕了 20 圈才摸到门。

2025 年底那篇 DCI（Direct Corpus Interaction）论文挺轰动的——把语料当文件系统，让 LLM 直接用 grep 类工具搜，BrowseComp-Plus 上把传统 RAG 干了 11 个点。但好景不长，故事到了 2026 年：

RISE 接过接力棒，用 BM25 给 DCI 套了个"圈"，把语料缩到 1000 篇里再让 agent 探索，效果还行，但成本还是偏高；
DR-DCI 又搞了个动态扩展工作区；
到了 2026 年 7 月，腾讯 + 中科院信工所这篇 RARG，终于把这件事的本质问题说清楚了——

**相关性不该只是个过滤器，它应该成为 grep 操作的"执行先验"。**

听上去像在玩文字游戏，但实际效果非常打：在 BrowseComp-Plus 100K 上准确率从 DCI 的 78% 涨到 84%（GPT-5.4-mini），工具调用数从 99.1 砍到 23.9；上 GPT-5.4 直接 91%，比 RISE 高 9 个点。

我读完第一反应是：这事怎么现在才被人想明白？

---

## 核心摘要

RARG 解决的是 DCI 范式下的"相关性真空"问题。原来的 DCI agent 拿到一个语料库就直接 grep，每个文件被平等对待——就像你在一个 10 万本书的图书馆里找一句话，不告诉你哪本书更可能藏着，你只能从第一本开始一本本翻。

RARG 的核心做法是引入一个 `embed_recall` 工具，让 embedding retriever 把语料按相关性排好序，然后**强制 ripgrep 按这个顺序扫**——`-j1` 强制单线程、顺序扫描，保证"相关文档里的命中先出来"。再叠两层增强：RARG+ 在最初给 agent 喂几段"查询相关段落"当起点；RARG++ 对 ripgrep 的所有命中做 match-level 重排，让低排名文档里的小亮点也能被看见。

效果：在 BrowseComp-Plus 上 84%/91% 准确率（mini/全量），BRIGHT 上 RARG+ 平均 nDCG@10 拿到 53.36，干掉专门的 NeMo retrieval agent（52.89）。

我的判断：这是一篇**方法论上很扎实、工程上很干净**的论文。它没有发明什么新模型，而是把"相关性"这个老概念在 DCI 这种新范式里重新定义了角色。但有个争议点我后面会讲——RISE 论文自己说在 1M 语料上拿到 81%，RARG 复现 RISE-BM25 只拿到 69%，这个 12 个点的 gap 让人觉得有点蹊跷。

---

## 论文信息

- **标题**：A New Role for Relevance: Guiding Corpus Interaction in Agentic Search
- **作者**：Jiangnan Li, Yuqing Li, Mo Yu, Jinchao Zhang, Jie Zhou
- **机构**：Tencent（腾讯）+ IIE-CAS（中科院信工所）
- **arXiv**：2607.24223（2026 年 7 月 27 日提交）
- **代码**：https://github.com/LeqsNaN/RARG

---

## 为什么需要这篇论文：相关性被用错了地方

先说 DCI 范式。Stanford + 滑铁卢 + TAMU 那批人 2025 年底搞出来的（arXiv:2605.05242），核心观点是：与其让 retriever 把语料压成 top-k 再喂给 LLM，不如把语料当文件系统，让 LLM 自己用 grep、ripgrep、cat 这些工具去翻。

为什么这事能 work？理由很直接——很多复杂问题需要的证据是"复合"的，要先用 grep 找到一个实体名，再 grep 找到这个实体的属性，再 grep 找到这个属性相关的论文。整个推理过程是动态的、单次 top-k 根本预测不到。

但 DCI 自己有个硬伤：**无差别扫描**。grep 不知道哪些文档更可能藏着答案，它就把所有文件平权处理。结果就是 agent 在长尾噪声里浪费大量调用，工具调用数能飙到 99 次/查询。

RISE（Jimmy Lin 团队，arXiv:2606.06880，2026 年 6 月）出来救场。它的思路是先用 BM25 把语料缩到 1000 篇，再让 agent 在这个 bounded workspace 里探索。在 100K 语料上用 GPT-5.4-mini 拿到 78% 准确率、$0.28/query 成本。

RARG 作者一看 RISE 的设计，眉头一皱——**BM25 还是太粗糙了，而且 RISE 把检索当"圈地"，圈完之后相关性就消失了**。在 RISE 内部，agent 还是无差别地在这个 1000 篇的圈里 grep，BM25 排出来的顺序基本没用上。

于是他们提了一个更尖锐的问题：

> 相关性为什么只能用来"圈地"？它为什么不能直接指导 grep 该先扫谁？

---

## 方法核心：从"圈地"到"执行先验"

RARG 的方法不复杂，但每一步都很有针对性。它由三个递进的变体组成：

### RARG：document-level relevance 做执行先验

新增一个工具 `embed_recall(scope_query)`，agent 给一个查询（可以多轮重写），embedding retriever 把整个语料按相似度排序，把 top-10000 个文档路径写到一个 scope 文件 `/tmp/scope_N.txt`，然后返回"这个 scope 文件 → 你的查询"这样一个映射给 agent。

接下来 agent 跑 ripgrep 的时候，用这种结构：

```bash
cat /tmp/scope_N.txt | xargs -d '\n' rg -j1 "PATTERN"
```

`-j1` 是关键——ripgrep 默认是多线程乱序扫，线程完成顺序决定命中顺序，会把相关性顺序打乱。`-j1` 强制单线程顺序扫描，保证"相关性高的文档里的命中先出来"。

直觉是什么？想象你在图书馆找一本特定主题的书，但不是从门口按字母顺序找，而是先看"哪些书架跟你的主题相关"，然后**按相关度从高到低逐架扫**。每扫完一个架子，你就能知道："啊，答案在第 3 架第 5 排"——而不是傻乎乎把整个图书馆每个角落都翻一遍。

但这里有个 trade-off：单线程 ripgrep 比多线程慢，所以 RARG 限制了 scope 大小（最多 10000 篇），并且单次 rg 只取前 30 个命中就停。论文里说这个开销"tolerable"，是性能-时间权衡后的选择。

### RARG+：给 agent 一个 entry point

光排好文档顺序还不够。RARG 作者发现 agent 拿到一个空 query 之后，前几步会"探索性乱试"——搜个关键词看什么匹配，再换个角度搜，浪费好几轮才摸到真正的搜索方向。

RARG+ 在 `embed_recall` 返回 scope 映射之后，**额外从 top 文档里捞几段最相关的段落**（按段落切分、embedding 评分、取 top-10），包在 `<qr_paragraph>` 标签里扔给 agent 当起点。

注意它不是"喂答案"——这些段落是给 agent 用的搜索入口参考，不是直接给答案。Context 满了之后这些段落会被压缩掉，因为它们不是核心证据。

### RARG++：match-level relevance rerank

第三个问题是：document-level 排序再准，document 内部也是混着的。一个相关文档里可能 1000 行只有 1 行是答案；而一个"看起来不那么相关"的文档（embedding 评分低）里，可能恰好藏着决定性的一句话——但因为它整个文档被排在后面，grep 在前面就触发 30-match 限制停掉了，根本轮不到它。

RARG++ 的解法是：对所有 ripgrep 命中（最多 500 条），用 embedding 重排，取 top 30 给 agent。重排 query 怎么构造？用 scope query + rg 当前 pattern 的关键词拼接：

```
Query: [scope query] 
RG focus: [keyword1] [keyword2] ...
```

这相当于"全局搜索目标 + 当前 grep 局部意图"组合出一个更精准的相关性判断。论文里试过让 LLM 自己生成这个 query（generative variant），但反而掉点 9 个，作者认为 LLM 训练时是直接发 Bash 命令的，加一个 rerank query 字段打破了它的行为模式。

---

### 架构图

![图1：RARG 与 DCI Agent 的整体对比——RARG 引入 embed_recall 工具，将相关文档路径按顺序写入 scope 文件；RARG+ 在此基础上添加 query-relevant paragraphs 作为入口；RARG++ 对 rg 命中做 match-level 重排](https://arxiv.org/html/2607.24223v1/x2.png)

*图 1：左边是 DCI-Agent 的无差别 grep（relevance-agnostic output）；右边三块分别是 RARG、RARG+、RARG++ 的递进设计。Document-level relevance 决定 grep 先扫谁，match-level relevance 决定哪些命中能到达模型。*

这张图把三个变体的关系画得很清楚——一层叠一层，每一层都解决一个具体的"信息没到 agent 眼前"的问题。

---

## 实验结果：精度和效率的帕累托前沿

主实验表（Table 1）很硬核，我直接给你看数据：

**BrowseComp-Plus 100K 文档、100 个 query、GPT-5.4-mini（medium reasoning）：**

| 方法 | Acc | Turns | Tools | Search | Bash | Read |
|------|-----|-------|-------|--------|------|------|
| RISE | 78% | 24.3 | 28.7 | 13.1 | 9.2 | 6.4 |
| RISE-BM25 | 77% | 23.0 | 29.6 | 14.8 | 9.6 | 5.2 |
| RISE-Q3-Emb-4B | 69% | 28.9 | 35.9 | 22.1 | 9.1 | 4.7 |
| Retrieval-Agent | 68% | 29.2 | 38.9 | 37.0 | – | 1.9 |
| DCI | 78% | 48.8 | **99.1** | – | 90.3 | 8.8 |
| RARG | 80% | 18.2 | 29.8 | 1.2 | 27.2 | 1.4 |
| RARG+ | 81% | 20.2 | 29.6 | 1.6 | 26.6 | 1.3 |
| RARG++ | 84 | **17.6** | **23.9** | 1.5 | **21.1** | 1.3 |

几个关键观察：

- RARG++ 拿到 84 vs RISE 78 个点：涨 6 个点，工具调用数从 28.7 砍到 23.9
- RARG++ 拿到 84 vs DCI 78 个点：同样涨 6 个点，但工具调用数从 **99.1 砍到 23.9**（降 76 个点）——这个效率提升非常能打
- **RISE 看起来 Search 用了 13 次**，本质是它在每次 Search 时把 top 文档的 snippet 也喂给 LLM 了——既做检索又做信息通道。所以 RISE 的 Search 数不是"圈地次数"，是"圈地 + 喂内容"的总和。RARG 反过来——Search 几乎只用 1.2 次来固定搜索顺序，剩下的工作全交给 Bash 里的 scoped rg

**GPT-5.4（更强的模型）上**：RARG++ 直接 91%，比 RISE 的 82% 高 9 个点，而且 turns 13.59、tools 25.43——又是又快又准。

**GPT-5.4-nano（小模型）上**：RARG++ 79% vs RISE 68%，仍然涨 11 个点。但 nano 上 Bash 比例明显下降（被指令跟随能力拖累），Search 调用数波动更大——论文后面也承认这点。

![图2：精度-效率的帕累托前沿——RARG 系列在 BC+ 和 BRIGHT 两个任务上都把 frontier 往左上推](https://arxiv.org/html/2607.24223v1/x1.png)

*图 2：左边 BC+ 100K 上的 Accuracy vs Cost（$ / query），RARG++ 在不同 backbone 下都把 frontier 推到了左上角；右边 BRIGHT 上 RARG+ nDCG@10 最高（53.36），超过专门做 retrieval 的 NeMo agent（52.89）。*

注意 Figure 1 的关键标注——`+9 pp / -46% cost`：RARG++ 用 GPT-5.4 比 RISE 涨 9 个点的同时成本降 46%。这是这次最有冲击力的一组数据。

---

## Scaling 测试：1M 文档时还撑得住吗？

RARG 作者做了一个非常硬核的实验——把 BC+ 的 100K 语料扩展到 1M，新增 90 万篇长文档（FineWeb-Edu）当干扰项。

**Table 2（GPT-5.4-mini，100 query）：**

| 方法 | Acc (100K → 1M) | Turns | Tools |
|------|------------------|-------|-------|
| RISE-BM25 | 77% → 69 | 25.4 | 32.1 |
| RARG | 80% → 78% | 19.3 | 31.8 |
| RARG+ | 81% → 78% | 21.9 | 30.4 |
| RARG++ | 84% → 79 | 17.8 | **24.7** |

1M 语料下 RARG++ 仍然领先 RISE-BM25 整整 10 个点。但有一个**值得说道的细节**——

RISE 自己的论文（arXiv:2606.06880）报告：在 1M 语料上 RISE-BM25 拿到 81 个点。而 RARG 论文复现 RISE-BM25 在 1M 上只拿到 69 个点。12 个点的 gap。

这可能有几种解释：
1. RARG 用的 900K 干扰文档是"长文档"（FineWeb-Edu），ripgrep 在长文档里命中噪声更多
2. RISE 原文用的 1M 语料可能干扰文档更短、更友好
3. 复现时的具体配置（compaction 阈值、上下文管理）可能有差异

但 RARG 论文没有详细讨论这个 gap，只是说"the difference may come from the design: our 900K added documents are long FineWeb-Edu articles"。

**我的判断**：这个 12 个点的复现差异是个重要信号。如果 RARG 在 1M 上的优势部分来自"它比 RISE 更适应长文档干扰"，那这个优势是有边界的；如果反过来 RISE 的 81% 是个偏乐观的数，那 RARG 的相对优势可能被高估了。**光看这一篇论文你没法判断，需要跑同条件对比实验**。

---

## BRIGHT：检索任务上的反直觉

BRIGHT 是个 reasoning-intensive retrieval benchmark（生物、地球科学、经济、机器人四个子集）。这里发生了一件很有意思的事——

**Table 3（GPT-5.4-mini，四个子集平均）：**

| 方法 | Avg nDCG@10 | Bio | Earth | Eco | Rob |
|------|-------------|-----|-------|-----|-----|
| DCI | 48.43 | 62.05 | 54.94 | 37.13 | 39.59 |
| RISE-BM25 | 41.60 | 50.27 | 47.80 | 33.65 | 34.67 |
| NeMo Agent | 52.89 | 65.15 | 61.85 | 39.05 | 45.49 |
| RARG+ | 53.36 | 66.70 | 62.16 | 37.23 | 47.34 |
| RARG | 51.75 | 63.87 | 60.54 | 38.50 | 44.07 |
| RARG++ | 50.55 | 61.65 | 61.32 | 36.14 | 43.10 |

**RARG+ 拿到 53.36，超过 NeMo 专门的 retrieval agent（52.89）。**

但更值得看的是三个 RARG 变体的**反序**：在 QA 上 `RARG++` > `RARG+` > `RARG`，在 BRIGHT 上变成 `RARG+` > `RARG` > `RARG++`。

作者的解释很直接——BRIGHT 要的是"广度优先"（recall 越多越好，再 rank 准），RARG++ 的 match-level reranking 把观察预算集中在局部强匹配上，反而不利于广召回；RARG+ 的入口段落提供了好的初始点，对广召回更友好。

**这个反序是个加分项**——说明 RARG 的三个设计组件不是捆绑死板的，可以根据任务特点选哪个变体。Fast convergence 对 QA 好，broad recall 对 retrieval 好。

---

## 行为分析：相关性真的被用上了吗？

光看主表涨点不够，RARG 作者还做了一组漂亮的"机制验证"实验。

### Figure 3：相关文档命中在 scope 排名上的分布

![图3：相关文档命中在 scope 排名上的分布热力图](https://arxiv.org/html/2607.24223v1/x3.png)

*图 3：横轴是 scope 内的排名分箱（1-500, 501-1000, ...），纵轴是不同 agent。颜色越深表示该排名段每查询的命中数。Embedding agent 的命中集中在 top-500 但总数少（30.3）；DCI 完全扁平（每个排名段都有命中，total 223.8）；RARG 集中在 top-500（72.0），RARG++ 兼顾 top 和中段。*

这张图直观展示了三种范式对"相关性"的使用方式：

- **Embedding Agent**：命中**极度集中在 top**，但**总数最少**（30.3/查询）——top-k 召回冗余、observation 被浪费
- **DCI**：命中**完全扁平**，每个排名段都差不多——典型的"无差别扫"
- **RARG**：命中**前重后轻**（top-500 拿 72 个）——document-level relevance 真的把 traversal 排序了
- `RARG++`：更**均衡**（top-500 拿 71.5，中段也能拿到）——match-level reranking 把低排名文档里的亮点救回来了

这就是 RARG 的"两层相关性"设计在数据上的体现。理论 → 实验的因果链很完整。

### Figure 4：Scope 质量和 Bash 组成

![图4：Scope 质量与 Bash 组成](https://arxiv.org/html/2607.24223v1/x4.png)

*图 4：左边是 scope recall（scope 覆盖金标文档的比例）和 RG coverage（rg 实际命中的覆盖率），三个 backbone/corpus 组合都给出误差棒。右边是 Bash 命令组成——scoped rg 在 mini 100K/1M 上占 87.5%/96.0%，说明 agent 确实在按设计协议使用 scoped rg。*

几个关键点：
- Scope recall 在 mini 100K 是 96.3 个点、1M 是 95.4 个点——embedding 几乎不受干扰文档影响，把 top-10K 砍掉后金标覆盖率仍然非常高
- RG coverage 在 mini 1M 掉到 75.9 个点——长文档里的"incidental match"（碰巧命中但不是答案）让 rg 命中率下降
- **Bash 组成**：mini 100K 上 scoped rg 占 87.5%，mini 1M 上占 96.0%——说明 agent 真的在用 scoped rg，不是又跑回全库扫

这组数据反过来印证了 RARG 的设计"起作用"的方式——不是魔法，是结构性的、协议性的。

---

## 案例分析：Russell David Lyons

论文做了一个定性 case study，找一个数学家的身份（要交叉 AMS fellowship + Ph.D. year + 两个共同作者 + 一个奖项 + 第二篇论文标题），三个 RARG 变体都答对 Russell David Lyons，但 tool use 严格递减：RARG 33 → RARG+ 18 → RARG++ 10，turns 17 → 11 → 10。

RARG 的答卷 CV 第一次出现在 turn 7（晚），RARG+/++ 在 turn 2 就已经看到了（早）——RARG+ 的 entry point 段落起效。RARG++ 又比 RARG+ 少 8 个 tool call——match-level reranking 让每一步看到的内容更准，减少了无效探索。

这个 case 单独看不算硬证据，但配合主表和行为分析看，三层递进的设计确实在工程上每加一层都更省。

---

## 我的判断

**亮点**：
- **问题定位精准**。它没有说"DCI 不好"，也没有说"RISE 不够好"，而是指出"相关性被用错了地方"——这是从已有工作中抽出"被忽视的维度"的能力，方法论味道很正
- **三层设计的因果链完整**。每个组件解决一个具体的"信息没到 agent 眼前"的问题，加一层涨一点不意外
- **效率提升非常硬**。GPT-5.4-mini 上 tools 从 DCI 的 99.1 砍到 23.9（降 76%），这是工程上立竿见影的改进
- **行为分析扎实**。Figure 3 的命中分布热力图和 Figure 4 的 scope 质量数据，让"为什么涨点"有据可依，不是黑盒吹

**问题/疑点**：
- **1M scaling 上 RISE-BM25 的复现差异**。RISE 原文 81%，RARG 复现 69%，12 个点。论文里轻描淡写说"可能是长文档干扰"，但这个 gap 值得更严肃的实验控制（比如同条件 ablation）
- **GPT-5.4-nano 上的不一致**。nano 模型的指令跟随问题让 Bash 组成变得不稳定（scoped rg 比例只有 65.5%），这意味着 RARG 的"协议正确性"对小模型是个硬依赖，迁移到其他弱模型可能掉点
- **Generative variant 的失败没展开**。让 LLM 自己生成 rerank query 反而掉 9 个点，作者归因于"train-evaluation gap"但只是一句话带过。这其实是个有意思的发现——LLM 在固定 protocol 上很稳，但一旦给它多一个自由度的字段就容易破坏 learned behavior
- **Embedding model 的依赖**。RARG 的有效性跟 embedding 质量强绑定，长文档排序和短匹配排序需要不同的模型（BRIGHT 上专门换了 Q3E）。这不是 RARG 独有的问题，但值得点出来

**对比同期工作**：
- DCI（arXiv:2605.05242）：范式开创者，但 scalability 是硬伤
- RISE（arXiv:2606.06880）：用 BM25 圈地解决 scalability，但 BM25 排序没下沉到交互层
- DR-DCI（arXiv:2606.14885）：动态扩展 workspace，跟 RARG 思路正交
- GrepSeek：把 grep 策略用 RL 训练出来，跟 RARG 互补
- RARG：在前人基础上把"相关性"的角色重新定义，方法最简洁、效果最硬

**工程上能不能用？** 能，而且挺直接。代码已经开源了（github.com/LeqsNaN/RARG），核心是 50 行不到的 embed_recall 工具和 prompt 模板。如果你在做一个 deep research agent 或者多跳 QA 系统，RARG 的设计可以直接借鉴——哪怕不用 ripgrep，把"按相关度排序的检索结果作为后续工具的输入"这个思路抽出来就够用。

---

## 写在最后的话

RARG 这篇论文让我重新审视了一个老问题——**检索在 agentic search 里的角色到底是什么**。

过去十年的 IR 研究几乎把"retriever"定义成"top-k selector"：给个 query，返回前 k 个，剩下的让 LLM 自己消化。但 agentic search 出来后，retriever 不应该只是个 selector 了——它应该是个**调度器**。

RARG 把这件事讲得最清楚：document-level 决定"先看谁"，match-level 决定"看到什么"，entry point 决定"从哪儿开始"。三层递进，组合起来就是一个"相关性驱动的搜索调度协议"。

不是颠覆，是把已有的组件重新摆位。这种工作其实最难——没有新模型、没有新 benchmark，但把整个 pipeline 的因果关系理顺了。

如果你也在做 deep research agent 或者多跳 QA，看到 "agent 在大语料里转圈找不到答案" 的现象，先想想你的 retriever 是不是只是个 selector，而不是个调度器。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。

## 研究文档（引用来源参考）
(no reference document available)

## 研究文档（引用来源参考）
(no reference document available)