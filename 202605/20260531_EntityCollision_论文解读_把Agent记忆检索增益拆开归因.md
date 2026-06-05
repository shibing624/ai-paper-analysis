# Entity-Collision：把 Agent 记忆系统的检索增益拆开归因

> arXiv: [2605.29630](https://arxiv.org/abs/2605.29630) ｜ 2026-05-28 ｜ [Code](https://github.com/youwangd/engram) (Apache 2.0)
> 关键词：Agent Memory ｜ Retrieval Evaluation ｜ Embedder Attribution ｜ BM25 Baseline ｜ Reproducibility

---

## 一句话结论

现在 Agent memory 评测榜上的"提升"绝大多数都是被**词汇泄漏（lexical leakage）和标签混合（tag-mixing）**污染的虚假信号；这篇论文构造了一个**让 BM25 永远占据下界、按查询类型严格分层**的 Entity-Collision 协议，发现**编码器容量根本不是决定性变量**——参数量大 2.7 倍的 BGE-large 在词汇类查询上反而输给了 MiniLM-384。

## 为什么值得读

如果你做过 Agent memory 系统，下面这个剧本一定不陌生：

> "我们在 LongMemEval / LoCoMo 上把 baseline 提升了 X%，因为我们的新 embedder 比旧的好。"

但这个"提升"到底来自哪里？通常无人能答。原因是端到端基准里有两个无声的污染源：

1. **词汇泄漏**：query 和 gold answer、distractor 之间共享实体词（人名、地名、时间），**BM25 这种纯词频检索就足够把对的捞出来**——所谓"embedder 提升"其实是 BM25 也能拿到的部分；
2. **标签混合**：测试集里"用户偏好"、"服务调用"、"工具使用"被混在一起平均，**某个 embedder 在 A 类强、B 类弱，平均后看不见这个事实**。

这两个污染让 Agent memory 领域过去一年的"哪个 embedder 更好"基本是糊涂账。**这篇论文是第一篇严肃尝试把这件事讲清楚的工作。**

它的方法论意义远大于具体数字：作者构造了一个**系统无关、可分层、BM25 锚定**的评估协议，并且交付了一份**26 个结果表格 + 37 个复现脚本 + Apache 2.0 开源**的复现包，复现性达到了"每个置信区间可从原始流逐字节重放"的程度。在 LLM 评测越来越像玄学的当下，这种工程严谨度本身就是一种宣言。

## 背景：Agent Memory 评测的两宗罪

Agent memory 的核心任务是：

> 在过往多轮对话/操作历史里，给定一个新 query，找到相关的过往片段（事实、偏好、约束）来支持当前回答。

主流基准：

- **LongMemEval**：长期记忆考察，含 single-session preference 等子任务；
- **LoCoMo**：长对话记忆考察。

主流做法：

- 把过往片段做 chunking + embedding；
- 在线对 query 做 embedding 后做 dense retrieval；
- top-k 拼回 prompt。

**评测的关键混淆：**

### 罪一：词汇泄漏

构造测试集时，gold answer 里"小明买了 iPhone 15 Pro Max"这个事实，几乎肯定出现在原始片段里——不是因为 embedder 抓得好，**而是因为词汇就是直接重复的**。BM25 这种 sparse retrieval 是这种场景的最强 baseline。

之前的 dense retrieval 评测要么不报 BM25，要么报但不控制 distractor 的词汇分布——**结果是 dense 模型的"提升"里有一大块是 BM25 也能拿到的**。

### 罪二：标签混合

LongMemEval 里"偏好类查询"和"工具调用类查询"被平均在一起。但这两类问题的检索性质完全不同：

- **偏好**：高度依赖具体实体词（"我喜欢的咖啡店"）；
- **工具调用**：更依赖意图（intent style）模糊匹配。

**一个 embedder 可能在意图风格上强、在词汇风格上弱**，但聚合分数掩盖了这个事实。

## 核心方法：Entity-Collision 协议

作者的设计目标是：

> **让 BM25 永远是下界**，让所有"超过 BM25 的提升"明确归因到 embedder 的语义能力，而不是词汇匹配。

### 关键设计：构造性的 BM25 锚定

对每个 query：

- gold passage 含一组实体标记（entity tokens）；
- **每个 distractor 必须共享同样的实体标记**——这一步是关键；
- 这样一来 BM25 区分不出 gold 和 distractor，**一切超过 BM25 baseline 的检索增益必须来自语义层面**。

这个简单但精巧的设计把"是不是真懂语义"和"靠词频蹭分"彻底剥离开。

### 按 discriminator tag 分层

作者按查询的判别器类型（intent / lexical / preference / service / tool 共 5 类）做分层评估，**每个 tag 单独算指标**，不再做粗暴的全量平均。这样可以看到：

> embedder X 在 lexical tag 上 +10%，在 intent tag 上 -3%——这种异质模式是混合平均掩盖的。

### 碰撞强度（Collision Degree）

作者把"distractor 与 gold 的实体重叠程度"分成 **5 级 collision degree**：

- 低 collision：distractor 实体很少重叠；
- 高 collision：每个 distractor 都共享所有 gold 实体词。

**碰撞越深，BM25 baseline 越无能，越能逼出 embedder 的纯语义能力。**

### 实验维度

| 维度 | 配置 |
|---|---|
| Tags | 5 类（intent / lexical / preference / service / tool） |
| Embedders | 3 个（256-d hash trigram / MiniLM-384 / BGE-large） |
| Collision degrees | 5 级 |
| 统计 | paired-bootstrap 95% CI |
| 外部验证集 | LongMemEval (n=500), LoCoMo |

## 关键发现

### 发现 1：Hash trigram 只在浅水区有用

**256 维 hash trigram**（一种简单的 sparse embedding）：

- 仅在 **closed-vocabulary lexical tag** 上有用；
- 仅在 **深度 collision** 时有用；
- 其他场景下 trigram 连 BM25 都不如。

**含义**：sparse embedding 不是 dense 的廉价替代品，**它有非常窄的适用区**。

### 发现 2：MiniLM-384 是双轴赢家

**MiniLM-384**（小模型）在两个轴向（tag × collision degree）上**主导式获胜**：

- 各类 tag 表现稳定；
- 在中高 collision degree 上提升明显。

### 发现 3：BGE-large 参数大 2.7 倍但不均匀

**BGE-large** 模型容量是 MiniLM 的 2.7 倍：

- ✅ 在 **intent-style 查询**（意图风格）上**获胜**；
- ❌ 在 **lexical 查询**（词汇风格）上**输给 MiniLM**。

**结论：encoder capacity is NOT the binding constraint**。**模型更大不等于更好——架构与训练目标决定了它擅长哪种 query 类型。**

这是一个对从业者非常重要的"反直觉"发现：盲目升级 embedder 大小可能在某些查询类型上反而退步。

### 发现 4：合成意图零假设可在 LongMemEval 上复现

作者的"合成 intent-tag null"在 LongMemEval 上复现出了**single-session-preference recall cliff**——一个早就被报告但解释不清的现象。Entity-Collision 协议给这个 cliff 提供了清晰的归因：**它不是模型变弱了，而是该子集的 lexical leakage 比其他子集更弱，所以 dense retrieval 的"虚假优势"消失了**。

### 发现 5：LoCoMo 上的 11.7 pp oracle 余量没有可恢复信号

在 LoCoMo 数据集上，作者做了一个 **adaptive 向量权重路由**实验：

- **Oracle margin**：11.7 个百分点（即如果你能完美选择每个 query 的最佳 embedder，可获得 11.7 pp 提升）；
- **Recoverable signal**：**0**（即没有任何可学习的特征能让你在线选对 embedder）。

**这是一个对"自适应路由"研究的冷水**：理论上限存在，但实际信号不可学习——**当前 Agent memory 的"自适应"研究方向可能在白费力气**。

## 这篇论文给"做 Agent Memory"的人的启示

1. **必须报告 BM25 baseline**，且 distractor 必须做实体控制；
2. **不要做全量平均**，按 query 类型分层报告；
3. **Embedder 大小不是单调有利的**，要按场景选——MiniLM-384 在词汇类够用，BGE-large 在意图类更强；
4. **自适应路由暂时别做**：oracle 余量在但 recoverable signal 没有；
5. **复现包应该是 26 表 + 37 脚本级别**——这是工业级 LLM 评测的应有姿态。

## 这篇论文给"做 LLM 评测"的人的启示

Entity-Collision 协议的方法论价值是**通用的**：

- 任何 retrieval 评测都可以引入"BM25 锚定 + 分层"思路；
- 任何"哪个模型更好"问题，都应该问"在哪个 query 类型上"；
- 任何端到端基准都该被警惕"是否有不受控的捷径"。

这种思路对 RAG、长上下文、Agent benchmark 都同样适用。

## 工程实现：Engram 复现包

代码已开源：[github.com/youwangd/engram](https://github.com/youwangd/engram)

亮点：

- **26 个结果表 + 37 个复现脚本**，覆盖论文中所有数字；
- **Event-sourced decision log + DAG-state-machine schema lifecycle**：内存测试平台是确定性的；
- **每个 CI 可从 ingest stream 逐字节复现**；
- 公共注册表（public registry）验证版本一致；
- Apache 2.0 许可。

这种复现工程对学术界是稀缺的，对工业界则提供了一个可直接接入的 memory 评测框架。

## 批判性看法

**优点：**

- **第一篇严肃做 Agent memory 评测污染分析**的工作；
- Entity-Collision 协议**简单、优雅、可外推**；
- BGE-large 在 lexical 上输 MiniLM 是**反直觉但可信**的发现；
- LoCoMo oracle margin = 11.7pp 但 recoverable = 0 是**对自适应路由方向的有力警告**；
- 复现工程级别难得一见。

**局限：**

- **Embedder 选了 3 个，覆盖不算充分**：缺少 GTE、E5、Voyage、OpenAI text-embedding 系列等当下主流；
- **作者只有一位（Youwang Deng）**，没有交叉审稿/二次复核，方法学风险尚需更广验证；
- 论文 48 页（正文 6 页 + 附录 42 页），**对 reviewer 阅读成本极高**，关键结论容易被淹没；
- "BGE-large 在 lexical 上输 MiniLM"的现象，作者**没有给出训练目标层面的解释**，只是观察；
- **没有讨论生产环境的延迟/成本权衡**：MiniLM 已经够好的话，BGE-large 在意图类的 marginal gain 是否值得 2.7 倍参数？

**不该过度外推：**

- 这些结论基于**论文构造的合成测试集 + LongMemEval/LoCoMo 子集**，迁移到真实生产 Agent memory 流量上还需进一步验证；
- "Encoder capacity is not the binding constraint"是**在当前 query 难度下成立**，未来若 query 变得更长、更复杂，结论可能反转。

## 给现有 Agent Memory 系统的 audit 建议

如果你正在维护一个 Agent memory 系统（aiserver / agentica / 自家系统），可以用 Entity-Collision 思路对现有评测做一次 audit：

1. **跑 BM25 baseline**：看你的"dense retrieval 提升"在 BM25 之上还剩多少；
2. **按 query 类型分层**：按 lexical / intent / preference 拆开看；
3. **构造 collision distractor**：手工挑一些与 gold 共享实体的 distractor，重新跑 recall；
4. **对比小 embedder**：用 MiniLM 跑一遍，看你是不是真的需要更大的 embedder。

很可能会发现：你以前花了大力气调的 embedder，**收益没有想象中那么大**。

---

**总结**：这是一篇 method-paper-as-meta-criticism 的工作。它对 Agent memory 评测领域的贡献，类似 SuperGLUE 之于 GLUE、HELM 之于早期 LLM 评测——**告诉社区"你以前的指标本身就有问题"**。即使你不立即换协议，至少应该把"BM25 锚定 + 标签分层"作为内部评测的标准动作。
