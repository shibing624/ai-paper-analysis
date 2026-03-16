# 🔬 Autoresearch 深度解读：Karpathy 的"AI 自主科研"到底有没有戏？

> **一句话总结**：Karpathy 用 630 行代码搭了一个"AI 研究员"的最小可行原型——让 Agent 在你睡觉时自动改代码、跑实验、筛结果，一晚上干完人类一周的活。它不是 AGI，但它可能是 AI 科研范式转移的第一声枪响。

| 项目信息 | 详情 |
|---------|------|
| **项目名称** | autoresearch |
| **作者** | Andrej Karpathy（前 Tesla AI 负责人、OpenAI 联合创始人） |
| **发布时间** | 2026 年 3 月 |
| **GitHub** | [karpathy/autoresearch](https://github.com/karpathy/autoresearch) |
| **Stars** | 33.3k+ |
| **代码量** | 约 630 行（核心三文件） |
| **许可证** | MIT |
| **硬件要求** | 单张 NVIDIA GPU（H100 测试） |

---

## 📖 引子：一条让 AI 圈炸锅的推文

2026 年 3 月的某个清晨，Karpathy 在 X 上发了一条推文：

> "ah yes, this is what post-AGI feels like"

背后的故事是：他的 AI Agent 在他睡觉的 12 小时里，自主提交了 110 次代码变更，把一个语言模型的验证损失从 0.862415 一路压到 0.858039。而 Karpathy 本人呢？去蒸桑拿了。

几天后，他把这套方法论打包成了一个仅 630 行代码的开源项目——**autoresearch**，扔到了 GitHub 上。48 小时内揽星 9500，截至发文已突破 33k。

这件事之所以炸裂，不在于技术多复杂，而在于它触及了一个根本问题：**AI 能不能自己搞科研？搞出来的东西到底靠不靠谱？**

这篇文章就来掰扯清楚这个问题。

---

## 🎯 autoresearch 在做什么？

把它想象成一个"永不下班的实习生"——你给它一份研究方向的说明书（`program.md`），它就开始不停地改代码、跑实验、看结果、决定保留还是丢弃，然后继续下一轮。整个过程完全自主，不需要人类干预。

具体来说，它的工作流是一个无限循环：

```
while True:
    1. 读取当前 train.py 和实验历史
    2. 形成改进假设（比如"把学习率调高试试"）
    3. 修改 train.py
    4. git commit（留痕）
    5. 运行训练（严格 5 分钟）
    6. 检查验证损失（val_bpb）
    7. if 改进 → 保留 commit，更新基线
       else → git reset，回退到上一个好的版本
    8. 记录结果到 results.tsv
    9. 继续下一轮
```

这个设计有几个精妙之处值得展开。

---

## 🏗️ 架构设计：极简到近乎偏执

整个项目只有三个核心文件，每个文件的角色界定极其清晰：

| 文件 | 角色 | 谁来改 | 功能 |
|------|------|--------|------|
| `prepare.py` | 基础设施 | **没人改** | 数据下载、BPE 分词器训练、数据加载器、评估函数 |
| `train.py` | 实验对象 | **AI Agent 改** | GPT 模型定义、优化器（Muon + AdamW）、训练循环 |
| `program.md` | 研究指令 | **人类改** | 给 Agent 的行为指南，定义实验规则和策略 |

这个分工像什么？像一个实验室的权限管理——`prepare.py` 是实验室的基础设施（水电气、仪器），不允许任何人动；`train.py` 是实验台上的东西，研究员可以随便折腾；`program.md` 是 PI（导师）写的研究计划书，决定这个实验室的研究方向。

### 固定 5 分钟时间预算：看似武断，实则精妙

每次实验严格跑 5 分钟墙钟时间（不包括编译启动）。你可能会觉得这是个随意的数字，但仔细想就会发现它解决了一个大问题：**怎么公平比较两个完全不同的模型配置？**

传统做法是固定训练步数或 epoch 数，但不同架构每步的计算量差异巨大。5 分钟时间预算把比较基准锚定在了"你能拿到的计算量"上——不管你把模型搞大搞小、换什么优化器，大家在同一个 GPU 上都只有 5 分钟。这意味着 Agent 不仅要优化模型质量，还要优化计算效率。

按这个节奏，每小时大约跑 12 次实验，一晚上就是 100 次左右。人类研究员一周能做几次实验？可能 5-10 次？这就是量级差异。

### 评估指标：val_bpb

用的是验证集每字节位数（validation Bits Per Byte）。和常见的 perplexity 不同，BPB 与词表大小无关——你换了一个全新的 tokenizer，结果依然可比。这一点很关键，因为 Agent 可能会去改词表相关的东西。

---

## 🧪 实验数据：让数据说话

Karpathy 在 GitHub 上公开了多轮完整的实验记录。最有代表性的是在 H100 上的一次夜间运行。

### 核心实验：H100 夜间 126 次实验

![autoresearch实验进展图——83次实验中15次保留的改进轨迹](https://raw.githubusercontent.com/shibing624/ai-paper-analysis/main/images/autoresearch/progress.png)

*图：autoresearch 实验进展图。绿色实心点是被保留的改进，灰色点是被丢弃的尝试。绿色阶梯线显示了 val_bpb 的持续下降趋势。*

| 指标 | 数据 |
|------|------|
| 初始 val_bpb | 0.9979 |
| 最终 val_bpb | 0.9697 |
| 总改进幅度 | 0.0282 |
| 实验总次数 | 126 |
| 保留的改进 | 23 次 |
| 丢弃的尝试 | 102 次 |
| 崩溃 | 1 次 |
| 总耗时 | 约 10.5 小时 |
| 使用的 Agent | Claude（Anthropic） |

保留率大约 18%——也就是说，Agent 大约每 5-6 次尝试才能找到一个有效的改进。这个比例其实挺合理的，人类做实验的成功率也差不多。

### 排名前七的改进

Agent 发现的改进按贡献大小排列：

| 排名 | 改进描述 | val_bpb 下降 |
|------|----------|-------------|
| 1 | 批量大小从 524K 减半到 262K（更多训练步数） | -0.0119 |
| 2 | 深度 9，宽高比 57（增加一层 Transformer） | -0.0043 |
| 3 | 嵌入层学习率从 0.6 提高到 0.8 | -0.0033 |
| 4 | RoPE 基础频率从 10K 提高到 200K | -0.0012 |
| 5 | Unembedding 学习率从 0.004 调到 0.006 | -0.0010 |
| 6 | 值嵌入添加微小权重衰减 0.001 | -0.0010 |
| 7 | 短窗口设为 1/8 上下文（256 tokens） | -0.0009 |

排名第一的发现很有意思：**在固定时间预算下，把 batch size 减半反而更好**。直觉上 batch size 越大、梯度越稳，效果应该越好。但 Agent 发现了一个反直觉的事实——在只有 5 分钟的约束下，小 batch 意味着更多的训练步数（更多参数更新），这比"每步梯度更准"带来的收益更大。

社区用户 snowkcon 证实，他在一个月前用完全不同的计算设置也独立发现了同样的规律。Agent 用一晚上重新发现了这个 insight。

### 更惊人的数据：两天 700 次实验

根据 VentureBeat 的报道，经过两天的持续自动调优，autoresearch 处理了约 700 次自主更改，发现了约 20 个可迁移的改进，将 "Time to GPT-2" 指标（在 [nanochat](https://github.com/karpathy/nanochat) 上的一个社区基准）从 2.02 小时缩短到 1.80 小时——效率提升 11%。

Karpathy 自己也说，Agent 发现了他在二十年工作中忽略的**注意力缩放和正则化方面的疏忽**。

### 那些失败的尝试同样有价值

Agent 也踩了不少坑，这些"负面结果"同样珍贵：

| 失败尝试 | val_bpb 变化 | 点评 |
|---------|-------------|------|
| 权重共享（weight tying） | +2.24 | 直接爆炸，完全失效 |
| 并行注意力+MLP | +0.011 | 在小模型上不 work |
| 多查询注意力 MQA（n_kv_head=1） | +0.008 | 过于激进的压缩 |
| 移除 careful WD mask | +0.005 | 说明精细的正则化掩码很重要 |
| 5% warmup | +0.0008 | 在本次运行中反而有害 |

权重共享那个尤其有意思——在大模型文献中，权重共享（embedding 和 unembedding 共享参数）是一个广泛采用的技巧，但在这个小模型、短训练的场景下，它带来了 +2.24 BPB 的灾难性退化。这提醒我们：**很多 ML 的"最佳实践"是有上下文的**，Agent 不迷信教条，纯粹用数据说话。

---

## 🧠 关键技术发现：打破传统智慧

这次实验中 Agent 发现了几个违反"深度学习 folklore"的结论：

### 1. 对所有参数施加权重衰减

传统智慧（包括 PyTorch 官方示例）通常建议：**不要对 bias、layernorm、embedding 参数施加权重衰减**。但 Agent 发现，对嵌入层加 0.001 的微小权重衰减，对值嵌入（Value Embeddings）加 0.001-0.003 的权重衰减，都能带来实打实的改进。

不过这里有个狭窄的最优区间：VE 权重衰减到 0.005 就开始变差了。这就像调盐一样——一点就好，多了就咸。

GitHub 用户 aniruddhaadak80 评论说："通常 folklore 建议排除 biases 和 layernorms 的权重衰减，但 Agent 发现打破这个规则有实证效果——这非常有趣。" 这正是自动化实验的价值：**它不会被先入为主的偏见束缚**。

### 2. 初始化缩放有"甜点"

Agent 逐步测试了 Transformer 初始化缩放比例：0.8x → 0.7x → 0.68x 都在改善，但到 0.66x 和 0.65x 就开始退化了。最终锁定 **0.68x** 为甜点。这种精细的网格搜索，人类研究员通常没耐心做到这么细。

### 3. 高嵌入层学习率 + 正则化

当嵌入层加上了权重衰减（正则化）后，更高的学习率（从 0.6 → 0.8 → 0.9）反而更好。这形成了一个有趣的相互作用：正则化"兜底"，所以可以放心给更大的学习率。

---

## 🔗 与相关工作的对比

autoresearch 并不是第一个尝试"AI 自动科研"的项目。把它放在生态里看，能更清楚地理解它的位置。

### 对比表：AI 自动科研的几种路径

| 维度 | autoresearch（Karpathy） | AI Scientist v2（Sakana AI） | AutoML/NAS（传统） | Hyperspace AI（分布式 autoresearch） |
|------|--------------------------|------------------------------|--------------------|------------------------------------|
| **目标** | 优化小模型的训练配置 | 端到端自动论文生产 | 自动搜索模型架构/超参 | 分布式 Agent 协作优化 |
| **搜索空间** | 代码级（改 train.py） | 全流程（想法→实验→写论文） | 架构/超参空间 | 代码级 + 多节点 |
| **Agent** | Claude/Codex 等编程 Agent | LLM 驱动的端到端系统 | 无 Agent，纯优化算法 | 多 Agent P2P 网络 |
| **评估标准** | val_bpb（单一数值） | 同行评审分数 | accuracy/loss | val_bpb |
| **代码量** | 630 行 | 大型系统 | 框架级（数万行） | 扩展自 autoresearch |
| **硬件** | 单 GPU | 多 GPU | 多 GPU 集群 | 分布式网络 |
| **特色** | 极简、可复现、教育价值高 | 首次在 ICLR 发表 AI 生成论文（评审分 6/7/6） | 工业成熟度高 | GossipSub 协议传播策略 |

### AI Scientist v2：另一个极端

Sakana AI 的 AI Scientist v2 走了完全不同的路——它不只是调超参，而是试图完成从"提出假设"到"写论文"再到"同行评审"的全流程。据报道，它在 ICLR 2025 Workshop 上成功发表了完全由 AI 生成的论文，评审打分 6/7/6，超过了人类论文的平均水平。

但两者的设计哲学截然相反：AI Scientist 追求**广度**（覆盖研究全流程），autoresearch 追求**深度**（在一个极小的空间里把实验做透）。Karpathy 的选择更"工程师"——与其做一个什么都能干但什么都不太靠谱的系统，不如在一个可控的沙盒里证明"自主实验循环"这个概念本身是 work 的。

### Hyperspace AI：分布式进化

更有意思的是 Hyperspace AI 的 Varun Mathur 把 autoresearch 的单 Agent 循环扩展到了一个点对点网络。多个节点上的 Agent 各自做实验，通过 GossipSub 协议实时共享有效策略。

结果相当惊人：不同硬件上的 Agent 发展出了**完全不同的策略**——高性能 GPU 节点倾向于"暴力破解"（堆参数量），而 CPU 节点被迫在初始化策略上更"聪明"（比如尝试 Kaiming 和 Xavier 初始化）。当一个节点发现 Kaiming 初始化使损失下降 21% 时，这个策略"像病毒一样"在网络中传播。

据报道，仅用 17 小时，这些分布式 Agent 就**独立重新发现了人类实验室花费 8 年才正式化的里程碑**（如 RMSNorm、绑定嵌入等）。

---

## 🤔 有没有戏？客观评估

好，回到核心问题：**这个项目到底有没有戏？**

### 乐观的一面

**1. 概念验证已经完成，而且很有说服力。**

val_bpb 从 0.9979 降到 0.9697，绝对值看起来不大，但 Karpathy 回应质疑时说得很清楚："这一切都是在优化计算性能比。数据中有固有熵，loss 为 0 是不可达的。这些收益是真实的、实质性的。" 换算成"Time to GPT-2"的加速，11% 的效率提升意味着每个训练 run 都在省钱。

**2. 发现了人类 20 年忽略的盲点。**

Karpathy 自己承认 Agent 发现了他在注意力缩放和正则化方面的疏忽。当一个深度学习领域工作了 20 年的顶级专家都承认 Agent 教了他新东西，这就不是客套话了。

**3. 教育价值极高。**

630 行代码、单 GPU、MIT 许可证。任何人都可以克隆下来，改改 `program.md`，自己跑一晚上看看结果。这是目前门槛最低的"体验 AI 自主科研"的方式。社区已经有了 MacOS（MLX）版、Windows 版等多个 fork。

**4. 生态潜力巨大。**

Karpathy 紧接着又发布了 [agenthub](https://github.com/karpathy/agenthub)——一个 Agent-first 的协作平台，用裸 Git 仓库 + 消息板的形式让多个 Agent 在同一代码库上协作。autoresearch + agenthub 的组合，可能是"AI Agent 协作做研究"这个方向的基础设施雏形。

### 谨慎的一面

**1. 验证集过拟合是真实风险。**

GitHub 用户 alexisthual 提出了一个关键质疑：跑 126 次实验，每次都看 val_bpb 来决定保留还是丢弃，这不就是在用验证集做选择了吗？从统计角度看，这相当于在验证集上做了 126 次假设检验，false discovery rate 会显著上升。

虽然 Karpathy 没有直接回应这个问题，但这确实是自动化超参搜索领域的老问题。HyperBand、Successive Halving 等方法在设计时都需要考虑多次评估带来的统计漂移。autoresearch 目前没有任何类似 held-out test set 的机制来做最终验证。

**2. 搜索空间太小，发现的都是"调参"级别的改进。**

说白了，Agent 做的事情本质上是**在一个 train.py 文件里做超参搜索和微架构调整**。减半 batch size、调学习率、改 RoPE 频率……这些都是人类研究员也会做的事情，只是 Agent 做得更快、更系统。

真正的科学突破——比如发明 Transformer、提出 RLHF、设计 FlashAttention——需要的不是在已知空间里搜索，而是**定义新的搜索空间**。autoresearch 目前完全不具备这个能力。

**3. Agent 的"脆弱性"问题。**

社区用户 Nyrok 指出了一个实操层面的问题：Agent 读取的是自由文本格式的 `program.md`，指令措辞的微小变化会导致不可预测的结果。比如 warmup 在一次实验中有效（-0.0009），在另一次实验中反而有害（+0.0008）。

这不完全是 Agent 的锅——小模型训练本身就有随机性。但它确实说明当前的 Agent 还缺乏对实验**可复现性**的理解。它不会主动跑多次取平均，也不会做 significance test。

**4. 只针对 LLM 训练这一个场景。**

autoresearch 目前只能优化一个特定的 GPT 训练脚本。你不能直接拿它去做图像分类、强化学习或者药物发现。当然，核心思想（自主实验循环）是通用的，但要迁移到其他领域需要大量的工程适配。

---

## 💡 我的判断：范式转移的先声，但离"AI 科学家"还很远

打个比方：autoresearch 就像早期的 AlphaGo。AlphaGo 证明了"AI 可以在围棋上超越人类"，但它并没有直接导致通用 AI 的出现。同样，autoresearch 证明了"AI Agent 可以在受限环境下做出比人类更快、有时更好的实验决策"，但这离"AI 自主做科学研究"还有很长的路。

**具体来说，我认为 autoresearch 在以下方面"有戏"：**

| 方向 | 可行性评估 | 理由 |
|------|-----------|------|
| 自动化超参搜索的新范式 | ⭐⭐⭐⭐⭐ | 已经证明了。比 grid search / random search 更聪明，因为 Agent 能利用上下文（实验历史）做决策 |
| 教育和科研工具 | ⭐⭐⭐⭐⭐ | 630 行代码，任何人都能跑，是理解"AI 做实验"的最佳教材 |
| 发现人类盲点 | ⭐⭐⭐⭐ | 已有实证。但受限于搜索空间，发现的都是"微调"级别的 insight |
| 分布式 Agent 研究网络 | ⭐⭐⭐⭐ | Hyperspace AI 的实验已经展示了可能性，多 Agent 协作的涌现行为值得探索 |
| 替代人类 ML 研究员 | ⭐⭐ | 短期内不可能。Agent 不能定义问题、不能设计新架构、不能做跨领域类比 |
| 通用科学发现 | ⭐ | 需要完全不同的系统设计，autoresearch 的框架太窄 |

**对从业者的建议：**

如果你是做 ML 训练的工程师，**现在就可以用起来**。不管你是训练 LLM、微调模型还是做 NAS，autoresearch 的核心思想（固定时间预算 + Agent 自主循环 + git 版本控制）可以直接迁移到你的工作流中。你不需要 H100，MacOS 的 MLX 版已经可用。

如果你是研究者，值得关注的不是 autoresearch 本身的实验结果，而是它背后的**设计模式**：人类定义搜索边界（写 `program.md`），Agent 在边界内做高通量探索。这可能是未来"人机协作科研"的基本范式。

---

## 🔧 动手指南：怎么跑起来

如果你想亲自试试，流程非常简单：

```bash
# 1. 克隆仓库
git clone https://github.com/karpathy/autoresearch.git
cd autoresearch

# 2. 安装依赖
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 3. 准备数据（约 2 分钟）
uv run prepare.py

# 4. 手动跑一次基线实验（约 5 分钟）
uv run train.py

# 5. 启动 AI Agent（以 Claude 为例）
# 在你的 AI 编程助手中指向这个仓库，然后说：
# "Hi have a look at program.md and let's kick off a new experiment!"
```

对于 MacBook 用户，可以用社区的 MLX 移植版：

```bash
git clone https://github.com/trevin-creator/autoresearch-mlx.git
```

**低算力设备的调参建议：**

| 参数 | 默认值 | 低算力建议 |
|------|--------|-----------|
| 数据集 | FineWeb | TinyStories |
| vocab_size | 8192 | 1024-2048 |
| MAX_SEQ_LEN | 2048 | 256 |
| DEPTH | 8 | 4 |
| TOTAL_BATCH_SIZE | 524K | 16K |
| WINDOW_PATTERN | "SSSL" | "L" |

---

## 📊 更广阔的图景：AI 科研自动化的生态

autoresearch 不是孤立事件。它处在一个快速发展的生态中：

```
                    ┌─────────────────┐
                    │  AI Scientist v2 │  ← 端到端论文生产
                    │  (Sakana AI)     │
                    └────────┬────────┘
                             │
    ┌──────────┐    ┌────────▼────────┐    ┌──────────────┐
    │ AutoML   │    │  autoresearch   │    │ researchpooler│
    │ (传统)    │◄──│  (Karpathy)     │───►│ (Karpathy)    │
    │ NAS/HPO  │    │  单Agent实验循环  │    │  多Agent研究池  │
    └──────────┘    └────────┬────────┘    └──────────────┘
                             │
                    ┌────────▼────────┐
                    │  agenthub       │  ← Agent 协作平台
                    │  (Karpathy)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Hyperspace AI  │  ← 分布式Agent网络
                    │  (P2P + Gossip) │
                    └─────────────────┘
```

Karpathy 的布局很清晰：autoresearch 是单兵作战的概念验证，agenthub 是多 Agent 协作的基础设施，而社区力量（Hyperspace AI 等）正在探索分布式进化的可能性。

VentureBeat 的评论可能有些夸张，但方向判断没错：**人类的角色正在从"实验者"变成"实验设计师"。** 你不再亲自跑实验、调参、看 loss 曲线，而是设计搜索空间、定义约束、审查 Agent 的发现。

---

## 📝 结语：起点，不是终点

回到开头的问题：autoresearch 有没有戏？

有戏，但别指望它马上颠覆什么。它最大的价值不在于那些具体的实验发现（减半 batch size、微调权重衰减这些），而在于它**用 630 行代码证明了一个概念**：AI Agent 可以在一个受控环境中自主进行有意义的实验探索，并且能发现人类专家遗漏的 insight。

这就像 2012 年 AlexNet 证明深度学习能 work 一样——重要的不是 ImageNet 上那个具体的准确率数字，而是它打开了一扇门。

autoresearch 打开的那扇门是：**AI 研究的"inner loop"可以自动化了**。至于 outer loop（定义问题、选择方向、评判意义），那是人类的活儿，至少目前是。

Karpathy 在蒸桑拿的时候，他的 Agent 在通宵干活。这个画面本身就是对未来科研模式的一个隐喻——人类负责"思考做什么"，机器负责"把它做到极致"。

有趣的时代。

---

> **参考资料**：
> 1. [karpathy/autoresearch - GitHub](https://github.com/karpathy/autoresearch)
> 2. [Session report: 0.9979 → 0.9697 - GitHub Discussions #43](https://github.com/karpathy/autoresearch/discussions/43)
> 3. [PR #44: 125 experiments on H100](https://github.com/karpathy/autoresearch/pull/44)
> 4. [VentureBeat: Karpathy's autoresearch lets you run hundreds of AI experiments a night](https://venturebeat.com/technology/andrej-karpathys-new-open-source-autoresearch-lets-you-run-hundreds-of-ai)
> 5. [AI Scientist v2 - Sakana AI](https://github.com/SakanaAI/AI-Scientist-ICLR2025-Workshop-Experiment)
> 6. [karpathy/agenthub - GitHub](https://github.com/karpathy/agenthub)
> 7. [trevin-creator/autoresearch-mlx - MacOS 移植版](https://github.com/trevin-creator/autoresearch-mlx)



*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我的微信公众号：机器懂语言。*