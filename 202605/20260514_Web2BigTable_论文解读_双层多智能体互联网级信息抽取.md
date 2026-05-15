# Web2BigTable：用 GPT-5 mini + Gemini 3 Flash 干翻 GPT-5 High——双层 Agent + 自演化 skill bank 把 SR 拉到 7.5 倍

## 核心摘要

如果让你的 agent 出一张表——"列出 2024 年所有市值过百亿美元的 AI 初创公司，含融资额、CEO、最大轮次"——现在主流的方案是啥？要么 deep research 类（Gemini DeepResearch、OpenAI Deep Research）端到端跑一遍，要么 multi-agent 框架并行查。但你试过就知道：**广度型任务（多实体 × 多字段）是这两条路线共同的死穴**。原因很无聊：单 agent 上下文撑不住表格规模，多 agent 又因为 task decomposition 太僵硬导致系统性的 coverage gap。

这篇 Web2BigTable 给出的方案是双层 multi-agent + 自演化 skill bank：上层 orchestrator 拆任务、下层 worker 并行干活，共享一个可读写的 Markdown workboard。**关键创新是 skill bank 不靠微调，靠 run-verify-reflect 循环不断 append SKILL.md 文件**——人类可读、版本控制友好。

效果让人有点吓到：在 WideSearch 上 Avg@4 Success Rate **38.50（第二名只有 5.10，7.5×）**，Row F1 63.53（+25.03），Item F1 80.12（+14.42）。而且后端模型只用了 GPT-5 mini + Gemini 3 Flash 这种"小钢炮"组合，**单 agent 模式下 Item F1 只有 33.28——框架本身带来了 46+ 点的提升**。在 XBench-DeepSearch 上也做到 73.0 准确率，超过 MiroFlow、Minimax-M2 等 deep research 旗舰。

一句话评价：**这是 2026 最值得看的 agent 框架设计之一**。skill bank 不靠 RL、不靠 SFT、靠可读 markdown 文件累积——把 agent 自进化的成本压到了一个新的低点。

---

## 论文信息

- **标题**：Web2BigTable: A Bi-Level Multi-Agent LLM System for Internet-Scale Information Search and Extraction
- **作者**：Yuxuan Huang, Yihang Chen, Zhiyuan He, Yuxiang Chen, Ka Yiu Lee, Huichi Zhou, Weilin Luo, Meng Fang, Jun Wang
- **日期**：2026/04/29
- **arXiv**：https://arxiv.org/abs/2604.27221
- **代码**：https://github.com/web2bigtable/web2bigtable

![图1：Web2BigTable 运行时界面——左侧任务拆解状态，右侧 worker 并行执行，下方共享 workboard](https://arxiv.org/html/2604.27221v1/figure/tui.jpg)

*图1：左上是用户 query，左中是 orchestrator 把任务切成的子分区状态，右侧是 N 个 worker 在并行跑（每个对应一个实体子集），底部是共享的 markdown workboard——所有 worker 都能读，写各自的 tagged region。这套 TUI 实际上把 agent 内部状态完全显式化了。*

---

## 问题动机：现在的 agent 在"宽"任务上都拉胯

说实话我之前用 deep research 类产品做企业调研，最大的痛是**列表型 query**——"列出所有 X"、"汇总 Y 行业 Top N"。表面看是个简单任务，实际上不管哪个 SOTA 系统都做不好：

- **单 agent（Claude Opus、GPT-5 High）**：上下文限制下，跑到 30-50 行就开始遗漏、重复、字段串行
- **End-to-end deep research（OpenAI/Gemini）**：报告写得花团锦簇，但 SR 普遍低于 5%——你让它列 100 家公司，它给你 60 家还有 20 家虚构
- **传统 multi-agent 框架**：拆任务靠人工模板或 LLM 一次性 decomposition，整个系统的 coverage 取决于一开始拆得对不对

WideSearch 这个 benchmark 把这种"广度型"任务的难度暴露得非常彻底——主流系统 Avg@4 Success Rate 全在 5% 以下。

论文的判断我觉得很准：**广度型任务的瓶颈不在 worker 能力，而在 decomposition 策略**。worker 个体能力再强，拆错了照样补不回来。

---

## 方法核心：双层结构 + 三种记忆 + 自演化

### Bi-level 架构

公式表达就一句：

$$\boldsymbol{\tau} = (\tau_1, \dots, \tau_N) \sim \pi_o(\cdot \mid q, \mathcal{S}_o), \quad x_i \sim \pi_w^{(i)}(\cdot \mid \tau_i, m_e, s_i)$$

- **Orchestrator $\pi_o$**：基于 query q 和 orchestrator skill bank $\mathcal{S}_o$，把任务拆成 N 个子任务 τ_1, ..., τ_N
- **Workers $\{\pi_w^{(i)}\}$**：每个 worker 拿一个子任务 τ_i，从 worker skill bank $\mathcal{S}_w$ 里 retrieve 对应的 execution skill，并能读写共享 workboard $m_e$
- **最终输出**：$X = (x_1, \dots, x_N)$

![图2：Web2BigTable 整体架构——双层 agent + 三类记忆](https://arxiv.org/html/2604.27221v1/x1.png)

*图2：上方是 orchestrator 调度，配 orchestrator skill bank（长期）；中间是 N 个并行 worker，每个配 worker skill bank（长期）；底层是 shared workboard（短期），所有 worker 通过文件锁安全并发读写。三类记忆三个时间尺度：workboard 分钟级、skill bank 小时/天级。*

### 三种记忆的时间尺度

| 记忆 | 时间尺度 | 形式 | 写入规则 |
|------|---------|------|----------|
| Workboard $m_e$ | 分钟 / 单 episode | Markdown 文件 | 每个 worker 写自己 tagged region，文件锁串行化 |
| Orchestrator skill bank $\mathcal{S}_o$ | 跨 episode 累积 | SKILL.md 文件集合 | 只 append，不微调底层 LLM |
| Worker skill bank $\mathcal{S}_w$ | 跨 episode 累积 | SKILL.md 文件集合 | 只 append，不微调底层 LLM |

**这个设计我觉得是论文最 elegant 的部分**——把"短期工作内存"和"长期知识"用不同的存储方式区隔开。Workboard 是临时的，episode 结束就丢；skill bank 是永久的，跨 episode 累积。

而且，**skill bank 是人类可读的 markdown**——意味着你可以 git diff 看 agent 学到了什么新技能、可以手工修剪烂技能、可以 cross-project 迁移。这对工程团队而言是个巨大的可控性优势。

### Run-Verify-Reflect 训练循环

训练阶段不更新 LLM 权重，只更新 skill banks：

```
for episode k in training_queries:
    Stage 1 (Run): orchestrator 用 S_o^k 拆任务 → workers 用 S_w^k 跑 → 输出 X_k
    Stage 2 (Verify): 用 gold table X_k^gold 评估 Item F1，生成结构化错误报告 r^{k+1}
    Stage 3 (Reflect): 用 LLM 把 r^{k+1} 蒸馏成新 skill → append 到 S_o, S_w
```

**关键设计**：

- 训练只用 ≤ 20 query（带 gold table）
- skill bank 是 monotone append（只加不删）
- 底层 LLM 全程不动权重

这种"测试时学习"+"参数无关进化"的范式，让我想起 Voyager（Minecraft 那篇）的设计哲学。但 Web2BigTable 在结构化任务（出表）上把这套方法推进得非常彻底。

---

## 实验：碾压式的领先

### WideSearch 主榜（Avg@4 Success Rate）

| 类别 | 系统 | SR | Row F1 | Item F1 |
|------|------|----|---------|---------| 
| Single Agent | Claude-4.5-Sonnet | – | – | 65.70 |
| Single Agent | GPT-5 High | – | – | 62.20 |
| Single Agent | OpenAI o3-high | 4.50 | 34.00 | 52.60 |
| End-to-end | Gemini | 4.30 | 36.60 | 59.10 |
| End-to-end | OpenAI o3 | 3.00 | 23.90 | 45.50 |
| Multi-Agent | Claude Sonnet 4 (Thinking) | 3.60 | 38.50 | 62.20 |
| Multi-Agent | OpenAI o3-high | 5.10 | 37.80 | 57.30 |
| **Web2BigTable** | **GPT-5 mini + Gemini 3 Flash** | **38.50** | **63.53** | **80.12** |

注意第二名：multi-agent o3-high 的 SR 是 5.10，**Web2BigTable 是它的 7.55 倍**。Row F1 +25.03、Item F1 +14.42。

更狠的对比在表 2：

| 系统 | WideSearch SR | XBench Acc |
|------|---------------|-------------|
| GPT-5 mini (single agent) | 4.00 | 35.0 |
| Gemini 3 Flash (single agent) | 3.00 | 28.0 |
| **Web2BigTable (GPT-5 mini + Gemini 3 Flash)** | **38.50** | **73.0** |

**同样的 backbone，加上框架，SR 从 4 跳到 38.5**。这种规模的提升不是参数堆出来的，是 architectural advantage。

### XBench-DeepSearch（深度推理 benchmark）

| 系统 | Accuracy |
|------|----------|
| Minimax-M2 | 72.0 |
| DeepSeek-V3.2 | 71.0 |
| GLM-4.5 | 70.0 |
| MiroFlow (GPT-5) | 72.0 |
| Kimi-Researcher | 69.0 |
| Claude-4.5-Sonnet | 66.0 |
| **Web2BigTable (GPT-5 mini + Gemini 3 Flash)** | **73.0** |

XBench 是深度推理任务（多跳推理 + 跨源验证）。Web2BigTable 用 5 个 worker（不是 WideSearch 的 10 个）也照样赢——说明这套架构同时适配 breadth 和 depth。

### 消融：到底是哪部分在贡献？

| 配置 | WS SR | WS Row F1 | WS Item F1 | XBench Acc |
|------|-------|-----------|------------|-------------|
| **Full system** | **38.50** | **63.53** | **80.12** | **73.0** |
| w/o learned orch. skills | 7.00 | 45.23 | 62.87 | 41.0 |
| w/o workboard | 27.50 | 54.81 | 73.45 | 60.0 |
| w/o worker skill evolution | 33.00 | 59.67 | 76.38 | 64.0 |

**最大头是 orchestrator skill**——去掉 SR 直接从 38.5 掉到 7.0，跟 baseline 一档。其次是 workboard，再次是 worker skill。这个排序挺有意思：

> **拆得对 > 协作得好 > 个体技能强**

这个排序我个人非常认同——多 agent 系统的天花板大概率不是个体能力，而是 task decomposition 的策略空间。Web2BigTable 把后者做成了一个跨 episode 累积的可学习对象，是个真正的范式升级。

---

## 我的判断

**亮点**：

- **skill bank 不微调**：用 SKILL.md 文件累积 agent 学到的东西，**人可读、可 git diff、可手工编辑**。这对工程团队的可控性、可迁移性都是巨大优势。
- **训练成本极低**：20 个带 gold table 的 training query 就够了。相比 RLHF 动辄百万 trace 的训练，便宜得不是一点半点。
- **backbone 不挑**：GPT-5 mini + Gemini 3 Flash 这种"次旗舰"组合也能 SOTA。框架价值不依赖最强 LLM。
- **breadth + depth 通吃**：同一架构，调整 worker 数量就能从广度任务切换到深度任务。
- **shared workboard 这个设计**：让多 worker 系统有了"共同视野"，避免重复探索和遗漏覆盖，对多 agent 系统是个核心补丁。

**问题**：

- **skill bank 怎么避免越积越乱**？论文用 monotone append，但跑久了 SKILL.md 会膨胀到几 MB 的量级，retrieve 时的效率怎么保证？这块论文里没系统讨论。
- **训练 query 怎么选**？说是用 20 个 query 训，但 query 分布对最终 skill bank 影响巨大。文章里是手工合成的扰动版，没有自动化方案。
- **WideSearch 用 20 个合成 query 训，再在原始 200 个测**：技术上是合规的（query 不同），但训练时见过的"任务结构"和测试时高度相关。这个 generalization gap 论文没量化。
- **第 2 名 SR 5.10 → 第 1 名 SR 38.50 这个差距太大**：让我有点警惕。是不是 baseline 没有用同样的 workboard 协调机制？看起来 multi-agent baseline 用的是"standardised orchestration framework"，但具体是否包括 shared memory 不清楚。如果不包括，这个对比可能高估了 Web2BigTable 的"独有"贡献。
- **代码可复现性**：依赖 OpenRouter API、8000+ 云端 skills、GPT-5 mini 和 Gemini 3 Flash 的具体 prompt——一般实验室复现门槛不低。

---

## 工程启发

如果你在做：

- **企业信息抽取 / 数据库 enrichment**：Web2BigTable 这套"orchestrator 拆 + worker 并行 + workboard 协作"的架构可以直接抄。特别是 SKILL.md 这种可累积、可审计的 memory 形式，对企业内部要求 traceable 的场景非常友好。
- **agent 训练范式选型**：在你的应用上能写出 gold-standard reference 的话，run-verify-reflect 比 RL/SFT 便宜很多个数量级。考虑作为 first try。
- **多 agent 系统设计**：把"短期工作内存"（workboard）和"长期知识"（skill bank）显式分离，避免把所有 state 都塞进 prompt。
- **deep research 类产品**：单 agent 路线在广度任务上有结构性短板，下一代产品大概率要走双层 multi-agent。

最后一个略微反向的判断——**SKILL.md 这种"markdown 累积"的范式可能会在 2026 接下来成为新标准**。它对应着一个朴素但深刻的洞察：**agent 自演化不一定要微调，可以靠"写文档"实现**。这跟人类工程师怎么变强的方式更接近——读文档、写文档、修文档。

某种意义上 Web2BigTable 是把"DevOps 的 runbook 文化"搬到了 agent 系统里。这个 vibe 我特别喜欢。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
