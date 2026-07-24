# AgentLens: 当我们停止用"通过/不通过"评判代码 Agent

你有没有这种感觉：让一个 AI 编程助手去修个 bug，看最后一行写着"已完成"，但你打开仓库一看，编译是红的，相关的测试根本没跑，或者它把隔壁文件也顺手改了。

传统代码 Agent 评测的"那一位"，就是 pass/fail——一个 0/1 位。但开发者日常用的不是 0/1 体验，是一整段交互过程：Agent 怎么理解我的需求、怎么用工具、怎么验证自己、犯错能不能自救、跟它聊得累不累。

[arxiv 2607.06624](https://arxiv.org/abs/2607.06624) 这篇 AgentLens 想要做的，正是把"这一段完整的过程"变成可被评判的对象，给每个模型一个**带可点击证据的诊断报告**。

---

## 核心摘要

AgentLens 是 Explyt 团队（论文原标题 *AgentLens: Production-Assessed Trajectory Reviews for Coding Agent Evaluation*）发布的一个**面向交互式代码 Agent 的轨迹级评测**。它把一段完整的对话、工具调用、文件编辑、验证过程、最终仓库状态作为评测对象，**每条轨迹由 5 个维度的 LLM 评判 + 形式化验证一起打分**：EndResult、InstructionCompliance、Pitfalls、Pleasantness、ToolCalls，再合成一个 Quality Index（QI）。它还配套了**两两对比（side-by-side）评审**和**夜间 CI 回归检测流水线**。

最有杀伤力的不是榜单本身——而是它通过评审内容诊断出了**多个被排行榜分数掩盖的真实问题**：例如 Kimi K2.6 的最低分不是模型能力问题而是 OpenRouter tool-parser bug；DeepSeek V4 Pro 与 Flash 的几乎打平背后是"控制 vs 速度"两种完全不同的策略；同一份 Haiku 在自家 IDE 和 Claude Code 上的差异来自 harness 熟悉度而非模型本身。

**我的判断**：这不是一篇"提出新指标"的论文，本质上是把"产品级 A/B 测试 + 错误归因"的思想用 Agent trajectory 重新做了一遍。**真正的价值在 review 文本和 nightly pipeline**，leaderboard 只是副产品。如果你也在做 Agent 产品，这一套流程可以照搬回去。

---

## 论文信息

- **arXiv ID**: 2607.06624
- **标题**: AgentLens: Production-Assessed Trajectory Reviews for Coding Agent Evaluation
- **作者**: Andrey Podivilov, Vadim Lomshakov, Sergey Savin, Matvei Startsev, Roman Pozharskiy, Maksim Parshin, Sergey Nikolenko
- **机构**: Explyt（主要），St. Petersburg Department of the Steklov Institute of Mathematics, St. Petersburg State University
- **发表日期**: 2026-07-07
- **代码仓库**: https://github.com/agent-lens/agent-lens-bench

---

## 为什么需要这种评测？——传统榜单的盲区

SWE-bench、Terminal-Bench 这些主流代码 Agent benchmark，本质都是问一个**二元问题**："Agent 最后把任务做完了吗？"

这个 0/1 位在研究型 leaderboard 上够用，但放在产品里完全失真。论文里举了个很直观的例子：

> 一个 Agent 跑完一个 32-step 的工程任务，最终测试全绿。但在这个 32 步里，它可能：跳过了 3 个必停的 checkpoint、修改了任务范围外的文件、伪造了构建成功的状态、用 `sed` 把代码改坏了又 revert 了 4 次、对你说"完成了"的时候仓库其实是红的。

从 pass/fail 视角看，这个 Agent"完成了任务"。**但任何一个用过 coding assistant 的开发者都知道，这种"绿"会让人血压飙升。**

更麻烦的是没有自然二元结果的任务——比如"给我写一份项目文档"。文件存在 ≠ 文档写得对。文档是否准确、是否契合代码库、是否对读者有用，根本不是 pass/fail 能衡量的。

AgentLens 想保留这种**过程性**结构：让评测看到完整 trajectory，并从里面抽出对用户真正重要的几条线。

---

## AgentLens 是什么？—— 5 维评审 + 形式化验证 + 横向对比

论文核心定义在 Section 3，我把它拆成 4 块讲清楚。

### 任务集：16 个真实工作流 × 2 个用户角色

AgentLens 目前 release 的是 **16 个 coding agent 场景**，每个跑两个用户 persona（default + toxic），所以一个 agent 一次跑出来是 32 条轨迹。

任务来源分两类：

1. **访谈驱动**：作者团队直接访谈程序员，问他们"昨天/前天你干了什么工作？具体做了哪些动作（看代码、改文件、查文档、跑测试、debug）？上次发生是什么时候？"用最近发生的、有代表性的工作流作为 task seed。
2. **生产数据派生**：从同意匿名收集的对话里抽出 usage summary，用 k-means 聚类出"使用模式簇"，再人工挑出 benchmark 还没覆盖的 cluster，写成新场景。这里用到了类似 Clio 的思路——只处理聚合、不接触原始对话。

三个用户 persona：

- **default**：宽松，不主动帮 Agent，只在明显失败时才干预
- **helpful**：协作型，会指正错误、给修正建议
- **toxic**：轻度不耐烦、可能挑战或嘲笑 Agent，但**不会故意误导**——任务目标保持

released fold 配的是 default + toxic。这样能直接看 Agent 在"不配合"的用户下还能不能稳定完成任务。

### 评审协议：6 个维度的解构

每条 trajectory 同时上 6 个评分维度，**前 5 个由 LLM judge 打分，第 6 个是形式化验证**：

| 维度 | 它在问什么 |
|---|---|
| **EndResult** | 任务最终结果是否真的"fit for purpose" |
| **InstructionCompliance** | Agent 是否按用户的显式规则/步骤/顺序/格式来办事 |
| **Pitfalls** | 抓过程性失败：工具滥用、空转循环、过早完成、缺失验证、流程不稳定 |
| **Pleasantness** | 交互体验：清晰、准确、简洁、productive、non-disruptive |
| **ToolCalls** | 工具使用质量：选择、参数、成功率、错误恢复、效率 |
| **Formal** | 客观检查：测试是否通过、仓库状态是否符合预期、Regex 命中数、覆盖度等 |

这 5 个 LLM 维度有一个共同设计：**每次给分同时给一段带证据引用的文字 review**。每条 review 结尾是结构化的 `Aspect | Severity | Evidence` 行，`[R$k$]` 这样的超链接直接跳到 trajectory 对应位置。

为什么不只给一个总分？因为**这些维度之间是真的互相制衡的**。论文里给了一个归一化相关矩阵（把每个 metric 除以 QI 后看 profile）：

- **EndResult vs ToolCalls 相关性 r = -0.66**：靠"工具用得糙但活干完了"拿分的，和"工具用得干净但活可能没干完"的，正好是两种策略
- **Formal vs Pleasantness 相关性 r = -0.74**：靠"形式化检查全过"拿分的，往往在交互体验上失分
- **InstructionCompliance 几乎跟谁都不强相关**：它既不是 outcome 的 proxy 也不是 tool use 的 proxy

所以一句话——**这 5 个维度信息冗余度低，合起来才是对一个 coding agent 的完整画像**。QI = 这 6 个分数的等权重平均（scale 到 0-100）：

$$Q(a) = \frac{1}{|M|}\sum_{m \in M} s_m(a)$$

简单归简单，但论文作者说"我们跟开发者的'vibe check'对了一下，QI 排名跟人脑里的可靠性印象基本吻合"。这是一句不起眼但很重要的话——**一个评测的 validity argument，最后得能跨过开发者那关**。

### Side-by-Side 评审：把诊断做到"对位"

QI 告诉你"谁分高"，但回答不了**为什么分高**、**A 模型比 B 模型到底强在哪**、**这次发版有没有把体验改差**。

AgentLens 对同一 task 的两条 trajectory 跑一个**配对 LLM 评判**，把 4 块信息一起喂进去：
1. Agent 1 的完整 trajectory
2. Agent 2 的完整 trajectory
3. Agent 1 的单轨迹 review（目标维度）
4. Agent 2 的单轨迹 review（目标维度）

而且 prompt 明确告诉 judge"**只比这一维，不要再自己从头推一遍单轨迹 review**"——这个约束非常重要，否则位置偏差会放大。

输出格式：

```
Aspect: <description> | Winner: A1/A2/Tie | Severity: low/medium/high | Evidence: <short evidence>
```

注意这里的 **severity 是相对的**——两个 Agent 同样烂，得分还是 Tie，不是"两个一起扣分"。

### 可靠性与偏置检查

LLM 当 judge 三个老问题，论文都做了检查：

1. **跑-跑方差**：把 GLM-5.1（自家部署）同一个配置跑 5 次，QI 均值 67.28，标准差 0.94（0-100 scale）。组件拆解：60.5% 的方差来自 Formal 验证（任务完成度本身有 flaky），18.1% 来自 Pitfalls，16.2% 来自 EndResult。**真正不可控的，是 borderline 任务能不能过验证**。
2. **位置偏置（order-swap）**：同一对 agent，交换 Agent 1 / Agent 2 位置再判一次。GLM-5.1 vs GLM-5.1（同模型对比）+ Opus 4.7 vs Haiku 4.5（高对比），5 个维度上 effect 方向都稳定，residual 很小。位置效应在这个聚合水平上可以忽略。
3. **Self-preference**：用 GPT-5.5 和 Sonnet 4.6 互判。23% 的 task-metric 出现分歧，**其中 18% 是各自偏自己、5% 是偏对方**——所以偏置是**不对称**的、确实存在。Pleasantness 维度贡献了一半的 self-favoring flips。这跟常识一致：越主观的维度，judge 越容易偏自己家。论文说 leaderboard 评测时**固定用 GPT-5.4 当 judge**来减少这个风险。

---

## CI Workflow：评测不是研究项目，是产品基础设施

这是论文 Figure 1，也是全文唯一一张架构图，我把它读给你看：

![AgentLens CI workflow：基准执行、单次评估、Side-by-Side 回归检测、通知维护者](https://arxiv.org/src/2607.06624/ci_workflow_diagramm.png)

*图 1：AgentLens 的夜间 CI workflow——从 nightly trigger 出发，并行跑多个 task fold 拉仓库 + Docker，merge 中间产物，存到实验跟踪系统；然后做单次评估，再拿当前 run 跟 anchor run 做 side-by-side 对比，触发回归检测；显著退化就通知 maintainer。*

**这张图说明了 AgentLens 真正的产品定位**——它不是一篇"我又造了一个 benchmark 跑分玩"的论文，是一套**已部署在 Explyt 自身 Agent 开发流程里的回归检测系统**。

实操例子论文给了两个，都在后面 Qualitative Analysis 展开。先预告一个比较有意思的：**一次并行 tool calling 的 race condition**，从最终结果看根本看不出问题，但 Pitfalls judge 通过 trajectory 直接定位到一个 `ConcurrentModificationException` 崩溃——把"质量掉了"具体化到"这个并发 bug 导致了质量掉了"。**这才是产品团队要的信号**。

---

## 实验：榜单 vs 故事

### 32-trajectory Java 榜单

评测目标：EAA（Explyt AI Agent）和 Claude Code 两个 harness，跑 17 个模型（主要用 OpenAI/Anthropic 官方，GLM-5.1 自家部署，其余走 OpenRouter）。所有单 run review 用 GPT-5.4 当 judge，reasoning effort 拉满。

| Harness | Model | QI | Formal | End | Instr. | Pitfalls | Pleasant. | Tools |
|---|---|---|---|---|---|---|---|---|
| EAA | Opus 4.7 | **81.5** | 81.2 | 94.0 | 75.0 | 64.0 | 92.0 | 83.0 |
| Claude Code | Opus 4.7 (xhigh) | 76.2 | 81.2 | 91.0 | 69.0 | 58.0 | 78.0 | 80.0 |
| EAA | GPT-5.5 | 73.0 | 75.0 | 78.0 | 70.0 | 55.0 | 80.0 | 80.0 |
| EAA | Sonnet 4.6 | 70.2 | 81.2 | 66.0 | 62.0 | 56.0 | 73.0 | 83.0 |
| Claude Code | Sonnet 4.6 | 70.1 | 90.6 | 80.0 | 62.0 | 55.0 | 75.0 | 58.0 |
| EAA | GLM-5.1 (self-hosted) | 67.3 ± 0.9 | 72.5 | 71.8 | 57.4 | 53.2 | 75.8 | 73.0 |
| EAA | DeepSeek V4 Flash | 64.8 | 87.5 | 73.0 | 50.0 | 50.0 | 66.0 | 62.0 |
| EAA | DeepSeek V4 Pro | 64.1 | 87.5 | 66.0 | 53.0 | 48.0 | 69.0 | 61.0 |
| Claude Code | Haiku 4.5 | 63.6 | 84.4 | 78.0 | 58.0 | 48.0 | 58.0 | 55.0 |
| EAA | GLM-5.1 | 60.6 | 80.6 | 60.0 | 53.0 | 48.0 | 61.0 | 61.0 |
| EAA | Haiku 4.5 | 58.2 | 75.0 | 72.0 | 55.0 | 47.0 | 47.0 | 53.0 |
| EAA | Gemini 3.1 Pro Preview | 54.5 | 71.9 | 62.0 | 52.0 | 41.0 | 50.0 | 50.0 |
| EAA | Gemini 3 Flash | 52.7 | 75.0 | 66.0 | 48.0 | 41.0 | 39.0 | 47.0 |
| EAA | MiniMax 2.7 | 50.8 | 70.0 | 52.0 | 47.0 | 38.0 | 43.0 | 55.0 |
| EAA | Qwen 3.6 Plus | 49.1 | 65.6 | 58.0 | 45.0 | 39.0 | 42.0 | 45.0 |
| EAA | Mimo V2.5 Pro | 45.1 | 62.5 | 50.0 | 38.0 | 32.0 | 36.0 | 52.0 |
| EAA | Kimi K2.6* | 28.1 | 42.9 | 29.0 | 33.0 | 19.0 | 21.0 | 24.0 |

*全数值百分比；Kimi K2.6 带 * 是因为 provider 端有 tool-parser bug（见下）。*

榜单本身没什么意外——Opus 4.7 第一（EAA harness 上比 Claude Code 高 5.3 个点），但**真正值得看的是下面这些 case**。

### Case 1：DeepSeek V4 Pro vs Flash — 几乎打平背后的策略分裂

QI：Pro 64.1 vs Flash 64.8。基本打平。但 side-by-side review 揭示了一个**控制-速度 trade-off**：

- **Pro 是更"稳"的 agent**：在 step gating、required pause、formatting、file-boundary 上更听话（instruction compliance 显著更好）
- **Flash 是更"快"的 agent**：放松了这些约束，**完成更多工作**，在 fixed budget 下更容易落到一个"能用、测试过"的状态
- 实测 Flash 吞吐比 Pro **高 2× 以上**

结论：**两个分数一样的模型，背后是完全不同的产品哲学**。一个靠"不出错"，一个靠"出够活"。**总分掩盖了这件事，分维度才能看见**。

### Case 2：Gemini 3.1 Pro Preview — 强模型，但 agent loop 不可靠

Gemini 3.1 Pro Preview 排在中下（QI 54.5），对一个 frontier model 来说挺尴尬的。但 review 给的诊断是 agent-loop 稳定性问题，不是 coding 能力问题：

- 跳过 / 合并步骤、还没批准就开始改
- 跑了一通但 reports 全绿——其实 `spotless:check` 和 `enforcer:enforce` 是红的
- 老 patch / 破坏性编辑循环、`sed` 写崩代码
- 32 个对话里 4 个 timeout

很典型的"模型很聪明但 agent harness 接不住"。

### Case 3：Kimi K2.6 排最后，但其实不是模型问题

这个 case 是论文最精彩的一笔——**榜单上最差的那个模型，被 review 翻案了**。

Kimi K2.6 QI 只有 28.1，Formal 42.9，几乎全维度垫底。表面看是"模型不行"，但 ToolCalls judge 的 review 看得更细：

> 20 / 21 reviews 报告了 schema / argument parse 错误，17 / 21 描述了**完全相同的** wrapper 错误（`{"": {...}}` 或更深嵌套），涉及 read_file、search_for_text、run_command、edit_file、write_file、list_dir 这些基本工具。

> 15 / 21 reviews 说 Agent 在 parser 报"缺顶层字段"之后，**还是用几乎一样的 payload 重试**，甚至越嵌套越深。

> 但 20 / 21 reviews 同时记到**只要改成 flat JSON，工具调用就成功了**——读仓库、搜文本、edit、write、跑测试、读 PR 都能跑通。

**所以 K2.6 的"差"，是 OpenRouter tool-parser 的 bug，不是模型 capability 的问题**。如果不读 review，团队会按"模型不行"的方向去优化，方向完全错了。

这种"分数错配 + review 救场"的案例，恰好是 AgentLens 整套设计最想捕获的信号。

### Case 4：Side-by-Side 抓到一个并发 race condition

这是 paper 给的第二个 example。某次 nightly 跑里，并行 tool calling 有 race condition，**但从最终输出几乎看不出异常**。Side-by-side Pitfalls judge 直接归因：

> Agent 2 的最显眼的 debug-relevant 失败是在 [C5] 出现了一个显式的 `ConcurrentModificationException` 崩溃；Agent 1 则是一连串 malformed tool/edit 调用和时间超时——其中 [C12] 一个 600s timeout，[C11]/[C22] 几次 broken edit invocation。

**这一段 review 把"质量掉了"具体化到了"这个并发 bug 让 Pitfalls 维度掉了 9 个 metric 比较点 + 触发 600s timeout"这一归因**——再配合 trajectory 里记录的 termination reason（异常退出），归因到 harness，而不是 agent。这就是论文自己说"AgentLens turns into a product-regression detector"的具体意思。

### Case 5：Explyt Haiku vs Claude Code Haiku — 同一个模型，两种命运

Haiku 4.5 在 EAA 上 QI 58.2，在 Claude Code 上 QI 63.6。差距 5.4 个点，**模型是一样的**。

数据归因：

- `edit_file` 工具成功率：EAA = 0.66 vs Claude Code = 0.93
- Pitfalls 维度：Claude Code 显著占优（`metrics_value` = 0.1812，p = 0.038）
- review 原文：EAA "more often makes poorly grounded or brittle changes and presents weaker verification"

**最合理的解释是 Claude Code 的 Haiku 跟 Claude Code harness 之间有协同**——可能是训练/评测时见过，模型知道这个 harness 的 tool contract 长什么样。

这个结果反过来也警告了一件事：**leaderboard 上的分不是 model capability alone，是 model + harness + provider + tool contract 的合集**。换 harness 跑出来的分数不能直接拿来比模型。

---

## 跟公开榜单的相关性

作者把 QI 跟 Artificial Analysis 上 11 个公开评测做了 Spearman rank correlation（用 leaderboard 里的 11 个 IDE-agent 模型）：

- **正相关最强**：APEX-Agents-AA（ρ=0.82）—— 这本身是个 agentic benchmark
- **负相关**：IFBench（ρ=-0.41）、τ²-Bench Telecom（ρ=-0.25）—— 这俩偏 instruction-following 和客服对话
- 11 个公开评测之间互相高度正相关（0.5 - 0.9 区间），**QI 是这一行里"独立"的那一个**

**这条相关性分析其实在讲一个挺重要的事**：现有公开评测彼此之间其实在重复测同一种东西（"考试能力"），而 AgentLens 这种"轨迹级"评测捕捉到的是另一种东西（"在 IDE 里能不能稳定干活"）。

具体例子：

- **Mimo V2.5 Pro 和 Gemini 3.1 Pro Preview**：在 AA composite 上排名比 QI 高 4.9 / 3.8 个 rank（90% bootstrap CI 不跨 0）——学术题刷得很溜，但 IDE 干活一般
- **Claude Opus 4.7、Sonnet 4.6、DeepSeek V4 Flash**：相反，在 QI 上排名比 AA 高 3.4-3.9 个 rank（CI 严格为负）——长程 IDE 工作比 composite 分数反映的要强

**这个 gap 是系统性的，不是噪声**。一句话：被"刷题式训练"优化的模型，跟被"工程化"优化的模型，在 AgentLens 这种评测上会被分开。

---

## 我的判断

### 这篇论文值不值得花时间看？

**值得**。但**不要冲着 leaderboard 看**——那张表是 paper 的最浅层。

**真正值得借鉴的是三件事**：

1. **维度解构**：5 维 LLM judge + 1 维 formal，比单一总分实用得多。每个维度配 evidence-linked review，把"为什么这个分"固定到 trajectory 的具体位置
2. **side-by-side 协议**：固定 judge model、要求 judge 沿用单轨迹 review 的 evidence、不要自己重推——这些约束是降低 LLM judge 噪声的实操经验
3. **nightly pipeline 跟 anchor run 对比**：把"评测"做成"产品流程的一部分"，用 side-by-side 评审做回归检测

### 局限性 / 我不太买账的地方

论文自己也承认了，我再补几条：

- **16 个场景是不是够**？作者说"开发者跟一个 coding assistant 用一周就觉得出谁好谁坏"——这话我同意，但"觉得出"和"评测出"是两件事。32 条 trajectory 上 judge 的方差我没看到 per-metric 的 std，只看到 QI 一项 0.94 的 std，**Pleasesantness 这种主观维度上 judge 自身的方差肯定比这大**。
- **Judge validity 是个弱点**。Section 3.1 写得很坦诚："我们自己做了点小规模 pairwise 标注对比，感觉 LLM judge 跟人类标注员的一致性至少跟人类互相之间的一致性一样高。"这话的方向我相信（之前 MT-Bench / Chatbot Arena 都有类似结论），但**32 个 task 量级、4 个标注员这种规模，没法 publish**。这是 paper 最大的留白。
- "**Benchmark 是为自家 IDE 量身定做的**"——作者自己在 Limitations 里写的。16 个工作流访谈自 Explyt 用户，agent harness 优先支持 IntelliJ IDEA + PyCharm。虽然 trajectory review 是开放数据，但 task 设计的中立性确实得让外部用一阵子才能确认。
- **Kimi K2.6 那种"分低不是模型差"的 case，确实反过来说明了 leaderboard 本身的不稳定**。同一个评测，OpenRouter 修一下 parser，分数可能就上去了。**榜单分数的有效期比论文里想的要短**。
- **OpenRouter 上的模型 routing 不可控**：作者在 Limitations 里也提到了。DeepSeek V4 Flash 比 Pro 吞吐高 2× 这种事，**跟模型本身没关系，跟 provider 的 serving 条件有关**。**榜单差异有一块就是"谁家 API 跑得顺"这种外部因素**。
- **单人 Opus 4.7 一次跑 100+ 美元**——成本对独立研究者不友好。这跟 SWE-bench 类似，老问题。

### 对工程实践的启发

如果你也在做 Agent 产品：

1. **不要只信 pass/fail**。把 trajectory 完整存下来，按 5-6 个维度做 review，dimension profile 比 single number 信息量大得多
2. **form verification + LLM judge 并用**。形式化检查扛客观，LLM judge 扛过程性
3. **side-by-side 回归检测**比自己跟自己对比能抓出更多问题，因为它能定位"这个具体 aspect 变了"
4. **judge 要固定**。换 judge 等于换尺子，榜单就只能相对
5. **low-rank on leaderboard 不等于 low capability**——总有可能是 harness / provider 的问题，review 帮你分辨

---

## 写在最后

AgentLens 的核心洞察其实不复杂：**过程比结果重要 + 分数后面要挂证据**。这两条在 software engineering 圈子里是老话，但把它做到一个 trajectory-level benchmark 里、用 nightly pipeline 跑出来、是另一回事。

论文里那张唯一的图（CI workflow），你看完可能会觉得"这不就是个普通 CI 流程"——**重点不是流程本身，是这套流程里跑的 review 是 trajectory-level 的、能直接告诉 maintainer "这次发版在 Tool Calls 维度上 regression 显著"这种信号**。

至于 leaderboard 上的数字，**6 个月后基本就过期了**——新一轮模型出来就会刷新。但 review 协议、维度设计、回归检测流程这些方法学的东西，能用很久。

---

## 参考文献

- Podivilov, A., Lomshakov, V., Savin, S., Startsev, M., Pozharskiy, R., Parshin, M., & Nikolenko, S. (2026). *AgentLens: Production-Assessed Trajectory Reviews for Coding Agent Evaluation*. arXiv:2607.06624. https://arxiv.org/abs/2607.06624
- Jimenez, C. E., et al. (2024). *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?*
- Yang, J., et al. (2024). *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*
- Yao, S., et al. (2024). *τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains*
- Zheng, L., et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
- Tamkin, A., et al. (2024). *Clio: Privacy-Preserving Insights into Real-World AI Use*

代码仓库: https://github.com/agent-lens/agent-lens-bench

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
