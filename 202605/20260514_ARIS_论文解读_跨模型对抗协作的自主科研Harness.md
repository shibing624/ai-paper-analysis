# ARIS：当你怀疑 AI 在"看起来很对"地胡说八道——上海交大开源的跨模型对抗科研 Harness

## 核心摘要

AI 自主做科研这件事，过去一年突然变成了 **harness engineering**（脚手架工程）的问题——不是模型不够强，是**包裹模型的那一层系统逻辑不够强**。

上海交大这篇 ARIS 工作把这个洞见做到了极致。先看他们点出的"中心失败模式"：

> The central failure mode is **not visible breakdown but plausible unsupported success**——长跑 agent 不会显式崩溃，而是产出**看起来合理、实际证据不足、甚至默默继承了执行者自身偏见**的 claim。

这句话杀伤力很大。所有用 Claude / Codex / GPT 做长任务的人都遇过这种情况：跑了 8 个小时，output 看起来超漂亮，**回去一查，里面引用是错的、数字是 hallucinate 的、结论被悄悄"圆"过的**。这才是 autonomous research 真正的死敌。

ARIS 给的解法是 **Cross-Model Adversarial Collaboration**：

- **Executor**（执行模型）负责推进进度
- **Reviewer**（来自**不同模型家族**的审查模型）负责挑刺、提修改请求
- 两个模型用不同 weights + 不同训练数据，**让"看不见的偏见"互相抵消**

整套系统三层架构：
1. **执行层（Execution）**：65+ 个 Markdown 定义的可复用 skill、MCP 接入、持久化 research wiki、确定性图表生成
2. **编排层（Orchestration）**：5 个端到端 workflow（idea discovery → experiment → review loop → paper writing → rebuttal）+ effort 等级配置
3. **保障层（Assurance）**：三阶段 evidence-to-claim 审计、五轮 scientific editing、proof checking、PDF visual inspection

一句话评价：**这不是一篇做"更强模型"的论文，而是做"如何让长跑 AI 别撒谎"的论文**——是 2026 时点最值得学的 system 工作之一。

---

## 论文信息

- **标题**：ARIS: Autonomous Research via Adversarial Multi-Agent Collaboration
- **作者**：Ruofeng Yang（项目负责人）、Yongcan Li、Shuai Li（通讯）
- **机构**：上海交通大学 / 上海创智学院
- **日期**：2026/05/04
- **arXiv**：https://arxiv.org/abs/2605.03042
- **代码**：https://github.com/wanshuiyin/Auto-claude-code-research-in-sleep

![图1：ARIS 五个 workflow 的端到端 composition](https://arxiv.org/html/2605.03042v1/x2.png)

*图1：ARIS workflow library。顶部展示了五个 workflow 在四个研究阶段（Discovery、Experimentation、Manuscript、Post-Submission）的组合关系；虚线连接表示可选路径，黄色方块表示需要 reviewer 介入的关键节点。*

---

## 问题动机：为什么 autonomous research 一定要"对抗式"

我自己做 agent 系统挺久了，对 "self-refine 陷阱" 深有体会——

让同一个模型自己评自己的输出，**它几乎不会发现自己的错**。原因很直接：

- **prior 一样**：同一个模型对同一个 prompt 的 belief 高度一致
- **盲区一样**：训练数据里的偏见、错误、空缺，self-eval 时同样存在
- **会"圆"自己**：模型有强烈的 self-consistency 倾向，会把"自己生成的内容"判定为合理

所以 self-refine 跑越多轮，**模型不是在变正确，而是在变自信**。

ARIS 的核心立场是：

> **Heterogeneous models over single-model self-refinement**——用不同模型家族（GPT vs Claude vs Gemini）互相 review，让训练数据/对齐目标/偏见的差异成为 cross-check 的天然力量。

![图2：跨模型对抗协作的工作循环](https://arxiv.org/html/2605.03042v1/x6.png)

*图2：Cross-model adversarial collaboration 的工作循环。Executor 生成 → 不同家族的 Reviewer 提出 critique → executor 改进 → 收敛检查。reviewer 的 access 范围可配置（仅文档 / 含 repo / 含 raw data），不同 effort level 控制 reviewer 严格程度。*

---

## 系统架构：三层 + 五 workflow

### 设计哲学（5 条）

论文里这 5 条原则我觉得每一条都值得反复读：

1. **Heterogeneous models over single-model self-refinement**——单模型自我精炼是个 anti-pattern。
2. **Modular skill files over monolithic agents**——别写"巨型 agent prompt"，把能力切成可组合的 Markdown skill。
3. **Composability over fixed pipelines**——workflow 是 skill 的组合方式，不是硬编码 pipeline。
4. **Portability over vendor lock-in**——不依赖任何具体厂商工具（OpenAI Tools API 等）。
5. **Persistent memory over ephemeral context**——给 agent 加 wiki，让多 session 的发现能累积。

第 5 条尤其重要——

![图3：Research Wiki 为什么关键](https://arxiv.org/html/2605.03042v1/x8.png)

*图3：Wiki 的核心价值。左：没有 wiki，每个 session 都从空白开始，同一个失败的 idea A 会被反复重试；右：有 wiki，Session 1 失败的 idea A 被记录为"已尝试且失败"，后续 session 直接跳过。这是 long-horizon research 必须的"记忆"基础设施。*

### 执行层（Execution）：Markdown-defined Skills

ARIS 提供了 **65+ 个可复用 skill**，全部以 Markdown 文件定义。每个 skill 是一段结构化的 prompt + 工具调用规约，比如：
- `/research-lit`：文献检索
- `/experiment-plan`：实验规划
- `/paper-write`：论文撰写
- `/proof-checker`：证明检查
- `/citation-audit`：引用审查

这种"skill as file"的设计 inherit 自 Claude 的 Agent Skills 系统，但 ARIS 把它推到了完整科研工作流的 scope。

工程上几个亮点：
- **MCP 接入**：通过 Model Context Protocol 集成不同模型作为 reviewer
- **Persistent research wiki**：跨 session 的发现累积
- **确定性 figure 生成**：FigureSpec renderer，避免 AI 生成图表的不可复现性

### 编排层（Orchestration）：5 个 Workflow

![图4：ARIS 系统拓扑](https://arxiv.org/html/2605.03042v1/x5.png)

*图4：ARIS 整体拓扑。六个组件群（Meta-Optimization 外环、Assurance 检查、Artifacts 产物、Workflow Orchestration、Skill Library、Tooling）通过标注的关系交互。Meta-Optimization 是 self-improvement 外环，gates 整个 Assurance 层。*

五个 end-to-end workflow：

| Workflow | 阶段 | 做什么 |
|----------|------|--------|
| 1: Idea Discovery | Discovery | 文献调研、跨模型生成想法、新颖性验证 |
| 1.5: Experiment Bridge | Experimentation | 实现脚本、代码 review、GPU 部署 |
| 2: Auto Review Loop | Experimentation/Manuscript | 跨模型 review 评分、提 action item、循环修订 |
| 3: Paper Writing | Manuscript | outline → figure → LaTeX → 五轮编辑 → claim audit → compile |
| 4: Rebuttal | Post-Submission | 解析 review、起草回复、stress-testing、三个安全闸 |

![图5：Workflow 2 Auto Review Loop 细节](https://arxiv.org/html/2605.03042v1/x3.png)

*图5：Workflow 2 - Auto Review Loop。每轮把 draft 提交给跨模型 reviewer 做结构化打分，提取 action items，可选触发 GPU 实验补充证据，修订受影响章节，收敛检查后进入下一轮或终止。*

![图6：Workflow 3 Paper Writing Pipeline](https://arxiv.org/html/2605.03042v1/x4.png)

*图6：Workflow 3 - Paper Writing Pipeline。三个阶段：Plan & Generate（outline、figures）→ Draft & Assure（LaTeX 起草，配五轮编辑、可选证明检查、claim audit）→ Compile & Inspect（编译 + 视觉检查 PDF）。*

### 保障层（Assurance）：Evidence-to-Claim Audit Cascade

这是 ARIS 我认为最关键的设计——

![图7：Evidence-to-Claim Audit 三阶段](https://arxiv.org/html/2605.03042v1/x7.png)

*图7：证据到 claim 的审计级联。Stage 1（experiment-audit）：reviewer 审计评估脚本和结果文件的 integrity；Stage 2（result-to-claim）：把结果映射到显式 claim ledger；Stage 3（paper-claim-audit）：交叉检查 manuscript 的每一句话是否能在 claim ledger 和原始证据里找到支持。*

**三阶段 Audit Cascade**：

1. **Stage 1: Experiment-integrity audit**
   - 检查评估脚本是否有 bug、metric 实现是否正确
   - 检查 result 文件格式、是否有 silent fail / NaN / OOM
   - **抓 fake ground truth、score normalization fraud 等"实验作弊"模式**

2. **Stage 2: Result-to-claim mapping**
   - 把每个数字结果**显式映射**到一个 claim
   - 形成 claim ledger（claim 账本）
   - "我们提升了 10% F1" → 必须能定位到具体 table / config / seed

3. **Stage 3: Paper-claim audit**
   - **逐句审查 manuscript**：每条声明能否在 claim ledger 找到支撑
   - 标注 "supported" / "partial" / "unsupported"
   - **unsupported 必须修改或删除**

我用 Claude / Codex 做 research 这一年，最痛的就是 stage 3 这个 gap——

> 模型在 paper 里写一句"Our method outperforms baselines on most benchmarks"——看起来 plausible，但什么叫 "most"？6 个 benchmark 赢了 4 个就敢这么写。**plausible unsupported success** 的活样本。

ARIS 把"逐句 audit"做成了系统的一部分。这是把 trust 工程化的关键。

**Manuscript Assurance** 还有：
- **Five-pass scientific editing**：术语统一、引用规范、行文一致性等
- **Proof verification**：理论证明独立检查（用 `/proof-checker` skill）
- **Visual PDF review**：实际渲染 PDF 后视觉检查（图表 overflow、表格断行等）
- **Citation audit**：每条引用核对作者、年份、出处真实性

### 五层 Effort 等级 + Reviewer Routing

ARIS 的 effort levels：从 quick（最低保障）到 max / beast（最严审）。
不同 level 控制：
- reviewer 模型用哪个（GPT-5.4 / Claude / Codex 等）
- 几轮 review loop
- audit 阶段开几个
- 提交前必须过哪些 gate

这种"配置 effort"的设计非常工程化——研究早期用 quick 快速迭代，投稿前切到 max 跑全套 audit。

### Meta-Optimization 外环

最高一层的 **self-improvement loop**——
- 记录每次 research session 的 trace
- 自动提出"哪个 skill 表现差"、"哪个 workflow 容易卡"
- **改动**必须经 reviewer 批准，**不允许 self-modify 自己**

这避免了"AI 让自己变得更宽松"的 alignment drift 问题。

---

## 我的判断

**亮点**：

- **诊断准、定位狠**：把 long-horizon agent 的核心失败模式定义为 "plausible unsupported success" 而不是 "visible breakdown"，这个 framing 非常深刻。
- **跨模型对抗作为默认配置**：把"用不同家族 LLM 互相 review"作为 default，让 alignment bias 互相 cancel——比 self-refine 强一个数量级。
- **Markdown skill 模块化**：65+ 个可组合 skill，比"一个超大 agent prompt"工程上可维护性强太多。这条线值得所有做 agent 系统的人借鉴。
- **Audit Cascade 把 trust 工程化**：claim-ledger + 逐句 audit 是真正的 industrial-strength assurance，不是 demo level 的"看起来对"。
- **开源 + 可配置**：完整 release 在 GitHub，可以本地跑、可以配自己的 reviewer model、不锁定厂商。
- **意识到 alignment drift 风险**：Meta-Optimization 外环不允许 self-modify，需要 human-approved 才落地。这种克制在 self-improving system 设计里很难得。

**问题**：

- **成本很高**：跨模型 review + 多阶段 audit + 五轮 editing，每完成一篇 paper 烧的 token 估计是单纯生成的 10× 起步。论文里没看到详细的 dollar cost / token cost 分析。
- **延迟也大**：很多 workflow 是串行 review-revise loop，wall-clock 时间可能要几小时甚至几天才出一个版本。
- **缺乏 controlled benchmark**：论文 Appendix E 标明 "Controlled Benchmark Protocol (Future Work)"——说明现在的 evidence 主要是 deployment experience，没有 head-to-head 量化对比 AI Scientist v2 / Agent Lab 之类的工作。
- **依赖外部模型 API 稳定性**：跨模型对抗对 API rate limit / 模型版本切换非常敏感。生产环境的 reliability 是个工程难题。
- **"reviewer 也会错"的问题没完全解决**：不同家族模型的 review 能 cancel 一部分偏见，但不能保证 review 自身正确。Stage 3 的 audit 还是依赖模型，本质上是"用模型 audit 模型"，存在天花板。

---

## 工程启发

如果你在做：

- **长跑 agent 系统**（autonomous coding / research / analysis）：**默认就要上跨模型 reviewer**。让 GPT 给 Claude 当裁判、让 Codex 给 Gemini 当裁判。**别再用 self-refine 了**。
- **企业内部 AI 工作流**：把"审查"独立成 separate model + separate context。reviewer 看不到 executor 的内部 reasoning，只看产出物——这种 information asymmetry 是 cross-check 的关键。
- **科研 / 写作 / 报告生成**：把"claim ledger"作为产出物的 first-class 输出。每一条数据声明、每一条对比、每一个数字，**显式映射到原始证据**。这比单纯"输出 Markdown"高一个层次的 deliverable。
- **长链推理产品**：把 effort level 暴露成用户配置——"快速答案" vs "审慎答案"，让用户根据场景权衡 cost / latency / reliability。
- **agent 工具链设计**：模仿 ARIS 的 "Markdown skill" 设计，把 capability 模块化为可组合文件。维护性、可观测性、版本管理都会好太多。

最后说一个个人 takeaway——

**LLM 自主科研的最大瓶颈不是"模型不够聪明"，而是"模型太聪明地骗自己"**。ARIS 把这件事讲透了——你给它一个长任务，它会用尽全力让 output 看起来合理。这是 RLHF + alignment 训出来的 super-power，但在 high-stakes 场景下也是 super-bug。

唯一的解药就是**异质监督**——用一个**不一样的 mind** 当 reviewer。ARIS 把这件事系统化了。

下一代 autonomous AI system 的核心竞争力不在 reasoning ability，而在 trust engineering。ARIS 是这条路上的早期典范。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
