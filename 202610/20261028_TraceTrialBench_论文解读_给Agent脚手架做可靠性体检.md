# 任务跑通了，但你的 Agent 框架可能在「撒谎」：3,420 次运行揭开 31.1% 的侥幸成功

## 🎯 一个让人后背发凉的数字

先说一个场景，做 Agent 产品的朋友大概率遇到过：同一个模型，套在 Claude Code 里是一个表现，套在另一个自研 harness 里又是另一个表现。任务成功率差不多，但线上事故率差很远。你盯着最终交付物看了半天，结论永远是「模型不太行」或者「模型又行了」——**至于中间那层脚手架到底干了什么，没人说得清**。

这篇论文甩出一个数字：**31.1%**。在他们跑的 3,420 次 Agent 任务里，超过三成的运行，任务明明「成功」了，但 harness 层至少有一条可靠性义务没履行。可能是 stderr 被吞了，可能是退出码被伪造了，可能是子智能体的完成状态根本没传回来——但最终产物是对的，于是所有人都假装无事发生。

反过来也成立：有些任务失败了，harness 却从头到尾忠实履职，锅完全在模型的决策上。只看任务结果，这两种情况根本分不开。

这就是 TraceTrial-Bench 要解决的问题。

**核心摘要**：这是第一个系统化评测 Agent harness（脚手架）可靠性的基准。作者把 harness 可靠性拆成七个组件——上下文、执行、治理、生命周期、可观测性、工具、验证，构建了 95 个机制导向的专家级任务，每个任务都带可执行或基于状态的真值锚点。配上轨迹追踪（TT）、多源证据采集（MEA）、失败归因（FA）三个模块，在 4 个 harness × 9 个模型上跑了 3,420 次。结果：harness 宏平均分从 59.8% 到 72.0%，任务成功率最高的 harness（Claude Code 44.0%）反而不是 harness 义务履行最好的（OpenClaw 72.0%）。我的判断：这不是一篇「刷榜」论文，是一篇「立规矩」的论文——它给 Agent 工程领域补上了一直缺失的计量学基础。

**论文信息**

- 标题：Trial by Trace: Benchmarking AI Agent Harnesses for Reliability and Failure Attribution
- arXiv：[2610.32285](https://arxiv.org/abs/2610.32285)，2026 年 10 月 23 日提交
- 作者：Zhiyong Cao、Hengjie Zhu、Yongxin Wang、Xuchen Li、Jintao Li、Shuxia Liu 等 23 人

---

## 📖 为什么「看结果」不够用

先厘清一个概念。论文给了一个很干净的定义：

$$\text{Agent} = \text{Model} + \text{Harness}$$

模型负责选动作，harness 负责剩下的一切：组装 prompt、传工具、管上下文压缩、跑权限审批、维护会话生命周期、把工具结果喂回给模型。我们平时说的「Claude Code」「Codex CLI」这些，模型只是其中一半，另一半全是 harness 工程。

问题在于，现有基准几乎全在测「Agent = 整体」的任务成功率。τ-Bench、SWE-bench、OSWorld 这类榜单，分数涨了你不知道是模型变聪明了还是 harness 的工程 trick 起效了，分数跌了你也不知道锅在哪。做 harness 的人想改进自己的产品，手里连个可复现的度量都没有。

![图 1：背景与动机](https://arxiv.org/html/2610.32285v1/background-cartoon.png)

*图 1：左栏点破核心痛点——任务结果会误导人。「成功的任务」可能是模型自己补救了 harness 丢失的信息（Agent pass / Harness shortfall）；「失败的任务」也可能发生在 harness 忠实执行的情况下（Agent fail / Harness faithful），左下角就是那个 31.1%。中栏给出系统边界：Agent = Model + Harness，模型选动作，harness 管上下文、工具与控制，外部工具和环境在 harness 边界之外。右栏是 TraceTrial-Bench 的解法：用原生轨迹、工作区产物、环境状态三类证据，先过准入闸门（apparatus admission gate），再把 Agent 任务结果和 Harness 原生义务分开评分，最后做基于证据的失败归因。顶部一排就是七个可靠性维度。*

这个思路说实话挺戳我的。我之前调 Agent 系统时就碰到过类似的事：线上一个 case 显示「工具调用成功」，但模型实际拿到的是被截断的输出，剩下的内容靠它自己脑补——最后还补对了。任务成功，过程造假。没有轨迹级的证据，这种事永远发现不了。

---

## 🏗️ 七组件框架：把「可靠性」拆成可测量的东西

作者做的第一件事，是给「harness 可靠性」这个模糊概念建立分类体系。关键设计原则是：**维度由被测的 harness 义务决定，而不是由应用场景决定**。同样是「状态隔离」，如果考的是信息有没有跨过会话边界，归 Context；如果考的是子智能体的创建与回收，归 Lifecycle。

| 维度 | 评估焦点 | 任务数 |
|---|---|---:|
| Context（上下文） | 跨边界的信息保存与披露 | 15 |
| Execution（执行） | 忠实的进程语义、状态与结果传递 | 27 |
| Governance（治理） | 审批、护栏、秘密信息脱敏、信任边界 | 9 |
| Lifecycle（生命周期） | 正确的状态转移、恢复、幂等重试 | 19 |
| Observability（可观测性） | 模型可见的诊断、可回放日志、来源溯源 | 7 |
| Tools（工具） | 有效的工具接口、忠实的传输、工具发现 | 17 |
| Verification（验证） | 完成声明与权威结果的一致性 | 1 |

95 个任务覆盖 **40 个机制级可靠性属性**。每个机制要被纳入，必须满足一个硬条件：触发条件、预期行为、观察边界、判定条件，四者在执行前就能写死。做不到这一点的「可靠性感觉」，一律不收。

![七组件与 40 个机制级叶子属性的分布](https://arxiv.org/html/2610.32285v1/figures/tracetrial-leaf-distribution.png)

*3.2 节的 taxonomy 旭日图：中心是 40 个叶子节点、7 个组件。内环占比——Tools 10 个叶子（25%）、Execution 9 个（22.5%）、Lifecycle 8 个（20%）、Context 6 个（15%）、Governance 4 个（10%）、Observability 2 个（5%）、Verification 只有 1 个（2.5%）。外环是每个叶子的具体机制名，随便挑几个都很有意思：Exit-code fidelity（退出码保真）、Stale-artifact rejection（拒绝过期产物）、PTY/EOF handling、Duplicate tool names（重名工具）、Compaction detail fidelity（压缩细节保真）、Pending children at exit（退出时未回收的子任务）——全是工程里真实踩过的坑。*

看这张叶子分布图我有个直观感受：这个 taxonomy 不是从文献里抄来的，是从事故报告里长出来的。「Submitted vs. confirmed」（提交了就当真了，没等确认）这种叶子，没被生产环境毒打过的人编不出来。

也要泼一盆冷水：Verification 维度只有 1 个任务，这个维度的任何结论都谈不上统计意义。作者自己也在局限性里承认了覆盖极薄。

## 🔧 95 个任务：机制导向，真值可执行

任务设计是这套基准最花力气的地方。每个任务从一个 taxonomy 子维度出发，固定四件事：要测的原生机制、被测属性、模型可见的后果、外部真值锚点。然后用「普通的领域工作」把机制触发包起来——模型不会意识到自己在被考。

每个任务包里钉死这些合同：

- **Task and pressure**：prompt、fixtures、机制触发器、资源限制；
- **Agent score sheet**：必需动作与最终交付物——从产物和权威状态判定任务完成度；
- **Harness score sheet**：原生义务、观察边界、证据选择器——判定信息与执行行为是否忠实；
- **Oracle and provenance**：私有状态、工作区基线、运行身份——把环境真相和模型能看到的东西隔离开。

任务构成：46 个单轮、45 个固定多轮、4 个模拟器驱动。触发器的例子包括「需要恢复的超大输出」「配着成功外表产物的非零退出码」「挂起的任务」「跨会话边界的事实」「只能走交互通道才能拿到的私有值」。

这里有一个概念值得展开：**「观察是边界特定的」**。一个名义上的「工具结果」其实有好几个不等价的表示——工具产出的外部结果、harness 转换后的版本、模型实际收到的载荷、原生会话存档里的记录、评估器看到的规范化事件。所以真值必须按问题选权威来源：环境/模拟器状态裁定「实际发生了什么」，变更后的产物裁定「最终持久化状态」，provider 请求证据只对工具声明的结构权威，模型面向的结果裁定「模型能看到什么」。

听着像废话？不是。市面上大量「Agent 评测」的 bug 就出在拿错了权威来源——用 harness 的日志去裁定环境状态，用最终产物去裁定模型当时看到了什么。这篇论文把这些边界全部写成了显式合同。

---

## 🧪 三大模块：TT、MEA、FA

![图 2：TraceTrial-Bench 总览](https://arxiv.org/html/2610.32285v1/tracetrial-benchmark-main-v2.png)

*图 2：四大块的完整流水线。① Task Design：领域专家围绕 7 个可靠性维度设计机制导向任务，配 ground truth，经专家评审产出 95 个专家级任务。② Evaluation Framework：受控原生执行——模型 → 原生 harness（被测系统 SUT）→ 环境，固定设置保证原生行为不被替换；Trajectory Tracing（TT）产出执行轨迹，确定性规则 + LLM judge 分别评 Agent performance 与 Harness reliability。③ MEA：从 Agent 上下文、harness 记录、环境与产物三个来源汇聚证据，统一成 Trace IR，按 criterion 抽取证据包，带完整性检查。④ FA：把 Harness verdicts、证据包、Agent verdicts 放进机制链（Harness failure → 可观察的 distortion → Agent consequence），候选链接加证据复核后给出运行级归因。实线是工作流，虚线是证据流。*

### Trajectory Tracing：把各家黑话统一成普通话

四个 harness 的原生记录格式完全不同。TT 模块的做法是给每家写一个 adapter：Codex 读 JSON 事件流，Claude Code 结合 stream-json 输出与会话记录，OpenClaw 读本地或网关后的 JSON 输出和 session JSONL，DeepSeek Harness 消费持久化的 JSON-RPC notification 流。全部归一化成共享的 Trace IR：按连续 seq 排序的 JSON 事件数组，每个事件带 schema_version、seq、timestamp、run_id、event_type 的信封。

规范事件分五族：生命周期（run_started/finished/failed）、模型界面（model_input、assistant_message 等）、动作/观察（tool_call、tool_result）、状态控制（context_compaction、subagent_lifecycle、policy_event）、记账/归档（token_usage、原始 stdout/stderr）。

有两个工程细节我很欣赏。一个是 call-result 关联：**只用 harness 原生 call ID 配对，对不上的、延迟的、重复的记录，宁可显式标记为未链接，也不搞启发式硬凑**。做过多并发工具调用日志分析的人都知道，启发式配对是评测数据污染的温床。另一个是工具可用性的裁定锚在 provider 边界实际提交的接口声明上，而不是 harness 的静态配置——配置里写了不等于真发出去了。

### MEA：证据采集，且「有界但不替代归档」

MEA 的核心是把**归档证据**和**判分证据**分开。归档全部保留用于审计；每个评分项只拿到自己 score sheet 里声明的选择器产出的有界证据包。

证据来源有六类，各自的「认知范围」写得很清楚：模拟器账本与最终状态（裁定精确请求、发出的值、不可逆状态）、工作区基线与变更产物（裁定交付文件）、归一化轨迹（裁定模型可见内容）、provider 请求证据（裁定工具声明）、执行/安全探针（独立于文字报告，裁定进程处置、超时、审批）、Agent 自己的声明（只代表它说了什么）。

选择器是声明式的：每个语义 rubric 声明 eligible 事件类型、字段、字面匹配条件、返回记录上限。超长字符串只保留边界段和关键词窗口，大集合保留完整匹配条目并附 reduction 记录，证据包自己报告 scanned/matched/returned 计数。**返回上限在评估前就被归档的 score sheet 钉死，judge 模型在评估时改不了**——这个设计直接把「judge 想看更多证据直到满意为止」这类微妙的偏差源堵死了。

### Failure Attribution：归因可以自动，但因果必须人工

FA 是我最欣赏也最想吐槽的模块，先说设计。

形式化上，$F_H = \operatorname{fail}(V_H)$ 是 harness 层失败集合，$Z_H = F_H \setminus B$ 去掉被测量旁路影响的项（比如 wrapper 命令用自己的退出码覆盖了被测命令的退出码），$Z_A = \operatorname{fail}(V_A)$ 是 Agent 层失败集合。任务执行前可以预声明机制链：

$$\ell = (h_\ell, a_\ell, \mu_\ell)$$

两个 criterion 端点加中间的 distortion 描述。运行时如果端点都在失败集里且激活条件满足，这条链进激活集 $L_{\mathrm{act}}$。

关键来了：**激活只证明「端点出现了、条件匹配了」，不证明失败真的沿这条链传播了**。候选链接要晋升成 validated，需要人工复核事件顺序、找到与声明机制一致的可观察中间 distortion、考虑来源和竞争性解释。

最后每个 run 按七条有序决策规则打六种标签之一：inconclusive_device（测量装置自己挂了）、inconclusive_unmeasured（某层没测到）、harness_caused（候选链激活）、agent_caused（有 Agent 侧问题但没有链接）、harness_defect_absorbed（harness 有零分但 Agent 全过）、clean。

我很喜欢作者的坦诚：这些标签是「操作性证据状态」，不是因果结论。agent_caused 不等于 harness 没毛病，harness_caused 只表示候选链激活。但说实话，这也就是说**自动归因的直接产出相当保守**——真正值钱的 validated propagation 还是要人看。论文没给出 corpus 级的 candidate-link precision，这是评估链上明显缺失的一环。

---

## 📊 两层评估与主实验：排名错位了

先准入，后解释。container 生命周期、归档完整性、轨迹可评分性、探针完整性、私有 oracle 可用性，全在语义评分之前验。准入不是第三个分数层，它不能补偿任何一层——只是保证你评的东西是真的。

评分协议很讲究顺序：**确定性规则先跑并钉死，Agent rubrics 先评，结果完全固定后再评 Harness rubrics**。Agent verdict 只作为下游上下文提供给 Harness 层——可以描述影响，但绝不能作为「harness criterion 失败」的证据。组合分：

$$S = 50\frac{p_H}{n_H} + 50\frac{p_A}{n_A}$$

只有 $S = 100$ 才算通过。

![图 7：准入与分层评估顺序](https://arxiv.org/html/2610.32285v1/layered-evaluation.png)

*图 7：四步流水线。① Admission checks：run id、lifecycle、archive、trace complete、probes intact 五项准入检查。② Deterministic rules：format、files、action 三类确定性规则加探针检查，先于任何 judge 调用钉死。③ Agent layer：基于 artifacts、actions、response 评分，$V_A = p_A/n_A$。④ Harness layer：基于 native trace、external state、semantic judge 评分，$V_H = p_H/n_H$。虚线分隔——两层保持独立的证据包与 verdict 向量。*

Judge 用 deepseek-v4-flash，输出四态：0 Unsupported（违规或证据不足）、1 Supported、2 Not exercised（机制没跑到，由 Agent 结果兜底解析）、3 Malfunction（机制到了但 harness 自己故障，直接判负）。

实验矩阵：4 个 harness（Claude Code、Codex、DeepSeek Harness、OpenClaw）× 9 个模型组（DeepSeek-V4-Pro、Gemini 3.7 Flash、GLM-5.3、GPT-5.6-Sol、Grok-4.6、HY3、Kimi K3、Opus 5、Qwen3.8-Max）× 95 任务 = 3,420 次运行。Docker 隔离执行，墙钟限制 120 到 5,400 秒，任务跨度 1–120 轮输入、5–300 次工具调用。

主结果（Table 6，每个 harness 855 个匹配单元格，单位 %）：

| Harness | Agent Pass | Agent Score | Harness Micro | Harness Macro | Clean | Overall |
|---|---:|---:|---:|---:|---:|---:|
| Claude Code | **44.0** | **75.9** | 71.5 | 71.2 | 9.1 | **73.5** |
| Codex | 35.8 | 70.7 | 60.2 | 59.8 | 2.8 | 65.2 |
| DeepSeek Harness | 40.0 | 73.3 | 69.5 | 68.8 | 9.8 | 71.0 |
| OpenClaw | 38.2 | 71.7 | **71.9** | **72.0** | **11.7** | 71.8 |
| 全部 | 39.5 | 72.9 | 68.3 | 67.9 | 8.4 | 70.4 |

几个数值得单独拎出来：

**排名错位**。任务成功率最高的是 Claude Code（44.0%），harness 义务履行最好的是 OpenClaw（macro 72.0%）。如果只看任务榜单，你会得出「Claude Code 是最强 Agent 系统」的结论；但这个系统里有一部分成功，是模型在 harness 丢信息的情况下自己补救回来的。这两种「强」对工程决策的含义完全不同——前者可以换模型继续榨，后者的坑换什么模型都在。

**8.4%**。这是两层全部 criterion 都通过的 run 占比。对比连续平均分 70.4、Agent 全过率 39.5%，差距悬殊。你想想看，平均每个 criterion 七成通过率听着还行，但「所有义务都履行了」的干净运行不到一成。可靠性是乘法不是加法。

**12.2 vs 8.2**。harness macro 的极差（59.8% 到 72.0%）比 Agent 任务成功率的极差（35.8% 到 44.0%）还大。固定同样的模型和任务，harness 之间的可靠性差异，已经大到超过它们对任务结果影响的差异。这个发现对「选哪家 Agent 框架」的决策权重，应该比现在行业惯例高得多。

Codex 垫底（macro 59.8%，clean 只有 2.8%）这个数也挺扎眼，但先别急着下结论——下面局限性部分会说为什么。

---

## 🔬 批判性审视：规矩立得好，但有几处得留心

**「首个」成色如何？** Agent 评估这两年很热，τ-Bench、AgentBench、OSWorld 都在测整体能力，学术界也有专门看工具调用、规划可靠性的工作。但把 harness 当作独立被测对象、建分类体系、配可执行真值、再做机制级归因的，据我所知这确实是第一个成体系的。工业界各家（包括这几个 harness 的厂商自己）内部肯定有类似的回归测试，公开、可复现、跨 harness 对齐语义的就是另一回事了。这个贡献是真的。

**但 3,420 次运行的统计含金量要打折**。论文自己承认：归档的是「选定结果」，包含早期失败后的恢复性重跑，不是每个单元格独立重复采样。所以它描述的是观察到的矩阵，不估计单元格内的可重复性。 harness 间差异不能做强因果解释——尤其是各家 adapter 成熟度、恢复策略都可能影响最终数字。Codex 垫底可能反映可靠性差距，也可能部分反映适配偏差，论文的框架本身无法区分这两者。

**judge 预算不统一是个小污点**。judge 身份固定（deepseek-v4-flash），但输出预算有三档：4,096 tokens（1,966 份）、8,192（1,452 份）、32,000（2 份）。对语义评分任务，输出预算影响评分宽松度是已知的偏差源。2 份 32,000 的无所谓，但 4,096 和 8,192 之间近 1,500 份的差异，论文没有给分层分析。

**归因模块的自动化程度别高估**。candidate link 激活是自动的，但 validated propagation 全靠人工。论文没报告人工复核的规模和一致性，也没报告 candidate-link precision。也就是说，「harness_caused」这个标签在全语料上占多少、准多少，目前是开放问题。

**成本问题论文几乎没谈**。95 个任务 × 每个模型-harness 组合，加上多源证据归档、语义 judge、人工归因复核——这套流程跑一轮的算力和人力成本不低。对于想拿它做日常回归测试的团队，成本结构可能比评测方法本身更劝退。这块我自己也没完全想清楚怎么摊薄，可能分层抽样是个方向。

---

## 💡 我的判断

这篇论文最值钱的不是任何单一数字，而是它把「harness 可靠性」从玄学变成了可计量的东西。七组件 taxonomy、边界特定的权威真值、声明式证据选择器、准入闸门、保守归因——每一环都在堵评测造假的洞。它更像计量学论文而不是模型论文：不告诉你谁最强，告诉你该怎么量。

对工程实践的直接启发：

如果你在做 Agent 产品，31.1% 这个数字应该让你今晚就打开日志系统，抽查「任务成功但过程异常」的 case——stderr 被吞、退出码被 wrapper 覆盖、子任务状态丢失，这三类是高频中的高频。

如果你在选型 Agent 框架，任务成功率榜单的权重该降了。问厂商要 harness 层的可靠性证据：轨迹可回放吗？工具结果保真吗？上下文压缩后细节还在吗？这些问题现在有了标准化的问法。

如果你在做评测研究，这套「先准入后解释、确定性先行、双层独立 verdict」的协议设计，值得原样搬到你自己的基准里。

遗留的大问题也有一个：这套框架评的是「harness 有没有履行义务」，但还没回答「义务清单本身完备吗」。40 个机制级属性是专家经验的上限，不是失败模式的上限。生产环境的黑天鹅会不断往这个 taxonomy 里加叶子——从这个角度看，TraceTrial-Bench 更像一个会持续生长的开源工程，而不是一篇发完就完事的论文。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
