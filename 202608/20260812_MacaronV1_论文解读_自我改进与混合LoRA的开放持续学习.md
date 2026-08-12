# Macaron-V1：冻结744B底座，用4个LoRA和一套自我改进循环，把"部署即固化"撕开一道口子

你有没有过这种感觉——一个 Agent 刚上线的时候还挺能打，但用了三个月，它还是在犯三个月前犯过的错。用户骂了、日志记了、badcase 攒了一硬盘，可模型是个冻结的 checkpoint，什么都改不了。想改？那就重新训练、重新评估、重新发布，走一遍以月为单位的流程。等你发版，世界又变了。

这其实是当前大模型后训练范式的一个结构性尴尬：模型在一批有界任务上训练、跟当时的 harness 对齐、固化、发布。做得再好，也只是在逼近一个**静态最优**。而真实世界的知识、工具、需求是持续涌进来的。Mind Lab 这份 49 页的技术报告 Macaron-V1，就是冲着这个口子来的——他们管它叫 **experiential intelligence（经验智能）**：在真实环境中从经验学习，部署之后继续学。

## 核心摘要

Macaron-V1 是一个围绕两个系统目标搭建的开放 Agent 模型家族。**适应**靠版本化 model-harness 对的递归自我改进：一个配置产生的经验在外部契约下被评估，用来构造它的后继版本。**协作**靠 Mixture-of-LoRA（MoL）架构：冻结 base model，挂多个专家 LoRA，每个用户轮次选一个。旗舰 Venti 用冻结的 744B GLM-5.2 加 4 个 1B 级 LoRA（对话、Agent、编程、GenUI），在内部 Personal Intelligence 基准上拿到 ChatBench 58.3、LivingBench 64.0，UI4A-Bench 上以 87.8 对 75.9 领先 Opus 4.8，TerminalBench 2.1 报出列表里最高的 87.6。我的判断：这不是一篇"刷榜论文"，而是一套**把模型、 harness、基础设施当做一个系统来协同设计**的工程宣言——真正的价值在架构和流程，基准数字反而是次要证据，而且作者自己对数字的局限标注得异常坦诚，这点很少见。

## 论文信息

- **标题**：Macaron-V1: Towards Open Continual Learning with Self-Improvement and Mixture-of-LoRA
- **作者/机构**：Mind Lab（团队署名，Vin Bo、Asher Cai、Jingwei Cao 等 70+ 位成员）
- **提交日期**：2026 年 8 月 10 日
- **链接**：https://arxiv.org/abs/2608.09819
- **开源**：模型见 Hugging Face `mindlab-research/macaron-v1`；serving harness 开源在 `MindLab-Research/Mixture-of-LoRA-Harness`

---

## 问题动机：后训练被环境绑死了

后训练时代有个不太被明说的依赖关系：后训练的有效性，和模型训练、评估、部署所处的**环境**是紧耦合的。你把模型在一批任务上优化好，跟当前的 harness 对齐，然后固化成 checkpoint 发出去。异构目标通过共享参数竞争，会产生跨任务干扰；更根本的是，系统被绑死在训练时可见的任务分布上。

作者把这个痛点拆成两条设计承诺，我觉得这两条比后面的所有技术细节都更能代表这篇报告的气质：

1. **模型越强，harness 应该越轻**。脆弱的 schema 是天花板，让通用模型干大部分活，专家 adapter 只留给真正需要判断的决策。
2. **训练和服务共享显式的 runtime 契约**。训练时用的 harness 配置和线上跑的必须是同一份可审计的东西，配置漂移要可见。

说实话，第一条我在自己的项目里踩过坑——schema 定得太死，模型能力一上来，schema 反而成了限制。这个判断我认。

---

## MoL 架构：冻结底座 + 热插拔权重

![三种能力扩展方式对比](https://arxiv.org/html/2608.09819v1/figures/mol_three_ways.png)

*图：三种扩展模型能力的方式。MoE 在 base 内部扩展专家（冻结权重）；Skills 在固定模型外做上下文工程（热插拔 context）；MoL 冻结 base，通过 Proxy 路由组合 LoRA adapters（热插拔权重）。作者注明这是实现基板说明，不是受控对比。*

MoL 的设计原则只有一条规则：**把技能和思维模式相近的任务聚到一个 LoRA，技能差异大的任务分开**。派生出两个性质：base 冻结（新能力靠注册 adapter 添加，不覆盖 base 权重）；adapter 可移植（不同团队训的 specialist 能在同一个 runtime 里组合）。

Venti 的配置：冻结 744B GLM-5.2 base，挂 4 个 release-labeled 1B 的 LoRA specialist——

| Adapter | 名称 | 职责 |
|---|---|---|
| L0 | Chat | 对话骨干、指令遵循、模型身份，**同时充当路由入口** |
| L1 | Agent | 长程、重工具使用任务（个人 Agent 工作流、服务集成） |
| L2 | Coding | 代码生成、SWE 类任务、终端使用 |
| L3 | GenUI | UI4A 渲染与 UI 驱动动作，专精 TSX |

adapter 配置 rank 16、alpha 32，每个含 7,688,042,496 个存储值。加上 base，整个系统约 774.8B 逻辑参数。

### 路由：L0 自己就是 router

这里有个挺反直觉的设计——**没有独立的 router 模型**。路由由入口 adapter L0 自己的推理决定，"L0 is the router"。每个用户轮次走三阶段：

1. **Route**：L0 在 24 token 的严格解码预算下，把请求分类成 L0–L3 中唯一的 canonical label，constrained-decoding grammar 保证输出合法。
2. **Answer**：被选中的 specialist 从自己的会话视图回答，以之前轮次的跨 adapter 摘要为种子。
3. **Summary**：specialist 生成不超过 192 token 的摘要，Proxy 存服务端，不返回客户端。

外加两个短路机制：tool-result stickiness（工具结果轮锁定同一 adapter，跳过路由）和 transactional rollback（每轮前 checkpoint 会话状态，故障可恢复）。

你想想看，这个设计其实是把"路由"也当成了一种对话能力来训练，而不是外挂一个分类器。好处是 router 天然理解对话上下文；代价是每个请求多了两跳开销。多大？看数据：

| 跳数 | Venti 平均 | 占比 | Tall 平均 | 占比 |
|---|---|---|---|---|
| Route（24 tok 受限解码） | 0.54s | 12% | 0.20s | 11% |
| Answer（specialist 生成） | 3.17s | 68% | 1.24s | 70% |
| Summary（192 tok 上限） | 0.97s | 20% | 0.32s | 19% |
| 合计 | 4.68s | 100% | 1.76s | 100% |

路由加摘要合计约占 32%（Tall 30%）。这个开销不算小，但考虑到换来的是模块化能力组合，算是合理代价。

路由准确率方面，6,448 样本的 trace 上 Venti 达到 **99.12%**（6391/6448），Tall 99.04%，100% canonical-label 合规、零解析错误。分类别看，L2 Coding 和 L3 GenUI 几乎满分，残余错误集中在 L0/L1 边界——这符合直觉，"闲聊"和"办事"的边界本来就模糊。不过得泼一盆冷水：作者自己明确标注，这个 trace 来自 LoRA 训练数据，不是独立的 held-out split，只能算实现诊断，不能当泛化估计。这种坦诚我喜欢。

路由之后的质量也有验证：Vita delivery 任务上，直调 L1（无路由）reward 0.636，路由后 0.650——没退化。样本小（5 seed × 100 任务），作者也只说"未检测到退化"，不主张等价性。

### 部署侧的数字

MoL 存 774.8B 逻辑参数 vs 复制 4 份 merged base 的 2.976T，只占 **26%**，省 74% 权重驻留。8×B300 上 CP8 LayerSplit 把 900K token 冷启动 needle-test 的 TTFT 从 107.1s 压到 49.2s。作者特意声明：不主张比独立部署 merged specialists 有更低 TTFT 或更高吞吐——MoL 的卖点是存储和可组合性，不是单点性能。

---

## Model-Harness 协同设计：harness 也是可训练的

这是我觉得全报告最值钱的部分。传统思路里 harness（Agent 的脚手架）是工程师手写的，模型只负责在里头跑。Macaron-V1 把 harness 也纳入了可迭代的范围。

### UI4A：教模型"何时"渲染 UI 比"如何"渲染更重要

生成式 UI 有三条历史路线：HTML-native（表达力最大、控制最小）、schema-native（可验证但被组件目录上限锁死）、UI4A 走的是第三条——**component-native**：在 runtime 强制的边界内写普通前端代码。

心智模型是 import + component + state + Action。Action contract 里有个很妙的设计：每个用户手势是四字段结构——Origin、State、Execution、Visibility，其中 Visibility 含 `NoAI` 边界，模型不可见的字段。渲染框架无关，React、Vue、Svelte、SolidJS 从同一份输出渲染。

关键经验一句话：**教模型何时渲染 UI 比教它如何渲染更重要**。L3 adapter 专精的就是"何时渲染"和组件选择绑定。收益也实在：48 案例 gallery 上 raw HTML 约 1,224 output tokens vs UI4A 约 672，省约 45%；结合 streaming 和 partial rendering，time-to-first-render 最高快约 6 倍。

### REPL：有状态的动作底座

L1 Agent specialist 的动作表面是一个有状态的 Python REPL——持久 namespace，位于离散 function calling、MCP endpoints 和 shell 命令之上。

两个核心机制。**Executable composition**：依赖值作为变量持久化，依赖操作链在单轮内解决，模型不用把中间值复述来覆去。**Validated reuse**：`save_tool` 把自推导的 helper 存进候选池，`promote_tool` 要在 held-out reference 上私有验证通过后才能被后续调用。"先验证后提升"这个顺序是承重的，事后证明不良的提升会被记录并降级。

![REPL可执行组合对比](https://arxiv.org/html/2608.09819v1/figures/repl_executable_composition.png)

*图：左：离散 function calling，每个中间观察都要经模型文本往返（蓝色粗箭头宽度≈观察大小）；右：REPL 把 orders、net 等依赖值持久化在可执行状态里，单轮解决。*

这个机制的收益有多直观？看 case study：一个查订单、筛 SLA 违约工单、映射订单、算风险营收的任务，离散 function calling 用了 **48 轮**，REPL 组合只用 **6 轮**——因为 `orders`、`breached`、`oids` 都留在持久 namespace 里，可以对集合做 map 而不用一轮一轮来回。

但这里有个必须说的反例：在 BFCL v4 的 200 任务评估上，REPL 得分 49.5%，**反而低于** function calling 的 54.0%。原因是有 stateful observe-before-commit API 的场景下，有些依赖调用必须先观察到前序结果才能提交，REPL 没有组合优势。作者没有藏着这个结果，而是直接写进报告，并允许 L1 在这类场景回退到 shell 或离散调用。这种把负面结果摆上桌的做法，在现在的技术报告里真的不多见。

### HCP：版本化的 runtime 契约

Harness Context Protocol 是一个版本化 TOML 契约，用来从可移植、可审计的 artifact 重建 runtime。它把可训练性界定得很精确：HCP 本身不携带梯度（就是个声明式 TOML 文档），但模型可以**重写** HCP 描述的 harness——prompts、skills、tool allowlists 都是可寻址字段。这是语言空间里的 harness 自迭代，不是参数空间的。

![系统总览](https://arxiv.org/html/2608.09819v1/figures/system_overview.png)

*图：共享一个 harness 的三个 loop。左：MindForge RL Rollout（任务生成→Agent episode→Verifier reward→GRPO 更新 LoRA）；中：HCP 契约序列化生产 harness 上下文（router、memory、resource、prompts、tool-call tokens），使训练与服务配置可显式比较；右：生产服务。三者底下是同一个 Agent Harness。*

这张图把整篇报告的核心思想说清楚了：训练和服务跑的是**同一份** harness 契约，配置漂移无处遁形。三个发布时钟也由此解耦——base 跟着 GLM-5.2 这类平台模型走，specialists 经 RSI 流程更新，harness 不用动权重可以更短周期发版。

---

## MindForge 与递归自我改进循环

MindForge 是 RSI 循环的控制平面。优化对象分得很清楚：$\theta$ 是冻结的稀疏 base 参数，$\phi$ 是可训练的 LoRA 参数，$c$ 是版本化 harness 配置。策略写成 $\pi_\phi(a_t \mid o_{\le t}; \theta, c)$，两条更新路径——模型优化改 $\phi$（$\theta$ 不动，用 GRPO 更新 adapter），配置搜索选新 $c$（HCP 不是学习参数）。

谱系公式一句话概括：

$$(\text{problem bank}, \text{model}, \text{HCP}) \longrightarrow \text{evaluated trajectories} \longrightarrow (\text{dataset}, \text{next model}, \text{next HCP})$$

三阶段循环：Discovery（当前模型提出更难的任务变体，候选要满足质量良定义 + 当前模型还解决不了）、Expansion（固定 model-HCP 对执行，审计把失败定位到模型、任务还是 harness）、Update（轨迹筛选→GRPO 更新 LoRA→接受的 HCP 注册为下一代配置）。涉及工具暴露或安全边界的变更保留人工审查——这个闸门设计很务实。

### Expansion 实验：模型冻结，只改 harness，122 个失败任务全部拿下

这个实验我愿称之为全报告最有说服力的一个。设置：122 个 simulation 任务（来自 29 个 TerminalBench 2.1 源家族），选取标准是冻结的 GLM-5.2-FP8 base 在官方 reward 下**全部 not-pass**——所以起点覆盖率是 0/122。全程模型冻结，只改 HCP 携带的资源、skills、工具暴露和 hooks。

| 阶段 | Jobs | 尝试 | 通过 | Pooled | 覆盖率 |
|---|---|---|---|---|---|
| Retry control | 1–10 | 50 | 6 | 12.0% | 2/122 |
| Portfolio v1 sweep | 11–12 | 244 | 15 | 6.1% | 14/122 |
| Skill/HCP search | 13–48 | 76 | 49 | 64.5% | 60/122 |
| + Stop-gate hooks | 49–69 | 80 | 65 | 81.2% | **122/122** |

69 个 job、450 次尝试，累积唯一覆盖率在第 69 个 job 达到 122/122。对比之下，两次全集单配置 sweep 只拿下 4/122 和 11/122。最终阶段与全集 sweep 的 per-attempt yield 比是 **13 倍**。

解读一下：base 模型"不会"的 122 个任务里，相当大一部分其实不是模型不会，而是 harness 没给对工具和上下文。改配置（不动权重）就能解锁。这对做 Agent 产品的人是个重要提醒——**你以为是模型能力瓶颈的问题，可能一半是 harness 工程问题**。

当然，作者的边界标注同样严格：这是自适应配置选择下的覆盖上限，不是任何单一配置的泛化估计；13× 描述的是搜索轨迹而非 hooks 的因果效应；实验没执行 $\phi$ 更新，不测迁移泛化。

### 基础设施一瞥

MinT 平台管 LoRA RL 的模型状态谱系，核心区分 adapter revision（不可变快照）和 policy record（可变服务状态），trainer checkpoint 和优化器状态绝不直接跨 serving 边界。adapter-only 交接相比 merge 路径，Qwen3-4B 减少 **18.3 倍**，Qwen3-30B 减少 **2.85 倍**（路径特定延迟比）。还验证了百万级 adapter 目录的可寻址性——打包 $10^6$ 条 rank-1 adapter 目录零构建错误，当然作者也说了这不等于单引擎显存里能装一百万个 adapter。LongStraw 是 architecture-aware、response-only 的长上下文执行栈，支撑数百万 token 级操作点。稀疏 MoE base 上还有 R3 rollout routing replay、IcePop-style token masking 这些解决 rollout–learner mismatch 的手段。这块细节多但都是工程承重墙，不展开。

---

## 实验结果：赢在哪，输在哪

![主结果总览](https://arxiv.org/html/2608.09819v1/figures/eval_bar.png)

*图：Macaron-V1-Venti 主结果。上排：ChatBench、LivingBench、VitaBench、PinchBench、ClawGym；下排：SWE Verified、TerminalBench 2.1、DeepSWE、SWE Atlas QnA、UI4A-Bench。蓝色柱为 Macaron-V1-Venti，对比 GLM-5.2、GPT-5.5、Claude Opus 4.8、Gemini 3.1 Pro、Qwen 3.7 Max、Minimax M3。*

对比六个模型：Opus 4.8、GPT-5.5、Gemini 3.1 Pro、GLM-5.2、Qwen 3.7 Max、Minimax M3。十二个基准行，完整数字如下（带 ∗ 的是导入的公开榜单值，仅供上下文参考；行内不带星号的用同一协议评测）：

| 基准 | Macaron-V1-Venti | GLM-5.2 | GPT-5.5 | Opus 4.8 | Gemini 3.1 | Qwen 3.7 | Minimax M3 |
|---|---|---|---|---|---|---|---|
| ChatBench | **58.3** | 54.5 | 55.5 | 52.8 | 52.0 | 52.5 | 49.1 |
| LivingBench | **64.0** | 60.5 | 61.9 | 63.8 | 52.1 | 56.1 | 57.1 |
| VitaBench | 60.0 | 55.8 | 55.8 | 56.5 | 55.2 | **61.2** | 56.8 |
| VitaBench2 | 46.0 | 43.1 | 47.4 | 46.3 | **50.2** | 47.6 | 39.4 |
| τ³-Bench | **69.3** | 69.1 | 61.1 | 67.7 | 67.1∗ | 63.0 | 61.2 |
| PinchBench | **94.0** | 88.1 | 89.0∗ | 91.8∗ | 82.9∗ | 93.4∗ | 86.1 |
| ClawGym | 77.7 | 74.6 | **82.5** | 80.5 | 77.5 | 75.7 | 76.2 |
| SWE-Verified | 85.6 | 80.4 | 82.9∗ | **88.6∗** | 80.6∗ | 80.4∗ | 80.5∗ |
| TerminalBench 2.1 | **87.6** | 82.7∗ | 83.4∗ | 78.9∗ | 70.7∗ | 73.5∗ | 66.0∗ |
| DeepSWE | 58.4 | 54.9∗ | **70.0∗** | 58.0∗ | 10.0∗ | 18.0∗ | 20.0∗ |
| SWE Atlas QnA | 49.5 | 48.9∗ | 45.4∗ | **57.3∗** | 13.5∗ | 22.6 | 37.9 |
| UI4A-Bench | **87.8** | 67.1 | 72.1 | 75.9 | 60.3 | 62.5 | 63.0 |

怎么读这张表？赢面最大的是 **UI4A-Bench**：87.8 对 Opus 4.8 的 75.9，拉开 **11.9 个点**。而且分层看更有说服力——Constraint Adherence 领先最强 baseline 12.0 分（94.2 vs 82.2），Visual Quality 领先 6.6 分（90.0 vs 83.4），Interaction 领先 1.6 分（95.0 vs 93.4）。这不是"能编译就行"的差距，是信息组织、视觉层级、控件连线全链路的差距。考虑到 L3 就是专精这个的，算是 MoL 专业化路线最直接的证据。

Personal Intelligence 两项第一，但差距很小：ChatBench 58.3 只比 GPT-5.5 高 2.8 分，LivingBench 64.0 只比 Opus 4.8 高 0.2 分。而且这里有两个必须指出的坑：ChatBench 用的是私有 GLM-5.2 judge，跟被测的 GLM 衍生模型同族，可能天然偏爱 GLM 风格的回答；这两个内部基准还跟 RSI 循环共享源域和失败分类法——**相当于在自己参与命题的考卷上考试**。作者自己把这些都写明了，还说没有区间估计和 judge 敏感性分析，0.2 分的差异不应被解读为确立的优势。但话说回来，既然知道有这些坑，这两个第一的宣传价值就得打折扣。

外部基准上是有输有赢的真实格局：TerminalBench 2.1 拿了列表最高（87.6），但 VitaBench 输 Qwen 3.7 Max，ClawGym 输 GPT-5.5 和 Opus 4.8，SWE-Verified 输 Opus 4.8，DeepSWE 被 GPT-5.5 甩开 11.6 分（58.4 vs 70.0），SWE Atlas QnA 也输 Opus 4.8。没什么"全面碾压"，就是专业化的钱花在哪儿、哪儿就强。

50B 的 Tall 对比自己的 Qwen3.6 35B-A3B base，七个基准行全部更高，差距从 LivingBench 的 1.3 分到 UI4A-Bench 的 **25.4 分**（59.3 vs 33.9）不等。还有个有意思的发现：Tall 的 specialist 只用纯文本数据训练，但路由后的服务在 OCRBench、MMBench-EN、MMMU、MME cognition 上反而比原生 base 还高一点点，只有 MME perception 掉了 52.99 分。作者照例标注：没有重复级方差，不构成多模态能力保持的证明。

---

## 基准本身也是设计出来的

顺带说说这套内部基准，因为设计得确实用心。

Macaron ChatBench 用动态宪法结构：六条公理固定（被理解感、诚实谏言、真实声音、推进感、校准的亲密度、增长的自主性），judge 结合 persona 和场景实例化具体标准。七个场景各有一对内在张力——比如"情感支持"场景的核心张力是"接住情绪 vs 推进理解"，"工具任务"是"效率 vs 诚实"。这比固定 rubric 高明的地方在于，同一个行为对不同用户、不同场景的最优判断本来就不同。

![LivingBench架构](https://arxiv.org/html/2608.09819v1/figures/livingbench_final.png)

*图：Macaron LivingBench 系统架构。① 数据合成：从产品失败和 UX 信号到测试用例包；② 环境构建：用户建模、世界建模、四类噪声；③ 多轮交互沙盒：用户模拟器、动态世界状态、工具系统、噪声路由器围绕被测 LLM Agent；④ 评估：需求满足 judge + 过程质量 judge 双裁判。*

LivingBench 的四层噪声系统挺讲究：用户噪声（情绪/时间压力）、世界噪声（事实真的变了，考重规划）、工具噪声（事实没变但返回值被篡改，考交叉验证）、观察噪声（信息不对称）。双 judge 设计——需求 judge 查结果、过程 judge 查路径，按 0.7×需求满足 + 0.3×过程质量合成。

案例里有个科伦坡的场景很能说明双 judge 的价值：母亲摔伤疑似髋骨骨折、坚持要说僧伽罗语的女医生、用户有 11:30 必到的会、没有网约车。Agent 先推荐了医院，查到骨科顾问主要说英语后主动重规划为上门医生 + 750 米外诊断中心；用户说"我们强迫她，我不管她哭不哭"时，Agent 指出髋伤加双膝关节炎被强拽的风险，并建议用"Renuka 在等你"这种社会关系话术。结果分满分，但过程分只有 0.593——只看结果的 benchmark 会给这个 trace 打高分，过程 judge 把它拦下来了。

![UI4A案例对比](https://arxiv.org/html/2608.09819v1/figures/ui4a_model_comparison.png)

*图：同一 UI4A-Bench 案例上 Macaron-V1-Venti vs GLM-5.2。用户附了张刚修好的照片，想挑风格。Venti 把照片放在卡片顶部、记录并总结所选风格、给出可用的确认控件；GLM-5.2 只用文字描述风格、无选择状态、留了个禁用的按钮。*

---

## 我的判断

这篇报告最值钱的东西，按我的排序：

**MoL 是 LoRA 工程化的一次正名。** "冻结 base + 每轮选一个 LoRA + L0 自己路由"这套组合，把多任务干扰问题从参数空间挪到了路由空间。74% 的权重驻留节省、99% 的路由准确率、32% 的路由开销——数字都摆在明处。它不新鲜（LoRA 组合和专家路由都有前人做过），但做成一个开源的、三种 API 表面可用的生产系统，这是工程贡献。

**Model-Harness Co-design 是被低估的范式提示。** Expansion 实验里 122 个 base 全挂的任务靠改 harness 配置全拿下——这个证据比任何 benchmark 分数都更能说明"harness 是能力的一部分"。做 Agent 的同学真该想想，你的 badcase 里有多少其实该改的是 harness 而不是模型。

**坦诚度是这篇报告的隐藏亮点。** 路由准确率标注"来自训练数据非 held-out"、LivingBench 高 0.2 分主动说"不应解读为优势"、REPL 在 BFCL v4 上输给 function calling 直接写进报告、ChatBench judge 同族风险自己点破。在一个充斥着"全面 SOTA"话术的时代，这种自我设限的写法反而让我更信它的正面结果。

问题也得说。**collective intelligence 是空头支票**——摘要说协作是核心目标之一，但报告明确说独立训练的 specialist 组合出超越单体的能力"保留给更强设定"，本版未评估。RSI 循环的三阶段里，Update 阶段的 $\phi$ 更新在 Expansion 实验里也没执行——也就是说"自我改进"的**参数更新这条腿，报告里基本没有端到端验证**，验证的是 harness 那条腿。内部基准自评的问题前面说了。还有 Venti 发布标签 748B vs base 实际 744B 这种细节，虽然作者解释了是 release-facing 标签，但总让人觉得营销还是没忍住。

回到标题里的"开放持续学习"：这篇报告给的是通往持续学习的**系统骨架**——MoL 让能力可插拔，HCP 让 harness 可版本化，MindForge 让迭代可谱系化。但"持续学习真的在复利式增长"这件事本身，还是个开放问题，作者自己也是这么说的。

如果你在做 Agent 产品，值得细读的是 Section 2（MoL 服务化）和 Section 3（harness 协同设计），这两块的可迁移性最强。如果你等的是"模型部署后自己越变越聪明"的证据，这篇还欠着——但骨架搭好了，迭代起来就是时间问题。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
