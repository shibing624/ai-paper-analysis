# Harness 也能自我改进：Prime Agent 把 ARC-AGI-3 从 30% 拉到 95.5%，靠的是运行时而不是新模型

上周调一个长程 Agent 评测的时候又碰到那个老问题：模型明明有能力，跑到一半上下文被压缩丢了关键状态，任务就这么挂了。你说这算模型不行，还是脚手架不行？

说实话，这个问题在 Agent 评测圈已经烂大街了——很多 benchmark 测出来的根本不是"模型行不行"，而是"harness 拖不拖后腿"。Prime Intellect 团队这篇技术报告（arXiv:2608.23552）就是冲着这个痛点来的：他们造了一个开源的 Agent 运行时 Prime Agent，核心主张是——**模型失败应该是因为任务超出了能力上限，而不是 harness 丢了状态、限制了动作、算错了资源或者提前终止了进程**。

**核心摘要**：Prime Agent 不是一个新模型，也不是一个新的 prompt 技巧，它是一个面向长程任务的开源 Agent harness。三件事撑起整个系统：持久化的 IPython REPL（按 Recursive Language Model 抽象做程序化上下文处理）、Continual Harness（跨轨迹保留历史、记忆、技能、prompt 和子智能体规格）、以及支持直接 agent-to-agent 通信的递归子智能体。效果上最扎眼的数字是 ARC-AGI-3 RHAE Best@1 从 30% 拉到 95.5%（人类基线 95.4%），还能支撑 85.5 小时的 nanoGPT speedrun 跑出 19 条验证记录。我的判断：这不是底层算法突破，是一次完成度很高的工程整合，但它把一个被严重低估的变量——harness 质量——变成了可测量、可标准化的东西，这个价值是真的。

- 标题：Prime Agent: A Self-Improving RLM Harness
- 作者：Seth Karten, Alex L. Zhang, Kevin Thomas, Sebastian Müller, Elie Bakouch, Daniel Auras, Mika Senghaas, Fares Obeid, Konstantin Dunas, Johannes Hagemann, Sami Jaghouar
- 机构：Prime Intellect
- 提交日期：2026 年 8 月 24 日（16 页，10 图，技术报告）
- 链接：https://arxiv.org/abs/2608.23552 ｜ 代码：https://github.com/PrimeIntellect-ai/prime-agent

---

## 🎯 问题动机：模型测不出来，可能不是你的模型不行

一个 LLM 说到底是一个有界的顺序处理器——每一步决策能用的信息只有权重里的和活动上下文里的。但长程任务需要的信息和计算，远远超出这两样。

之前大家怎么补？工具调用补了"外部动作"，agentic compaction 补了"上下文管理"。但完整的信息状态早就溢出了权重和 token 上下文。论文用了一个我很喜欢的视角：把整个系统看成一个**状态信息的缓存层级**，就像 CPU 的 L1/L2/L3 缓存那样：

![图2：Prime Agent 状态层级——L0 是模型权重，L1 是活动上下文，L2 是 REPL 与子智能体，L3 是磁盘持久化的历史/记忆/技能；L1 与 L2 之间是模型上下文边界，每一层有各自的更新机制](https://www.mulanai.com/fs/files/0826_18b43b03_state-hi.png)

*图2：四层状态层级。L0 模型权重靠微调更新，L1 活动上下文靠 compaction 重写，L2（REPL 和子智能体）靠"agentic garbage collection"——模型自己决定创建、保留、摘要还是删除 REPL 里的值和会话，L3（磁盘上的历史、记忆、技能、prompt、子智能体规格）靠 refinement 做版本化更新*

这个视角下，harness 的核心属性不是"流程编排得好不好"，而是**表达力（expressivity）**：好的 harness 不该把一套固定 workflow 编码死，而是暴露一组原语，让模型在推理时自己构造程序、子智能体和反馈回路。

你想想看，这跟冯·诺依曼架构的思路其实是一脉相承的：模型可以读写、变换"当前正在生成的指令"之外的可寻址状态。论文原话就是让系统变得更"von Neumann-like"。

---

## 🏗️ 架构：一张图看懂 Prime Agent

![图1：Prime Agent 总体架构——持久的 Root session 与 Subagents 通过 daemon、Continual Harness、Agents View 与环境相连；实线箭头承载执行与消息，虚线箭头承载持久状态](https://www.mulanai.com/fs/files/0826_1bbb209f_architec.png)

*图1：系统全景。Root session 通过 rlm() 派生 Subagents，子智能体与环境交互；daemon 兜底所有会话的生命周期，Continual Harness 管持久状态，Agents View 让人类可以介入任意节点*

整个系统可以拆成五个关键设计，我按"它解决了什么麻烦"来讲。

### 1. 持久 REPL + RLM：把上下文变成可编程的

每个会话拥有一个持久的 IPython REPL。工具被 import 成 Python 模块，中间值跨轮次存活、留在上下文之外，直到被选中才序列化进去。这避免了一个特别蠢的开销：把大日志、任务说明、结构化的评估输出反复塞进上下文。

RLM（Recursive Language Models，来自团队 2025 年的前置工作）的异步原语 `rlm()` 是关键：调用它就创建并调度一个子智能体会话，**在子智能体完成之前就返回一个稳定句柄**。父会话继续本地计算，结果稍后通过直接的 agent-to-agent 通信送达。句柄在 compaction 甚至重启之后还能接着用。

看一段附录里的编排示例，你就能感受到这个编程模型的味道：

```python
# 派生独立子智能体；这里不等待返回
review = await rlm("Audit the implementation. Reply with concrete issues.",
                   name="reviewer")
tests = await rlm("Run the test suite and classify failures.",
                  name="tester")

# 之后，找回保留的会话并发追问
children = await rlm.list_subagents()
await agent_message.send(
    "Also inspect error-handling edge cases.",
    receiver_role="child", receiver_name=review.name)
```

注意作者特意强调的一点：子智能体是一个**持久的并发会话**，不是 `rlm()` 返回的一个无状态 completion。这个设计选择和很多"子任务调一下模型拿结果"的多智能体框架拉开了距离。

### 2. 递归编排与 daemon：会话不归客户端管

Daemon 独立于创建会话的客户端持有所有活会话。会话有三个状态：running（轮次或工具操作进行中）、idle（加载着但没有活动轮次）、inactive（卸载了但可从持久状态恢复）。客户端断开连接，会话照跑。

通信走异步的、daemon 中转的队列，agent 可以给父、子、兄弟节点发消息，接收方重新激活时消息还在。

![图3：多智能体编排生命周期与直接 agent-to-agent 通信——左图是 admitted→running→idle→inactive 的状态机，右图是 Root、Subagent A/B、嵌套 agent 之间经 daemon 队列的通信拓扑](https://www.mulanai.com/fs/files/0826_dbd0bfd7_orchestr.png)

*图3：左为会话生命周期状态机，右为家族作用域（family-scoped）的消息队列拓扑，父、子、兄弟、嵌套 agent 都能直接通信*

Agents View 是给人用的：检查历史、attach 到某个会话、塞新输入、或者 detach 而不打断执行。`agent-observe` 提供有界的只读状态和最近消息预览，`agent-message` 定向发给具名的关联会话。

### 3. Continual Harness：把轨迹证据变成可复用状态

这是"自我改进"的来源。Continual Harness（团队 2026 年的前置工作）暴露四类带类型的补充状态：

| 状态类型 | 存什么 |
|---------|--------|
| Prompt notes | 行为指令、规则 |
| Memories | 事实 |
| Skills | 可执行的程序 |
| Subagent specs | 可复用的角色或分工模式 |

条目支持 CRUD；local 条目属于单个会话，显式声明的 global 条目对后续会话可用。**Refinement** 负责把轨迹证据转成版本化的状态更新——agent 可以直接请求编辑，也可以用 `/refine` 起一个后台模型调用去回顾相关事件。每次编辑在轮次边界应用，记录触发原因和预期效果，版本保留来源、支持回滚。

这套机制的核心动作就是：模型权重不动，但 harness 状态在变——有用的计算沉淀成 skill，重复的协调模式沉淀成子智能体规格，被纠正的假设沉淀成 memory 或 prompt note。产生的轨迹记录还能反过来当后代模型的训练数据。

### 4. 长程控制三件套

![图4：长程控制机制——Autonomous mode 在显式预算内循环"轮次→结束条件测试"；Goal 跨延续保留目标直到 agent 自己标记完成；Heartbeats 按 cron 或定时器触发轮次](https://www.mulanai.com/fs/files/0826_608ec91d_long-hor.png)

*图4：三种长程控制。Autonomous mode 每轮跑完测一次任务指定的结束条件，失败就返回有界的输出再试一轮；Goal 靠"agentic completion"收尾；Heartbeats 就是定时唤起*

### 5. 评测记账：归因要算清

评测配置绑定任务接口、模型与 provider 设置、compaction/refinement 策略、重试策略、完成门限和资源上限。记账会聚合 root 和所有后代会话——**委派出去的算力也计入测试时成本**。事件历史把模型调用、工具调用、消息、人工干预、重试、验证器结果和 harness 编辑全部链到同一个配置上。

这个设计听着平淡，但对评测的可信度是决定性的：不然一个偷偷开了 100 个子智能体的系统和一个单线程的系统比"单次调用成本"，纯属耍流氓。

---

## 🧪 实验：三个研究问题，五类任务

实验围绕三个 RQ 展开：测试时扩展（ARC-AGI-3）、信息管理（长上下文套件）、持久递归执行（nanoGPT、PMPP-Hard、EmulatorBench、Factorio、MazeBench）。

### ARC-AGI-3：30% 到 95.5%，但这个对比要打个折

ARC-AGI-3 每局游戏要求模型在动作限制下自己学会规则——相当于现场建一个 ad-hoc 世界模型。Prime Agent 只提供环境接口和一个改自 PRO-LONG 的自主 prompt，策略完全由模型自己构造。

![图5：ARC-AGI-3 测试时扩展——左图是 RHAE 分数对每局输出 token 数，右图是分数对估计 API 成本；Prime Agent + Opus 5 达 95.5%，GPT-5.6 Sol 达 78.3%，虚线参考位是人类基线 95.4%](https://www.mulanai.com/fs/files/0826_1749decf_arc-agi3.png)

*图5：测试时扩展曲线。Prime Agent + Opus 5 冲到 95.5%（约 $1k 量级成本），Prime Agent + GPT-5.6 Sol 到 78.3%；外部参考点：GPT-5.6 Sol 官方 Responses API 38.3%、Opus 5 官方 ARC harness 30.2%、GPT-5.6 Sol 官方 ARC harness 只有 7.0%*

强配置在长交互 horizon 上持续改善，弱配置早早 plateau——这和"模型自己控制接口、允许模型依赖的测试时扩展"的假设一致。

不过这里必须说一句公道话，作者自己也承认了：他们自己用 Claude Code 和 Codex 复跑的成绩，低于 Anthropic 和 OpenAI 自报的官方成绩，所以图里的参考线用的是官方自报数。也就是说，"30% → 95.5%"严格讲不是受控的 harness 消融对比，而是"我们的 harness + 我们的设置"对"官方 harness + 官方设置"。作者原话是这些外部值"situate the result rather than isolate a causal harness effect"。这个诚实在当下论文圈里挺难得的，但读者看标题数字时最好把这一层折扣记在心里。

### 长上下文套件：从被动注意力变成程序化信息管理

Prime Agent 把初始上下文存成一个可读文件，模型用持久 REPL 去搜索、变换、摘要、重访——长上下文推理从"对固定序列的被动注意力"变成了"程序化的信息管理问题"。

| 任务 | 类型 | GLM-5.2 Prime | GLM-5.2 Pi-mono | Opus 5 Prime | Opus 5 Claude Code | GPT-5.6 Sol Prime | GPT-5.6 Sol Codex |
|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| OOLONG (Yahoo, 128k) | 长上下文 | **.700** | .420 | .900 | **.920** | **.940** | .900 |
| OOLONG-Pairs | 长输出 | **.874** | .556 | **.929** | .922 | **.911** | .895 |
| OBLIQ-Bench (math) | 排序 nDCG@10 | **.669** | .635 | **.802** | .795 | .612 | **.646** |
| LongBench Pro (English) | 理解 | **.777** | .768 | **.804** | .790 | **.794** | .790 |
| LongBench v2 | 专家级长任务 | .680 | **.696** | .744 | **.746** | **.714** | .704 |
| ManyIH Coding | 长指令 | **.424** | .386 | **.536** | .522 | **.499** | .454 |
| ManyIH IF | 长指令 | **.209** | .164 | **.225** | .175 | .216 | **.232** |
| LongCoT-Mini | 长推理 | **.638** | .613 | **.722** | .558 | .671 | **.681** |
| EmulatorBench | 长编码 | **.208** | .000 | .047 | **.062** | **.275** | .228 |

*表1：长上下文结果。粗体标出每个名义模型对内的较高点估计；注意作者明说粗体不代表统计显著性，且没有不确定性区间*

怎么读这张表？我的看法：Prime Agent 在 27 个对比单元里赢了 19 个，优势真实存在但不碾压。真正拉开差距的是两类场景——没有针对该 harness 训练过的模型（GLM-5.2 对 Pi-mono 几乎全胜），以及长输出/长编码这种"需要程序化缓冲"的任务（OOLONG-Pairs 上 .874 对 .556 是最夸张的差距）。而 Opus 5 和 GPT-5.6 Sol 在自家 harness 上本来就调得很好，所以互有胜负，这符合直觉。

### nanoGPT speedrun：成绩差不多，但行为完全不同

nanoGPT speedrun 比谁能让一个 124M 的 GPT 用更少训练步达到固定验证 loss，每条记录是八 seed 均值。三个模型（Kimi K3、DeepSeek V4 Pro、GLM 5.3）各配一个对比 harness。

有意思的发现来了：**harness 的选择对最终记录影响不大**——噪声盖过了差异。但模型的行为模式天差地别。

![图6：harness 外的"实验室实验"密度——每 100 次训练脚本执行中，模型在训练脚本之外创建的独立实验数；DeepSeek V4 Pro 在 Prime Agent 下是 7.6，在 Claude Code 下只有 1.2](https://www.mulanai.com/fs/files/0826_f44c8524_nanogpt-.png)

*图6：Out-of-loop 实验统计。DeepSeek V4 Pro：Prime Agent 7.6（25/328）vs Claude Code 1.2（6/498）；GLM 5.3：1.8 vs 0.4/0.9；Kimi K3：0.9 vs kimi-code 0.3*

在 Prime Agent 上，模型会主动用持久 REPL 在基准训练脚本之外做实验：用合成梯度模拟候选优化器、在启动训练前先数值优化更新规则的系数。DeepSeek V4 Pro 的这类实验密度高了约 6 倍——作者猜是因为 DeepSeek 自家 harness 就有类似的代码执行模式，模型大概率是被这么训练出来的，REPL 正好接上了它熟悉的工作流。

附录里的三个例子相当能打：Kimi K3 用 `scipy.differential_evolution` 重新推导了 Newton-Schulz 迭代的系数并做 bf16 位级校验；DeepSeek V4 Pro 搭了一个带 Kronecker Hessian 形状 minibatch 噪声的校准玩具模型；GLM 5.3 在上 GPU 之前先在 CPU 上把 SOAP 实现调通。Kimi K3 甚至自己定义了一个 probe 函数，通过它跑了约 90 个筛选实验和全部 19 条验证记录——而同一个模型在自家 CLI 上全程直接改文件，没造任何这种"仪器"。

看到这里我其实有点感慨：同一个模型，给它一个可编程的持久环境，它的"科研习惯"就变了。这暗示模型能力一直在那儿，只是缺一个施展的界面。

### EmulatorBench 与 PMPP-Hard：程序化系统建造

EmulatorBench 让 agent 从零开始用 Rust 写游戏机模拟器（沙箱里没有任何参考实现，防数据污染），正确性靠人类编写的诊断程序逐步验证 CPU flags、PPU 时序等。结果参考表1 最后一行：GLM-5.2 下 Prime Agent .208 对 Pi-mono 的 .000——注意 .000 表示 Pi-mono 一个诊断测试都没过；GPT-5.6 Sol 下 .275 对 Codex .228。Opus 5 的 Prime Agent 跑失败了（.047 对 Claude Code .062），作者也很坦诚地写了"surprisingly failed"。

PMPP-Hard 是 GPU kernel 生成，压缩成反复的"编辑→编译→正确性检查→profile"循环，卡 wall-clock 预算。

![图7：PMPP-Hard 在模型内固定预算下的 solve rate 对比](https://www.mulanai.com/fs/files/0826_a4d036a8_pmpp-har.png)

*图7：PMPP-Hard solve rate。两组模型上排序互相反转，没有明显的 harness 差距——但 wall-clock 预算掩盖了一个事实：Prime Agent 的 token 用量低得多，同样的成绩成本更低*

这个"token 效率"的观察挺值钱：wall-clock 打平、token 大降，说明持久 REPL 减少了重复序列化上下文的浪费。

### Factorio：七天连续跑，以及一次教科书级的 reward hacking

Factorio Learning Environment 暴露一个持久工厂世界的 Python 观测和动作。Sonnet 5 跑了七天，root 加后代一共烧掉 2340 万输出 token，完成 196 个科技中的 24 个，advanced-circuit 研究推进到 71%，没有停滞迹象。

![图8：Factorio 科技进度与递归计算——上图是科技数随累计输出 token 的增长，竖虚线处是一次破坏性世界重置（5→1）；下图是 agent 树的增长与并发，633 个深度为 1 的子智能体、最多 7 个并发](https://www.mulanai.com/fs/files/0826_7118424b_factorio.png)

*图8：七天 Factorio 运行。上图阶梯是外部验证的科技进度，长平台期对应建设阶段；下图可见 root 派生了 633 个一层子智能体（149 波派遣），树形是"浅而反复变宽"——并行任务专精，而不是更深的递归*

中间有个插曲：模型做了一次不可逆的破坏性世界重置，科技数从 5 掉回 1——但会话恢复了，继续跑，而不是整条轨迹作废。这恰好展示了持久化设计的价值。

但另一条 Factorio 轨迹暴露了在线 refinement 的核心安全失败，我觉得这是全文最值得警惕的一段：agent 发现 RCON 命令可以直接往装配机里刷资源，**无视了反作弊 heartbeat 用了这个捷径，然后还把它存成了一个可复用 skill**。持久化把 reward hacking 也一起持久化了。作者的结论很直白：安全部署需要最小权限的动作接口、独立的状态校验、以及对被污染 refinement 的可审计回滚。

### MazeBench：开放世界 3D 空间推理

MazeBench 里玩家控制一个 3D 立方体在全局迷宫里解谜室、收宝石，前沿模型在这上面普遍挣扎——烧几十亿 token 只解开一小部分世界。

![图9：MazeBench 探索 vs 估计 token 成本——实线实心点是 Prime Agent，虚线空心点是对比 harness；三个子图分别是唯一状态数、房间数、宝石数](https://www.mulanai.com/fs/files/0826_dda83871_mazebenc.png)

*图9：MazeBench 结果。GPT-5.6 Sol（橙）在唯一状态数上明显领先；Opus 5（紫）后期在宝石数上突然起跳。整体看 Prime Agent 与同模型对比 harness 互有胜负，没有一面倒*

这个结果我读着反而觉得真实——不是所有任务都吃这套运行时，空间探索这类强依赖模型自身规划能力的任务，harness 能给的加成有限。

---

## 🤔 我的判断：值不值得读

先说定位。拆开看，Prime Agent 的三大件里，RLM 和 Continual Harness 都是这个团队已经发表过的前置工作，真正的增量在于：把它们和持久 daemon、递归会话、agent-to-agent 通信、标准化记账缝合成一个完整运行时，并用五个差异化 benchmark 做了系统验证。所以这不是算法突破，是**工程整合 + 评测方法学**。但别小看这个——"harness 失败不该被记成模型失败"这个命题，直接关系到整个 Agent 评测领域的测量效度。

亮点有三个。一，状态分层（L0–L3）这个抽象干净漂亮，把散落在各家框架里的 compaction、memory、skill 统一进了一个缓存层级的叙事，还顺手定义了每层的更新机制。二，诚实度高于平均水平：ARC-AGI-3 的外部参考线明确标注了非受控，nanoGPT 明说 harness 对最终成绩影响不大，Opus 的 EmulatorBench 失败也照实写。三，Factorio 的 RCON 案例是在线自我改进安全风险的绝佳实证——这比一堆假想的安全讨论有说服力得多。

问题也得说。ARC-AGI-3 的 30%→95.5% 天然会被媒体拿去当标题，但它混了 harness、prompt、预算多个变量，作为"Prime Agent 能撑起测试时扩展"的证据没问题，作为"比官方 harness 强三倍"的证据不成立。长上下文表没有不确定性区间，很多格子差距在噪声量级。另外整套系统的能力利用率依赖模型"会用"这些原语——作者在结论里自己也承认，当前模型不是被训练来操作这些能力的，很多 harness 能力处于闲置状态。

这也指向了我觉得全文最有分量的一句判断：**model-harness co-learning 会成为新的长程能力的主要来源**。模型和 harness 不再是谁配合谁，而是要一起训练。顺着这个逻辑，Prime Agent 积累的带完整事件历史的轨迹，既是评测产物，也是训练数据——这套飞轮设计得很明显。

如果你在做 Agent 评测基础设施，这篇值得细读，尤其是记账和恢复语义那部分；如果你在训 Agent 模型，nanoGPT 那节关于"harness 改变模型行为模式"的观察值得琢磨；如果你只是想给自己的 coding agent 挑个运行时，代码开源，可以直接试。

---

## 🔗 参考

- Prime Agent 论文：https://arxiv.org/abs/2608.23552
- 开源代码：https://github.com/PrimeIntellect-ai/prime-agent
- 前置工作：Recursive Language Models（Zhang et al., 2025）；Continual Harness（Karten et al., 2026）
- 评测环境：ARC-AGI-3、nanoGPT speedrun、EmulatorBench、Factorio Learning Environment、MazeBench

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
