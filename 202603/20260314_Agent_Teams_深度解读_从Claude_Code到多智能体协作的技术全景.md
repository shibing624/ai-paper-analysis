# Agent Teams 深度解读：从 Claude Code 到多智能体协作的技术全景

> 当一个 Agent 不够用，那就来一个团队。本文拆解 Agent Teams 的核心工作机制，横向对比 Claude Code、MetaGPT、AutoGen 等主流多智能体框架，纵向追溯学术脉络，试图回答一个根本问题：**让 AI "组队干活"到底难在哪？**

---

## 🎯 一句话总结

Agent Teams 是一种**主从式多智能体协作架构**，通过任务分解（DAG）、消息传递（Mailbox）和生命周期管理（Shutdown Protocol）三大核心机制，让多个 LLM Agent 像真实团队一样分工协作、并行执行、协调收敛。Claude Code 的 Agent Teams 是当前工程实现最完整的产品级方案，而这一方向背后有一整条从 CAMEL 到 AdaptOrch 的学术演化链。

---

## 📖 起源：一篇"Agent 自我观察"的实验报告

之前我做了一件很有意思的事情——**让 Agent Teams 自己观察自己**。我创建了一个5人"观察团队"，由不同角色的 Agent 分别记录协作机制的工作细节。团队角色设计如下：

| 角色名称 | 类型 | 职责 |
|---------|------|------|
| team-lead | 协调型 | 统筹全局、分配任务、收集结果 |
| task-manager | 调度型 | 管理任务队列、处理依赖关系 |
| executor-alpha | 执行型 | 执行具体子任务 |
| executor-beta | 执行型 | 与 alpha 并行执行，测试竞争机制 |
| protocol-tester | 边界测试型 | 探测消息协议边界、异常处理 |

这种"让被观察者自己写报告"的实验设计颇为巧妙。结果也确实揭示了几个核心机制，下面我们逐一拆解。

---

## 🏗️ 核心机制一：文件系统即状态后端

Agent Teams 没有用数据库，甚至没有用内存中的共享状态，而是**直接拿文件系统当状态后端**。这个选择乍一看很"土"，但仔细想想非常合理——每个 Agent 本质上是一个独立的 CLI 进程，用文件系统做进程间通信是最简单、最可靠的方案。

```
~/.codebuddy/teams/{team-name}/
├── config.json              # 团队配置：成员列表、角色定义
├── members/
│   ├── team-lead/
│   │   └── inbox/           # 收件箱：其他 Agent 发来的消息
│   ├── executor-alpha/
│   │   └── inbox/
│   └── ...
└── tasks/
    ├── task-001.json         # 任务元数据：状态、依赖、所有者
    ├── task-002.json
    └── ...
```

每个任务用一个 JSON 文件记录状态，包含 `status`（pending/in_progress/completed/deleted）、`blockedBy`（依赖的其他任务 ID）和 `owner`（认领者）三个关键字段。这个设计有几个优点：

**原子性靠文件系统保证。** 两个 Agent 同时想认领同一个任务？`TaskUpdate` 操作设置 owner 字段时，文件系统的写入锁天然保证了原子性，不需要额外的分布式锁机制。

**可观测性极强。** 整个团队的状态就是一堆 JSON 文件，用 `ls` 和 `cat` 就能看到全貌。调试的时候不用接 debugger，直接 `watch -n1 cat tasks/task-001.json` 就行。

**崩溃恢复简单。** 某个 Agent 挂了？重启后读一遍文件就能恢复状态，不需要事务日志或 WAL。

当然也有明显的缺点——性能上限低。报告里发现广播消息的成本随团队规模线性增长（N 个队友 = N 次文件写入），这在 10 人以内的小团队可以接受，但如果要扩展到几十个 Agent 并行就会成为瓶颈。

---

## 🏗️ 核心机制二：任务系统与 DAG 依赖

任务不是简单的列表，而是一个**有向无环图（DAG）**。每个任务可以通过 `blockedBy` 字段声明自己依赖哪些前置任务，只有前置任务全部 completed 之后，当前任务才会从"阻塞"变为"可认领"。

这个设计直接借鉴了构建系统（如 Make、Bazel）的依赖拓扑排序思想。举个例子，一个典型的前后端开发任务可能长这样：

```
[定义 API Schema] ──→ [实现后端接口] ──→ [集成测试]
         │                                    ↑
         └──→ [实现前端组件] ─────────────────┘
```

后端和前端可以并行开发（都只依赖 Schema），但集成测试必须等两者都完成。Agent Teams 的任务系统会自动处理这种依赖关系，编排器不需要手动调度"先做 A 再做 B"。

**四态流转模型**也值得关注：

```
pending ──→ in_progress ──→ completed
                │
                └──→ deleted (取消/失败)
```

没有 `failed` 状态——这是个有意思的设计决策。失败的任务直接标记为 deleted，然后由编排器决定是重新创建一个同样的任务让别的 Agent 认领，还是调整整个执行计划。这种"失败即重来"的策略比复杂的错误恢复机制要简单得多。

---

## 🏗️ 核心机制三：消息传递与通信协议

Agent 之间怎么交流？Agent Teams 设计了一套基于收件箱的消息系统，支持以下消息类型：

| 消息类型 | 用途 | 是否需要回复 |
|---------|------|------------|
| message | 点对点定向消息 | 否 |
| broadcast | 广播给所有队友 | 否 |
| shutdown_request | 请求某个 Agent 关闭 | 是（shutdown_response） |
| shutdown_response | 确认关闭请求 | 否 |
| plan_approval_response | 审批执行计划 | 否 |

报告中 protocol-tester 做了几个很有价值的边界测试：

**超长消息测试：** 发送了 1085 字符的消息，系统正常接收，没有截断。这说明消息长度限制比较宽松，Agent 之间可以传递详细的上下文信息。

**无效收件人测试：** 给一个不存在的 Agent 发消息，系统直接拒绝，不会静默丢弃。这是一个关键的安全特性——如果消息被静默丢弃，调试起来会非常痛苦。

**空消息测试：** 空内容消息也被拒绝。看起来是小事，但防止了 Agent 因为幻觉（hallucination）产生空消息进而触发其他 Agent 的无意义处理。

消息投递采用的是**客户端驱动的推送模式**，不是 Agent 自己轮询收件箱。当 Agent 执行完一个 tool call、进入下一个 agentic turn 时，系统会自动把收件箱里的新消息注入到 Agent 的上下文中。这比轮询高效得多，也避免了轮询间隔的配置问题。

---

## 🏗️ 核心机制四：生命周期管理

一个经常被忽视但极其重要的问题：**Agent 什么时候停下来？**

LLM Agent 没有"完成"的天然概念。如果不加约束，一个 Agent 可能会无限循环地"思考"——每次都觉得还有事情没做完。Agent Teams 用两个机制解决这个问题：

**`max_turns` 硬上限。** 每个 Agent 在创建时可以设定最大执行轮次。到达上限后 Agent 强制停止，无论任务是否完成。这既是成本控制（每个 turn 都烧钱），也是安全边界（防止失控）。

**优雅关闭协议（Graceful Shutdown Protocol）。** Team Lead 关闭团队时，不是直接杀进程，而是：
1. 依次向每个成员发送 `shutdown_request`
2. 每个成员收到后完成当前手头工作，然后回复 `shutdown_response`（approve=true）
3. Team Lead 收集全部 response 后，才执行最终的 `team_delete` 操作

这跟 Kubernetes 的 Pod 终止流程很像——先发 SIGTERM，给 graceful period 让进程收尾，最后再 SIGKILL。只不过这里是用自然语言消息代替了 Unix 信号。

---

## 🔍 Claude Code Agent Teams：产品级实现的技术细节

报告观察到的机制，跟 Claude Code 官方 Agent Teams 的实现高度一致。Claude Code 在 2026 年 2 月正式推出 Agent Teams 功能，把社区之前通过 OpenClaw 等工具零散实现的多 Agent 协作能力做成了原生支持。我们来看看它的具体工程实现。

### 工具体系

根据 Claude Code 系统提示词（version 2.1.76）的公开拆解，Agent Teams 依赖以下核心工具：

| 工具名称 | Token 数 | 功能 |
|---------|---------|------|
| **TeammateTool** | 1645 | 核心工具：创建团队、管理成员、协调队友 |
| **SendMessageTool** (Teams版) | 1205 | 智能体之间的消息传递 |
| **TaskCreate** | 528 | 创建任务或启动子智能体 |
| **TeamDelete** | 154 | 删除团队、清理资源 |

此外还有几个关键的 System Prompt：

**Team Coordination（250 tokens）：** 告诉 Agent 在团队环境中如何与其他成员配合——什么时候该发消息、什么时候该等待、什么时候该认领任务。

**Teammate Communication（130 tokens）：** 定义了 swarm 模式下智能体之间的通信行为准则。

**Fork Usage Guidelines（339 tokens）：** 规定了什么时候该 fork 子智能体。有一条很关键的规则：**禁止在 fork 过程中读取输出或伪造结果**。这是为了防止 Agent 偷懒——如果允许主 Agent 在子 Agent 还没执行完就"猜"一个结果，整个并行执行的可靠性就没了。

### 编排模式

Claude Code 的编排器（主 Agent）根据任务性质动态决定生成多少个子 Agent、分别承担什么角色。触发方式有两种：

**隐式触发：** 用户描述的任务暗示了需要并行处理。比如"重构整个项目的认证模块，同时更新所有相关的单元测试"，编排器会自动判断可以拆成"重构"和"测试"两个并行子任务。

**显式触发：** 用户明确要求。比如"用 3 个 Agent 分别处理前端、后端和数据库"。

每个子 Agent 有独立的上下文窗口，只加载与自己任务相关的文件和信息。这解决了单 Agent 架构的一个根本问题：**上下文窗口是有限的**。一个 Agent 试图同时理解整个项目的前端、后端、数据库代码，很容易因为上下文过长而丢失关键细节。分成多个 Agent 后，每个只需关注自己那一块。

### Worker Fork 模式

Claude Code 有一个特殊的子 Agent 类型叫 **Worker Fork**，它的系统提示词（370 tokens）明确要求：

> 这是一个分叉的子智能体，它直接执行指令而不生成更多的子智能体，然后报告结构化结果。

换句话说，Worker Fork 是**叶子节点**——只干活、不分活。这防止了子 Agent 无限递归地创建更多子 Agent，避免了"Agent 繁殖爆炸"的问题。这跟操作系统中的 fork bomb 防护逻辑是一个道理。

---

## 📚 学术脉络：Agent Teams 背后的论文演化

Agent Teams 不是凭空出现的，背后有一条清晰的学术演化链。从2023年的角色扮演通信框架，到2026年的自适应编排，我们来梳理关键节点。

### CAMEL（2023）：角色扮演通信框架的开山之作

> Li et al. "CAMEL: Communicative Agents for 'Mind' Exploration of Large Language Model Society." arXiv:2303.17760, 2023.

CAMEL 提出了一个核心洞察：**两个 LLM Agent 可以通过角色扮演（role-playing）实现自主协作，不需要人类持续介入**。具体做法是用一个叫 "inception prompting" 的技术，给两个 Agent 分别设定角色（比如一个是"用户"、一个是"AI助手"），然后让它们自动对话推进任务。

这个工作的意义在于证明了 LLM 多智能体协作的可行性，但它的限制也很明显——只支持两个 Agent 之间的对话，没有任务分解、没有依赖管理、没有生命周期控制。可以理解为 Agent Teams 的"最小可行验证"。

### MetaGPT（2023，ICLR 2024 Oral）：把软件公司搬进 LLM

> Hong et al. "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework." ICLR 2024 Oral. arXiv:2308.00352.

MetaGPT 把"Agent 协作"这件事向前推了一大步，它的核心创新是**把人类组织的标准操作流程（SOP）嵌入到多智能体系统中**。具体来说，MetaGPT 模拟了一个软件开发公司的组织结构：

| Agent 角色 | 对应人类角色 | 输出物 |
|-----------|-----------|--------|
| Product Manager | 产品经理 | PRD（需求文档） |
| Architect | 架构师 | 系统设计文档 |
| Project Manager | 项目经理 | 任务分解和排期 |
| Engineer | 工程师 | 代码实现 |
| QA Engineer | 测试工程师 | 测试用例和报告 |

每个角色只接收上游角色的标准化输出（比如 Engineer 只看 Architect 的设计文档，不需要看 PRD），通过**结构化的中间制品（artifacts）**实现角色间的信息传递。这比 CAMEL 的自由对话高效得多——就像真实公司里工程师不需要参加产品讨论会，只需要看 PRD 和技术方案就行。

MetaGPT 对 Agent Teams 的影响主要体现在"角色专业化"和"工作流管理"两个维度。Claude Code 的 Team Coordination 系统提示词中，关于角色分配和任务边界的规定，很大程度上继承了 MetaGPT 的设计理念。

### AutoGen（2023，Microsoft）：可编程的多智能体对话

> Wu et al. "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." arXiv:2308.08155, 2023.

如果说 MetaGPT 是"固定流水线"，AutoGen 则提供了"可编程的对话拓扑"。它的核心概念是 **ConversableAgent**——每个 Agent 都可以与其他任意 Agent 对话，开发者通过代码定义谁跟谁说话、在什么条件下说什么。

AutoGen 对 Agent Teams 最大的启发是**人机协作模式**。在 AutoGen 中，人类可以作为一个特殊的 Agent 参与到多智能体对话中，在关键决策点介入。Claude Code 的 `plan_approval_response` 消息类型就是这个思路的产物——Team Lead 可以在执行计划生成后请求人类审批，而不是自动开干。

### Multi-Agent Collaboration Mechanisms Survey（2025）：系统性分类框架

> Tran et al. "Multi-Agent Collaboration Mechanisms: A Survey of LLMs." arXiv:2501.06322, 2025.

这篇综述提出了一个五维分类框架来表征多智能体协作机制：

| 维度 | 说明 | Claude Code 对应 |
|-----|------|-----------------|
| **参与者** | 涉及哪些智能体 | 主 Agent + N 个 Worker Fork |
| **协作类型** | 合作/竞争/混合 | 纯合作（cooperative） |
| **结构** | 集中式/分布式/点对点 | 集中式（主从架构） |
| **策略** | 基于角色/基于模型 | 基于角色（role-based） |
| **协调协议** | 通信规则 | Mailbox + DAG 依赖 |

按照这个分类框架，Claude Code Agent Teams 属于**集中式、合作型、基于角色的多智能体系统**，协调协议结合了消息传递和任务依赖两种机制。

### LLM-based Multi-Agents Survey（2024）：进展与挑战

> Guo et al. "Large Language Model based Multi-Agents: A Survey of Progress and Challenges." arXiv:2402.01680, 2024.

这篇来自圣母大学（Notre Dame）等机构的综述，系统回答了四个问题：

1. **环境**：多智能体系统在什么样的环境中运行？（沙盒、代码库、Web、模拟器）
2. **角色设定**：如何给每个 Agent 分配身份和能力？（prompt-based profiling）
3. **通信**：Agent 之间怎么交换信息？（自然语言、结构化消息、共享内存）
4. **能力增长**：Agent 如何在协作中变得更强？（经验积累、反馈学习）

对于 Claude Code Agent Teams 来说，第 3 点特别值得关注。论文指出，纯自然语言通信会引入大量冗余信息，降低协作效率。Claude Code 的解决方案是**结构化消息**——`send_message` 工具要求提供 `type`、`recipient`、`content`、`summary` 四个字段，其中 `summary` 是一个 5-10 词的摘要。这个设计让接收方可以先看摘要决定是否需要读全文，减少不必要的上下文注入。

### AdaptOrch（2026）：编排拓扑比模型选择更重要

> Yu. "AdaptOrch: Task-Adaptive Multi-Agent Orchestration in the Era of LLM Performance Convergence." arXiv:2602.16873, 2026.

这篇论文提出了一个很有挑衅性的观点：**当各家 LLM 的性能趋于收敛时，优化编排拓扑比选择更强的模型更有效**。

AdaptOrch 定义了四种基础编排拓扑——并行、顺序、分层、混合——并提出了一个拓扑路由算法，能在 $O(|V| + |E|)$ 时间内将任务依赖 DAG 映射到最优编排模式。实验表明，即使使用相同的底层模型，拓扑感知的编排比固定拓扑实现了 12-23% 的性能提升。

这个结论对 Claude Code Agent Teams 有直接的工程启示：与其纠结用 Sonnet 还是 Opus 执行某个子任务，不如花精力优化任务拆解和依赖关系的设计。目前 Claude Code 的编排器主要依赖 LLM 自身的判断力来决定拆分策略，还没有引入形式化的拓扑优化算法，这可能是未来改进的方向。

---

## 🔧 工程落地：主流框架横向对比

| 维度 | Claude Code Agent Teams | MetaGPT | AutoGen | CrewAI | LangGraph |
|-----|------------------------|---------|---------|--------|-----------|
| **架构模式** | 主从式（Leader + Workers） | 流水线式（SOP chain） | 可编程对话图 | 基于角色的编排 | 有向图状态机 |
| **通信机制** | Mailbox + 文件系统 | 结构化制品传递 | Agent 对话链 | 任务委托 | 状态传递 |
| **任务管理** | DAG + 四态流转 | SOP 驱动 | 开发者定义 | 自动/手动 | 图节点 |
| **并行支持** | 原生并行 spawn | 有限 | 需配置 | 支持 | 支持 |
| **生命周期** | shutdown 协议 | 流程终止 | 开发者控制 | 自动 | 图终态 |
| **人机协作** | plan_approval | 有限 | human-in-the-loop | 支持 | 支持 |
| **上手门槛** | 低（CLI 原生） | 中 | 高（需编程） | 低 | 高 |
| **适用场景** | 代码开发/重构 | 软件工程全流程 | 通用多 Agent | 业务自动化 | 复杂工作流 |

几个值得注意的差异：

**Claude Code 的优势在于"开箱即用"。** 用户不需要写一行编排代码，只需要自然语言描述任务，编排器自动决定拆分策略。这是 MetaGPT 和 AutoGen 做不到的——它们需要开发者预先定义角色和交互拓扑。

**但灵活性也因此受限。** 如果你对任务拆分有特定要求（比如"一定要先做 code review 再合并"），Claude Code 的编排器可能不会完全按你想的来。MetaGPT 和 LangGraph 在这方面更可控。

**CrewAI 和 Claude Code 在理念上最接近——** 都是"给 Agent 分角色，然后让它们自己干"。但 CrewAI 更偏业务自动化场景（营销、客服、研究），Claude Code 专注于代码开发。

---

## 💡 我的思考与判断

### Agent Teams 的真正瓶颈不是技术，是成本

报告里提到广播消息的成本随团队规模线性增长，但这其实是小问题。**真正的成本瓶颈是 token 消费**。每个子 Agent 都是一个独立的 LLM 会话，需要独立的系统提示词（light thousands of tokens）、上下文注入、多轮推理。一个 5 Agent 的团队，token 消费大约是单 Agent 的 3-5 倍（不是 5 倍，因为每个子 Agent 的上下文比单 Agent 处理全部任务要短）。对于使用量大的团队来说，这笔账得算清楚。

### "编排质量"将成为核心竞争力

AdaptOrch 论文的核心结论——编排拓扑比模型能力更重要——我认为说到了点子上。同样一个"重构认证模块"的任务，好的编排器会把它拆成"分析现有代码 → 设计新接口 → 实现核心逻辑 → 迁移调用方 → 编写测试"五步，有明确的依赖关系；差的编排器可能就拆成"改后端"和"改前端"两坨，导致接口不一致需要大量返工。

目前这个编排质量完全依赖 LLM 自身的规划能力，没有形式化的验证或优化。我觉得接下来 6 个月内，我们会看到更多工作把图优化算法引入 Agent 编排——类似编译器优化 AST 那样，优化任务 DAG。

### 消息系统会从"无结构"走向"强类型"

现在 Agent 之间传递的消息基本是自然语言 + 少量结构化字段（type、recipient）。但随着 Agent 数量增加和协作复杂度提升，我预测消息系统会向"强类型"方向演化——每种消息有 schema 定义，接收方可以校验消息格式，甚至可以自动路由。这本质上就是把微服务架构中的 protobuf/gRPC 那套搬过来。

### 人机协作的比例问题

Agent Teams 目前的定位是"自主执行 + 关键节点人类审批"。但"哪些节点需要人类介入"这个问题还没有好的答案。太多审批会降低效率（你一个下午就在按 approve），太少审批又可能让 Agent 团队跑偏。理想的方案可能是**基于置信度的动态审批**——Agent 对自己的决策有高置信度时自动执行，低置信度时请求人类确认。这也是一个值得研究的方向。

---

## 📝 相关论文速查表

| 论文 | 年份 | 核心贡献 | 与 Agent Teams 的关系 |
|-----|------|---------|---------------------|
| CAMEL | 2023 | 角色扮演通信框架 | 证明 LLM 多 Agent 协作可行性 |
| MetaGPT | 2023/2024 | SOP 嵌入多 Agent 系统 | 角色专业化、工作流管理范式 |
| AutoGen | 2023 | 可编程多 Agent 对话 | 人机协作、灵活拓扑 |
| LLM Multi-Agents Survey | 2024 | 系统性综述 | 四维分析框架 |
| Multi-Agent Collaboration Survey | 2025 | 五维分类框架 | 协作机制分类学 |
| AdaptOrch | 2026 | 自适应编排拓扑 | 编排 > 模型选择 |

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我的微信公众号：机器懂语言。*