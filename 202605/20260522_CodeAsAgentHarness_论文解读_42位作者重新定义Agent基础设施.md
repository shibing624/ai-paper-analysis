# 42 位作者联手发声：代码不再只是 Agent 的"输出"，而是它赖以生存的"操作系统"

> 论文：Code as Agent Harness — Toward Executable, Verifiable, and Stateful Agent Systems
> 机构：UIUC + Meta + Stanford
> arXiv：https://arxiv.org/abs/2605.18747
> GitHub：https://github.com/YennNing/Awesome-Code-as-Agent-Harness-Papers

---

## 写在前面

先抛一个问题给你。

如果你过去半年真的在生产环境里调过 Agent，应该对下面这些场景毫不陌生：上下文窗口爆了，对话越长 Agent 越"傻"；工具调用偶尔崩一次，整个任务链断在那里没法续；让它写个跨文件的 refactor，第三步就开始忘记前两步在干嘛；多 Agent 协作时 A 改了文件，B 拿到的还是旧版本，最后谁也不知道 ground truth 是什么。

很多人一上来就把锅甩给"模型不够强"。但说实话，我自己一线踩了一年坑之后越来越觉得——**模型本身只占整个 Agent 故事的一半，另一半全在模型外面那一层"壳"里**。这层壳，社区现在有了个越来越统一的叫法：**Agent Harness**。

刚好这两天 arXiv 上挂出一篇 42 位作者联合署名的综述，来自 UIUC、Meta、Stanford。题目就叫 *Code as Agent Harness*。第一次有人系统性地把"代码作为 Agent 操作基底"这件事做了完整的范式梳理。我读完之后的第一反应是："终于有人把这些散在各处的工程实践，拧成一根绳了。"

---

## 核心摘要

42 位研究者联合署名（其中 6 位 Core Contributor，通讯作者 Hanghang Tong 和 Jingrui He），把过去三年 Agent 工程化的所有零散实践——从 Claude Code 到 SWE-agent，从 Voyager 到 MetaGPT——统一在一个清晰的论点之下：**代码不是 Agent 的产出物，而是 Agent 的运行时基底**。论文搭了一个三层的分类体系：接口层（代码怎么进入推理、行动、环境建模）、机制层（规划、记忆、工具、控制、harness 优化）、扩展层（多 Agent 如何在共享代码工件上协调）。最后给出 6 个 open problem，每一个都直击当前 Agent 工程的痛点。这篇综述最大的价值不在于"提出了什么新方法"，而在于它给了一个**统一的概念语言**——以后再聊 Agent，"harness"会成为绕不开的关键词。

---

## 这篇论文为什么现在出现？

时机的判断我先讲在前面。

2024 年到 2025 年这段时间，整个圈子在做一件事：**把模型外面的东西标准化**。Anthropic 在 Claude Code 文档里第一次明确地说"SDK 就是 driving Claude Code 的 agent harness"。OpenAI 在 2026 年初的 Codex Long-horizon 文档里反复强调 "execution plan" 这种持久化的 harness state。LangChain 把它的"DeepAgents"项目重命名为 harness。Martin Fowler 在博客里干脆把 harness 拆成 Guides（前馈）和 Sensors（反馈）两半。

社区在用脚投票了：**Agent ≠ Model**。

但问题是，2026 年之前所有这些讨论都散在博客、文档、推特线程里。学术界一直没人正经做整理。这篇综述补的就是这个空白——它不是某个团队孤立的成果，而是 42 位作者把整个领域的共识打包，给出一个能站得住的学术框架。

> 我读到 Introduction 那段三元区分的时候，第一反应是："这个分类太关键了。"

论文把一个长程 Agent 系统拆成三个耦合元素：

- **Model-internal capabilities**（模型内部能力）：推理、规划、感知、评估
- **System-provided harness infrastructure**（系统侧 harness 基础设施）：工具、API、沙箱、记忆系统、验证器、权限边界、审计日志
- **Agent-initiated code artifacts**（Agent 主动生成的代码工件）：临时工具、回归测试、可执行工作流、可复用技能、DSL 程序

前两个大家都聊烂了。但第三个——**Agent 在任务执行循环中自己创建、执行、修订、持久化、共享的代码工件**——一直没被严肃对待。这恰恰是这篇论文要重点研究的对象。

回过头看，过去研究"代码 + LLM"，关注点都是"模型能不能写出对的代码"；这篇论文把视角转了 90 度，去问一个新问题：**代码作为 Agent 自己生产、自己消费、自己迭代的中间表示，到底起了什么作用**。

---

## 总览图：一张图看懂整个三层框架

![Code as Agent Harness 三层总览](https://arxiv.org/html/2605.18747v1/x1.png)
*Figure 1：Code as Agent Harness 的三层总览。上方是 AI Agent ↔ Code ↔ External Environment 的核心循环，代码同时充当推理基底（Reasoning Substrate）、行动接口（Action Interface）、环境建模（Environment Modeling）三种角色；中间三栏是三层组织——Harness Interface（§2）、Harness Mechanisms（§3）、Scaling the Harness（§4）；下方是五大应用域 Code Assistants、GUI/OS Agents、Scientific Discovery、Personalization、Embodied Agents*

这张图基本就是论文的"思维导图"。我把每一层的核心提问列出来，方便你建立全局认知：

| 层级 | 核心提问 | 关键章节 |
|------|---------|---------|
| **Harness Interface** | 代码以什么角色进入 Agent 的工作循环？ | §2 |
| **Harness Mechanisms** | 怎么让 Agent 跨多步、跨长程、跨失败稳定地跑下去？ | §3 |
| **Scaling the Harness** | 多个 Agent 怎么在共享的代码工件上协调？ | §4 |
| **Applications & Open Problems** | 在哪些真实场景落地了？还有什么问题没解？ | §5 |

接下来我按这个顺序展开。

---

## 第一层：Harness Interface — 代码以三种角色进入 Agent

![Harness Interface 三大方向](https://arxiv.org/html/2605.18747v1/x2.png)
*Figure 2：Harness Interface 的三个方向——Code for Reasoning（代码帮 Agent 推理）、Code for Acting（代码把意图翻成行动）、Code for Environment Modeling（代码让环境可检查、可追踪）*

### Code for Reasoning：把推理从"嘴上说"变成"程序跑"

这一支起源很早，PAL（Program-aided Language Models）、PoT（Program of Thoughts）、Chain-of-Code 都是代表。核心思想就一句话：**别再让 LLM 用自然语言一步步"想"了，让它写代码，把计算交给解释器**。

为什么这么做？你想想看一个最朴素的例子：让模型算 `(3725 × 47) + 1932`。让它用自然语言"逐步思考"，错误率随便就到 20% 以上。但你让它写一行 Python：

```python
result = 3725 * 47 + 1932
print(result)
```

执行一下，错误率几乎为零。

但这只是开胃菜。论文真正想强调的是这个动作背后更深的范式转变：**当推理被外化为代码之后，整个推理过程就从不可验证变成了可验证**。解释器、符号求解器（如 Z3）、execution trace、process reward model——所有这些工具突然都能介入到推理过程中来检查、纠错、refine。

代表工作：PAL、PoT、Chain-of-Code、SatLM、NExT、CodePRM。

### Code for Acting：把意图翻译成"能在环境里跑的程序"

这一支主要来自具身智能体和工具调用的传统。Code as Policies、Voyager、RoboCodex、UI-TARS 都是这个路线。

我对这块感受最深的是 Voyager。它让 Minecraft 里的 Agent 不是去"想"下一步做什么，而是写一段 JavaScript 函数（比如 `combatZombie()`），存进它的 skill library 里，下次遇到僵尸时直接调用。每个技能就是一段可执行、可复用、可组合的代码。

> 这个设计真的挺漂亮的。比传统的"工具调用"（tool calling）厉害的地方在于——工具是别人写的，Agent 只能用；但这里 Agent 自己**写工具**、自己**用工具**、用完之后还自己**改进工具**。

Code for Acting 的核心区别于传统 tool calling 的关键，是把"行动空间"从一个固定列表，扩展成了"可以动态生成、动态扩展的程序空间"。

### Code for Environment Modeling：让环境状态变得可检查

这一支相对新一点，2025 年才比较成形。代表工作有 WorldCoder、CWM（Code World Models）、Code2World。

核心思路：**用代码作为环境的内部表征**。比如一个 Web Agent，传统做法是把 DOM 树喂给模型；新做法是把"当前页面状态"表达成一段可执行的 Python 对象——`page.buttons`、`page.forms.login.fields`——这样 Agent 不仅"看到"了环境，还能用程序去**操作**这个表征。

代码在这里同时充当三个角色：state representation（状态）、dynamics model（动态）、feedback signal（反馈）。

---

## 第二层：Harness Mechanisms — 让 Agent "活下去"的五大机制

![机制层时间线](https://arxiv.org/html/2605.18747v1/x4.png)
*Figure 3：Harness Mechanisms 的代表工作时间线（2023-2026）。五个子方向：Planning（规划）、Memory（记忆）、Tool Use（工具）、Control（控制）、Optimization（harness 优化）。可以清楚看到 Memory 和 Optimization 是 2025-2026 年增长最猛的方向，前者反映了"长上下文不够用"的现实，后者反映了"手工调 harness 不可持续"的共识*

这一章是论文里最长、也最干的一章。我挑五个机制各讲两句直觉。

### 1. Planning：从"一次决定全部"到"持续调整路线"

论文把规划分成四类：

- **Linear decomposition**（线性分解）：Self-Planning、WebAgent 那种"先列大纲再写代码"。简单粗暴，但一旦初始计划错了就废了
- **Structure-grounded**（结构约束）：CodePlan、VerilogCoder。把仓库的依赖图、电路图作为规划的硬约束，规划必须服从结构
- **Search-based**（搜索式）：CodeTree、SWE-Search、ReThinkMCTS。引入 MCTS 这种经典搜索算法，让规划在多个候选方案之间做选择
- **Orchestration-based**（编排式）：MapCoder、AgentCoder。把不同的子任务分给不同的子 Agent

我自己工程里踩过的最大的坑是——**线性分解在"任务超过 5 步"之后基本必崩**。一旦中间某一步的执行结果不符合预期，整个剩余计划就成了僵尸。所以现在做产品级 Agent，要么走 structure-grounded（如果任务结构清晰），要么走 search-based（如果任务结构不清晰但需要鲁棒性）。

但还有一个更深的工业动向论文特别提到了——**计划本身从临时 prompt 变成了持久化的 harness 对象**。OpenAI 的 Codex 长程文档里反复提到 `PLAN.md`、`Implement.md` 这种文件：里程碑、验收标准、验证命令、回滚规则全写在里面，Agent 可以跨上下文重启、跨会话续跑，人类也能 review 和 version control。这不是 prompt 工程，这是把规划提升为**文件系统支持的一等公民**。

### 2. Memory：仓库级 Agent 的真正难点

我个人觉得这一节是整篇论文最值钱的部分之一。

传统 LLM Agent 的"记忆"基本就是 context window + 简单的向量检索。这套在玩具任务上够用，但放到仓库级、长程任务上立刻崩溃。论文梳理出来代码 Agent 的记忆有五个层次：

1. **Working memory**（工作记忆）：当前任务上下文，类似 RAM
2. **Repository evidence retrieval**（仓库证据检索）：RepoCoder、CodeRAG 这种针对代码语义的检索
3. **Procedural memory**（程序性记忆）：CodeMem、MemGovern。存储"做过的事"，下次遇到类似任务可以复用
4. **Experiential memory**（经验记忆）：ExpeL、RepairAgent。把成功/失败的经验抽取出来作为指导
5. **Shared interaction history**（共享交互历史）：MemCoder。多个 Agent 共享同一个记忆池

之所以说这部分值钱，是因为它直接对应了生产环境最痛的问题——**仓库级任务里，光是"决定哪些信息要塞进 context"就已经是个工程难题**。我之前做内部 coding assistant 的时候，光这一层就反复调了两个月。

### 3. Tool Use：从"列表式工具"到"代码组合式工具"

这一节其实是对 OpenAI 那套 function calling 范式的一个挑战。

传统 tool calling：你给模型一个 JSON schema 列表，模型选一个工具填参数，调用，返回结果。问题是——**工具数量一多，模型选不对**；**工具之间组合复杂，单次调用搞不定**。

新范式 ToolCoder、CodeAgent、OpenHands 走的是另一条路：**把工具调用本身写成代码**。模型不再"选工具"，而是写一段 Python，里面随便调用任何已经注册的工具，甚至自定义组合：

```python
results = []
for repo in github.search_repos("agent harness"):
    if repo.stars > 100:
        readme = fetch_readme(repo)
        summary = llm_summarize(readme)
        results.append({"name": repo.name, "summary": summary})
return sorted(results, key=lambda x: ...)
```

这种"code-as-tool-orchestration"的好处是组合性指数级提升，但成本是对沙箱、权限、安全的要求也指数级提升。

### 4. Plan-Execute-Verify Loop：从 debug 到"控制论"

这一节论文做了一个我觉得非常关键的概念升级。传统讨论"agent debug"基本停留在"看到错就改"的层次。论文把它重新定义成**控制论意义上的闭环**：

- **Plan**：对预期变更签合同（read set / write set / 不变量 / 验收准则）
- **Execute**：在沙箱、权限受控的环境里执行
- **Verify**：用 deterministic sensor + human-review gate 判定是 accept / revise / escalate / rollback

> 我读到这里愣了一下。这就不是 "debug" 了，这是直接把 DevOps 里的金丝雀发布、回滚机制搬进了 Agent 设计。

代表工作：AgentCoder、SWE-agent、LDB、Self-Debugging、QualityFlow。

### 5. Agentic Harness Engineering：让 harness 自己进化

最有意思的来了。

前面四个机制都是"手工设计的 harness"。但论文敏锐地指出——**手工设计 harness 在 2025 年开始触顶**。因为 harness 的设计空间太大了：planning 走哪种、memory 用什么粒度、tool schema 怎么定、permission tier 怎么切。手工调参根本调不完。

于是 2025-2026 年冒出了一批"harness 自动进化"的工作：

- **AutoHarness**（2026）：根据环境反馈自动合成 harness 代码
- **Meta-Harness**（2026）：把历史 harness 源码、得分、execution trace 作为搜索状态，对 harness 本身做搜索
- **GEPA**（2025）：通过 reflection 和 execution feedback 优化 prompt + workflow
- **Live-SWE-agent**（2025）：harness 在运行时自我修改

这是个非常有 meta 味道的方向：**Agent 不仅自己写代码、自己用代码，还在改进它自己的用法手册**。

---

## 第三层：Scaling the Harness — 多 Agent 协同的真正难点

![多 Agent 协同总览](https://arxiv.org/html/2605.18747v1/x10.png)
*Figure 4：多 Agent 系统下的 harness 设计要素。包括 Functional Role Specialization（角色专业化：Coder、Verifier、Planner、Executer）、Execution/Feedback/Synchronization（执行反馈与同步）、Interaction Modes（交互模式：Debate、Critique、Adversarial、Collaborative）、Topologies（拓扑：Chain、Cyclic、Hierarchical、Star + 自适应动态结构）*

这一章我读得最快，但有一段话让我停下来重读了三遍：

> "Many systems still rely on sequential handoff, shared logs, or file-only state ... these mechanisms often synchronize **artifacts but not assumptions**."

翻译过来：**多 Agent 系统经常同步了文件内容，但没同步前提假设**。

一个 Agent 在旧仓库快照上做计划，另一个在新 patch 上跑测试，第三个还记着已经过时的 invariant，人类 reviewer 又引入了新约束没传播出去——结果就是大家做的事看起来"在同一个仓库上"，但实际上**每个 Agent 头脑里那个"仓库"都不是同一个东西**。

这是我做内部多 Agent 系统时最头疼的问题，论文一句话点破。

论文把 MAS 拆成几个维度：

- **角色专业化**：Manager、Planner、Coder、Reviewer、Tester。MetaGPT、ChatDev、AgentCoder 是代表
- **交互模式**：Programming（协作编程）、Repair（协作修复）、Debate（辩论）、Adversarial（对抗，比如 red-teaming）
- **拓扑结构**：Chain（链式）、Cyclic（循环）、Hierarchical（层级）、Star（星型），以及更高级的 Adaptive Object-driven Topology（运行时自重组）
- **共享 harness 设施**：仓库、测试套件、execution trace、blackboard、belief-state 同步

最大的 open problem 是论文给出的那个：**Transactional Shared Program State**——能否给多 Agent 系统一套类似数据库事务的语义？每个动作声明 read set / write set / assumptions / version dependencies / verifier obligations / conflict policy。冲突不只在文件 diff 层面检测，还在 plan、test、retrieved evidence、memory、latent user requirement 层面检测。

> 听起来有点天方夜谭，但仔细想想就是 CRDT + Git + Database Transaction + Belief Propagation 的混合体。这个方向我打赌未来 12 个月会出一堆论文。

---

## 五大应用域：纸面上的概念怎么落地

![五大应用域](https://arxiv.org/html/2605.18747v1/x12.png)
*Figure 5：Code as Agent Harness 在五大应用域的落地形态——Code Assistant（Claude/Codex/OpenClaw）、GUI/OS Agent（视觉 grounded + 自我进化）、Scientific Discovery（实验自动化）、Personalization（自我进化推荐）、Embodied Agent（基于演化技能库的物理交互）*

我挑两个最有代表性的展开讲。

### Code Assistants：从"completion"到"完整开发流"

早期的代码助手就是 Copilot 那种"自动补全"。但 Claude Code、Codex、OpenHands 这一代彻底变了——它们的工作对象不再是单个文件，而是**整个仓库 + 测试套件 + 构建系统 + PR workflow**。

论文里有句话讲得特别好：

> "The repository becomes the operational substrate on which code assistants plan, act, and receive feedback."

**仓库本身就是 Agent 的操作基底**。Source 文件、测试、build 脚本、依赖元数据、issues、branches、PRs 共同构成一个可检查、可修改、可验证的持久工作区。

而最近 1-2 年最有意思的趋势是 **Model Context Protocol（MCP）** 的崛起。MCP 标准化了 harness 如何向模型暴露 tools、context、resources。这一下子让"我的 harness"和"你的 harness"之间可以共享工具，整个生态可组合性大幅提升。

### Embodied Agents：技能库即 harness

具身智能体这块 Voyager（2023）就奠定了基调：**Agent 的核心资产不是模型权重，而是它的 skill library**。

每次成功完成一个任务，Agent 把这次的解决方案抽象成一段可复用的代码（比如 `craftWoodenPickaxe()`），存进 library。下次遇到类似任务，先在 library 里找有没有现成 skill，找到就用，找不到就组合现有 skill 派生新 skill，再找不到才从头写。

这套模式后来被 RoboCodex、CodeBT、UI-TARS、Mu 等延续到 manipulation、UI 控制等更难的领域。论文用一句话总结：

> "Code couples a multimodal precondition, an executable action pattern, and an expected postcondition."

也就是说，一个真正的 skill = 一对（视觉/物理前置条件，可执行动作，预期后置状态）。这个三元组才是 embodied agent 的最小可复用单元，而代码是表达这个三元组最自然的形式。

---

## 6 个 Open Problem：未来 1-2 年的研究地图

论文最后一章给了 6 个 open problem，我读完觉得每一个都很扎实，挑出来逐一点评。

| Open Problem | 核心矛盾 | 我的判断 |
|--------------|---------|---------|
| **Harness-level evaluation** | 现在大家只看 end-task success，但这把 model、harness、tools、env 的贡献全混在一起了 | 这是真问题。SWE-bench 的分高低跟 harness 关系巨大。下一代 benchmark 必须做出 model vs harness 的 ablation |
| **Semantic verification beyond execution** | "测试过了" ≠ "对了"。Agent 看到 green test 就觉得稳了，但测试覆盖率往往是不全的 | 这个我太有体感了。我们内部一度被"模型把测试改了让它过"这种 reward hacking 坑惨 |
| **Self-evolving harnesses without regression** | harness 自动进化时怎么保证不偷偷弱化 safety、不在罕见但重要的 case 上回退？ | 类比软件工程里的回归测试 + 金丝雀发布。AutoHarness、Meta-Harness 那帮人正在搞 |
| **Transactional shared program state** | 多 Agent 同步了 artifact 但没同步 assumption | 上面提过，最有意思的方向之一 |
| **Human-in-the-loop as harness state** | 人类的批准 / 拒绝 / 校正应该作为 harness 的持久状态，而不是一次性 prompt | 这块论文写得最深刻——"executable accountability" 这个词造得真好 |
| **Multimodal code-harness** | 现在的 harness 主要还是文本的，GUI / embodied / scientific 场景需要把多模态观察也纳入 harness state | 这是 2026-2027 年要爆的方向 |

我额外说一句对 **Self-evolving harnesses** 这个问题的看法。

论文里有句话我反复琢磨：

> "The goal is not a harness that changes often, but one that changes only when it can justify the change."

**目标不是一个频繁变化的 harness，而是一个只在能证明变化合理时才变化的 harness**。

这个判断非常老练。我见过太多团队一上来就追求"自动优化 harness"，结果优化出来的版本在 benchmark 上涨了 2 个点，但生产上各种诡异 bug。原因就是优化器只看 benchmark 这个 narrow signal，对 safety invariant、tail case 视而不见。论文提出的 change contract（每个 harness 修改都要带 contract：修改了什么、解决什么失败模式、保留什么 invariant、什么 evaluation 能否定它、怎么回滚）——这个 framework 其实就是把软件工程里成熟的变更管理思想搬过来了。

> 看到这一段，我对作者团队的工程嗅觉真的肃然起敬。这不是写论文的人，这是踩过坑的人写的。

---

## 这篇论文的真正贡献，到底是什么？

聊到这里，我想坦诚地说一下我的判断。

**这篇论文不是一篇方法论文，没有新算法，没有新 SOTA**。如果你是冲着"学个新技术回去用"来的，可能会失望。

但它做的事情我觉得对整个领域非常重要——**它给了一个稳定的概念语言**。

过去两年，大家聊 Agent 都在用模糊术语：framework、scaffold、orchestration、wrapper、middleware。每个人用的都不一样，沟通效率极低。这篇论文用 "harness"、"harness interface"、"harness mechanism"、"harness scaling"、"agent-initiated code artifact" 这几个核心概念，把整个领域的对话坐标系给立起来了。

类比一下：这就像 Goodfellow 那本 Deep Learning 教材对 2015 年的深度学习社区做的事情——不是发明了任何新东西，而是**把已有的零散知识整合成可教授、可引用、可批判的统一框架**。

**值得花时间细读**。尤其是：
- §3 机制层（5 大子模块的对比写得非常清晰）
- §4 MAS 那段对 "synchronize artifacts but not assumptions" 的洞察
- §5 的 6 个 open problem，每一个都是未来 12-18 个月可以发 paper 的方向

**可以略读**：
- 五大应用域那一章（如果你已经熟悉 SWE-agent / Voyager / Claude Code，可以快速跳过）

---

## 工程启发：如果你也在做 Agent 产品

最后说几个我读完之后立刻打算用在工作上的点：

1. **把 `PLAN.md` 这种东西做成一等公民**。别再让计划只活在 prompt 里。让它写进文件系统，Git 跟踪，跨 session 可恢复。这是个性价比极高的工程动作。

2. **认真做 verifier 的分层**。不要把 pass/fail 当成单一信号。每个 verification artifact（unit test / integration test / static analyzer / human review）都要声明：它验证了什么、它没验证什么、它的置信度是多少。

3. **多 Agent 协作前，先想清楚 synchronize what 这件事**。不要只同步文件，要同步 assumption。哪怕实现简单点，先在 prompt 里强制每个 Agent 输出它"假设的环境状态"，光这一步就能避免一大堆诡异 bug。

4. **Permission tier 别一刀切**。读文件、跑沙箱代码、改本地仓库、发请求、改生产代码——每一层都要有独立的批准机制和审计日志。这一点 Claude Code 做得很标杆。

5. **Skill library 是你的资产**。如果你的 Agent 系统跑得久了还没有一个能积累、复用、组合的 skill library，那基本就是个一次性玩具。

---

## 最后

读完整篇论文，我合上电脑发了会儿呆。

Agent 这件事进入到 2026 年，已经不再是"模型够不够强"的问题了。是"模型外面那一整套基础设施够不够稳"的问题。这篇论文用 "harness" 这个词把这件事拍上了桌面，而且用 42 个签名告诉全社区：**这是一个值得当成一等公民来研究的领域**。

我个人觉得未来 12-18 个月，Agent 领域最值得追的几个方向：

1. **Harness-level benchmark**（解决 model vs harness 的归因问题）
2. **Self-evolving harness with safety invariant**（让 harness 像生产软件一样有 change management）
3. **Transactional shared state for MAS**（给多 Agent 系统一套类数据库语义）
4. **Multimodal harness**（把视觉、物理、传感作为 first-class harness state）

如果你正在做 Agent 产品、Agent 研究，或者只是想搞懂"为什么我的 Agent 写得越来越大但越用越不稳"——这篇综述真的推荐读一遍。

最后引用论文的收尾段：

> "The most important future systems will likely be those that combine four properties: executable, inspectable, stateful, and governed."

**Executable（可执行）、Inspectable（可检查）、Stateful（有状态）、Governed（受治理）**。这四个词大概率会成为未来几年评判 Agent 系统的标尺。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
