# MUSE-Autoskill：把 Agent 技能从"一次性产物"管成"有生命周期的资产"

## 核心摘要

最近几个月看了一连串"自动造 skill"的 Agent 论文——Voyager、AutoSkill、EvoSkill、SkillGen、Skill1、SkillOS……每篇看完心里都有一个相似的疑虑：**skill 被造出来之后，就完事儿了？**

绝大多数工作把 skill 当作一次性产物——任务来了，让 LLM 生成一段代码或一段 procedure，挂到 skill bank 上，下次再有类似任务就检索调用。但真实的工程系统不是这么用 skill 的。代码会出 bug、会过时、文件名会变；同一个 skill 在不同任务里会暴露不同的失败模式；不同 agent 会需要"用同一份 skill 但带着各自的踩坑笔记"。

字节跳动 ByteBrain 团队这篇 MUSE-Autoskill 想把这件事彻底框起来：**skill 不应该是一次性 artifact，而是一个有完整 lifecycle 的资产**——创建（Creation）、记忆（Memory）、管理（Management）、评估（Evaluation）、精炼（Refinement）。每个 skill 不仅有 SKILL.md 描述自己，还配一个 `.memory.md` 攒经验，配 `tests/` 跑单元测试，配 `scripts/` 装可执行代码，跑失败就让 agent 自己 patch 后重跑。

实验在 SkillsBench（51 个任务、4 个领域）上，用 GPT-5.5 做 backbone，对比 Codex、Hermes、MUSE-Autoskill 三个 agent。最有意思的不是 MUSE 自己拿了 68.40% 的 SOTA，而是另外两个数：**MUSE 自己生成的 skill，在能生成出来的 35 个任务上，Phase 2 准确率干到 87.94 个百分点**，比人类专家手写的 skill 平均水平（68.40%）高将近 20 个点；更狠的是，把这批 MUSE 生成的 skill 直接丢给 Hermes，**Hermes 不做任何改动就能恢复 79 个百分点的人类 skill gap**，证明这些 skill 是真正可迁移的"知识资产"，不是绑定在 MUSE 自己内部的胶水。

我读完的判断是：这篇论文最值钱的不是某个具体 trick，而是把"skill = lifecycle-managed asset"这个 framing 钉死，并且配了一套相当工程化的实现（DAG-of-ReAct、两级上下文压缩、per-skill memory）。它不会是某种范式革命——很多组件其实是 Voyager / Reflexion / MemGPT / Anthropic Agent Skills 已有思路的整合——但**整合本身做得很扎实，而且整个东西已经在线上跑（SkillMarket、ArkClaw、SkillHub），不是 demo**。

---

## 论文信息

- **标题**：MUSE-Autoskill: Self-Evolving Agents via Skill Creation, Memory, Management, and Evaluation
- **作者**：Huawei Lin, Peng Li, Jie Song, Fuxin Jiang, Tieying Zhang
- **机构**：ByteDance Inc.（ByteBrain 团队）+ Rochester Institute of Technology（第一作者实习期间完成）
- **通讯作者**：Tieying Zhang（tieying.zhang@bytedance.com）
- **arXiv**：https://arxiv.org/abs/2605.27366（2026 年 5 月 26 日）
- **规模**：30 页正文，8 张图，13 张表

---

## 问题：现有"自动造 skill"方法到底缺什么

先说为什么这件事值得做。

ReAct 时代的 agent 是把 tool 列表写死在 prompt 里。Voyager 那一拨开始让 agent 自己写 skill 代码，Anthropic Agent Skills 把 skill 标准化成可移植的 SKILL.md + scripts/ 文件夹。从 Voyager 到 AutoSkill / EvoSkill / SkillGen，再到用 RL 联合优化的 Skill1 / SkillOS / SkillMaster——thread 本身一直在往前走。

但作者提了一个非常具体的批评，把现有方法的四个 gap 数得很清楚：

1. **Creation–usage mismatch**：很多方法是"离线挖 skill"，从历史 trajectory 里 distill 一批 skill 然后再上线。问题是 skill 生成时**根本不知道 agent 当前的 runtime context**——它生成的 skill 长什么样、依赖什么文件路径、需要什么环境，跟实际调用时的状态可能对不上
2. **没有 per-skill memory**：现有 skill 一旦生成就是"死的"。同一个 skill 这次跑挂了的原因、上次发现的 input 格式坑，全部丢进 trajectory 里就过了。下次再调用这个 skill，agent 还得重新踩一遍坑
3. **Static、unvalidated**：skill 生成出来直接进 bank，没有人测过它能不能跑、能不能通过单元测试。Voyager 倒是有 self-verification，但只针对 Minecraft 那种封闭环境
4. **Context handling 烂**：长 horizon 任务里 trajectory 越来越长，flat conversation history 直接塞模型，要么超 token 预算，要么把关键信息埋在中间被忽略（即典型的 "lost in the middle" 问题）

这四个 gap 拼起来就是一句话：**现在的 skill 是 disposable artifact，不是 managed infrastructure**。

而 MUSE 想要的是后者。

---

## MUSE 总览：把 skill 拆成五个 lifecycle 阶段

![图1：MUSE-Autoskill 架构总览](https://arxiv.org/html/2605.27366v1/x1.png)

*图1（论文 Figure 2）：四个象限分别对应 lifecycle 的四组关键能力——左上 Skill Creation（Task → LLM → Scripts/Tools/Resources → Skill）；右上 Skill Management（Skill Bank + Retrieval → Merge/Update/Forget）；左下 Self-Evolution & Correction（Unit Test → Evaluate → Refine 闭环 → Updated）；右下 Memory Mechanism（Short-term / Long-term / Skill-level，Skill-level 还可以 Share 给其他 Agent）。Creation 和 Evaluation 在论文里其实是一体的——skill 生成完会立即跑 tests/，过了才注册进 bank。*

五个 lifecycle 阶段写在论文里是 creation、memory、management、evaluation、refinement。我个人的理解是 evaluation 和 refinement 应该看成同一个机制的两端：evaluate 给信号，refine 用这个信号 patch skill。

整套系统启动时只有两个 built-in skill：`skill_create` 和 `web_search`。**其他全部 skill 都是 agent 自己在跑任务过程中现造出来的**。这点其实挺关键——它不像 SkillOS 需要专门的 curator 模型，也不像 Voyager 绑定在 Minecraft 这种封闭世界，MUSE 整套机制是 training-free 的、跨 backbone 的。

### 端到端流程

![图2：MUSE 端到端工程流程图](https://arxiv.org/html/2605.27366v1/x2.png)

*图2（论文 Figure 3）：Master Agent 跑 ReAct 主循环（Plan → Action → Observation）。当某一步需要 skill：先去 Skill Bank 查（query），命中就 load；没命中走右下的 Synthesis 分支，调 Skill Creator 生成新 package（SKILL.md + scripts/ + tests/）。生成完进入橙色的 Sandbox，由 Skill Executor 加载 SKILL.md、隔离运行 scripts、产出 observation 和 artifact，然后送进 Evaluator 跑测试。Evaluator pass → log 到 Memory（按 skill / long / short 三层组织）；Evaluator fail → 红色的 Refiner 分支，根据失败信号 patch 对应文件（图里画了一个真实的 diff：给 parse 函数加 timeout=30 retry），patch 完回到 Sandbox 重测。*

这张图其实回答了"creation–usage mismatch"那个 gap：**skill_create 不是一个离线脚本，是 ReAct 主循环里的一个 tool**。Agent 在做任务的当下决定"现有 skill 不够，我得造一个"，造的时候就在当前 task 的 runtime context 里，造完就用，用完就攒经验。

### Skill 包结构

每个 skill 是一个文件夹，跟 Anthropic Agent Skills 兼容：

```
my-skill/
├── SKILL.md          # 接口：name、description、inputs、outputs、procedure
├── .memory.md        # 跨任务累积的经验（这一篇论文的关键创新）
├── scripts/          # 可执行代码（可选）
├── resources/        # 辅助数据（可选）
└── tests/            # 单元测试（用于 evaluation 阶段）
```

注意 `.memory.md` 是 MUSE 相对 Anthropic Agent Skills 加上去的"per-skill memory scope"。论文里提到这是它跟 MemGPT、Generative Agents、Reflexion 这些经典 memory 工作的关键区别——**它把记忆挂在 skill 这个抽象上，而不是挂在 agent 或 episode 上**。

后面会看到，这个设计在 cross-agent transfer 实验里非常关键。

---

## 关键机制 1：DAG-of-ReAct + 两级自适应上下文压缩

这是论文里我个人最喜欢的一块工程实现。说实话第一眼看到 "DAG of ReAct turns" 我还以为是花式包装，看完发现真的是把上下文管理这件事想得很透。

![图3：两级自适应上下文压缩](https://arxiv.org/html/2605.27366v1/x3.png)

*图3（论文 Figure 4）：每个 ReAct turn 是一个 (plan, action, observation) 三元组。系统恒定保留最前面 KEEP_FIRST 个 turn（system prompt + 早期规划）和最后面 KEEP_LAST 个 turn（最近上下文），中间区域才允许压缩。* 

*顶部 → 中部（L1 触发）：当某个 turn 的 token 超过 per-node 阈值（图里 T3 占了 20K，T4 占了 15K，total 71K 超出 50K budget），L1 在原位重写这两个 turn 为 5K 的摘要，结构保持不变。*

*中部 → 底部（L2 触发）：当 L1 之后单 turn 都不超标但总长度仍超预算（56K vs 50K），L2 把一段连续的 turn（图里 T3∑、T4∑、T5）合并成一个 6K 的合成 node，topology 上替代原来这一段。*

*关键：原始 turn 全部保留在 full history 里（用 immutable history_prev / history_next 指针链着），任何先前状态都可以被 replay 或在不同会话间 resume。*

我把它的精妙之处拆开聊：

**为什么 L1 优先于 L2？**论文给了一个非常工程的判断：L1 只重写"惹事的那个 node"的 payload，每个 turn 的边界和 plan/action/observation 结构完整保留，**下游 turn 引用上游 turn 的位置编号不会变**。L2 一旦合并多个 turn 进同一个 synthetic node，turn-level 结构就丢了，所以 L2 是退路。

**为什么要 pin 住首尾？**这点其实呼应 "lost in the middle" 那篇研究——长上下文里中间内容关注度最低。把 system prompt + 早期 plan 钉在首部、最近 turns 钉在尾部，被压缩的恰好是模型本来也最容易忽视的中段。

**为什么 full history 要保留？**因为 long-term memory 和 skill bank 是按 session 持久化的，full history 用 immutable pointer 串起来，**任何中间状态都可以 replay**——这对长 horizon 任务在多个 session 之间接力非常关键。MUSE 的真实部署场景里，一个任务跨多个 session 是常态。

放在跟其他工作的坐标里看：LLMLingua 这类是 token-level prompt 压缩；StreamingLLM 是 attention-sink-based KV 保留；MemGPT 是 OS 风格的虚拟内存分页。MUSE 选了一个更"agent-aware"的层级：**以 ReAct turn 为基本单位**，保持语义粒度的同时控制 token 预算。

---

## 关键机制 2：三层记忆 + Skill-level Memory

记忆这块的层级是这样的：

| 层级 | 存什么 | 是否压缩 | 作用域 |
|---|---|---|---|
| **Short-term** | 当前 task 的中间推理、observation、临时执行结果 | 是（用上面那套自适应压缩） | 单任务、单 session |
| **Long-term** | 跨 session 的可复用结论、环境怪癖、通用经验（如 "prefer batched I/O"、"项目 pin 死了 package 版本"） | **否** | 跨任务、跨 session |
| **Skill-level**（MUSE 独有） | 挂在每个 skill 旁边的 `.memory.md`：已知失败模式、输入格式坑、性能注意事项 | 否 | 跟着 skill 走 |

Skill-level memory 是 MUSE 相对前作真正新的东西。设想一下：你写过一个解析 PDF 的 skill，第一次用发现某些扫描件 OCR 后会乱码、需要先做去噪；第二次用发现 100MB 以上的 PDF 要分页处理避免 OOM。这两条经验如果只丢进 episodic memory 或 long-term memory，下次别的任务调这个 skill 时，系统不会自动把这两条经验拉出来——它根本不知道这两条经验跟这个 skill 有关。

MUSE 的做法是把它直接钉在 skill 上。每次加载 SKILL.md 的同时，`.memory.md` 也会被 surface 给 agent。这就把"经验"从 agent-bound 解耦成了 **skill-bound**。

后面 cross-agent 实验会发现，这个设计直接让 skill（连同它的经验）在不同 agent 之间可移植。

---

## 实验 1：人类 skill 提供了一个"统一基线"

实验设计本身挺干净。三个 agent 都用 GPT-5.5 做 backbone：

- **Codex**（OpenAI 的 coding agent）
- **Hermes**（另一个 ReAct 风 agent）
- **MUSE-Autoskill**（本文）

51 个 SkillsBench 任务，每个任务 5 次独立运行，macro-average over tasks。

### Table 2：人类 skill 普遍涨点，MUSE 涨得最多

| Agent | Without Skills | With Human Skills | Lift |
|---|---|---|---|
| Codex | 52.11% | 67.28% | +15.17 |
| Hermes | 47.89% | 61.21% | +13.33 |
| **MUSE-Autoskill** (Ours) | 53.19% | 68.40% | +15.21 |

这个结果其实有两层意思：

第一层是验证"skill 机制本身有用"——三个 agent 加了 skill 都涨了 13-15 个点，这跟具体哪个 agent 没关系，是 skill 这个抽象在起作用。

第二层有点微妙——**MUSE 不仅在 with-skills 拿了最高分（68.40%），在 lift 上也最大（+15.21）**。论文的解释是"MUSE 在 reading、interpreting、applying skill content 上更强"。我个人对这个解释保持谨慎——MUSE 可能也只是因为 ReAct loop 跑得更深（后面会看到 18-19 turn vs Codex 11-12、Hermes 13-14），有更多机会调用 skill。这两种解释从这张表上看不出来谁更对。

### Table 3：分领域看，MUSE 在 3/4 领域领先

| 领域 | # | Codex w/ hum | Hermes w/ hum | MUSE w/ hum | 最优 |
|---|---|---|---|---|---|
| Science & Engineering | 14 | **78.57** | 72.86 | 72.86 | Codex |
| Data Analysis | 15 | 60.22 | 47.39 | **61.78** | Ours |
| Document Processing | 9 | 84.44 | 82.22 | **88.89** | Ours |
| Ops & Planning | 13 | 51.38 | 50.08 | **57.08** | Ours |
| **Macro-average** | **51** | 67.28 | 61.21 | **68.40** | Ours |

Science & Engineering 那一栏 MUSE 输 Codex 5.7 个点，论文坦诚地点了三个具体失败任务（lake-warming-attribution、flood-risk-analysis、radar-vital-signs），原因是 "verifier 惩罚没有在 task spec 里钉死的方法学选择"。这种诚实度在 paper 里挺难得，没把所有数据都洗成"我赢"的样子。

---

## 实验 2：MUSE 自己造的 skill，反而比人类 skill 还强（在它能造出来的任务上）

这是论文真正的"亮点"实验。流程是 two-phase：

- **Phase 1**：MUSE 不带任何 skill 跑这 51 个任务（5 次/任务）。在至少有一次成功的任务里，挑最佳 trajectory，调用 `skill_create` 把它 distill 成 SKILL.md（+ 可选 scripts）
- **Phase 2**：把生成的 skill 注入回去，重跑 5 次

51 个任务里 MUSE 成功生成 skill 的有 35 个（68.6%）。剩下 16 个 Phase 1 完全失败、没 trajectory 可 distill，在 51-task average 里按 0 计入。

### Table 4：Self-created skill 涨 +7.16

| Configuration | Accuracy (51 tasks) |
|---|---|
| MUSE without skills (baseline) | 53.19% |
| MUSE with human skills (reference) | 68.40% |
| **MUSE self-created skills** (Ours) | 60.35% |

光看 51-task 的 60.35% 你可能觉得"还不如人类 skill"。但论文紧接着给了一个更关键的数：

> **在 35 个能生成 skill 的任务上，Phase 2 准确率达到 87.94%——比人类 skill 的 68.40% 还高 19.5 个点**。

51-task 的总分被拉低，是因为另外 16 个任务 Phase 1 完全失败、贡献了 0%。论文很诚实地点了这个：

> The primary bottleneck is therefore **coverage** (the agent's ability to solve tasks without skills) rather than the quality of the skills it generates.

这个表态我比较喜欢——它没有把瓶颈归咎于"skill 生成模块还需优化"，而是直接指出"Phase 1 走不下去"才是真正的天花板。后面 Bottleneck Analysis 还专门分析了这 16 个任务，主要集中在 scientific computing 和 system operations 这种 baseline 本身就弱的领域。

---

## 实验 3：MUSE 造的 skill 能跨 agent——这个数才是真正的"知识资产"证据

把 MUSE 在实验 2 里生成的那 35 个 skill 文件**原封不动**丢给 Hermes，再跑 51 个任务。

### Table 5：Hermes 用 MUSE 的 skill，恢复了 79% 的 gap

| Configuration | Hermes | MUSE-Autoskill |
|---|---|---|
| Without skills | 47.89% | 53.19% |
| **With MUSE-Autoskill generated skills** | 58.40% | 60.35% |
| With human skills (reference) | 61.21% | 68.40% |

我把这张表的几个数字挑出来看：

- Hermes 加上 MUSE 的 skill：47.89% → 58.40%，**涨 +10.51**
- Hermes 在人类 skill 下：61.21%
- 也就是说，MUSE 生成的 skill 在 Hermes 上把 47.89 → 61.21 这个 13.32 的 gap，**关掉了大约 79 个百分点**（10.51 / 13.32）
- 而且 Hermes 用 MUSE skill（58.40%）跟 MUSE 自己用 MUSE skill（60.35%），**只差 1.95 个点**

这才是论文真正硬的地方。

为什么这个数字硬？因为它直接打脸了一个常见怀疑——"自动生成的 skill 是不是只对生成它的 agent 有用？是不是 over-fit 到 MUSE 自己的内部 prompt 风格？"

这个 1.95 个点的 residual 说明：MUSE 生成的 skill 是**写成可读文档 + 脚本**的真实知识，不是某种隐式 prompt 工程。它能跨 agent 转移，意味着 skill 真的是一个 portable 的"knowledge asset"。

跟 SkillOS 那种"训出一个 curator"的路线对比就更清楚了：SkillOS 的 curator 跨 backbone 可迁移，但 skill 本身绑定 curator；MUSE 反过来，curator（其实就是 skill_create tool 调 LLM）不需要训练，**skill 自己就是迁移的单位**。两条路各有取舍，但 MUSE 这条路对工程落地确实更友好——它意味着公司内不同团队的 agent 可以共享同一份 skill 仓库。

---

## 实验 4：成本视角下的 Pareto 改进

很多 skill 论文吐槽点都在"涨点是涨了，但 token 涨得更狠"。MUSE 这块的数据看起来挺好。

### Table 6：Skill 生成是一次性成本，使用反而省 token

| Configuration | Tokens | Latency (s) | Turns |
|---|---|---|---|
| Skill 生成（Phase 2） | 383K | 164 | 7 |
| MUSE without skills | 578K | 684 | 20 |
| MUSE with human skills | 615K | 656 | 19 |
| **MUSE with generated skill** | **493K** | **411** | **15** |
| Hermes without skills | 181K | 370 | 14 |
| Hermes with human skills | 186K | 369 | 14 |
| **Hermes with generated skill** | **97K** | **257** | **13** |

注意几个有意思的数：

- **MUSE 用自己生成的 skill 反而比 without skills 省 token**（493K vs 578K，少 85K），同时 latency 从 684s 降到 411s，turn 数从 20 降到 15。这件事其实挺反直觉的——加了 skill 应该 prompt 变长才对——直觉上应该 token 涨但 turn 数降。这里之所以总 token 反而降，是因为 turn 数显著缩短（20→15），节省的 reasoning token 超过了 skill 本身占的 prompt token
- **Hermes 用 MUSE 生成的 skill 比用人类 skill 还省**（97K vs 186K，少一半）。这其实暴露了一个事实：人类写的 skill 描述往往不够"机器友好"，需要 agent 花更多 token 推理；MUSE 生成的 skill 因为是从成功 trajectory 里 distill 出来的，**结构跟 agent 的执行路径天然对齐**

![图4：MUSE 生成 skill 是 Pareto 最优](https://arxiv.org/html/2605.27366v1/x4.png)

*图4（论文 Figure 5）：双面板——(A) reward vs latency；(B) reward vs tokens。两个面板里 MUSE 生成 skill（深蓝实心圆）相对自己 without skills（蓝色空心圆）和 with human skills（蓝色浅填充）都是右上角的 Pareto 点：reward 更高（+11.0pp）+ latency 更低（-273s）+ token 更少（-85K）。Hermes 这条线（青色）也是同样的趋势：用 MUSE skill 比 without skills 涨 +15.3pp、latency -113s、token -84K。*

注意 Hermes 那条线：它用 MUSE 生成的 skill 涨幅（+15.3）比 MUSE 自己用（+11.0）还大。这其实是因为 Hermes 的 baseline（47.89%）比 MUSE 的 baseline（53.19%）更低，提升空间更大。但绝对值上，MUSE + MUSE-skill 还是最高的。

### 三个 case study 看 skill 到底在干嘛

论文挑了三个有意思的任务展开讲：

1. **adaptive-cruise-control**：要写一个离散 PID 控制器满足 verifier 对 overshoot、稳态误差、上升时间的约束。MUSE 不带 skill 时 5 次中 2 次成功（40%）。生成的 skill `adaptive-cruise-pid-controller` 把离散 PID 方程、anti-windup、增益调整启发式、verifier 要求的 JSON 格式都写死。Phase 2 准确率达到 **100 个百分点**。Hermes 用同一个 skill 从 20% 涨到 60%——证明 skill 装的是真实领域知识，不是任务记忆

2. **flink-query**：要写一个 Apache Flink Java 作业，处理 gzipped Google ClusterData trace、按微秒事件时间做会话化、按精确格式输出元组。Baseline 5 次中 1 次成功（20%），原因是 agent 在 turn budget 内从文档里恢复不了项目的 POJO 和 AppBase 骨架约定。生成的 skill 把 schema 解析、`clusterdata.utils.AppBase` 扩展协议、event-time session trigger、Maven 验证 recipe（带合成 gzipped 数据）全打包。Phase 2 准确率拉到 **满分**

3. **weighted-gdp-calc**：要在 Excel 里填两条件查找 + SUMPRODUCT 加权均值，保留原格式且不能用 macro/VBA。生成的 skill `excel-financial-formula-modeling` 钉死用 openpyxl、列出公式 pattern、加了一步用源数据重算目标 cell 的验证步骤。从 20% 直接拉到 **满分**。同一个 skill 描述把 Hermes 也带过去了

这三个 case 有个共同点：**都不是"靠 LLM 多想就能解决"的任务，而是"需要某段精确的领域过程性知识"的任务**。这恰好是 skill 这种抽象最擅长的场景——把模糊的领域过程变成确定性的可执行包。

论文也老实承认了一个 regression：**hvac-control 任务从 80 个点跌到 20 个点**。源 trajectory 用了一个标定窗口 + 增益估计例程，distill 出的 skill 把这套流程钉死了，但 Phase 2 在不同初始条件下这套流程不通用，反而拖累。这其实是 trajectory-distillation 的固有问题——distill 自一条成功路径，可能 over-fit 这条路径的特定假设。

---

## 关键机制审视：MUSE 生成的 skill 长什么样？

![图5：人类 skill vs MUSE 生成 skill 的解剖学对比](https://arxiv.org/html/2605.27366v1/x5.png)

*图5（论文 Figure 6）：(A) SKILL.md 行数分布——MUSE 生成的 skill 中位数 326 行，人类写的 146 行，**MUSE 大约长 2.2 倍**，且 IQR 更紧凑（更一致）；(B) Skill 包里有什么子目录——69% 人类 skill 只有 SKILL.md（无子目录），91% MUSE skill 也只有 SKILL.md。23% 人类 skill 带 scripts/、9% MUSE skill 带 scripts/。0% 人类 skill 带 tests/、9% MUSE skill 带 tests/（lifecycle 强制要求测试通过才注册）。*

这张图的 takeaway 我觉得有两层：

**第一层——MUSE 生成的 skill 不是更冗长，而是更过程化（procedural）**。论文检视过这些 skill，多出来的内容是 input/output schema、failure mode、step-by-step procedure 这种**人类作者会默认略掉的隐性知识**。LLM 写 skill 反而把这些显式化了。

**第二层：MUSE 的 skill 有 9% 带 tests 目录，而人类写的 skill 一个都没有**。这是 lifecycle 设计的强制结果——MUSE 的 skill 必须通过 `tests/` 才能注册进 bank。0 vs 9 个百分点听起来差距小，但意义不一样：**人类作者天然不爱写测试，机器反而更愿意配测试**。从工程角度这是一个挺有价值的"颠倒"。

不过这里我也想吐槽一句：91% MUSE skill 仍然只有 SKILL.md（无 scripts、无 tests）。这意味着 lifecycle 里"评估 + 精炼"这条线在大部分 skill 上**根本没启动**——因为没有 tests 就没法 evaluate。所以 evaluation/refinement 这两个 lifecycle 阶段在实际数据里更像"少数 skill 上能跑"的能力，不是默认每个 skill 都过的环节。这个 gap 论文没有特别突出讲，但数据是放在那的。

---

## 三个 agent 的成本-质量权衡

![图6：3 个 agent × {with/without skills} 在两维度上的权衡](https://arxiv.org/html/2605.27366v1/x6.png)

*图6（论文 Figure 7）：(A) latency vs reward 上，三个 agent 加 skill 都是"上 + 左"的 Pareto 改进——reward 涨且 latency 不涨甚至降。MUSE 的 ReAct loop 跑得最深（中位 18-19 turns），但 wall-clock 跟 Codex（11-12）相近，因为 adaptive context compression 把每个 turn 的 prompt size 压住了。(B) tokens vs reward 上，arrow 仍向上但向右漂移——加 skill 会让 prompt 加长 + sandbox 工具调用增加，token 不是 free lunch。最 token-hungry 的 MUSE 用 ~12% 多的 token 换来 ~15pp reward 提升。*

这张图的关键 insight 不是"MUSE 最好"，而是**所有 agent 加 skill 在 reward-latency 上都是免费午餐，但在 reward-token 上不是**。论文说：

> Even the most token-hungry agent, MUSE-Autoskill, recovers ~15 pp of reward for ~12% more tokens. Coupled with prompt caching (which absorbs about half of the marginal input cost), this puts the dollar cost of the skill-induced lift well below what the raw token deltas suggest.

prompt caching 这个观察是真实工程视角——cache 命中率高的话，新增 skill prompt 的实际 dollar 成本远比 token 数显示的低。这是论文里少数显示出"我是真在工业系统跑这个"的细节之一。

![图7：51 任务 × 5 次的 per-task 成本分布](https://arxiv.org/html/2605.27366v1/x7.png)

*图7（论文 Figure 8）：(A) per-task latency；(B) per-task tokens。Hermes 在两个轴上都最 lean（中位 latency 351-370s，中位 token 163-172K，反映其更短的 13-14 turn loop）。MUSE 是最 token-heavy 的（515-577K，~Codex 286-312K 的 1.8 倍），因为 18-19 turn 更深。但 MUSE 的 IQR 比 Codex 窄——adaptive compression 把 per-turn prompt size 上限钉住了。*

这里有个我比较欣赏的细节：MUSE 的 token 中位数高，但**分布更紧**。这正是 adaptive compression 的目的——不是让中位数下降，而是让长尾被压住。在生产系统里这件事其实比"中位数低"更重要：你不希望某个任务突然消耗 5 倍的 token。

---

## 真实部署：SkillMarket / ArkClaw / SkillHub

论文第 5 节有意思——它直接讲 MUSE 已经在三个生产系统里被用了：

- **SkillMarket**：把 skill 创建 pipeline 暴露给终端用户，从一次成功的 trajectory 直接 distill 出可复用、自带测试的 skill 包，不需要人工编写。计划版本会加上 skill 版本管理和迭代刷新
- **ArkClaw**：把 skill retrieval 集成为 `find-skill` 能力——agent 在合成新 skill 之前先找现有 skill。计划扩展是把整个 agent 当作一个可调用的 sub-agent，让单个 skill 能封装委托式多 agent 行为
- **SkillHub**：把整个 skill lifecycle（创建、评估、记忆、管理、精炼）做成一个托管服务，团队可以一起存储、评估、治理 skill 以及它们累积的 per-skill 经验

这块挺有说服力的，把"实验 + 生产"这个闭环展示出来了。很多 agent 论文止步于 benchmark，MUSE 直接告诉你"这套 lifecycle 抽象已经在线上验证过了"。

不过我得稍微保留一下：论文没披露这些系统的规模数据（多少用户、多少 skill、QPS 多少），所以"在用"这个表述的强弱无法判断。但至少这套 framing 不是纯学术想象。

---

## 我的判断：哪些是真贡献，哪些是包装

写到这儿可以做一个总结评估了。

### 真正的贡献

1. **Lifecycle framing 本身**。把 skill 提升为 "managed, testable, transferable infrastructure" 这个 framing，比某个具体 trick 更有长期价值。后续做 skill 系统的工作绕不开这个抽象
2. **Skill-level memory**。把 memory 从 agent-bound 解耦成 skill-bound，是论文里最干净的创新点。这件事在 cross-agent transfer 里直接 pay off
3. **DAG-of-ReAct + 两级压缩**。这是相当工程化的 contribution，对长 horizon agent 的状态管理是有用的工具
4. **Cross-agent transfer 那个 1.95 pp 的 residual**。这个数比"我们涨了 X 个点"硬太多——它直接证明 skill 是 portable 资产
5. **失败诚实度**。承认 16/51 任务 Phase 1 完全失败、hvac-control regression、Science & Engineering 输 Codex 5.7 个点。这种诚实在工业实验室论文里不多见

### 我会打折看的部分

1. **MUSE-Autoskill 拿了 SOTA 这件事本身意义有限**。68.40% vs Codex 67.28% 这个 1.12 个点的 gap 在 5-run macro-average 下置信区间不一定显著（论文 Appendix G 给了 std，但作者自己也提到 "with 5 runs per task, confidence intervals for individual tasks are wide"）。真正硬的是 87.94% on 35 tasks 和 cross-agent transfer，不是这 1.12 的 gap
2. **51/94 任务子集的选择偏差**。论文承认排除的任务"Docker 环境更复杂、可能更难"，所以汇报数字可能高估了系统级表现。这是个挺 honest 的 caveat 但确实削弱了主结论的强度
3. **每个 skill 从单条成功 trajectory distill**。论文自己也说 "may not represent the most general solution path"。hvac-control 的 regression 就是个例子。理论上应该 distill 多条成功 trajectory 的共性，而不是单条的细节
4. **每个 skill 在同一个任务上 evaluate**。Phase 1 → distill → Phase 2 都在同一个任务上，verifier 是确定的、ground truth 没塞进 skill，但**协议本身仍然耦合了 skill 内容和源 trajectory**。论文 limitations 自己也承认这点。真正可信的 skill 评估应该是 distill 自任务 A、evaluate 在任务 A'（同分布但不同实例）
5. **Lifecycle 五阶段里只有部分阶段在大部分 skill 上 active**。91% 的 MUSE skill 只有 SKILL.md，没 tests/、没 scripts/。evaluation 和 refinement 在大部分 skill 上没启动。lifecycle framing 在概念上完整，但落地数据上是"creation + memory 强、evaluation + refinement 弱"

### 跟同期工作怎么比

我把之前看过的几篇放一起对比：

| 工作 | 创建 | Per-skill 经验 | 单元测试评估 | 跨 agent 可移植 | Training-free |
|---|---|---|---|---|---|
| Voyager（2023） | ✓ | ✗ | △（self-verify） | ✗ | ✓ |
| AutoSkill | ✓ | ✗ | ✗ | △ | ✓ |
| EvoSkill | ✓ | ✗ | △（Pareto select） | △ | ✓ |
| SkillGen | ✓ | ✗ | △（contrastive） | △ | ✓ |
| Skill1（RL） | ✓ | ✗ | ✗ | ✗ | ✗ |
| SkillOS（RL curator） | ✓ | △ | ✗ | △（curator 可移植） | ✗ |
| Anthropic Agent Skills | ✓ | ✗ | ✗ | ✓（标准化格式） | ✓ |
| **MUSE-Autoskill** | **✓** | **✓** | **✓** | **✓（实测）** | **✓** |

MUSE 在表格上是"全占"的那一栏。但要诚实地说，**它的每一项单独看都不是首创**——Voyager 早就有 self-verification，Reflexion 早就有 reflective memory，Anthropic Agent Skills 早就标准化了 SKILL.md 格式。MUSE 的工作是把这些拼成一个 lifecycle 完整闭环并实测了 cross-agent transfer。

这种"整合型贡献"的价值取决于读者立场。如果你做研究，可能觉得这篇没有惊艳的新点子。如果你做工程系统，这篇相当于一份"该这么做"的蓝图。

---

## 工程启发：如果你要构建自己的 skill 系统

读完之后，我觉得有几条经验是可以直接拿走的：

1. **Skill 包格式向 Anthropic Agent Skills 看齐**。SKILL.md + scripts/ + tests/ 这个目录结构已经是事实标准，没必要自创格式。可移植性只跟标准化挂钩
2. **加一个 `.memory.md` 是 ROI 极高的设计**。挂在 skill 旁边，不需要改 agent 主流程。Agent 加载 skill 时把 `.memory.md` 一起 surface 进 prompt 即可。每次任务结束让 agent 自己 append 几条经验
3. **`skill_create` 必须做成 in-loop tool 而不是离线脚本**。这是 MUSE 解决 creation–usage mismatch 的核心。Skill 必须在它将被使用的同一个 runtime context 里被创造
4. **Tests 是 evaluation 的核心信号**。如果让 LLM 生成 skill 的同时生成 tests/，并且把 tests pass 作为 register-into-bank 的 gate，整个 skill bank 的可靠性会被实质性抬高一个台阶
5. **DAG-of-turns 是长 horizon agent 的合理状态抽象**。比 flat conversation history 强，比 OS 风格虚拟内存（MemGPT）轻量。如果你做的 agent 单任务跨 50+ turn，这套机制值得借鉴
6. **从单条成功 trajectory distill skill 有 over-fit 风险**。如果可能，应该收集多条成功 trajectory 后做 contrastive 抽取（这块 SkillGen 那套思路更安全）。MUSE 的 hvac-control regression 就是反例

---

## 收尾

这篇论文给我的最大启发不是某个具体机制，而是 **skill 不是 disposable artifact，而是 lifecycle-managed asset** 这个 framing 本身。

Agent 圈这两年一直在讨论"如何让 agent 自我进化"。大多数答案集中在 RL、self-reflection、memory architecture 这种偏算法的方向。MUSE 提了一个更工程化的答案：**让你的 capability primitives（skill）有版本、有测试、有经验、可治理、可迁移**。

这听起来不性感，但生产系统里通常是这种东西救命。

如果你团队正在搭一个长期演进的 agent 系统——不是 demo、不是 benchmark 跑分、是真的要让 agent 在三个月、半年、一年的尺度上变得更能干——MUSE 这套 lifecycle 抽象值得抄。

至于"自我进化"那个最高目标，我自己的看法是：MUSE 还没真的做到。**真正的自我进化要求 skill 在跨任务跨时间维度上变得更通用**，而 MUSE 目前每个 skill 仍是 task-bounded 的产物。下一步要解决的可能是 SkillGen 那种 contrastive 路径——从多条成功 + 失败 trajectory 里抽公共结构。

但这是个公平的下一步问题。MUSE 把 lifecycle 这层地基已经打得不错了。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
