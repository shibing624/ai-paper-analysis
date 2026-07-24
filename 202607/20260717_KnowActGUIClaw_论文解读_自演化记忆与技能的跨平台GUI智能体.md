---
title: "自演化记忆与技能的跨平台GUI智能体"
date: 2026-07-17
arxiv: 2607.12625
short_name: KnowActGUIClaw
authors: Yunxin Li, Jinchao Li, Shibo Su, Zhenran Xu, Chenrui Zhao, Tongshu Bian, Xiaoman Liang, Meishan Zhang, Baotian Hu, Min Zhang
institution: Lychee Team, Harbin Institute of Technology, Shenzhen
license: CC BY-NC-SA 4.0
---

# KnowAct-GUIClaw：把记忆和技能闭环塞进 GUI 智能体，让"用过就记住"在手机上变成现实

你有没有被手机 AI 助手气到过？让它把一封邮件里明天的会议时间记到日历，结果它打开邮件 App、把会议时间从一堆英文里抠出来、再打开日历、一点一点填——填错了时间格式，或者填完忘了"地点"那一栏。你再试一次，它又从头来一遍，不会从上一回学任何东西。

这类场景暴露的，是当下"个人助手式"智能体（personal assistant agent）最让人头疼的两件事：没法跨 App 协作搞定一条长链路任务，更不会从执行经验里学习。哈工大深圳 Lychee 团队提出的 **KnowAct-GUIClaw**（arXiv:2607.12625）就是冲这两件事去的。

---

## 核心摘要

OpenClaw、Nanobot 这类配置型（configuration-centric）的本地智能体框架虽然能挂工具、挂记忆、挂会话，但只要碰上"必须用 GUI 操作"的任务就露怯——没有跨平台 GUI 操控能力，也没有自我演化机制，每次都得从零摸索。

KnowAct-GUIClaw 的解题思路很直接：把 GUI 任务当作一个**部分可观测的决策过程（POMDP）**，套上一个 **Know-Route-Act-Reflect** 的四阶段闭环，靠两个长期存储（attribution-aware 经验记忆 + 自演化技能库）让每一次执行都能"反哺"下一次。两个最关键的工程效果：

- 在长程移动端基准 **MobileWorld** 上，用开源 **Kimi-K2.6** 当基座、开了完整 Know-记忆-技能之后拿到了 **64.1%** 的 pass@1，论文里标注"超过了 Seed-2.0-Pro（63.2）和 GPT-5.5（62.4）等所有代理框架和闭源代理模型"。但这个差距说实话只有不到 1 个百分点的领先，不算压倒性优势。
- 经验记忆和技能可以**跨基座模型迁移**：用 Kimi-K2.6 跑出来的经验丢给 35B 级的 Qwen3.5 executor，能把后者在 MobileWorld 上的 pass@1 从 24.8% 抬到 41.0%（+16.2 个点），抬到 Kimi 自身水平的 70% 左右。

我的判断：这是一份**扎实的工程整合**工作，把现有范式——记忆 + 技能 + 多级规划 + POMDP 化的 GUI 决策——按 Know-Route-Act-Reflect 的明确节奏串起来，在 MobileWorld 上做出了开源 SOTA。但其本质是系统级整合而非底层突破，"超过闭源模型"这种对比也要看具体口径。

---

## 论文信息

- **标题**：KnowAct-GUIClaw: Know Deeply, Act Perfectly, Personal GUI Assistant with Self-Evolving Memory and Skill
- **作者**：Yunxin Li, Jinchao Li, Shibo Su, Zhenran Xu, Chenrui Zhao, Tongshu Bian, Xiaoman Liang, Meishan Zhang, Baotian Hu, Min Zhang
- **机构**：Lychee Team, Harbin Institute of Technology, Shenzhen（深圳 AI 训练平台 / 深圳 Loop Area Institute 联合署名）
- **发表**：arXiv:2607.12625（v1 于 2026-07-14 提交，v2 于 2026-07-15 修订）
- **代码 & 复现日志**：https://github.com/HITsz-TMG/KnowAct/releases/tag/Result
- **许可证**：CC BY-NC-SA 4.0

---

## 问题动机：把 GUI 塞进 OpenClaw 不是个"集成一下"的问题

OpenClaw 那一类本地优先（local-first）的助手平台，主打"配置即能力"——把消息渠道、工具、记忆、会话、用户写好的 skill 都当成可声明的一等公民，让模型去调度。这种范式下，**结构化 API 工具调用**顺风顺水，但用户真正想干的活大半要靠 GUI：手机里翻 App、跨 App 搬数据、处理权限弹窗、在登录态里做连续操作。

直接塞一个 GUI 模型进去行不行？论文点出了四个具体痛点：

1. **跨 App 任务容易失忆**。一条指令可能横跨五个 App，靠一条 GUI 轨迹去"摘要所有中间结果"既冗余又易丢信息。简单写"目标 App = 高德"又会在指令模糊时强行制造幻觉。
2. **GUI 观察是部分可观测的**。截图 + 屏幕元数据 + 前台 App + 历史动作 + 内部推理日志，每一种都只是底层设备状态的一小片投影——后台导航栈、登录态、异步加载、表单"非可视值"都看不到。必须把历史轨迹记下来再喂给后续的轻量 GUI executor。
3. **成败轨迹用完就丢**。成功和失败的尝试在任务结束那一刻就被扔掉，下次跑类似任务又得重新打开 App、重新摸路径、重新踩同样的坑。
4. **GUI 流程很少合并"非视觉捷径"**。网页搜索、Android deeplink、系统 intent、可复用的预定动作序列……这些都能省一大段 GUI 步骤，但直接当持久 skill 又不安全——必须验证"此刻真的能跳到对的那一页"。

这四点单独看都不新鲜，但 KnowAct-GUIClaw 的贡献是把它们**统一收编到一个范式**里：**让 GUI 操控变成 personal assistant 的一个子智能体**（subagent），把"我以前怎么干的"和"我刚才是怎么干的"沉淀成结构化资产。

---

## 方法核心：Know–Route–Act–Reflect 四阶段闭环

整篇论文的骨架就是这张图（Figure 2）：

![Figure 2: KnowAct-GUIClaw 的四阶段执行循环和两个持久存储](https://arxiv.org/html/2607.12625v2/figures/know-route-act-reflect.png)

*Figure 2: 顶部的 Memory & History Store（Session Context / Agent Memory / History / GUI Policy）和 Skill & Shortcut Store（Agent Skills / GUI Shortcuts）给四个阶段持续供料。Know 阶段主动收集证据并组装推理上下文；Route 阶段做 App 候选排序并把任务拆成单 GUI 任务或多 App 工作流；Act 阶段在 GUI 原语、Skill、Deeplink/Intent、Ask-User 这四类动作空间里跑 Observe-Reason-Act；Reflect 阶段把轨迹蒸馏成可复用的技能和经验记忆。*

下面按四阶段展开讲。

### 1) Know：把"我应该这么干"的提示先拉齐再动手

在 GUI 行动之前，Know 阶段主动**主动取**相关证据——不是简单查 top-1，而是：

- 通过**语义相似度**检索先前的 GUI 经验和候选 Skill，结果只作"建议性"上下文，**不覆盖当前指令**；
- 把 **Policy Memory**（来自经验库里的"做事准则"）直接注入 prompt，不参与排序；
- Host（主代理）只把 GUI 子任务需要的子集上下文下发，**自己持有的会话上下文、Agent Memory、用户画像**按相关性主动召回；
- 当用户指令模糊时，host 主动用记忆**提议可被显式覆盖的默认假设**（比如"默认目的地 = 上次出差的城市"）。

这套设计的关键在"经验记忆"上不是简单的"以前跑过什么"文本，而是带**归因（attribution）**的——能告诉你"这个经验是因为哪个 App、哪次失败而留下来的"。Figure 3 用一个 Mastodon 创建邀请链接的案例做对比：

![Figure 3: 经验记忆在 GUI 任务开始前就把"该走哪条路"换掉了](https://arxiv.org/html/2607.12625v2/figures/gui_memory_case.png)

*Figure 3: 上半部分是没启用经验记忆的轨迹——GUIClaw 在 Mastodon 移动 App 设置里点了一圈，最终撞到不支持的入口失败（红色 X）。下半部分有经验记忆：一条"Mastodon 邀请链接需要 Web Admin Panel"的政策先被注入，agent 直接用浏览器跳到 web 管理界面，在 Step 6 成功生成邀请链接（绿色对勾）。注意，经验记忆是"建议性"的，每一步仍然靠实时屏幕证据。*

这是一个挺值得琢磨的细节：**经验记忆改的是"任务上下文"而非"动作选择"**。GUI executor 在每一步仍然以截图和 UI 树为准，记忆只是把"我之前走过这条死路，换一条"这件事提前说出来，避免重蹈覆辙。

### 2) Route：拆任务 + 黑板显式传值

Host 把用户请求拆成两类：

- **单 GUI 任务**：直接交给 executor；
- **多 App 工作流**：每个子任务是一个 goal-level 元组 $(g_i, h_i, I_i, O_i)$，$g_i$ 是 App 范围内的目标，$h_i$ 进一步限定 App，$I_i$ 是声明的**输入**（从黑板读），$O_i$ 是声明的**输出**（写回黑板）。

跨子任务的信息传递通过**短期黑板（blackboard）**完成，论文把流程抽象成：

$$G(g_i, h_i, B_{i-1}[I_i]) \rightarrow \tau_i, \quad E(\tau_i, O_i) \rightarrow B_i[O_i]$$

其中 $G$ 是 GUIClaw 在 POMDP 上执行子任务，$\tau_i$ 是轨迹证据，$E$ 是把证据映射到声明输出的提取器，$B_i$ 是第 $i$ 步的黑板。**任何声明的输入或输出缺失，整个工作流直接 fail closed**——这避免了"前一个子任务传了模糊描述，下一个子任务猜"的情况。

下图（Figure 4）画得很清楚：

![Figure 4: Route 阶段的黑板中介执行](https://arxiv.org/html/2607.12625v2/figures/blackboard-subgoal.png)

*Figure 4: 左侧的 Short-lived Blackboard B 存已知的 typed 输入输出（user.name, trip.from, trip.to, trip.date, flight.option 等），中间是顺序执行的 Ordered Subtasks，每个子任务先"Check Declared Inputs I_i"再"Append known values from B"，然后跑 GUIClaw 的 Observe-Reason-Act 循环，最终用 Output Extractor 把声明的输出 $O_i$ 写回 B。右侧是结构化工作流结果：Final State / Collected Outputs / Evidence / Status / Next Suggestions。*

这套设计的好处是**数据流是显式的、类型化的、不可伪造的**。你说 trip.date 必须是 2025-06-24，那它就是 2025-06-24，不会被某个 App 改成本地格式。Host 和 GUI 子代理都不必把所有上下文塞在 prompt 里——只需要查表。

### 3) Act：GUI / Skill / 捷径 / 求助四类动作空间

GUIClaw 执行的混合动作空间是：

$$\mathcal{A} = \mathcal{A}_{\text{gui}} \cup \mathcal{A}_{\text{skill}} \cup \mathcal{A}_{\text{shortcut}} \cup \mathcal{A}_{\text{ask}}$$

- $\mathcal{A}_{\text{gui}}$：点击、滑、滚、输文字、打开/关闭 App、等待这些"类人原语"；
- $\mathcal{A}_{\text{skill}}$：从历史轨迹蒸馏出的**带状态校验的可复用流程**，包含 ID、App 范围、描述、参数、可靠度计数和步骤序列；
- $\mathcal{A}_{\text{shortcut}}$：**页面校验过的** Android deeplink / intent，能省掉一长串导航；
- $\mathcal{A}_{\text{ask}}$：需要用户输入或授权的介入动作。

**关键工程细节是 Skill 的执行不是"按序列重放"**。每一步前都校验期望状态：能查的用确定性 state contract，查不到的就用视觉 valid-state check；对不上就启动有界恢复子目标、跳过可选障碍步骤，或者退回普通 GUI 执行。Shortcut 也一样——先验证"此刻真的能跳到对的那一页，参数没被吞"，否则不会触发。

这避免了"manifest 里声明了 SEARCH intent 就被无脑调用"这种粗糙做法。Android 的常量太多了（SEARCH、SEND、GET_CONTENT），没有页面级校验的 shortcut 库就是定时炸弹。

### 4) Reflect：把轨迹蒸馏成下一轮可用的资产

Reflection 阶段是**闭环的关键**。它把 GUI 任务结束后的轨迹做四件事：

1. **后置总结**：把"停在哪、还差什么"压缩成短记录，让被中断或阻塞的任务变成 host 可以接管的检查点，而不是从头再来。
2. **Skill 抽取**：用受限的 prompt（只允许声明式动作序列）让视觉 LLM 从轨迹里挖出可复用流程，标准化为"ID / App 范围 / 描述 / 参数 / 固定字段 / 状态契约"。**修复优先于新提取**——如果一条轨迹里发现了失败的重用 skill，in-place 更新它（窄化描述、加 guarded 可选障碍、刷新过期目标），而不是用别的 workflow 覆盖。
3. **Shortcut 校验**：把 manifest 里挖出来的候选 shortcut 当作"发现证据"而不是可信动作；实际跑一次、记录前台 App、ADB 输出、UI 树、截图，让验证器决定"是不是真的能用作 one-step skill"。
4. **经验记忆归纳**：仿照 ReasoningBank 的思路，把轨迹总结成"做法准则"，按 success / failure 两条 prompt 分别诱导。短轨迹、异常轨迹直接跳过，重复条目在 App 级别去重。

整条 Know→Route→Act→Reflect 形成一个**经验可迁移的闭环**：路由阶段用经验决定"拆成几个 App、用哪条路"；执行阶段用 Skill 缩短步骤；反思阶段把这次跑出来的新经验又塞回记忆库。

---

## 实验结果：在 MobileWorld 上打了个公开 SOTA，技能带来的提升其实没那么夸张

### 5.1 主表：MobileWorld GUI-Only 117 任务

论文的核心 benchmark 是 MobileWorld 的 117 个 GUI-Only 任务，用 50 步上限、原始确定性评估器、pass@1 单跑成功率。图 1 把所有外部模型按 leaderboard 分类排开：

![Figure 1: MobileWorld GUI-Only SR 对比](https://arxiv.org/html/2607.12625v2/figures/front.png)

*Figure 1: 灰条是专用 GUI 模型，彩色条是通用模型家族，紫色虚线显示"+13.1、+9.7、+16.2、+3.5、+5.9、+8.5"这些 KnowAct-GUIClaw 相对不同基线的提升幅度。最右侧的蓝色条 64.1% 是 KnowAct-GUIClaw + memory & skills（Kimi-K2.6），超过了 63.2 的 Seed-2.0-Pro、62.4 的 GPT-5.5、58.1 的 Gemini-3.1-Preview、56.4 的 Claude-Opus-4.7。*

把主表抽出来读一读：

| 模型 / 配置 | SR (pass@1, %) |
|---|---|
| Seed-2.0-Pro | 63.2 |
| **GPT-5.5** | **62.4** |
| Gemini-3.1-Pro-Preview | 58.1 |
| Claude-Opus-4.7 | 56.4 |
| Kimi-K2.6 | 55.6 |
| Kimi-K2.5 | 49.6 |
| Claude-Sonnet-4.5 | 47.8 |
| Qwen3.5-397B-A17B | 42.7 |
| GUI-Owl-1.5-32B（专用 GUI 模型） | 43.9 |
| MAI-UI-235B-A22B | 39.7 |
| ForgeOwl-8B | 41.0 |
| UI-Venus-72B | 16.4 |
| **KnowAct-GUIClaw（Qwen3.5-35B + host & mem）** | **34.5** |
| **KnowAct-GUIClaw（Qwen3.5-35B + host & mem & skills）** | **37.9** |
| **KnowAct-GUIClaw（Qwen3.5-35B + host + Kimi-derived mem & skills）** | **41.0** |
| **KnowAct-GUIClaw（Qwen3.5-397B host acts directly）** | **46.2** |
| **KnowAct-GUIClaw（Kimi-K2.6 + host & mem）** | **61.5** |
| **KnowAct-GUIClaw（Kimi-K2.6 + host, mem & skills）** | **64.1** |

**几个值得说一说的观察**：

- **"超过闭源模型"这个说法要拆开看**。在公开 leaderboard 上 Kimi-K2.6 跑 55.6%，加上 host + memory + skills 后到 64.1%，确实超过了 Seed-2.0-Pro（63.2）和 GPT-5.5（62.4）。但领先幅度只有 0.9-1.7 个百分点，这种量级在统计上其实挺容易被一次重新评测翻盘——别忘了 MobileWorld 用了 50 步上限和确定性评估器。
- **跨模型迁移是真东西**。Qwen3.5-35B 自己跑只有 24.8%，加上 host & memory 升到 34.5%，再加上 skills 升到 37.9%；但**用 Kimi-K2.6 的轨迹蒸馏出来的经验 + 技能喂给同一个 35B executor，能拉到 41.0%**——比标准 35B host-memory-skills 配置（37.9）还高 3.1 个点。这是整篇论文最漂亮的结果之一。
- **专用 GUI 模型在 MobileWorld 上打不过通用大模型 + 框架**。GUI-Owl-1.5-32B 这种 43.9% 的"专门做 GUI 任务"模型，比 Kimi-K2.6 + KnowAct-GUIClaw（61.5）差了 17.6 个点。这一点我觉得值得深思——GUI 任务到了一定复杂度后，**通用推理 + 框架编排**可能比"专门在 GUI 数据上 fine-tune"更有上限。

### 5.2 消融：host / memory / skills 的边际收益

表 2 给了系统级消融（A-F 六组），拆出 host + memory 和 skills 的贡献：

| 配置 | 设置 | SR (%) | GUI 步数 | GUI 任务数 | Total | Host Total |
|---|---|---|---|---|---|---|
| A | 35B executor 单独跑 | 24.8 | 26.7 | 1.0 | 281,266 | – |
| B | + host & mem | 34.5 | 26.8 | 2.3 | 279,211 | 65,224 |
| C | + skills | 37.9 | 25.1 | 2.5 | 278,289 | 63,792 |
| D | 397B executor 单独跑 | 40.7 | 26.1 | 1.0 | 254,459 | – |
| E | + host & mem | 43.3 | 26.8 | 1.4 | 273,352 | 10,096 |
| F | + skills | **46.2** | **23.7** | 1.7 | **260,516** | 10,982 |

**A → B**：host + memory 给 35B executor 带来 +9.7 个点（24.8→34.5），给 397B 带来 +2.6 个点（40.7→43.3）。**Total token 几乎不动**，说明 host 不靠堆截图，赢在"任务组织得更好"。

**B → C / E → F**：skill 进一步抬 SR（B→C: +3.4, E→F: +2.9），还**减少了 GUI 步数和总 token**（E→F：26.8 步降到 23.7 步，254k token 降到 260k token 之间虽然还小涨，但单步成本明显下来）。**SR 涨的同时 token 不涨甚至降**，这是少见的"省力又能干"组合。

**F 配置的细节很关键**。397B host 不是把每个子任务都丢给 35B executor，而是**自己直接处理"够格的"信息查询子任务**（比如用 web 搜索、查邮件摘要），只把真正需要 GUI 操作的子任务交给 executor。结果是 GUI 任务数从 2.5 降到 1.7，host total token 也从 63k 降到 11k——**host 直接处理反而比全部委派更省钱、更准**。

论文也单独算了一笔账给真正用到 skill 的子集（Table 3，35B executor 用了 skill 的 83 个任务，397B host 用了 skill 的 87 个任务）：

| 指标 | 35B + skills | 35B no skills | Δ | 397B + skills | 397B no skills | Δ |
|---|---|---|---|---|---|---|
| GUI 步数 / 任务 | 25.7 | 29.0 | -3.3 | 22.3 | 25.6 | -3.3 |
| 总 token / 任务 | 284,279 | 303,014 | -6.2% | 242,930 | 258,084 | -5.9% |
| 单跑 SR (%) | 40.6 | 35.7 | +4.9 | 50.2 | 48.3 | +1.9 |
| pass@3 (%) | 54.2 | 53.0 | +1.2 | 63.2 | 63.2 | 0.0 |

在"确实用 skill 的子集"上，技能带来的是**稳定的 3 步缩短 + 6% token 节约 + 1.9~4.9 个点 SR 提升**。这部分的提升幅度才是真材实料。

### 5.3 AndroidDaily：跨 App 分析任务上拉开更大差距

AndroidDaily 是按任务类型、复杂度、歧义分组的端到端基准。论文用 iOS 设备跑，按 resolved（去掉 41 个不可用任务评 194 个）和 all（全部 235 个，未可用直接判 0）两种口径打分：

| 模型 / 配置 | Filter | Query | Analyze | Atomic | Comp. | Cond. | Low | Mid | High | Total |
|---|---|---|---|---|---|---|---|---|---|---|
| UI-TARS-1.5 | 57.64 | 65.97 | 36.71 | 61.41 | 13.64 | 60.38 | 57.05 | 54.90 | 57.89 | 56.64 |
| Step-GUI-4B | 44.77 | 64.29 | 33.72 | 54.03 | 19.61 | 42.86 | 51.21 | 38.32 | 59.52 | 49.06 |
| Step-GUI-8B | 52.50 | 63.82 | 32.95 | 59.09 | 14.00 | 42.86 | 54.08 | 44.55 | 61.54 | 52.50 |
| **Ours (Resolved)** | **80.56** | **76.88** | **77.08** | **81.25** | **62.50** | **75.00** | **78.89** | **77.50** | **78.95** | **78.61** |
| **Ours (All)** | 68.40 | 66.13 | 51.39 | 70.98 | 41.67 | 53.23 | 64.94 | 65.96 | 62.50 | 64.89 |

最大的优势在 **Analyze** 和 **Comp.**（复杂）任务——前者 77.08% 对比 UI-TARS-1.5 的 36.71%，后者 62.50% 对比 13.64%。这正是 KnowAct-GUIClaw 显式子任务分解 + 黑板信息传递 + 经验检索能大放异彩的地方。简单 atomic 任务上它也有 +20 个点左右的优势，但绝对值差距没那么离谱。

论文自己也承认 resolved 和 all 的差异反映了**App 可用性和环境不匹配**——这跟策略质量没关系，更像是测试环境问题。所以引用时我建议优先看 resolved（78.61%）这个数字。

### 5.4 跨平台：HarmonyOS 76.2%，Windows 70.0%

为了证明框架不是 Android 专属，作者在 HarmonyOS 和 Windows 上各跑了一组：

- **HarmonyOS 63 个 MobileWorld-派生任务**（48/63 通过，76.2%）
- **Windows 30 个手工设计的桌面任务**（21/30 通过，70.0%）

这是 Framework 的**真正卖点**之一：GUI 操控的具体原语（点击/滑动/输入）可能因平台而异，但 Know-Route-Act-Reflect 的骨架、记忆接口、技能校验机制都是平台无关的。换基座比换框架容易。

### 5.5 案例研究：四种典型场景

论文给了四组真实截图案例，配合 host 卡片、步骤标签、每步思考和黑板 pill 的 overlay 解释（这些图太大上传前已被压缩）：

- **路由 + 信息传递**：Email-to-Clock 工作流把"明晚 8 点的派对"时间通过黑板传到 Clock 任务里。
- **经验记忆纠错**：Figure 3 的 Mastodon 案例（图见上）——有记忆时切到 web 管理面板，没有时死磕移动 App 设置。
- **导航压缩 + 捷径**：JD 搜索捷径一步到结果页，淘宝分支需要 5 步普通 GUI 步骤。SMS 捷径能直接打开带"收件人 + 正文"预填的短信撰写页。
- **host 恢复 + 直接参与**：Figure 6(b) 失败后 host 重新规划、用简历文件恢复联系人电话；Figure 7 的会议地点任务里 host 直接用 web 搜索补全地址，省掉 GUI 里"翻邮件找街道号"那一段。

每个案例都强调**边界是 typed 的**——黑板里传的是结构化值，不是自由文本摘要。

### 5.6 跨平台失败模式：暴露的也是基座模型的局限

作者没藏失败——HarmonyOS 上闹钟分钟滚轮选不到 25 分钟（惯性滚轮 picker 难精准控制），淘宝把"立即支付"按钮误判为"已经在支付页"导致提前停手；Windows 上 WeChat 案例发对了文字但选错了表情图标（自然语言情绪标签→视觉相似图标的映射出错），通知 toast 的"瞬时观察 + 立即点击"做不稳。

这些失败模式其实不全是"框架"的锅——很多是**基座视觉模型的语义接地（semantic grounding）能力**问题。框架能保证流程不出错，但"在表情面板里找 laughing 表情"这种细粒度视觉语义，还是得靠更强的多模态基座。

---

## 我的判断

### 亮点

1. **范式统一**。把"记忆 + 技能 + 多级规划 + POMDP 化 GUI"这几条分散的线，**用一个清晰的 Know-Route-Act-Reflect 节奏串起来**，并用 Memory & History Store + Skill & Shortcut Store 两个长期存储把闭环扣死。读下来像一份能直接照着落地的工程蓝图。
2. **黑板 + typed 契约**。跨 App 工作流里用 typed tuple + 短期黑板传值，是把"自由文本摘要会丢"这件事**工程化**地解决了。`fail closed` 这个规则很重要——比"尽力猜"靠谱太多。
3. **Skill 的状态校验机制**。技能不是"动作重放"，每步前都要 state contract 或 visual check。对不上要么恢复、要么跳过、要么回退普通 GUI 执行。这避免了 skill 库越长越容易"撞上下文过期"的常见反模式。
4. **跨模型迁移的实证**。把 Kimi-K2.6 的经验丢给 35B Qwen executor 能拉到 41.0%，跨基座"经验可迁移"这件事**真有数据支撑**——很多类似工作只说"理论上可迁移"就完事了。
5. **跨平台验证**。Android + iOS + HarmonyOS + Windows 四平台都跑过，Framework 不绑死平台是 mobile GUI 智能体走向工程化的必要条件。

### 我得提的问题

1. **"超过闭源模型"的对比口径需要细看**。64.1% vs 63.2% 的 Seed-2.0-Pro 只领先 0.9 个点；57% 这种量级在 MobileWorld 的 50 步上限 + 确定性评估器上，**一次重新评测可能就翻盘**。论文里说"beat all agential frameworks and closed-source agential models"听上去很猛，但表 1 的差距比"碾压"温和多了。
2. **Skills 在 35B 配置上的提升只有 +3.4 个点**（34.5→37.9），比我预想的小。论文强调 skills 在 "skill-using 子集" 上能 +4.9 个点（35B executor 子集），听起来好，但**这是用 skill 的子集里的提升，不是全局**。全局 +3.4 是更诚实的数字。
3. **AndroidDaily 上的"跨模型迁移证据"其实只做了一个方向**——Kimi → Qwen。论文明确说"establishes transferability in the tested Kimi-to-Qwen direction without assuming universal model-independent transfer"。这是诚实的提法，但工程上谁都想问一句：**反过来 Qwen→Kimi 呢？GPT-5.5→任何开源模型呢？**论文没给数据。
4. **跨平台任务数偏少**。HarmonyOS 63 个 + Windows 30 个，跟 MobileWorld 的 117 个不在一个量级。70% / 76% 的成功率虽然亮眼，但**评测集太小**——我比较关心这 63 个 / 30 个是怎么选出来的，是否覆盖了 HarmonyOS 特有的控件（带滚轮的 picker、右侧下滑通知等已经在失败模式里被提到的）。
5. **重复尝试上界 +3 个百分点，pass@3 all-3 的提升更小**。A→F 的 single-run SR 从 24.8% 涨到 46.2%（+21.4），但 pass@3 any-of-3 从 34.2% 到 59.8%（+25.6），pass@3 all-3 从 15.4% 到 32.5%（+17.1）。**all-3 提升比 any-of-3 小**，说明技能 + host 提升里有不少是"减小方差"的成分，真正稳定的提升没单一指标看上去那么夸张。
6. **GUI 子任务委派的 host 过滤条件没公开**。E→F 的"host 直接处理够格的子任务"是关键设计，但论文没给"够格"的判断阈值。在工程落地时，这一刀切不准就会两头不讨好——要么 host 抢活太多（GUI 模型浪费）、要么 host 放活太松（依然低效）。

### 工程启发

如果你也在做 GUI 智能体（无论是 web、移动端还是桌面），KnowAct-GUIClaw 给的几个套路值得借鉴：

- **给长期记忆加 attribution 标签**。只存"上次是怎么做的"不够，要存"上次为什么这么做、在哪个 App、因为什么失败"——attribution 让"经验"变成"可调用的政策"，而不是"翻不动的日记本"。
- **跨 App 工作流用 typed blackboard 传值**。比"上一个 agent 用自然语言写一段总结给下一个 agent"靠谱一个数量级。
- **Skill 必须带状态校验**。不带 state contract 的 skill 库，最终都会因为"界面改了"全失效。
- **Host 不是吉祥物**。它应该有权**拒绝委派**、**直接调用非 GUI 工具**、**接管失败子任务**——全委派 vs 全不委派是两个极端，配置得当的 host 应该能省下 60%+ 的 host token。
- **跨基座迁移是 ROI 最高的方向**。与其把所有精力都花在"训更强的专用 GUI 模型"上，不如**先做好经验 + 技能的可移植性**——一次蒸馏，多次复用，比单点提升 5 个点的边际收益高。

---

## 写在末尾

KnowAct-GUIClaw 让我最喜欢的一点是它**没有发明新概念**——POMDP、黑板、记忆、Skill、Reflection、跨 App 工作流，每一项都已经在相关领域被反复验证。它做的事情，是把这些东西**按 Know-Route-Act-Reflect 的节奏对齐到一个移动 GUI 智能体的具体实现里**，并把"经验可跨基座迁移"用数据证实。

但如果你期待它"彻底改变 GUI 智能体"，可能会失望。**它的创新是工程整合层面的，不是底层突破层面的**。在 SOTA 的数字上虽然过了几个闭源模型，但优势非常薄，更像是"在开源 SOTA 上站住了脚"，而非"打破了某个天花板"。

真正让我眼前一亮的，是它在 HarmonyOS / Windows 上的 76% / 70% 通过率——这种"框架不绑平台"的能力，比单一基准上的 0.9 个点领先要更有长线价值。

如果你正在选型 GUI 智能体的执行框架，KnowAct-GUIClaw 值得你花两个小时读一遍原文。它的复现日志（GitHub Releases 的 Result 标签页）是公开的，跑一遍同款 benchmark 就能验证文中所有数字。我个人比较关心的是未来工作里承诺的"统一联合规划器"——把 Knowledge 检索、工具调用、GUI 动作**放进一个决策循环**而不是分四个阶段串行做，如果真做出来，会比现在这种"接力棒"式的编排再高一档。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
