---
title: "行为图谱让 Agent Harness 可读可改：Harness Handbook 是怎么把"改哪段代码"这件事画清楚的"
date: 2026-07-17
arxiv: 2607.13285
authors: "Ruhan Wang, Yucheng Shi, Zongxia Li, Zhongzhi Li, Yue Yu, Junyao Yang, Kishan Panaganti, Haitao Mi, Dongruo Zhou, Leoweiliang"
source: arXiv
---

## 核心摘要

我读完这篇论文的第一反应是：终于有人把"想改 harness 但不知道从哪下手"这件事掰开揉碎画出来了。论文的核心贡献是一个叫 **Harness Handbook** 的"行为中心表示"——通过静态分析 + LLM 辅助结构化，从代码库里自动合成一份"行为图谱"，把每个行为显式链接到对应源码；外加一个 **BGPD（行为引导渐进式披露）**算法，让 planning agent 从高层行为一路定位到具体修改位置。

效果相当能打：在两个开源 harness（Terminus-2 和 Codex）共 60 个修改请求上，Handbook-Assisted 规划的整体胜率分别比纯 baseline 高出 18.9 和 10.0 个百分点，文件级 F1 在 Terminus-2 上摸到 89.3%，**完全未命中（Wrong）** 指标最高下降 25.9 个点，与此同时 planner 用的 token 还少了 8.6%–12.7%。增益最大的是"散布实现点""很少执行的路径""跨模块交互"这三种老大难问题。

坦率讲，**它不是又一个"AI 写代码更准"的论文**，而是把"改哪段代码"这个上游瓶颈单独拎出来打。**对正在维护或开发生产级 agent 的人，这篇是真值得读的**。

---

## 论文信息

- **标题**：Harness Handbook: Making Evolving Agent Harnesses Readable, Navigable, and Editable
- **作者**：Ruhan Wang, Yucheng Shi, Zongxia Li, Zhongzhi Li, Yue Yu, Junyao Yang, Kishan Panaganti, Haitao Mi, Dongruo Zhou, Leoweiliang
- **arXiv**：[2607.13285](https://arxiv.org/abs/2607.13285)
- **提交日期**：2026-07-14（v1）
- **页数 / 图数**：29 页 / 6 figures
- **项目页**：[ruhan-wang.github.io/Harness-Handbook](https://ruhan-wang.github.io/Harness-Handbook/)

---

## 痛点：你以为在改 bug，其实在做考古

你有没有这种经历：接到一个"小需求"——比如让 agent 在多轮工具调用之间持久化一个状态，或者在某个边界条件下多走一条分支——你信心满满打开 IDE 准备改，结果花了两天在几千行代码里翻来翻去，不是找不到位置，就是改了一处发现另一处偷偷依赖了原来的行为。

这不是你菜，是 **production 级 harness 天生如此**：动辄上百个函数、十几个 stage、执行逻辑分散在多个模块、通过共享状态连在一起。一个"行为"——比如"工具调用失败时重试"——可能同时牵涉初始化、规划、错误处理、状态序列化、prompt 模板五个地方。

更麻烦的是：现代 AI agent 的能力不只看 foundation model，更看 harness。Anthropic 之前就反复强调过——"真正决定 AI 编程工具能不能在大规模代码库里打胜仗的，不是模型本身，而是围绕模型搭建的那套马具（harness）"。换 harness 的回报比换模型更稳定。

但 harness 难改啊。当模型升级了、API 改了、新工具加了、用户行为变了，harness 必须跟着改。问题是：**修改请求描述的是「系统应该做什么」**（行为层语言），**仓库组织的是文件、函数、模块**（实现层语言）。这两套语言之间存在语义鸿沟，而这座桥，现在主要靠人脑自己建。

**Behavior localization**（行为定位）——找到实现某个行为的所有代码位置——就成了 harness 演进的核心瓶颈。

---

## 现有方法为什么没解掉这个瓶颈

论文把现有工作归了几类：

- **Repository maps / code search / repository memory / long-context editing**：让代码库更易探索，但它们**说到底还是 implementation-centric**——围绕文件、函数、模块组织信息。
- **Code summarization / 自然语言 artifacts**：把代码翻译成更可读的描述，但仍然围绕"代码长什么样"组织。

这些方法可以识别"个别相关代码片段"，但回答不了两个问题：

1. 这些片段怎么协同产生一个行为？
2. 我是不是漏了某些不在主路径上的位置（比如 fallback 分支、冷路径、镜像实现）？

一句话总结：**它们让你更容易看到代码，但没让你更容易看到行为**。

而 coding agent 在这个问题上还多一层枷锁：受限于上下文窗口，没办法一次性把所有相关代码都塞进 prompt，必须迭代探索仓库，然后……就可能漏掉 scattered 或者 rarely executed 的路径。

---

## Harness Handbook：把"行为-代码"的桥画出来

论文的核心创意是一个**以行为为中心（behavior-centric）的表示**——Harness Handbook。它不再围绕"代码在哪个文件"组织知识，而是围绕"系统能做什么行为"组织知识，然后把每个行为链接到实现它的源码。

### L1–L3 文档树

Handbook 由两级结构组成。第一级是 **L1–L3 文档树** $\mathcal{D}$，三级层次渐进披露：

| 层级 | 名称 | 内容 | 信息密度 |
|------|------|------|----------|
| **L1** | System Overview | 架构、执行模型、主要 stage、全局数据流 | 浅（高层概览） |
| **L2** | Component Overview | 选定 stage 的职责、输入输出、依赖、局部状态 | 中（组件细节） |
| **L3** | Unit Deep Dive | 链接到源码的实现条目，每个有 statically identified source location | 深（实现细节） |

L3 条目按 `leaf mode` 决定粒度：

- **function-as-leaf**：L3 覆盖整个函数或连续区域，适合有可信 seed skeleton 且函数级别组织可控的 harness（比如 Terminus-2，6 个文件、103 个内部函数）。
- **file-as-leaf**：L3 覆盖一个文件，适合没有 seed skeleton 或函数级别组织超预算的大仓库（比如 Codex，2,267 个文件、34,363 个内部函数）。

两个 L3 条目体只在需要时披露，限制不必要的上下文——这就是"渐进式披露"的字面意思。

### 状态寄存器视图 $\mathcal{Z}$

第二级是 **状态寄存器视图（State Register View）** $\mathcal{Z}$，记录跨 stage 边界的状态关系。这个东西看着不起眼，但其实非常关键——它专门捕捉"一个 stage 写入的值在数个 stage 后被消费"这种结构上距离远但相互依赖的关系。**这种关系，是自顶向下代码阅读最容易漏掉的**。

![图1：Harness Handbook 结构与渐进披露](https://arxiv.org/html/2607.13285v1/x1.png)

*图1：左边是 Handbook 的目录结构示例（按 1.System Overview → 4.Components 组织的 L1–L3 文档树，展示了 6 个 stage、Main Loop 的 9 项属性、6 项收益）；右边是 L1–L2–L3 三层信息密度的渐进披露。*

---

## 怎么自动造出这本 Handbook

Handbook 不是手写的，是从代码库**自动合成**的。整个构造流水线分三个阶段：

### Phase I：静态事实抽取（无 LLM 调用，纯确定性）

语言特定的适配器解析仓库，提取函数、命名外部边界、源码位置、签名和调用边，构建 program graph $\mathcal{G}$。**只保留解析到内部函数或命名边界的调用**——未解析的调用写进 audit log，不去猜目标。

这一步的好处是**确定性强**，不会被 LLM 幻觉带偏；坏处是 dynamic language 解析可能漏调用，但作者宁漏不猜，**让不确定的部分被显式记录，而不是悄悄假设**。

### Phase II：行为组织

这是 LLM 出场的阶段，但有两种组织变体存放在 $\Theta$ 中：

- **function-as-leaf 分支**：从 seed skeleton $\mathcal{S}_0$ 开始，对每个可分析的内部函数，proposer 用源码、caller/callee 上下文和 stage 描述提议 function-to-stage 分配；reviewer 通过迭代 review 精炼分配（可添加/删除/合并/拆分 stage）；收敛条件是 stage skeleton 和所有分配与前一轮匹配且无函数等待重新考虑。
- **file-as-leaf 分支**：为每个文件构建 file card，从 card 描述推断 stage skeleton，把每个文件分配到一个 primary stage（跨切面文件可命名 1-2 个 secondary stage），可选的 proposal-and-review 精炼。

三种组织变体：**`oneshot`（默认）**、**`doctor`（草拟 + 反复 review）**、**`agent`（agent 生成草拟 + 同样的 loop）**。这其实是在"成本 vs 质量"之间给用户三档选择。

### Phase III：层次合成与打包

把 stage skeleton 和源码组织转换成 L1–L3 文档树和跨 stage 状态寄存器视图。**每个 L3 条目链接到静态识别的源码位置，并针对当前仓库验证**——能解析到当前源码的条目是"活跃"的，无法重新验证的条目会被 **frozen** 并排除在定位之外。

这条规则非常务实：Handbook 不是建好就永远对的，它会**随仓库演变而失效**，但**用一种保守、可审计的方式失效**——而不是悄悄骗你。

![图2：Handbook 构造流水线](https://arxiv.org/html/2607.13285v1/x2.png)

*图2：Handbook 构造的三个阶段。Phase I 用静态分析从源码仓库抽取函数、调用图、状态读写等事实，输出 Program Graph；Phase II 通过 Propose-Review 循环把函数/文件映射到执行 stage（Init → Setup → Loop → Tool Call → Parse → Finish），输出 Behavioral Mapping；Phase III 合成 Tier 1 Overview + Tier 2 Stage Cards + Tier 3 Unit Cards，输出最终 Handbook。*

---

## BGPD：让 agent 沿行为图谱定位源码

Handbook 本身只是一份文档。要让它在 planning 时真正起作用，需要一个导航算法——**Behavior-Guided Progressive Disclosure**（BGPD）。

BGPD 是个四步 coarse-to-fine 流程：

### Step 1：Stage Selection
从 L1 系统概览和 L2 stage 索引开始，agent 选最匹配修改请求 $q$ 的 stage，然后**跟随状态寄存器视图 $\mathcal{Z}$ 添加通过共享状态耦合的 stage**——这一步是**关键创新**，专门捕捉远距离 stage 间的隐式依赖。

### Step 2：Entry Selection
在选定的 stage 内打开对应 stage 页，选最相关的 L3 条目。每个条目暴露摘要和源码 locator，**条目体仅在需要时才披露**。

### Step 3：Call-Relation Expansion
沿 program graph $\mathcal{G}$ 中的调用关系扩展候选集。function-as-leaf 模式跟随 function-call graph，file-as-leaf 模式跟随 induced file-call graph。**命名的外部边界节点提供上下文但永远不会作为编辑位置返回**——避免把"调用了外部 API"误判为"需要改的地方"。

### Step 4：Source Verification
前三步都在 handbook 上操作。**最后一步打开当前仓库 $\mathcal{R}$，解析每个候选 locator，只保留对请求 $q$ 仍然相关的位置**，产出验证后的证据集 $\hat{\mathcal{E}}_q$。每条记录包含 file path、可选的 function/region anchor、当前源码摘录。

> "The handbook guides the search, while the repository remains the basis for the edit plan."

——这句话是 BGPD 的哲学：**Handbook 是导航图，但仓库是事实来源**。它不让你跳过"看真实代码"那一步，只是让你**知道往哪看**。

Handbook 通过一个 **SKILL.md manifest** 暴露给 planner：

- 名称：`<harness>-handbook`
- 描述：结构化 map，用于查找变更必须触及的每个代码位置
- 引用文件：`overview.md` / `index.md` / `registers.md` / `stages/<id>.md`
- 使用指南：6 步导航流程，要求 plan 必须覆盖 handbook 暴露的每个位置

这个 manifest 的设计很工程化——**Handbook 不是个内嵌在某个 agent 框架里的私货，而是个独立的 skill 文件**。任何支持 SKILL.md 协议的 agent 都能消费它。

---

## 完整修改工作流：BGPD → Plan → Execute → Resync

把 BGPD 嵌入端到端修改工作流，论文给出了 Algorithm 1，四步走：

| 步骤 | 输入 | 输出 | 关键 |
|------|------|------|------|
| **1. BGPD 定位** | $q, \mathcal{H}, \mathcal{R}$ | 验证后证据集 $\hat{\mathcal{E}}_q$ | 只读 |
| **2. Edit Planning** | $q, \hat{\mathcal{E}}_q$ | edit plan $\mathcal{P}$ + action declarations $\Gamma$ | 三类动作：modify / add / remove（rename = remove + add）|
| **3. Execution** | $\mathcal{R}, \mathcal{P}$ | 新仓库 $\mathcal{R}'$ + diff $\Delta$ | executor 不能列目录、不能访问 shell；先 verbatim 替换，失败才读源码 |
| **4. Handbook Resync** | $\mathcal{H}, \mathcal{R}, \mathcal{R}', \Delta, \Gamma$ | 更新后 $\mathcal{H}'$ | 每个非空 diff 触发；模型调用限于 4 个语义步骤，其余都是确定性的 |

**Resync 是这套系统的隐藏宝石**。它解决了一个所有"代码索引"类工具的通病：建索引那是一瞬间的事，但仓库每天在变。

Resync 怎么搞：

- **Version alignment**：重新解析源码、刷新 program graph，对齐新旧版本。function-as-leaf 用不依赖行号的 body fingerprints 匹配函数（避免有人加一行 import 就把整张表搅乱）；file-as-leaf 用 file-set 差异和 content hashes。
- **Scoped update**：stage skeleton $\mathcal{S}$ 仍然有效就只更新受影响部分；否则用相同 leaf mode 重跑 Phase II–III。
- **Conservative handling**：无法解析或分类的内容被 frozen 或记录在 coverage record 中，**而非猜测**。

坦率讲，这种"诚实承认失败"的处理方式比大多数工具的"尽力猜"强太多。

---

## 实验设计：60 个修改请求、3 个 judge、2 个 harness

### 评估对象

| 特性 | **Terminus-2** | **Codex** |
|------|---------------|----------|
| 来源 | Harbor 框架的 Python terminal agent | Codex coding agent 的 Rust monorepo |
| 语言 | Python | Rust |
| Leaf mode | function-as-leaf | file-as-leaf |
| 源文件数 | 6 | 2,267 |
| 内部函数节点 | 103 | 34,363 |
| 边界节点 | 4 | 14,016 |
| 解析调用边 | 257 | 159,960 |
| Stages (L2) | 20 | 140 |
| L3 条目 | 10 | 6,227 |
| State registers | 10 | 62 |

Codex 在规模和复杂度上都比 Terminus-2 大两个数量级——所以 Codex 的提升更难得。

### 修改请求

每个 harness 贡献 **30 个 behavior-driven 修改请求**（共 60 个），按三种类型分布：

- **Query**（Q）：修改现有行为，不揭示目标位置
- **Cross-file**（CF）：添加跨文件/模块的端到端能力
- **Search-Hostile**（SH）：把相关实现放在关键词搜索难以恢复的位置（fallback 分支、镜像实现、冷路径等）

按定位难度标注 **Easy / Medium / Hard**。

### 评估臂

- **Baseline**：planner 用只读工具直接探索仓库（file read、in-file search、directory listing）
- **Handbook-Assisted**：planner 额外获得 handbook 作为可导航 skill，遵循 BGPD 策略

两个臂在请求、模型、仓库、工具权限和解码设置上**完全相同**。Planner 由 **DeepSeek-V4-Pro** 驱动，基于 **NexAU** 构建。

### 评测指标

**Plan quality** 用三个独立 judge（GPT-5.5、Opus 4.8、DeepSeek-V4-Pro）在三个维度评分：

| 维度 | 权重 | 含义 |
|------|------|------|
| Localization | 0.5 | 编辑位置是否准确 |
| Scope Control | 0.25 | 计划是否保持聚焦 |
| Reasoning | 0.25 | 理由和支撑证据是否充分 |

加权得分 $S = 0.5 \times S_{Loc} + 0.25 \times S_{Scope} + 0.25 \times S_{Reason}$（0–100 分制），分差 $\geq \delta = 3$ 判定为胜。

**Localization accuracy** 与 Opus 4.8 和 GPT-5.5 的独立参考计划比较，报告 **Recall / Precision / F1 / Wrong**（Wrong = 零重合率，越低越好）。

**Planning cost** = 每个请求的平均 planner tokens。

---

## 主实验结果：胜率全面碾压，token 反而更省

### 整体胜率

| Harness | Baseline | Handbook-Assisted | 增益 |
|---------|----------|-------------------|------|
| Codex | 28.3% | 38.3% | **+10.0 pp** |
| Terminus-2 | 26.7% | 45.6% | **+18.9 pp** |

三个 judge 在 Codex 上增益都是 10.0 pp，方向高度一致。Terminus-2 增益范围 13.3–26.7 pp（看哪个 judge），最大的 GPT-5.5 给出 26.7 pp。

**Token 使用反而更少**：

| Harness | Baseline | Handbook-Assisted | 降幅 |
|---------|----------|-------------------|------|
| Codex | 0.102 M / 请求 | 0.089 M / 请求 | **−12.7** 个点 |
| Terminus-2 | 0.058 M / 请求 | 0.053 M / 请求 | **−8.6** 个点 |

这事儿值得多看一眼——**质量提升不依赖更大的 token 预算，反而更省**。原因是 BGPD 把 agent 直接带到正确位置，省了 baseline 那种"撒网式 grep 失败 → 重新读大文件 → 再 grep"的反复探索。

![图3：整体胜率、Per-Judge 胜率、Token 成本](https://arxiv.org/html/2607.13285v1/x3.png)

*图3：(a) Codex 和 Terminus 上整体胜率分别提升 10.0 和 18.9 pp；(b) Per-Judge 胜率，三个 judge 在两个 harness 上都偏向 Handbook-Assisted；(c) Token 成本分别下降 12.7% 和 8.6%。*

---

## 定位质量：F1 提升 5–18.8 个点，Wrong 大幅下降

这是我觉得最硬核的表——和独立参考计划直接比定位准确率。

### Codex vs Opus 4.8 参考

| 粒度 | 指标 | Baseline | Handbook | Gap |
|------|------|----------|----------|-----|
| File | F1 | 46.6 | 61.8 | **+15.2** |
| File | Wrong ↓ | 37.0 | 14.8 | **−22.2** |
| Symbol | F1 | 38.3 | 57.1 | **+18.8** |
| Symbol | Wrong ↓ | 44.4 | 18.5 | **−25.9** |

### Terminus-2 vs GPT-5.5 参考

| 粒度 | 指标 | Baseline | Handbook | Gap |
|------|------|----------|----------|-----|
| File | F1 | 76.5 | 89.3 | **+12.8** |
| File | Wrong ↓ | 20.0 | 6.7 | **−13.3** |
| Symbol | F1 | 73.0 | 89.3 | **+16.3** |
| Symbol | Precision | 73.9 | 93.3 | **+19.4** |

F1 增益范围 **5.0 到 18.8 个点**，**24 个 Recall/Precision/F1 比较中 Handbook 全部胜出**。Wrong（完全未命中率）从不增加，最大降幅 25.9 个点。

我觉得 **Wrong 这个指标其实比 F1 更能讲故事**——它衡量的是"这次修改完全改错了位置"的概率。F1 提升可能来自"找对了 70%"，但 Wrong 下降说明"完全找错"这种情况被显著压住了。在生产环境里，**完全改错位置比改不全的代价大得多**——前者直接发布事故，后者最多是个 bug。

---

## 按维度细分：Localization 是主要受益者

按 Localization / Scope Control / Reasoning 三个维度看，三 judge 平均：

| 维度 | Terminus-2 增益 | Codex 增益 |
|------|----------------|------------|
| **Localization** | **+12.2 pp** | +2.2 pp |
| **Scope Control** | +6.7 pp | +1.1 pp |
| Reasoning | +4.5 pp | +3.3 pp |

**Localization 是主要受益维度**——这跟论文的核心定位完全一致：Handbook 解决的就是"在哪改"的问题。但 Scope Control 和 Reasoning 也有正增益，说明**更好的位置理解 → 更好的计划聚焦度 → 更有逻辑的论证**。这套正向链路是真实的。

![图4：按 Localization、Scope Control、Reasoning 三维度细分](https://arxiv.org/html/2607.13285v1/x4.png)

*图4：Codex（上排）和 Terminus（下排）在三个维度上的 per-judge 胜率。Localization 维度 Terminus 平均提升 12.2 pp，Codex 提升 2.2 pp。*

Codex 维度提升比 Terminus-2 小很多，**这其实合理**——Codex 仓库大 100 倍，本来 LLM 还能靠 long-context 硬读；Terminus-2 体量小、逻辑密度高，行为分布在少数 stage 里，Handbook 把这些 stage 显式画出来效果就立竿见影。

---

## 按请求类型和难度：老大难问题反而提升最大

### 按请求类型

| Harness | 类型 | 增益 |
|---------|------|------|
| Codex | Query (Q) | **+26.7 pp** |
| Codex | Cross-file (CF) | **+16.3 pp** |
| Codex | Search-Hostile (SH) | **+16.7 pp** |
| Terminus-2 | Query (Q) | **+3.3 pp** |
| Terminus-2 | Cross-file (CF) | **+20.0 pp** |
| Terminus-2 | Search-Hostile (SH) | **+33.3 pp** |

**6 个 harness-by-type 比较全部偏向 Handbook-Assisted**，增益范围 16.3–33.3 pp。

**Search-Hostile 在 Terminus-2 上提升 33.3 pp**——这是整篇论文最让我惊喜的数据。SH 请求的相关位置被故意放在关键词搜索难以恢复的地方（fallback 分支、镜像实现、冷路径），**正是 baseline 必然翻车的问题域**。Handbook 用状态寄存器和显式 L3 链接把这些位置拉到阳光下，**就赢了**。

### 按定位难度

按 Easy / Medium / Hard 标注，6 个比较全部正增益，范围 **3.7–33.3 pp**。**增益不随难度标注单调递增**——说明难度标注本身不能解释增益，Handbook 在不同难度上都有效。

![图5：按修改请求类型与定位难度细分](https://arxiv.org/html/2607.13285v1/x5.png)

*图5：(a) 在 Codex 上 Query 类型提升最大（26.7 pp），在 Terminus 上 Search-Hostile 提升最大（33.3 pp）；(b) 按难度标注分层，Codex 的 Easy、Terminus 的 Medium 增益最显著，整体 6 个比较全部为正。*

---

## 我的判断：真亮点 vs 还没说清的地方

### 亮点：把语义鸿沟具象化

我觉得这篇论文最值钱的地方是**把"行为到代码"这个一直被默认为"靠脑子想"的步骤，做成了一个可工程化、可评测的子任务**。一旦把它单独拎出来，就能：

1. 量化（Localization Recall / Precision / F1 / Wrong）
2. 优化（BGPD 算法、resync 机制）
3. 评估（独立 judge、per-type 切片）

这种**把模糊的人类认知成本变成可测量的机器问题**的思路，是工程化 AI 系统的核心方法论。

**另一个亮点是「诚实失败」**。Phase I 只保留解析到的调用，未解析的写进 audit log；Resync 拿不准的内容 frozen；Handbook 里无法重新验证的 L3 条目排除在定位之外。这种"宁漏不猜"的风格，让 Handbook 在生产环境里更可信赖。

### 几个我还想问的问题

第一，**seed skeleton 依赖**。function-as-leaf 模式需要 "a trustworthy seed skeleton that faithfully reflects the harness's execution stages is available"，当不可用时必须退回 file-as-leaf。对于一个完全陌生的 harness，**谁去提供这个 seed skeleton**？论文没展开。如果要靠 LLM 自己生成，那它就退化成了"agent 模式"，和 file-as-leaf 的区别又在哪里？

第二，**评估只到 plan，没到 execution**。论文自己声明"This experiment evaluates localization and edit planning in the modification workflow"——也就是说**没真正跑过端到端修改**。Handbook 帮 agent 选对了位置，但 agent 改出来的代码到底能不能 work、能不能不破坏现有行为，**这事儿论文没回答**。这其实是个挺大的缺口——选对位置只是必要条件，不是充分条件。

第三，**Handbook 构建成本没说**。Codex 那种 34,363 个内部函数、159,960 条调用边的仓库，Phase II 跑一遍要多少 token、要多少时间？能不能增量构建？论文里没看到。**如果构建成本远超后续收益，那这套系统就只能用在长期维护的高价值 harness 上，不能做一次性项目**。

第四，**只有两个开源 harness、60 个修改请求**。Codex 是 OpenAI 的 monorepo，Terminus-2 是 Harbor 框架里的 Python 终端 agent。**完全没有覆盖闭源、私有、领域特定的 harness**。这些 harness 是不是也能用同样方法做 Handbook、效果是不是一样好，目前是 open question。

第五，**planner 模型用 DeepSeek-V4-Pro，这是个较弱模型**。论文 RQ2 说"有了 handbook，较弱 planner 能匹配更强模型"，但 baseline 和 Handbook-Assisted 都用同一个 planner——**没法直接回答 RQ2**。要回答 RQ2 应该是 baseline 用 DeepSeek-V4-Pro vs Handbook-Assisted 用更弱模型（比如 DeepSeek-V4-Flash 之类），看 Handbook 能不能把弱模型拉到强模型水平。

### 它到底是工程整合还是底层突破？

我的判断：**90% 工程整合 + 10% 真亮点**。

真亮点是那个**「以行为为中心」**的表示视角——把组织知识的单位从文件/函数/模块换成行为，这件事之前没人系统性地做，更没人做出完整的工具链。

剩下 90% 是**成熟的工程整合**：静态分析（几十年技术）、LLM-assisted structuring（2024 后的标配）、SKILL.md manifest 暴露（已有协议）、eval suite with 3 judges（标准做法）、program graph 增量 resync（已知模式）。

但这套整合做得**相当扎实**——Phase I 完全确定性、Phase II 收敛条件清晰、Resync 限定模型调用范围在 4 个语义步骤、保守失败处理。**它不是"我又把已有东西拼了一遍"的凑数论文**——它把每一块都打磨到生产可用。

---

## 工程启发：谁该把这套东西用起来

如果满足下面任一条件，建议认真读一下这篇论文 + 项目页：

- 你在维护一个 production 级 agent harness（数千行起步、多 stage 协作），并且**迭代频率高**（每月几次修改）。
- 你在用 coding agent 改一个**你不完全懂**的大仓库，每次修改都要花大量时间"先理解代码再下笔"。
- 你在做**多 agent 协作**，需要让一个 agent 接手另一个 agent 改过的代码——Handbook 在这里是天然的"交接文档"。

如果你的项目还处于 prototype 阶段、单文件几百行、每个 stage 都清楚，**Handbook 的 ROI 大概率是负的**——维护它比维护代码本身还累。

---

## 结语：Handbook-as-Memory 是 Agent 自我进化的关键

论文在 Conclusion 提到下一步是 **harness self-evolving**——"using the handbook as a shared behavioral memory, an agent can autonomously close the loop of localization, planning, execution, and resynchronization as the repository evolves"。

这是我觉得最值得追的方向：**当 Handbook 稳定到一定程度，它就能充当 agent 的「行为记忆」**——新会话的 agent 不用从零理解整个仓库，读 Handbook 就知道"系统有哪些行为、行为如何分布、哪里是热点、哪里是冷路径"。这个意义远超"帮人改代码"——它是让 **agent 系统具备长期可演化性**的基础设施。

说到底，这篇论文卖的不是"我用 AI 改代码改得更准"，而是让 **agent harness 这件事变得可读、可导航、可演化**。后者才是大规模 agent 系统的真正瓶颈。

arXiv 链接：[https://arxiv.org/abs/2607.13285](https://arxiv.org/abs/2607.13285)，项目页有 SKILL.md manifest 示例和完整的 BGPD prompt 模板，感兴趣可以自己跑一下。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
