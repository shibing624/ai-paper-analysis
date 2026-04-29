# 给Agent装"持续学习"开关：写技能容易，修技能难——SkillFlow揭开11个前沿模型的真实差距

## 核心摘要

如果你最近在调一个带 Skills 的 Coding Agent，应该会有这种感觉：模型刚部署的时候表现还不错，跑了一阵子后总结出一堆所谓"经验"写进 skill 库，结果新任务来了反而比一开始更糟。这不是错觉。

SkillFlow 是中科大、多伦多大学、悉尼大学合作的一个新基准，专门来测**自主 Agent 能不能在执行任务过程中自己挖出可复用 skill、出错了能修、攒一段时间后还稳得住**。166 个真实工作流任务、20 个 task family、跑遍 11 个当前最强的 Agent 模型组合（包括 Claude Opus 4.6、GPT-5.3 Codex、Kimi K2.5、Qwen-Coder-Next 等）。

数据相当扎心：Claude Opus 4.6 在 lifelong 评测下从 62.65% 涨到 71.08%，**+8.43 个点**，这是全场最强；但 Kimi K2.5 用了 66.87% 的 skill 调用率，只多涨了 **+0.60 个点**；Qwen-Coder-Next 直接是 **−0.60，越学越差**。说到底，技能库这个机制不是给所有模型都送钱的——只有那些本身能"识别错误技能并修复"的模型才能真的吃到红利。

这篇论文最值钱的地方不是又造了个 benchmark，而是它把"模型能不能自我进化"这件事拆解成了**写 skill、用 skill、修 skill** 三个能力，并且发现**前两个全员都会，唯独第三个是真正的能力分水岭**。

---

## 论文信息

- **标题**：SkillFlow: Benchmarking Lifelong Skill Discovery and Evolution for Autonomous Agents
- **作者**：Ziao Zhang, Kou Shi, Shiting Huang, Avery Nie, Yu Zeng, Yiming Zhao, Zhen Fang, Qisheng Su, Haibo Qiu, Wei Yang, Qingnan Ren, Shun Zou, Wenxuan Huang, Lin Chen, Zehui Chen, Feng Zhao
- **机构**：中国科学技术大学（USTC）、多伦多大学、悉尼大学
- **arXiv**：[https://arxiv.org/abs/2604.17308](https://arxiv.org/abs/2604.17308)
- **代码**：[https://github.com/ZhangZi-a/SkillFlow](https://github.com/ZhangZi-a/SkillFlow)

---

## 痛点：现在测 Agent Skill 的 benchmark，全在测错地方

Skill 这个东西最近一两年在 Agent 圈非常火。Anthropic 把 Skills 直接做进 Claude Code 主仓，OpenAI 的 Codex CLI、Google 的 Gemini CLI、阿里的 Qwen Code、Moonshot 的 Kimi CLI，全部内置了 plug-and-play 的 skill 加载机制。所谓 skill，简化理解就是一段把"特定领域的操作经验、踩过的坑、可复用代码"打包成 markdown + 脚本的目录，让 Agent 在做新任务时可以"读说明书"。

但现有 benchmark 的问题在于——**它们只测"模型能不能用现成的 skill"**。你给一个写好的 PDF 处理 skill，让 Claude 用它解决一个 PDF 任务，搞定；这就是 SkillsBench 这类工作做的事。

可工业里大家真正想要的是另一件事：**能不能在没有人写 skill 的情况下，让 Agent 自己一边干一边总结，攒出一个越用越好用的 skill 库？**

这个区别非常关键。第一种是"会不会用工具"，第二种是"会不会从经验里造工具"。前者就像考你用扳手能不能拧螺丝，后者是考你能不能自己打磨出一把更好用的扳手——更接近"持续学习"的本质。

我之前在做 Coding Agent 的时候踩过类似的坑：让模型每跑完一个任务就把"教训"写成一个 cheatsheet 存起来，结果跑到第 5、6 个任务的时候，模型开始疯狂把每个新任务都总结成新 cheatsheet，库越来越大，但调用率反而下降——因为同一个能力被分散在七八个文件里，模型自己都找不到了。这种"碎片化技能膨胀"问题，论文里有非常精彩的图表来量化。

![图1：SkillFlow 概念总览图](https://arxiv.org/html/2604.17308v1/x1.png)

*图1：SkillFlow 的核心设定。上半部分是传统的"静态 skill 评测"，给一组任务，Agent 直接用预制 skill 解决；下半部分是 SkillFlow 的 lifelong 设定——Agent 从空 skill 库开始，每完成一个任务就把经验"外化"成 skill patch（增/改/删），下一个任务带着进化后的库继续做。这中间会暴露一个核心问题：Agent 能不能识别自己写错的 skill 并修正它？*

---

## 方法核心：怎么造一个真正能测"技能进化"的基准

要测"技能能不能跨任务复用"，最大的难点是**怎么定义"同一类任务"**。如果任务之间太相似（比如做 5 个完全一样的财务报表），那纯属抄答案；如果任务之间太不一样（比如这个任务是 OCR、下一个是写代码），那 skill 复用根本无从谈起。SkillFlow 给出的答案是 **Domain-Agnostic Execution Flow**（DAEF）。

### DAEF：用工作流"骨架"定义任务族

DAEF 的核心想法很简洁：**剥离掉任务里所有具体的领域信息（公司名、字段名、文件名），只保留操作步骤的拓扑结构**。形式化定义是把每个任务表示成一个工作流图：

$$\mathcal{T}=(V,E,\lambda,\gamma)$$

其中 $V$ 是子目标节点集合，$E$ 是依赖边，$\lambda(v)$ 给每个节点打一个领域无关的操作类型标签，$\gamma(v)$ 是任务相关的具体绑定。DAEF 就是把 $\gamma$ 抹掉之后剩下的东西：

$$\mathcal{F}=\phi(\mathcal{T})=(V_F, E_F, \lambda_F)$$

听起来抽象，但看图就直观了：

![图3：DAEF 在不同领域间的对应关系](https://arxiv.org/html/2604.17308v1/x3.png)

*图3：左边是"FX 汇率即期点差修正"任务，右边是"催化剂用量平衡"任务。两个看似毫不相关，但抽出 DAEF 后骨架完全一致：read → extract → retrieve → update → validate → output。这就是 SkillFlow 任务族构造的核心——**同一个 DAEF 下的任务，理论上可以共享同一个底层 skill**。*

DAEF 节点标签来自一个固定词表：read / extract / retrieve / normalize / filter / align / compute / compare / detect / update / validate / output（共 12 个），由两个独立专家分别标注、互验之后才会被纳入。每个 DAEF 限定在 5–8 个节点之间，太短没有进化空间，太长无法保证一致性。

我得说，DAEF 这个设计是这篇论文最有想法的地方。它解决了一个非常实际的问题：**怎么在不让模型作弊（直接复用题面）的前提下，让它有机会学习并迁移过程性知识**。我之前看 SkillWeaver、SkillRL 那批工作，痛点就是任务相似度难以控制——SkillFlow 用 DAEF 把这条边界画清楚了。

### 任务构建：双 Agent 流水线 + 人工审核

任务怎么造出来的？过程有点像内容流水线：

![图2：SkillFlow 任务构建流水线](https://arxiv.org/html/2604.17308v1/x2.png)

*图2：四步构建流程。Step 1 收集种子任务和 skill 池；Step 2 用 embedding 做 task-skill 配对；Step 3（图中右侧 Architect/Critic 部分有省略）是双 Agent 反复迭代生成新任务族；Step 4 是人工审核。*


**Step 1**：从 GDPval 和 SkillsBench 里挑出 64 个真实工作场景的种子任务，同时从 Anthropic Skills 仓库、Awesome-Claude-Skills 等公共仓库里收集 8000+ 个开源 skill，过滤后留下 2,318 个。

**Step 2**：用 Qwen3-Embedding-4B 计算 task description 和 skill description 的语义相似度，给每个种子任务匹配 5–10 个候选 skill。

**Step 3**：把每个种子任务标注成 DAEF（去重后留 30 个），扔进**双 Agent 循环**：
- **Architect Agent**（Cursor + GPT-5.3-Codex）负责按 DAEF 生成新任务、构造 Docker 环境、写 verifier；
- **Critic Agent**（Claude Opus 4.6）负责在真实 Docker 里跑、检查工作流一致性、难度梯度、可解性；
- 两边来回拉扯，每个 family 跑两轮，每轮 4–5 个任务，最终每个 family 8–9 个任务。

**Step 4**：人工审核四个维度——指令是否泄露 skill、逻辑是否自洽、环境是否完整、难度梯度是否合理。

最终留下 **20 个 task family、166 个任务**，覆盖 5 大领域：

![图4：SkillFlow Taxonomy](https://arxiv.org/html/2604.17308v1/x4.png)

*图4：内圈是五大领域（金融经济、运营供应链、医疗、治理战略、数据文档智能），外圈是 20 个具体的 DAEF 工作流家族。注意 "Data & Document Intelligence" 这一块占比最大——OCR、PPT 格式化、HWPX 自动化这种重复性高、规则强的工作场景，正是 skill 机制最该发力的地方。*

### Agentic Lifelong Learning Protocol：核心评测协议

这是整个 benchmark 的灵魂。给定一个有序的任务族 $\mathcal{F} = \{T_1, T_2, \ldots, T_n\}$（按难度排序），Agent 要按下面的流程跑：

**首任务**：$T_1$ 必须裸跑，不能用任何 skill。完成后拿到 verifier 给的 rubric 反馈 $r_1$。

**Skill patch 生成**：每跑完一个任务，模型自己根据轨迹 $\tau_t$ 和反馈 $r_t$ 生成一个 patch：

$$\Delta_t = \text{Model}_g(\mathcal{S}_{t-1}, \tau_t, r_t), \quad \mathcal{S}_t = \text{Apply}(\Delta_t, \mathcal{S}_{t-1})$$

这里 $g$ 是固定的 prompt 模板，$\Delta_t$ 是一个 JSON，包含三个字段：
- `summary`：自然语言描述这次学到了什么；
- `upsert_files`：要新增/覆盖的 skill 文件路径 → 内容映射；
- `delete_paths`：要删除的过期 skill 路径。

注意 patch 里**可以删 skill**——这一点很关键。现实里的 skill 库经常是越攒越烂的，删除能力本身是评测的一部分。

**后续任务**：$T_t$ 带着进化后的 $\mathcal{S}_{t-1}$ 跑。所有任务跑完后，每个 family 内部有一份 family-local 的 skill 库；不同 family 之间是隔离的（**Family Reset**），避免引入跨工作流的检索噪声。

为什么 family 内部隔离？作者的解释是：lifelong 学习的核心是"在同一个任务类下持续优化"，不是"跨异构工作流的 skill 检索"。后者是另一个问题，会被 retrieval mechanism 的混入因素污染评测。我觉得这个 trade-off 是合理的——先把"同类任务里能不能持续提升"测清楚，再谈"跨域迁移"会更靠谱。

---

## 实验结果：能力差距比想象中更大

实验部分的核心问题是：**lifelong skill evolution 真的有用吗？对哪些模型有用？**

测了 11 个模型在 4 个 Agent harness 下的组合：Claude Code（4 个 Claude 变体 + 2 个 MiniMax）、Codex CLI（GPT-5.4 + GPT-5.3-Codex）、Qwen Coder（Qwen-Coder-Next + Qwen3-Coder-480B）、Kimi CLI（Kimi K2.5）。

### 主实验表：vanilla vs skill evolution

| Agent / Model | vanilla %comp ↑ | skill %comp ↑ | Δ%comp | #Skills | %use |
|--------------|----------------|---------------|--------|---------|------|
| Claude Code / **Claude Opus 4.6** | 62.65 | **71.08** | **+8.43** | 1.05 | 45.78 |
| Claude Code / Claude Sonnet 4.5 | 49.40 | 55.42 | +6.02 | 2.55 | 72.89 |
| Claude Code / MiniMax M2.5 | 28.31 | 34.94 | +6.63 | 2.50 | 32.53 |
| Claude Code / Claude Opus 4.5 | 58.43 | 60.84 | +2.41 | 1.50 | 60.84 |
| Claude Code / Claude Sonnet 4.6 | 56.63 | 56.63 | +0.00 | 2.55 | 53.01 |
| Claude Code / MiniMax M2.7 | 37.35 | 36.75 | -0.60 | 4.60 | 51.20 |
| Codex CLI / GPT-5.4 | 33.13 | 36.75 | +3.62 | 1.05 | 81.33 |
| Codex CLI / **GPT-5.3 Codex** | 52.41 | 46.39 | **-6.02** | 1.10 | 84.94 |
| Qwen Coder / Qwen-Coder-Next | 45.18 | 44.58 | -0.60 | 5.45 | 12.05 |
| Qwen Coder / **Qwen3-Coder-480B** | 24.70 | 24.10 | -0.60 | 5.20 | 66.87 |
| Kimi CLI / **Kimi K2.5** | 55.42 | 56.02 | +0.60 | 1.50 | 66.87 |

光看这张表就能感受到差距：

**Claude Opus 4.6 是绝对的赢家**——从 62.65% 干到 71.08%，多解出了 14 道题（166 道里），最终 skill 库平均只有 1.05 个文件。它学会了"少而精"的库管理策略。

**GPT-5.3 Codex 是最大的反例**——掉了 6 个点，而且它的 skill 调用率高达 84.94%，**说明它在频繁调用一个错误的 skill 库**。这个现象论文里有详细分析，很值得展开（后面 Finding 2 会讲）。

**Qwen3-Coder-480B 看起来最反直觉**——66.87% 的调用率（很高），但只掉了 0.6 个点，且最终库 5.2 个文件（明显碎片化）。它的能力不在于不写 skill，而在于"写了一堆没啥用的 skill"。

**Kimi K2.5** 是个有趣的中间案例：调用率 66.87%、库精简（1.5），却只涨了 0.6 个点。说明**高调用率 ≠ 高效用**。

### Pareto 前沿：完成率 vs 成本

![图5：Completion-Cost Pareto Frontier](https://arxiv.org/html/2604.17308v1/x5.png)

*图5：横轴是平均成本（USD，对数刻度），纵轴是平均完成率。蓝点是 vanilla 跑，红点是 skill 进化后的跑（带星号），虚线连接同模型的两次结果。Opus 4.6\* 在最右上角，是最强但也最贵的；Kimi-K2.5\* 在 Pareto 前沿上，是性价比最高的；MiniMax 系列在最便宜区间靠 skill 进化吃到了不小提升。但有意思的是，GPT-5.3 Codex\* 比 vanilla 反而向下掉了，这就是"用错 skill 不如不用"的直接证据。*

这张图比单看主表更有信息量。它告诉你：**skill evolution 不是免费午餐——它把模型推向 Pareto 前沿的不同位置，但对一些模型来说，是把它推下了前沿**。这跟 GRPO/RLHF 时代我们对"训练时缩放"的直觉一样——不是把所有模型都喂同样的训练数据就都能提升，模型本身的"接受能力"才是瓶颈。

### 域级热力图：哪些任务受益最大

![图6：Domain-Grouped Completion Gains](https://arxiv.org/html/2604.17308v1/x6.png)

*图6：颜色越红代表 skill evolution 带来的提升越大，越蓝代表掉得越多。HLS（医疗）这一行 Sonnet 4.6 飙到 +0.33 但 GPT-5.3-Codex 掉到 -0.28，差距巨大。整体观察：**没有一个模型在所有领域都涨**，也**没有一个领域所有模型都涨**。skill evolution 的有效性是模型 × 任务双重耦合的。*

### 最重要的对照实验：history-context vs skill library

论文里有一个非常关键的消融——他们在 Claude Opus 4.6 上做了个对照：把所有之前任务的完整对话历史直接拼接成 context 喂给模型（不让它显式总结成 skill），看看能不能复现 skill evolution 的提升。

| Setting | Completion (%) | Δ vs Vanilla |
|---------|--------------|--------------|
| Vanilla | 62.65 | – |
| Full historical trajectory context | **51.04** | **-11.61** |
| Full protocol (skill library) | **71.08** | **+8.43** |

对照组（拼历史）不仅没涨，还**掉了 11.61 个点**。这说明 skill evolution 的收益不是"上下文变长"带来的——它来自于**结构化的经验外化**。模型主动从轨迹里提炼出 decision rule，比看一堆原始 trace 要有价值得多。

这个对照做得相当干净。我自己之前的直觉是"反正 long context 模型这么强，把历史塞进去也差不多"，结果数据告诉我完全不是这样——**塞历史会污染推理，而结构化总结才是关键**。

---

## 六个核心发现：写 skill 不难，修 skill 才是分水岭

论文 3.3 节给了 6 个 finding，我觉得每一个都值得展开，因为这些观察可能比 benchmark 数据本身更有长期价值。

**Finding 1：Opus 4.6 是唯一接近"稳定 skill 学习"的模型。** 它的核心证据不是涨幅最大，而是它真的在**修复**之前写错的 skill。Trace 里能看到：第 N 个任务出错了，模型识别出是某个 skill 文件里的指引有问题，下一个任务它就把那个 skill 改了，然后再下一个任务用新版本一次跑通。这就是"library-level improvement"——不是叠加新经验，是**修旧经验**。

**Finding 2：错误 skill 会引发系统性下游漂移。** 一旦一个错的 skill 被写进库，后面所有任务都会继承这个错误抽象。这跟 RL 里的"路径依赖"问题非常像——模型很难跳出自己生成的逻辑分布。这就是 GPT-5.3 Codex 掉 6 个点的原因：它写了个早期看起来合理但其实不对的 skill，后面持续被这个 skill 带偏。

**Finding 3：统一的高效用 skill > 分散的任务专属 skill。** 看主表里的 #Skills 列：Opus 4.6 的最终库平均只有 1.05 个文件，Sonnet 4.6 也是 2.55，但是 MiniMax M2.7 是 4.60、Qwen-Coder-Next 是 5.45。**强模型倾向于把多个变体合并成同一个不断进化的核心 skill**，弱模型则不停地新建。这个数字非常有说服力。

**Finding 4：Qwen 和部分 MiniMax 主要败在"skill 通胀"。** 

![图7：Skill Count Growth by Task](https://arxiv.org/html/2604.17308v1/x7.png)

*图7：每条曲线是一个模型在 task index 推进时的累积 skill 数。Qwen3-Mext 和 Qwen3-480B-A35B（左上两条）几乎是单调线性增长——每个新任务都额外造一个新 skill。相比之下 Codex CLI / GPT-5.3-Codex 的曲线（最下面的蓝色和红色）几乎平在 1 上，整个任务族只维护一个核心 skill。*

这张图非常直观地展示了"碎片化 vs 整合"两种风格。我读到这里的第一反应是：**Qwen 系列的问题不是不会写 skill，而是不会"丢"skill**——它对每次成功都过度归因，把偶然的局部经验都当成可复用知识保存下来。

**Finding 5：Codex 在"把变体合并到核心 skill"上做得不错——但不一定换来更高完成率。** Codex 库很精简，但完成率比 Opus 系列差不少。这说明**精简不是万能解，精简且修复能力强才是**。

**Finding 6：模型的核心差距在"修复 skill"，不在"写 skill"。** 这是整篇论文最值得记住的一句话。所有模型都能写出某种 skill，原始 generation 不是 bottleneck。真正决定胜负的是：**你能不能识别一个 skill 是错的，改它，然后看到后续任务表现变好**。

我觉得 Finding 6 是这篇 paper 对工业落地最有价值的洞察。换个说法——**与其卷"让模型生成更多 skill"的能力，不如想办法让模型有更强的 critical reflection、能把 verifier 反馈和已有 skill 文件关联起来**。

---

## 失败模式：4 类典型坑

论文 Appendix D.3 列了一份很实在的失败分类，每一类都贴了 trace 片段。挑几个我觉得有共性的：

**D.3.1 验证器/工具链不兼容**：Agent 生成了 SUMPRODUCT 公式，但 verifier 用 `ssconvert --recalculate` 重算时这个 flag 在容器版本里不存在。模型判断"已保存成功"就交付了，没人追究 verifier 是否真的能跑通。这类问题的根因是**模型对自己能力的边界不敏感**——它不知道自己生成的东西要在什么环境下被验证。

**D.3.2 缺失缓存值（最常见）**：openpyxl 写公式但不计算值，verifier 用 `data_only=True` 读取时全部返回 None。Trace 显示 Agent 反复说"公式已写入"，但 verifier 期望的是数值结果。这是个**典型的"过程对了但产物不对"**的失败——Agent 的 self-report 跟 verifier 反馈完全脱钩。

**D.3.3 验证不充分**：Agent 用 Python 自己算了一遍 INDEX/MATCH 结果，"看起来对"就交差，结果 verifier 跑 pytest 直接失败。

**D.3.4 过早宣布成功**：第 N 步还有 verifier 红灯，Agent 已经说"任务完成"。

这些失败模式有个共同点——**它们都跟"信任谁"有关**。Agent 总倾向于相信自己的 self-report，而 verifier evidence 才是 ground truth。论文 Appendix B.2 的 prompt template 里有一句话我觉得点睛：*"Prefer verifier evidence, failed tests, and concrete execution results over the agent's self-report when they conflict."* 这句话应该贴在每个 Coding Agent 项目的 SKILL.md 第一行。

---

## Family-level 热力图：粒度更细的诊断

![图11：Family-Level Skill-Gain Heatmap](https://arxiv.org/html/2604.17308v1/x11.png)

*图11：把图6 的 5 个领域细化到 20 个 task family 后的全景图。能看到一些非常戏剧性的局部表现——比如 Sonnet 4.6 在 Sales Pivot Analysis 上 +1.00（满分提升），在 Compensation Scenario Modeling 上 +0.75；Opus 4.6 在 Weighted Risk Assessment 上 +0.88；但同样的 Sonnet 4.6 在 Distribution Center Auditing 上掉 -0.50、Healthcare Cost-Benefit Analysis 上掉 -0.44。**没有任何一个模型在所有 20 个 family 上都有正提升**。*

这张图给出的工程启发是：**别指望靠 skill evolution 通吃所有任务，应该按 family 维度去做能力适配**——同一个模型在不同 task family 上的表现差异，可能比换模型还大。

---

## 我的判断：这篇论文真正值得记住的几件事

**亮点：**

1. **DAEF 这个抽象层是真的有用**。它给"任务族"画了一条清晰的边界，让 cross-task transfer 评测从"靠相似度玄学"变成了"靠工作流拓扑"。这个抽象有可能成为后续 Agent benchmark 的标配。

2. **history-context 对照实验做得干净**。直接证明了"光给长上下文不够，必须显式结构化"。这对所有在做 Agent 长上下文管理的团队都是一记警钟。

3. **Finding 6 的洞察非常有产品价值**。"修复 skill 比写 skill 更难"——也就是说工业上的 skill 系统设计应该优先投资 reflection/critique 能力，而不是 generation 能力。

**问题：**

1. **Family Reset 设计是把双刃剑**。论文承认这是为了避免 retrieval 噪声，但**真实工业场景里的 skill 库就是要跨 family 检索的**。SkillFlow 测的是"理想隔离条件下的 lifelong learning"，离实际部署还有距离。如果想解决跨域 skill 检索，需要别的 benchmark。

2. **task family 数量偏少**。20 个 family，每个 8–9 个任务，跨 family 间的统计 power 不算特别强。某些 +0.50 / -0.50 的 cell 可能只是 1–2 道题的差别，不太能下"模型 A 比模型 B 在 family X 上强"的结论。

3. **"修复"能力没有独立指标**。论文反复强调 modify > generate，但 metric 里没有 explicit 的 "skill repair rate"——只能从 #Skills 趋势 + 完成率综合推断。如果能加一个"被修改过的 skill 在后续任务里复用率"指标，结论会更硬。

4. **Skill patch prompt 可能过强**。Appendix B.2 的 prompt 模板写得非常细致（YAML frontmatter、目录结构、命名规则全规定死了）。这一定程度上"喂"了模型怎么写 skill，可能掩盖了基础模型在自由 schema 下的真实能力。

**对工程的启发：**

如果你正在做带 Skills 的 Agent 系统，这篇论文的几个直接 actionable item：

- **加 verifier-driven feedback loop**，让 skill patch 强制基于 rubric 输出，而不是 Agent 自评；
- **限制 skill 库膨胀**——比如设个上限触发强制合并，或者在 patch prompt 里明确"优先 update 现有 skill 而不是新建"；
- **重点测试模型的"识别错误 skill"能力**，而不是只看 generation；
- **跨 family 测试要分开做**——SkillFlow 的 family reset 设定告诉你，跨域 skill 检索是另一个独立问题，需要单独的 retrieval 机制设计。

---

## 收尾

读完这篇论文我有个挺强的感觉：**Agent skill 这套机制正在从"宣传卖点"变成"严肃工程问题"**。Anthropic、OpenAI、Google 都把 skill 内置进了主流 CLI，但底层能力到底怎么样，之前其实没有靠谱的横向测评。SkillFlow 把这个空缺补上了，而且补得很扎实——20 个真实工作流、166 个任务、11 个模型组合、双 Agent 流水线 + 人工审核，工程量不小。

但更让我在意的不是这个 benchmark 本身，而是它揭示出的能力分层：**前沿模型在"持续学习"这件事上的差距，已经大到跨越一个量级**。Opus 4.6 的 +8.43 和 Qwen3-Coder-480B 的 −0.60，甚至不是同一种行为模式——前者在主动修复，后者在被动膨胀。

如果你在做长程 Agent 系统、特别是要让 Agent 在生产环境里持续运行的，这篇 paper 值得仔细读。它给出的不是某个具体方案，而是一套**诊断框架**——你可以用 DAEF + Lifelong Protocol 的思路去测自己 Agent 的"skill 健康度"，看它是属于"会修能用"的那一类，还是"越长越烂"的那一类。

更长远来看，能不能从经验里持续提炼可复用的过程性知识，可能是未来通用 Agent 走向"真正可部署"的核心能力之一。SkillFlow 让我们第一次能用数字来谈这件事，这本身就值得一个赞。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
