# Agent 当裁判光看 Trajectory 不够，它得自己去环境里查证 —— AJ-Bench 论文解读

## 核心摘要

我们做 Agent RL 训练的时候有个特别恶心的问题：用 LLM-as-a-Judge 给 trajectory 打分，模型经常一本正经地胡说八道。比如让它判断"某论文最新版的发布日期是不是 2025-08-09"，它没法上网查，到头来只能"我无法确认"或者瞎蒙一个。这篇来自中科大、新国立和美团的工作，把"让 Judge 也变成 Agent"这件事第一次系统化做出来了 —— 给 Judge 配上浏览器、文件系统、Postgres、PPT/Word/Excel 等 60 个工具，让它自己去环境里复现状态、查证据、再下结论。

实测下来，**同一个 base model，开启 Agentic 模式之后，平均 F1 涨 13 个点**（gpt-5-mini-low 从 59.0 涨到 72.4，deepseek-v3.2 从 64.5 涨到 77.3）。更有意思的是，开了工具的 deepseek-v3.2，反过来把不开工具的 gpt-5、claude-opus-4.5 都甩在身后。但论文也很坦诚地说：F1 离饱和还远，**60–80% 的 failure case 是"工具调对了、证据拿到了，结论却推错了"或者"看不懂工具输出"** —— 这暴露的是另一个层面的能力鸿沟。

值得一读，尤其如果你正在搞 Agent 强化学习、Reward Modeling、或者在思考 LLM-as-a-Judge 在复杂任务上为什么不靠谱。

## 论文信息

- **标题**：AJ-Bench: Benchmarking Agent-as-a-Judge for Environment-Aware Evaluation
- **作者**：Wentao Shi*, Yu Wang*, Yuyang Zhao*, Yuxin Chen, Fuli Feng, Xueyuan Hao, Xi Su, Qi Gu†, Hui Su, Xunliang Cai, Xiangnan He†
  - *为共同一作；†为通讯作者
- **机构**：中国科学技术大学（USTC）、新加坡国立大学（NUS）、美团（Meituan）
- **链接**：https://arxiv.org/abs/2604.18240
- **项目页**：https://aj-bench.github.io/

---

## 一、为什么 LLM-as-a-Judge 在 Agent 评测上撑不住了

先讲个我自己被坑过的场景。

去年我在做一个 Agent RL 训练 pipeline，用 LLM-as-a-Judge 给 trajectory 打 reward。task 是"给我找出 2024 年在中美同步上映的电影并提供官方来源"。Agent 跑完吐出一个 markdown 表格，看起来很像那么回事 —— 标题、日期、来源链接齐全。LLM Judge（当时用的 GPT-4）一看：结构清晰、来源可信，给个高分。

问题来了。我抽查的时候发现，里面三部电影里有一部根本不存在。Agent 编了个名字、编了个日期、连 IMDB 链接都是 hallucination 出来的。LLM Judge 因为没法点开链接验证，只能从"看起来合理"这个层面给分 —— 结果就是 reward signal 完全是噪声。

这就是 **LLM-as-a-Judge 的根本局限**：它的判断只能 ground 在文本表面信号上。一旦 task 涉及"事实是不是真的"、"这个文件是不是真的被改了"、"这个 PPT 的某个对象是不是真的存在"这类需要**环境状态**才能验证的事情，纯文本 Judge 立刻就废了。

![图1：LLM-as-a-Judge vs Agent-as-a-Judge —— 上半部分是纯文本 Judge 因为没法上网，只能写"我无法确认"；下半部分是 Agent-Judge 直接调 browser_navigate 拿到 arXiv 页面，看到真实日期是 09-19 而非 08-09，正确判错](https://arxiv.org/html/2604.18240v1/x1.png)

*图1 这个例子特别戳：query 问 LongCat-Flash 技术报告的发布日期，response 给了 2025-08-09。LLM-as-a-Judge 因为无从查证，只能含糊地说"I cannot definitively verify"，给了一个错误的判断；Agent-Judge 直接调 browser 打开 arxiv.org/abs/2509.01322，拿到真实日期 19 Sep 2025，判定 response 错误。同样的问题，工具有没有，结论是相反的。*

Rule-based verifier 也救不了。RLHF 那一套规则匹配在数学题、代码题这种**输出空间窄**的场景能 work，但一到"开放式问答"、"GUI 操作是否成功"、"文件系统是不是被正确改写"这种场景，规则就写不出来或写了也不全。

之前 Zhuge et al. 2025 那篇 ICML paper 第一个提了 Agent-as-a-Judge 的 idea，但他们只在 DevAI 上做了实验，covered 的就是 code verification 这一个窄场景。**Agent-as-a-Judge 到底在更广泛的 task 上能不能 work、能 work 到什么程度，是空白的**。

这就是 AJ-Bench 想填的坑。

---

## 二、AJ-Bench 怎么造的：3 个 domain × 3 个验证维度

### 2.1 整体框架

![图2：AJ-Bench 的整体框架 —— 上半部分是 benchmark 构造（task design → trajectory collection → label annotation），下半部分是评测流程（环境 replay 到 final state → Agent-as-a-Judge 在 3 种验证模式下工作）](https://arxiv.org/html/2604.18240v1/x2.png)

*图2 把整套方法的骨架画清楚了：先从 4 个数据源（Mind2Web2、WideSearch、MCPMark、OSWorld）筛 task，跑模型生成 trajectory，再人工标注 binary label。评测时把环境 replay 到 task 执行完的 final state，让 Judge Agent 在这个活的环境里通过 process verification、state verification、information acquisition 三种模式去判 trajectory 对错。*

注意"replay 到 final state"这个设计 —— 它的妙处在于：评测 GUI 任务时，你不需要真的把 trajectory 重新跑一遍（GUI 操作慢、容易因为非确定性失败），而是把环境直接快进到"trajectory 跑完应该到达的状态"，然后让 Judge 自己去这个状态里翻看：文件改了没？数据库写入了没？PPT 里那个图表加了没？

### 2.2 三个 domain 各自考察什么

不同 domain 强调不同的验证能力：

| Domain | 子领域 | 任务来源 | 主要考察 |
|--------|-------|---------|---------|
| **Search** | Wide / Deep | Mind2Web2 + WideSearch | 信息获取（External web search） |
| **DS** | FileSystem / Postgres | MCPMark | 状态验证（环境内查询） |
| **GUI** | PPT / Word / Excel | OSWorld | 状态验证 + 过程验证（关键操作 audit） |

这三个 domain 的选择不是拍脑袋。Search 这块用 deep search（多跳推理）+ wide search（广覆盖检索），是"一句答案直接 verifiable" 不够时必须的能力；DS 选 FileSystem 和 Postgres 是因为环境状态可以**直接查**，对错没歧义；GUI 选 office 三件套是因为它们对"动作精确性"的要求最高 —— PPT 里某个图位置错了、Excel 公式错了，光看 trajectory 判不出来，得真的打开文件看。

### 2.3 数据规模

![图3：AJ-Bench 任务分布 —— 内圈是 3 个 domain（Search/DS/GUI），中圈是 7 个子领域，外圈是细分类别。Deep search 占了最大块（52 task），覆盖娱乐、学术、科技专利、游戏、餐饮、体育等 21 个细类](https://arxiv.org/html/2604.18240v1/x3.png)

*图3 是个旭日图，把 155 个 task 在不同子领域的分布画出来。Deep search 是最大头（52 task），覆盖面最广 —— 学术研究、科技专利、人物关系、体育赛事、地理环境，几乎你能想到的"得多跳查证"的领域都有。GUI 这块 PPT 21 task、Excel 19 task、Word 12 task，每个细类都打磨过。*

总数据量：**155 tasks + 516 trajectories + 60 tools**。每个 task 配 3 个成功 trajectory + 3 个失败 trajectory（DS 这边），保证 label 平衡。

### 2.4 标注怎么做的

这里有几个细节我觉得挺扎实：

- **Search domain**：Mind2Web2 部分用人工标注 + GPT-4.1 做 rubric 拆分到 single-item 级别；WideSearch 部分用 6 个模型 majority voting（gpt-4.1、gpt-5、o4-mini、claude-sonnet-4、gemini-2.5-pro、grok-3）作为 ground truth
- **DS domain**：用 MCPMark 自带的 verifier script（人工写的高质量校验脚本）+ 人工 review
- **GUI domain**：先用 OSWorld 的规则脚本对照 golden file，然后**全部人工复核**

binary label（1/0）。Search domain 是 item-level 标注，DS/GUI 是 trajectory-level。

---

## 三、Judge Agent 怎么实现的

实现框架基于 [MCPMark](https://github.com/eval-sys/mcpmark)，它的 design philosophy 是"考察 LLM 的 intrinsic 能力 —— 自己决定何时用工具、用哪个工具、怎么解析结果"，**不写复杂的 workflow 编排**。这个选择我挺认同的：如果 Judge 还得靠人手写状态机来兜底，那它说到底还是 LLM-as-a-Judge 加了点 if-else，没意思。

整个评测以 **F1 score** 为主指标 —— 因为 binary classification（PASS/FAIL）下，accuracy 在样本不平衡时会骗人，F1 才是诚实的。所有结果跑 3 次取平均。

---

## 四、主实验：Agent 模式带来的提升有多大

### 4.1 主表数据

直接看 Table 3 这张表（论文里最关键的数据）：

| 模型 | Agentic | Search-Wide | Search-Deep | FileSystem | Postgres | PPT | Word | Excel | **Overall Avg@3** |
|------|--------|-------------|-------------|------------|----------|-----|------|-------|------------------|
| gemini-3-pro-preview | ✗ | 72.70 | 81.26 | 75.69 | 73.20 | 76.10 | 72.14 | 74.28 | **75.05** ± 1.26 |
| gemini-2.5-pro | ✗ | 66.35 | 81.22 | 66.10 | 68.96 | 68.72 | 60.13 | 66.67 | 68.31 ± 0.95 |
| claude-opus-4.5 | ✗ | 64.26 | 81.11 | 66.06 | 69.66 | 59.21 | 51.45 | 75.77 | 66.79 ± 1.33 |
| claude-sonnet-4.5 | ✗ | 61.02 | 81.34 | 69.26 | 68.36 | 75.61 | 61.56 | 71.24 | 69.77 ± 1.18 |
| gpt-5 | ✗ | 66.33 | 80.37 | 59.09 | 62.84 | 51.90 | 44.81 | 61.78 | 61.02 ± 0.13 |
| gpt-5.1 | ✗ | 58.02 | 70.90 | 46.27 | 57.53 | 41.90 | 39.54 | 60.33 | 53.50 ± 3.56 |
| grok-4 | ✗ | 69.18 | 78.32 | 75.70 | 59.57 | 61.11 | 65.26 | 75.52 | 69.24 ± 1.11 |
| **gpt-5-mini-low** | ✗ | 60.84 | 68.42 | 60.41 | 65.52 | 45.05 | 48.41 | 64.36 | 59.00 ± 0.91 |
| **gpt-5-mini-low** | **✓** | 65.93 | 75.69 | 67.54 | 67.30 | 76.28 | 72.22 | 81.89 | **72.41** ± 1.68 |
| **Improvement** |   | +5.09 | +7.27 | +7.13 | +1.78 | **+31.23** | **+23.81** | **+17.53** | **+13.41** |
| kimi-k2-0905-preview | ✗ | 63.52 | 80.17 | 55.96 | 65.85 | 65.53 | 55.39 | 63.90 | 64.33 ± 2.07 |
| qwen3-235b-a22b | ✗ | 62.69 | 81.33 | 64.66 | 64.32 | 45.50 | 36.82 | 53.97 | 58.47 ± 2.32 |
| glm-4.6 | ✗ | 66.61 | 77.88 | 60.86 | 64.94 | 60.82 | 50.07 | 72.49 | 64.81 ± 0.96 |
| **deepseek-v3.2** | ✗ | 63.65 | 62.91 | 60.31 | 66.31 | 58.38 | 69.77 | 70.12 | 64.49 ± 0.50 |
| **deepseek-v3.2** | **✓** | 72.47 | 82.14 | 72.60 | 72.70 | 83.14 | 78.64 | 79.71 | **77.34** ± 1.36 |
| **Improvement** |   | +8.82 | +19.23 | +12.29 | +6.39 | +24.76 | +8.87 | +9.59 | **+12.85** |

几个我看完皱眉的点：

**第一，开 Agentic 之后，gpt-5-mini-low 直接超过了不开 Agentic 的 gpt-5、claude-opus-4.5、grok-4。** 一个"小模型 + 工具"打过"大模型 + 没工具"，这个结论非常 telling：Judge 任务的 bottleneck 不是模型推理能力，而是**信息访问能力**。

**第二，GUI 这块的提升幅度极其夸张。** PPT/Word/Excel 三项，gpt-5-mini-low 开 Agentic 后涨了 17–31 个点，deepseek-v3.2 也涨了 9–25 个点。这背后的逻辑很直接 —— 你光看 trajectory 文本，根本判不出"这张 PPT 现在是不是符合要求"，必须真的去看那张 PPT。

**第三，开了工具的 deepseek-v3.2（77.34）是全表第一，超过了不开工具的 gemini-3-pro-preview（75.05）。** 这一刀砍得挺有意思 —— 在 Agent 评测这件事上，开源模型 + 工具完全可以打赢闭源 SOTA + 没工具的组合。

但等等。我得提一个 caveat。这个对比不算严格公平 —— 闭源模型那一栏是 LLM-only，没给它们配工具。如果 gemini-3-pro-preview 也开 Agentic 模式，会不会又是另一番景象？论文在 Appendix A.5 补了 Gemini 3 Flash Preview 等几个模型开 Agentic 的小规模实验，结果是**几乎所有模型开 Agentic 都涨**。所以 deepseek-v3.2 这个第一不能解读为"开源模型已经超过闭源"，而是"工具增强这件事本身收益巨大"。

### 4.2 Reasoning effort 没那么重要

Table 4 是个有意思的反 intuition 结果：

| Model | Reasoning | Wide | Deep | FileSystem | Postgres | PPT | Word | Excel | Overall |
|-------|-----------|------|------|------------|----------|-----|------|-------|---------|
| gpt-5-mini | low | 65.93 | 75.69 | 67.54 | 67.30 | 76.28 | 72.22 | 81.89 | 72.41 |
| gpt-5-mini | medium | 72.76 | 77.11 | 75.80 | 69.84 | 82.05 | 72.00 | 82.35 | **75.99** |
| gpt-5-mini | high | 74.48 | 79.19 | 71.53 | 67.92 | 78.95 | 63.64 | 81.08 | 73.83 |
| deepseek-v3.2 | N/A | 72.47 | 82.14 | 72.60 | 72.70 | 83.14 | 78.64 | 79.71 | **77.34** |
| deepseek-v3.2 | thinking | 70.37 | 79.31 | 68.83 | 74.13 | 82.05 | 78.57 | 86.49 | 77.11 |

**reasoning effort 拉到 high 反而比 medium 差，deepseek-v3.2 的 thinking 版本反而比 no-thinking 还稍微低一点。** 这个反 intuition 在哪里？我们一般觉得"思考越多越好"，但在这个 task 上不对 —— Judge 任务的关键技能是"决策什么时候调工具、调哪个工具、怎么读工具返回值"，**这是 procedural skill 而不是 cognitive depth**。让模型多思考一会儿，可能反而让它"想多了 / 自信地走偏了"。

这点其实跟最近 RL 圈一些经验吻合 —— overthinking 在 tool-use 类任务上有时是负担。

### 4.3 交互轮数：早期收益最大

![图4：交互轮数对评测结果的影响 —— PPT/Word/Excel/FileSystem 四个子领域随着 max interaction turns 从 1 到 32，F1 都在涨。Word 最敏感（25 → 78），PPT 次之（44 → 83），Excel 已经在前几轮就到 75+ 不太涨了](https://arxiv.org/html/2604.18240v1/x4.png)

*图4 给我两个直观判断：第一，Judge 给越多交互预算就越准，没有"够用就行"的临界点；第二，**前 8 轮收益最大，后面是边际递减**。Word 这条绿线最陡，说明 Word 任务最依赖"反复查"。Excel 那条紫线最早平 —— Excel 验证比较"原子化"，看一眼公式或者表格就能下结论。*

工程上的启发：**配 Judge Agent 的 interaction budget 时，8–16 轮是性价比最高的甜区**，再往上 ROI 急剧下降。

### 4.4 多模态：mixed 不一定最好

![图5：GUI domain 的多模态 ablation —— A11y Tree、Screenshot、Mixed 三种输入对比。PPT 上 A11y Tree 和 Mixed 差不多；Word 上 gemini-3-flash 偏爱 Screenshot（91.7 vs A11y 80）；Excel 上 Mixed 普遍最好（92.3）](https://arxiv.org/html/2604.18240v1/x5.png)

*图5 这张图有点反 intuition —— 我以为"输入信息越多越好"，但在 PPT 和 Word 上，Mixed 反而经常不是最好的。比如 gpt-5-mini-low 在 Word 上，Screenshot 单独给（76.9）比 Mixed（54.5）好整整 22 个点。作者的解读是 mixed 输入引入了 noise 和 redundancy，把模型搞分心了。*

这个发现其实挺值得多模态 Agent 圈关注的。一直有种朴素假设："模态越多信息越完整"，但实际上**多模态融合的代价是 attention 被分散** —— 文本 + 截图同时塞进去，模型需要消耗更多的 context budget 去对齐两边的信息，反而可能漏掉关键 cue。

### 4.5 Failure modes：60–80% 是"看不懂工具输出"

论文 Appendix A.7 这张 failure mode 表格，是我觉得整篇文章最有 actionable insight 的部分：

| Domain | 模型 | (a) 工具没调 | (b) 调错工具 | (c) 看不懂输出 | (d) 证据对了但推理错 |
|--------|------|------------|------------|--------------|------------------|
| Search-Deep | deepseek-v3.2 | ~4% | ~2% | **~40%** | **~54%** |
| Search-Wide | deepseek-v3.2 | ~12% | negligible | **~57%** | ~31% |
| FileSystem | deepseek-v3.2 (thinking) | negligible | negligible | **~79.2%** | ~20.8% |
| Postgres | deepseek-v3.2 | ~6.1% | ~3.0% | ~48.5% | ~42.4% |
| Excel | deepseek-v3.2 (thinking) | negligible | negligible | **~80%** | ~20% |
| Word | deepseek-v3.2 (thinking) | negligible | negligible | **~83.3%** | ~16.7% |
| PPT | deepseek-v3.2 | ~22.2% | negligible | ~66.7% | ~11.1% |

把这张表的结论压缩成一句话：**Judge 现在的瓶颈不是"会不会调工具"，而是"调完工具后能不能正确解读返回值 + 严格按规则推理"**。

**(a)+(b) 加起来通常不到 30%**，说明现代 LLM 已经基本会"挑工具调"了；
**(c) 看不懂输出是最大头**，比如读了 Excel 截图但没意识到某个 cell 的格式错了；
**(d) 证据对了推错了**特别坑 —— Appendix 里有个例子，task 是"找出所有重复内容的文件并移到 duplicates/ 目录"，Judge 自己列了两种解读（严格 vs 宽松），自己也指出 verify script 的逻辑是严格版本，**到头来还是给了 PASS** —— 这种"知道规则但不应用"的问题很难训。

---

## 五、ReAct vs MCPMark：framework 也有讲究

论文 Appendix A.4 还做了 framework 的 ablation：把 Agent-as-a-Judge 换成 ReAct framework（显式 thought + action），对比 MCPMark 的隐式自由发挥。

| Framework | Model | Wide | Deep | PPT | Word | Excel |
|-----------|-------|------|------|-----|------|-------|
| LLM-Judge | gpt-5-mini-low | 60.84 | 68.42 | 45.05 | 48.41 | 64.36 |
| MCPMark | gpt-5-mini-low | 65.93 | 75.69 | 76.28 | 72.22 | 81.89 |
| ReAct | gpt-5-mini-low | 51.13 | 71.84 | 64.86 | 63.64 | 76.47 |
| LLM-Judge | deepseek-v3.2 | 63.65 | 62.91 | 58.38 | 69.77 | 70.12 |
| MCPMark | deepseek-v3.2 | 72.47 | 82.14 | 83.14 | 78.64 | 79.71 |
| ReAct | deepseek-v3.2 | 70.51 | 77.88 | **95.24** | 75.00 | **85.17** |

ReAct 在 deepseek-v3.2 + GUI 上反而表现更好（PPT 从 83 涨到 95.24，Excel 从 79.7 涨到 85.17），但在 gpt-5-mini-low 上明显不如 MCPMark。**Framework 的最优选择跟 base model 的特性强相关**，没有"一招鲜"的方案。这点对工程实践挺重要的：选 Judge framework 不能照搬 paper，得在自己的 model 上跑一下。

---

## 六、我的几个判断

**第一，AJ-Bench 解决的是真问题。** 不是那种"为了 paper 凑个 benchmark"的工作。Agent RL 训练已经卷到大家都在拼 reward 质量了，传统 LLM-as-a-Judge 的 ceiling 早就摸到了。Agent-as-a-Judge 是个绕不开的方向，AJ-Bench 把"什么是 good Agent-Judge"这个问题给了一个可量化的 evaluator —— 对整个 community 的价值是 infrastructure 级别的。

**第二，13 个点的提升没那么"颠覆"，但很"扎实"。** 别被宣传话术忽悠 —— Agent-as-a-Judge 不是新概念（Zhuge 2025 提过），核心 idea 也不复杂（给 Judge 工具用）。这篇 paper 的贡献是把它**做出来 + 系统评测**，证明这条路 work。13 个点的平均提升在 RL 训练 reward 这种应用场景下其实非常可观 —— reward signal 准确率涨 13 个点，下游 policy 训练的 sample efficiency 可能能涨一两倍。

**第三，77% 这个数还远不够用。** 如果你打算把这套东西直接接到 RL pipeline 里当 reward model，要小心 —— 23% 的 false signal 在 long-horizon RL 里会被持续放大。Failure mode 分析显示主要问题是"读不懂工具输出"和"推理跑偏"，这两个都不是简单"加数据"能解的问题，可能需要专门 fine-tune 一个 Judge model（用 trajectory + 正确判断作为 SFT 数据）。

**第四，对工程的启发**：
- 如果你的 Agent 评测涉及 GUI 操作、文件状态、数据库变更，**别再用纯 LLM-as-a-Judge 了** —— 给它配工具，回报很高
- Interaction budget 8–16 轮是甜区，别开太多浪费钱
- Reasoning effort 别拉满，medium 通常是最优的
- 多模态输入要 task-by-task 调，不要默认 mixed 最好
- ReAct vs 自由发挥，跟你的 base model 强相关，必须实测

**第五，一个值得追问的方向**：论文用 majority voting + 人工 review 做的 ground truth，本身也有 noise。当 Judge Agent 的 F1 接近 90 的时候，到底是 Judge 真的做得好，还是 ground truth 里那些"边界 case"恰好被它猜中？**评测的天花板被 ground truth 质量决定**，未来要做更难的 benchmark，annotation 这块的成本会指数级上升。

---

## 收尾：这是 Agent RL 训练 infra 必经的一步

Agent training 这两年的瓶颈一直在 reward。从最早的 outcome reward → process reward → LLM-as-a-Judge → 现在的 Agent-as-a-Judge，每一步都是在把"判断对错"这件事变得**更接近真实环境**。AJ-Bench 这篇 paper 的位置，就是给"Agent-as-a-Judge 这个范式到底能到什么水平"画了第一张地图 —— 现在我们知道 ceiling 在 70–80% F1，知道主要瓶颈在 tool output 解读和 reasoning rigor，知道 reasoning effort 拉满反而有害，知道 8–16 轮 interaction 是性价比最高的。

如果你正在做 Agent RL，或者在设计下一代 Reward Model 的 pipeline，这篇值得仔细啃一遍 —— 不是为了 cite，是为了把里面的工程经验直接搬进自己的系统。

代码和数据都开源了：[https://aj-bench.github.io/](https://aj-bench.github.io/)

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
