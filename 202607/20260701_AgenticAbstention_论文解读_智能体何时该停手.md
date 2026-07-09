---
title: 智能体何时该停手：28,000 任务下 LLM 弃权能力全景实测
date: 2026-07-01
arxiv: 2606.28733
authors: Han Luo, Bingbing Wen, Lucy Lu Wang
short_name: AgenticAbstention
---

# 智能体何时该停手：28,000 任务下 LLM 弃权能力全景实测

你有没有这种感觉——把一个看起来挺复杂的任务丢给 AI 智能体，它就一头扎进去开始调用工具，搜索、点击、试错、再搜索……它"看起来根本不知道什么时候该承认"这件事做不了"是常有的事。哪怕任务压根不靠谱，或者环境里压根没有它要找的东西，它也照样一通操作猛如虎，结果交给你一堆没用的中间状态。

这不是个别现象。Han Luo、Bingbing Wen、Lucy Lu Wang 在 arXiv:2606.28733 这篇《Agentic Abstention: Do Agents Know When to Stop Instead of Act?》里，把这件事系统地拆开来研究了。他们构建了一个 28,000 任务的基准（覆盖 Web 购物、终端操作、QA 三大场景），跑了 13 个主流 LLM 加 2 个 agent scaffold，结论相当扎心：

> 8 个 web 模型里有 6 个在 10 轮后 abstention recall 还不到 0.5；Terminal 两个 scaffold 都低于 0.5；QA 5 个模型有 4 个低于 0.5。**绝大多数智能体既不会及时停下，也不会在应该停下的时候停下。**

更颠覆的是几个反直觉的发现：
- **模型越大 ≠ 越会及时停下**。规模上去之后最终会停下来（overall recall 上去了），但"早停"的 timely recall 几乎纹丝不动
- **推理能力不是越多越好**。Qwen-3-235B 的 Thinking 版本比 Instruct 版 timely recall 更高，但 overall recall 反倒下降——推理让它"早判断"，却也让它"过早放弃"
- **Agent 框架比模型本身更重要**。同一个 GPT-5.4-mini，换个 scaffold（Codex CLI vs Terminus 2），Terminal 场景的 overall recall 从 0.18 拉到 0.38，差了一倍多

最后他们还提了一个叫 **CONVOLVE** 的上下文工程方法，只用 20 条轨迹就能把 Llama-3.3-70B 在 WebShop 上的 timely recall 从 **26.7 拉到 57.4**，overall recall 直接到 **100**——**完全不改模型参数，纯靠 playbook 蒸馏**。

我读完的感受是：这是一篇把"agent 知道自己几斤几两"这件事从经验直觉变成可量化研究的论文。它不教你怎么让模型变强，它教你**怎么让模型在该停的时候停下来**。下面我把它的核心设计、关键数据和我的判断拆开讲讲。

---

## 论文信息

- **标题**：Agentic Abstention: Do Agents Know When to Stop Instead of Act?
- **作者**：Han Luo, Bingbing Wen, Lucy Lu Wang
- **机构**：艾伦人工智能研究所（AI2）及相关合作机构
- **链接**：https://arxiv.org/abs/2606.28733
- **代码与数据**：https://lhannnn.github.io/agentic-abstention

---

## 为什么要重新定义"abstention"？

老实说，"模型知道自己不会"这件事在 LLM 时代并不新鲜——标准的 LLM abstention 已经被研究很多年了，评测标准也成熟。但作者认为，**agent 场景下的 abstention 是个完全不同的物种**。

**标准 LLM abstention** 是个单轮决策：拿到问题，要么答，要么弃答。判断标准清晰，benchmark 一把刷完。

**Agentic abstention** 是个序贯决策问题。作者把它形式化成了 POMDP：

- **状态空间** $\mathcal{S}$：包含任务是否可解这种不可直接观察的属性
- **动作空间** $\mathcal{A}=\{\text{ANSWER}, \text{ABSTAIN}, \text{ACT}\}$
  - **ANSWER**：终结动作，提交答案/下单/交方案
  - **ABSTAIN**：终结动作，明确告诉用户"我做不了"
  - **ACT**：非终结动作，调用工具、和环境交互
- **终止条件**：选 ANSWER/ABSTAIN，或者预算步数用完

关键区别在于：agent 在每一步都面对"继续搜、还是停下"的选择。而且"该不该停"这个信号，往往要交互了几步之后才会浮现——比如一开始指令看似可行，搜了一圈才发现环境里压根没这玩意儿。

一句话概括：**standard abstention 是"问"完再决定弃不弃；agentic abstention 是边"问"边决定，而且答案往往藏在环境里。**

下图给了一个具体的例子——Web 购物场景里，任务是"find and buy a pink living room pillow"（粉色的客厅枕头），但环境里其实没有粉色枕头：

![图1：Environment-based Abstention 的三种 agent 轨迹](https://arxiv.org/html/2606.28733v1/x1.png)

*图1：Environment-based Abstention 的三种 agent 轨迹对比。任务指令"Find and buy a pink living room pillow"，但环境里其实没有粉色枕头（Missing Target）。三种轨迹分别是：t1 搜了一下，t2 立刻发现搜不到，**及时弃权**（Timely success）；t1-t4 搜了 4 次才意识到不行，**延迟弃权**（Delayed success）；t1-t10 一直在搜，**10 轮里完全没弃权**（Failure to abstain）。*

这个图很好地说明了一件事：abstention 不仅是"会不会"的问题，还是"什么时候"的问题。一个 agent 哪怕最终能弃权，要是拖到第 8 轮才反应过来，中间那些无效的搜索、点击、token 消耗全是浪费——更糟的是用户体验直接崩了。

---

## 三大类弃权场景

为了让评估不空对空，作者把"agent 应该弃权"这件事细分成几个具体的子类型，覆盖了"指令本身就有问题"和"环境里找不到"两大根源。

**第一类：Request-based Abstention（基于请求的弃权）**——光看指令就知道做不了：
- **False Premise or Contradiction**：指令依赖错误假设或自相矛盾。比如"BPA-free 且 PVC-containing 的环保罐"——BPA-free 和 PVC-containing 本来就是对立属性
- **Subjective Preference**：成功依赖用户主观审美，比如"挑一款你觉得我会喜欢的口味的蛋白奶昔"
- **Underspecified Intent**：缺关键上下文，比如"买和之前一样颜色的裤子"——但没有"之前"的记录

**第二类：Environment-based Abstention（基于环境的弃权）**——指令看着能跑，跑了才知道不行：
- **Missing Target**（WebShop）：指令格式没问题，但环境里压根没这商品。作者的做法是**从商品目录里删掉目标商品，重建 Lucene 索引**，让 agent 必须搜到空结果才能反应过来
- **Missing Prerequisite**（Terminal）：环境里缺完成任务的必要资源——文件、依赖、权限、服务都被手动移除了

**第三类：QA 场景的 5 类弃权**——Answer Unknown、False Premise、Subjective、Underspecified Context、Underspecified Intent。这块是借用了已有 AbstentionBench 16 个数据集的标注。

这个分类有个细节我挺喜欢：作者没有简单二分"能/不能"，而是分了两轮——"看指令就知道"和"跑了才知道"——这正好对应了 agent 的两种典型失败模式：盲目开工和无效试错。后续主实验也证实了这种区分的必要性（后面会讲到）。

---

## 28,000 任务基准长什么样？

这套基准叫 **AbstentionBench**（准确说是 AbstentionBench subset），跨三大场景，**总计 28,000+ 任务**：

| 场景 | 数据源 | 任务数 | 可解 | 不可解 |
|------|--------|--------|------|--------|
| **WebShop** | WebShop 500 原始指令 | 1,000 | 500 | 500（249 Request + 251 Environment） |
| **Terminal** | Terminal-Bench 2.0（89 原始任务） | 277 | 89 | 188（87 False Premise + 80 Underspecified + 21 Missing Prerequisite） |
| **QA** | AbstentionBench（16 datasets） | 27,073 | 含可解与应弃权 | 5 类弃权场景 |

![图10：AbstentionBench 16 数据集的组成](https://arxiv.org/html/2606.28733v1/x10.png)

*图10：AbstentionBench 子集的构成。左图是 16 个数据集的样本量（ALCUNA、BBQ、UMWP、KUQ 等），右图是按弃权场景聚合的分布。可以看到 **Underspecified Context 占大头（69.9%）**，False Premise 只有 9.9%。*

评测覆盖了 **13 个 LLM-as-agent 系统**（GPT-5.4-mini、Grok-4-Fast、Llama-3.3-8B/70B、GPT-OSS-120B、MiniMax-M2.5、Qwen3-8B/14B/32B/235B Instruct & Thinking、Gemma-4-31B、GLM-5.1）和 **2 个 agent scaffold**（Terminus 2、Codex CLI）。

**指标设计**也相当讲究，作者没只看一个数：
- **AbsRec@K**：弃权成为正确选择后的 K 步内正确弃权的实例比例。K 越大越宽松
- **Timely Recall（AbsRec@1）**：最早可弃权步就弃权
- **Overall Recall（AbsRec@10）**：最大步数（K_max=10）内能弃权
- **SPL**：成功弃权按路径长度归一化（延迟弃权要被惩罚）
- **Over-abstention rate**：在可解任务上错误弃权的比例

这套指标体系是这篇论文另一个我觉得值钱的地方——它把"会不会弃权"和"会不会过早弃权"拆开了量化。后面的实验你会看到，**这两件事常常是 trade-off**。

---

## 主实验：13 个模型在 AbstentionBench 上集体翻车

我直接上图说话吧。先看 Abstention Recall 随 K 变化的全景图：

![图3：Abstention Recall 随 K 变化（Web/Terminal/QA 三场景）](https://arxiv.org/html/2606.28733v1/x3.png)

*图3：Abstention Recall 随 K 变化的三场景全景图。横轴 K 是弃权窗口（1-10 步），纵轴是 recall。Web 场景下 Llama-3.3-70B（红线）一支独大，K=10 接近 0.83；其他大部分模型在 10 步后都还卡在 0.2-0.5 之间。Terminal 两个 scaffold 都不行，最高的 Codex CLI 也就 0.38。QA 里 Qwen3-235B（橙线）一路领先，K=10 接近 0.71。*

几个我看完数据后印象深刻的点：

**1. Timely abstention 是真正的硬骨头。** 看所有曲线在 K=1 那一列，几乎都贴着底部——**绝大多数 agent 在最早可弃权点都判断不出来**。Web 场景下 8 个模型在 K=1 时 timely recall 最高也就 0.27（Llama-3.3-70B），其他基本在 0.0-0.1 之间。这意味着绝大多数 agent 都要"先错几轮再说"。

**2. 模型规模和 abstention 能力不是线性关系。** 8B → 14B → 32B → 235B 这一路上去，**overall recall 是涨的，但 timely recall 基本没动**。说得直白点：模型变大会让你"最终能停下来"，但不会让你"更早停下来"——它还是要把环境榨干到最后一刻才肯认输。

**3. Reasoning（推理）是把双刃剑。** Qwen-3-235B 的 Thinking 版本比 Instruct 版本的 AbsRec@1 更高（timely recall 改善），但 AbsRec@10 反倒下降。意思是：推理让它"早判断"，但也让它更容易"过早放弃"。Terminal 场景里 GPT-5.4-mini 的 medium reasoning effort 是最佳折中，high reasoning 没进一步收益。

**4. 推理还能缓解 over-abstention（过度弃权）。** Web 场景下 Qwen3-235B-Instruct 在第 10 轮 over-abstention 率升到 34%，Thinking 版本降到 24 个点——多推理一步让它在可解任务上不那么"神经质"。

![图6：Web 和 Terminal 场景下 over-abstention 累积率随 turn 变化](https://arxiv.org/html/2606.28733v1/x8.png)

*图6：Over-abstention 累积率随 turn 变化。Web 场景下 Qwen3-235B-Instruct（橙）一路爬到 0.34，Thinking 版（蓝）控制在 0.24。Terminal 场景各 reasoning 等级都很低（最高 0.08）。说明"长交互"是过度弃权的主要温床，推理可以缓解但消除不了。*

按 abstention 类别拆开看更有意思：

![图4：Web 场景各 abstention 类别下的 AbsRec@K](https://arxiv.org/html/2606.28733v1/x4.png)

*图4：Web 场景四个 abstention 类别的 AbsRec@K 对比。从左到右：Missing Target、Subjective Preference、Underspecified Intent、False Premises。Llama-3.3-70B（红线）在所有类别都领先，但**Missing Target 上其他模型几乎全趴下**——K=1 时基本都在 0 附近。False Premises 最容易，连小模型都能早期弃权，因为错误前提通过推理就能识别。*

**最难啃的骨头是 Missing Target**——指令看着没毛病，但环境里没货。这正是 agentic abstention 区别于 standard abstention 的核心：环境的不确定性是 novel 的，光靠理解指令不够，必须和搜索结果对账。

最容易的是 False Premises——错误前提通过推理就能识别（"BPA-free 和 PVC-containing 矛盾"），不用看环境。

---

## Scaffold 比模型更重要（这点我真没想到）

Terminal 场景有个对比让我挺意外：**同一个 GPT-5.4-mini，换个 scaffold 效果差一倍多**。

![图12：Terminus 2 vs Codex CLI 在 Terminal 场景的 abstention 表现](https://arxiv.org/html/2606.28733v1/x11.png)

*图12：同样基于 GPT-5.4-mini，Terminus 2（橙）和 Codex CLI（蓝）在 Terminal 场景的对比。左：改写指令的 overall recall，Codex CLI 10 轮后到 0.32，Terminus 2 只有 0.12。中：按类别拆，Underspecified Intent（虚线）两者都吃力，但 Codex CLI 还是明显好一截。右：延迟任务（环境没反馈必须猜的）两者都不错，Codex CLI 10 轮 0.85，Terminus 2 0.66。*

**Codex CLI 在几乎所有维度都碾压 Terminus 2**——overall recall 差 2 倍多，Underspecified Intent 也是 Codex CLI 完胜。

这件事的含义比数字本身大：它说明**abstention 能力不只是模型属性，更是"agent 如何和环境交互"这个系统级问题**。一个好的 scaffold 会把"该停下"的信号更好地组织到 prompt 里、把不必要的工具调用截断在中间层。这给工业界的启示是：**别只盯着换模型，agent 框架的 prompt 结构和工具调用策略可能比 scale 升级更划算**。

---

## CONVOLVE：20 条轨迹把 timely recall 翻倍

最后讲 CONVOLVE——这是论文给我的最大惊喜，因为它**不改模型参数**，纯靠上下文工程。

**核心思想**：把 agent 完整的失败/成功交互轨迹喂给一个 reflection model，让它分析"什么时候该停、什么时候停晚了"，再让一个 curator 把这些反思凝练成可复用的 **playbook**（规则手册），后续推理时把 playbook 塞进 prompt 就行。

**三个步骤**：

1. **环境交互 + 轨迹记录**：每个 episode 跑完，记下指令、观察、动作、终止原因、奖励、abstention 步骤、是否及时
2. **反思**：reflection model 分析"agent 是否在任务已不可行后还在行动？哪些观察触发了 abstention？哪些动作延迟了？"
3. **Playbook 更新**：curator 把反思凝练成结构化条目，按 section 组织（不匹配现有 section 的归"other"），通过 add 操作更新

公式上可以写成 $c^{(k+1)} = \mathcal{U}(c^{(k)}, \tau^{(k)}, y^{(k)})$，curator 在 playbook 预算（80K tokens）和反思输入（6K tokens）约束下做上下文管理。

**最关键的是训练数据只用 20 条轨迹**——基本是 few-shot 量级。

效果呢？**在 WebShop 上直接把 Llama-3.3-70B 的 timely recall 从 26.7 拉到 57.4，整体 recall 到 100**：

| 方法 | AbsRec@1 (Timely) | AbsRec@10 (Overall) | SPL |
|------|------|------|------|
| Llama-3.3-70B（base） | 26.7 | 83.2 | 55.3 |
| + In-Context Learning | 55.1 | 97.0 | 77.2 |
| **CONVOLVE（70B+70B）** | **57.4** | **100.0** | **78.9** |
| 70B 模型 + 8B 生成的 playbook | 55.3 | 99.0 | 76.4 |

最后一行是另一个让我惊讶的发现：**8B 小模型生成的 playbook 应用到 70B 大模型上，效果几乎和 70B 自己生成的 playbook 一样**（timely 55.3 vs 57.4，overall 99.0 vs 100）。说明 **CONVOLVE 的价值不依赖反思模型的能力**，蒸馏出来的 stopping rules 有相当的通用性。

**在其他基准上效果依然显著**：
- AbstentionBench：Llama-3.3-70B + CONVOLVE 的 AbsRec@1 从 25.0 涨到 39.7，SPL 从 33.2 涨到 47.5
- TerminalBench：AbsRec@1 从 37.6 直接拉到 68.9，提升 **31.3 个点**，是所有基准里最大的

CONVOLVE 的设计我读完其实有点感慨：它骨子里和"reflection / self-critique"那一脉方法论一脉相承，但关键差别在于：它不靠模型自己生成 reflection，而是把 reflection 显式外化成了可读、可调试、可迁移的 playbook。这在工业界特别实用——你可以把 playbook 拿出来人审、迭代、版本化，不用每次重新跑训练。

---

## 我的判断

这篇论文值得花时间读，但它不是没槽点。我把我觉得**最值钱的地方**和**最该警惕的地方**都说一下。

**最值钱的三个点**：

1. **把 "abstention" 这件事从"回答或弃答"重新定义成序贯决策问题**——这是概念级的贡献。所有做 agent 评估的人都会被影响
2. **28K 任务的 AbstentionBench + 13 模型 + 2 scaffold 的横评**——给出了一个扎实的基线，未来做 agent 评估绕不开它
3. **Model scale、reasoning、scaffolding 三因素的解耦分析**——发现"规模不改善 timely recall"、"推理是把双刃剑"、"scaffold 影响比模型大" 这三个反直觉结论，是任何做 agent 落地的人都需要记住的

**最该警惕的点**：

1. **CONVOLVE 只在 WebShop 上做了端到端验证，AbstentionBench/TerminalBench 上只给了 final recall，没展示 playbook 迭代过程和反思质量**。20 条轨迹够不够、reflection model 的质量会不会是瓶颈、playbook 会不会随 episode 变臃肿——这些工程问题论文没展开。如果你要在生产环境用 CONVOLVE，还得自己做几轮 ablation
2. **评测的模型列表里有些非常新的/小众的模型**（比如 MiniMax-M2.5、GLM-5.1），这些模型可能 agentic 能力本来就没优化好，不代表 abstention 一定差。要复现的话建议先看具体 scaffold 配置
3. **Missing Target 这个类别的构造方式**（从商品目录里删掉目标商品并重建 Lucene 索引）这种做法本身要打问号——真实用户场景里"环境里没货"通常伴随着很多杂音（推荐相似商品、相关搜索建议），比纯 Lucene 搜索空结果复杂得多

**对工程实践的启发**：

- 如果你也在做 agent 框架，**最该优化的可能不是基础模型，而是 scaffold 里的"该停就停"信号**——把环境反馈的"搜不到"、"权限不足"、"文件不存在"这些信号在 prompt 里强化，把"已尝试 3 次未果"这种累计状态显式化
- **ReAct 风格的纯推理对 abstention 不够**，需要把"我应该放弃"作为一种可被显式触发的动作，而不是隐式涌现
- **CONVOLVE 的思路可以马上借鉴**——即使你不用它的 reflection-curator pipeline，也可以做"hard-negative playbook"：收集 100 条"agent 失败地坚持了 N 轮"的轨迹，把教训写进 system prompt，单这一招可能就能救 10-20 个点的及时性

最后一句话总结：**这篇论文告诉你，agent 的"刹车系统"不是免费的——它需要专门的设计、专门的评测、专门的优化。** 比起"让模型更会做"，"让模型知道什么时候不做"可能是 agent 走向可靠部署更关键的一步。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
