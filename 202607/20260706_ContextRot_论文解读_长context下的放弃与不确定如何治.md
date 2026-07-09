---
title: "长 context 一拉长，LLM 就开始"摆烂"——Context Rot 论文解读"
date: 2026-07-06
arxiv: 2606.29718
authors: "Shijie Xia, Yikun Wang, Zhen Huang, Pengfei Liu"
affiliations: "Shanghai Jiao Tong University, Fudan University, SII, GAIR"
code: https://github.com/GAIR-NLP/ContextRot
subjects: cs.IR, cs.AI, cs.CL
---

# 长 context 一拉长，LLM 就开始"摆烂"——一篇论文讲透 deep search 里的 context rot

你有没有发现，部署一个 deep search 智能体，简单的 query 它挺利索，一旦问题变复杂、轨迹拉到 100K tokens 以上，模型就开始"撂挑子"——要么直接说"unable to determine"，要么给一个看起来挺像回事但其实没把握的答案？

这种现象不是 bug，是 bug 之前的一个更隐蔽的问题。GAIR 团队（上海交大 SII + 复旦）最近的一篇论文 **"Diagnosing and Mitigating Context Rot in Long-horizon Search"**（arXiv:2606.29718）就专门盯着这件事，把"context rot"这个词从一个含糊的体感问题，落成了一个有明确定义、可量化、可缓解的工程问题。

## 一句话摘要

> 在 4 个开源旗舰模型 × 3 个 deep search benchmark 上，作者发现 rot 的真正表现不是"答错了"，而是"模型在长 context 下越来越倾向于直接放弃或给出不确定的答案"；通过剪枝实验证明 rot 与 context **内容**直接相关，并通过系统对比 7 种 context management 方法 + 后验拒采，给出了一份**可落地的策略选择清单**。

我的第一反应是：这事其实大家都遇到过，但没人把它的形态分得这么清。"context rot" 这个词在 Anthropic、谷歌的 blog 里都出现过，但论文作者给了一个非常硬的工程化定义——**terminal state 分类 + struggle score 量化**——这才是这篇论文最值钱的地方。

---

## 论文基本信息

- **标题**：Diagnosing and Mitigating Context Rot in Long-horizon Search
- **作者**：Shijie Xia¹, Yikun Wang², Zhen Huang², Pengfei Liu¹
- **机构**：¹ Shanghai Jiao Tong University, SII, GAIR　² Fudan University, SII, GAIR
- **arXiv ID**：2606.29718　　**链接**：[arXiv:2606.29718](https://arxiv.org/abs/2606.29718)
- **代码**：https://github.com/GAIR-NLP/ContextRot
- **领域**：cs.IR / cs.AI / cs.CL

---

## 一、Context rot 到底是什么？以前的认知有什么盲区

context rot 不是一个新词。Lost-in-the-middle、NoLiMa、RULER 这些工作都讲过——把一个关键信息塞到长 context 的中间，模型就找不到了；或者塞点无关内容进去，模型就分心了。

但这些研究有一个共同点：**都是单轮**——把一个长 prompt 一次塞给模型，看它怎么反应。

而真实的 deep search 智能体完全不是这么工作的。它是**多轮、多源、渐进式积累**的：

```
(r1, T1, o1), (r2, T2, o2), …, (rk, Tk, ok)
```

每一步里，模型先思考，然后调工具（search / visit），拿到观察结果，再思考，再调工具……一直这么滚下去。几十甚至上百次 search 调用下来，context 里塞满了：工具返回的网页摘要、模型自己写过的 reasoning 段落、中间可能调皮的搜索词、还有早期那些"后来被发现没用"的搜索结果。

那这种**真实的、长程的、混合 reasoning 和 observation 的 context**会不会 rot？以什么形式 rot？怎么缓解？

这是这篇论文要回答的问题。

---

## 二、诊断：把 rot 拆成四种 terminal state

作者的第一招是**给出可量化的 rot 定义**。他们没沿用传统的"答对/答错"二元分类，而是把"答案形态"和"reasoning 形态"绑在一起，定义出四种 terminal state（用 GPT-OSS-120B 当 judge，重复 5 次投票；与人工标注 98.7% 一致）：

| 类别 | 解释 | 模型是否在"放弃" |
|------|------|--------|
| **Confident Correct (CC)** | 给答案，且 reasoning 显示满足所有约束 | — |
| **Confident Incorrect (CI)** | 给答案，且 reasoning 显示满足所有约束（但答错了） | — |
| **Uncertain Incorrect (UI)** | 给答案，但 reasoning 里写"我没法完全验证" | 半放弃 |
| **Give Up (GU)** | 明确说"找不到/无法确定"，不给答案 | 完全放弃 |
| **No Answer (NA)** | 跑到 turn 上限也没给出答案 | 资源耗尽 |

**rot 现象 = UI + GU 占比随 context 长度上升**——这是论文后续所有实验的"rot 指标"。

这个定义很关键，因为它把"rot"从一个主观感受变成了一个可计数的比率。读到这里我得说，作者是懂工程的——他们没有去搞一个"rot score"神经网络，直接复用 LLM-as-a-judge 加 5 票多数，工程上最容易复现，也最容易扩展。

下图是论文最核心的 overview 之一：trajectory 越长，Confident Correct 比例断崖式下跌，Give Up 和 Uncertain Incorrect 一起抬升。

![图1：长 context 导致 rot 现象的概览。Confident Correct 比例从 16K 时的近 95% 掉到 64K+ 时的不到 20%，Give Up 和 Uncertain Incorrect 反向上升](https://arxiv.org/html/2606.29718v1/x1.png)

*图 1：rot 现象概览。在一个 BrowseComp 例子上，模型从"几乎全对"被推到"主要在 give up / uncertain incorrect"。右上和右下两个对话框就是这两种典型 terminal state 的形态。*

---

## 三、四个模型 × 三个 benchmark：rot 是不是真的普遍？

实验用了 4 个开源旗舰模型（GLM-4.7、GLM-5.0、Qwen3.5-397B-A17B、MiniMax-M2.5）和 3 个数据集：

- **BrowseComp**（web 搜索）
- **BrowseComp-Plus**（local corpus 检索）
- **xbench-DeepSearch**（web 搜索）

每个实验跑 5 次，最多 100 turn。

**Table 2**（去掉置信类型，只看错误分布）大致长这样：

| 模型 | BrowseComp UI+GU | BrowseComp-Plus UI+GU | xbench-DeepSearch UI+GU |
|------|---|---|---|
| Qwen3.5-397B-A17B | 53.4% | 24.0% | 20.4% |
| GLM-4.7 | 58.2% | 26.0% | 17.0% |
| GLM-5.0 | 44.8% | 15.3% | 14.4% |
| MiniMax-M2.5 | 53.4% | 26.3% | 22.2% |

这数字挺刺眼的——**没有任何一个模型能在长 trajectory 下保持 80%+ 准确率**。即便是最强的 GLM-5.0，在 BrowseComp 上也有一半问题会以"放弃"或"不确定"收场。

把 trajectory 长度按 16K 一档切片，看不同区间的 terminal state 分布：

![图 2：4 个模型 × 3 个 benchmark 上 terminal state 随 trajectory 长度的变化。Confident Correct 曲线在所有 12 个子图里都一致地随长度下滑，UI/GU 反向上升](https://arxiv.org/html/2606.29718v1/figure/length_label_subplot.png)

*图 2：4 个模型 × 3 个 benchmark 上 terminal state 随 trajectory 长度的变化。绿色（CC）随长度上升单调下跌，黄色（UI）和橙色（GU）反向上升——rot 不是某个模型的特殊问题，而是普遍现象。*

从这张图里能读出几个关键事实：

1. **Context window 本身不是瓶颈**。Table 2 中所有数据集的 NA 都接近 0——模型压根没跑满 context，是它自己"早退"了。
2. **rot 跟 context 长度强相关，跟数据集难度不是强相关**。BrowseComp-Plus 上模型正确率更高（数据相对简单），但 UI+GU 比例跟 BrowseComp 几乎相当——rot 看的是"agent 在长 context 下的行为模式"，不是"问题难不难"。
3. **"give up" 还是 "uncertain" 是 model-dependent 的**。GLM-4.7 更倾向 give up，Qwen3.5 更倾向 uncertain incorrect。这条信息对后面做 model routing 有用。

作者又做了一步很聪明的分析——给轨迹标一个 **struggle score**（reasoning 阶段出现"还是不行"、"换条路"、"我卡住了"这类反复失败话语的比例，judge 与人工 91.4% 一致）。结果在 BrowseComp 上：

![图 3：BrowseComp 上 struggle score 与 terminal state 的关系。struggle score 越高，UI 和 GU 占比越高，CC 占比越低](https://arxiv.org/html/2606.29718v1/figure/struggle_behavior_subplots.png)

*图 3：BrowseComp 上 struggle score 与 terminal state 的关系。struggle score 越高（横轴越往右），UI（黄）和 GU（橙）占比越高——"语义层面"已经在挣扎的轨迹，最终基本就是 rot 收场。*

这告诉我们：rot 不只是 token 数字的函数，它有**语义信号**。一个能检测 struggle 模式的系统，理论上能比单纯看 token 长度更早地预测"这条轨迹要烂了"。这个观察直接影响了后面"语义触发 summary"的设计。

---

## 四、剪枝实验：rot 真的是 context 本身造成的吗？

作者这一步我认为是全文设计得最巧的一处。**如果 rot 是 context 长度造成的，那把 context 整个删掉，rot 应该消失**。他们做了四种"剪枝"：

- **N/A**：原始 ReAct 轨迹，啥也不删
- **Tool**：只删掉工具响应，保留 reasoning
- **Reason.**：只删掉 reasoning，保留工具响应
- **All**：整个累积 context 全部丢掉，只保留最近 3 轮

关键对比在图 4：

![图 4：四种剪枝策略下 terminal state 分布。All 几乎消灭了 UI 和 GU，但 NA（no answer）爆表——模型失去了所有"前情提要"，根本走不到终点](https://arxiv.org/html/2606.29718v1/figure/qwen_perturbation_subplot.png)

*图 4：四种剪枝策略下 terminal state 分布（Qwen3.5 / GLM-4.7 在三个数据集上）。蓝色斜线 N/A 是 baseline，绿色 Reason.、黄色 Tool 都在不同程度上缓解 rot，但红色 All 几乎把 UI 和 GU 砍到 0。代价看右边的 NA 柱：All 策略的 NA 飙到 20%-45%。*

这段数据非常关键地讲清楚了**两件事**：

1. **rot 跟 context 长度不是简单线性关系**。Table 3 里"Reason."的 trajectory 比 N/A 还长（删了 reasoning 后模型需要更多 search 来弥补），但 rot 反而更低。所以"rot 是因为 context 太长"这个朴素直觉是错的，**rot 跟 context 的内容有关**。
2. **rot 是被 context 推出来的，但粗暴丢掉 context 不可行**。All 策略 rot 几乎归零，但 NA（没答完）爆表到 20-45%——模型丢了所有累积信息，找不着北。

这是个非常工程化的洞察：rot 是个"该被治"的病，但**治疗方案要在"保留信息"和"避免 rot"之间找平衡**，不是单纯把 context 砍掉。

---

## 五、缓解方案 1：七种 context management 方法怎么选？

作者把 7 种方法归到三大类（覆盖了主流的工业方案）：

| 类别 | 触发方式 | 方法 |
|------|---------|------|
| **Context compaction** | 触发后把历史重写成 summary | Summary (Length)：长度超阈值就 summary<br>Summary (Turn)：轮数超阈值就 summary<br>Summary (Semantic)：struggle score 达 0.5 就 summary |
| **Context trimming** | 触发后直接丢 token | Discard：超阈值就把所有 tool 响应全丢<br>Keep Latest：始终保留最近 3 轮，丢更早的<br>Keep Latest (w/ sum.)：Keep Latest + 长度触发 summary |
| **Context isolation** | 把任务挪出 active context | FoldAgent：主 agent 通过 tool call 调子 agent，子 agent 只回 summary |

Table 4 给出了所有方法在三个数据集上的 Acc / # Tool / Rot / NA。我把 Overall（三个数据集平均 Acc）这一列抽出来最直观：

| 方法 | Qwen3.5 Overall | GLM-4.7 Overall | 性质 |
|------|---|---|---|
| ReAct (baseline) | 54.4 | 51.8 | — |
| Summary (Length) | 59.3 | **60.4** | 主动型 compactor |
| Summary (Turn) | 60.7 | 58.2 | 轮数触发 compactor |
| Summary (Semantic) | 59.3 | 60.3 | 语义触发 compactor |
| Discard | 60.3 | 59.5 | 暴力丢弃 |
| Keep Latest | 60.1 | 59.9 | 滑动窗口 |
| **Keep Latest (w/ sum.)** | **62.9** | 60.2 | 混合策略 |
| **FoldAgent** | **64.9** | 57.5 | 子 agent 隔离 |

三个能直接拿到答案的关键 finding：

1. **没有 silver bullet**。7 种方法没有一种在所有模型 × 所有数据集上都是最优。**Keep Latest (w/ sum.)** 是综合最稳的（混了 trimming + compaction），FoldAgent 在强模型上拔得头筹但在 GLM-4.7 上反而垫底。
2. **context compaction 的代价不容忽视**。Summary (Length) 在 BrowseComp 上把 tool calls 从 21.7 拉到 57.7（接近 3 倍）。这是用"钱"换"准确度"的典型选择。
3. **FoldAgent 高度依赖基座模型能力**。Qwen3.5 + FoldAgent = 64.9% 拿下 Overall 第一，但 GLM-4.7 + FoldAgent = 57.5% 是它的最低值。原因是 FoldAgent 让主 agent 决定什么时候调子 agent、怎么拆任务——这本身需要 agent 有强元认知能力。如果主 agent 都不太会调度，sub-agent 隔离只会更糟。

**Threshold sensitivity**（Table 5）里还有一个工程上很实用的发现：把 Summary (Length) 的触发阈值从 64K 压到 32K，rot 从 20.2% 降到 2.4%，准确率从 45.0% 涨到 46.6%——但 tool calls 从 41.7 涨到 57.7。**触发的越频繁，rot 越低，但成本越高**。这条 trade-off 实际部署时需要根据 SLA 选。

---

## 六、缓解方案 2：后验拒采——一种"不动 context"的退路

context management 是"在线"操作，要在 trajectory 跑的时候决定怎么处理 context。还有另一条路——**跑完之后再挑**。这就是 rejection sampling。

做法很直接：对每个 query 跑多条 trajectory，然后用某种规则选一条作为最终答案。作者设计的规则叫 **rot-aware filter**：

1. 让 GPT-OSS-120B judge 每条 trajectory 的 terminal state（CC / UC / CI / UI / GU / NA）
2. **把 UI 和 GU 滤掉**，只保留 confident answer 的 trajectory
3. 在剩余 trajectory 上做聚合（三种方式：FT = 选最少 turn、FL = 选最短 trajectory、MV = 多数投票）

Table 6 的结果（取 Overall 那一列做对比）：

| 模型 | FT | FL | MV | + Filter FT | + Filter FL | + Filter MV |
|------|---|---|---|---|---|---|
| Qwen3.5 | 63.1 | 65.5 | 64.9 | **68.0** | **68.8** | **69.4** |
| GLM-4.7 | 62.1 | 63.5 | 64.0 | **66.2** | **66.7** | **66.6** |

**+ Filter 之后三个聚合方法都涨 2.6% 到 4.9 个点**。在 BrowseComp 和 BrowseComp-Plus 这种 rot 严重的数据集上提升更大（BrowseComp Qwen3.5 FT 从 54.0 涨到 61.7，+7.7 个点）。

这个发现的工程价值很大——**它不需要改模型、不需要改 prompt、不需要动 context，只需要在最后一道"门"里多一个 LLM judge**。对于已经部署的 deep search 系统，这是一个零侵入的优化。

**filter 的精度验证**（Table 10）也很有说服力：用"confident answer"作为正确性的代理，在 BrowseComp-Plus 上 precision 0.981、recall 0.796——**模型说"我确定"时，绝大多数时候确实是对的**。这是 filter 能 work 的根本原因。

---

## 七、组合拳：context management + rejection sampling

最后一步，把两条线合起来。context management 减少每条 trajectory 的 rot，rejection sampling 在结果层面再筛一遍。

图 5 展示了 ReAct + 各种 context management 方法 + 拒采的 acc 随 tool calls 的变化曲线：

![图 5：ReAct + 各种 context management + rejection sampling 的 cost-accuracy 曲线](https://arxiv.org/html/2606.29718v1/figure/tool_call_acc_scaling_subplots.png)

*图 5：Qwen3.5（上行）和 GLM-4.7（下行）在 3 个数据集上的 cost-accuracy 曲线。x 轴是 tool calls，y 轴是准确率。曲线越靠左上越好。可以看到 Keep Latest (w/ sum.) + MV（橙色）和 FoldAgent + MV（蓝色）通常是 Pareto 前沿上的点。*

几个观察：

- **BrowseComp / BrowseComp-Plus（rot 严重）**：best context management + 拒采 > ReAct + 拒采。context management 在重灾区能"治本"，rejection sampling 在"治标"上再做一次过滤。
- **xbench-DeepSearch（rot 较轻）**：ReAct + 拒采 跟 最好的 context management + 拒采 **基本打平甚至略胜**。因为 xbench 上 rot 比例本来就低（10-15%），context management 的边际收益小，反而是直接多 sample 几条更划算。
- **每多 sample 1 次的边际收益递减**。曲线在 tool calls 翻倍的后段趋于平缓。

---

## 八、我的判断

这篇论文不是又一个"context 太长所以模型变笨"的现象描述。**它把 context rot 工程化了**——定义了四种 terminal state、定义了 struggle score、给出了 7 种 context management 的横向对比表、外加一个零侵入的 rejection sampling 后验方案。这些东西都直接能拿到 deep search 智能体里用。

几个我自己想强调的点：

**第一，"context rot" 的真实表现是"放弃"和"不确定"，不是"答错"**。这点在 Table 2 上非常清楚——长 trajectory 下的 CI（Confident Incorrect）比例远低于 UI + GU。**这意味着传统的 accuracy-only 评估会严重低估 rot 的影响**——它把"模型主动放弃"和"模型自信答错"混在一起计入了"错误"，但前者其实是个更严重的失败信号。**用"rot 指标（UI + GU 比例）"作为副指标来监控 deep search 系统，是这篇论文最值得工程团队抄走的实践**。

**第二，rot 跟 context 长度不是简单线性，跟 context 的内容直接相关**。"Reason."剪枝后 trajectory 反而更长但 rot 更低这个反直觉结果很关键——它意味着**"什么时候砍"和"砍什么"比"砍多少"更重要**。这也解释了为什么"语义触发的 summary"能 work：它在模型已经陷入 struggle 模式时才动手，而不是机械地按 token 阈值。

**第三，没有银弹**。Keep Latest (w/ sum.) 是综合最稳的，FoldAgent 在强基座上最强，Summary 系列在中等基座上性价比高。**对工业团队的建议是：基座弱的，先上 Keep Latest (w/ sum.)；基座强（Qwen3.5 这种 397B 量级）的，可以试 FoldAgent**。直接堆 summary 阈值不是好方案——成本翻 3 倍但准确率只涨 1-2 个点。

**第四，rot-aware filter 是个"零侵入"的工程利器**。它不需要改模型 / prompt / context 策略，只需要在最后加一道 LLM judge。如果你的 deep search 系统已经在做 rejection sampling（多数投票、最短选优），加一个"先 filter 掉 give up 和 uncertain"的前置步骤，**平均涨 2.6-4.9 个点**。这种"白捡"的优化在工业里特别有价值。

**第五，我对作者没展开讨论的两个点有点小疑问**：
1. **rot 是不是模型训练的副产物**？论文主要在"behavior"层面描述，没讨论是不是某个 RLHF / SFT 阶段放大了 rot。如果基座模型在长 context 下都 rot，那 post-training 的着力点应该改；如果是 post-training 引入的，方向又不同。
2. **rot-aware filter 的 judge 是 GPT-OSS-120B**——它本身也可能 rot。如果把 judge 也换成 rot 的模型，filter 的 reliability 会怎么变？这个问题论文没回答。

**第六，论文的 Limitations 也得提一下**：只测了 4 个开源模型（闭源加密了 reasoning，看不到）、只在 deep search 场景验证。其他长程 agentic 任务（软件工程、code agent）rot 是不是同形态，不确定。

但不管怎么说，这篇论文的工程价值已经能直接落地了——光那 7 种 context management 方法的对比表和 rot-aware filter 的设计，就值回票价。**如果你们团队也在做 deep search / long-horizon agent，建议先按 Table 4 的方法名 + Table 6 的 filter 设计搭一个 baseline**。剩下的优化可以后面慢慢加。

---

## 附：方法速查表

| 类别 | 方法 | 关键 trigger | 适合场景 |
|------|------|------------|---------|
| Compaction | Summary (Length) | token 长度超阈值 | 资源充足、追求稳定 |
| Compaction | Summary (Turn) | turn 数超阈值 | 节奏可控 |
| Compaction | Summary (Semantic) | struggle score 达 0.5 | 想"治未病"——症状出现前就动手 |
| Trimming | Discard | 长度超阈值 | 极致省 token |
| Trimming | Keep Latest | 滑动窗口（最近 3 轮） | 简单稳 |
| Trim+Compact | **Keep Latest (w/ sum.)** | 双触发 | **综合最稳的混合方案** |
| Isolation | FoldAgent | 主 agent 主动调用 | **强基座模型专用** |
| Post-hoc | **Rot-aware Filter + 拒采** | 在结果层过滤 UI/GU | **零侵入优化，所有方案都能套** |

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
