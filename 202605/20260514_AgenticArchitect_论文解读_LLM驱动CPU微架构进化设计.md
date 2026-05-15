# Agentic Architect：让 LLM 帮你"进化"出 CPU 微架构——预取器跑赢 SMS 21%

## 核心摘要

体系结构这行业最痛苦的事情之一：**设计空间太大、人工 sweep 太慢**。Cache 替换策略、prefetcher、branch predictor 这些子系统的微观策略，几十年来靠的是人工的"设计-实现-仿真-分析-迭代"循环——一个有经验的架构师可能花两年把 cache 替换策略从 LRU 推进到 RRIP，再花两年推进到 Mockingjay。

CMU 这篇 Agentic Architect 想把这个循环"AI 化"：**LLM 驱动代码进化 + cycle-accurate 仿真**做评估。架构师定义优化目标、seed 设计、scoring function、simulator 接口、benchmark 划分，**LLM 在这些约束里探索具体实现代码**。整套框架是模块化的，evolutionary backbone、LLM、simulator 都可以换。

结果让人意外的"扎实"：
- **Cache replacement**：1.062× IPC over LRU，**比 Mockingjay 还多 0.6%**——这可是十年研究累积出来的 SOTA
- **Branch prediction**：1.100× over Bimodal，比 Hashed Perceptron seed +1.5%
- **Data prefetching**：1.76× over no-prefetch，**比 VA/AMPM Lite seed +17%、比 SMS +21%**

但最值钱的不是这些数字，而是论文的几个清醒结论：**evolved 设计的组件大多是已知技术，新意在"怎么协调它们"；seed 质量 bound 了搜索上限**；prompt 设计比 LLM 选择更重要；最反直觉的——**minimal prompt 比 prescriptive prompt 效果更好**。

一句话评价：这是 LLM-driven scientific discovery 在系统结构领域的一次扎实落地，**给"AI for chip design"提供了一个端到端开源框架**。

---

## 论文信息

- **标题**：Agentic Architect: An Agentic AI Framework for Architecture Design Exploration and Optimization
- **作者**：Alexander Blasberg, Vasilis Kypriotis, Dimitrios Skarlatos
- **机构**：Carnegie Mellon University
- **日期**：2026/04/28
- **arXiv**：https://arxiv.org/abs/2604.25083

![图1：Agentic Architect 整体框架与进化循环](https://arxiv.org/html/2604.25083v1/x1.png)

*图1：人类架构师提供 system prompt、seed policy、evaluator/score function、benchmark split（图中所有 human icon 标的输入）；LLM 在这些约束下生成候选实现；候选代码进入 cycle-accurate simulator 评估；评估结果回到 LLM 指导下一轮 evolution。这是一个标准的 LLM-driven evolutionary loop，但每个轮次的 fitness 是真的拿仿真器跑出来的 IPC，不是某种 surrogate。*

---

## 问题动机：为什么是微架构？

说实话我做过一段时间 CPU 微架构相关的事，对那种"调一个 RRPV 阈值跑三天 simulation"的工作流深有体会。这种工作有几个特点：

1. **搜索空间巨大且离散**：cache 替换策略一个 PC-based predictor 的设计空间随便就是 2^N 的组合
2. **每次评估很贵但准**：cycle-accurate simulation 慢，但跑完了 IPC 就是 ground truth
3. **设计天然是模块化代码**：cache replacement 一段函数、prefetcher 一段函数，很适合 LLM 生成
4. **领域有成熟 benchmark**：SPEC、PARSEC、CloudSuite 等——给 RL/evolutionary search 提供了相对公平的评估底座

论文的判断特别清晰：**这四个性质让微架构成为 LLM-driven discovery 的理想 target**。

我尤其认同第 3 点。同样的进化范式如果用在"生成芯片版图"或者"生成 RTL 时序约束"上，复杂度数量级会高很多。微架构这种"几百行 C++ policy 函数"的颗粒度，正好是 LLM 当前能稳定 hold 住的尺度。

---

## 方法核心：进化 + 仿真 + 人机协作

### 框架的"co-design"立场

论文反复强调一点：**不是要 LLM 取代架构师，是要 LLM 加速架构师的搜索**。

```
Human Architect:           LLM Agent:
  - 优化目标               - 生成候选代码  
  - seed policy            - 调用 evaluator
  - scoring function       - 根据反馈 mutate / crossover
  - simulator 接口          
  - benchmark split        
```

这种划分让我特别有共鸣——它跟我们之前讨论的"程序员的角色不会消失，只是上移"是同一个判断。架构师不写具体策略代码了，但要定义"什么是好策略"的目标函数；这件事的认知门槛反而更高。

### Evolutionary frameworks

论文里测试了两个进化框架——OpenEvolve 和 AdaEvolve。后者整体表现更好，特别是在带 seed 的场景下。但作者明确说**框架是可换的**——这是"基础设施型论文"的一种成熟姿态。

---

## 实验：三个领域全面 SOTA

### 整体 IPC 速度提升

![图2：三个领域的 geomean IPC speedup](https://arxiv.org/html/2604.25083v1/x2.png)

*图2：左 cache replacement vs LRU、中 prefetch vs no-prefetch、右 branch prediction vs Bimodal。可以看到 evolved 设计在三个领域都达到或超过 SOTA。特别 prefetch 这一栏，evolved 设计的 1.76× 显著高于 SMS 和 VA/AMPM Lite。*

### Cache Replacement：把 Mockingjay 又推了 0.6%

![图3：每条 trace 上 cache replacement 的 IPC speedup over LRU](https://arxiv.org/html/2604.25083v1/x3.png)

*图3：横轴是 SPEC trace，纵轴是相对 LRU 的 IPC speedup。橙色 Mockingjay 是十年研究累积的 SOTA，灰色 SHIP 是另一条强 baseline，红色 evolved policy 在大多数 trace 上是头部。几何平均上 evolved +0.6% over Mockingjay——别小看这 0.6%，在 cache 替换这种已经被研究到极限的方向上，能涨 0.6% 是很硬的。*

Mockingjay 是 belady-imitating 范式的代表——训练一个硬件可实现的预测器去逼近 OPT 的 reuse-distance 决策。**Agentic Architect 居然在这种已经极度成熟的设计上又找到了 0.6% 的增量**。这个数字本身不大，但 message 很强：现有 SOTA 不是真正的局部最优。

### Prefetcher：21% 击败 SMS

![图4：每条 trace 上 prefetcher 的 IPC speedup over no-prefetch](https://arxiv.org/html/2604.25083v1/x4.png)

*图4：相对 no-prefetch 的 IPC speedup。evolved prefetcher（红色）几乎在所有 trace 上压制 SMS、VA/AMPM Lite、IPCP 等经典设计。Geomean 上 +21% over SMS 这个数字相当夸张——在 prefetcher 这种长跑领域罕见。*

### Branch Prediction：39% misprediction reduction

![图5：每条 trace 上 branch prediction 的 IPC over Bimodal](https://arxiv.org/html/2604.25083v1/x5.png)

*图5：evolved branch predictor 在 Hashed Perceptron seed 基础上多涨 1.5% geomean IPC。论文里提到在"最 prediction-sensitive 的 trace"上 misprediction 减少 39%——这是真正硬的 architectural improvement，不是 trace-specific 的 overfit。*

---

## 几个让我皱眉又拍腿的发现

### 1. seed quality bound 搜索上限

![图6：seed 强度对 evolved 结果的影响](https://arxiv.org/html/2604.25083v1/x7.png)

*图6：同样 framework，不同 seed policy。可以看到强 seed（Mockingjay）evolved 出来的结果就更好；弱 seed（LRU）即使跑同样多轮也追不上。结论一句话：**evolution 能 refine、不能凭空创造。***

这一点我特别认同。LLM-driven search 不是炼金术——它的能力本质上是"在已知技术空间里做组合优化 + 局部探索"。如果你从一个 weak seed 起步，evolution 顶多帮你爬到这个 seed 邻域的局部最优。

工程上的意义：**别指望用 Agentic Architect 从 LRU 进化出 Mockingjay**。你得自己先做出 Mockingjay 级别的 seed，然后 Agentic Architect 帮你榨出最后 5-10%。

### 2. Minimal prompt > Prescriptive prompt

![图7：minimal prompt vs full prompt 在 branch prediction 上的对比](https://arxiv.org/html/2604.25083v1/x8.png)

*图7：左侧 minimal prompt（只描述问题）、右侧 prescriptive prompt（明确指定可能用什么技术）。横轴是 iteration，纵轴是 evaluator score。**反直觉的是：minimal prompt 收敛得更稳、最终分数更高**。*

我看到这个结果第一反应是有点不敢相信，但仔细想想又非常合理——

**prescriptive prompt 会限制 LLM 的搜索空间**。你告诉它"考虑用 PC-indexed predictor"，它就大概率局限在这个 family 里探索；你不说，它会自己尝试更多角度。这跟 RLHF 里"too much human guidance hurts exploration"是同一个原理。

工程启发：**写 prompt 的时候别太聪明**——把问题描述清楚就行，让 LLM 自己探索路径。如果 LLM 比你笨，那这条评论就不成立；但 GPT-5 / Claude Opus 在 cache 替换这种 well-bounded 问题上其实已经有相当好的 prior 了。

### 3. 评估漏洞和 trace 选择

论文里特别强调 trace selection 的重要性——**用太"敏感"的 trace 训会过拟合**。这个我想起 RL 圈类似的发现：训练任务的分布选择对最终 policy 的 generalization 有决定性影响。

![图8：训练 trace vs held-out trace 的 geomean speedup](https://arxiv.org/html/2604.25083v1/x10.png)

*图8：训练集（蓝）和 held-out 集（橙）的 evolved 结果对比。可以看到 held-out 上的提升和训练上 comparable，没有明显 overfit——但这是在"多样化训练集"前提下的。论文里有专门讨论"如果训练 trace 太窄会怎样"。*

### 4. evolved 设计的"novelty 在协调而非组件"

> Across all evolved designs, individual components correspond to known techniques; what is novel is the mechanisms and policies that coordinate them.

这是论文里我觉得最值得反复读的一句话。**evolved policy 用的是已知技术**（PC indexing、reuse distance prediction、satcounter saturation 等），新意在"如何把这些技术拼起来"。

这跟过去几十年系统结构论文的"创新方式"是一样的——**几乎所有突破都是 architecture-level 的 recomposition**，组件本身是已有的。Agentic Architect 不是在发明新组件，是在自动化"如何 recompose"这件事。

我觉得这个判断对整个 LLM-driven discovery 方向都有启发意义：**LLM 的能耐不在于"发明全新概念"，而在于"高效搜索现有概念的组合空间"**。

---

## 我的判断

**亮点**：

- **首个端到端开源的 architecture exploration 框架**：之前类似工作（FunSearch、AlphaEvolve）大多是在数学/scheduling 这种相对抽象的领域。Agentic Architect 是真的能跑 SPEC、跑 ChampSim 这种工业级 benchmark 的工具。
- **三个领域全面 SOTA**：cache、prefetch、branch——三个最经典且被研究最透的子系统都拿到 SOTA。这种 breadth 让框架的 generality 立得住。
- **+21% over SMS 这个数字**：在 prefetcher 领域是个相当响的成绩。
- **方法论上的清醒**：seed bound、prompt counterintuition、组件 vs 协调的洞察——这些 takeaway 比单纯的 IPC 数字更有 lasting value。

**问题**：

- **"+0.6% over Mockingjay" 的统计显著性**：cache 替换上的差距很小，且没有多次 run 的置信区间。SPEC trace 之间的 IPC variance 本身就有几个百分点，0.6% geomean 的 significance 需要更严格的统计验证。
- **simulation cost 没量化**：跑一轮 evaluation 多少 CPU·小时？整套 evolution 的开销是多少 GPU·小时？这些信息对其他实验室复现 / 跟进非常关键，但论文披露得不够。
- **LLM 选择的影响**：论文说"prompt 比 LLM 选择更重要"，但具体测试的 LLM 列表和对比数据相对有限。我想看更系统的"GPT-5 vs Claude Opus vs DeepSeek vs Qwen"对比。
- **"open source after publication"**：到本文写作时（2026/05）代码还没公开发布。

---

## 工程启发

如果你在做：

- **芯片 / 处理器 设计**：Agentic Architect 这套框架可以直接迁移到 GPU 调度、内存控制器、cache hierarchy 等任何"行为可由代码描述、性能可由仿真衡量"的子系统。**关键是把 evaluator 做得快且准**。
- **EDA 工具链**：把"LLM-driven evolutionary search"作为现有 DSE 工具的一个补充层——人工指定搜索空间、LLM 帮你穷举。
- **任何"参数空间巨大且有靠谱评估"的优化问题**：DBMS query optimizer、compiler pass ordering、kernel scheduler——本质上是同一类问题。Agentic Architect 提供了一个"baseline 框架"可以套用。
- **写 prompt**：记住论文的反直觉发现——**minimal prompt 往往更好**。别试图把你的所有先验都写进 system prompt，给 LLM 留点探索空间。

最后说一个我觉得对整个领域都有启发的判断——**"AI for systems" 这条线在 2026 接下来会有更多 papers 出来**，但能不能跳出"刷某个 benchmark"的窠臼，关键看两件事：

1. **开源生态能不能形成**：Agentic Architect 承诺开源，如果真的兑现，会成为后续工作的 baseline 平台。
2. **评估基础设施跟得上吗**：cycle-accurate simulation 是这个领域的命脉。LLM 再聪明，evaluator 不靠谱、不可复现、不快，整个 loop 就跑不起来。

我比较乐观的是：**LLM 在工程问题上的应用很可能比在科学发现上来得更快**——因为工程问题的评估更可定义、更可自动化。Agentic Architect 这种"在已知工业 benchmark 上跑出可验证 SOTA"的做法，正是 LLM-driven discovery 走向成熟的标志。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
