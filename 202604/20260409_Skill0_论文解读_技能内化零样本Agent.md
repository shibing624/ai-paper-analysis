# 训练时把技能"喂"进参数里，推理时不带小抄也能满分——Skill0的技能内化之路

你有没有遇到过这种情况：给Agent配了一堆skill文档，检索命中率还行，但推理的时候token蹭蹭涨，效果却不见得好多少？

这其实是当前Agent skill体系的一个老问题了——技能停留在context里，模型只是在"照着念"，而不是真正"学会了"。就像一个人做菜永远盯着菜谱，和一个已经把菜谱内化成肌肉记忆的大厨，完全是两回事。

这篇来自浙江大学和美团的论文提出了 Skill0，核心思路很直白：训练的时候给skill文档当辅助轮，然后一点一点把辅助轮撤掉，直到模型能零样本独立完成任务。在ALFWorld上比标准RL baseline涨了9.7%，Search-QA涨了6.6%，而且推理时每步token开销压到0.5k以下。这不是一个全新范式的突破，但它在"怎么让Agent真正学到技能"这个工程问题上给了一个干净利落的解法。

---

## 论文信息

**标题**：SKILL0: In-Context Agentic Reinforcement Learning for Skill Internalization

**作者**：Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen

**机构**：Zhejiang University, Meituan

**日期**：2026年4月2日

**代码**：https://github.com/ZJU-REAL/SkillZero

**论文**：https://arxiv.org/abs/2604.02268

---

## 问题出在哪：推理时带着skill文档跑，是治标不治本

当前LLM Agent的技能增强路线，主流做法是运行时检索：任务来了，从SkillBank里找到相关的skill文档，塞进context里当指导。SkillRL就是这条线上的代表工作，它甚至会在RL训练过程中让skill文档一起进化。

但这条路有三个越来越明显的问题：

**检索噪声**。相似度检索不是万能的，经常会把不相关的skill塞进去，模型被误导。你想想，模型正在做"把杯子放微波炉加热"这个任务，结果检索给了一个"如何在冰箱里冷藏东西"的skill，这不是帮倒忙吗？

**token开销爆炸**。SkillRL每步消耗2.21k tokens，其中大量是skill文档占据的。这在实际部署中是很大的成本。

**知识没有被真正学会**。模型只是在follow instruction，并没有把技能内化到参数里。一旦撤掉skill文档，性能直接崩塌。

![图1：Skill Augmentation vs Skill Internalization的对比](https://arxiv.org/html/2604.02268v1/x1.png)

*图1：上面是传统的Skill Augmentation方式（以SkillRL为代表），skill文档随训练越积越多，推理时需要检索；下面是Skill0的Skill Internalization方式，skill数量随训练递减，最终推理时不需要任何skill文档。*

这张图把两种范式的差异画得很清楚：一个是skill越积越多（红色箭头上升），一个是skill逐步撤掉（绿色箭头下降）。方向完全相反。

---

## Skill0怎么做：三步走的课程学习

Skill0的整体框架分成三个模块，配合得挺紧凑的。

![图2：Skill0的整体框架](https://arxiv.org/html/2604.02268v1/x2.png)

*图2：Skill0的完整流程——(a) 按类别对skill进行分组，(b) 上下文强化学习的单步交互循环，(c) 课程学习过程中skill逐步减少直到零样本推理。*

### 第一步：Skill分组（Relevance-Driven Skill Grouping）

不是简单的相似度匹配，而是按任务类别把skill聚类。比如ALFWorld里有6种任务类型（Pick、Look、Clean、Heat、Cool、Pick2），每种任务类型对应一组相关skill。这个分组是离线完成的，不吃推理时间。

具体来说，SkillBank由两层组成：通用技能（exploration strategies、goal-tracking heuristics）和任务特定技能（domain-specific action sequences和preconditions）。分组完之后，每个验证子任务会被分配一组候选skill。

### 第二步：上下文强化学习（In-Context RL with Visual Context）

这里有个有趣的设计——交互历史不是用纯文本渲染的，而是用视觉方式呈现。Agent在环境中每步交互时，skill文档和历史观察被压缩成紧凑的视觉context，模型基于此生成动作。

训练用的是类PPO的框架，reward由两部分组成：

$$\tilde{r}_t = r_t + \lambda \cdot r_t^{comp}$$

其中 $r_t$ 是任务完成奖励，$r_t^{comp}$ 是context压缩奖励。压缩奖励只在任务成功时给：

$$r_t^{comp} = \ln(c_t)$$

$c_t$ 是当前步使用的token数。这个设计的直觉很好理解：完成同样的任务，用的token越少越好。对数函数保证了压缩奖励不会压过任务奖励。

### 第三步：动态课程（Dynamic Curriculum）——核心中的核心

这是Skill0最关键的设计。训练被划分为 $N_S$ 个阶段，每个阶段允许携带的skill数量按线性衰减：

$$|\mathcal{S}^{(s)}| \leq M^{(s)} = \lceil N \cdot \frac{N_S - s}{N_S - 1} \rceil$$

$N$ 是总skill数，$s$ 是当前阶段编号。第一阶段带全部skill，最后阶段skill降到0。

但光衰减还不够。哪些skill该先撤、哪些该保留？Skill0用了一个 **on-policy helpfulness** 指标：每隔 $d$ 步在验证集上跑两次——一次带skill、一次不带。如果带skill后成功率反而不涨甚至更低，说明模型已经内化了这个skill，可以撤掉。

整个筛选流程是 **Filter -> Rank -> Select**：
1. **Filter**：去掉helpfulness为负的skill（模型已经不需要了）
2. **Rank**：按helpfulness降序排列（越有帮助的越靠前）
3. **Select**：在当前budget内选top-k

这三步缺一不可，后面消融实验会看到去掉任何一步都会崩。

---

## 实验结果：数据说话

实验在两个环境上做：ALFWorld（家居交互任务）和Search-QA（搜索增强问答），基座模型用的是Qwen2.5-VL的3B和7B版本。

### ALFWorld主实验（3B模型）

| 方法 | Pick | Look | Clean | Heat | Cool | Pick2 | Avg | Cost(k) |
|------|------|------|-------|------|------|-------|-----|---------|
| Zero-Shot | 27.0 | 24.3 | 4.5 | 20.5 | 10.2 | 0.0 | 15.2 | 1.21 |
| GRPO | 92.6 | 85.7 | 70.6 | 86.6 | 79.3 | 65.0 | 79.9 | 1.02 |
| AgentOCR | 91.9 | 81.8 | 76.0 | 73.3 | 76.1 | 70.0 | 78.2 | 0.38 |
| SkillRL | 91.9 | 100 | 82.9 | 87.4 | 78.7 | 70.0 | 82.4 | 2.21 |
| **Skill0** | **95.6** | **80.4** | **100** | **86.7** | **78.7** | **75.2** | **87.9** | **0.38** |

几个数字值得注意：

Skill0在Clean任务上打到100%，这个很夸张。但Look任务只有80.4%，反而不如SkillRL的100%。说明skill内化过程中，不同任务类型的学习难度差异很大——有些操作性知识更容易被参数化，有些需要更多探索的任务则没那么好内化。

token开销方面，Skill0和AgentOCR一样只有0.38k，而SkillRL需要2.21k。这是接近6倍的压缩。坦率讲，如果效果相当但token省6倍，在工程上是很有吸引力的。

### Search-QA主实验（3B模型）

| 方法 | NQ | Triv | Pop | Hotp | 2Wk | MuS | Bam | Avg | Cost(k) |
|------|------|------|------|------|------|------|------|------|---------|
| GRPO | 39.3 | 60.6 | 41.1 | 37.4 | 34.6 | 15.4 | 26.4 | 36.4 | 0.61 |
| AgentOCR | 38.6 | 56.5 | 41.7 | 33.6 | 30.7 | 14.6 | 24.0 | 34.2 | 0.26 |
| SkillRL | 38.6 | 57.6 | 40.3 | 33.6 | 31.1 | 13.3 | 58.1 | 38.9 | 0.87 |
| **Skill0** | **39.8** | **57.5** | **42.3** | **35.1** | **33.7** | **13.3** | **63.7** | **40.8** | **0.18** |

Search-QA上Skill0的提升没那么dramatic，比GRPO涨了4.4个点。不过在Bamboogle（Bam）这个子任务上涨到63.7%，比GRPO的26.4%翻了一倍多。token只用0.18k，这个数字很能打。

但说实话，在7B模型上，Skill0的Search-QA成绩（44.4%）被SkillRL（47.1%）反超了。这说明在知识密集型任务上，运行时携带skill文档的优势可能更大——有些搜索策略的知识确实不太容易纯靠参数学到。

### 训练过程曲线

![图3：ALFWorld训练Reward曲线](https://arxiv.org/html/2604.02268v1/x3.png)

*图3：ALFWorld上Skill0（红色）vs AgentOCR（蓝色）的训练Reward曲线。Skill0在约40步后开始明显拉开差距。*

![图4：Search-QA训练Reward曲线](https://arxiv.org/html/2604.02268v1/x4.png)

*图4：Search-QA上的训练曲线，两条线差距不如ALFWorld明显，Skill0在后期略微领先。*

从训练曲线看，ALFWorld上Skill0的优势在中后期体现得很清楚，大约step 40之后reward开始拉开。Search-QA上差距没那么大，两条线缠绕在一起，后期Skill0稍微领先。

---

## 关键分析：技能到底有没有被内化？

这是整篇论文最该追问的问题。作者给了三组很有说服力的证据。

### 证据一：有无skill的精度差逐渐消失

![图5：训练过程中有/无skill的精度对比](https://arxiv.org/html/2604.02268v1/x5.png)

*图5：三个子图分别展示——(a) Skill0训练时有/无skill的精度曲线收敛到一起；(b) Skill0 vs AgentOCR在无skill推理时的对比；(c) Skill0 vs GRPO vs SkillRL在无skill推理时的对比。*

图5(a)是关键证据：绿色线（带skill）和紫色线（不带skill）在训练后期几乎重合，说明模型确实不再依赖外部skill context了。图5(b)则显示，同样在无skill条件下推理，Skill0比AgentOCR高出不少，说明Skill0在训练过程中确实学到了更多。

### 证据二：Helpfulness的"先升后降"模式

![图6：各任务类型的skill helpfulness随训练步数变化](https://arxiv.org/html/2604.02268v1/x6.png)

*图6：ALFWorld各任务类型的skill helpfulness随训练步数变化。大部分任务呈现"先升后降"的特征模式，最终趋向零或负值。*

这组图特别有意思。以"Pick Heat Then Place In Recep"为例，helpfulness在60步左右达到峰值0.8，然后一路降到训练结束时接近0。这说明skill在训练早期确实很有用（模型还在学），但随着知识被内化，skill变得多余甚至有害。

右下角的总体Success Rate也呈现同样的pattern——先升后降，最终skill的边际收益归零。这个"先升后降"的曲线形状，就是技能内化成功的指纹。

### 证据三：各任务类型的学习轨迹

![图8：各任务的有/无skill学习曲线](https://arxiv.org/html/2604.02268v1/x8.png)

*图8：ALFWorld六种任务类型的学习曲线。绿线（无skill）和紫线（有skill）在大部分任务上最终收敛，验证了技能内化的效果。*

从各个子任务的角度看，"Pick And Place"和"Pick Clean Then Place In Recep"的内化效果最好，两条线后期几乎完全重合。"Pick Heat Then Place In Recep"的内化速度最慢，在训练后期才追上来。

![图9：Search-QA各任务的学习曲线](https://arxiv.org/html/2604.02268v1/x9.png)

*图9：Search-QA四种任务类型的学习曲线，同样呈现收敛趋势。*

---

## 消融实验：每一步都不能省

消融实验做得很扎实，把我最关心的几个问题都回答了。

### 动态课程的三个组件缺一不可

| 配置 | 带skill精度 | 不带skill精度 | 差值 |
|------|-----------|-------------|------|
| Filter + Rank + Select（完整版） | 86.3 | 87.9 | +1.6 |
| 去掉Filter | 81.6 | 78.9 | -2.7 |
| 去掉Rank | 76.6 | 62.9 | -13.7 |

去掉Rank之后，不带skill时直接从87.9%掉到62.9%，暴跌25个点。这个数字太恐怖了。它说明如果不按helpfulness排序就随机保留skill，模型会学到一堆没用甚至有害的"知识"，内化过程完全失败。

### Budget衰减策略的选择

![消融图：不同budget策略的对比](https://arxiv.org/html/2604.02268v1/figures/ablations.png)

*消融实验柱状图：[6,3,0]是Skill0的默认策略（从6个skill线性衰减到0），对比固定保持全量（Fixed Full）、均匀分配[6,6,6]、更快衰减[3,3,3]、无skill[0,0,0]等策略。*

这张柱状图一目了然。[6,3,0]（Skill0的默认设定）是唯一一个"不带skill时精度反而比带skill更高"的配置，涨了1.6个点。

Fixed Full（一直带着全部skill训练）的问题最严重：带skill时84%还行，不带skill直接掉到72%左右，差了12.3个点。这正好验证了论文的核心论点——一直给拐杖，模型永远学不会走路。

[6,6,6]（每阶段都保持6个skill）也不行，不带skill时掉到72.6%。[0,0,0]（完全不给skill）好于这两者（78.9%），但不如[6,3,0]的87.9%。这说明"先给再撤"的课程学习确实比两个极端都好。

![图7：课程策略对比的训练曲线](https://arxiv.org/html/2604.02268v1/x7.png)

*图7：(a) 带skill推理时，Skill0 vs [3,3,3]策略的精度曲线；(b) 不带skill推理时，Skill0 vs Fixed Full策略的精度曲线。Skill0在两种条件下都有明显优势。*

### 验证间隔的影响

| 验证间隔 d | ALFWorld | Search-QA |
|-----------|----------|-----------|
| 10（默认） | 87.9 | 48.9 |
| 5 | 87.5 | 49.6 |
| 20 | 78.1 | 42.3 |

$d=20$ 时ALFWorld掉了将近10个点，说明评估太不频繁的话，helpfulness信号滞后太多，会导致该撤的skill没及时撤、不该撤的被误撤。$d=5$ 和 $d=10$ 差不多，但 $d=5$ 的计算开销翻倍，性价比不高。

---

## 我的判断：亮点与局限

### 亮点

**思路清晰且自洽**。"先给辅助轮再逐步撤掉"这个课程学习的直觉，加上"on-policy helpfulness驱动的动态筛选"这个机制设计，整条技术链路很干净。不是那种七拼八凑的复杂系统。

**消融做得非常充分**。Filter/Rank/Select的逐个消融、budget策略的系统对比、验证间隔的敏感性分析，该做的都做了，而且数据说服力强（去掉Rank暴跌25个点这种现象很能说明问题）。

**工程价值明确**。token开销从2.21k压到0.38k，接近6倍压缩，同时精度还涨了。这个trade-off在实际部署中是非常有吸引力的。

### 我觉得值得追问的地方

**SkillBank的质量依赖**。论文坦言Skill0依赖SkillBank的初始质量。但这个SkillBank是怎么构建的？如果原始skill本身就写得不好，或者覆盖不全，内化效果会怎样？这个问题论文没深入讨论。

**7B在Search-QA上不如SkillRL**。在知识密集型任务上，内化方案被推理时检索方案反超（44.4% vs 47.1%），这暗示了一个边界条件：当任务需要的知识量超过模型参数能承载的范围时，运行时检索可能仍然是更好的选择。论文对这个现象没有展开分析，但我觉得这其实是一个很重要的信号。

**基座模型偏小**。实验只做了3B和7B，没有更大模型的结果。参数量更大的模型本身"装"知识的容量就更大，skill内化的效果是否会更好？有没有一个模型大小的临界点？这些都值得探索。

**泛化性存疑**。ALFWorld和Search-QA是两个特定的环境，任务类型相对固定。在更开放的Agent场景（比如WebArena、OSWorld）里，这套课程学习框架是否还能work？技能的种类更多、任务更复杂时，动态课程的scalability是个问题。

**与同期工作的定位**。SkillRL是作者团队的前作（从GitHub组织名ZJU-REAL就能看出来），Skill0可以看作是在SkillRL基础上的"反方向"探索。SkillRL的路线是让skill和agent co-evolve（技能越来越多），Skill0的路线是让agent internalize skill（技能越来越少）。两种范式可能各有适用场景，但论文倾向于把Skill0包装成一个普遍更好的方案，这个结论可能过于绝对。

---

## 技术细节补充

训练配置上，用的是4块H800 GPU，ALFWorld每批16个任务各rollout 8次，max token 3072；Search-QA每批128个任务，max token 4096。课程阶段数 $N_S = 3$，验证集大小1000个实例。检索器用的是E5 embeddings。

训练目标是经典的PPO变体，加了KL正则：

$$\mathcal{L}_{Skill0}(\theta) = \mathbb{E}\left[\frac{1}{\sum|{\tau_i}|} \sum_i \sum_t \text{clip}(r_{i,t}(\theta), A_i, \epsilon)\right] - \beta \cdot \mathbb{D}_{KL}[\pi_\theta \| \pi_{ref}]$$

这里 $\pi_{ref}$ 是参考策略，$\beta$ 控制KL惩罚的强度，防止训练后的策略偏离太远。

---

## 写在最后

Skill0给我的感觉是：idea很直觉但执行很扎实。"训练时当拐杖、推理时扔掉"这个思路谁都能想到，但怎么让"扔掉"的过程平滑而不崩塌，这才是技术含量所在。动态helpfulness评估 + 线性衰减budget的组合，给出了一个干净的解法。

如果你在做Agent相关的工作，尤其是面对推理时token成本压力大的场景，Skill0的思路值得一试——先用skill文档做warm-up训练，然后把辅助轮撤掉，让模型真正"学会"而不是"照抄"。

但也要清醒地看到，在知识密集型任务上（Search-QA 7B的结果就是例子），模型参数不一定能完全替代外部知识的运行时注入。skill内化有它的适用边界，不是万能药。

有个更根本的问题可能值得整个领域思考：Agent到底应该把知识存在参数里还是存在外部工具里？Skill0和SkillRL分别代表了这两个方向。可能最终答案是——看场景。但Skill0至少证明了，对于操作性强、模式相对固定的任务，内化是更高效的选择。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我
