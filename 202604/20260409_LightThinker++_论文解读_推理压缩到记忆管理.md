# LightThinker++：让大模型学会"记笔记"，推理token砍掉70%还能涨点

你有没有观察过自己解一道复杂数学题的过程？一开始你会在纸上写满草稿，但写到后面，你不会回头把每一步都重读一遍——你只看自己画的关键标注和中间结论。大脑天然就有这种"压缩-检索"的记忆管理机制。

但大模型不会。它做推理的时候，每走一步都把所有历史token拖着走，上下文越来越长，KV cache线性膨胀，注意力计算二次方增长。DeepSeek-R1这类推理模型动不动生成几千个token的思维链，在复杂任务上context window直接炸了。

浙大和蚂蚁集团的团队先前提出了LightThinker，用隐式压缩把中间推理步骤"打包"成少量gist token。这次的LightThinker++更进一步——直接给模型装了一套显式的"记忆管理系统"，让它像人一样决定哪些信息该归档、哪些该展开、哪些该折叠。效果也确实能打：**peak token减少70%，推理加速26%，精度只掉1个点**；在长程Agent任务上，80轮之后context还能稳定在3-4万token（baseline已经飙到10万了），性能反而涨了14.8%。

---

## 📖 论文信息

- **标题**：LightThinker++: From Reasoning Compression to Memory Management
- **作者**：Yuqi Zhu, Jintian Zhang, Zhenjie Wan, Yujie Luo, Shuofei Qiao, Zhengke Gui, Da Zheng, Lei Liang, Huajun Chen, Ningyu Zhang
- **机构**：浙江大学、蚂蚁集团、浙江大学-蚂蚁集团知识图谱联合实验室
- **链接**：https://arxiv.org/abs/2604.03679
- **项目**：https://github.com/zjunlp/LightThinker

---

## 🎯 为什么需要这篇论文？问题到底出在哪

推理模型（如DeepSeek-R1、Qwen2.5-7B with CoT）在解决复杂问题时会生成很长的思维链。这带来一个很实际的工程问题：

**token越推越多，资源消耗失控。** 具体来说：
- KV cache随序列长度线性增长
- 注意力计算随序列长度二次方增长
- 在Agent场景下（比如Deep Research），模型要进行几十甚至上百轮交互，context轻松突破10万token

现有的解决方案要么太粗暴、要么不够灵活：

| 方法 | 策略 | 问题 |
|------|------|------|
| Vanilla | 保留所有token | context无限膨胀，Peak Token最高 |
| H2O | 基于注意力分数淘汰KV cache | 只减Peak不减Dependency，重要信息可能被误删 |
| AnLLM | 压缩中间步骤 | 压缩不可逆，关键信息丢了就找不回来 |

![图1：四种方法的context增长对比——Vanilla线性增长、H2O固定上限但总依赖不变、LightThinker锯齿状增长压缩Peak、LightThinker++在此基础上加入Expand动作实现可逆管理](https://arxiv.org/html/2604.03679v1/x3.png)

*图1：四种context管理策略的对比。Vanilla的context随生成长度线性增长（左上）；H2O通过KV淘汰固定了Peak但总Dependency面积不变（右上）；LightThinker/AnLLM通过周期性压缩形成锯齿波形（左下）；LightThinker++增加了Expand动作，允许在需要时临时恢复被压缩的内容（右下）。*

这张图很直观地展示了问题的核心：**不光要压得下去，还得在需要的时候展得开。** LightThinker++的Expand Action就是那个关键差异。

---

## 🏗 方法详解：从隐式压缩到显式记忆管理

这篇论文其实包含两个方法：LightThinker（隐式压缩）和LightThinker++（显式记忆管理），后者是前者的演进。我分别拆开讲。

### Part 1: LightThinker——用gist token压缩思维链

核心思路就一句话：**把一大段推理过程压缩成几个特殊token的hidden state，然后把原文扔掉。**

具体实现分三步：

**第一步：数据重构。** 把原始思维链拆成若干段（segment），在每段末尾插入特殊标记：`<w>` 表示"该压缩了"，`[c1][c2]...[cn]` 是cache token用于承载压缩信息，`[o]` 表示"继续生成"。模型通过SFT学会在正确的时机触发压缩。

**第二步：注意力掩码设计。** 这是技术上最精巧的部分。

![图2：LightThinker的注意力掩码设计和推理过程](https://arxiv.org/html/2604.03679v1/x2.png)

*图2：左侧(b)展示了训练时的注意力掩码——cache token只能看到当前段和之前的cache token，后续生成只能看到cache而看不到原始推理文本。右侧(c)展示推理过程——每次压缩后，原始token被丢弃，context窗口从62降到28-33之间波动。*

看这张图你就明白了：假设原始序列 $X, S_1, S_2, S_3$ 总共62个token。经过两次压缩后，context在28-33之间波动，而不是一路涨到62。节省的不只是Peak，整个推理过程中的"总依赖量"（Dependency）也大幅降低。

**第三步：两种压缩粒度。**

- **Token级**：每生成固定数量token就压缩一次。简单粗暴，但可能把一句话从中间截断。
- **Thought级**：等模型写完一个完整的推理步骤再压缩。语义更完整，效果也更好（后面实验会看到，thought级比token级准确率高6.2%）。

训练目标很标准——最大化在压缩注意力掩码下的生成概率：

$$\mathcal{L} = -\mathbb{E}\left[\log P_\theta(S_1|X) + \sum_{k=2}^{n} \log P_\theta(S_k | X, \{C^{(i)}, [o]^{(i)}\}_{i=1}^{k-1})\right]$$

### Part 2: LightThinker++——给模型装"记忆管理器"

LightThinker有一个根本性的局限——**压缩是不可逆的**。一旦某段推理被压缩成gist token，原始信息就丢了。这在某些场景下会出大问题。

![图3：压缩错误的案例](https://arxiv.org/html/2604.03679v1/x6.png)

*图3：左侧案例中，模型在压缩过程中把关键数值（8000, 4000等）丢失了，导致最终答案错误；右侧案例展示了LightThinker++如何通过expand动作恢复归档内容来纠正推理。*

上面这个案例很有说服力：模型在计算巧克力工厂产量时，压缩后把关键数字8000和4000弄丢了，最终算错。但如果有expand能力，模型可以在发现信息不足时主动展开之前归档的内容。

LightThinker++引入了三个显式的**记忆操作原语**：

| 操作 | 语义 | 效果 |
|------|------|------|
| **commit**(R, Z) | 归档——把详细推理R存起来，只保留摘要Z | context减少，只看得到Z |
| **expand**(k) | 展开——恢复第k步的完整推理R | context临时增加，获取细节 |
| **fold**(k) | 折叠——用完后再把第k步收起来 | context恢复到归档状态 |

![图4：LightThinker++的推理过程示意](https://arxiv.org/html/2604.03679v1/x4.png)

*图4：(a) 记忆操作空间——每个内容项维护原始推理 $R_k$ 和语义摘要 $Z_k$ 的双重表示，通过commit/expand/fold三个动作在两种状态间切换。(b) 推理时，模型先生成内容，然后决定commit归档，需要时fold某步或expand某步，直到answer。*

这个设计我觉得是真的漂亮。它把KV cache管理从"模型内部的隐式操作"提升为"模型可以主动控制的显式行为"。模型不再是被动地被压缩，而是自己决定什么时候该记、什么时候该翻、什么时候该收。

#### 训练pipeline也值得聊聊

LightThinker++的训练分两阶段：

**1. 环境感知轨迹合成（Environment-Aware Trajectory Synthesis）**

用一个teacher模型（DeepSeek-V3.2-Thinking）在带有记忆限制的环境中生成推理轨迹。关键是，teacher模型每一步都能看到当前的记忆状态（哪些被归档了、哪些是展开的），然后做出记忆管理决策。

**2. 行为剪枝（Behavioral Pruning）**

teacher生成的轨迹不一定都是好的，需要清洗。这里有三条启发式规则：
- **生命周期完整性**：每个commit的内容必须有对应的完整生命周期
- **对称性约束**：expand和fold必须成对出现
- **防抖动**：过滤掉那些频繁expand-fold-expand-fold的"抖动"轨迹

训练损失就是标准的SFT：

$$\mathcal{L}_\text{SFT}(\theta) = -\mathbb{E}\left[\sum_k \log P_\theta(\mathcal{I}_k | X, \tilde{\mathcal{H}}_k)\right]$$

其中 $\tilde{\mathcal{H}}_k$ 是根据当前记忆状态重建的上下文。

---

## 🧪 实验结果

### LightThinker主实验

在GSM8K、MMLU、GPQA、BBH四个benchmark上的结果：

| 方法 | 模型 | Avg Acc | Peak Token | Dep | Time(h) |
|------|------|---------|------------|-----|---------|
| Vanilla | Qwen2.5-7B | 62.05 | 2086 | 3.9M | 11.83 |
| H2O | Qwen2.5-7B | 57.57 | 597 | 2.3M | - |
| AnLLM | Qwen2.5-7B | 58.50 | 655 | 0.7M | - |
| LightThinker (token) | Qwen2.5-7B | 55.76 | 1038 | 1.5M | 11.48 |
| LightThinker (thought) | Qwen2.5-7B | 61.93 | 676 | 1.0M | 11.46 |
| Vanilla | Llama-3.1-8B | 61.33 | 1986 | 3.0M | - |
| LightThinker (thought) | Llama-3.1-8B | 57.14 | 629 | 0.9M | - |

几个值得注意的数据：

1. **Thought级压缩比Token级好6.2%**（61.93 vs 55.76），这很合理——按语义边界压缩保留的信息更完整。

2. **Peak Token从2086降到676**，减少了67.6%，而平均准确率只从62.05降到61.93——几乎无损。

3. 和H2O、AnLLM比，LightThinker在准确率上明显更高（61.93 vs 57.57/58.50），说明基于模型训练的压缩确实比基于规则的KV淘汰更智能。

![图5：LightThinker的详细分析](https://arxiv.org/html/2604.03679v1/x5.png)

*图5：(a)生成token总量对比；(b)GPQA上压缩频率分布；(c)不同输出长度下的推理时间对比——LightThinker在长输出时优势明显；(d)跨数据集的压缩比（Qwen约5-7倍，Llama约4.5-7倍）；(e-g)cache size对准确率、推理时间、Dep和Peak的影响。*

子图(c)很有意思：当输出长度从1024增长到32768时，LightThinker的推理时间增长远慢于Vanilla。在32768 token的生成长度下，Peak token只增加了4%——这说明压缩机制在长序列场景下优势巨大。

### LightThinker++主实验

LightThinker++在两种设置下评测：**Throughput**（追求最大压缩）和**Budget**（限定context预算内追求最高精度）。

| 设置 | 方法 | Avg Acc | Peak Reduction | 精度变化 |
|------|------|---------|----------------|---------|
| Throughput | Vanilla | ~87% | - | - |
| Throughput | LightThinker++ | ~87% | 69.9% | 保持 |
| Budget | Vanilla | ~87% | - | - |
| Budget | LightThinker++ | ~89% | 45.0% | **+2.42%** |

Budget设置下准确率反而涨了2.42%，这个结果挺反直觉的。我的理解是：**适当的记忆管理强迫模型做更结构化的推理**，就像你做题的时候有意识地记笔记，反而比漫无目的地写草稿更清晰。

![图6：LightThinker++的详细分析](https://arxiv.org/html/2604.03679v1/x7.png)

*图6：(a) visible token大幅减少——Qwen从2982降到511，Llama从3007降到590；(b) expand+fold操作占比约80-95%，commit占5-20%；(c) pre-commit token长度分布集中在128-384；(d) peak token随生成预算的变化；(e) commit前后的压缩比；(f) 消融实验——去掉expand和fold后peak升高、准确率下降。*

消融实验(f)的数据很清楚：

| 配置 | Avg Acc | Avg Peak |
|------|---------|----------|
| Upperbound | 60.1% | 3120 |
| No-Expand&Fold | 53.6% | 811 |
| LightThinker++ | 60.8% | 940 |

去掉expand和fold（只保留commit），准确率从60.8%暴跌到53.6%，降了7.2个点。这说明**可逆性是LightThinker++的命脉**——光能压缩不够，必须能在需要时展开。

### 长程Agent任务

这是我觉得最有说服力的实验。在Deep Research类的Agent任务上，模型需要进行多轮搜索和推理，context会随交互轮次持续增长。

![图7：长程Agent任务中的context管理](https://arxiv.org/html/2604.03679v1/x12.png)

*图7：(a) 三个数据集上的Active Context Window Size随轮次变化——Vanilla在40-60轮后context飙到9-10万token，而LightThinker++稳定在3-4万token，分别减少了67.6%、49.9%和63.4%。(b) Mean和P95 context使用量对比，LightThinker++在所有数据集上均减少约46-58%。*

看xbench这条线：Vanilla在60轮左右context达到96488 token，而LightThinker++只有31252——减少了67.6%。更关键的是，**LightThinker++的曲线是平的**，不会随着轮次增加而持续膨胀。这对实际部署来说太重要了——你不用担心跑着跑着就OOM。

性能方面，LightThinker++在xbench上Pass@1提升4.4%，hard子集上性能提升2.51倍。context更少、效果更好——这不是"有损压缩"，而是"有益压缩"。

---

## 🔬 关键案例分析

论文给了一个很好的案例来说明LightThinker++的记忆管理过程：

![图8：Agent任务中Vanilla vs LightThinker++的推理过程对比](https://arxiv.org/html/2604.03679v1/x10.png)

*图8：一个多跳问答任务"谁是任期最短的美国总统？他的副总统毕业于哪所大学？"的完整推理过程对比。上方(a) Vanilla方案context持续增长；下方(b) LightThinker++通过commit归档搜索结果、expand展开需要的内容、fold收起用完的内容，全程维持紧凑的context。*

这个案例展示得很清晰：模型先搜索"最短任期总统"，得到Harrison的wiki内容后commit归档。发现需要查副总统信息，先expand展开Harrison的内容确认副总统是John Tyler，然后fold收起，再搜索Tyler的信息。全程context保持可控，而Vanilla会把所有搜索结果都堆在context里。

---

## 🤔 批判性分析

说实话，这篇论文整体完成度很高，但有几个地方我觉得需要认真审视：

**1. 训练数据依赖teacher模型的质量**

LightThinker++的训练轨迹由DeepSeek-V3.2-Thinking生成。如果teacher模型本身的记忆管理决策不够好（比如在某些场景下该expand的时候没expand），学生模型也会继承这些缺陷。论文没有分析teacher模型的轨迹质量——行为剪枝能过滤掉明显的错误，但"次优决策"这种软性问题很难被规则捕获。

**2. 隐式压缩的信息损失问题并没有被LightThinker++彻底解决**

LightThinker++的commit操作说到底还是把详细推理R压缩成摘要Z。虽然expand可以恢复R的可见性，但Z本身的质量直接决定了模型在归档状态下能不能做出正确判断。论文展示了压缩错误的案例，但没有系统性地分析"什么类型的信息在摘要过程中最容易丢失"。

**3. 评测benchmark偏向推理任务**

GSM8K、MMLU、GPQA、BBH都是偏推理和知识的benchmark。对于代码生成、创意写作等需要长程连贯性的任务，记忆管理是否同样有效？论文虽然做了Agent任务的评测，但覆盖面还可以更广。

**4. 和最近的一些工作的对比不够充分**

推理效率优化是2025年的热门方向，除了H2O和AnLLM，还有很多同期工作在做类似的事（比如token pruning、early exit、自适应思维链长度控制等）。论文的baseline选择偏少，特别是缺少和"动态推理预算分配"这类方法的对比。

**5. 实际部署的额外开销**

LightThinker++需要模型在每一步做"记忆管理决策"（commit/expand/fold/answer），这相当于在原有推理任务上增加了一个额外的决策负担。虽然论文报告了推理时间减少26%，但这主要来自context缩短带来的注意力计算节省。在context本身不长的简单任务上，记忆管理的overhead是否会成为负担，论文没有讨论。

---

## 💡 工程启发

1. **如果你在做Agent系统，记忆管理应该是必选项而非可选项。** 从实验数据看，长程Agent任务中LightThinker++不仅省资源，还提升了效果。这说明"适度遗忘"对模型来说其实是一种正则化。

2. **显式 > 隐式。** LightThinker++比LightThinker效果更好，核心原因是显式操作给了模型可控、可逆的管理能力。如果你在设计类似系统，建议直接走显式路线——让模型自己决定什么该记什么该忘，而不是用规则帮它决定。

3. **思维级压缩 > Token级压缩。** 6.2%的准确率差距不是小数目。按语义边界做分割是值得花精力的。

4. **Dependency这个指标值得借鉴。** 传统上我们用Peak Token来衡量内存效率，但Dependency（推理过程中的累积信息使用量）其实更能反映真实的计算开销。这个视角在做推理优化时很有参考价值。

---

## 📝 总结

LightThinker++把大模型推理效率优化从"粗暴压缩"推进到了"智能记忆管理"。核心贡献是三个记忆操作原语（commit/expand/fold）加上配套的训练方案，让模型学会像人一样管理推理过程中的上下文。

这不是一个从零到一的突破——它站在LightThinker、AnLLM、gist token等前人工作的肩膀上。但它提出了一个重要的范式转变：**context管理应该是模型能力的一部分，而不是工程层面的外部优化。** 从效果来看，70%的peak token减少、长程任务上的稳定context和性能提升，都是实打实的数据。

说实话，我觉得这个方向还有很大的探索空间。比如，能不能让模型通过RL自己学会最优的记忆管理策略，而不是依赖teacher模型的示范？能不能把记忆管理和CoT的长度自适应结合起来？这些问题留给后续工作来回答。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我
