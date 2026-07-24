---
title: 多教师在线蒸馏翻车现场：当一个 token 决定整条轨迹
date: 2026-07-10
arxiv: 2607.07050
authors: Jiabin Shen, Guang Chen, Chengjun Mao
affiliation: Ant Group 等
tags: [LLM, 智能体, 工具调用, 知识蒸馏, On-Policy Distillation]
---

# 多教师在线蒸馏翻车现场：当一个 token 决定整条轨迹

你有没有遇到过这种"看起来哪里都对、跑起来就是不对劲"的训练？一个学生模型被两路教师监督——一路专门教它什么时候该调用工具，另一路专门教它什么时候该直接回答——理论上这不就是把两个专家的能力拼起来吗？结果训练完一测，确实学会调用工具了，但毛病也跟着来了：明明不该调用的地方它也调了，APIGen-MT 上的 over-calling 从 4.9% 飙到 13.7%。

更让人挠头的是看 loss 曲线——两路教师贡献"挺均衡"的啊。token 曝光量是均衡的，token 级 JSD 也是均衡的，gradient proxy 也看不出哪边在喧宾夺主。那学生到底是怎么被"教坏"的？

这篇 [arXiv:2607.07050](https://arxiv.org/abs/2607.07050) 揭开了这个谜：错不在总量，在**位置**。它给出了一个叫 **behavior leverage imbalance**（行为杠杆失衡）的概念，并配套提了一个简单到令人发指的解决方法 **Soft Clamp**。

---

## 核心摘要

**痛点**：多教师在线蒸馏（MOPD）训练智能体模型时，常用"工具调用教师 + 直接回复教师"分工合作。但学生学完之后虽然调用准确率上去了，却开始**过度调用**——本该直接回答的也去调工具，APIGen-MT 上 over-calling 从 4.9% 飙到 13.7%，多轮场景里反复调同一个工具的比例高达 16.7%。这件事诡异的地方在于：从 token 曝光量、整段 JSD、gradient proxy 这些"总账"看，两路教师是均衡的，看不出谁在喧宾夺主。

**核心方案**：论文把锅精准地甩到了一个被忽略的维度——**信号落点**。在工具调用场景里，`<tool_call>` 标签、函数名这些"模式入口 token"具有不成比例的全局控制力——只要这几个位置被推一下，后面整条轨迹就跟着跑偏了。Soft Clamp 干的事情极其简单：对 batch 内的 token 级 JSD 设一个动态阈值 $C = k \cdot \mathrm{mean}(d)$，超出的 token 前向值被钳到 $C$，梯度按 $C/d$ 缩放而不是直接截断。

**关键效果**：在 APIGen-MT 上，Soft Clamp 把 over-calling 从 13.7% 压到 9.0%，决策准确率 89.2% 与 vanilla GKD（88.9%）持平甚至反超；在 BFCL 多轮循环诊断里，Loop@3 从 14.8% 降到 10.1%，重复调用率从 16.7% 降到 11.1%。

**一针见血的评价**：这篇不是新方法、新范式，是一篇"诊断+小修小补"的工程型工作。但它戳中了一个真实痛点——多教师 OPD 的失效模式不能只看总账，必须看**信号落点**。Soft Clamp 的代码改动小到几乎可以一行解决，APIGen 决策精度不掉，多轮循环砍掉三分之一，性价比很高。值得每个做 agent 训练的人读完反思。

---

## 论文信息

- **标题**：Behavior Leverage Imbalance in Multi-Teacher On-Policy Distillation
- **作者**：Jiabin Shen, Guang Chen, Chengjun Mao
- **机构**：Ant Group（蚂蚁集团）等
- **发表日期**：2026 年 7 月 8 日
- **链接**：[arXiv:2607.07050](https://arxiv.org/abs/2607.07050)
- **篇幅**：17 页（含附录），6 张图

---

## 问题动机：为什么总账看不对？

先说场景：智能体语言模型在多轮工具调用轨迹里的行为大概是这样——

```
System → User → ToolCall₁ → ToolResponse₁ → ToolCall₂ → ToolResponse₂ → Response₁ → User → ...
```

在这样一个轨迹里，模型需要做的关键决策不是"怎么调工具"，而是"**要不要调**"。多教师 OPD 看起来是个自然解法：

- 一个工具调用教师，专门教"什么时候该调结构化函数"
- 一个直接回复教师，专门教"什么时候该自然语言回答"
- 学生在自己生成的轨迹上做 on-policy 学习（GKD 风格）

理论上完美——专业化分工、行为互补。论文做的实验就是在这个设定下，结果翻车了。

### 翻车现场：APIGen-MT 主结果

| 方法 | 决策准确率 | Over-calling | 调用召回 | 回复召回 |
|---|---|---|---|---|
| Base | 80.7 | 7.2 | 68.5 | 92.8 |
| Base SFT | 85.3 | 4.9 | 75.5 | 95.1 |
| Vanilla GKD | 88.9 | **13.7** | 91.4 | 86.4 |
| Hard Clip | 89.2 | 12.0 | 90.4 | 88.0 |
| Global Reweight | 89.1 | 10.1 | 88.2 | 90.0 |
| **Soft Clamp** | **89.2** | **9.0** | 87.5 | **91.0** |

*表 1：APIGen-MT 决策结果。Vanilla GKD 调用召回从 75.5% 飙到 91.4%——但代价是 over-calling 从 4.9% 跳到 13.7%，回复召回从 95.1% 跌到 86.4%。*（数据来源：论文 Table 3）

这就是那个让人困惑的"跷跷板"：调得动了，但不该调的地方也调了。Base SFT 是"少调"但"调得准"；Vanilla GKD 是"敢调"但"调得乱"；Soft Clamp 试图找到平衡点。

更糟的是多轮场景。BFCL 多轮循环诊断（800 个任务、3136 轮）显示，Vanilla GKD 的 **Loop@3** 高达 14.8%（连续三轮都在调工具），**重复调用**占 16.7%，**最终回复率**只有 89.6%——意味着 10% 的对话压根结束不了，模型陷入"调工具 → 收到观察 → 再调同一个工具"的死循环。

### 总账的迷惑

直觉上你要解释这个现象，第一反应肯定是"那肯定是工具调用教师贡献太大了吧？"

论文做了 sanity check（Table 1）：

| 方法 | Token 曝光 T/R | Token 级 JSD T/R | Gradient proxy T/R |
|---|---|---|---|
| Vanilla GKD | 0.867 | 0.881 | 0.957 |
| Hard Clip | 0.870 | 0.893 | 0.980 |
| Global Reweight | 0.874 | 0.902 | 0.985 |
| Soft Clamp | 0.881 | 0.895 | 0.976 |

*表 2：聚合层 sanity check。所有 GKD 变体的 T/R 比都接近 1.0，甚至略低于 1.0——意味着工具调用教师并没有"霸占"训练信号。*（数据来源：论文 Table 1）

T/R 是 tool-call 除以 response 的比值。三种聚合指标下，T/R 都接近 1 甚至略小于 1。也就是说：

- 工具调用样本没有获得更多 token 曝光
- 整段 token 级 JSD 没有明显偏向工具调用教师
- gradient proxy 同样没看出偏向

**结果：总账是平的，但行为已经歪了。** 你用任何"总量分析"的方法都查不出来。问题出在哪儿？

![图 1：Token 曝光和聚合 JSD 的 sanity check。横轴是训练 step，纵轴是各聚合指标。Tool-call 信号在总量上不占优，所以答案一定在"信号落点"上。](https://arxiv.org/html/2607.07050v1/x1.png)

*图 1：聚合层 sanity check。横轴是训练 step，纵轴是各聚合指标。可以看到四组 GKD 变体的 token 曝光、JSD 比例曲线高度重合，没有任何"哪路教师贡献过大"的迹象。*（图片来源：论文 Figure 1）

这就是论文最精彩的切入——既然总账没毛病，那一定有个"被忽略的维度"在搞事。

---

## 方法核心：行为杠杆与 Soft Clamp

### 关键洞察：行为杠杆（Behavior Leverage）

论文给了一个非常直觉化的概念——**behavior leverage**（行为杠杆）：

> 一个 token 位置对"未来生成模式"的控制程度，叫它的行为杠杆。

为什么这个概念重要？因为在工具调用场景里，存在一类特殊的"模式入口 token"：

- `<tool_call>` 标签
- `<function=get_xxx>` 函数名
- 结构化 schema 的边界符

这些 token 一旦被推一下，整个轨迹就跟着跑偏。打个比方：

> 你在十字路口准备左转还是直行，本来五五开。这时候"左转"或"直行"这两个字只占你说话内容的一小部分——但**这一个 token 决定了后面所有话**。这两个字就是"高行为杠杆"位置。

相对地，自然语言回复里的内容 token 杠杆就低很多：你把"今天天气不错"改成"今天天气挺好"，影响的是局部措辞，不会把整段对话从"回答"模式拉成"调工具"模式。

论文的诊断假设非常清晰：

> **多教师 OPD 的脆弱性，来自于当一路教师的信号集中在高杠杆位置（模式入口），而另一路教师的信号分散在低杠杆位置（内容词）时，前者对全局行为的控制力远大于它在总账里的占比。**

这个假设也很容易验证——把"工具调用样本上 student 的 P(<tool_call>) 与 teacher 的差距"作为一个 step-level 指标（论文叫 **signed pressure**），看它和最终 over-calling 是否对齐：

$$P_{\mathrm{signed}} = p_t(\texttt{<tool\_call>} \mid x, y_{<1}) - p_s(\texttt{<tool\_call>} \mid x, y_{<1})$$

| 方法 | APIGen over-call | P(tool) on response | Top-1 tool on response | BFCL irrel. refusal |
|---|---|---|---|---|
| Vanilla GKD | 13.7 | 0.164 | 15.1 | 77.8 |
| Hard Clip | 12.0 | 0.157 | 14.4 | 81.9 |
| Global Reweight | 10.1 | 0.138 | 11.3 | 80.9 |
| **Soft Clamp** | **9.0** | **0.135** | **11.2** | **83.5** |

*表 3：响应侧决策压力 vs. 最终 over-calling。Soft Clamp 在四个指标上都是 GKD 变体里最优的。*（数据来源：论文 Table 2）

![图 2：响应侧决策压力 vs. 最终 over-calling。四个点对应四个 GKD 变体，P(tool) on response 越低，over-calling 越低。](https://arxiv.org/html/2607.07050v1/x2.png)

*图 2：响应侧决策压力和 over-calling 的关系。每个点对应一个 GKD 变体，红色 Soft Clamp 在右下角——决策压力最低、over-calling 也最低。*（图片来源：论文 Figure 2）

训练过程中的 step-level 曲线就更直观了：

![图 3：训练过程中响应侧决策压力演化。上排是滚动均值，下排是累计均值。Soft Clamp 红色线在最下方。](https://arxiv.org/html/2607.07050v1/x3.png)

*图 3：训练 step 级别的 response 侧决策压力（P(tool) 和 Top-1 tool 率）。Soft Clamp 红色线在累计均值（下方两张图）里始终是最低的，而 Vanilla GKD 蓝色线在第 100-200 步间出现明显高峰。*（图片来源：论文 Figure 3）

这个图值得多看一眼。上排的滚动均值在训练初期所有方法都差不多，但 Vanilla GKD 在第 100-200 步出现一个明显的峰——这就是"工具调用入口"被高杠杆推高的瞬间。下排的累计均值差异就更稳定了，Soft Clamp 一直贴在最低位。

### Soft Clamp 算法

有了诊断，解决方案就水到渠成了：**把那些"极端的 token 级 JSD"按位置压下去，但保留梯度信号。**

对 batch 内的所有监督 token 的 JSD 值 $\{d_i\}$，定义动态阈值：

$$C = k \cdot \mathrm{mean}_i(d_i)$$

校准后的 divergence：

$$d'_i = \begin{cases} d_i, & d_i \leq C \\ d_i \cdot \dfrac{C}{\mathrm{stopgrad}(d_i)}, & d_i > C \end{cases}$$

主实验用 $k=3.0$。

这个设计有几点小心思：

1. **动态阈值**：用 batch 内均值做归一化，不依赖全局统计
2. **Forward 钳到 C**：极端 token 的 loss 不再爆炸，但也不会被截断到 0
3. **Stop-gradient 妙用**：$\mathrm{stopgrad}(d_i)$ 让分母不传梯度，所以 $C/d_i$ 是个**常数缩放因子**——极端 token 的梯度被等比缩小，但**方向和符号都还在**。这是它和 Hard Clip 的本质区别：Hard Clip 把超出部分直接截到常数，梯度按"超出部分"算（也就是 0），极端 token 的学习信号被直接掐断；Soft Clamp 保留了一个 $\frac{C}{d_i}$ 的比例梯度，模型仍然能继续学，只是步子小了点

对比另外两个 baseline：

| 方法 | 操作 | 缺点 |
|---|---|---|
| Hard Clip | $d'_i = \min(d_i, c)$，$c=0.5$ | 极端 token 失去边际梯度，学不动 |
| Global Reweight | 按 z-score 全局调权重 | 改动大段 loss，不是局部干预 |
| **Soft Clamp** | 动态阈值 + 梯度保留 | 只动极端值，普通 token 不受影响 |

---

## 实验结果：Soft Clamp 真的管用吗？

### 主实验：APIGen-MT

回到 Table 1 的主结果。Soft Clamp 把 over-calling 从 13.7% 压到 9.0%（降幅 34%），同时决策准确率 89.2% 不掉。这个图更直观：

![图 4：APIGen-MT 行为 trade-off。四个子图分别是 Decision accuracy、Over-calling、Call recall、Respond recall。Soft Clamp 红点同时满足低 over-calling 和高 respond recall。](https://arxiv.org/html/2607.07050v1/x5.png)

*图 4：APIGen-MT 四个决策指标的水平条形图。Decision accuracy 各 GKD 变体差异不大（约 89%）；Over-calling 差异最显著，Vanilla 13.7% → Soft Clamp 9.0%；Call recall Vanilla 最高（91.4%），但 Respond recall Vanilla 最低（86.4%）。*（图片来源：论文 Figure 5）

这个 trade-off 让我想起 RLHF 里的 reward hacking：你在优化一个总目标，单看总目标确实涨了，但拆开看子指标会发现"一好一坏"。Vanilla GKD 的问题是把"调用能力"学过头了，挤占了"判断什么时候不该调用"的能力。

### 跨域验证：BFCL 和 When2Call

光在 APIGen 上好不算数，作者还在两个外部 benchmark 上做了验证。

| 方法 | BFCL Overall | Tool Call Quality | Irrelevance Refusal |
|---|---|---|---|
| Base | 82.2 | 83.5 | 79.8 |
| Base SFT | 82.9 | 82.0 | 84.4 |
| Vanilla GKD | 79.8 | 80.9 | 77.8 |
| Hard Clip | 80.2 | 79.3 | 81.9 |
| Global Reweight | 80.0 | 79.5 | 80.9 |
| **Soft Clamp** | 80.6 | 79.1 | **83.5** |

*表 4：BFCL 结果。Soft Clamp 在 GKD 变体里 irrelevance refusal 最高（83.5%），符合"减少不必要工具调用"的预期。但所有 GKD 变体都没超过 Base SFT 的 84.4%——说明 OPD 本身在 BFCL 上是个负向迁移。*（数据来源：论文 Table 4）

When2Call 结果（表 5）则给论文的适用范围划了边：

| 方法 | MCQ Acc | tool_call | request_for_info | cannot_answer |
|---|---|---|---|---|
| Base | 72.8 | 88.1 | 61.6 | 68.1 |
| Base SFT | 71.6 | 88.8 | 63.3 | 63.1 |
| Vanilla GKD | 64.9 | 89.7 | 52.6 | 52.9 |
| **Soft Clamp** | 65.0 | 89.1 | 57.8 | 49.7 |

*表 5：When2Call 结果。所有 GKD 变体的 MCQ 准确率都掉到 65% 左右，明显低于 Base 的 72.8%。request_for_info 和 cannot_answer 这两个"判断不调用"的能力，GKD 反而变差了。*（数据来源：论文 Table 5）

作者诚实地承认了：**GKD 变体在 When2Call 上整体是负向的**。这其实是在提醒读者：Soft Clamp 是 GKD **内部**的行为校准方法，不解决"OPD 是否值得做"这个更上层的问题。When2Call 的失败说明 base 模型本身的工具决策能力就很强，强行用 OPD 去"学"反而会学偏。

### 最有说服力的实验：多轮循环诊断

这是我最看重的实验。多轮场景里，single-turn 的 over-calling 倾向会被**放大成循环**——一旦模型偏向调工具，多轮里就是"调工具 → 收到观察 → 再调"。

| 方法 | Calls/turn | Loop@3 | Loop@5 | Max-step | Repeat call | Final answer |
|---|---|---|---|---|---|---|
| Base SFT | 0.974 | 5.1 | 0.7 | 0.7 | 2.5 | 96.5 |
| Vanilla GKD | 1.494 | 14.8 | 8.6 | 8.6 | 16.7 | 89.6 |
| Hard Clip | 1.348 | 11.5 | 6.3 | 6.3 | 14.4 | 91.5 |
| Global Reweight | 1.398 | 12.1 | 6.6 | 6.6 | 14.8 | 92.1 |
| **Soft Clamp** | **1.268** | **10.1** | **4.7** | **4.7** | **11.1** | **94.1** |

*表 6：BFCL 多轮循环诊断。Soft Clamp 在 Loop@3（10.1%）、Repeat call（11.1%）、Final answer（94.1%）三个核心指标上都是 GKD 变体最优。*（数据来源：论文 Table 6）

![图 5：BFCL 多轮循环诊断。四个柱状图分别显示 Calls/turn、Loop@3、Repeat same call、Final answer rate。Soft Clamp 红柱带星星标记。](https://arxiv.org/html/2607.07050v1/x6.png)

*图 5：BFCL 多轮循环诊断。Vanilla GKD 蓝柱在 Loop@3（14.8%）和 Repeat same call（16.7%）上都是最高的，Soft Clamp 红柱带星星标记——GKD 变体里表现最好。*（图片来源：论文 Figure 6）

这些数字背后是真实的用户体验差异。Loop@3 = 14.8% 意味着每 7 次对话就有 1 次陷入"调三次工具以上"的死循环；Soft Clamp 把它压到 10.1%——不是消灭循环，但确实让循环少了三分之一。Final answer rate 从 89.6% 升到 94.1%，意味着多 4.5% 的对话能正常结束。这个改进幅度对生产环境的 agent 来说是实打实可感知的。

### Intervention strength：越多越好吗？

论文还做了一个"插值实验"：把 Soft Clamp 的阈值 $k$ 调高，相当于"压得更狠"。

![图 6：干预强度 vs. 行为。横轴是 clamp 阈值 $k$（越大越激进），纵轴是 APIGen over-calling。](https://arxiv.org/html/2607.07050v1/x4.png)

*图 6：干预强度（x 轴是 clamp 阈值 $k$）vs. APIGen over-calling 的关系。$k$ 越大压缩越激进，over-calling 越低，但作者明确提醒"不能读作越多越好"——过度的压缩会让极端 token 的学习信号归零。*（图片来源：论文 Figure 4）

作者在 Figure 4 caption 里特意写了一句"the result should not be read as 'more compression is always better'"。这是个诚实的提醒：Soft Clamp 不是"调参调到无脑压缩"，而是一个**平衡点**——压太狠会让模型学不到那些"真正难的高杠杆 token"，反而欠拟合。

---

## 我的判断：这篇论文值不值得读？

### 亮点

**1. 诊断比方法更值钱**

Soft Clamp 本身的代码改动确实小到可笑——batch 内求个 mean、设个阈值、乘个 stopgrad 因子就行。但**提出"行为杠杆"这个概念、并系统地用 sanity check 排除聚合解释**这个诊断过程，才是这篇论文真正的价值。

"多教师 OPD 翻车"是个老问题，之前大家都是"调调 loss 权重试试"或者"换个教师组合试试"。这篇论文告诉你：**别只盯着总量，去看信号落点**。这个思路可以直接迁移到很多场景——RLHF 的 reward hacking、多任务学习的任务干扰、混合数据 SFT 的能力漂移，本质上都是"信号落点"问题。

**2. 多轮诊断的设置很有说服力**

很多论文停在"single-turn 指标涨了 X 个点"就完了。这篇专门搭了一个 800 任务、3136 轮的 multi-turn harness，专门去量"循环"、"重复调用"、"能不能给出最终回复"——这些才是用户真正能感知的失败模式。Base SFT 调工具只调 0.97 次/轮，Vanilla GKD 调到 1.49 次/轮，多调的那 0.5 次就藏在 16.7% 的重复调用里。

**3. 诚实承认边界**

论文没有吹"我们解决了多教师蒸馏的所有问题"。When2Call 上所有 GKD 变体都是负向的，作者直接写"the current GKD variants do not improve all out-of-domain tool-use decisions"，并把 Soft Clamp 定位为"GKD 内部行为校准方法"，不是"通用工具决策解决方案"。这种分寸感在 AI 论文里挺难得的。

### 问题

**1. 没做多 seed 验证**

作者在 Limitations 里坦承："报告数值为当前训练运行的点估计；多种子研究能更好区分系统性效应与运行间变异。" 这是个实打实的缺口——over-calling 13.7% vs. 9.0% 这个差异，跨 seed 跑一遍是否稳定？论文没给数据。

**2. 双教师设定太窄**

实验是 "tool-call teacher vs. response teacher" 的二元分工。实际工程里多教师设定比这复杂得多——拒绝 vs. 有用性、代码 vs. 自然语言、短答案 vs. 长推理…… Soft Clamp 在这些更异构的设定下是否依然有效，论文没验证。从原理上讲应该有效（行为杠杆的概念不依赖具体教师类型），但缺数据。

**3. SFT anchor = 0.3 是"作弊"吗？**

主实验里所有 GKD 变体都加了一个 sft_alpha=0.3 的监督 loss 作为"格式锚"。这意味着 GKD 变体之间的对比是**纯 OPD 行为校准的对比**，但 GKD 变体和 Base SFT 的对比**已经包含格式锚的影响**。换句话说，你不能直接说"我们比 Base SFT 好"——APIGen 上的 89.2% vs. 85.3% 包含了 sft_alpha=0.3 的贡献。

作者在 Appendix E 里专门讨论了这点（"schema drift 是另一个问题，格式锚是保持 schema 可解析的"），技术上站得住脚，但读者看主表时容易高估 GKD 的实际提升。

**4. 阈值 k=3.0 是怎么来的？**

主实验固定 $k=3.0$，没做 $k$ 的消融（Figure 4 只展示趋势，没列不同 $k$ 下其他指标的变化）。从 Figure 4 看 $k=3$ 到 $k=5$ 还能继续压 over-calling，但其他指标（call recall、format 质量）会不会跌？不知道。

### 和同期工作的对比

提到多教师 OPD 的稳定性问题，2026 年还有两篇相关工作值得关注：

- **Stabilizing On-Policy Distillation for MLLM Reasoning**（Hao et al., 2606.09091）——从 MLLM 视角做 OPD 稳定化，用的是"全局归一化"，更接近本论文的 Global Reweight baseline
- **Entropy-Aware On-Policy Distillation**（Jin et al., ICML 2026）——从熵的角度做教师信号平衡

这些工作和 Soft Clamp 的差异在于：它们改的是**总账**（全局归一化、熵加权），Soft Clamp 改的是**局部极端值**（per-token 动态阈值）。从行为杠杆的视角看，本论文更精准——它不动普通 token，只针对高杠杆位置可能出现的极端值。

---

## 工程启发：怎么把这套思路用到自己的训练里？

如果你也在做 agent 训练，这篇论文至少给你三个具体的启发：

**1. 训练完先看"信号落点"，再看 loss 曲线**

loss 涨了不代表模型学对了，loss 稳不代表模型没问题。**把你的决策边界 token 拎出来单独看**——比如工具调用场景里的 `<tool_call>`、分类任务里的标签 token、代码生成里的 import 语句。画一个"step-level 决策压力"曲线，看它和最终指标的对齐关系。如果对不齐，说明你的总账在撒谎。

**2. 给极端 token 的 loss 加个动态钳位**

Soft Clamp 这个 trick 真的就几行代码——batch 内求 mean，设定阈值，超出部分 forward 钳位 + stopgrad 缩放梯度。即使你没做多教师 OPD，单教师 SFT 里也可能出现类似问题：某些 token 的 loss 异常高（数据噪声、模型不擅长），这些 token 拉爆梯度导致训练不稳。Soft Clamp 的设计哲学（"只动极端值，保留梯度方向"）可以原样套用。

**3. 评估必须覆盖多轮和"用户体验指标"**

APIGen 单点准确率从 88.9% 到 89.2%，看起来没涨多少；但多轮循环率从 14.8% 降到 10.1%、重复调用从 16.7% 降到 11.1%，这些才是用户能感知的改进。**写评估脚本时，把"对话能不能正常结束"也列为一级指标**，比单纯的任务准确率更能反映真实使用体验。

---

## 收尾

这篇论文解决的不是"怎么把多教师蒸馏做得更强"，而是"为什么多教师蒸馏明明在'工作'，行为却在偷偷变差"。答案是行为杠杆——少数高杠杆 token 位置上的极端信号，可以在总账毫无波澜的情况下，把模型推向一个错误的行为模式。

Soft Clamp 的方案非常克制：它不动普通 token、不改教师路由、不需要新的 reward model，只针对极端 per-token divergence 做一次动态压缩，**既限制了爆炸，又保留了学习信号**。

从这个角度看，它不是一个"突破性方法"，但它是一个"工程上立刻能用、且背后有清晰诊断支撑的小修小补"。

如果让我给一个总评：**这是一篇被低估的、值得在 agent 训练里立刻落地的工程型工作。** 它不会上 leaderboard，但每段实验设计都直击真实痛点，every step 都有清晰的"为什么"。

---

**论文**：https://arxiv.org/abs/2607.07050
**作者机构**：Jiabin Shen, Guang Chen, Chengjun Mao (Ant Group 等)
**发表日期**：2026 年 7 月 8 日

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。
