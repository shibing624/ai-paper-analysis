# ShadowPEFT：把 LoRA 的"分散低秩"换成共享影子网络，顺便让 PEFT 模块可拆可装

## 核心摘要

LoRA 已经做到极致了，还能怎么改？这篇论文给了一个让我眼前一亮的角度——它指出 LoRA 这一类方法说到底就是"分散式参数化"：每个被选中的 Linear 层各自学一组低秩扰动，互相之间完全没有协调。作者提出 **ShadowPEFT**，把适配能力从"分散在各 Linear 上的微扰"集中到一个**跨层共享、随深度演化的影子网络**里——基础模型仍然冻结，但每一层的 hidden state 会和影子网络的状态做差，再通过低秩瓶颈注入回去；影子状态本身也会用 GRU 风格的门控随深度更新。

带来的两个新性质很值钱：**这个影子模块是可拆卸的**（detached shadow-only inference，能直接当个小模型独立跑）、**也是可预训练的**（用一个 0.5B 的小模型预训练后，附在 8B 的大模型上做适配）。在 Qwen3 0.6B/4B/8B 上，ShadowPEFT 在更少可训练参数下平均分都比 LoRA/DoRA 高 0.4–0.9 分；OOD 泛化、参数 scaling、推理延迟、机器人意图理解等几个 ablation 都能站住脚。

平心而论，绝对涨点不是惊艳级的（不是那种"暴打 SOTA"的故事），但论文里**对 PEFT 范式的重新解读**和**云边协同部署的可行性**，是它真正值得读的地方。

---

## 论文信息

- **标题**: ShadowPEFT: Shadow Network for Parameter-Efficient Fine-Tuning  
- **作者**: Xianming Li, Zongxi Li, Tsz-fung Andrew Lee, Jing Li, Haoran Xie, Qing Li
- **机构**: 香港理工大学（PolyU）COMP 系；岭南大学人工智能部
- **链接**: <https://arxiv.org/abs/2604.19254>
- **代码**: <https://github.com/ShadowLLM/shadow-peft>
- **模型**: <https://hf.co/collections/shadow-llm/shadow-peft-models>

---

## 我之前调 LoRA 的时候碰到的那个直觉

先说个工程上很真实的感受。

LoRA 这套东西用了好几年了，思路简单到让人怀疑——给每个 q_proj / k_proj / v_proj 加一对低秩矩阵 $W_{down}$ 和 $W_{up}$，让每层各自学一个小扰动。但只要你认真看过 PEFT 训练的 loss 曲线，应该都见过这种现象：**前面几层和后面几层的扰动量级差别很大、方向也不一致**。这其实很合理——因为每一层的 LoRA 模块都是独立优化的，它们之间没有任何"信息互通"的机制。

这就埋下了一个潜在问题：模型的内部表示是逐层流动的（layer-by-layer），但"如何调整这些表示"是被分散在各层独立学到的。你想想看，LoRA 等于把一个原本具有**跨层连续性**的适配过程，强行做成了**逐层独立的局部修补**。

ShadowPEFT 这篇论文就是从这个角度切入的。作者把它说成"linear-local parameterization vs. centralized layer-space refinement"——读起来有点学术，但翻译成人话其实就一句：

> 与其让每层各自瞎学一个扰动，不如让一个共享的小网络从头到尾跟着模型一起跑，每层都用它来给基模型"打个补丁"。

这就是 ShadowPEFT 的核心 idea。

---

## 方法核心：影子网络是怎么"跟着跑"的

![图1：左边是常规 LoRA（每个 Linear 都挂独立的低秩适配，云端边端都要带着完整 LLM），右边是 ShadowPEFT（影子网络跨层共享，可拆卸、可预训练，边端只用影子模型，云端跑完整 ShadowPEFT）](https://arxiv.org/html/2604.19254v1/x2.png)

*图1：LoRA 范式 vs ShadowPEFT 范式。左侧 LoRA 把适配能力分散到每层 Linear 上（红色 ✗ 表示不可拆卸、不可预训练）；右侧 ShadowPEFT 引入一个跨层共享的 Shadow 模块，绿色 ✓ 表示既可以独立部署到边端、也可以独立预训练。这张图把两个范式的对比拍得非常清楚。*

这张图把整个论文的卖点压缩到了一张图里。我特别喜欢它把"可拆卸性"和"可预训练性"明确画了出来——这是 ShadowPEFT 区别于所有 LoRA 变体的本质特征。

### 三步走：注入 → 编码 → 更新

具体怎么实现的？冻结的 base 模型有 $L$ 层 Transformer decoder，每一层的输出 hidden state 是 $\mathbf{h}_{out}^{(\ell)}$。ShadowPEFT 额外维护一个**和深度对齐的影子状态** $\mathbf{s}^{(\ell)}$，从输入开始就通过一个 shadow backbone $f_{\text{shadow}}$ 初始化：

$$\mathbf{s}^{(0)} = f_{\text{shadow}}(\mathbf{x};\,\theta_{\text{shadow}})$$

然后从第 1 层开始，每一层做三件事：

![图2：ShadowPEFT 三个核心模块。(a) Shadow Inject 把 base 与 shadow 的差 δ 通过低秩瓶颈投影后加回 base hidden state；(b) Base Encode 是冻结的原始 Transformer 层处理修正后的输入；(c) Shadow Update 用 LayerNorm + 双分支（变换分支 W_t 和门控分支 σ(W_g)），通过门控残差更新 shadow 状态。](https://arxiv.org/html/2604.19254v1/x3.png)

*图2：ShadowPEFT 的三模块架构图。整个流程像一条"影子轨道"在沿着 base model 的深度方向跑：注入时把差异 δ 滤一下打回去，编码时让 base 自己跑（base 完全冻结），更新时用 GRU 风格的门控让影子状态吸收新信息。*

#### Step 1：Shadow Injection

第 $\ell$ 层进入 base layer 之前，先把当前 hidden state 与影子状态做差：

$$\boldsymbol{\delta}^{(\ell)} = \mathbf{h}_{out}^{(\ell-1)} - \mathbf{s}^{(\ell-1)}$$

这个差 $\delta$ 表示当前 base 表示离影子参考的距离。然后过一个低秩瓶颈（这部分和 LoRA 几乎一样）：

$$\tilde{\boldsymbol{\delta}}^{(\ell)} = \operatorname{Dropout}(\boldsymbol{\delta}^{(\ell)}\,\mathbf{W}^{(\ell)}_{\text{down}})\,\mathbf{W}^{(\ell)}_{\text{up}}$$

其中 $\mathbf{W}_{\text{down}} \sim \mathcal{N}(0, \sigma^2)$、$\mathbf{W}_{\text{up}} = 0$（标准的 LoRA 初始化方式，保证训练初始时扰动为零、模型不被破坏）。最后通过残差注入回去：

$$\mathbf{h}^{(\ell)} \leftarrow \mathbf{h}_{out}^{(\ell-1)} + \alpha\,\tilde{\boldsymbol{\delta}}^{(\ell)}$$

这里有个细节我觉得是真的精巧的——**为什么要先做差再过瓶颈，而不是直接学一个独立扰动？** 因为差值 $\delta$ 本身就携带了"base 和 task 期望的偏离方向"，瓶颈做的事变成了"在这个差异里挑出有用的 component"，相当于把适配从"凭空生成扰动"变成了"在已有差异里筛信号"。这是个巧妙的归纳偏置。

#### Step 2：Base Encoding

第 2 步就是让冻结的 base layer 处理修正后的 hidden state，没什么特别的——base model 完全不动。

#### Step 3：Shadow Update（GRU 风格）

这是整篇论文我最喜欢的设计。base 层吐出新的 $\mathbf{h}^{(\ell)}_{\text{out}}$ 之后，影子状态需要更新，否则下一层的 $\delta$ 就没意义了。作者用了一个 GRU 风格的门控残差：

$$\mathbf{t}^{(\ell)} = T^{(\ell)}(\mathbf{h}^{(\ell)}_{\text{out}}),\quad \mathbf{g}^{(\ell)} = \sigma(G^{(\ell)}(\mathbf{h}^{(\ell)}_{\text{out}}))$$

$$\mathbf{s}^{(\ell)} = (1 - \mathbf{g}^{(\ell)}) \odot \mathbf{s}^{(\ell-1)} + \mathbf{g}^{(\ell)} \odot \mathbf{t}^{(\ell)}$$

$T$ 和 $G$ 都是两层的轻量 MLP（`SiLU` 激活，无 bias）。门控 $\mathbf{g}$ 决定影子状态在每一层"被新信息冲刷多少 vs. 保留多少之前积累的 task context"。

为什么用 GRU 而不是直接拼接或加权？作者明确写了一句："This GRU-style design is helpful to prevent shadow collapse and improve optimization stability"。说实话这个 motivation 在论文里没有充分展开，我怀疑前期实验确实出现过 shadow 状态被覆盖、训练不稳定的情况，门控是个工程修补。但这个修补本身是合理的——状态空间模型从 LSTM 到 GRU 再到 Mamba，这种"残差 + 门控"的设计已经被反复证明有效。

### 训练目标：联合损失里的"影子辅助 loss"

ShadowPEFT 的训练 loss 也很有意思。除了主任务的 next-token 预测，作者额外加了一个**影子分支的辅助 loss**：

$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\mathbf{h}_{\text{base}}\,\mathbf{W}_{\text{lm}},\;\mathbf{y}) + \lambda\,\mathcal{L}_{\text{CE}}(\mathbf{s}^{(0)}\,\mathbf{W}_{\text{shadow}},\;\mathbf{y})$$

默认 $\lambda = 0.05$。这个辅助 loss 在 Appendix E 被解释为"正则项"，但我觉得它真正的作用是**为 detached deployment 服务**——如果不加这个 loss，影子分支可能会退化成只会"配合 base 做修补"，独立时啥都不会。加了之后，shadow 自己也被监督要预测 token，这才能保证它**离开 base 模型也能用**。这个设计直接呼应了它"可拆卸"的卖点。

---

## 实验：涨点不爆炸，但很扎实

### 主实验：Qwen3 0.6B / 4B / 8B 三个尺度

作者在 Qwen3 三个尺度上对比了 LoRA、DoRA、ShadowPEFT。Benchmark 横跨生成（MMLU/GSM8K/SQuAD V2）和理解（Amazon/20News）。

| 方法 | 可训练参数 | MMLU | GSM8K | SQuAD V2 | Amazon | 20News | 平均 |
|------|----------|------|-------|----------|--------|--------|------|
| **Qwen3 0.6B** | | | | | | | |
| LoRA | 9.18M | 49.91 | 48.22 | 80.75 | 60.40 | 69.76 | 61.81 |
| DoRA | 9.32M | 50.34 | 48.37 | 80.91 | 60.92 | 69.88 | 62.08 |
| **ShadowPEFT** | **9.07M** | **50.63** | **48.90** | 80.54 | **61.18** | **70.10** | **62.27** |
| ↪ Detached Shadow Only | 9.07M | 24.62 | 1.90 | 42.10 | 50.90 | 64.42 | 36.79 |
| **Qwen3 4B** | | | | | | | |
| LoRA | 23.59M | 72.37 | 76.80 | 86.55 | 61.78 | 75.23 | 74.55 |
| DoRA | 23.91M | 72.56 | 77.86 | 86.48 | 62.02 | 75.31 | 74.85 |
| **ShadowPEFT** | **23.44M** | **72.91** | **79.00** | **86.84** | **62.66** | **75.73** | **75.43** |
| **Qwen3 8B** | | | | | | | |
| LoRA | 30.67M | 76.46 | 79.76 | 86.90 | 62.42 | 77.03 | 76.51 |
| DoRA | 31.04M | 75.79 | 78.39 | 86.79 | 62.22 | 76.78 | 75.99 |
| **ShadowPEFT** | **29.12M** | **76.51** | **80.74** | **87.51** | **62.84** | 76.99 | **76.92** |
| **Qwen3 8B + 0.5B Shadow** | | | | | | | |
| ShadowPEFT w/ random shadow | 455M | 76.82 | 80.21 | 87.39 | 62.68 | 75.88 | 76.60 |
| **ShadowPEFT w/ pretrained shadow** | 455M | 76.54 | **82.18** | **87.78** | 62.72 | 76.31 | **77.11** |
| ↪ Detached Shadow Only（预训练） | 455M | 50.03 | 48.45 | 78.93 | 60.52 | 72.63 | **62.11** |

我直接讲我看这张表的几个真实反应：

**1. 平均涨点 0.4–0.9，扎实但不惊艳。** 在三个尺度上 ShadowPEFT 都拿了平均第一，但绝对值就是 0.46（0.6B）→ 0.58（4B）→ 0.41（8B）这个量级。说实话这种幅度在 PEFT 论文里不算特别能打——同一数据集上，LoRA 不同实现之间的方差都可能差这么多。

**2. ShadowPEFT 用的可训练参数是最少的。** 9.07M vs LoRA 9.18M 和 DoRA 9.32M（0.6B 这一行）；29.12M vs 30.67M / 31.04M（8B 这一行）。这个细节其实挺重要——参数更少还能涨点，意味着新设计在"参数效率"维度上确实有优势，不是靠堆参数赢的。

**3. Detached Shadow Only（不预训练）直接崩了。** 0.6B 时拿 36.79 平均分，4B 时 38.40，8B 时 36.09——基本都在"乱猜"水平。这个结果其实很反直觉但很合理：随机初始化的影子模型本身没有 base 的能力，被设计成"贴在 base 旁边打补丁"的角色，单独拿出来当然不能用。

**4. 预训练的 0.5B Shadow 是真正的亮点。** 看最后一行：用一个独立预训练过的 0.5B 模型当 shadow，detached 时还能拿 62.11 的平均分——这个分数**比微调过 LoRA/DoRA 的 Qwen3 0.6B（约 61.8/62.0）还要高**。这是论文里我觉得最有说服力的一个数。它证明了"shadow 可以是个独立可用的模型"，从而 enable 真正的云边协同。

### Ablation：影子更新模块对生成任务必要

| 任务 | ShadowPEFT 完整版 | 去掉 Update Module |
|------|------------------|------------------|
| GSM8K | 79.00 | **76.57**（-2.43） |
| Amazon | 62.66 | 62.64（基本不变） |

这个 ablation 很说明问题：**Update 模块对推理类生成任务很重要，对分类任务影响不大**。作者的解释是分类任务靠初始 shadow 表示就够了，而多步推理需要每层不断刷新影子状态。我觉得这个解释合理，但也暴露了一个潜在问题——如果你的下游任务全是分类，那其实 shadow update 这部分参数基本是浪费的。

### 参数 scaling：DoRA 居然崩了

这块的结果挺有意思。作者把可训练参数从 0.1B scale 到 0.5B（base 模型固定 Qwen3 8B），看看三种方法的表现：

| Shadow 规模 | LoRA | DoRA | ShadowPEFT |
|-----------|------|------|-----------|
| 0.1B | ~80.5 | 81.12 | 81.35 |
| 0.2B | ~81.0 | ~80 | ~82 |
| 0.3B | ~81.0 | ~79 | ~82.1 |
| 0.4B | ~81.3 | ~78 | **82.12**（峰值） |
| 0.5B | ~80.9 | **77.79** | 81.80 |

LoRA 几乎是平的（0.5–1 分波动），ShadowPEFT 缓慢上升然后在 0.4B 触顶，DoRA 则**单调下降**——从 81.12 一路掉到 77.79。

DoRA 这个表现挺让我意外的。作者引用了 Rathore et al. 2025 的结论：低秩 PEFT 增加 rank 超过某个阈值后会损害泛化、加速遗忘。也就是说 DoRA 把 magnitude 和 direction 解耦的设计，在低 rank 时是优势，但参数一上去这个优势就反转成了劣势。**ShadowPEFT 通过引入一个独立的 shadow 模型来扩容，绕开了"一直加 rank"这条路**——这个思路其实很值得借鉴。

### 推理延迟：相比 LoRA 多 4–6%

Figure 3(b) 的延迟数据：

| Base 模型 | LoRA | DoRA | ShadowPEFT |
|---------|------|------|-----------|
| Qwen3 0.6B | 81 ms | 121.5 ms | 84 ms（+3.7%） |
| Qwen3 4B | 101.2 ms | 156 ms | 107.2 ms（+5.9%） |
| Qwen3 8B | 103.3 ms | 152.7 ms | 109.2 ms（+5.7%） |

ShadowPEFT 的额外开销在 4–6% 左右，明显比 DoRA（+50% 左右）小很多。原因是 shadow 的 forward 可以和 base 的 forward 并行执行（两个独立的网络嘛），加上 injection/update 都是轻量的小 MLP。这个数字我是信的——shadow 模型只有 base 的几十分之一规模，并行起来开销确实不会大。

### OOD 泛化：在分布外任务上稳定胜出

| 训练集 | 方法 | OOD 平均 |
|------|------|---------|
| GSM8K | LoRA | 50.40 |
| | DoRA | 48.57（**显著掉点**） |
| | **ShadowPEFT** | **50.61** |
| SQuAD V2 | LoRA | 52.41 |
| | DoRA | 52.92 |
| | **ShadowPEFT** | **53.23** |
| MMLU | LoRA | 76.57 |
| | DoRA | 75.18 |
| | **ShadowPEFT** | **76.64** |

DoRA 在 GSM8K 上训练后 OOD 直接掉了 2 个点（vs LoRA），这又是一个有意思的发现——DoRA 的 weight magnitude decomposition 设计在某些场景下会损害泛化。ShadowPEFT 的优势倒是不大（每次比 LoRA 高 0.07–0.82），但**至少没掉点**，这本身在 OOD 上就不容易。

### 系统实验：机器人意图理解

最后一个实验是 ShadowPEFT 真正展示"detached + cloud" 协同价值的地方。作者拿 Unitree Go2 机器狗做意图理解：

- ShadowPEFT 的 detached shadow（0.5B 预训练版本）部署在边端，处理常规技能命令（StandUp / TurnRight / PlayKungFu 等 34 种预定义动作）
- 复杂或开放域请求（"今天天气怎么样"、"推荐附近餐厅"），detached shadow 会输出 `[REMOTE]` 标签，路由到云端的完整 ShadowPEFT 模型

测试集准确率：

| 方法 | Test Time | Accuracy |
|------|----------|----------|
| LoRA | ~1100s | 97.7% |
| DoRA | ~1100s | 97.7% |
| **ShadowPEFT（云边协同）** | **~150s** | **99.35%** |

时间快了将近 7 倍，准确率反超 1.65 个点。Table 3 里还给了几个具体的 case study——"今天天气怎么样"这种问题，detached shadow 直接返回 `[REMOTE]`（聪明地拒答并路由），而不会像某些方案瞎调一个 `CheckWeather()` 函数。

**这才是 ShadowPEFT 的真正杀手锏**——前面所有的涨点都只是基本盘，这种"边端模型本身就是大模型适配模块的一部分"的设计，是 LoRA 完全做不到的。

---

## 我的判断

### 亮点

**1. 范式重构是这篇论文最值钱的部分。** 不是又一个 LoRA 变体，而是把"分散 vs 集中"这个轴做了重新思考。从 LoRA 的"per-Linear local perturbation"换到 ShadowPEFT 的"shared layer-level refinement"，是真的在 PEFT 设计空间里开了一个新方向。

**2. 可拆卸 + 可预训练这两个性质在工程上极其有用。** 想象你有一个 8B 的云端模型 + 一个 0.5B 的边端模型，传统做法你得分别为它们做适配；而 ShadowPEFT 让你做**一份适配**就能云边都用——只是云端跑全套 ShadowPEFT，边端只跑 detached shadow。这种"一份训练，两种部署"的能力，在 LLM 落地的成本结构里非常重要。

**3. Update Module 的 GRU 设计简单但好使。** 状态空间建模这一套思路在长上下文/序列建模里已经被反复验证，把它平移到"跨层影子状态更新"是个自然的扩展。GSM8K 上去掉这个模块掉 2.43 分，证明它对推理任务确实关键。

**4. 实验里几个 ablation 出乎意料。** DoRA 在参数 scaling 时单调下降、在 OOD 上不如 LoRA，这些发现本身就有独立价值——它告诉我们一个长期被忽略的事实：**LoRA 的所有"改进版"未必在所有维度上都更好**。

### 我觉得的问题

**1. 涨点幅度其实很有限。** 0.4–0.9 的平均分提升，在 PEFT 论文的语境里只能说"足够发表但说服力一般"。如果你纯粹追求微调效果，没必要换。

**2. Detached shadow 不预训练就崩了。** 这是个实际的工程门槛——你想真正用到 detached deployment，必须额外预训练一个 0.5B 的 shadow model（论文用 100K FineWeb-Edu + 100K Wudao 做 continual pretraining，加上 Moore-Penrose 伪逆做 head 对齐）。这一步在 LoRA 那边是完全不需要的。所以"开箱即用"的工程便利性是不如 LoRA 的。

**3. "影子模型"这个抽象的可解释性其实还很模糊。** 作者反复用 "task-adaptive reference trajectory"、"functional overlay" 这些词来描述影子状态的角色，但论文里没有一个直观的可视化告诉你 shadow 学到了什么——比如在不同任务上 shadow 状态的差异、shadow 对 base 表示的修正方向是什么样的。这块缺失的分析让"为什么 shadow 比独立 LoRA 更好"这个问题没有特别完美的答案。

**4. 参数预算可比性的问题。** 主表里 ShadowPEFT 用的可训练参数都比 LoRA/DoRA 略少（约 1–6%），表面上看是"更少参数还更好"。但 ShadowPEFT 的架构里有一些隐含的归纳偏置（共享 + GRU 门控）是 LoRA 没有的，单纯比参数数量未必能完全反映"信息利用效率"。理想情况下应该再做个 LoRA 等 rank 提升到相同参数数的对比——论文里 Figure 3(a) 部分回应了这个问题，但只有 GSM8K 一个数据集。

### 工程启发

**如果你在做边端推理 + 云端备援的架构**，ShadowPEFT 这套思路绝对值得仔细看一下。把"边端小模型"和"云端大模型的适配模块"做成同一个东西，本身就是个新颖的产品形态——你训练一次，就拿到了**一对联动的模型**，云边切换无缝。

**如果你在做 PEFT 方法本身的研究**，"集中 vs 分散"这个轴的探讨其实才刚开始。ShadowPEFT 用一个完整的小 LLM 当 shadow，参数其实还蛮重的（0.5B vs Qwen3 8B 的 30M LoRA）。有没有更轻的 shadow 设计？能不能把 shadow 设计成一个小 SSM 或者 TTT-style 的内层循环？这些都是开放问题。

**如果你纯粹想要更好的微调效果**，老老实实用 LoRA 就行——0.4 分的提升换来一堆新工程复杂度，不划算。但如果你的场景里"可拆卸 + 可预训练"是 hard requirement，那 ShadowPEFT 是目前我看到唯一比较成熟的答案。

### 横向对比：和同期工作的位置

把 ShadowPEFT 放在 PEFT 演进的脉络里看：

| 范式 | 代表方法 | 核心思想 | 跨层协调 | 可拆卸 |
|------|---------|---------|---------|-------|
| Soft Prompt | Prefix-Tuning, P-Tuning | 输入端学连续 prompt | 无 | 部分 |
| Adapter | Houlsby Adapter, Compacter | 在每个 Block 插入瓶颈 | 弱（独立 adapter） | 部分 |
| Low-Rank | LoRA, DoRA, AdaLoRA | 给每个 Linear 加低秩扰动 | 无 | 弱（嵌入到权重里） |
| **Centralized** | **ShadowPEFT** | **跨层共享的状态化影子模型** | **强（GRU 门控更新）** | **强（独立模型）** |

这个表是我自己梳理的，不是论文原文。从这个角度看，ShadowPEFT 确实开了个新的格子——以前的方法都没在"跨层协调"和"可拆卸"这两个维度上同时有突破。AdapterSoup 之类的 adapter 共享方法其实是最接近 ShadowPEFT 的，但 ShadowPEFT 多了一个**持久演化的状态**，这是 adapter 共享做不到的。

---

## 收尾

ShadowPEFT 不是那种"涨 5 个点暴打 SOTA"的论文，但它是一篇我读完之后会记住的论文。原因不是它在 benchmark 上的具体数字，而是它**重新定义了 PEFT 这件事可以怎么做**——把适配从"权重空间的局部微扰"变成"层空间的共享演化"，顺便把"边端部署"这个工程刚需第一次自然地融进了 PEFT 框架里。

我其实蛮期待看到接下来一两年这个方向上的工作：能不能把 shadow 做得更轻？能不能让 shadow 跨任务迁移？能不能用更小的 shadow 服务更大的 base（比如 0.1B → 70B）？这些问题里任何一个被攻克，都可能让 PEFT 这个领域产生质的变化。

如果你在做 LLM 的边端部署、或者对 PEFT 范式本身有兴趣，强烈建议把这篇论文细读一遍。代码已经在 GitHub 开源，模型也放到了 Hugging Face，复现门槛不高。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
