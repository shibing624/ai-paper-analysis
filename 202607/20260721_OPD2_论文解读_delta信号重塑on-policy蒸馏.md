# 蒸馏信号再设计：把"推理知识"和"自然偏好"拆开，OPD² 在 Qwen3 / Gemma4 上全面打赢了 OPD 和 ExOPD

> 论文：On-Policy Delta Distillation
> 作者：Byeongho Heo, Jaehui Hwang, Sangdoo Yun, Dongyoon Han（NAVER AI Lab）
> arXiv：2607.15161
> 代码：https://github.com/naver-ai/opd2

---

## 核心摘要

读完摘要的第一反应是：等等，这个想法居然没人做过？

On-Policy Distillation（OPD）的标准奖励信号是 $\log \pi^* - \log \pi_\theta$，也就是 teacher 和 student 的对数概率差。直觉上合理，问题是它把两件不同的事情搅在一起——teacher 通过 reasoning tuning（指令微调、RL）获得的能力，**以及** teacher 从预训练里继承的语言习惯。

NAVER 这篇 OPD² 提出一个非常干净的解法：把 teacher 减去它的"上一代"，也就是 $\log \pi^* - \log \pi^*_{\text{base}}$（paper 里叫 delta signal）。这个差值纯粹刻画"reasoning tuning 到底让模型学到了什么"，把"模型本来就爱这么说话"的部分直接减掉。

效果在 Qwen3 和 Gemma4 上一致：non-thinking 模式数学平均涨 3-4 个点，thinking 模式别人都在掉分的时候它还在涨。代价是训练时间多 24-28%，因为要多跑一次 base model 的前向。

值不值得细读？我的判断是：**如果你在做小模型蒸馏 post-training，这篇是必读的，方法干净、可复现、信号分析到位**。它不是 GRPO 那种"训练 pipeline 大改"的工程活，而是一个奖励函数层面的 insight，迁移成本极低。

---

## 论文信息

| 项目 | 内容 |
|---|---|
| 标题 | On-Policy Delta Distillation |
| 作者 | Byeongho Heo, Jaehui Hwang, Sangdoo Yun, Dongyoon Han |
| 机构 | NAVER AI Lab |
| 提交日期 | 2026/07/16 |
| arXiv | 2607.15161 |
| 篇幅 | 19 页，4 张图，12 个表 |
| 代码 | https://github.com/naver-ai/opd2 |

---

## 为什么要重新设计蒸馏信号？

OPD 这个范式最近一两年被 Qwen3 团队带火了，说到底是用一个强 teacher 在自己 student 生成的 token 上打 token 级分数，绕开了 RL 里 reward 设计难、reward hacking 的问题。Qwen3 技术报告里专门提到，1.7B 这种小模型用 OPD 比 GRPO 更省、更稳。

但 paper 的作者们指出，OPD 这么多研究，**奖励函数本身几乎没人动过**。大家要么改改 top-k 处理、要么加加 advantage 归一化，但 $\log \pi^* - \log \pi_\theta$ 这个核心信号始终是雷打不动的默认设置。

他们的观察是：

> teacher 不是一个"纯粹推理机"，而是一个被预训练 + 推理微调两层塑造的混合体。

举个例子。Qwen3-4B-Thinking 经过 RL 之后，模型可能确实学会了"先在脑里想一步再下笔"——这是 reasoning tuning 教会的。但它同时也"自带"了某些语言偏好，比如喜欢用 "however"、"thus"、"see"、"try" 这些高频词。这些偏好**不是 reasoning 的产物**，是从万亿 token 的预训练里继承下来的。

用 $\log \pi^* - \log \pi_\theta$ 去学，等于让学生**同时模仿**了 reasoning 能力和语言习惯。语言习惯这部分其实 student 自己就差不多有了，强 teacher 教学生说"hence"其实没什么增量信息。

**所以真正的 reasoning knowledge，应该从 teacher 身上"减掉"它本来就有的东西，再去学剩下的。**

这就是 delta signal 的直觉。

---

## 方法核心：delta signal + centering + 联合条件

### 1. 形式化定义

标准的 OPD 奖励：

$$R_t^{\text{OPD}} = \log \pi^*(y_t \mid x, y_{<t}) - \log \pi_\theta(y_t \mid x, y_{<t})$$

作者提出的 delta signal：

$$R_t^{\Delta} = \log \pi^*(y_t \mid x, y_{<t}) - \log \pi^*_{\text{base}}(y_t \mid x, y_{<t})$$

区别就一项：右边第二项从 student 换成了 teacher 的 base 模型（同家族、同规模、未经过 reasoning tuning 的版本）。比如 Qwen3-4B-Thinking-2507 的 base 就是 Qwen3-4B-Base。

### 2. 为什么直接用 $R_t^{\Delta}$ 不够？

作者写得很坦诚：用 $R_t^{\Delta}$ 做训练有个**收敛性问题**——因为奖励里完全没有 student 的位置，模型会一路 push 到"teacher 最想要的 token 概率为 1"的 one-hot 极值。

实践中虽然到不了那个点（强 teacher → 弱 student），但训练会不稳定。所以他们又加了两层设计：

**a) Centering（去偏）**：减掉 student 当前分布下的期望奖励，让优势函数有正有负。

$$A_t^{\Delta} = R_t^{\Delta} - \mathbb{E}_{\tilde{y}_t \sim \pi_\theta} \left[ \log \pi^*(\tilde{y}_t) - \log \pi^*_{\text{base}}(\tilde{y}_t) \right]$$

实现上对 student 概率最高的 top-1024 个 token 算期望，省显存。

**b) 联合条件（Joint Conditioning）**：只在 $\Delta$ 信号和原始 OPD 信号**方向一致**的时候才更新：

$$A_t^{D^2} = \begin{cases} A_t^{\Delta} & \text{if } A_t^{\Delta} \cdot A_t^{\text{OPD}} > 0 \\ 0 & \text{otherwise} \end{cases}$$

这个 trick 漂亮的地方在于：它解决了上面那个收敛性 bug——当 student 和 teacher 完全一致的时候，$A_t^{\text{OPD}} = 0$，联合条件直接关掉梯度。同时，$A_t^{\Delta}$ 控制梯度大小，$A_t^{\text{OPD}}$ 控制方向，**sign-consistent 的方向**才更新，避免 student 偏离 teacher 太远。

### 3. 整体流程图

下面这张图把 OPD 和 OPD² 的区别讲得很清楚：

![图1：OPD vs OPD² 流程对比](https://arxiv.org/html/2607.15161v1/x1.png)

*图 1：左图是传统 OPD——student 把"草稿"交给 teacher，teacher 直接反馈；右图是 OPD²——除了 student 草稿，teacher 还要跟自己没做 reasoning tuning 之前的"上一代"做对比，把 Δ 信号反馈给学生。简单说就是**老师批改作业时先回忆一下自己高中是怎么学的，把"风格偏好"和"真本事"分开打分**。*

---

## 信号到底有什么不一样？

paper 没有停在"理论看起来对"上，而是用三组分析把 delta signal 的"行为"展示出来，每一组都很有说服力。

### 分析 1：词云

作者用 Qwen3-1.7B 做 student、Qwen3-4B-Thinking-2507 做 teacher、Qwen3-4B-Base 做 base，在 10k 数学题上做 rollout，然后分别画 OPD / Base / Δ 三个信号的词云（取 positive signal 强度）。

结论非常直观：

- **OPD 词云** 偏向一般性的高频词：see、try、verify、confirm——这些是"探索/验证"型语言，也是 base 模型自带的高频词。
- **Δ 词云** 强调逻辑连接词：hence、note、however、instead、yet；**抑制** 探索型词汇。

直接看 token 列表更明显：Δ 信号主动"减掉"了 base 模型自带的探索性表达，把蒸馏重心推到推理逻辑词上。

### 分析 2：Token 级可视化

作者构造了三道简单的推理题（数学、科学、代码），但**故意把答案写错**。然后看 OPD 和 Δ 各自在错误 token 上的信号方向。

![图3：token 级信号对比](https://arxiv.org/html/2607.15161v1/x5.png)

*图 3：左边一列是 OPD 信号，右边一列是 Δ 信号。蓝=促进，红=抑制。三个例子分别是数学（红球蓝球求和）、科学（水蒸气，错误回答"水会变暖蒸发"）、代码（Python 加法）。可以看到 Δ 在 "becomes warmer"、"joins" 这些错误推理 token 上是**红色抑制**的，而 OPD 经常给这些 token 打**蓝色促进**——因为 student 错得太离谱，student 概率极低，$\log \pi^* - \log \pi_\theta$ 反而变正了。Δ 因为 teacher 和 base 同源，对"推理是否在正轨上"更敏感。*

这一段是整篇 paper 我觉得最漂亮的部分。**信号的方向性是真正能反映 reasoning correctness 的**，而不是被 log-prob 差值的 magnitude 噪声给污染。

### 分析 3：统计

作者在 1 万题 × 3 个领域（Math/Code/Science）上统计了"从 OPD 换成 Δ，token 信号变化超过 1 的比例"，列出了 top 增强词和 top 抑制词。

跨域一致：增强 hence、thus、however、regardless；抑制 perhaps、tackle、computing、consider、analyze。

`perhaps` 这个词很有意思——它代表"我不确定"，在推理里几乎不贡献信息。Δ 信号主动压它，符合"推理要确定"的逻辑。

---

## 实验结果

实验设计是 paper 的另一个亮点。覆盖了 Qwen3 三个尺寸（1.7B / 4B / 8B）× 两种模式（thinking / non-thinking）× 三个领域（Math / Code / Science），再加上 Gemma4-E4B-it 做跨家族验证。总共 14 个 benchmark（7 Math + 4 Code + 3 Science）。

### Non-thinking 模式

非思考模式下，Qwen3 本身的推理能力**有明显的提升空间**，三种 OPD 方法都能涨分。

**Qwen3-1.7B 数学平均分：**

| 方法 | AIME24 | AIME25 | AMC23 | HMMT25 | MATH500 | Olympiad | RGMath | **Avg** |
|---|---|---|---|---|---|---|---|---|
| Base | 14.2 | 9.4 | 41.2 | 5.0 | 68.6 | 26.0 | 79.5 | 34.8 |
| +OPD | 36.7 | 23.8 | 70.8 | 17.0 | 81.0 | 37.1 | 90.9 | 51.0 |
| +ExOPD | 35.4 | 23.3 | 75.5 | 14.7 | 81.9 | 37.8 | 90.9 | 51.4 |
| +OPD² | **41.0** | **28.8** | **79.5** | 15.0 | **83.9** | **40.1** | **93.9** | **54.6** |

1.7B 这个尺寸上 OPD² 比 ExOPD 平均多 **3.2 个点**，比 OPD 多 **3.6 个点**。在 Code 领域优势更大：Qwen3-1.7B Code 平均从 10.5（base）到 29.4（OPD²），比 ExOPD 24.6 高 **4.8 个点**。

一个特别值得提的细节：**4B + OPD² 的数学平均 70.3，已经超过了 8B + ExOPD 的 67.8**。算一笔账就是，**用 OPD² 等于"免费"加一层参数**。

### Thinking 模式

Thinking 模式下 Qwen3 已经很强了，**OPD 在这里经常是负优化**。

**Qwen3-1.7B thinking 数学平均：**

| 方法 | Avg |
|---|---|
| Base | 59.2 |
| +OPD | 57.1 ⬇️ |
| +ExOPD | 58.4 ⬇️ |
| +OPD² | **62.7** ⬆️ |

**Qwen3-8B thinking 数学平均：**

| 方法 | Avg |
|---|---|
| Base | 73.7 |
| +OPD | 72.2 ⬇️ |
| +ExOPD | 73.6 ⬇️ |
| +OPD² | **75.9** ⬆️ |

这是 paper 里我觉得最值得高亮的结果。**当 baseline 很强时，OPD / ExOPD 都在掉分，OPD² 是唯一一个还在涨的**。原因很自然：thinking 模型自身的"自然偏好"已经和 reasoning 高度耦合了，OPD 的 log-prob 差信号噪声大、容易被这部分干扰；Δ 信号因为消去了 base 的 baseline，**只关注 reasoning tuning 带来的"增量"**，所以在已经训练好的模型上还能继续微调。

在 HMMT25 这种竞赛级 benchmark 上差距尤其夸张：Qwen3-8B 的 HMMT25 从 44.3 涨到 52.3，**OPD² 比 ExOPD 高 6 个点**。

### 跨家族：Gemma4

Gemma4-E4B-it 的实验有两个看点。

**第一，常规 OPD 跨家族会崩**。在 Gemma4 数学上，base 60.6，OPD 反而掉到 58.9，Code 从 55.2 砸到 36.9（掉了快 20 个点）。这跟 Qwen3 上的"温和涨分"完全不同，**说明 OPD 对模型家族/训练范式其实是敏感的**。

**第二，OPD² 在跨家族上依然稳**。Gemma4 数学 60.6 → 67.8，Code 55.2 → 49.5（虽然没涨但保住了大部分能力，OPD 只剩 36.9），Science 47.0 → 48.8。

paper 给出的解释是：Δ 信号"减掉了"teacher 自带的、与 reasoning 无关的家族风格偏好，所以不同家族 base 模型带来的"风格干扰"被消去了，蒸馏信号更纯粹。

### 训练动态

![图4：Qwen3-4B 训练曲线（AIME 24 & 25）](https://arxiv.org/html/2607.15161v1/x6.png)

*图 4：Qwen3-4B non-thinking 在 AIME24 & AIME25 上的训练曲线（pass@1）。三种方法在前 20 步都急速上涨，但之后 OPD 和 ExOPD 逐渐回落到 ~50%，OPD² 一直维持在 60% 以上。**注意所有表格报的是 final step 而不是 peak**——意思是 OPD² 哪怕没在最好的 checkpoint 取数，依然赢。*

CodeContests（未贴图）和 GPQA 上的曲线趋势一致：OPD / ExOPD 在第 20 步后就开始 plateau 或缓慢下降，OPD² 始终保持高水位。说明这个优势不是"刚好踩到好 checkpoint"，是训练轨迹层面的稳定优势。

---

## 消融实验

paper 在 Qwen3-1.7B 上做了三组消融：

| 变体 | Math (NT) | Code (NT) | Science (NT) | Math (T) | Code (T) | Science (T) |
|---|---|---|---|---|---|---|
| 完整 OPD² | 54.6 | 29.4 | 38.8 | 62.7 | 40.4 | 43.5 |
| 无 Δ 信号（换回 OPD） | 50.5 | 22.5 | 35.9 | 57.4 | 32.1 | 41.3 |
| 无联合条件 | 55.8 | 28.8 | 38.8 | 61.5 | 39.3 | 43.8 |
| 无 centering | 54.9 | 28.5 | 38.8 | 63.1 | 39.3 | 43.3 |

**最核心的发现**：去掉 Δ 信号，6 个指标里 5 个都出现最大幅度的下降（Math NT 从 54.6 掉到 50.5，Code NT 从 29.4 掉到 22.5）。这验证了 paper 的标题——**delta signal 才是 OPD² 的灵魂，centering 和联合条件是工程稳定性补丁**。

让我有点意外的是：去掉联合条件在 Math NT 上反而从 54.6 涨到 55.8。这种小幅波动其实是噪声级别的，但说明联合条件不是"必须"的。作者自己也在 ablation 后说了，这两个组件"contribute additional but relatively modest improvements"。

---

## 计算开销

| 方法 | Qwen3-1.7B | Qwen3-4B | Qwen3-8B | Gemma4-E4B |
|---|---|---|---|---|
| OPD | 4.4h | 7.3h | 7.6h | 12.7h |
| ExOPD | 5.3h (+19%) | 9.1h (+26%) | 9.4h (+24%) | 13.8h (+9%) |
| OPD² | 5.5h (+24%) | 9.3h (+28%) | 9.6h (+27%) | 13.8h (+8%) |

多 24-28% 训练时间，主要来自 teacher-base 那次前向。paper 老实承认：当前实现没专门优化这部分，进一步压一压完全可行。

公平地说，**多 25% 时间换 3-4 个点的提升，在小模型蒸馏场景下绝对划算**。1.7B 模型本来训练几小时就结束，多一小时换一个 4-5% 的稳定涨分，工程上几乎不用想。

---

## 我的判断

### 亮点

1. **问题定位精准**。"OPD 的奖励函数本身没被认真设计过"——这是 paper 一开始就挑明的话，比那些"我们提出了 X 方法在 Y 上取得 SOTA"的论文坦率得多。
2. **delta signal 的物理意义非常清晰**。它不是一个为了涨分而拍出来的工程 trick，而是有明确语义：teacher 学到的 reasoning 增量 = teacher 减 base。这个解释对任何人来说都站得住脚。
3. **三组信号分析是真正的干货**。词云、token 级可视化、统计变化——很多 paper 给一张表就完了，这篇是把"为什么这个方法 work"讲透了。
4. **消融诚实**。联合条件去掉反而涨了一个点，作者没藏，照实写出来。
5. **跨家族验证充分**。Gemma4 上 OPD 直接掉 20 个点，OPD² 还能涨，这种 stress test 比单纯在 Qwen3 上跑 3 个尺寸更有说服力。

### 我保留的地方

1. **ExOPD 这个 baseline 选的有点微妙**。paper 里说 ExOPD 是 "state-of-the-art"，但 ExOPD 本身在 Gemma4 Code 上也只能勉强维持原始能力（55.2 → 45.1），和 OPD² 的 49.5 差距也不大。如果想真正证明 Δ 信号"普适性更强"，最好跟其他蒸馏变种（比如最近几个月的 Distill-Mix、Self-Rewarding、Reasoning-Prompting）也对比一下。
2. **teacher-base 的可获得性是隐藏前提**。OPD² 严格要求 teacher 有公开的 base 模型，这对 Qwen3、Gemma 这种开源模型没问题，但对闭源 teacher（GPT、Claude）就完全没法用。paper 没在 limitation 里专门提这一点，我觉得是个疏漏——如果你想用 OPD² 蒸馏一个 Qwen3 学生 + GPT-5 teacher 的组合，对不起，你得知道 GPT-5 的 base。
3. **"短 post-training 周期" 是个营销话术**。100 步 on-policy 蒸馏加 4× 8-GPU 节点的训练，说"短"是相对的。paper 没有跟"完整的 long-form RL 训练"做 head-to-head 对比，所以"短"的优势只是相对于"更长的 RL"——这个比较不算太公平。
4. **centering 在 top-1024 token 上做**。对 vocab size 128k+ 的 Qwen3，1024 占了不到 1%，会不会有偏？这是个 implementation detail，作者没分析，我也没法判断，但实际效果是 work 的。

### 工程上能不能直接抄？

可以。paper 的代码会开源在 naver-ai/opd2，关键实现细节（top-k 期望、联合条件、reward scale 0.1）都在 Table 10/11 里写清楚了。如果你的场景是：

- ✅ 有开源 teacher（Qwen3、LLaMA、Gemma）
- ✅ teacher 家族里有同款 base 模型
- ✅ 你在做小模型蒸馏 post-training
- ✅ 训练时间能多承担 25%

那 OPD² 几乎是 free lunch。**哪怕你不打算用 OPD² 全套，光是"teacher 减 teacher-base"这个 insight 也值得在你的 pipeline 里试试**——我赌大概率有正向收益。

---

## 一些追问

paper 没明说但值得想想的问题：

1. **Δ 信号的"粒度"问题**。paper 默认 teacher 和 base 是"同架构、同 tokenizer、同训练数据预训练"的，但实际工业里 teacher 经常是 fine-tune 过的、backbone 都换过的版本。base 怎么选？是不是可以用"早一版 checkpoint"代替？
2. **联合条件的可解释性**。$A_t^{\Delta} \cdot A_t^{\text{OPD}} > 0$ 这个条件数学上很 elegant，但直觉上更像是"我俩都觉得这个 token 好"。有没有可能 $A_t^{\Delta}$ 单独足够好，联合条件反而限制了 $\Delta$ 的发挥？ablation 里 Math NT 那个 54.6 → 55.8 让人浮想联翩。
3. **跟 RL 的关系**。paper 把 OPD 和 RL 并列（"alternative"），但实际工业里 OPD 经常作为 RL 的暖启动。**OPD² 训完的学生能不能直接接 GRPO 继续练**？如果能，那 Δ 信号和 reward model 的兼容性如何？这个问题 paper 没碰，但实操中大概率有人会问。

---

## 写在最后

整篇 paper 给我最大的感受是：作者**没有把简单问题复杂化**。一个直觉干净的 insight（teacher 减 base），加上一组诚实的实验分析（词云、token 可视化、消融），就足以撑起一篇 19 页的 paper。

不是所有 SOTA 都需要 GRPO、RLHF、Process Reward Model、Tree Search 这套大阵仗。有时候——尤其是在小模型 post-training 这种**边际收益敏感**的场景下——奖励函数层的 5 行代码改动，能换回 4 个点的稳定提升。

这才是工程师最想要的 paper。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
