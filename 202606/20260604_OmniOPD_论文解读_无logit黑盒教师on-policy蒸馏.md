# 教师不给 logit 也能搞 on-policy 蒸馏？OmniOPD 用 chunk 级语义投票把 Claude/Gemini 拉进了蒸馏管线

## 核心摘要

最近 on-policy distillation（OPD）很热——学生自己生成轨迹，老师在每个 token 位置上给 logit 反馈，号称兼顾 SFT 的稳定和 RL 的探索性。听起来很美，**但有一个被一直回避的尴尬：你必须能直接读到老师的 logits**。GPT、Claude、Gemini 这几位现在最能打的老师，全都只给文本不给概率分布——OPD 派对，他们一个都进不来。

更扎心的是，就算能拿到 logits，token 级别的 logit 匹配本身也很脆：老师和学生的高概率 token 集合重叠很窄，一旦风格不同、tokenizer 不一样、或者轨迹陷入 repetition，监督信号要么消失要么直接放大病态模式。

这篇 **OmniOPD**（arXiv 2606.01476）干的事情很直接：**把 token 级 logit 匹配换成 chunk 级语义验证**。学生自己生成轨迹，挑出 entropy 最高的几个"关键岔路口"，每个岔路口圈一个 chunk，老师在前缀上跑 N 次蒙特卡洛 rollout，用 ROUGE-1 或编辑距离算语义重合度——这样一来老师只需要返回文本，黑盒模型也能当老师了。

数字相当能打：在数学推理上相对标准 OPD 最高涨 **28.64 个点**；用 Claude-4.5-Haiku/Gemini-2.5-Flash 当老师，比开源教师再涨 **9.54 个点**，把 Qwen3-4B 直接推过了 GRPO 自探索的天花板。这是一篇我读到一半就在想"对，应该是这个方向"的论文。

---

## 论文信息

- **标题**：OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification
- **作者**：Yuhang Zhou, Lizhu Zhang, Yifan Wu, Mingyi Wang, Peng Bo, Jiayi Liu, Xiangjun Fan, Zhuokai Zhao
- **arXiv**：[2606.01476](https://arxiv.org/abs/2606.01476)（2026 年 5 月 31 日）
- **方向**：On-Policy Distillation / LLM Post-training

---

## 为什么需要这篇论文？标准 OPD 的两个死结

先把 OPD 的来龙去脉讲清楚。SFT 的问题是 **off-policy**：学生学的是老师写的轨迹，自己生成时分布对不上，error 复利累积。RL 的问题是 **稀疏 credit assignment**：通常只有最终答案对错的 reward，链路上每一步对不对说不清。OPD 的解法挺漂亮——学生自己生成（保 on-policy），老师在每一步给 dense 的 logit 反馈（保 dense 监督），两边都救了。

但这套方案有两个绕不过去的现实问题。

**第一个问题：教师必须能给 logits。** 商业 API 全都只给文本。也就是说，OPD 这个范式天然把 GPT-4、Claude、Gemini 全部排除在教师之外——而恰恰是这些模型在数学、代码上有最强的能力。你能用来当老师的，只有自己跑得起的开源模型。**这是一个很难的 ceiling**。

**第二个问题：就算能拿到 logits，token 级 matching 本身也脆。** 论文引用了几篇 2026 年的工作（Li et al., Luo et al., Fu et al.）分析 OPD 的失败模式——有效监督信号其实集中在老师和学生 plausible next token 的重叠区，而这个重叠区非常窄。一旦：

- 老师和学生推理风格不同（比如一个爱用"Let's think step by step"，一个直接列方程）
- 学生陷入 repetition loop 这种退化前缀
- 老师和学生的 tokenizer 不一致（这在跨 family 蒸馏几乎是常态）

监督信号要么蒸发，要么 actively harmful——把退化模式放大。

我之前在做小模型蒸馏的时候碰到过类似的情况：用 Qwen 当老师蒸 Llama 的学生，token 对不上根本没法做 logit 匹配。当时我们的解法很笨——退回到 SFT。这篇论文给的答案是：**别在 token 级别上较劲，往上抬一层**。

---

## OmniOPD 怎么做？三件事 + 两个稳定锚

![图1：OmniOPD 框架概览](https://arxiv.org/html/2606.01476v1/figure/omniopd_diagram.png)

*图 1：OmniOPD 三步走——(1) 学生生成轨迹后用 entropy 找推理岔路口选出 chunk；(2) 老师只在每个被选中的 chunk 上跑 N 次蒙特卡洛 rollout，用语义相似度打分；(3) 损失只作用于审计 token，未审计区域用 KL 锚定防止 policy 漂移。*

整个框架围绕一个核心问题：**没有 logits，怎么造出稠密的 on-policy 监督信号？** OmniOPD 的回答可以拆成三件事。

### 1. Chunk 级监督：把 token 抬到 chunk

不再追求"老师在每个 token 位置上的概率分布"，改成"老师在 C 个连续 token（chunk）上整体上倾向生成什么样的内容"。学生生成的 chunk $c$，老师拿到前缀 $y_{<c}$ 跑 N 次独立的蒙特卡洛 rollout，得到 N 个候选 $\{y_{teacher}^{(i)}\}$。然后用一个连续语义相似度函数 $\phi$（默认是归一化编辑距离，备选 ROUGE-1）打分：

$$k_{sem}^{(c)} = \sum_{i=1}^{N} \phi(y_c, y_{teacher}^{(i)})$$

这一下解决了两个问题：第一，**老师只要返回文本就行**，黑盒 API 全部解锁；第二，监督信号从"必须 token 一致"放宽到"语义接近"，对 tokenizer 差异和风格差异天然鲁棒。论文在定理 4.4 给了一个简洁的不变性证明——只要两套 rollout 在 $\phi$ 度量下等价，损失就完全相同。

代价当然有：每个 chunk 一次教师查询而不是每个 token 一次，调用次数从 $T$ 降到 $T/C$，但每次调用要采 $N$ 个样本——所以总 token 数会上升。这块在论文 Appendix 9 有详细的成本分析。

### 2. Peak-Entropy Chunk Selection：只看推理岔路口

如果在整条轨迹上均匀采 chunk，会非常浪费——大部分 token 都是"the"、"is"、"=" 这种确定性极高的位置，老师审不审计结果都一样。真正有信息量的位置是**推理岔路口**：模型纠结要不要换思路、是用代数还是几何、是继续展开还是回头检查的那种位置。

OmniOPD 的做法是用学生自己的预测熵当 proxy。每一步 $t$ 计算：

$$\mathcal{H}_t = -\sum_{v \in \mathcal{V}} \pi_\theta(v|x, y_{<t}) \log \pi_\theta(v|x, y_{<t})$$

然后挑 entropy 最高的 $M$ 个位置作为 anchor，每个 anchor 周围圈一个 $C$-token 的 chunk 当作"被审计区"。直觉很好理解——熵高的地方就是模型自己也拿不准的地方，监督信号在这里给最值钱；熵低的地方学生自己心里有数，没必要再喂一遍。

我个人挺喜欢这个设计的。它本质上把"哪里需要监督"这件事从 hyperparameter 抬成了**可微调的策略**——而且还是用学生自己的状态来决定。这种"学生自己挑题让老师批"的味道，比死板地等距采样优雅多了。

### 3. Dirichlet-Multinomial 贝叶斯先验：稳定方差

但 chunk 级设计有个新问题：老师只跑 $N$ 次（默认 N=10），是个有限样本估计，方差极大。如果用最朴素的频率估计：

$$\hat{\pi}_{freq}^{(c)} = k_{sem}^{(c)} / N$$

只要老师那 10 次 rollout 一次都没匹配上学生的 chunk，这个估计直接是 0，log loss 立刻爆炸。这是经典的稀疏采样陷阱。

解法是上贝叶斯平滑——拿学生自己 chunk 的归一化概率当先验：

$$\bar{\pi}_\theta^{(c)} = \left(\prod_{t \in c} \pi_\theta(y_t|x, y_{<t})\right)^{1/C}$$

然后用 Dirichlet-Multinomial 共轭得到稳定后的目标：

$$\hat{\pi}_{teacher}^{(c)} = \frac{k_{sem}^{(c)} + \alpha \cdot \bar{\pi}_\theta^{(c)}}{N + \alpha}$$

写成凸组合更直观：

$$\hat{\pi}_{teacher}^{(c)} = \frac{N}{N+\alpha}\hat{\pi}_{freq}^{(c)} + \frac{\alpha}{N+\alpha}\bar{\pi}_\theta^{(c)}$$

$\alpha$ 控制偏差-方差权衡：$\alpha$ 小则估计更接近频率值（方差大但偏差小），$\alpha$ 大则更靠近学生自己的先验（方差小但偏差大）。论文证明了即使 $k_{sem}^{(c)} = 0$ 这个最坏情况，估计也始终大于 $\alpha \bar{\pi}_\theta^{(c)} / (N+\alpha) > 0$，**梯度永远不爆炸**。

### 4. Base-Model KL Anchor：锁住未审计区域

到这里还有一个隐患：审计 chunk 只占轨迹的 $M \cdot C$ 个 token，剩下大约 $T - M \cdot C$ 个 token 是没有任何监督的。如果只在审计点上推梯度，学生很容易学会"在没人看的地方乱写"——比如缩短 trajectory 到只有 audited chunk 长度，或者在中间塞乱码节省算力。

OmniOPD 的解法是在未审计区上加一个 KL 信任域，把学生拉回到自己的初始权重 $\pi_{ref}$（也就是冻结的初始 checkpoint）：

$$\mathcal{L}_{OmniOPD}(\theta) = -\mathbb{E}_{\hat{y} \sim \pi_\theta}\left[\sum_{c=1}^{M} \hat{\pi}_{teacher}^{(c)} \sum_{t \in c} \log \pi_\theta(y_t|x, y_{<t})\right] + \beta \sum_{t \in \mathcal{U}} D_{KL}(\pi_{ref} \| \pi_\theta)$$

通过 Pinsker 不等式可以证明，这等价于约束未审计区的 total variation drift。这个设计后面会看到——它不是"锦上添花"，而是 **OmniOPD 能不能 work 的生死开关**。

---

## 实验结果：数学和代码的双线压制

### 主实验：数学推理

设置很标准：训练用 DAPO-Math-17K，评测覆盖 AIME-2024、AIME-2025、AMC23、MATH-500、OlympiadBench。学生选 Qwen3-1.7B 和 Qwen3-4B，老师有四档：开源（Qwen3-32B、Qwen3-30B-A3B-Instruct）和黑盒（Claude-4.5-Haiku、Gemini-2.5-Flash）。

| 学生 / 老师 | SFT | OPD | OmniOPD | 相对 OPD 提升 |
|---|---|---|---|---|
| Qwen3-4B / Qwen3-32B | 63.80 | 64.16 | **69.08** | +7.7 |
| Qwen3-4B / Qwen3-30B-A3B | 49.77 | 56.22 | **72.32** | 涨 28.64 个点 |
| Qwen3-4B / Claude-4.5-Haiku | 67.52 | N/A | **74.92** | — |
| Qwen3-4B / Gemini-2.5-Flash | 73.51 | N/A | **75.67** | — |

几个观察。

**第一**，开源老师组里 Qwen3-30B-A3B-Instruct 的提升最猛（+28.64%）。这其实暴露了标准 OPD 的脆弱性——A3B 是 instruct 模型，输出风格和 base 学生差异大，token 级 logit 匹配命中率就低。OmniOPD 的 chunk 级语义匹配天然抹平了这种 style gap，所以涨幅最大。

**第二**，黑盒老师组（Claude/Gemini）是 OmniOPD 独享的。论文重点 highlight 的数字是：用 Gemini-2.5-Flash 当老师把 Qwen3-4B 蒸到 75.67%，**直接超过同一学生的 GRPO 自探索结果（70.24%）**。这点挺关键——以前 RL 派的一个论据是"老师再强也有教不出来的东西，自探索才是 ceiling"，OmniOPD 用前沿黑盒老师把这个 ceiling 又拉高了一档。

**第三**，黑盒老师对 SFT 也是 work 的，差距没那么夸张。从 67.52%（SFT）到 74.92%（OmniOPD with Claude）是 +7.40 个点——这部分增益就是"on-policy + chunk 验证"相对纯 imitation 的本质优势。

### 竞争编程

| 学生 / 老师 | SFT | OPD | **OmniOPD** | 相对 OPD |
|---|---|---|---|---|
| Qwen3-1.7B / Qwen3-32B | 40.44% | 47.06% | **47.93%** | +1.85% |
| Qwen3-4B / Qwen3-32B | 62.03% | 65.26% | 63.78% | -2.27% |

代码任务这边就比较诚实——**1.7B 上能赢，4B 上反而输了 2.27 个点**。这是个挺有意思的现象。我的猜测是代码任务对 token 级精确性更敏感（一个变量名错了整段就崩），chunk 级语义匹配在精确度要求高的场景下反而有损耗。论文没在这点上展开太多，但能看出来作者也没刻意藏——主表完整放出来了，没做 cherry-picking。

### 消融实验：KL 锚定才是命门

| 配置 | 平均准确率 | 损失 |
|---|---|---|
| **完整 OmniOPD** | **69.08%** | — |
| 移除 entropy 选择 | 68.45% | -0.63 |
| 移除贝叶斯平滑 | 68.63% | -0.45 |
| **移除 KL 锚定** | **8.28%** | **-60.80** |

这张表是全文最戏剧性的一个数据。entropy 选择和贝叶斯平滑都是"调味料"级别——去掉了下降不到 1 个点。但 **KL 锚定一去掉，性能从 69.08% 直接崩到 8.28%**，掉了 60 个点。

这恰好印证了我前面提到的疑虑：审计 token 只是轨迹的一小部分，未审计区如果不锁住，学生会立刻学会"在没人看的地方乱写"。论文用定理 4.3 给了形式化的证明——没有 KL 信任域的话，policy 在未审计区会迅速漂移到任意位置。8.28% 这个数其实很说明问题：不是退回到 base 模型（base 也有几十个点），而是**模型在训练中 actively 学坏了**。

### 贝叶斯先验 α 的敏感性

![图2：贝叶斯先验 α 的敏感性曲线](https://arxiv.org/html/2606.01476v1/x1.png)

*图 2：先验强度 α 对最终性能的影响。α=1.0 处达到峰值 69.08%，α 太小（0.1）方差爆炸降到 67.52%，α 太大（≥4.0）偏差累积，在 α=5.0 处崩到 60.77%。曲线形状是教科书级别的偏差-方差权衡。*

这个图算是论文中"理论很美但工程要小心"的最好注脚。$\alpha$ 是个范围很窄的甜蜜区——0.5 到 2.0 之间都很稳，但偏离这个范围两侧都会出事。这种 hyperparameter 敏感性在生产里要慎重对待——好在 $\alpha=1.0$ 这个默认值刚好在峰值附近，可以直接用不调。

### 训练动态

![图3：Qwen3-4B 学生在两种教师下的训练动态](https://arxiv.org/html/2606.01476v1/x2.png)

*图 3 (a)：Qwen3-32B 教师下的 on-policy loss 曲线，前 100 步从 0.33 快速降到 0.24 后稳定。*

![图3-b](https://arxiv.org/html/2606.01476v1/x3.png)

*图 3 (b)：Qwen3-32B 教师下的 reference KL 散度——KL 持续小幅上升但保持在合理范围内，说明信任域约束在生效。*

![图3-c](https://arxiv.org/html/2606.01476v1/x4.png)

*图 3 (c)：Qwen3-32B 教师下的 AIME 2025 准确率持续提升，从初始水平稳定爬升到训练结束。*

![图3-d](https://arxiv.org/html/2606.01476v1/x5.png)

*图 3 (d)：换成 Gemini-2.5-Flash 教师后的 on-policy loss 曲线——下降速度和 Qwen3-32B 相当，说明黑盒教师的训练动态非常稳定。*

![图3-e](https://arxiv.org/html/2606.01476v1/x6.png)

*图 3 (e)：Gemini-2.5-Flash 教师下的 reference KL 曲线，整体形态与开源教师对照组一致。*

![图3-f](https://arxiv.org/html/2606.01476v1/x7.png)

*图 3 (f)：Gemini-2.5-Flash 教师下的 AIME 2025 准确率提升曲线——爬升速度比 Qwen3-32B 教师更快，最终达到的高度也更高。*

我看完这六张图最深的一个观察是：**用黑盒教师训练的曲线看起来跟开源教师没什么两样**。这其实是个挺重要的工程信号——之前我担心黑盒老师的"高方差 N=10 蒙特卡洛"会导致训练 noisy 不稳定，但实际曲线很稳。说明贝叶斯平滑 + KL 锚这套组合拳确实把 noise 压下去了。

---

## 我的判断：值不值得跟？

先说结论：**值得，但要看你的场景**。

### 漂亮在哪

**第一个值钱的地方是把 logit 这个枷锁拆掉了**。这个意义不只是"现在能用 Claude 当老师了"——更深层的影响是 OPD 这个范式之前被严格限制在白盒老师的小圈子里，现在可以用任何能产 rollout 的系统当老师，包括（推得极致一点）人类标注、agent 系统、外部检索增强后的输出。chunk 级语义验证打开的是一整片新空间。

**第二个值钱的地方是 entropy-driven chunk selection 这个动作**。它其实可以从 OmniOPD 里拆出来单独用——不管你是 SFT、RLHF 还是 RL，"哪里需要监督"这件事都可以让学生自己的不确定性来决定。我个人觉得这块的方法论意义比 chunk 级 matching 还要长远一点。

**第三个值钱的地方是消融实验的诚实性**。代码任务上 4B 输给标准 OPD 这个数据没藏，KL 锚去掉崩到 8% 这个戏剧性结果也直接放出来。这种"主体很硬，但弱点也明牌"的论文在现在挺难得。

### 让我皱眉的地方

**成本是绕不过去的**。每个 chunk 要采 $N=10$ 次 rollout，意味着教师调用的 token 量比标准 OPD 大很多倍——尤其是用 Claude/Gemini 这种按 token 计费的 API 当老师时。论文 Appendix 9 给了 cost 分析，但实际部署时这是要算账的，不是免费午餐。

**chunk size $C$ 的脆弱性**。论文里 $C=25$ 时性能从 69.08% 崩到 24.48%——这是个非常 sharp 的悬崖。再加上前面提到的 $\alpha$ 也敏感，总的来看 OmniOPD 在生产中调参空间不大。好处是默认值在峰值附近，坏处是换一个 domain 重新调时风险高。

**代码任务的 4B 反例**。这个反例其实不是 bug 而是 feature——它告诉我们 chunk 级语义匹配并不是普适的"更好的监督"，在精确性要求高的场景下反而是损耗。这对工程选型是一个重要提示：**不要无脑 OmniOPD，先想清楚自己的任务特性是更近 reasoning 还是更近 generation**。

**理论框架略偏 retrofit**。四个定理（梯度稳定、集中性、信任域、不变性）确实给了 OmniOPD 不错的理论 backing，但读起来更像"先有了 method，再补的证明"。这不是缺点，但要分清——论文真正的贡献是在工程直觉层面把 OPD 的两个死结都解了，理论是配套保障，不是出发点。

### 适合谁

如果你是在做：
- **小模型 reasoning 蒸馏**：尤其是想用 Claude/Gemini 当老师把 7B 以下学生推到 frontier，这套方案现在是最直接的路径
- **跨 family 蒸馏**：不同 tokenizer / 不同 architecture，token 级 OPD 卡死，OmniOPD 几乎是唯一解
- **复杂推理任务**：数学、agent planning 这类需要"在岔路口正确决策"的任务，entropy chunk 选择会非常受用

如果你是在做：
- **代码生成**：先看看你的任务对 token 级精确性的依赖度——可能标准 OPD 或者 SFT + RL 还是更好的选择
- **格式化输出**（JSON、SQL）：chunk 级语义松弛对这类任务会引入风险

---

## 收尾

OmniOPD 的核心洞察其实可以浓缩成一句话：**当 token 级监督信号脆而稀时，往上抬一层到 chunk 级，配合 entropy 选点和 KL 锚定，能换来更鲁棒、更通用、且兼容黑盒教师的 dense 监督**。

这种"换个粒度看问题"的思路其实很多 RL 工作都在用——比如 PRM 把 reward 从 outcome 级抬到 step 级，agent RL 把动作从 token 级抬到 tool call 级。OmniOPD 把这套抬粒度的逻辑用到了 OPD 上，而且赶上了一个非常好的时间窗口：黑盒前沿模型能力大幅领先开源老师。

我个人对这条线的判断是——**chunk 级 + 黑盒老师的蒸馏，未来一年大概率会成为小模型 post-training 的标配工具**。原因很简单：算账。当 Gemini-2.5-Flash 这种老师便宜到能批量当 teacher 时，你愿不愿意为了 5-10 个点的效果再多花一倍 API 费？大部分团队会愿意。

如果你正在做小模型 reasoning 训练，建议这两件事可以马上试：第一是把 entropy-driven chunk selection 这个动作单独抽出来叠加到现有 pipeline 上，零成本验证；第二是用 Gemini-2.5-Flash 跑一个 mini OmniOPD 实验对比下你现有的 SFT/RL baseline，看看黑盒老师到底能给你的特定任务带来多少 ceiling。

最后留一个我自己也没完全想明白的问题——OmniOPD 本质上把"老师的偏好"压缩成了"chunk 级语义相似度"这一维信号。但很多 reasoning 错误其实不是 chunk 级的（写出来语义上很像，但底层 logic 错了）。这种 deeper 的错误，OmniOPD 能不能监督到？我的直觉是不太够，可能还需要再叠一层 outcome 验证。但这是另一篇论文的事了。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我*
