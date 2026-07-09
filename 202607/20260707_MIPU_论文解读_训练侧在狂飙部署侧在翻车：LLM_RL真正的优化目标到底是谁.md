# 训练侧在狂飙，部署侧在翻车：LLM RL 真正的优化目标到底是谁

你有没有碰到过这种情况——RL 后训练跑起来，pass@1 在训练集上一路高歌猛进，结果一上线推理，模型反而答得更差？或者更糟，跑着跑着训练分数突然断崖式下跌，直接回到解放前？

这篇来自阿里 + 天津大学的论文 [The Mirage of Optimizing Training Policies: Monotonic Inference Policies as the Real Objective for LLM Reinforcement Learning](https://arxiv.org/abs/2606.29526)（arXiv:2606.29526，2026 年 6 月）就盯着这个问题。它的核心观点很扎心：**我们一直在优化训练策略，但上线用的是推理策略，这两个东西根本不是一回事。** 论文把这叫 "Objective Misalignment"（目标错位），并提出了一套叫 **MIPU**（Monotonic Inference Policy Update）的两步训练框架。

说真的，这篇论文对我启发最大的是它把训练-推理不一致这个"工程现象"提升到了"目标层面"——以前我们把它当作一个系统 bug 在修（更精确的精度、更小的 KL、更稳的 LR），作者说：不对，这根本是优化目标选错了。

---

## 核心摘要

**痛点**：LLM RL 后训练常出现"训练分数高但推理质量崩"或"训练中突然崩溃"的现象。根因是训练引擎（FSDP/Megatron）和推理引擎（vLLM/SGLang）参数同步后，对同一条轨迹仍会给出不同概率——这造成了持续的 off-policyness。

**核心洞察**：现有工作（TIS、MIS、LR-decay、FP16、QuRL 等）都在"修正"训练侧的更新，让训练策略看起来在变好。但作者说：你修的是训练策略 π，真正上线的是推理策略 μ。这两者的 transition 可以完全背离。提升训练策略不等于提升推理策略。

**方案**：提出 **MIPI**（Monotonic Inference Policy Improvement）作为新优化目标，并把目标函数分解成三项：(1) post-update inference gap、(2) training-side update、(3) pre-update inference gap。**MIPU** 框架分两步实现——Step 1 用截断的 sampler-referenced 重要性权重（相对 μ 而非 π）构造候选更新；Step 2 用后更新推理 gap 作为代理信号决定是否接受同步候选。

**效果**：在 FP8-quantized rollout 这个高不匹配设置下，Qwen3-4B 的平均 pass@1 从 baseline 的 64.42 提到 66.71（+2.29 个点），Qwen3-1.7B 从 50.86 提到 53.97（+3.11 个点）。但更关键的是，**所有 baseline 都在训练后期崩溃了，MIPU 是唯一一个稳定收敛的**。

**一针见血的评价**：这是一个"问题重新定义"型的工作。把工程问题升维成目标问题，迫使后续工作必须回答"你在优化谁的策略"。但方法本身严格说并没有提供形式化单调性保证（论文自己承认了），更像是一套经验性的过滤器。在我看来真正的贡献是那个三段分解公式——它把混乱的工程现象拆成了一个可以逐项讨论的优化目标。

---

## 论文信息

- **标题**：The Mirage of Optimizing Training Policies: Monotonic Inference Policies as the Real Objective for LLM Reinforcement Learning
- **作者**：Jing Liang, Hongyao Tang, Yi Ma（共同一作）, Yancheng He, Weixun Wang（通讯）, Xiaoyang Li, Ju Huang, Wenbo Su, Jinyi Liu, Yan Zheng（通讯）, Jianye Hao（通讯）, Bo Zheng（通讯）
- **机构**：天津大学，阿里巴巴
- **发表日期**：2026-06-28
- **论文链接**：https://arxiv.org/abs/2606.29526
- **项目页**：https://anitaleungxx.github.io/MIPU

---

## 一、问题动机：你修的策略，根本不是你在用的那个

要把这个问题讲清楚，得先理解 LLM RL 的基础设施现状。

现代 LLM RL 流水线基本都是"两条腿走路"：

- **推理引擎**（vLLM、SGLang）：负责 rollout 阶段采样轨迹，要求快
- **训练引擎**（FSDP、Megatron）：负责算 log-probability 和梯度更新，要求精度

这两条腿是分开跑的，每隔若干步同步一次权重。理论上同步之后应该是同一个模型，对同一条轨迹给出相同概率。但实际上，由于量化精度、attention 实现、kernel 选择等差异，同步后的两个引擎对**同一条轨迹**给的概率就是不一样的。

这种"概率不一致"是 LLM RL 训练不稳定的公认元凶之一。近两年的工作主要从三个方向修这个问题：

| 方向 | 代表工作 | 思路 |
|------|---------|------|
| 训练侧算法修正 | TIS（Yao et al., 2025）、MIS（Liu et al., 2025） | 用 trainer-to-sampler 重要性比做截断校正，或过滤掉极端 mismatch 的 token/序列 |
| 优化器侧修正 | LR-decay（Zhang et al., 2026） | 学习率衰减，让训练过程在 mismatch 累积到崩溃前手动刹车 |
| 系统侧修正 | QuRL（Li et al., 2026）、FP16（Qi et al., 2025） | 用 FP16 量化或低精度推理减小 mismatch 来源 |

但作者说，**所有这些工作都在"训练侧"做文章**——它们的目标函数还是 J(π_{k+1}) - J(π_k)，只是修得让训练更稳。

问题来了：即使你修得再稳，**训练侧在变好 ≠ 推理侧在变好**。

下面这张图是整篇论文的核心图，它把这个错位讲得清清楚楚：

![MIPU 框架总览：训练侧优化 vs 推理侧真实目标](https://arxiv.org/html/2606.29526v1/x2.png)

*图 1：MIPU 框架总览。上半部分是 Canonical RL 的流程：训练侧更新 → 同步到推理侧 → 没有 inference-side 验证 → 同步后可能是 beneficial 也可能是 risky；中间是 Objective Misalignment：训练侧提升 J(π_{k+1}) - J(π_k) ≥ 0 推不出推理侧提升 J(μ_{k+1}) - J(μ_k) ≥ 0；下半部分把推理侧提升分解为三项，并给出 MIPU 的两步实现。*

我自己读这张图第一反应是 `"**哦，原来我一直把'我让 loss 下降'等同于'我让模型变好'了**"`。但仔细想想，这里有一个微妙的认知偏差——在大多数 RL 设定里，训练侧策略和推理侧策略是同一个东西，所以目标错位根本不会出现。但 LLM RL 出于工程效率把这两者物理分离了，错位就成了一种常态。

---

## 二、MIPI 公式：把推理侧提升拆成三项

作者的解法很简洁——既然你想保证推理侧 J(μ_{k+1}) - J(μ_k) ≥ 0，那我就在数学上把它拆成可以分别优化的三项。

引入 π_{k+1}（更新后的训练策略）作为中间桥梁，得到：

$$
J(\mu_{k+1}) - J(\mu_k) = \underbrace{J(\mu_{k+1}) - J(\pi_{k+1})}_{\text{① post-update inference gap}} + \underbrace{J(\pi_{k+1}) - J(\pi_k)}_{\text{② training-side update}} + \underbrace{J(\pi_k) - J(\mu_k)}_{\text{③ pre-update inference gap}}
$$

三项的物理意义很直白：

- **② 是我们本来就在做的事**：在训练引擎里用 GRPO/PPO 之类的算法更新 π
- **③ 是同步偏差**：π 和 μ 同步之后仍然存在的概率差。这个值通常较小（毕竟是同步过的），但 FP8 之类的量化会把它放大
- **① 是真正的未知数**：同步之后，推理侧 μ_{k+1} 是不是真的兑现了 π_{k+1} 的"承诺"？这才是 mismatch 的核心所在

作者把这个公式叫做 **MIPI**（Monotonic Inference Policy Improvement）原则。它的逻辑是：要保证推理侧提升，三项必须**合起来** ≥ 0，单看 ② 是不够的。

坦率讲，这个公式本身并不复杂。但它提供了一个非常重要的"分析工具"——以前讨论 LLM RL 不稳定只能模糊地说"KL 飘了"、"reward 塌了"，现在可以明确说"是 ① 那项没被控制"。

---

## 三、MIPU 框架：两步实现 MIPI

有了上面的分解，MIPU 的两步实现就很自然了：

### Step 1：Sampler-Referenced Policy Update（优化 ②+③）

传统的 GRPO surrogate 用的是 π_θ/π_k 作为重要性权重（centered at training policy）。但 rollout 实际是从推理侧 μ_k 采样的，这个权重就和数据来源对不上了。

MIPU 改成 π_θ/μ_k 风格（centered at sampler）。具体地，把重要性比分解为两部分：

$$
\rho_i(\theta) = \underbrace{\frac{\pi_k(y_i|x)}{\mu_k(y_i|x)}}_{w_i^k:\ \text{pre-update mismatch}} \cdot \underbrace{\frac{\pi_\theta(y_i|x)}{\pi_k(y_i|x)}}_{r_i(\theta):\ \text{current update}}
$$

- w_i^k 是 π_k 和 μ_k 之间的预同步偏差（量化越狠越大）
- r_i(θ) 是当前更新的方向

三种"修正 w_i^k 但 clip 方式不同"的实现：

| 方法 | 权重 | Clipping | 效果 |
|------|------|----------|------|
| PPO-IS | w_i^k · r_i(θ) | 对整个 ρ clip | w_i^k 已经在 clip 范围外时可能过度限制 |
| Vanilla-IS | w_i^k · r_i(θ) | 只对 r_i(θ) clip | 无界 w_i^k 会带来大方差 |
| TIS（论文采用） | w̄_i^k · r_i(θ) | 只对 r_i(θ) clip | w̄_i^k = min(w_i^k, w_max=2) 截断，平衡稳定性和方差 |

Step 1 的目标函数是：

$$
\mathcal{J}_{\mathrm{S1}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{T_{i}}\sum_{t=1}^{T_{i}} \bar{w}^{k}_{i,t}\,\ell^{\mathrm{dc}}_{i,t}\right]
$$

ℓ^{dc} 是 dual-clipped GRPO loss（论文 baseline 也用这个，所以比较公平）。

### Step 2：Inference-Gap-Aware Acceptance（验证 ①）

Step 1 训出 π_{k+1} 之后，同步到推理引擎得到 μ_{k+1}。但这个 μ_{k+1} 是不是真的兑现了 π_{k+1} 的"承诺"？直接测 J(μ_{k+1}) - J(π_{k+1}) 没法算（需要 μ_{k+1} 自己的 advantage）。作者用反向恒等式转换：

$$
T_{\mathrm{post}} = -\Delta(\pi_{k+1}, \mu_{k+1}) = -\mathbb{E}_{s,a\sim d^{\pi_{k+1}}}\left[A^{\mu_{k+1}}(s,a)\right]
$$

然后用一个稳定的代理：

$$
\widehat{T}_{\mathrm{post}} = -\mathbb{E}_{x\sim\mathcal{D}_{\mathrm{val}},\,y_i\sim\mu_{k+1}}\left[\rho_i \hat{A}^{\mu_{k+1}}_i\right]
$$

其中 ρ_i 是长度归一化的序列级重要性比。这个代理在 validation 集上采样几条响应就能算。

**接受准则**：

$$
\widehat{T}_{\mathrm{post}} \geq -c \quad\Rightarrow\quad \text{accept} \quad\text{else} \quad \text{rollback to } (\pi_k, \mu_k)
$$

c 是一个 tolerance 参数，对代理噪声的容忍度。作者做了一个挺巧妙的设计：c 是动态的，前 100 步从 c_start 线性衰减到 c_end（4B 上 c_start=1e-3, c_end=0；1.7B 上 c_start=4e-3, c_end=1e-3），因为前期 T̂_post 本来就低，固定阈值会过度拒绝。

直觉上可以这么理解 Step 2：π_{k+1} 提交了一个候选 μ_{k+1}，我们用推理侧的表现反向打一个分，分太差的"打回重做"。这个机制不提供形式化单调性保证（论文自己承认了），但经验上能挡住 mismatch 累积导致的崩溃。

---

## 四、实验：FP8 高不匹配下，MIPU 是唯一不崩的

### 4.1 实验设置

- **模型**：Qwen3-1.7B、Qwen3-4B
- **训练设置**：FP8-quantized rollout（人为放大 mismatch 模拟高不匹配场景）
- **训练数据**：DAPO-Math-17（1.7B 用了 5759 条筛选样本）、DeepMath-103K（4B 用了 1491 条筛选样本）
- **评估基准**：MATH-500、AIME24（avg@16）、AMC23（avg@16）、Minerva、OlympiadBench
- **关键超参**：learning_rate 1e-6、num_return 8、batch 64、dual_clip、KL 系数 0.001、response_length 8192
- **硬件**：H100 × 8
- **框架**：ROLL（阿里自研 RL 框架），训练引擎 Megatron + 推理引擎 vLLM

注意一个细节：**论文没有和 FP16/BF16 rollout 下的基线对比**。这意味着所有数字都是"高 mismatch 条件下的胜负"，而不是普适的 SOTA。这也是我后面要吐槽的点之一。

### 4.2 主结果

| Model | Method | MATH-500 | AIME24 | OlympiadBench | Minerva | AMC23 | Avg. | Stable |
|-------|--------|----------|--------|---------------|---------|-------|------|--------|
| Qwen3-4B | Baseline (GRPO) | 89.34 | 42.00 | 64.89 | 43.39 | 82.50 | 64.42 | ✗ |
| Qwen3-4B | MIS | 90.95 | 38.44 | 62.50 | 44.12 | 81.09 | 63.42 | ✗ |
| Qwen3-4B | LR-decay | 90.34 | **44.00** | 67.26 | 43.75 | 82.97 | 65.66 | ✗ |
| Qwen3-4B | **MIPU (Ours)** | **91.15** | 43.56 | **67.86** | **45.96** | **85.00** | **66.71** | ✓ |
| Qwen3-1.7B | Baseline (GRPO) | 83.10 | **25.33** | 56.55 | 31.68 | 57.66 | 50.86 | ✗ |
| Qwen3-1.7B | MIS | 81.29 | 24.67 | 58.33 | **34.19** | 60.16 | 51.73 | ✗ |
| Qwen3-1.7B | LR-decay | 82.09 | 26.00 | 58.93 | 28.68 | **65.47** | 52.23 | ✗ |
| Qwen3-1.7B | **MIPU (Ours)** | **86.52** | 24.67 | **59.52** | 33.82 | 65.31 | **53.97** | ✓ |

看几个有意思的点：

- **Stable 列只有 MIPU 是 ✓**，其他全是 ✗。这是论文最想强调的故事——**分数高不高先不论，至少没崩**
- Qwen3-4B 上 MIPU 平均分比 baseline 高 2.29 个点（66.71 vs 64.42），比 LR-decay 高 1.05 个点
- Qwen3-1.7B 上 MIPU 平均分比 baseline 高 3.11 个点（53.97 vs 50.86）
- 一些 baseline 在某些数据集上甚至能超过 MIPU（比如 1.7B 上的 AIME24 baseline 25.33 > MIPU 24.67，LR-decay 的 AMC23 65.47 > MIPU 65.31），但这些"亮点"都是昙花一现

但光看表格看不出"崩溃"是什么样子。下面这张训练曲线是关键：

![FP8 rollout 下不同方法的训练曲线](https://arxiv.org/html/2606.29526v1/x3.png)

*图 2：FP8-quantized rollout 下 Qwen3-4B 的训练曲线。Baseline（灰）在 150 步左右直接掉到 0.6 以下；MIS（红）虽然早期稳定但 240 步左右训练崩溃；LR-decay（黄）前 200 步表现不错但 300-450 步之间断崖式跌到 0.1；MIPU（橙）从一开始稳定上升到 0.85 一直保持。*

这张图很能说明问题。**当 reward 信号在 0.85 附近徘徊时，所有方法看起来都还行；真正区分它们的是 200 步之后谁还站着**。MIS 跑了 240 步就断了（崩溃了），LR-decay 跑到 480 步左右崩了，baseline 在 220 步就消失，只有 MIPU 跑满 780 步还稳定。

### 4.3 消融：Step 1 和 Step 2 各自能干啥

| Method | MATH-500 | AIME24 | OlympiadBench | Minerva | AMC23 | Avg. |
|--------|----------|--------|---------------|---------|-------|------|
| Baseline | 89.34 | 42.00 | 64.89 | 43.39 | 82.50 | 64.42 |
| + Step 1 only（TIS） | 90.34 | 41.11 | **68.45** | 44.85 | 82.03 | 65.36 |
| + Step 2 only | 90.34 | 40.44 | 64.88 | 43.38 | 75.00 | 62.81 |
| + Step 1 + Step 2（Ours） | **91.15** | **43.56** | 67.86 | **45.96** | **85.00** | **66.71** |

消融说明了几个关键点：

- **Step 1 单独用（等价于 TIS）**：平均 65.36，比 baseline 高 0.94 个点。它能改善候选更新的方向（OlympiadBench 68.45 是全表最高的），但仍然会崩溃
- **Step 2 单独用**：平均 62.81，**比 baseline 还差**。原因是它拒绝了很多候选，但候选本身质量不高，结果就是"原地踏步"。论文里说"Step 2 mostly keeps the previous inference policy rather than creating further improvement"——这个观察很重要，光过滤不行，还得有好的候选
- **Step 1 + Step 2 组合**：平均 66.71，比单独的 Step 1 高 1.35 个点，AMC23 直接飙到 85.00。两者是互补的

下面这张图更直观：

![消融实验的 4 个训练曲线](https://arxiv.org/html/2606.29526v1/x4.png)

*图 3：Qwen3-4B FP8 设置下消融实验的四个子图：Reward (pass@1)、Mismatch-K3 (训练-推理 KL)、Inference Gap (T̂_post)、Rollback Rate（100 步滑动平均）。baseline（红）在 Reward 子图里 200 步后崩溃；+Step 1（橙）和 +Step 2（浅橙）以及 Ours（黄）的 Rollback Rate 在 400 步后都上升到 0.5 以上，但只有 Ours 维持了高 Reward。*

我特别想指出 **Rollback Rate** 这个图。Step 1 的 rollback 率从一开始的 0.5 持续下降到 0.25 左右（说明它对候选越来越满意），而 Step 2 / Ours 的 rollback 率从 0.5 上升到 0.9（说明它越来越挑剔）。这个对比在视觉上很能说明问题——**Step 1 是个温和的改进者，Step 2 是个严格的守门员**，两者配合才有效。

### 4.4 Step 2 是不是"靠多拒绝"才稳？

这一节我必须单独拎出来说——因为这是我觉得论文里最有说服力的一组实验。

要回答的问题是：Step 2 之所以有效，会不会只是因为它拒绝了更多更新？

实验对比：把 rollback 概率随机设到 70%（比 Step 2 的 53.5% 还高），结果：

![Step 2 vs. 随机回滚的对比](https://arxiv.org/html/2606.29526v1/fig/random.png)

*图 4(b)：Step 2 vs. 随机回滚对比。深橙是 Step 2，浅橙是 Random rollback。Random 拒绝了 67% 的更新（比 Step 2 的 53.5% 更保守），但在 200 步之后开始崩，300 步跌到 0.15 附近。Step 2 拒绝了 53.5% 的更新却一直稳定在 0.85 以上。*

这个实验很关键。它说明 Step 2 不是个"通用稀疏化机制"——**随机拒绝更多更新照样崩**。区别在于 Step 2 拒绝的是"有害的"，而随机拒绝不分青红皂白。

论文里还对比了 4B 和 1.7B 的 mismatch 强度：

![不同模型规模下的 mismatch 强度对比](https://arxiv.org/html/2606.29526v1/x5.png)

*图 4(a)：Qwen3-4B FP8（橙）和 Qwen3-1.7B FP8（红）的 Mismatch-K3 和 Inference Gap 对比。1.7B 的 KL 更大（峰值 0.016 vs 0.011）、Inference Gap 更不稳定（震荡到 ±0.0008）。这说明小模型对 FP8 量化更敏感，需要更大的 c tolerance（c_start=4e-3 vs 4B 的 1e-3）。*

### 4.5 Step 1 变体对比

作者在 Appendix B 里还对比了 PPO-IS、Vanilla-IS、TIS 三种 Step 1 实现：

![Step 1 三种实现的对比](https://arxiv.org/html/2606.29526v1/x6.png)

*图 5：Qwen3-4B FP8 设置下 Step 1 三种实现的对比。Reward 子图里 PPO-IS（红）200 步后崩溃；其他三个都稳定。Grad Norm 子图里 PPO-IS 的梯度范数明显低于其他三个（被过度限制了）。Clip Ratio 子图里 TIS 的 clip ratio 是 7.18e-3，比其他三个（1.8e-04 到 2.10e-04）高一个数量级。*

最有意思的细节是 **Clip Ratio** 这一项。TIS 的 clip ratio 比其他高一个数量级，说明它在更新时更激进（不轻易触发 clip），但反而因此获得了更稳定的训练。这反过来说明 PPO-IS 那种"对完整 ρ_i clip"的方式确实过度限制了更新——π 和 μ 之间的系统性偏差 w_i^k 让 clip 频繁触发，gradient signal 被削弱了。

### 4.6 容忍度 c 的敏感性

最后一个实验是 c tolerance 的敏感性：

![c tolerance 的敏感性](https://arxiv.org/html/2606.29526v1/x7.png)

*图 6：Qwen3-4B FP8 设置下不同 c 取值的对比。c=0（灰）200 步后崩溃；c=-0.0001（橙）200 步前训练很慢；c=0.0001（红）和 Ours（黄）效果最好。*

注意论文里 Ours 在 Qwen3-4B 上设的是 c_start=1e-3, c_end=0（动态），等同于 c=0 的更激进版本。但图里 c=0 反而崩了。原因是 c=0 没有"前期宽松、后期严格"的过渡——前期 T̂_post 本来就低（图中能看到训练早期 Inference Gap 在 -0.0001 以下），c=0 会过度拒绝。**这个动态 tolerance 设计看似细节，其实是 MIPU 能 work 的关键。**

---

## 五、我的判断：问题重新定义 > 方法贡献

读到论文最后，我反而对 MIPU 本身没那么兴奋，倒是被那个三段分解公式启发了一下。

### 5.1 这篇论文真正值钱的地方

把训练-推理 mismatch 从"工程问题"重新定义为"目标问题"，这是我觉得最值的贡献。TIS、MIS、LR-decay 这些工作都在说"我们要修 mismatch"，但都默认优化目标 J(π_{k+1}) - J(π_k) 是对的。**作者说，不对，你优化的对象错了**——这种视角转换在研究里比方法本身更稀缺。

具体到工程上，那个分解公式把以前模糊的"模型在 FP8 下不稳"拆解成了"①没被控制"这种可量化、可讨论、可改进的子问题。以后我们再看到 RL 训练曲线上的异常，至少可以问"是哪一项没控制好"。

### 5.2 但方法本身有水分

说几个不太满意的地方：

**第一，"Monotonic"这个帽子有点大**。论文标题叫 Monotonic Inference Policy Improvement / Update，但实际方法并没有形式化单调性保证。Step 2 的 T̂_post 是个 noisy proxy，c tolerance 是经验值，rollback 也是离散的二值决策。论文里也老老实实说了"this mechanism does not provide a formal monotonic-improvement guarantee"，但读者很容易被标题误导。我理解它的意图是"工程上尽量朝单调性方向走"，但叫"Monotonic"还是有点 over claim。

**第二，实验设置对 MIPU 有点偏 friendly**。FP8-quantized rollout 是人为放大的高 mismatch 场景，在这个场景下所有 baseline 都会崩。如果在 BF16 或者 FP32 rollout 下重新跑一次，可能 TIS 本身表现就不错了，MIPU 的额外收益会被大幅压缩。论文没给这个 baseline 对比，我只能猜测。

**第三，Step 1 实质上就是 TIS**。作者自己承认 Step 1 的实现就是 TIS 风格的截断重要性权重。真正属于 MIPU 的新东西主要是 Step 2。消融里 Step 1 only 跑出来的 65.36 平均分，和 TIS 原始论文里的成绩应该差不了太多。

**第四，T̂_post 引入的额外开销没说清楚**。validation rollout 本身就要采样响应+算 log-prob，这一步会不会拖慢整体训练流程？Step 2 接受候选的话这部分开销是"白花"的（接受了还要重做一次吗？还是只算 proxy 就行？）。论文没给出 wall-clock 时间对比。

**第五，规模限制**。论文实验只在 1.7B-4B 模型上跑。Limitations 章节也坦承 "future work should examine whether the same mismatch patterns also appear in larger models"。但大模型训练 7B-70B 之间 FP8 rollout 也很常见，缺这一段的实验是挺可惜的。

### 5.3 和同期工作的关系

同期类似问题的工作挺多，这里列一下我从论文和相关搜索里看到的：

| 工作 | 方法 | 与 MIPU 的关系 |
|------|------|----------------|
| TIS (Yao et al., 2025) | trainer-to-sampler ratio 做 clip correction | MIPU 的 Step 1 直接采用 TIS 风格 |
| MIS (Liu et al., 2025) | token/sequence 级 mismatch filtering | 被作为 baseline 比较；强调"过滤"和"接受"策略差异 |
| LR-decay (Zhang et al., 2026) | 学习率衰减 | 论文里也作为 baseline，证明它不能根本解决 mismatch |
| QuRL (Li et al., 2026) | 低精度 RL | 系统级修复，论文里没直接比 |
| FP16 (Qi et al., 2025) | 用 FP16 取代 FP8/BF16 rollout | 同样系统级修复 |

可以看到，这片田地过去一年很热闹，但大家的切入点都不同——有的在算法层、有的在系统层、有的在数据层。MIPU 选了"目标层"，这个角度我个人觉得是更根本的，但**还需要在更多模型规模、更多推理设置、更多 RL 算法上验证**。

### 5.4 适合谁来读、用

- **做 LLM RL 训练的工程同学**：值得读，重点看那个三段分解——以后排查训练不稳定时可以按这个框架分类
- **做 RL 理论的同学**：谨慎对待"Monotonic"这个名词，作者自己也没给形式化保证
- **做系统优化的同学**：Step 2 的 T̂_post 思路可以借鉴到其他领域（比如 off-policy 评估）

如果你真要复现，注意几个工程细节：动态 c tolerance 是个 trick（不是论文亮点但没它不 work）、用 dual-clipped GRPO loss 作为基础（论文 baseline 也有）、w_max=2 这个超参在 TIS 里也用过所以相对通用。

---

## 六、写在最后

这篇论文最让我感慨的不是方法，而是**对"工程问题"和"目标问题"的区分能力**。

TIS、MIS、LR-decay、FP16、QuRL……这些工作都在说"我们有办法让训练更稳"，MIPU 站出来说"等一下，你确定你修的是对的东西吗？"

当然这个区分也有点事后诸葛——一旦说穿了"训练侧 ≠ 推理侧"，很多人会觉得"这不是显然的吗"。但显然的事情在工程里经常被忽略——尤其是在 GRPO 类算法不断刷分的环境下，能跳出"SFT → GRPO → 看分"的循环去思考"我到底在优化什么"的研究者并不多。

所以如果让我用一句话总结这篇论文，我会说：

> **MIPU 不是 GRPO 的替代品，而是一面镜子——它照出了 LLM RL 训练中"训练目标和部署目标不一致"这个被长期忽视的工程真相。**

这个真相会越来越重要。随着 RL 后训练扩展到更大的模型、更长的 context、更多的 step，mismatch 累积会越来越严重。"修了又崩，崩了再修"的循环必须被一个更根本的视角打破——MIPU 在这个方向上迈了一步，虽然方法本身还在经验阶段。

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
