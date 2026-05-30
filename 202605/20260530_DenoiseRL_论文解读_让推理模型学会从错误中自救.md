# 别总找更强老师了，让推理模型从错误里爬出来

你有没有遇到过这种训练推理模型的尴尬：模型刷题刷得越来越像样，reward 也在涨，可一碰到稍微绕一点的数学题，它还是会沿着一个早早犯下的小错一路狂奔。更麻烦的是，这种错不一定发生在最终答案那一刻，而是藏在中间推理里：一个变量设错、一个条件漏读、一个枚举范围偏掉，后面再怎么写得头头是道，也只是把错误包装得更漂亮。

DenoiseRL 这篇论文抓住的就是这个点。

它没有继续往“找更强 teacher”“构造更难数据”“堆更多 rollouts”那条路上卷，而是干了一个挺反直觉的事：让弱模型先生成错误推理，把错误轨迹截一段塞给当前策略模型，然后训练它从这个“烂摊子”里把解题过程救回来。

这事听起来像折磨模型，但思路很干净。

---

## 核心摘要

DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes，arXiv:2605.28421，来自 Caijun Xu、Changyi Xiao、Zhongyuan Peng、Yixin Cao。论文把弱模型的错误推理轨迹当成结构化噪声，不再把弱模型当老师，而是把它当“错误状态生成器”。训练时，策略模型一部分正常从题目开始解，另一部分从弱模型错误前缀后继续生成；最终答案对了就给奖励，PPO/GRPO 更新时只更新策略模型自己生成的 continuation，不碰离线 prefix。实验上，在 Qwen3-4B-Base 上，GRPO 平均分从 39.6 提到 42.0；在 Qwen3-8B-Base 上，DAPO 平均分从 42.8 提到 44.8。我的判断是：这不是那种换个 loss 名字包装成新范式的论文，它最有价值的地方在于把“错误恢复能力”变成了可训练的状态分布扩展问题。

---

## 论文信息

| 项目 | 信息 |
|---|---|
| 论文 | DenoiseRL: Bootstrapping Reasoning Models to Recover from Noisy Prefixes |
| arXiv | arXiv:2605.28421v1 |
| 作者 | Caijun Xu, Changyi Xiao, Zhongyuan Peng, Yixin Cao |
| 机构 | Fudan University；Shanghai Innovation Institute |
| 提交时间 | 2026-05-27 |
| 代码 | https://github.com/ALEX-nlp/DenoiseRL |
| 任务 | 数学推理与通用推理强化学习训练 |

---

## 这篇论文在反驳一个默认假设

现在训练 reasoning model，大家默认有几条路：

| 路线 | 直觉 | 问题 |
|---|---|---|
| 更强 teacher 蒸馏 | 让强模型教弱模型 | 贵，而且能力上限容易被 teacher 锁住 |
| 人工筛难题 | 给模型更难的训练样本 | 数据工程很重，难度分布不好控 |
| 普通 on-policy RL | 模型自己采样，答案对了就奖励 | 错误状态覆盖不够，模型很少学会“半路修车” |
| DenoiseRL | 把弱模型错误前缀当训练入口 | 逼模型从错误推理状态恢复 |

说实话，我第一反应是：这和 denoising autoencoder 很像。

在 BART、去噪自编码器那条线里，训练目标不是从干净输入复制干净输出，而是先把输入弄脏，再让模型恢复。DenoiseRL 把这个思路搬到了 reasoning RL 里：错误推理前缀就是噪声，正确解题路径就是要恢复出来的结构。

但这里有个关键差别。

文本去噪通常有标准答案，reasoning RL 里没有完整的中间步骤标注。DenoiseRL 不要求“修复后的每一步”都被监督，只看最终答案。模型能不能从错误里爬出来，全靠终止奖励和组内 advantage 去推动。

这就有点意思了。

---

## 方法一句话：让模型从坏开局里练自救

DenoiseRL 的训练 batch 里同时放两类 rollout：

- main rollout：正常从问题 $q$ 开始解题；
- denoise rollout：先给模型一段弱模型生成的错误前缀 $w_{1:p}$，再让它继续写。

弱模型在这里不是 teacher。

它更像一个“故障注入器”。

![图1：DenoiseRL 的整体流程](https://arxiv.org/html/2605.28421v1/x1.png)

*图1：弱模型先生成错误推理路径，DenoiseRL 截取其中一段 confusing calculation / wrong theory / error accumulation 作为 noisy prefix。策略模型不是模仿弱模型，而是从这个错误状态切回正确路径，最终得到 verified answer。这个图最重要的信息是：红色路径负责制造坏开局，绿色路径才是策略模型要学会的恢复路线。*

更具体一点，训练前先离线构造错误轨迹池。对每个问题 $q \in \mathcal{D}$，弱模型 $\pi_w$ 多次采样答案，然后 verifier 过滤出错误解答，形成 $\mathcal{W}(q)$。

这一步是离线的。

论文里用的弱模型是 Qwen2.5-1.5B-Instruct，在 MATH-7.5K 上每题采样 8 条 rollout，只保留 verifier 判错且格式合规的轨迹。训练时不再反复调用弱模型，成本不会滚雪球。

对 denoise rollout，先从错误池采一条轨迹：

$$
w \sim \mathcal{W}(q)
$$

再按 prefix ratio $\rho$ 截断：

$$
p = \max(1, \lfloor \rho |w| \rfloor)
$$

保留 $w_{1:p}$ 作为 assistant prefix，让策略模型继续生成：

$$
y_{>p} \sim \pi_{\theta}(\cdot \mid q, w_{1:p})
$$

这里很容易误解。DenoiseRL 不是让模型“接着错误写下去”，而是让模型在一个已经偏离的状态里发现问题、重构路径，进而给出正确答案。最终 verifier 看的也是完整 folded response。

---

## Folding：一个看起来细、但非常关键的公平性设计

如果直接把错误 prefix 塞进去，再让模型生成完整长度，那 denoise rollout 就会比 main rollout 多一截 token 预算。这样不公平，也会鼓励模型用更长的推理去硬凑答案。

DenoiseRL 用了一个 folding 约束：

$$
\tilde{y} = [w_{1:p}, y_{p+1:p+L}], \quad p+L \leq R
$$

其中 $R$ 是最大响应长度，论文默认 $R=4096$。prefix 已经占掉 $p$ 个 token，continuation 最多只能用 $R-p$ 个 token。

这个细节挺工程。

它不是为了公式漂亮，而是为了让“带错误前缀的训练样本”不要偷偷获得更大的计算预算。后面的消融也证明了，不做 length-fair cap，平均分会从 42.0 掉到 40.2。

---

## Reward 和 advantage：把恢复轨迹放进同一个组里比较

DenoiseRL 的 reward 很朴素：最终答案正确就是 1，错误就是 0。

同一个问题下，有 $N$ 条 main rollout 和 $K$ 条 denoise rollout。论文默认：

$$
N=12, \quad K=4
$$

也就是每题 16 条轨迹，其中 12 条正常解题，4 条从错误前缀开始恢复。

组内 advantage 按 GRPO 的方式归一化：

$$
A_i = \frac{r_i - \mu_q}{\sigma_q + \varepsilon}
$$

其中 $\mu_q$ 和 $\sigma_q$ 来自同一个问题下 $N+K$ 条轨迹的 reward 分布。

这背后的直觉是：正常从头做对、从错误里救回来做对，都进入同一组比较。模型不只是学“哪条完整答案更好”，也学“哪种状态下还能翻盘”。

PPO/GRPO 更新时还有一个非常重要的 mask：denoise rollout 只更新 continuation token，不更新弱模型给出的 noisy prefix。

这个选择后来被实验证明不是可有可无。

---

## 为什么不能更新 noisy prefix？因为那是 off-policy 噪声

这篇论文里我最喜欢的消融是 Figure 5。

如果把离线弱模型生成的 prefix 也纳入 PPO 更新，会发生什么？

短期看，好像有点用。平均验证准确率在 step 80 达到 34.7%。然后开始崩。

到 step 400，所有 benchmark accuracy 掉到 0；response length 也冲到 4096 token 上限。

![图5：更新 off-policy noisy prefix 会训练崩溃](https://arxiv.org/html/2605.28421v1/x5.png)

*图5：左图是平均验证准确率，右图是响应长度。把 weak-model prefix 也拿来算 PPO loss，前期准确率短暂上升，随后快速恶化；长度先缩短到约 450 tokens，后面暴涨到 4K budget。这个图把“只更新 continuation”这个设计的必要性讲得很清楚。*

原因其实不难理解。

PPO 的 ratio 假设 token 来自当前策略附近的分布。可是 noisy prefix 是弱模型离线生成的，和当前策略不在一个分布上。你把这些 token 也拿来算重要性比率，相当于强行给一堆 heavily off-policy token 分配梯度。

梯度会非常吵。

更糟的是，这种噪声不是随机噪声，而是结构化错误。模型可能会被拉向错误推理风格，到头来既不会短推理，也不会长推理，只会在长度和答案上一起失控。

---

## 实验设置：别忽略这些训练细节

论文主实验用了两个 policy model：

| 配置项 | 设定 |
|---|---|
| Policy model | Qwen3-4B-Base；Qwen3-8B-Base |
| Weak model | Qwen2.5-1.5B-Instruct |
| 训练数据 | MATH-7.5K |
| 每题弱模型采样 | 8 条 rollout |
| 默认 main rollouts | $N=12$ |
| 默认 denoise rollouts | $K=4$ |
| 最大响应长度 | $R=4096$ |
| Prefix ratio | $\rho=0.2$ |
| Prompt batch size | 16 |
| Learning rate | $10^{-6}$ |
| KL loss | 无 |
| Length loss | 无 |
| PPO clipping | $\varepsilon_{low}=\varepsilon_{high}=0.2$ |
| 训练采样 | temperature 1.0，top-p 1.0 |

评测集包括 MATH500、AMC23、AIME2024、AIME2025、BBEH。AMC23、AIME2024、AIME2025 用 AVG@16；MATH500、BBEH 用 AVG@1。验证采样参数是 temperature 0.6、top-p 0.95。

这个设置有个值得细看的点：DenoiseRL 没有加 KL loss，也没有 length loss。也就是说，论文希望证明“恢复式状态扩展”自己能带来收益，而不是靠额外正则把训练拉住。

---

## 主结果：提升不夸张，但挺扎实

Table 1 是主结果。

| 模型 | 方法 | MATH500 | AMC23 | AIME24 | AIME25 | BBEH | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B-Base | Base | 70.0 | 43.1 | 8.3 | 7.7 | 4.1 | 26.6 |
| Qwen3-4B-Base | GRPO | 83.6 | 63.1 | 22.1 | 18.1 | 11.1 | 39.6 |
| Qwen3-4B-Base | DAPO | 83.8 | 62.5 | 20.6 | 21.5 | 10.4 | 39.8 |
| Qwen3-4B-Base | DenoiseRL-GRPO | 85.8 | 61.4 | 24.8 | 23.3 | 14.8 | 42.0 |
| Qwen3-4B-Base | DenoiseRL-DAPO | 84.6 | 63.6 | 21.9 | 21.7 | 15.7 | 41.5 |
| Qwen3-8B-Base | Base | 70.4 | 49.2 | 11.9 | 10.8 | 4.1 | 29.3 |
| Qwen3-8B-Base | GRPO | 87.8 | 69.7 | 24.0 | 22.9 | 10.6 | 43.0 |
| Qwen3-8B-Base | DAPO | 87.0 | 69.7 | 23.8 | 21.7 | 11.7 | 42.8 |
| Qwen3-8B-Base | DenoiseRL-GRPO | 87.2 | 70.3 | 24.6 | 23.1 | 11.5 | 43.3 |
| Qwen3-8B-Base | DenoiseRL-DAPO | 88.2 | 71.4 | 27.0 | 24.8 | 12.6 | 44.8 |

看数字，别急着喊“碾压”。

在 Qwen3-4B 上，DenoiseRL-GRPO 比 GRPO 高 2.4 个平均点，DenoiseRL-DAPO 比 DAPO 高 1.7 个平均点。这个幅度对于 MATH/AIME 这类 benchmark 已经不小，但也不是换代式飞跃。

在 Qwen3-8B 上更微妙。DenoiseRL-GRPO 只比 GRPO 高 0.3 个平均点；真正亮眼的是 DenoiseRL-DAPO，平均分 44.8，比 DAPO 高 2.0 个点，而且 MATH500、AMC23、AIME24、AIME25、BBEH 全部是该模型组里最好。

我的理解是：DenoiseRL 更像一个能插到现有 RL backbone 里的训练策略，而不是要替代 GRPO 或 DAPO。DAPO 本来就在长 CoT RL 里加入了若干工程技巧，DenoiseRL 再给它补一类“错误恢复状态”，两者比较互补。

---

## 噪声不是越强越好：prefix 太长会把模型逼成复读检查器

DenoiseRL 最自然的问题是：错误前缀截多长合适？

论文比较了 $\rho \in \{0.2,0.5,0.8\}$，固定 $K=4$。

![图2：不同 prefix ratio 下的平均响应长度](https://arxiv.org/html/2605.28421v1/x2.png)

*图2：蓝色 $\rho=0.2$ 的长度最稳，训练后期平均约 1.38K tokens；$\rho=0.5$ 和 $\rho=0.8$ 会出现明显 length spike，最高接近 4K budget。这个图说明错误前缀越长，模型越容易陷入过度检查和长推理。*

论文给出的后期平均响应长度很直观：

| Prefix ratio | 训练后期平均响应长度 | 现象 |
|---:|---:|---|
| $\rho=0.2$ | 1.38K tokens | 较稳定，生成紧凑 |
| $\rho=0.5$ | 3.87K tokens | 长度不稳定，明显变长 |
| $\rho=0.8$ | 2.26K tokens | 会触达 4096 token 上限，过度思考更明显 |

这里有个挺真实的现象：坏开局太坏，模型会越来越不信自己。

Figure 3 里给了一个例子。问题是两个正三位回文数的乘积为 436,995，求它们的和。noisy prefix 一开始就把 digit positions 认错了。策略模型接下去写，先继续沿错误方向走，后来开始反复重新检查、重新因式分解、重新验证，最终进入 endless verify / self doubt。

![图3：高 prefix ratio 下的过度验证案例](https://arxiv.org/html/2605.28421v1/x3.png)

*图3：左侧红框是错误前缀，蓝框是策略模型 continuation，右侧展示 recover loop。模型并不是单纯答错，而是在错误起点下不断重查、重写、重启推导，直到 reach the max response length。这个案例很像实际长 CoT 训练中的“过度思考病”。*

坦率地讲，这个结果也提醒我们：recovery training 不是越狠越好。太轻，模型学不到纠错；太重，模型天天从烂泥坑里爬，到头来会把“怀疑一切”学成默认行为。

---

## $K=4$ 的甜点区：恢复任务要有，但不能抢走主任务

另一个旋钮是 denoise rollout 数量 $K$。

论文固定 $\rho=0.2$，比较 $K \in \{1,4,8\}$。为了让每步采样预算接近，$K$ 增大时会减少普通 on-policy rollout 数量。

![图4：不同 denoise rollout 数量下的 benchmark 增益](https://arxiv.org/html/2605.28421v1/x4.png)

*图4：$K=4$ 的平均增益最高，Avg 上是 15.4%；AIME24、AIME25、BBEH 上也比 $K=1$ 和 $K=8$ 更均衡。$K=1$ 的 recovery 信号太稀，$K=8$ 又过度强调修错，反而削弱从头解题能力。*

论文给出的总结是：

| $K$ | 平均增益 | 解读 |
|---:|---:|---|
| 1 | 14.9% | recovery 信号偏稀 |
| 4 | 16.3% | 恢复任务和从头解题之间的较好折中 |
| 8 | 11.9% | 过度强调恢复，主任务被挤压 |

这块我觉得很像 curriculum 的配比问题。

模型当然需要学会从错误状态恢复，但它更常见的部署场景还是从干净问题开始解。训练里如果错误前缀样本占比太高，模型会把“修别人写烂的半成品”当成主业。结果它恢复能力可能上去了，从头解题的决策路径反而被干扰。

---

## 长度公平真的有用：不是给模型更多 token 就更强

Table 2 专门看了 length-fair output budget 的影响。

| Folding mode | MATH500 | AMC23 | AIME2024 | AIME2025 | BBEH | Average |
|---|---:|---:|---:|---:|---:|---:|
| Length-fair | 85.8 | 61.4 | 24.8 | 23.3 | 14.8 | 42.0 |
| No length cap | 84.2 | 60.6 | 18.8 | 24.2 | 13.5 | 40.2 |

平均分 42.0 对 40.2，差 1.8 个点。

AIME2024 上差得更明显：24.8 对 18.8。

这个结果很实用。很多长 CoT RL 训练里，长度预算一松，模型就会把“多想一会儿”当成万金油。DenoiseRL 如果不给 denoise rollout 做 folding，模型会隐性拿到更多 token，短期看可能更容易翻盘，长期看会学出冗长、不稳定的推理风格。

---

## 训练成本：稍慢，但没有贵到离谱

Table 3 给了训练时间。

| Method | Rollouts / problem | Time 秒/step |
|---|---:|---:|
| GRPO baseline | 16 on-policy | 43.8 |
| DenoiseRL-GRPO | 12 + 4 | 49.7 |

硬件是 4 张 H100。DenoiseRL 每步从 43.8 秒变成 49.7 秒，大概慢 13.5%。

Figure 6 解释了原因：DenoiseRL 的 continuation 后期比 GRPO 长，论文说训练末段 100 个 step 里 continuation token 数约为 GRPO 的 1.27 倍。再加上 folded response 还包含 prefix，采样和反传自然更慢。

![图6：训练期间 rollout 长度变化](https://arxiv.org/html/2605.28421v1/x6.png)

*图6：蓝线是 GRPO，黄色实线是 DenoiseRL continuation，橙色虚线是 prefix + continuation 的完整 folded response。DenoiseRL 后期 continuation 比 GRPO 更长，完整 folded response 更长，所以单步训练时间更高。*

我觉得这个成本是能接受的。

因为它没有引入在线强 teacher，也没有用更复杂 verifier。弱模型错误池是离线生成的，训练时主要多出来的是 denoise continuation 的生成和反传成本。

---

## 代表性恢复案例：模型到底学到了什么

论文附录里有几个 recovery case，Table 4 的例子很清楚。

题目大意是：学校有 150 到 200 名学生；如果 1 人缺席，剩下学生可以分成 6 个相等 section；求所有可能 enrollment number 的和。

noisy prefix 走错了枚举，只得到 193 和 199，并给出错误答案 392。

continuation 修正时重新建立关系：

$$
n = 6k + 1
$$

因为学生数在 150 和 200 之间：

$$
150 \lt 6k + 1 \lt 200
$$

于是：

$$
24.833\ldots \lt k \lt 33.166\ldots
$$

所以 $k=25,\ldots,33$，对应学生数：

$$
151,157,163,169,175,181,187,193,199
$$

总和是 1575。

这个例子里的恢复不只是“改最终答案”。模型需要识别前面枚举范围错了，回到约束条件重新推一遍。这正是 DenoiseRL 想训练的能力：不是在答案末尾补救，而是在中间推理状态发现路线偏了。

---

## 和普通 GRPO / DAPO 的关系：不是替代，而是补状态分布

GRPO 的核心直觉是组内比较：同一道题采多条回答，用 reward 的均值和方差做相对 advantage。相比 PPO 里训练 value model，GRPO 更轻，也更适合数学题这种可验证奖励场景。

DAPO 则是在长 CoT RL 里补了更多工程技巧，比如 clip 解耦、动态采样等，让训练更稳、更高效。

DenoiseRL 不直接挑战这些算法。它干的是另一层事：改变 rollout 的起点分布。

普通 on-policy RL 的状态分布大多来自模型自己从题目开始生成的轨迹。如果模型很少自然走到某些“可恢复但困难”的错误状态，那它就很少学会怎么从那里出来。DenoiseRL 用弱模型错误前缀人为扩展这些状态，让 RL 信号覆盖到更像真实错误恢复的场景。

你可以把它理解为一种 reasoning 状态空间的数据增强。

但这个增强不是改题面，也不是生成新题，而是改推理过程的起点。

---

## 我觉得这篇论文最漂亮的地方

第一，弱模型的位置摆得很准。

很多 weak-to-strong 论文容易把弱模型包装成 teacher，但弱 teacher 最大的问题就是上限低。DenoiseRL 很聪明地避开了这个坑：弱模型只负责制造错误，不负责提供正确监督。弱模型越容易犯错，反而越能提供丰富的 corrupted states。

第二，mask off-policy prefix 这个设计很克制。

作者没有为了“更多监督信号”把 prefix 也塞进 loss，而是明确区分了可见上下文和可更新 token。这个边界感很重要。Figure 5 的训练崩溃也说明，如果把 off-policy token 当 on-policy token 更新，整个训练会被噪声拖垮。

第三，length-fair folding 是工程上真正会踩坑的细节。

如果没有这个约束，很容易把性能提升误判为“模型恢复能力更强”，其实只是 denoise rollout 拿到了更多 token。论文把这个变量控制住，实验可信度更高。

---

## 我也有几个疑问

说实话，这篇论文也不是没有让人皱眉的地方。

最明显的是，实验范围主要集中在数学与推理 benchmark。DenoiseRL 的思想听起来也适合代码修复、多步工具调用、Agent 任务恢复，但这些场景里的 verifier 往往更贵、更慢，错误状态也更复杂。数学题只看 boxed answer，奖励很干净；真实任务里“恢复成功”没这么好判。

第二，弱模型错误轨迹的质量会不会决定上限？

论文强调不依赖强 teacher，但弱模型如果生成的错误太浅，可能学不到深层恢复；如果错误太离谱，模型又会陷入 Figure 3 那种 overthinking。也就是说，DenoiseRL 省掉了人工构造困难数据，但并没有完全省掉“错误分布设计”这个问题。

第三，结果提升在不同 backbone 上不均衡。

Qwen3-8B + GRPO 只涨 0.3 个平均点，说明 DenoiseRL 并不是每个设置下都稳定带来大收益。它和 backbone、模型尺度、训练阶段、prefix 分布之间的耦合还需要更多实验拆开看。

这些问题不影响论文价值，但会影响工程落地时的预期。

---

## 如果你想把它用到自己的训练里

我会优先试这几个方向：

| 工程问题 | DenoiseRL 式做法 |
|---|---|
| 数学模型常在中间步骤犯错 | 收集弱模型错误 CoT，截短 prefix 做 recovery rollout |
| 代码模型 patch 经常局部修坏 | 把错误 patch 或错误调试日志当 prefix，训练模型从坏状态修复 |
| Agent 工具调用链中途选错工具 | 用失败轨迹前半段作为状态，让模型继续恢复计划 |
| 模型过度思考 | 限制 prefix ratio，做 length-fair cap，监控 response length |
| 训练不稳定 | prefix 只作为上下文，不参与 PPO loss |

真正落地时，我会把 $\rho=0.2, K=4$ 当起点，不会一上来把错误前缀塞得很长。然后盯三个指标：最终准确率、平均 response length、恢复样本上的成功率。

如果准确率涨但长度暴涨，我会先怀疑 prefix 太长或 folding 没控住。

---

## 总结：让模型学会“坏开局也能翻盘”

DenoiseRL 这篇论文最打动我的地方，不是平均分多涨了几个点，而是它把 reasoning RL 里一个长期被忽略的问题拎了出来：模型不只要会从干净题面推到正确答案，还要会从错误中间状态恢复。

这很像真实解题。

人做题也不是每次从头到尾都走最短路径。更常见的是：做着做着发现哪里不对，退一步，重建变量，换一条路。很多推理模型现在缺的恰恰是这种“发现自己偏了并拉回来”的能力。

DenoiseRL 给了一个简单、可复用、成本不算离谱的训练框架。它不神奇，也不完美，但方向挺对。

如果你也在训数学推理、代码推理或长链路 Agent，我觉得这篇值得细读。尤其是 Figure 5 和 Table 2，里面藏着两个很实际的训练教训：别乱更新 off-policy 错误前缀，别让恢复样本偷偷拿更多 token。

这两个坑，工程里真的很容易踩。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
