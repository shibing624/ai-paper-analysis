# GRPO 把奖励均匀撒给每个 token，这篇论文让模型用自己的解法当"老师"重新分配

你有没有想过一个很别扭的事：RLVR 训练里，一道数学题做对了，GRPO 会把同一个 advantage 平摊给这条回答里的**每一个 token**。包括那句"Let me think step by step."，包括"所以答案是"这种过场话，也包括真正卡住命门、决定成败的那一步推导——全都拿一样的梯度。

这显然不对。常规 token 在那儿白白吃梯度，真正关键的推理步骤反而没被额外奖励。问题是，想做 token 级的信用分配（credit assignment），你得知道"哪个 token 重要"——而这个信息从哪来？

过去的答案都要往模型外面伸手：要么训一个 process reward model（PRM）给每步打分，要么用 ground-truth 答案对齐，要么搞知识蒸馏拉一个外部 teacher 来。可纯 RLVR 的设定里，你手上只有一个 binary verifier——对就是对，错就是错，啥额外信息都没有。

这篇 6 月刚挂出来的 **SC-GRPO**（arXiv:2606.18810）给了个我觉得挺漂亮的解法：不用外部 teacher，让模型**拿自己已经做对的解法**当老师，去给每个 token 算一个"重要性权重"。说白了——既然你已经做对过，那就照着对的答案回头看，哪些 token 让分布发生了剧烈变化，哪些就是关键步骤。

---

## 核心摘要

RLVR 里 GRPO 的硬伤是 token 级信用分配缺失：整条回答共享一个 advantage，关键推理步骤拿不到额外信用。已有的 token 级方法（PRM、On-Policy Distillation）都要外部资源，在纯 RLVR 下用不了。

SC-GRPO 的核心观察是：**把模型条件化在它自己验证过的正确轨迹上，原始分布和条件化分布之间会产生一个可测量的 per-token KL 散度**——这个散度天然标记出了哪些 token 是"看到答案后才恍然大悟"的关键位置。论文把这个 KL 做成 GRPO 梯度上的**乘性权重**，而不是加一项蒸馏 loss（这点很关键，后面讲）。

在 math、code、agentic 横跨五个 benchmark 上，SC-GRPO 比 GRPO 平均高 8.1%，比 DAPO 高 5.9%，OOD 还更稳，甚至打过了需要外部 teacher 的 OPD。我的判断：这是一个把"自蒸馏信号"用对地方的工程巧思，理论上还顺手证明了"为什么不能直接当蒸馏目标"，值得细读。

---

## 论文信息

- **标题**：Learning from Own Solutions: Self-Conditioned Credit Assignment for Reinforcement Learning with Verifiable Rewards
- **作者**：Yingyu Shan, Yuhang Guo, Zihao Cheng, Zeming Liu, Xiangrong Zhu, Xinyi Wang, Jiashu Yao, Wei Lin, Hongru Wang, Heyan Huang
- **arXiv**：[2606.18810](https://arxiv.org/abs/2606.18810)（2026 年 6 月 17 日提交）
- **基座模型**：Qwen3-8B（thinking disabled）

![图1：SC-GRPO 总览](https://arxiv.org/html/2606.18810v1/figures/teaserv3.png)

*图1（teaser）：上半部分是 SC-GRPO 在现有方法谱系中的定位——它既不依赖 PRM，也不依赖外部 teacher，只用模型自己的 rollout；下半部分用一个 LiveCodeBench 的真实例子展示核心机制：把已验证的正确解法塞进 context 当条件，模型对同一段前缀的预测分布会变，变化最大的那些 token 就是被信用分配"点名"的关键步骤。*

---

## 🎯 为什么 GRPO 的"均匀分配"是个真问题

先把 GRPO 的 advantage 写出来，这是理解一切的起点：

$$\hat{A}_{i,t}=\hat{A}_i=\frac{r(x,o_i)-\mathrm{mean}(\{r(x,o_k)\}_{k=1}^{G})}{\mathrm{std}(\{r(x,o_k)\}_{k=1}^{G})}$$

注意左边那个下标 $t$——它压根不出现在右边。也就是说，同一条 rollout $o_i$ 里，第 1 个 token 和第 500 个 token 拿到的 advantage **完全一样**，都等于 $\hat{A}_i$。这就是"uniform credit"的数学本质。

这在短回答里也许无所谓，但推理任务的回答动辄上千 token。一条正确解法里，可能只有那么三五步是真正的"题眼"，其余都是格式、过渡、复述。GRPO 对这一切一视同仁，梯度被大量常规 token 稀释掉了。

你可能会说，那上 PRM 啊，给每步打分不就行了。问题是 PRM 自己就要标注数据训练，而且 reward hacking 风险一堆；用 ground-truth 答案做对齐又只在有标准答案的任务上成立。On-Policy Distillation 要拉个更强的外部模型当 teacher，On-Policy Self-Distillation 要 privileged information——这些在"我手上只有一个 verifier"的纯 RLVR 场景里统统失效。

SC-GRPO 想问的是：**能不能完全不向外伸手，只用模型自己的 rollout，就把这个 token 级的重要性信号挖出来？**

---

## 🧠 核心 idea：让模型看着自己的正确答案，量它"恍然大悟"的程度

这套方法最妙的直觉，我用一句话讲：**给你一道题让你重做，和给你这道题、再把一份标准答案摆在你面前让你重做，你在哪些地方的"想法"会变得最厉害？那些地方就是关键。**

形式化一下。对一条 rollout $o_i$，它在第 $t$ 步的前缀状态是 $s_{i,t}=(x,o_{i,<t})$。现在取一条已经验证正确的轨迹 $\tau$（从模型自己的正确 rollout 里采的），构造一个 **self-conditioned teacher**：

$$\widetilde{\pi}_{\theta}(\cdot\mid s_{i,t},\tau) := \operatorname{sg}[\pi_{\theta}(\cdot\mid x,\tau,o_{i,<t})]$$

这里 $\operatorname{sg}[\cdot]$ 是 stop-gradient，$\tau$ 只是塞进 teacher 的 system prompt 里。关键在于：teacher 和 student 是**同一个模型**，区别只是 teacher 的上下文里多了一份正确答案 $\tau$。两者在 $o_i$ 的**相同前缀** $o_{i,<t}$ 上各自给出下一个 token 的分布。

然后在每个 token 处算从 teacher 到 student 的 forward KL：

$$D_{i,t}=\mathrm{KL}\left(\widetilde{\pi}_{\theta}(\cdot\mid s_{i,t},\tau)\;\middle\|\;\pi_{\theta}(\cdot\mid s_{i,t})\right)$$

$D_{i,t}$ 大，意味着"看到正确答案后，模型在这个 token 上的预测和原来差很多"——这正是关键推理步骤的特征。常规 token（标点、过渡词）看不看答案都那样，KL 接近 0。

![图2：SC-GRPO 方法总览](https://arxiv.org/html/2606.18810v1/figures/methodv1.5.png)

*图2（method）：完整流程一图流——左边把 reference trajectory 条件化进去构造 self-conditioned teacher，中间在每个 response token 上算 teacher 与 student 的 KL 散度，右边把这个 token 级 KL 当成权重去缩放 GRPO 的梯度。整条链路没有任何外部模型介入。*

### 把 KL 变成一个 [0,1) 的权重

原始 KL 值域无界、还稀疏，不能直接当权重乘。作者用了个有界映射：

$$f(D_{i,t})=\frac{D_{i,t}}{D_{i,t}+c}, \quad c=\max\left(P_{75}(\mathcal{D}_{\mathrm{act}}),\,c_{\min}\right)$$

$D$ 越大 $f$ 越接近 1，$D$ 接近 0 则 $f$ 接近 0。分母里的 $c$ 是个自适应阈值，取当前 micro-batch 里 active token 的 KL 值的**第 75 百分位**和一个下界 $c_{\min}$ 的较大者。

为什么是 $P_{75}$ 而不是中位数？因为 token 级 KL 是**右偏**的——绝大多数 token KL 趋近 0，只有少数尖峰。用 $P_{50}$（中位数）的话，$c$ 会被压得很小（甚至小于 $c_{\min}$），结果所有 token 的 $f$ 都接近 1，退化回 uniform weighting，等于白做。$P_{75}$ 才能把 low-KL token 压下去、保留 high-KL 尖峰。这个细节在消融里有实打实的数据支撑，后面会看到。

### 完整目标函数

$$J_{\mathrm{SC\text{-}GRPO}}(\theta)=\mathbb{E}\Big[\tfrac{1}{G}\sum_{i=1}^{G}\tfrac{1}{|o_i|}\sum_{t=1}^{|o_i|} f(D_{i,t})\,\min\big(\rho_{i,t}\hat{A}_i,\,\mathrm{clip}(\rho_{i,t},1{-}\epsilon,1{+}\epsilon)\hat{A}_i\big)\Big]$$

跟标准 GRPO 比，差别就是前面多乘了一个 $f(D_{i,t})$。其余 importance ratio $\rho_{i,t}$、clip 都原样保留。有一点要强调：这个 KL 权重对正确和错误 rollout 是**对称施加**的，不管 advantage 正负。错误回答里 KL 大的 token，同样会被放大梯度——因为那也是模型该重点学习"别这么走"的地方。

---

## 🔬 一个我没料到的理论结果：为什么不能直接拿它当蒸馏目标

这篇论文最让我意外的不是方法，是它顺手证明的一个东西。

最自然的想法其实是：既然有了 self-conditioned teacher，那直接做蒸馏不就好了？让 student 去拟合 teacher 的分布。这就是把 OPSD（On-Policy Self-Distillation）朴素地搬过来。

但作者证明了这条路有根本性的坑。设 $\mathcal{C}(x)$ 是所有验证正确的轨迹集合，当存在**多条**正确轨迹 $\tau_j$ 时，对每条做一个 teacher $p_j$，那么在 forward / reverse KL 下，使 loss 最小的 student 分布是这些 teacher 的聚合：

$$q^{\star}\propto\begin{cases}\sum_{j}\mu_{x}(j)\,p_{j}&\text{(Forward KL)}\\ \prod_{j}p_{j}^{\,\mu_{x}(j)}&\text{(Reverse KL)}\end{cases}$$

问题来了，这个加权平均/几何平均出来的分布有三宗罪：

1. **它不一定对应任何一条 feasible 的轨迹**——平均出来的东西可能哪条正确路径都不是；
2. 它对所有前缀给同等权重，哪怕几条正确轨迹在某个 token 上分歧很大；
3. 它继承了每条 $\tau_j$ 到达正确答案的随意路径——有些正确答案是瞎蒙对的，你不该照学。

一句话：**把 self-teacher 当目标分布去拟合，会逼模型去学一个"四不像"的平均解**。这对任何把 teacher 视作目标的蒸馏方法都成立。所以 SC-GRPO 才坚决地把 KL 当**权重**用，而不是当 **loss** 用——它只借 KL 来判断"哪个 token 重要"，绝不让模型去拟合 teacher 的具体分布。

这个区分很重要，也是后面消融里 RQ1 专门要打的点。

---

## 🏗️ 工程细节：Group Routing 怎么处理不同难度的题

光有权重还不够。一组 G 个 rollout，根据做对的数量 $n_c$ 不同，要分情况处理：

| 组类型 | 条件 | 处理方式 |
|---|---|---|
| **Partial-solve** | $2\leq n_c \lt G$ | 每条 rollout 均匀采一条正确轨迹 $\tau$ 当 reference（自己是对的就排除自己），用标准 GRPO advantage + KL 权重 |
| **Solve-none** | $n_c=0$ | 全错，没有正确轨迹可用。改用 diversity score 构造 pseudo-advantage 鼓励探索 |
| **Fallback** | $n_c=1$ 或 $n_c=G$ | 退回标准 GRPO，不加 KL 权重 |

Solve-none 这块设计得有意思。全错了没有正确答案当 reference 怎么办？作者随机挑一条 rollout $o_r$ 当参照，对其余每条算和它的 diversity：

$$s_i=\frac{1}{L_i}\sum_{t=1}^{L_i}f(D_{i,t}),\quad L_i=\min(|o_i|,|o_r|)$$

再归一化成 pseudo-advantage $\hat{A}^{\mathrm{div}}_i$，用 $\alpha\hat{A}^{\mathrm{div}}_i$ 替换原 advantage。意思是：全军覆没的题上，鼓励模型生成跟参照不一样的解法，多探索。

至于为什么 $n_c=1$ 和 $n_c=G$ 要直接退回 GRPO——$n_c=G$ 全对，没有错误 rollout 做对比；$n_c=1$ 强行加 KL 权重会触发 entropy collapse。消融数据会印证这一点。

---

## 🧪 实验：五个 benchmark，横跨数学、代码、智能体

实验设置我觉得是诚实的。所有 run 共享 Qwen3-8B（thinking disabled），任务覆盖三类：

- **Math**：训练用 DAPO-Math-17k，评测 AIME 2024 & 2025
- **Code**：LiveCodeBench v6，一半 unit test 训练、一半评测
- **Agentic**：AppWorld 和 WebShop，官方 split

指标是 Avg@8（8 个样本的平均 verifier reward）和 Pass@8（至少 1 个样本解决的比例）。baseline 包括 GRPO、DAPO、REINFORCE++，以及三种不同质量外部 demonstration 的 OPSD 变体（MiniMax-M2.7、DeepSeekv4-Pro、Oracle）。

### 主结果（Table 1）

| 方法 | AIME24 Avg@8 | AIME25 Avg@8 | LCB v6 Avg@8 | AppWorld Avg@8 | WebShop Avg@8 | **Avg. Avg@8** | **Avg. Pass@8** |
|---|---|---|---|---|---|---|---|
| Qwen3-8B（base） | 25.41 | 18.75 | 26.33 | 9.65 | 8.13 | 17.65 | 30.42 |
| GRPO | 41.67 | 34.16 | 39.59 | 12.50 | 68.93 | 39.37 | 53.25 |
| DAPO | 43.75 | 37.91 | 40.74 | 13.99 | 71.50 | 41.58 | 55.19 |
| MiniMax-M2.7（OPSD）† | 29.17 | 19.58 | 26.05 | 9.87 | 11.40 | 19.21 | 35.53 |
| DeepSeekv4-Pro（OPSD）† | 33.75 | 22.08 | 26.05 | 9.43 | 9.88 | 20.24 | 34.50 |
| Oracle（OPSD） | 33.75 | 22.08 | 25.38 | 9.43 | 10.38 | 20.20 | 35.96 |
| **SC-GRPO（ours）** | **51.67** | **42.08** | **43.22** | **22.36** | **77.88** | **47.44** | **64.11** |

几个数字值得停下来看：

平均 Avg@8 从 GRPO 的 39.37 涨到 47.44，比 DAPO 的 41.58 高了 **5.86 个点**；Pass@8 更猛，比 DAPO 高 **8.92 个点**。最大的提升出现在 multi-turn 的 agentic 任务上——AppWorld 上 SC-GRPO 拿到 22.36，几乎是 DAPO（13.99）的 1.6 倍。这其实合直觉：multi-turn 任务里关键决策点更稀疏、回答更长，信用分配的价值更大。

再看 OPSD 那三行——这是我觉得最能说明问题的对比。三个 OPSD 变体全都贴着 base model（17.65）打转，远低于 RL baseline。最扎心的是 **Oracle 在 LCB 上只有 25.38，比 base model 的 26.33 还低**。Oracle 是尽可能多收集正确 demonstration 的版本，连它都救不了——这直接说明：**demonstration 的数量和质量根本不是瓶颈，朴素蒸馏这条路本身就是错的**。正好印证了前面那个理论结果。

### 跟 OPD 正面刚

![图4：SC-GRPO vs OPD](https://arxiv.org/html/2606.18810v1/x4.png)

*图4：在 AIME 24 & 25 上用两个不同 student 对比 SC-GRPO 与 On-Policy Distillation（OPD），百分比是相对各自 student baseline 的增益。左边是 OPD 会失败的设置（Qwen3-1.7B-Base student / Qwen3-4B-Base teacher），SC-GRPO 拿到 23% 提升；右边是 OPD 能成功的设置（DeepSeek-R1-Distill-Qwen-1.5B student / JustRL-DeepSeek-1.5B teacher），SC-GRPO 在两个 AIME 任务上仍然超过 OPD——而且全程不需要外部 teacher。*

这个对比挺有说服力的。OPD 是要外部 teacher 的，且对 teacher 质量敏感，配不好就崩。SC-GRPO 不挑 teacher（因为它就是模型自己），OPD 能赢的场景它追平甚至超过，OPD 崩的场景它照样涨。

### 计算开销：看着吓人，其实还好

有人肯定关心，每个 token 都要跑一遍 teacher 的 forward，是不是慢死了？

Actor update 这一步确实慢了不少——LCB 上慢 51%，DAPO-Math 上慢 37%。但端到端看，每个 step 的总时间只涨了 **13%** 和 **2%**。原因是 RLVR 的时间大头在 rollout 生成上，actor update 占比本来就小。这个 trade-off 我觉得能接受。

![图3：计算开销对比](https://arxiv.org/html/2606.18810v1/x1.png)

*图3：SC-GRPO 与 GRPO 每个训练 step 的平均 wall-clock 时间（秒）对比。虽然 actor update 环节有明显增量，但被 rollout 生成的延迟摊薄后，端到端开销增长有限。*

---

## 📊 消融实验：每个设计决策都有数据撑腰

消融做得相当扎实，全在 LiveCodeBench v6 上。

### RQ1：KL 该当权重，还是当 loss？（Table 2）

这是直接回应前面理论的实验。把 self-teacher 的 KL 改成附加蒸馏 loss（$\mathcal{L}_{\mathrm{GRPO}}+\beta\mathcal{L}_{\mathrm{distill}}$）试了 5 种配置：

| 变体 | Avg@8 | Pass@8 |
|---|---|---|
| GRPO | 39.59 | 43.51 |
| **SC-GRPO（权重，ours）** | **43.22** | **48.85** |
| Forward KL loss（$\beta=0.1$） | 38.35 | 41.98 |
| Forward KL loss（$\beta:0.03\to0.1$） | 36.35 | 38.16 |
| Reverse KL loss（$\beta=0.1$） | 36.06 | 38.93 |
| Reverse KL loss（$\beta:0.03\to0.1$） | 39.11 | 41.98 |
| Bidirectional KL loss | 34.35 | 36.64 |

5 个 additive 变体**没一个打得过 SC-GRPO**，好几个甚至跌破 GRPO baseline。理论预言的"拟合平均解会伤害性能"在这里被实测坐实了。KL 信号只配当权重，不配当目标。

### RQ2：Group Routing、阈值、探索系数（Table 3）

| 变体 | Avg@8 | Pass@8 |
|---|---|---|
| **A1. Group Routing** | | |
| Partial-solve only（$2\leq n_c\leq7$） | 41.41 | 45.80 |
| **Partial + solve-none [ours]** | **43.22** | **48.85** |
| Any partial（$1\leq n_c\leq7$） | 38.55 | 41.98 |
| All-correct only（$n_c=8$） | 40.17 | 42.74 |
| All groups | 36.64 | 42.75 |
| **A2. 阈值 $c$** | | |
| **Adaptive $\max(p_{75},10^{-4})$ [ours]** | **43.22** | **48.85** |
| Fixed $c=10^{-4}$ | 42.93 | 48.09 |
| Adaptive $p_{50}$ | 41.98 | 45.03 |
| **A3. 探索系数 $\alpha$** | | |
| **$\alpha=0.1$ [ours]** | **43.22** | **48.85** |
| $\alpha=0.2$ | 40.74 | 45.80 |

读这张表能学到不少：

**A1** 验证了 routing 的每个分支。只用 partial-solve 已经不错（41.41），加上 solve-none 的 diversity 探索再涨到 43.22。但一旦把 $n_c=1$ 也塞进来（Any partial），掉到 38.55——entropy collapse 应验；只用 all-correct（$n_c=8$）因为没有错误对比，40.17；最离谱的是无脑 all groups，36.64，直接跌破 GRPO。

**A2** 是 $P_{75}$ vs $P_{50}$ 的正面交锋。用 $p_{50}$ 只有 41.98——因为 KL 右偏让阈值塌到 $c_{\min}$ 以下，退化成均匀加权了。而 $\max(p_{75},10^{-4})$ 既能自适应又免去 per-task 调参，是 hyperparameter-free 的。

**A3** 探索系数 $\alpha=0.1$ 最优，调到 0.2 反而掉到 40.74——探索信号太强会跟 RL 主目标抢方向。

### RQ3：OOD 泛化

![图5：In-Domain vs OOD 性能](https://arxiv.org/html/2606.18810v1/x5.png)

*图5：在 LiveCodeBench 上训练、在 Codeforces 上评测的 OOD 表现。SC-GRPO 的优势没有因为换了分布就消失——OOD 上 Avg@8 拿到 5.2% 相对提升，和 in-domain 的 6.1% 很接近，说明增益来自更好的信用分配，而不是对训练集的过拟合。*

这个实验我挺看重。信用分配方法最容易被质疑的就是"你是不是只是过拟合了训练分布的某些模式"。OOD 上还能保持接近 in-domain 的提升幅度，这个反驳是有力的。

### 看一眼 KL 到底点名了哪些 token

![图10：Token 级 KL 热力图案例](https://arxiv.org/html/2606.18810v1/figures/case_heatmapv1.png)

*图10（附录 D 案例研究）：图1 底部那个例子的完整版热力图，按 token 展示 KL 信号的分布。颜色深的 token 是被信用分配"点名"的关键步骤——能直观看到 KL 权重确实集中在推理的转折点上，而不是均匀铺满或乱点一气。*

---

## 🤔 我的判断

先说我喜欢的地方。

**这个 idea 的经济性很高。** 不要 PRM、不要外部 teacher、不要 ground-truth 对齐，就靠模型自己已经做对的轨迹，把 token 级的重要性信号挖出来。在纯 RLVR 这个"什么外部信息都没有"的苛刻设定下，这几乎是把现有信息榨到极致的做法。

**理论和方法咬合得很紧。** 它没停在"我有个 trick 很 work"，而是先证明了"为什么朴素蒸馏不行"，再把 KL 从"目标"降级成"权重"。RQ1 的消融又用实验把理论钉死。这种"理论说不能这么用→那就换个用法→实验验证换法对了"的闭环，比单纯刷 SOTA 有说服力。

再说我打问号的地方。

**额外 forward 的成本被 rollout 摊薄了，但这是有前提的。** 端到端只涨 2%-13%，是因为这几个任务 rollout 生成本来就慢、占大头。如果换成 rollout 很快、回答很短的任务，actor update 慢 51% 这个数就会暴露出来。论文报的是对它有利的场景。

**self-teacher 的质量上限就是模型自己。** 模型当前做对的轨迹质量越差，它当 teacher 给出的 KL 信号也越糊。这方法本质上是"用现在的自己教现在的自己"，对一个推理能力本就很弱的基座，可能挖不出多少有效信号。论文用的是 Qwen3-8B 这种已经不弱的基座，换个弱基座会怎样，没看到数据。

**agentic 任务上 1.6 倍的提升很亮眼，但也要警惕。** AppWorld、WebShop 这种 multi-turn 任务方差大、baseline 普遍偏低（GRPO 在 AppWorld 才 12.50），在低基数上涨绝对值容易显得夸张。我更信 LCB 和 AIME 这种成熟 benchmark 上 5-6 个点的提升。

总的来说，这是一篇"小切口、想得透、做得实"的论文。它没有发明新范式，但把"自蒸馏信号用在哪、怎么用"这件事想明白了，还顺手证明了一条看似诱人实则有坑的路。如果你在做 RLVR、被 GRPO 的均匀信用分配困扰，又不想引入 PRM 这种重资产，SC-GRPO 这个思路值得在你的 pipeline 上试一把——改动量其实就是在 GRPO 梯度前乘一个权重项。

最后留个开放问题：self-conditioned teacher 这个构造，能不能推广到没有 verifier 的开放式生成任务上？毕竟"看到一个好答案后，模型在哪些 token 上想法变了"这个信号，本身并不要求答案必须可验证。这或许是个更大的口子。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
