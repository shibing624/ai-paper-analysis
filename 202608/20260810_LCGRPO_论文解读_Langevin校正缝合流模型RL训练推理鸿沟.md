# LC-GRPO：一步 Langevin 校正，缝合 Flow 模型 RL 训练与推理之间的那道缝

上周翻到一篇让我眼前一亮的 paper。事情是这样的：最近用 GRPO 给文生图/文生视频模型做 RL 后训练的工作不少（Flow-GRPO、DanceGRPO 这些），但大家一直绕开一个尴尬的问题——训练时用随机 SDE 采样做探索，测试时却用确定性 ODE 采样出图。连续时间里这俩是等价的，可一旦离散化成几十步，差别就出来了：SDE rollout 一加噪声就糊，模型在模糊的样本上学到的奖励信号，跟测试时清晰样本的分布根本对不上。

LC-GRPO 给了一个特别干净的解法：每一步先走和推理完全一致的 ODE Euler 步（predictor），再补一步 Langevin 随机校正（corrector）注入探索噪声。score 不用额外训练，直接从 flow 的速度场里用 Tweedie 公式恢复。理论上证明了一步 Langevin 能严格收缩 Euler 步的 Wasserstein 误差，匹配噪声水平下甚至比标准 Euler–Maruyama SDE 离散化更准。实验上，SD3.5-Medium、FLUX.1-Dev、HunyuanVideo 三个模型上全面压过 Flow-GRPO / DanceGRPO / CPS，而且训练-评估差距从 0.063 缩到 0.016。这篇论文值钱的地方不在于又刷了多少分，而在于它把"训练-推理失配"这个大家一直在凑合的问题，用一个有理论保证的方式讲清楚了。

---

**论文信息**

- 标题：LC-GRPO: Bridging Train-Inference Gap for Flow-Based GRPO with Langevin Correction
- 作者：Yingqing Guo, Hui Yuan, Zijian He, Mengdi Wang, Zheng Ding
- arXiv：https://arxiv.org/abs/2608.05600 （2026 年 8 月 6 日提交）

---

## 🎯 问题到底出在哪：SDE 探索的代价

先快速回顾下背景。Flow matching 模型推理时走确定性 ODE：$\mathrm{d}\bm{x}_t = \bm{v}_t\mathrm{d}t$，离散化就是大家熟悉的 Euler 步：

$$\bm{x}_{t-\Delta t} = \bm{x}_t - \bm{v}_t(\bm{x}_t, t)\Delta t$$

但 RL 需要随机 rollout 来探索。Flow-GRPO 们的做法是引入一个与 ODE 共享边缘分布的 SDE：

$$\mathrm{d}\bm{x}_t = \Big[\bm{v}_t + \frac{\sigma_t^2}{2t}\big(\bm{x}_t+(1-t)\bm{v}_t\big)\Big]\mathrm{d}t + \sigma_t\mathrm{d}\bm{w}_t$$

用 Euler–Maruyama 离散化后，每步在漂移项之外额外注入 $\sigma_t\sqrt{\Delta t}\,\xi$ 的高斯噪声，Flow-GRPO 里取 $\sigma_t = \eta\sqrt{t/(1-t)}$，$\eta$ 越大探索越猛。

理论上很漂亮——连续时间里 ODE 和 SDE 的边缘分布一模一样。但说实话，我第一次看到"有限步数下两者差异巨大"这个论断时并不意外，做扩散模型采样的都知道，SDE 采样器和 ODE 采样器在少步数下出来的图质量差距是肉眼可见的。图 2 把这个现象展示得很直白：

![图2：不同噪声水平下 SDE 与 ODE+Langevin 采样质量对比](https://arxiv.org/html/2608.05600v1/x3.png)

*图 2：同一个 $\eta$，SDE 采样（上排）在噪声水平 0.6、0.9 时面包店招牌上的文字已经开始糊成一团，企鹅的围巾细节也在劣化；而 ODE + Langevin（下排）在同等噪声下几乎保持了 ODE 的清晰度。左半部分是 SD3.5-M（10 步），右半是 FLUX.1-Dev（6 步）。*

这就形成了一个两难：噪声小了探索不足，RL 学不动；噪声大了 rollout 全是模糊图，奖励模型在模糊图上的打分跟在清晰图上的打分根本不是一回事。模型被迫在一个"错误的分布"上优化。

## 🧠 核心思路：预测-校正，把探索和采样解耦

LC-GRPO 的切入点我觉得挺聪明的——为什么一定要把噪声揉进采样动力学里？能不能先走一步干净的 ODE，再单独加一步"只改分布不改边缘"的随机化？

具体每个 rollout 转移分两步：

**第一步，Predictor（ODE Euler）**：

$$\bm{x}' = \bm{x}_t - \Delta t\,\bm{v}_\theta(\bm{x}_t,t)$$

这一步和测试时推理一模一样，保证 rollout 的主体轨迹始终贴着推理分布走。

**第二步，Corrector（一步 Langevin）**：

$$\bm{x}_{t-\Delta t} = \bm{x}' + \epsilon_t\,\bm{s}(\bm{x}', t-\Delta t) + \sqrt{2\epsilon_t}\,\xi,\quad \xi\sim\mathcal{N}(0,I_d)$$

Langevin 动力学是老朋友了——MALA 那一套，朝着 score 方向走一小步再加噪声，目标分布是不变的。这里的关键问题是：score 从哪来？总不能为了 RL 再训一个 score 模型吧？

不用。Tweedie 公式直接把 score 从 flow 速度场里恢复出来：

$$\bm{s}(\bm{x},t) = -\frac{\bm{x} + (1-t)\,\bm{v}(\bm{x},t)}{t}$$

零额外参数，零额外训练。这个 trick 在扩散模型社区其实用得不少，但用在 GRPO rollout 的校正步上，配合后面的理论分析，就讲出了一个完整的故事。

步长遵循 Song 等人的设置取期望形式：$\sqrt{2\epsilon_t} = \eta(t-\Delta t)$，$\eta$ 控制探索强度。有个细节很有意思——论文脚注里提到，相同 $\eta$ 下 Langevin 在所有时间步上的高斯噪声方差之和实际上比 SDE 还大。也就是说，LC-GRPO 的样本更清晰，不是因为"作弊"少加了噪声，而是噪声加得更聪明：加在 corrector 里，由 score 引导着加，而不是盲目地往轨迹里灌。

对 RL 来说最重要的是：这个转移核仍然是各向同性高斯，

$$p_\theta(\bm{x}_{t-\Delta t}\mid\bm{x}_t,\bm{c}) = \mathcal{N}\big(\bm{m}_\theta(\bm{x}_t),\ 2\epsilon_t I_d\big),\quad \bm{m}_\theta(\bm{x}_t) = \bm{x}' + \epsilon_t\,\bm{s}_\theta(\bm{x}', t-\Delta t)$$

似然可以闭式计算，GRPO 的 importance ratio $r^i_t(\theta)$ 照常算，整个 GRPO 目标（组内归一化优势 + clip + KL 惩罚）原封不动搬过来用。算法流程上只有一个小细节：最后一步不加 Langevin 校正，走纯 ODE 步，保证最终输出的样本就是推理分布的。

## 📐 理论部分：不只是工程 trick

这篇论文让我加分的地方是它给了理论保证，而不是只扔个方法就跑实验。

**定理 1（Langevin 严格改善 Euler 步）**：在目标分布 $\alpha$-强对数凹、score $L$-Lipschitz 等常规假设下，如果 Euler 步存在 Wasserstein 误差 $\varepsilon = W_2(\mathrm{law}(\bm{x}_{\mathrm{ode}}), p_r) > 0$，那么只要步长满足

$$\epsilon \leq \frac{\alpha}{L^2}\quad\text{且}\quad \epsilon \lt \frac{9}{100}\cdot\frac{\alpha^2\varepsilon^2}{L^2 d}$$

一步 Langevin 校正就严格降低误差：$W_2(\mathrm{law}(\bm{x}_{\mathrm{lc}}), p_r) \lt \varepsilon$。证明的核心是一个收缩界——Langevin 步把分布往目标拉近的速率是 $(1-\alpha\epsilon/2)$，而它自己引入的离散化误差只有 $O(\epsilon^{3/2})$ 阶，步长足够小时收缩项稳稳主导。

**定理 2（比 SDE 步更准）**：这个是更直接的打脸。把两种方法的主阶误差都展开到 $O(h^2)$，在匹配噪声水平 $\epsilon = \sigma^2 h/2$ 下，只要满足一个比较条件 $\langle \bm{E}, \dot{\bm{s}}\rangle_{p_t} \lt \frac{\sigma^2}{2}\|\dot{\bm{s}}\|^2_{p_t}$（$\bm{E}$ 和 $\dot{\bm{s}}$ 是沿流的速度/score 物质导数组合），Langevin 校正步的 $W_2$ 误差就严格小于 Euler–Maruyama SDE 步。

坦白说，这些假设（强对数凹之类）在真实图像分布上肯定是过于理想化的，这个比较条件也没法在实践里直接验证。但理论的价值在于给出直觉：corrector 步是"朝着目标分布收缩"的，而 SDE 步只是"边缘分布正确的随机游走"，两者的误差结构有本质差异。这就够了。

## 🧪 实验：三个模型，全面压过 baseline

实验设置上有个值得点赞的地方：LC-GRPO 每步要 2 次 NFE（Euler 步一次速度评估，corrector 的 score 又一次），所以作者让每个 baseline 分别跑"相同步数"和"两倍步数"，取较好的结果来比。这种自缚手脚的比法，比那些偷偷占算力便宜的工作厚道多了。

| 模型 | LC-GRPO rollout 步数 | Baseline 步数 | 评测采样 |
|---|---|---|---|
| SD3.5-Medium | 10 | 10 / 20 | 统一 40 步 ODE |
| FLUX.1-Dev | 6 | 6 / 12 | 统一 28 步 ODE |
| HunyuanVideo | 8 | 16 | 统一 50 步 ODE |

baseline 有三个：Flow-GRPO（标准 SDE 采样）、CPS（用 DDIM 采样 rollout 的方案）、DanceGRPO（视频侧）。奖励函数覆盖 OCR（可验证的文字渲染）、HPS-v2.1（人类偏好）、CLIP + HPS-v2.1 多奖励，视频侧用 VideoAlign 的 Visual Quality。

**文生图主结果（SD3.5-Medium，任务指标列是该设置下优化的奖励）**：

| 设置 | 方法 | 任务指标 | Aesthetic | CLIP | ImgRwd | HPS-v2.1 | PickScore |
|---|---|---|---|---|---|---|---|
| OCR | Flow-GRPO | 0.914 | 5.31 | 0.288 | 0.94 | 0.281 | 22.43 |
| | CPS | 0.935 | 5.20 | 0.287 | 0.75 | 0.265 | 22.16 |
| | **LC-GRPO** | **0.960** | 5.33 | 0.291 | 1.00 | 0.280 | 22.46 |
| HPS-v2.1 | Flow-GRPO | 0.381 | 6.30 | 0.267 | 1.41 | 0.357 | 22.89 |
| | CPS | 0.371 | 6.12 | 0.270 | 1.38 | 0.344 | 22.62 |
| | **LC-GRPO** | **0.393** | 6.11 | 0.280 | 1.44 | 0.367 | 23.03 |
| HPS+CLIP | Flow-GRPO | 0.351 / 0.296 | 5.97 | 0.289 | 1.28 | 0.327 | 22.76 |
| | CPS | 0.345 / 0.291 | 5.73 | 0.285 | 1.31 | 0.323 | 22.68 |
| | **LC-GRPO** | **0.356 / 0.302** | 5.73 | 0.297 | 1.33 | 0.332 | 22.93 |

FLUX.1-Dev 上的趋势一致（HPS 设置下 0.384 vs Flow-GRPO 的 0.378），多奖励设置下 PickScore 23.40 也是全场最高。幅度不算爆炸，但注意这是在 baseline 已经跑了两倍步数的前提下拿到的，而且右侧那列泛化指标（Aesthetic、CLIP、ImageReward、PickScore）基本没有塌——说明没有在奖励模型上过拟合，没有明显的 reward hacking。

**文生视频（HunyuanVideo + Visual Quality 奖励）**：

| 方法 | Visual Quality | Motion Quality | Text Alignment | VBench Total | Quality | Semantic | Dynamic Degree | Aesthetic |
|---|---|---|---|---|---|---|---|---|
| HunyuanVideo 基座 | -0.368 | – | 0.351 | 78.74 | 80.94 | 69.96 | 52.8 | 61.28 |
| DanceGRPO | -0.205 | -0.279 | 1.312 | 78.85 | 81.18 | 69.51 | 12.5 | 64.03 |
| CPS | -0.276 | -0.178 | 1.048 | 76.86 | 79.70 | 65.49 | 19.4 | 59.02 |
| **LC-GRPO** | **0.063** | -0.223 | 1.151 | **79.10** | **81.92** | 67.83 | 45.8 | **64.70** |

这个表里我觉得最值得盯的不是 LC-GRPO 把 Visual Quality 从基座的 -0.368 拉到了 0.063（优化目标本身涨是应该的），而是 Dynamic Degree 这一列：DanceGRPO 训完直接崩到 12.5（基座 52.8），CPS 也只有 19.4，LC-GRPO 守住了 45.8。你想想看，视频生成 RL 训完动态程度掉成原来四分之一，那基本就是模型学会了"少动保平安"——画面不动了，画质自然好，reward 自然高。这是典型的 reward hacking 前兆。LC-GRPO 能守住动态性，很大程度上就是因为 rollout 质量和推理分布贴得紧，奖励信号没有失真。

![图5：HunyuanVideo 视频定性对比（猫）](https://arxiv.org/html/2608.05600v1/x6.png)

*图 5：prompt 是"a cat grooming itself meticulously with its tongue"。DanceGRPO 直接训歪了——猫头人身，恐怖谷拉满；CPS 的猫基本静止；LC-GRPO 的猫真的在低头舔爪子，动作幅度和语义都对。*

![图5：HunyuanVideo 视频定性对比（自行车）](https://arxiv.org/html/2608.05600v1/x7.png)

*图 5（续）："a bicycle gliding through a snowy field"。DanceGRPO 色调发灰发暗，CPS 的画面几乎不动，LC-GRPO 生成了黄昏光线下骑手穿越雪地的完整运动。*

## 📊 训练-推理差距：这篇论文的题眼

图 1 是全篇最核心的一张图，也是标题里 "Bridging Train-Inference Gap" 的直接证据：

![图1上：训练奖励与评估奖励的差距对比](https://arxiv.org/html/2608.05600v1/x1.png)

*图 1（上）：FLUX.1-Dev 上 HPS 奖励的训练曲线。实线是训练时 rollout 上的奖励，虚线是 ODE 推理评估的奖励。看 step 0 处的初始差距：Flow-GRPO 是 -0.063，CPS 是 -0.035，LC-GRPO 只有 -0.016。*

仔细品一下这三条线。Flow-GRPO（绿）训练奖励明显低于评估奖励——rollout 太糊，奖励模型给模糊图打低分，模型在低质量的信号上爬坡。CPS（蓝）反过来，训练奖励虚高，评估时掉下来。LC-GRPO（橙）两条线几乎贴在一起，而且收敛高度也是三者中最高的。

![图1下：DrawBench 上的泛化指标](https://arxiv.org/html/2608.05600v1/x2.png)

*图 1（下）：DrawBench 上的五个泛化指标。LC-GRPO（橙）在 HPS-v2.1、PickScore、CLIP 上都是最优或接近最优，没有被优化目标"带偏"。*

说实话，训练-评估 gap 这个视角比单纯刷分有价值得多。LLM 的 RL 后训练里类似的失配问题（比如训练和推理数值精度不一致导致的概率比偏差）最近也被反复讨论，扩散模型这边的问题其实是同一个家族的病。LC-GRPO 相当于给这个病开了一剂有理论说明书的药。

## 🔬 消融：两个反直觉的发现

![图4a：Langevin 步梯度回传的影响（FLUX.1-Dev）](https://arxiv.org/html/2608.05600v1/x4.png)

*图 4（a）：红线是梯度穿过 Langevin 校正步，蓝线是 stop-grad。回传梯度全程稳定领先。*

第一个发现：虽然 Langevin 校正在测试时根本不用，但训练时让梯度流过这个校正步，效果更好更稳。直觉上你可能觉得 corrector 只是个随机化装置，梯度走 ODE 主干就够了——但数据说不行，corrector 里的 score 项本身携带了有价值的优化信号。

![图4b：CFG 与梯度的组合影响（SD3.5-Medium）](https://arxiv.org/html/2608.05600v1/x5.png)

*图 4（b）：四条线对比梯度回传 × CFG 的组合。绿线（stop-grad + 无 CFG）在 600 步附近出现明显塌陷，而"回传梯度 + 无 CFG"（红线）是最终赢家。*

第二个发现更反直觉：Langevin 校正步里**不要**用 CFG。作者的解释我很认同——校正步的职责是定义随机化的采样分布、注入探索，而不是提升视觉质量，加 CFG 反而扭曲了 score 的方向。不用 CFG 还省计算，一举两得。

## 🤔 我的判断

**亮点**：

1. 问题定位准。Flow-based RL 的 train-inference gap 之前大家要么忍着要么用工程手段缓解（比如 MixGRPO 混着用 ODE/SDE），LC-GRPO 第一次把它形式化成"predictor 对齐推理 + corrector 注入探索"的结构，还给了 Wasserstein 收缩的理论保证。
2. 方法够轻。score 白嫖自速度场，转移核还是各向同性高斯，GRPO 的脚手架一行不用改。这种"插进去就能用"的方法，工程 adoption 成本最低。
3. 实验厚道。两倍 NFE 主动让 baseline 补步数，Dynamic Degree 这种容易翻车的指标全程监控。

**保留意见**：

- 理论假设（强对数凹）离真实图像分布挺远，定理 2 的比较条件实践中不可验证。理论更多是直觉背书，不是保证书。
- 每步 2 次 NFE 的成本是实打实的。虽然作者公平地让 baseline 跑了双倍步数，但在真实训练预算下，这个开销换来的收益是否始终划算，论文没有给 wall-clock 层面的对比。
- HunyuanVideo 的 Text Alignment 上 DanceGRPO（1.312）其实高于 LC-GRPO（1.151），说明不同方法在不同指标维度上各有取舍，不是全方位碾压。
- 相关工作里把 SAGE-GRPO、MixGRPO 列为同路线方案，但正文实验只比了 Flow-GRPO / CPS / DanceGRPO，和这两个最直接竞品的正面对决缺席，有点遗憾。

**工程启发**：如果你正在做 flow/diffusion 模型的 RL 后训练，这篇论文最值得抄走的有两样东西——一是"监控训练-评估 gap"这个习惯（在 step 0 量一下初始差距，就能知道你的 rollout 分布偏了多少），二是 predictor-corrector 的解耦思路。就算不用 Langevin，把"对齐推理的确定性步"和"注入探索的随机步"分开设计，这个抽象本身就很值钱。

说到底，RL 的 reward 信号质量上限，是被 rollout 分布决定的。在错误的分布上采集的数据，再精巧的算法也救不回来。LC-GRPO 把这个朴素的道理，用一步校正讲明白了。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
