---
title: "把轮次当成分配单位：TurnOPD 如何让长程 Agent 蒸馏提速 2.29 倍"
date: 2026-07-09
arxiv: 2607.05804
categories:
  - 论文解读
  - Agent 训练
  - On-Policy Distillation
tags:
  - TurnOPD
  - On-Policy Distillation
  - Long-Horizon Agent
  - KL Budgeting
---

# 把轮次当成分配单位：TurnOPD 如何让长程 Agent 蒸馏提速 2.29 倍

你有没有被这样一种情况搞崩溃过：模型在长程 Agent 任务上跑 OPD（On-Policy Distillation），每一步都按部就班做了，KL 损失也在下降，log 看起来很正常——可最后的成功率死活上不去？

我之前在调类似流程的时候碰到过。当时第一反应是"teacher 是不是还不够强"，第二反应是"rollout 太短了所以信号不够"，第三反应是"loss 加权是不是不对"。但说真的，这三个直觉都没说清楚问题到底出在哪。

这篇 [arXiv:2607.05804](https://arxiv.org/abs/2607.05804) 给了我一记相当清醒的"诊断三连"。它先告诉你 vanilla OPD 在长程 Agent 任务里到底把钱（compute）和损失（loss）花在了哪些"轮次"上，再告诉你这个分配为什么本身就是结构性的浪费，最后才给出两个简单得有点不好意思的修复。

## 核心摘要

**痛点**：在长程语言 Agent 任务上，沿用 vanilla OPD（固定 rollout 深度 + trajectory 级 KL 归约）会把绝大部分 compute 浪费在尾部低信息轮次上，同时把绝大部分 KL 损失压在浅层 token 上。结果是：训得越久，越难看到准确率提升，wall-clock 时间却越堆越离谱。

**核心方案**：TurnOPD 用两个 turn-level 预算控制器把"分配单位"从 token 抬回"轮次"——（1）**自适应 rollout 深度控制器** 用周期性的探针 rollout 估算 survivor-weighted KL 质心 $H_{\mathrm{eff}}$ 和成功完成深度的 80% 分位数 $H_{\mathrm{cov}}$，取二者较大者作为下一轮的 rollout cap；（2）**渐进式 turn 归一化损失** 把 KL 聚合权重从 token mass 平滑地插值到按 turn 均匀的 $q_t^{\mathrm{turn}}=1/T$，让训练后期能更公平地把梯度砸到深轮决策上。

**关键效果**：在 ALFWorld（Qwen3-1.7B 学生）、WebShop（Qwen3-1.7B）、Multi-Hop Search（Qwen3.5-2B）三个长程任务上，TurnOPD 把 100 步训练的 wall-clock 时间从 4.42h 砍到 1.93h（ALFWorld-1.7B），2.29 倍加速的同时 Same-Step Avg@4 从 83.0 涨到 86.3；WebShop 上 1.24h vs 1.57h，82.80 vs 76.98。

**一针见血的评价**：这不是模型架构创新，而是"训练循环里被默认假设拖垮的算力分配"的一次系统性重设。方法本身高度模块化，思路甚至可以照搬到 GRPO 类的策略优化上。如果你也在做长程 Agent 的训练，强烈建议先把这篇里的诊断图看一遍，再决定要不要改自己的 recipe。

---

## 论文信息

- **标题**：TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training
- **作者**：Yuhang Zhou（复旦大学 / 腾讯混元）、Kai Zheng（腾讯混元，项目负责人）、Haoling Li（腾讯混元）、Dengyun Peng（复旦大学）、Can Xu（腾讯混元，通讯）、Jingjing Chen（复旦大学，通讯）
- **链接**：https://arxiv.org/abs/2607.05804
- **会议**：ICLR 2026 conference（参考 .bst 与 cls 命名）

## 问题动机：OPD 在长程 Agent 上的"水土不服"

On-Policy Distillation 的好处是"在学生自己的轨迹上对齐 teacher"——teacher 评估学生走过的状态，反向 KL 提供稠密 token 级监督。听起来对长程任务应该很香，理由是：

- 不依赖稀疏 reward；
- teacher 是在学生自己诱导的状态上查询，避免了 off-policy 分布漂移；
- token 级 KL 是稠密的，可以一直监督到 trajectory 末尾。

但问题来了：**长程 Agent 的 trajectory 不是"一段文本"，而是一串"轮次"**。每轮里学生做出一个动作（带推理 + 工具调用），环境给一个观察，下一轮又基于这个新观察继续。这个 turn 结构和 vanilla OPD 的 token 级目标之间有一个根本性的错配。

具体两个错配，论文给了名字——**外部错配**和**内部错配**：

1. **外部错配（rollout 深度层面）**：固定 rollout 深度意味着，不论每轮的"修正信号"有多强，都要采到最大 horizon。但实际上，越往后能"幸存"到那一轮的 trajectory 越少，深轮次 KL 的 survivor 权重迅速衰减。采到的尾部 token 大部分是"已经被 student 自己吃掉的 context"逼出来的低熵续写，KL 噪声大、信号弱。
2. **内部错配（损失分配层面）**：trajectory 级 KL 归约用 token 数做分母。浅轮次 token 密度高（输入长、模型生成多），深轮次幸存少但每轮 token 也不少；综合下来**浅轮次贡献了绝大部分 KL 损失，深轮次被结构性饿死**。

直觉上的"加大 KL、加长 rollout"在这两个错配上都是错的——前者加深了损失不均，后者把算力塞进了噪声区。

## 诊断：把"轮次"作为分配单位会带来什么

论文用 vanilla OPD 在 ALFWorld-4B 和 Multi-Hop Search-2B 上跑了 100 步，把监督信号按 turn 展开。诊断一共有三张图，先看最直觉的那张。

### 诊断 1：每轮 KL 的真实分布

![ALFWorld-4B 与 Multi-Hop Search-2B 上 vanilla OPD 的教师熵与 reverse-KL 随 turn 演变](https://www.mulanai.com/fs/files/0709_6f3ddc12_fig_vani.png)

*图 1：turn-resolved 的教师熵（左）、每轮 reverse-KL（中）和 step-by-turn KL heatmap（右）。上排是 ALFWorld-4B，下排是 Multi-Hop Search-2B。颜色由深到浅对应 turn 0 → 41/11。ALFWorld 上教师熵随 turn 升高而下降，KL 集中在前 10 轮；Multi-Hop Search 上教师熵随 turn 升高，但 KL 仍然前重后轻。*

论文的核心观察：**KL 不是平的、也不是稳的**。

- ALFWorld 上，turn 0 的 KL 在 0.4-0.5 之间，前 20 步就掉到 0.2 以下，剩下 20 步的深轮次 KL 几乎在 0.05-0.15 区间震荡。teacher entropy 也是随 turn 升高而下降的，但**KL 的衰减比 entropy 的衰减更快**。
- Multi-Hop Search 上更戏剧化：teacher entropy 反而随 turn 升高（深轮是 search/read，需要更长尾的分布），但 KL 仍然前重后轻。说明 KL 不完全被 teacher uncertainty 决定，**还有别的结构在压它**。

这种"KL 在尾部塌掉"的现象，第一直觉是"学生学会了"，但论文立刻做了一组 control 来打脸这个直觉。

### 诊断 2：KL 还能不能区分成功/失败轨迹？

定义 $G_t = K_t^{\mathrm{fail}} - K_t^{\mathrm{succ}}$，把成功和失败的 rollout 分别按 turn 算 KL 差。如果 KL 是干净的"还需修正"信号，失败轨迹应该 KL 更大，$G_t > 0$。

![成功与失败轨迹的 per-turn KL gap $G_t$](https://www.mulanai.com/fs/files/0709_38c70013_success_.png)

*图 2：$G_t = K_t^{\mathrm{fail}} - K_t^{\mathrm{succ}}$ 随 turn 的变化。Early / Mid / Late 分别对应训练 0-30 / 30-60 / 60-100 步。ALFWorld 上 $G_t$ 随 turn 上升而**变负**：深轮次反而是成功轨迹 KL 更大；Multi-Hop Search 上 $G_t$ 始终为正但随 turn 衰减。*

结果让人意外：

- **ALFWorld 上 $G_t$ 是负的**——尤其在深轮次。看到这个负号挺反直觉的：deep turn 的 KL 不是"还需修正"，反而是"成功轨迹在更复杂的 context 上 teacher 也更不确定"。失败轨迹在 deep turn 的低 KL，是因为它已经陷入重复/模板化 context，**teacher 也懒得给信号了**。
- **Multi-Hop Search 上 $G_t$ 为正但很弱**，且随 turn 衰减。说明即便在有成功/失败区分的任务上，深轮次 KL 的"结果预测力"也在弱化。

这个 control 实验直接把"深轮次 KL 低 = 学得好"这个直觉打掉了。

### 诊断 3：污染-压缩机制（Contamination-Compression）

论文给了一个形式化解释。学生和 teacher 在某个 context $c$ 下的 next-token 分布可以分解为

$$\pi_S(\cdot\mid c) = \lambda(c) p_F(\cdot\mid c) + (1-\lambda(c)) p_S^{\mathrm{free}}(\cdot\mid c)$$

$$\pi_T(\cdot\mid c) = \lambda(c) p_F(\cdot\mid c) + (1-\lambda(c)) p_T^{\mathrm{free}}(\cdot\mid c)$$

其中 $p_F$ 是被 context 几乎"锁死"的那部分（重复字符串、格式、抄录实体、闭合分隔符等），$\lambda(c)$ 是 context-forced mass。

由 KL 联合凸性立得（论文里给了证明）：

$$\mathrm{KL}\!\left(\pi_S\;\|\;\pi_T\right) \le (1-\lambda(c)) \cdot \mathrm{KL}\!\left(p_S^{\mathrm{free}}\;\|\;p_T^{\mathrm{free}}\right)$$

这就是 Proposition 1 里的 **contamination compression bound**：观测到的 KL 只能捕捉到"自由分量"中 $\le 1-\lambda(c)$ 的那部分。

直觉含义非常朴素——**在 student 自己生成的 context 上，两边都被 context 拉成同一个 surface continuation，policy 级的真实分歧被压缩进自由分量**。越长的 rollout、越多 student 自己的输出，$\lambda(c)$ 越大，观测到的 KL 越虚。

由此得到非对称推论：**失败轨迹往往陷入重复动作、循环、模板化续写**——这些恰恰是 $p_F$ 最强的位置，因此 $\lambda_t^{\mathrm{fail}} > \lambda_t^{\mathrm{succ}}$，失败轨迹的 KL 被**更狠地压下去**。这就解释了图 2 里 $G_t$ 变负。

### 诊断 4：损失预算的去向

把诊断推进到"优化器把钱花到了哪"。定义 turn-$t$ 的损失占比 $s_t^{\mathrm{traj}}$ 为该 turn 的 KL 总和除以整 batch 的 KL 总和。

![vanilla OPD 下 trajectory 级 KL 归约的 turn 损失占比](https://www.mulanai.com/fs/files/0709_1dbdc6ae_fig_loss.png)

*图 3：ALFWorld-4B 与 Multi-Hop Search-2B 上每轮 KL 损失占比。线条面板给出每轮随训练步的曲线，heatmap 给出原始 step-by-turn 分布。ALFWorld 上 turn 0 一个轮就吃掉约 25% 的 KL 损失，前 3 轮接近 50%，可靠深轮次（第 3 段）只分到 3.6%–4.5%；Multi-Hop Search 上前 3 轮占 38%–40%，深轮次 11%–13%。*

把这张图和论文里 Table 2 的数字合在一起看（深/浅 raw KL 比、deep support、deep loss budget）：

| 任务 | 阶段 | Deep/Shallow raw KL | Deep support | Deep loss budget |
| --- | --- | --- | --- | --- |
| ALFWorld | 早训 | 31% | 23.0% | **3.6%** |
| ALFWorld | 晚训 | 42% | 18.2% | **4.5%** |
| Multi-Hop Search | 早训 | 90% | 17.3% | **12.9%** |
| Multi-Hop Search | 晚训 | 92% | 15.5% | **11.1%** |

两点结论：

1. **trajectory 级归约产生预算错配**——深轮次从优化器拿到的损失份额严重不足；
2. **硬性 turn-level 归约（直接给每轮等权）可能矫枉过正**——ALFWorld 深轮次的 raw KL 只有浅轮的 31%–42%，Multi-Hop Search 上 raw KL 几乎平，但 deep support 只有 15%–17%。一上来就把所有 turn 等权，**会过快地放大低支持度的深轮估计**。

## TurnOPD 的方法：把"轮次"做成真正的分配单位

基于诊断，论文提出两个 turn-level 预算控制器。一个管"采多深"（rollout 深度），一个管"怎么分"（loss 归一化）。

![TurnOPD 总体框架](https://www.mulanai.com/fs/files/0709_3017b06d_TurnOPD_.png)

*图 4：TurnOPD 总体框架。左边（红）是 vanilla OPD 的两个错配：external mismatch（采到最大 horizon 浪费尾部 compute）和 internal mismatch（trajectory 级归约把 KL 损失压在浅 token 上，深轮次饿死）。右边（青）是 TurnOPD 的两个修复：自适应 rollout 深度控制器 + 线性 round-normalization mixing。底部：训练过程产生更好的 accuracy-time frontier。*

### 控制器 1：自适应 rollout 深度

设 $H^\star$ 是最优（但不可观测）的 rollout 深度——浅了欠探索，深了浪费 compute。论文假设它存在（附录里给了一个 efficiency-coverage sandwich 的存在性证明），然后用两个互补信号构造一个在线代理：

**（a）效率侧 $H_{\mathrm{eff}}$**：把每轮 KL 当成"蒸馏价值"的一个代理，乘以该轮的幸存率 $n_t/n_0$，得到一个 survivor-weighted 分布 $q_t$，再取一阶矩：

$$m_t = [K_t]_+ \cdot \frac{n_t}{n_0}, \quad q_t = \frac{m_t}{\sum_j m_j + \epsilon}$$

$$\bar H_{\mathrm{eff}} = \sum_t t \, q_t, \quad H_{\mathrm{eff}} = \mathrm{round}(\bar H_{\mathrm{eff}})$$

直觉上：$H_{\mathrm{eff}}$ 跟着"监督信号的 centroid"走，浅轮次 KL 大时它浅，深轮次持续有信号时它自然加深。

**（b）覆盖侧 $H_{\mathrm{cov}}$**：只看成功轨迹的完成深度，取 80% 分位数 $H_{\mathrm{cov}} = \hat Q_{0.8}(L_{\mathrm{succ}})$。这是一个**下界**——确保 rollout 不会短到丢掉大多数成功完成。

最后：

$$H_{\mathrm{ctrl}} = \max(H_{\mathrm{eff}}, H_{\mathrm{cov}})$$

外加一个 EMA 平滑：$\bar H_k = (1-\alpha_{\mathrm{ema}}) \bar H_{k-1} + \alpha_{\mathrm{ema}} H_{\mathrm{ctrl},k}$，再用 $\hat H_{k+1} = \mathrm{clip}(\mathrm{round}(\bar H_k)+1, H_{\min}, H_{\max})$ 作为下一次的 rollout cap。

实现上的一个细节是 **probe rollout**——每隔 $r_{\mathrm{probe}}$ 步（论文设 8 步）做一次完整 horizon 的 rollout，**只用 probe 出来的未截断轨迹去估 $H_{\mathrm{eff}}$ 和 $H_{\mathrm{cov}}$**。常规训练步用当前 cap $\hat H_k$ 截断即可。这一步对正确性很关键，否则用截断后的轨迹估 $H_{\mathrm{eff}}$ 会持续系统性低估。

### 控制器 2：渐进式 turn 归一化

损失归一化用线性插值：

$$q_t^{\mathrm{blend}} = (1-\alpha) q_t^{\mathrm{traj}} + \alpha \, q_t^{\mathrm{turn}}$$

其中 $q_t^{\mathrm{traj}} = n_t / \sum_j n_j$ 是 trajectory 级（token 质量加权），$q_t^{\mathrm{turn}} = 1/T$ 是 turn 级（轮次均匀）。插值系数 $\alpha$ 绑在训练进度上：

$$\alpha = \mathrm{clip}((\mathrm{progress} - s)/(e - s), 0, 1)$$

论文里 $s=0, e=1$，$\mathrm{progress} = k/K$。也就是 $\alpha$ 从 0 线性增长到 1。

直觉含义：**早期保持 token-level 归约稳定（KL 自然集中在浅轮次，token-level 归约不会引入额外噪声），后期逐渐把"注意力"挪到深轮次，让那些 survivor 少但有信号的位置也能拿到梯度**。

两个控制器一前一后地接上：rollout 深度决定"采多深"，loss 归一化决定"采到之后怎么分"。

## 实验：把 frontier 往外推

### 主结果

| 任务 | 学生 | Teacher | 方法 | Least-Time Avg@4 | Same-Step Avg@4 | Wall (h) | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ALFWorld | Qwen3-1.7B | Qwen3-8B-GRPO | Zero-Shot | 0.00 | 0.00 | -- | -- |
| ALFWorld | Qwen3-1.7B | Qwen3-8B-GRPO | Teacher | 90.75 | 90.75 | -- | -- |
| ALFWorld | Qwen3-1.7B | Qwen3-8B-GRPO | Vanilla OPD | 73.52 | 83.00 | 4.42 | 1.00× |
| ALFWorld | Qwen3-1.7B | Qwen3-8B-GRPO | TCOD-F2B | 80.06 | 80.06 | **1.87** | **2.37×** |
| ALFWorld | Qwen3-1.7B | Qwen3-8B-GRPO | **TurnOPD** | **85.60** | **86.29** | 1.93 | 2.29× |
| ALFWorld | Qwen3-4B | Qwen3-8B-GRPO | Vanilla OPD | 90.79 | 91.81 | 2.86 | 1.00× |
| ALFWorld | Qwen3-4B | Qwen3-8B-GRPO | TCOD-F2B | 86.50 | 86.50 | **1.89** | **1.51×** |
| ALFWorld | Qwen3-4B | Qwen3-8B-GRPO | **TurnOPD** | **91.73** | **92.21** | 2.16 | 1.33× |
| Multi-Hop Search | Qwen3.5-2B | Qwen3.5-9B-GRPO | Vanilla OPD | 45.77 | **47.82** | 4.45 | 1.00× |
| Multi-Hop Search | Qwen3.5-2B | Qwen3.5-9B-GRPO | TCOD-F2B | 45.64 | 47.77 | 3.80 | 1.17× |
| Multi-Hop Search | Qwen3.5-2B | Qwen3.5-9B-GRPO | **TurnOPD** | **47.24** | 47.24 | **2.94** | **1.51×** |
| WebShop | Qwen3-1.7B | Qwen3-8B-GRPO | Vanilla OPD | 76.98 | 81.65 | 1.57 | 1.00× |
| WebShop | Qwen3-1.7B | Qwen3-8B-GRPO | TCOD-F2B | 80.45 | 81.66 | 1.33 | 1.18× |
| WebShop | Qwen3-1.7B | Qwen3-8B-GRPO | **TurnOPD** | **82.80** | **82.80** | **1.24** | **1.26×** |

把数字串成 accuracy-time frontier 图：

![四组 (任务, 学生) 设置下的 Val Avg@4 随 wall-clock training time 的演变](https://www.mulanai.com/fs/files/0709_5fcc6770_iso_time.png)

*图 5：横轴是累计 wall-clock 训练时间（小时），纵轴是 Val Avg@4。浅青为 vanilla OPD baseline，中蓝为 TCOD-F2B，深蓝为 TurnOPD。四张子图分别是 ALFWorld-1.7B / ALFWorld-4B / Multi-Hop Search 2B / WebShop-1.7B。TurnOPD 的曲线整体处于 frontier 的左上外侧。*

**Least-Time 评估**（看最小 wall-clock 即可达到的精度）下，TurnOPD 在所有 4 组 (任务, 学生) 设定上都是最强学生训练方法。**Same-Step 评估**（同 100 步后看精度）下，ALFWorld-1.7B / 4B、WebShop 上 TurnOPD 也是最优；Multi-Hop Search 上 vanilla OPD 略胜（47.82 vs 47.24），但 TurnOPD 的 wall-clock 仅 2.94h，是 vanilla 的 66%。

特别值得注意的是 **ALFWorld-4B 上 TurnOPD（91.73）甚至超过了 teacher 引用（90.75）**。这一度让我有点怀疑是不是 teacher 引用没给对——但论文里 teacher 的 GRPO 训练是独立的、引用为 90.75 没有算错，TurnOPD 真的超过了 8B-GRPO teacher。这在 OPD 类工作里其实并不奇怪（student 走自己的轨迹，teacher 给的 KL 信号会被"再加工"），但能稳超 1 个点说明两个控制器一起工作确实让优化过程非常高效。

### Diagnostic–Controller Alignment：控制器真的在做它该做的事吗？

论文在主结果之外专门验证了 $H_{\mathrm{eff}}$ / $H_{\mathrm{cov}}$ 的实际行为。

![ALFWorld / WebShop / Multi-Hop Search 上 $H_{\mathrm{eff}}$、$H_{\mathrm{cov}}$ 和 ema-H 随训练步的变化](https://www.mulanai.com/fs/files/0709_90afd2f7_horizon.png)

*图 6：横轴是训练步（注意 x 轴标签与论文正文一致，单位是 step 而非小时——这是图本身的轴），纵轴是 rollout depth（0-based turn index，实际 cap 大约 +1）。浅青是 $H_{\mathrm{eff}}$，深蓝是 $H_{\mathrm{cov}}$，红色是 EMA 后的实际 cap。ALFWorld 上 $H_{\mathrm{cov}}$ 主导（深轮次出现成功完成时驱动 cap 加深），WebShop 上保持温和的中等长度，Multi-Hop Search 上 $H_{\mathrm{eff}}$ 和 $H_{\mathrm{cov}}$ 交替主导。*

这张图说明：

- ALFWorld 上，**$H_{\mathrm{cov}}$ 在训练中段之后稳定主导**。当成功 rollout 出现，coverage 拉高 cap；这恰好对应 rollout 长度的 5 → 25 跳变；
- WebShop 上，**$H_{\mathrm{cov}}$ 始终温和**（保持在 5-8 区间），这与 WebShop 任务相对短是一致的；
- Multi-Hop Search 上，**$H_{\mathrm{eff}}$ 在前期主导**（任务长尾但前几轮信号强），到训练后期 $H_{\mathrm{cov}}$ 偶尔拉高。

这其实是非常好的"代理变量对齐 sanity check"——控制器没有出现"统一拍一个常数"或者"震荡"那种失控行为，而是**跟着任务和训练阶段自适应**。

## 消融：两个控制器各干了什么

### 拆解两个控制器

ALFWorld-1.7B 上的 Same-Step Avg@4 / 100 步 wall time：

| 配置 | Adaptive Depth | Linear Blend | Avg@4 | Wall (h) |
| --- | --- | --- | --- | --- |
| Vanilla OPD | | | 83.0 | 4.42 |
| + Adaptive Depth | ✓ | | 82.8 | 1.96 |
| + Linear blend norm | | ✓ | 85.1 | 2.59 |
| **TurnOPD** | ✓ | ✓ | **86.3** | **1.93** |

配合训练曲线：

![ALFWorld-1.7B 上 5 个变体的 Val Avg@4 随训练时间的演变](https://www.mulanai.com/fs/files/0709_a317acbf_componen.png)

*图 7：5 个变体——Vanilla OPD、Linear Blend Norm、Adaptive Rollout Depth、TurnOPD（两者合用）、Adaptive Rollout Depth + Turn Norm。横轴是 wall time（小时），纵轴是 Val Avg@4。Adaptive Depth 单用省时间但精度持平；Linear Blend 单用涨精度但耗时间；二者合用在 1.9h 处达到 86.3 最高点。*

两个控制器**功能正交**：

- **Adaptive Depth 是"效率杠杆"**：单用能把 wall time 从 4.42h 砍到 1.96h（-56%），但精度只微降 0.2 个点（83.0 → 82.8）。说明单纯截断 rollout 不够，**还得让优化器把钱花对地方**。
- **Linear Blend 是"优化杠杆"**：单用让精度涨 2.1 个点（83.0 → 85.1），wall time 只降到 2.59h。说明 turn 归一化直接提升了训练效率，但没解决"采到尾部低信息"问题。
- **合用**：精度 86.3（最高），wall time 1.93h（最低之一）。两个杠杆叠加，既有免费的 compute 节省，又有精度提升。

这印证了论文的核心论点：**external mismatch 和 internal mismatch 是两个独立的问题，必须分别处理**。

### KL 归一化方案的对比

| KL 归一化方案 | Same-Step Avg@4 | Deep budget (E/M/L) | α (E/M/L) |
| --- | --- | --- | --- |
| Trajectory-level | 83.0 | 3.2% / 0.7% / 1.2% | 0 / 0 / 0 |
| Turn-level（硬切） | 85.0 | 29.9% / 32.0% / 31.9% | 1 / 1 / 1 |
| **Linear blend (ours)** | **85.1** | 12.8% / 21.6% / 27.7% | 0.17 / 0.50 / 0.83 |

trajectory-level 把深轮次预算压到几乎 0；硬切直接给 30%+，但**早训阶段会过快地放大低支持度估计**；Linear blend 在早训时 α=0.17（基本还是 token 主导），到中训 α=0.50，到晚训 α=0.83。**进度条式的迁移**让深轮次的预算份额从 12.8% 涨到 27.7%，稳且有用。

### 覆盖分位数 $p$ 的敏感性

![Success-CDF 与 Full-CDF 下不同 $p \in \{0.4, 0.6, 0.8\}$ 的 Val Avg@4 与 EMA horizon](https://www.mulanai.com/fs/files/0709_56b4aaab_CDF.png)

*图 8：四张子图——(a) Success-CDF 下的 Val Avg@4 随 wall time；(b) Success-CDF 下的 ema-H 随 step；(c) Full-CDF 下的 Val Avg@4 随 wall time；(d) Full-CDF 下的 ema-H 随 step。三条曲线分别是 $p=0.4, 0.6, 0.8$。Full-CDF 的 ema-H 整体被推高（30-50），Success-CDF 保持 5-25。*

两个观察：

1. **$p$ 直接控制 horizon**。$p=0.6$（Full-CDF）下达到 85.1 / 1.66h，$p=0.8$ 下达到 85.8 / 1.83h——稍高的 $p$ 换稍高的精度和稍多的时间。**这是一个可调旋钮**，不是固定超参。
2. **CDF 来源显著影响效率**。Full-CDF 比 Success-CDF 激进——它把所有 trajectory（含卡到 max horizon 的失败）都算进去，会**系统性高估需要的深度**。这给了一个有用的工程提示：**如果你的任务很难定义"成功"（比如开放式 web research），用 Full-CDF 时把 $p$ 调小**（比如 0.4-0.6）就能匹配 Success-CDF 的效率。

## 我的判断

读完后我有几个比较明确的感受，按重要性排：

**第一，诊断是这篇最值钱的部分**。两张诊断图（per-turn KL、loss share）一旦看进去，就很难再用"loss 在掉、reward 在涨但准确率不动"来搪塞自己了。它把"为什么长程 OPD 训不到位"从一句感叹，拆成了两个可观测、可定位的错配。这个分析能力本身比 TurnOPD 方法更值得带回家。

**第二，方法本身的"工程聪明"大于"理论突破"**。两个控制器用到的工具都很标准——EMA、quantile、线性插值——但合起来恰好踩中两个错配。这其实是个非常"研究员式"的判断：不要在每个错配上各拍一个超参，**先把结构性的分配错误识别出来，再各给一个对应的修**。这种思路在 GRPO 类工作里也很容易复用（reward shaping、curriculum 设计都能用类似框架）。

**第三，contribution 是真实但不夸张的**。ALFWorld-1.7B 的 Same-Step Avg@4 涨 3.3 个点（83.0 → 86.3），ALFWorld-4B 涨 0.4 个点（91.81 → 92.21），WebShop 涨 1.15 个点，Multi-Hop Search 在 Least-Time 上涨 1.47 个点。**没有一个任务有"突破性"涨幅，但全部任务都同时涨精度和降时间**。这正是把 frontier 外推应有的样子：不是单点大胜，是整条曲线被推出去。

**第四，与 TCOD-F2B 的对比值得展开说一下**。TCOD-F2B 是同期工作，用 curriculum over trajectory length 来节省 compute。在 ALFWorld-1.7B 上它比 TurnOPD 还快（1.87h vs 1.93h），但精度掉到 80.06。**它只解决 external mismatch，不解决 internal mismatch**——这正好印证了 TurnOPD 的论点：只截 rollout 不重分配 loss，**省下的时间是被错配的优化浪费的**。我反而觉得 TurnOPD 的 ablation 表比它的主结果更说服人。

**第五，几个我有点想追问的细节**：

- **probe rollout 的频率 8 步是否普遍合适**？论文没给 sensitivity study 扫 probe_interval。直觉上，频率太低会让控制器跟不上训练动态，太高又稀释了 compute 节省。这是一个值得在你自己场景里重新调的超参。
- **teacher query 的成本**没讨论。OPD 本身要 teacher 在 student 的 prefix 上 forward 一次；如果 student rollout 变短，teacher 的总 query 量等比下降，这是 TurnOPD 真正能省 compute 的地方。但论文没量化这个 teacher 端的节省，给的 wall-clock 是总训练时间。
- **深轮次的 raw KL 衰减到底在 ALFWorld 上是 teacher 自身 entropy 衰减造成的，还是 contamination compression 造成的**？论文做了一个"非对称压缩解释 $G_t$ 变负"的理论，但没有用 entropy normalization 控制变量来切分两者的相对贡献。这是一个可以做 follow-up 的方向。

**最后一个比较元层面的感受**：长程 Agent 训练这个领域，目前最缺的不是"再训一个 SOTA 模型"，而是**把 compute 分配这件事讲清楚**。Rollout 太长、token-level 归约、token-level reward shaping、token-level advantage、token-level credit assignment——这些"默认假设"在 turn 这个粒度上其实都站不住。这篇论文做的事就是把其中两个默认假设显式化，然后用"两个简单的控制器"修复。**对任何在长程 Agent 训练里摸爬滚打过的人，这种"把经验直觉形式化"的工作比又一个新 benchmark 实在得多**。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
