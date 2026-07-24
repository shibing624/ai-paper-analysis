---
title: 黎曼等距优化破解LLM强化学习探索坍缩
date: 2026-07-24
arxiv_id: 2607.10169
authors: Zhicheng Cai, Xinyuan Guo, Hanlin Wu, Mingxuan Wang, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou
venue: ICML 2026
short_name: RiemannianIsometric
---

# 黎曼等距优化破解LLM强化学习探索坍缩

你有没有这种感觉:LLM 强化学习训到一半,reward 还在涨,但生成的多样性肉眼可见地在崩塌。不同 prompt 出来的回答越来越像,推理路径越走越窄,最后模型就像一个"答套路题机器"——它确实答得越来越准,但凡遇到稍微偏一点的题,直接翻车。

这个问题有个专门的名字,叫 **exploration collapse**(探索坍缩),是 LLM RL 训练里最让人头疼的顽疾之一。从 GRPO 到 DAPO 再到 GSPO,过去一年的改进基本都集中在经验调参层面——加 Clip-Higher、把重要性采样从 token 级升到序列级、动态采样等等。但这些方法有一个共同点:它们都是"症状性修复",没人说清楚 PPO-Clip 为什么会失败。

直到这篇 [arXiv:2607.10169](https://arxiv.org/abs/2607.10169) 出现。作者直接撕开了一个更底层的口子:**PPO-Clip 用错了度量衡**。它隐式地拿 Euclidean 距离衡量策略差异,但策略天然住在由 KL 散度诱导的 **Riemannian 流形**上——这俩压根不是一回事。几何不匹配才是探索坍缩的根因。

## 核心摘要

**痛点**:PPO-Clip 让 LLM 强化学习在低概率动作上几乎不更新,高概率动作上又过度激进,导致策略快速坍缩到少数高概率路径上,丢掉所有"歪门邪道"的探索可能。

**方案**:**RIPO**(Riemannian Isometric Policy Optimization),用基于概率依赖的动态裁剪边界 $\epsilon_{s,a} = \sqrt{\delta / \pi_{\theta_{\text{old}}}(a|s)}$ 替换 PPO-Clip 的固定边界,让每个动作在流形上走等距的一步。

**效果**:在 4 个模型 × 7 个数学竞赛 benchmark 上,相对 GRPO 平均涨 17.1%~37.2%;Qwen3-1.7B 上 AIME24 涨 62%(从 11.3 涨到 18.3);在编程和搜索任务上也稳定领先。

**定位**:**底层原理突破,不是工程整合**。把"信息几何"这套在传统 RL 里被遗忘的老工具,精准地用到了 LLM 这个被"经验调参"统治了一年多的领域里。

---

## 论文信息

| 字段 | 内容 |
|---|---|
| 标题 | Beyond Euclidean Clipping: Overcoming Exploration Collapse in LLM RL via Riemannian Isometric Policy Optimization |
| 作者 | Zhicheng Cai, Xinyuan Guo, Hanlin Wu, Mingxuan Wang, Wei-Ying Ma, Ya-Qin Zhang, Hao Zhou |
| 提交日期 | 2026-07-11 |
| 会议 | ICML 2026 |
| 链接 | https://arxiv.org/abs/2607.10169 |

---

## 问题动机:PPO-Clip 到底错在哪?

先说背景。PPO 的 clip 机制是过去 8 年 RL 训练的基石:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}\left[\min\left(r(\theta)\hat{A}, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\hat{A}\right)\right]$$

其中 $r(\theta) = \pi_\theta(a|s) / \pi_{\theta_{\text{old}}}(a|s)$ 是重要性采样比。这个 $\epsilon$ 通常是 0.2,对所有动作一视同仁。

但这就有问题了:同一个 $|r-1| = 0.2$ 的更新幅度,对高概率动作($\pi=0.8$)和低概率动作($\pi=0.01$)来说,**几何含义完全不同**。

KL 散度的二阶 Taylor 展开告诉你,真正衡量"两个策略离多远"的应该是:

$$d_{\text{geom}}(\pi_{\theta_{\text{old}}}, \pi_\theta) = \frac{1}{2}\sum_a \pi_{\theta_{\text{old}}}(a|s) (r_{s,a} - 1)^2$$

注意这里有个 $\pi_{\theta_{\text{old}}}$ 在前面——**这是 Riemannian 流形的 Fisher 信息矩阵在指数族上的对角元素**。换句话说,流形上的"距离"是概率加权的。

举个例子就明白了:设 $\delta = 0.02$,看一个 PPO-Clip 的更新:

| 动作概率 $\pi_{\text{old}}$ | PPO-Clip 允许的 $\pi_{\text{new}}$ | PPO-Clip 消耗的几何距离 | 问题 |
|---|---|---|---|
| 0.8 (高概率) | 0.96 | $0.5 \times 0.8 \times 0.2^2 = 0.016$ | 正常 trust region |
| 0.01 (低概率) | 0.012 | $0.5 \times 0.01 \times 0.2^2 = 0.0002$ | 远没花完预算 |

**同一个 $\epsilon$,高概率动作走了一大步(占满 trust region),低概率动作几乎没动。** 这就是为什么 LLM 训练中那些"罕见但关键的推理 token"——比如突然想到用某个反证法、突然切换到几何思维——永远得不到强化。模型只在"老套路"上越来越熟练,这就是探索坍缩。

> 直觉地说:PPO-Clip 是在用平面直角坐标系量"多远",但策略住在曲面上,曲面上各点的"单位长度"本身就不一样。

作者把这件事上升到了信息几何的层面:KL 散度在指数族策略空间上诱导的 Fisher 信息矩阵是对角的 $\text{diag}(\pi_{\theta_{\text{old}}})$,所以自然距离是非均匀的。PPO-Clip 的固定边界相当于强行让所有方向上的"步长"一致,这就是**几何不匹配**(geometric mismatch)。

---

## 方法核心:RIPO 的等距更新

RIC(Riemannian Isometric Clip)的核心想法其实非常干净——既然流形距离长这样:

$$d_{\text{geom}} = \frac{1}{2} \pi_{\theta_{\text{old}}}(a|s) (r-1)^2 \leq \delta$$

那我直接解出 $r$ 的允许范围:

$$|r - 1| \leq \sqrt{\frac{2\delta}{\pi_{\theta_{\text{old}}}(a|s)}}$$

吸掉常数 2 之后,写成:

$$\boxed{\epsilon_{s,a}(\pi_{\theta_{\text{old}}}) = \sqrt{\frac{\delta}{\pi_{\theta_{\text{old}}}(a|s)}}}$$

这就是 RIPO 的动态裁剪边界。再用 $\delta = 0.02$ 算一次:

| 动作概率 $\pi_{\text{old}}$ | RIPO 边界 $\epsilon$ | RIPO 允许的 $\pi_{\text{new}}$ | 几何距离 |
|---|---|---|---|
| 0.8 (高概率) | $\sqrt{0.025} \approx 0.158$ | 0.926 | $0.5 \times 0.8 \times 0.158^2 \approx 0.01$ |
| 0.01 (低概率) | $\sqrt{2} \approx 1.414$ | 0.024 | $0.5 \times 0.01 \times 1.414^2 \approx 0.01$ |

**两个动作消耗的 trust region 都是 0.01**——真正做到了 Riemannian 等距。0.8 那个高概率动作被更紧地限制(从 0.96 收到 0.926),0.01 那个低概率动作从 0.012 开放到 0.024,直接翻倍。

> 通俗点说:PPO-Clip 是"不管远近都迈同样大小的步子",RIPO 是"不管远近都走同样的几何距离"。后者的世界才是平的。

完整的 RIPO 目标函数(公式 15)就是用这个动态 $\epsilon_{i,t}(\pi_{\theta_{\text{old}}})$ 替换 PPO-Clip 的固定 $\epsilon$:

$$\mathcal{J}_{\text{RIPO}} = \mathbb{E}\left[\frac{1}{\sum|o_i|}\sum_{i,t}\min\left(r_{i,t}\hat{A}_{i,t},\, \text{clip}\!\left(r_{i,t}, 1-\epsilon_{i,t}, 1+\epsilon_{i,t}\right)\hat{A}_{i,t}\right)\right]$$

其他部分沿用 GRPO 的组相对优势 + DAPO 的 token 级策略梯度。

### Bias-Variance Trade-off 也讲得通

RIPO 还能从另一个角度看——重要性采样的方差上。

PPO-Clip 把 $r$ 截断到 $[1-\epsilon, 1+\epsilon]$,对低概率 token $x'$,方差贡献 $v(x') = \pi_{\theta_{\text{old}}}(x')(1+\epsilon)^2$ 随 $\pi \to 0$ 趋零,**这能压方差但也引入了大偏差**——低概率动作的信号直接被砍掉。

RIPO 的方差贡献是:

$$v(x') = \pi_{\theta_{\text{old}}}(x')\left(1 + \sqrt{\frac{\delta}{\pi_{\theta_{\text{old}}}(x')}}\right)^2 \approx \mathcal{O}(\delta)$$

**密度无关的常数阶方差**——低概率动作也有方差的"地板",但又比朴素重要性采样的 $\pi_{\text{old}} r^2$ 小(因为 $r$ 被压住了)。这是一个有原则的偏差-方差权衡:方差严格小于标准 IS,偏差远小于 PPO-Clip。

---

## 实验结果

### 主实验:7 个数学 benchmark,4 个模型

作者在 Qwen3-1.7B/4B/8B-Base、Llama3.2-3B-Instruct 上对比了 GRPO、DAPO、GSPO、GMPO、DCPO。**核心结果(Qwen3-1.7B-Base,Avg@8)**:

| Method | AIME24 | AIME25 | AMC23 | HMMT25 | BRUMO25 | CMIMC25 | SMT25 | 平均 |
|---|---|---|---|---|---|---|---|---|
| GRPO | 11.3 | 10.8 | 33.8 | 0.0 | 15.0 | 0.3 | 7.3 | 11.2 |
| DAPO | 15.0 | 12.1 | 39.7 | 0.8 | 9.2 | 1.9 | 7.8 | 12.4 |
| GSPO | 16.3 | 11.2 | 40.0 | 1.6 | 12.9 | 2.8 | 8.5 | 13.3 |
| GMPO | 17.5 | 12.5 | 40.9 | 0.8 | 15.0 | 3.8 | 6.6 | 13.9 |
| DCPO | 17.1 | 11.3 | 46.9 | 0.8 | 15.4 | 1.3 | 10.8 | 14.8 |
| **RIPO** | **18.3** | **12.9** | 46.1 | 1.3 | 15.4 | 3.2 | **10.4** | **15.4** |

AIME24 从 11.3 涨到 18.3,**62% 的相对提升**——这就是论文摘要里那个"up to 60% improvement on AIME24"的来源。

更大的模型同样稳赢。Qwen3-8B-Base 平均 38.5 vs GRPO 28.5(**35.1%**),AIME24 涨到 43.8(GRPO 才 31.7)。

### 训练动态:RIPO 真的"治"住了探索坍缩

下面这张图是 Qwen3-8B 在 DAPO-Math-17k 上训了 300 步的全过程:

![图1:RIPO 训练动态](https://arxiv.org/html/2607.10169v1/figs/aime-acc.png)

*图1(a):AIME24 Avg@8 随训练步数变化。RIPO 200 多步就到 44,其他方法 300 步还在 30 出头晃。*

![图1(b):策略熵曲线](https://arxiv.org/html/2607.10169v1/figs/aime-entropy.png)

*图1(b):策略熵(对数轴)。GRPO 熵一路崩到 $10^{-1.5}$ 附近——基本已经在"答套路题";DAPO 熵失控涨到 5 以上——完全散乱;RIPO 熵先快速下降到 $10^{-1}$ 附近,然后稳稳维持,这是理想的"持续探索"区间。*

![图1(c):梯度范数](https://arxiv.org/html/2607.10169v1/figs/aime-grad.png)

*图1(c):梯度范数。DAPO 后期出现剧烈尖峰(典型的训练崩溃前兆),GRPO 持续在 0.15-0.2 区间震荡,RIPO 几乎贴在 0.1 的一条平线上——这是非常稳的优化信号。*

![图1(d):裁剪 token 比例](https://arxiv.org/html/2607.10169v1/figs/aime-clipratio.png)

*图1(d):被 clip 掉的比例。GSPO 大概 8%(序列级裁剪太狠),DCPO/GMPO 极低(几乎不裁剪,意味着 trust region 失效),RIPO 维持在 $5 \times 10^{-5}$ 这个很合理的水位——既没在瞎放飞,也没把梯度全砍掉。*

### δ 的对称性很关键

![图2:δ 消融](https://arxiv.org/html/2607.10169v1/figs/abs-re.png)

*图2(a):不同 {δ_low, δ_high} 组合的 reward 曲线。对称设置 {0.02,0.02} 到 {0.08,0.08} 全都稳步涨到 0.6+,但只要 δ_high 远小于 δ_low(不对称),训练就在 50 步内彻底崩掉——奖励从 0.4 砸到 0.2,熵爆到 5 以上。*

直觉上为什么不对称会崩?因为 δ_high 太小意味着低概率动作的更新被压住,跟 PPO-Clip 没区别,直接退化成探索坍缩;δ_low 太大又意味着高概率动作放飞,Loss 爆炸。

> 这张图其实是个**重要的工程警告**:虽然公式 $\epsilon = \sqrt{\delta/\pi}$ 看起来很优雅,但 δ_low/δ_high 仍然是要调的超参,不能像某些论文宣传的那样"零调参"。

### 泛化性:PPO 目标、编程、搜索

把 RIPO 的裁剪机制迁到传统 PPO(不是 GRPO)上,GSM8K 上 0.5B~14B 四个模型全部涨点(论文表 4)。0.5B 涨 3.1 个点,1.5B 涨 2.4,7B 涨 1.8,14B 涨 1.2——小模型受益更明显,这也符合"小模型探索能力更弱"的常识。

在 Qwen3-8B 上做 Codeforces/CodeContest/TACO/APPS 等编程任务,平均 44.9 vs GRPO 39.7(涨 13.2%);在 TriviaQA/PopQA/HotpotQA 等搜索任务上,平均 43.4 vs GRPO 37.7(涨 15.1%)。这说明 RIPO 的几何不匹配修复不是数学推理特有的,是个**通用机制**。

### Pass@k:这是最让我信服的数据

AIME25 上(Qwen3-8B-Base,表 5):

| Method | Pass@1 | Pass@8 | Pass@16 | Pass@32 | Pass@64 | Pass@128 |
|---|---|---|---|---|---|---|
| GRPO | 20.4 | 36.5 | 40.8 | 45.5 | 50.1 | 53.3 |
| DAPO | 20.6 | 33.4 | 35.7 | 37.4 | 39.1 | 40.0 |
| DCPO | 26.4 | 39.5 | 44.7 | 48.5 | 54.7 | 58.9 |
| **RIPO** | **30.4** | **43.2** | **47.0** | **50.8** | **55.6** | **60.0** |

Pass@1 涨 10 个点,Pass@128 涨 6.7 个点——这才是真正的"探索能力"指标,因为它衡量的是"采样足够多次时,模型能不能找到正确答案"。RIPO 在所有 $k$ 上都最高,而且 $k$ 越大领先优势保持得越稳(GRPO 从 Pass@1 的 20.4 一路爬到 Pass@128 的 53.3,说明它其实藏着多样性,只是平均分上不去)。

> 我自己的解读:Pass@k 的数据基本能"实锤" RIPO 是真的在维持探索能力,而不只是把答案集中到某个模式上。这是这篇文章最让我觉得信服的地方。

---

## 我的判断

**亮点**:
- **理论框架站得住脚**。KL→Fisher metric→Riemannian distance 的推导是教科书级的清晰,不是硬凑的数学包装。命题 3.1 把 PPO-Clip 的失败明确归因到几何不匹配,这是过去一年 RL-on-LLM 文献里少见的"从第一性原理出发的批评"。
- **公式简洁、计算开销几乎不变**。$\epsilon_{s,a} = \sqrt{\delta / \pi_{\theta_{\text{old}}}}$ 只多了一个按 token 计算的 sqrt 和除法,在工程实现上完全可以一次前向扫完。
- **跨任务泛化扎实**。数学、代码、搜索都验证过,不是"刷榜论文"。

**要小心的几点**:
- **δ_low/δ_high 仍然要调**。论文图 2 已经说明这件事,而表 2 显示在 {0.02, 0.08} 范围里都很稳——这点比 DAPO 的"Clip-Higher 单参数"略复杂,但也不算灾难。
- **计算 $\pi_{\theta_{\text{old}}}$ 的开销**。每次 clip 都要按当前 token 重新算 old 概率,显存和计算都有一些代价——论文没给 latency 数字,实战部署时需要自己测。
- **对 MoE 模型是否还成立?** GSPO 论文特别强调过 token 级 IS 在 MoE 上会因路由抖动崩溃,RIPO 仍然是 token 级,只是边界动态化。论文没有在 MoE 上做实验,这是个开放问题。
- **"60% improvement"是 1.7B 小模型**。这个数字的"震撼力"主要来自小模型。大模型 Qwen3-8B 上 AIME24 涨 38%(43.8 vs 31.7),仍然是 SOTA 水平,但没到 60% 那么夸张。我倾向于认为 RIPO 的"理论美感"比"工程涨幅"更值钱。

**和同期工作的位置**:
- **DAPO**(字节+清华 AIR,2025.03)是"GRPO 的工程修补包",Clip-Higher 只是单参数手调。
- **GSPO**(阿里 Qwen,2025.07)是从 token 级跳到序列级,核心论点是 MoE 上的稳定性。
- **DCPO**(2025)开始做动态 clip,但用经验启发式。
- **GMPO**(2025)用几何均值改 ratio,没有改边界。
- **RIPO** 是第一个**从信息几何的角度**给出"为什么 PPO-Clip 注定会塌"的理论解释,然后推导出唯一一个符合该理论的形式。

> 说实话,过去一年 LLM RL 论文我已经看麻了——大部分是"A 加 B 加 C 加 D 涨 1-2 个点"式的缝合。但 RIPO 是少有让我重新翻回 PPO 原论文、把 Fisher 信息矩阵那一节重新看一遍的工作。这个思路一旦你接受了,回头看 DAPO 的 Clip-Higher 就会觉得:"那只是给一个流形问题打了个补丁"。

**对工程实践的启发**:
1. 如果你在跑 GRPO/DAPO 训推理模型,熵曲线塌得很快——直接换成 RIPO 的 clip 边界,基本是 zero-cost 升级。
2. 如果你在做 on-policy RLHF(传统 PPO 目标),也可以直接套用 RIPO 的裁剪——论文表 4 已经在 GSM8K 上验证过。
3. 如果你做 MoE,先在自家场景小规模 A/B 一下,别直接全量切。

---

## 收尾

LLM 强化学习这个领域,过去一年被"clip 怎么改"和"advantage 怎么估"这两条线索占满。RIPO 的价值在于把一个**早就该被问的问题**——"我们用的度量衡对吗"——重新摆到台面上。Fisher 信息矩阵在 90 年代的信息几何里就是常识,但在过去一年的 LLM RL 喧嚣里被彻底遗忘了。

如果你也在调 LLM 的 RL 训练,试试把 clip 边界改成 $\sqrt{\delta / \pi_{\text{old}}}$——成本几乎为零,理论上站得住脚,实验上也确实打得过。

剩下的开放问题:**MoE 上还成立吗?** **多轮 agent RL 的回合级 clip 怎么办?** 这些可能都是 RIPO 之后值得追的方向。

---

*觉得有启发的话,欢迎点赞、在看、转发。跟进最新 AI 前沿,关注我。*
