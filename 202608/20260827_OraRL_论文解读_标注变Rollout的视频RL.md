# 把标准答案直接塞进 Rollout 组：OraRL 用 2.2 倍 SFT 的成本，把视频大模型 RL 训到了新高度

做过多模态 RL 后训练的人，应该都体会过这种肉疼：一个 prompt 要采样 8 条甚至 16 条 CoT rollout，每条 rollout 都要让模型吭哧吭哧生成几百个 token 的推理链，然后 reward 一打，发现一大半 rollout 是废的——要么全对要么全错，组内方差为零，梯度直接消失。算力烧了，信号没拿到多少。

更气人的是，训练数据里明明躺着标准答案（时间区间、框坐标、分割点、选项字母），它们从头到尾只干一件事：给 rollout 打分。答案本身那么好的监督信号，就这么浪费了。

arXiv 2608.20492 这篇论文干的事，一句话讲完：**把标注本身序列化成一条 oracle rollout，直接塞进 on-policy 组里参与策略更新**。想法朴素到不行，但直接做会翻车——他们管这个坑叫 advantage inversion，后文细讲。解决之后，这套叫 OraRL 的方法只要 SFT **2.2 倍**的步时间（GRPO+CoT 要 4.9 倍），训出的 Video-ORA-9B 在时序定位、跟踪、分割、空间智能上全面刷新：VSI-Bench 拿到 **73.1 分**，GPT-5 是 55.0，Gemini-3-Pro 是 55.1。

我的判断先放在这：这不是什么花里胡哨的新 RL 算法，而是一个"把已有监督信号用到极致"的漂亮工程设计，而且实验做得相当扎实，控制变量做得比大多数 RL 论文都干净。值得细读。

## 核心摘要

视频 MLLM 的 RL 后训练有个结构性浪费：on-policy 采样又贵又低效，标注却只用来算 reward。OraRL 把标注追加为组内第 n+1 条 oracle rollout，但直接混进 GRPO 的组归一化会让高 reward 的 oracle 抬高 baseline，把本来优于平均的 rollout 的 advantage 翻成负数（实测 22.4% 的 rollout 被翻转）。解法是解耦优势估计：baseline 只用 on-policy 奖励算，oracle 通过 directional gain 和一个 detached、有上界的 advantage 单独起作用，再配符号均衡剪枝把 9 条 rollout 砍到 4 条进更新。结果是 0.8B 到 9B 全尺度超过 backbone 和 GRPO，训练成本只有 GRPO+CoT 的一半不到，推理还不用 CoT，解码延迟 130 ms 对 4780 ms。

**论文信息**

- 标题：Annotations as Rollouts: Efficient and Scalable Reinforcement Learning for Video MLLMs
- 作者：Yunheng Li, Guohong Mu, Hao Li, Shengsheng Qian, Dingwen Zhang, Qibin Hou, Ming-Ming Cheng
- arXiv：https://arxiv.org/abs/2608.20492 （2026 年 8 月 20 日提交）
- 项目页：https://orarl.github.io/

---

## 🎯 问题：标注只配当裁判吗？

先把范式摆清楚。视频 MLLM 后训练现在主流是 GRPO 一类的方法：每个 query 采样 n 条 on-policy rollout，用组内均值当 baseline 算相对 advantage，不需要 critic。标注 y 的唯一作用是算 $r_i = R(o_i, q)$。

作者把"标注在模型适配里的角色"整理成了四个范式，这张图信息密度很高：

![图2：标注使用的四种范式](https://www.mulanai.com/fs/files/0827_44da8798_S1F2.png)

*图2：四种利用标注的范式对比。(a) SFT：直接复制标注，没有探索；(b) GRPO：标注只打分，低 reward 组没有锚点；(c) 混合策略归一化：把 GT 混进组里一起算 baseline，oracle 会把符号搞乱；(d) OraRL：GT 保留为优化目标但不进 baseline，正负符号保持均衡。*

SFT 是纯模仿，不探索；GRPO 探索了但组里全是"考生"没有"标准答案"做参照；直接把答案混进组（mixed-policy）看似两全，实则埋雷。这个雷就是全文的技术起点。

## ⚠️ Advantage Inversion：好心办坏事的数学

直觉上，把一条满分 rollout 放进组里，应该是"带着大家往高处走"。但 GRPO 的 advantage 是相对组均值算的——你放个满分选手进来，均值被抬高了。

具体算一下。设 on-policy 均值为 $\mu_{\mathrm{op}}$，oracle reward 为 $r_{\mathrm{gt}}$，混合 n+1 条后的组均值变成：

$$\mu_{\mathrm{aug}} = \frac{n\mu_{\mathrm{op}} + r_{\mathrm{gt}}}{n+1} = \mu_{\mathrm{op}} + \frac{r_{\mathrm{gt}} - \mu_{\mathrm{op}}}{n+1}$$

问题来了：所有满足 $\mu_{\mathrm{op}} \lt r_i \lt \mu_{\mathrm{aug}}$ 的 rollout，明明比当前策略的平均水平好，却拿到了负 advantage，被往下压。oracle 越强（$r_{\mathrm{gt}}$ 比 $\mu_{\mathrm{op}}$ 高得越多），这个翻转区间越宽，宽度正好是 $(r_{\mathrm{gt}}-\mu_{\mathrm{op}})/(n+1)$，随 oracle-policy 差距线性增长。

还有第二重伤害：oracle 会撑大组内标准差，归一化之后所有 on-policy advantage 的幅度被压缩，梯度贡献进一步被稀释。

这不是理论洁癖。作者在 11,503 个组、92,024 条 rollout 上做了统计：

![图6：Advantage inversion 散点图](https://www.mulanai.com/fs/files/0827_e76e6633_advantag.png)

*图6：横轴是标准 GRPO 下的 advantage，纵轴分别是朴素混合（左）和 OraRL（右）下的 advantage，红色阴影区是被翻转的 rollout。朴素混合翻转率 22.4%，OraRL 只有 1.9%。*

数字更扎心：朴素混合下 **42.5%** 的组至少有一条 rollout 被翻转，**8.3%** 的组直接失去全部正 rollout——时序定位任务因为 reward 稀疏，这个比例高达 17.5%。你想想看，一组采样里好不容易出的几条好答案，全被"标准答案"亲手打下去了。这就是 Table 13 里朴素注入把 GRPO 从 60.3 拉到 55.4 的原因，tracking 单项暴跌 11.9 分。

顺带一提，作者之前的工作 Tempsamp-R1 是用任务特定的 reward shaping 来缓解这个问题，能把翻转率压到 11.9%，但每个任务都得手工设计 shaping 函数。OraRL 的野心是用一套规则通吃。

## 🏗️ OraRL：解耦优势估计 + 符号均衡剪枝

![图3：OraRL 框架总览](https://www.mulanai.com/fs/files/0827_ee212d0b_framewor.png)

*图3：OraRL 总览。标注经 Annotation Oracle 序列化后追加为第 n+1 条 rollout；on-policy 奖励单独定 baseline；Policy-Oracle Gap 一路做成 directional gain 强化正样本，一路做成 detached anchor 给 oracle 分配 advantage；最后 Balanced Select 选出正负均衡的子集进入更新。*

整套设计拆开是四块。

**其一，oracle 追加而非替换。** 标注 y 经任务适配器序列化成模型响应格式 $o_{\mathrm{gt}} = T_{\mathrm{task}}(y)$——时序任务序列化成时间区间，跟踪序列化成框轨迹，Video QA 序列化成答案选项——追加到 n 条 on-policy rollout 后面，组大小 n+1。注意是追加不是替换，探索一条没少。

**其二，baseline 把 oracle 排除在外。** on-policy 部分的 advantage 就是：

$$A_i^{(0)} = r_i - \mu_{\mathrm{op}}$$

没有方差归一化，构造上翻转区间为空——比均值高就是正，不可能被谁翻成负。说实话，这个处理比想象中激进，连 GRPO 的标准差归一化都扔了，离散度只留给后面的 gain 用。

**其三，oracle-policy 差距拆成两个独立通道。**

一路是 directional gain，只放大正样本：

$$g_q = \mathrm{clip}\left[\left(\frac{\sigma_{\mathrm{aug}}}{\sigma_{\mathrm{op}}+\epsilon}\right)^{1/4},\, 1,\, 4\right], \qquad U_i = \begin{cases} g_q A_i^{(0)}, & A_i^{(0)} \gt 0 \\ A_i^{(0)}, & A_i^{(0)} \leq 0 \end{cases}$$

oracle 偏离 on-policy 分布越远（$\sigma_{\mathrm{aug}}$ 越大），说明策略离正确答案还差得远，就把正 rollout 的信号放大，最高 4 倍。放大后再做一次 re-centering 恢复零均值。这个设计挺精巧的：差距大的时候重学好的，差距小的时候不折腾。

另一路是 detached oracle advantage，给 oracle 自己分配多大权重：

$$w_q = \left[\mathrm{clip}\left(\frac{r_{\mathrm{gt}} - \mu_{\mathrm{op}}}{r_{\mathrm{gt}}+\epsilon},\, 0,\, 1\right)\right]^2, \qquad A_{\mathrm{gt}} = \min\left(2w_q,\; \mathrm{clip}(1.2\,A^+_{\max},\, 0.05,\, 1)\right)$$

$w_q$ 度量"策略离满分还差多少"，平方让衰减更陡——策略快追上了，oracle 权重自动趋零。$A_{\mathrm{gt}}$ 又被最强正 on-policy advantage 的 1.2 倍封顶，防止 oracle 一家独大主导整个梯度。全组一条正 rollout 都没有的时候，给个 0.05 的小 bootstrap 信号兜底。这两个通道都响应同一个 gap，所以作者特意不让 $g_q$ 作用在 $A_{\mathrm{gt}}$ 上，避免重复计数——这种细节能看出是真调过的。

**其四，符号均衡剪枝（sign-balanced pruning）。** 9 条 rollout 全进更新太贵，那就砍：oracle 永远保留，剩余名额在正、负 rollout 之间尽量均分，各符号内部按 advantage 幅度排序取 top。$n=8$、$\kappa=0.5$ 时保留 4 条：oracle + 1 条最强正样本 + 2 条最强负样本。

剪枝会引入偏置（永远留正锚点、偏好大幅度样本），所以补一步 moment correction：先恢复零均值；如果中心化把 oracle 压成了负的（惩罚已知正确答案就荒谬了），就把 oracle advantage 置零、偏移量均摊给其他 rollout；最后对齐剪枝前的 RMS 尺度，缩放系数 clip 在 $[0.25, 1]$，只缩不放。

这块的工程收益看 Table 16 就很清楚：

| 剪枝率 κ | 保留条数 | 步时间 (s) | 峰值显存 (GB) | 平均分 | 每丢 1 分省的时间 (s/pt) |
|---|---|---|---|---|---|
| 0 | 8 | 92.5 | 62.4 | 63.1 | – |
| 0.25 | 6 | 75.0 | 60.0 | 62.8 | 58.3 |
| **0.50** | **4** | **62.4** | **50.9** | **62.7** | **75.3** |
| 0.75 | 2 | 45.0 | 47.5 | 60.2 | 16.4 |

*表：符号均衡剪枝的精度-效率权衡（n=8）。κ=0.5 是甜点位：加速 1.48 倍，平均只丢 0.4 分。κ=0.75 只剩 oracle + 1 条 policy rollout，符号对比没了，分数掉到 60.2。*

消融还验证了两个设计都不是摆设：剪掉 moment correction 丢 1.1 分；把符号均衡换成纯幅度选择（CPPO 那种），temporal 掉 1.8 分——只留一个符号，所有梯度把模型往同一个方向推，对比信号没了。

## 📊 实验：全尺度、全任务赢一遍

先看招牌结果。Video-ORA-9B 对比各任务此前最强模型：

| 任务 | 指标 | 此前最佳 | Video-ORA-9B |
|---|---|---|---|
| 时序定位 | mIoU（三基准宏观） | 62.5（TimeLens2-8B） | **66.0** |
| 视觉跟踪 | GOT-10k AO | 73.0（OneThinker-8B） | **78.2** |
| 分割 | 宏观平均 | 64.3 | **70.4** |
| 空间智能 | 三基准宏观 | 51.0 | **56.1** |

时序定位细项上，Video-ORA-9B 在 Charades/ActivityNet/QVHighlights 的 mIoU 分别拿到 61.8 / 63.6 / 72.5，比专门做时序的 TimeLens2-8B 高 2.3 到 5.0 分，也全部压过 Gemini-2.5-Pro。跟踪上 AO 78.2 对 OneThinker-8B 的 73.0，而且优势随 IoU 阈值变严还在扩大（R@0.7 高 6.5 分），说明框是真的准，不是碰运气。

最唬人的是 VSI-Bench 空间智能：**73.1 分**，GPT-5 是 55.0，Gemini-3-Pro 是 55.1，Kimi-K2.5 是 54.5。一个 9B 开源模型在这个榜上领先闭源旗舰 18 分。

但我得泼点冷水，有两个地方要看清楚。一是这个 73.1 里 backbone 的功劳不小——Qwen3.5-4B 自己就有 59.7，Qwen3-VL-8B 是 57.9，OraRL 是站在一个本来就很强的基础上再拉 13 分，而不是把废柴点石成金。二是 Route Planning 子项上它（47.4）还是输给 Gemini-3-Pro（61.9）和 GPT-5（50.2），MMSI-Bench + MindCube 的平均（47.7）也仍略低于 Grok-4（50.7）和 GPT-5（49.1）。空间智能这块不能说全面超越，是"VSI-Bench 这一项赢很大"。

视频分割是提升最夸张的任务：MeViS 的 J&F 从 backbone 的 32.1 拉到 61.3，ReasonVOS 从 21.5 拉到 63.7。原因有点黑色幽默——backbone 压根不会输出可用的 mask prompt，相当于从零教会。Video QA 那边也稳，VideoHolmes 涨 15.2 分、VideoMME 76.7，七个基准里五个开源第一。

再看扩展性，这是我觉得全文最值钱的两张图：

![图4：模型规模扩展](https://www.mulanai.com/fs/files/0827_d052d737_S4F4.png)

*图4：0.8B 到 9B 的模型扩展曲线。绿色实线是 Video-ORA，灰色虚线是对应的 Qwen3.5 backbone，八个任务族全部随规模稳定上升且每个点都高于 backbone。*

![图5：数据扩展与 reward 动态](https://www.mulanai.com/fs/files/0827_fceaee26_S4F5.png)

*图5：(a) 视频感知聚合分随训练 prompt 数（6.4k 到 100k）的变化：OraRL 一路涨到 70.35，GRPO 到 66.3 趋于平缓，SFT 反而越训越差。(b) 空间智能聚合分同样趋势，OraRL 56.14 对 GRPO 53.5、SFT 51.8。(c) 训练全程的 reward 曲线，OraRL 持续上行，GRPO 早早进入平台期。*

两个观察。SFT 曲线往下掉这个现象很有意思——数据加到 100k，continued SFT 不升反降，说明单纯模仿在数据规模化时会饱和甚至退化，RL 的探索信号是真有增量。OraRL 和 GRPO 的差距（100k 时大约 3 到 4 分）从头到尾没有收窄，说明 oracle 锚点带来的不是一次性提升，而是持续更高的样本效率。这种"曲线一直分开"的证据比单点对比有说服力得多。

## 🔬 消融：增益到底从哪来

RL 论文最容易注水的地方就是"我的 estimator 比你的 estimator 好"，所以 Table 12 这组对照我很在意——同一个 SFT 初始化、同数据、同预算，把主流优势估计器全拉来打：

| 方法 | Temporal | Tracking | Video Seg. | 平均 |
|---|---|---|---|---|
| Continued SFT | 55.2 | 63.1 | 57.7 | 58.7 |
| GRPO | 58.1 | 63.9 | 58.9 | 60.3 |
| Dr. GRPO | 58.3 | 65.1 | 58.6 | 60.7 |
| GDPO | 58.0 | 64.4 | 58.9 | 60.4 |
| CPPO | 58.5 | 64.2 | 58.8 | 60.5 |
| LUFFY-style | 57.0 | 50.0 | 57.2 | 54.7 |
| **OraRL** | **60.5** | **67.1** | **60.6** | **62.7** |

*表12：训练范式对比。四个 on-policy 变体之间差距不超过 0.4 分，OraRL 比最强的 Dr. GRPO 高 2.0 分。*

这张表其实说了两件事。GRPO / Dr. GRPO / GDPO / CPPO 挤在 60.3 到 60.7 之间——优势估计器的微调在这个场景下基本是噪音级别，大家别再为这点差异卷了。LUFFY-style（把外部轨迹混进 on-policy 组的做法）反而掉到 54.7，tracking 崩到 50.0，等于从反面又验证了一次 advantage inversion 的杀伤力。OraRL 的 62.7 和它们之间隔着 2 分，这个增益只能归因于 oracle 监督本身。对照实验做到这个份上，结论就比较硬了。

backbone 泛化也测了：Qwen3-VL-8B 上 OraRL 比 GRPO 高 2.6 分（64.0 对 61.4），Qwen3.5-9B 上高 1.4 分，同一协议直接用、不重新调参。

## ⚡ 效率：不用 CoT 的底气从哪来

很多人看到"视频 RL 不用 CoT"会本能怀疑：不推理能行吗？Table 11 的回答挺反直觉的——CoT 在这个 setting 下是负资产。加 CoT 先把 backbone 平均拉低 3.9 分（51.4 对 55.3），GRPO+CoT 每步 135.6 s 比 answer-only 的 93.9 s 贵 44.4%，分数却一分没多拿（58.5 对 58.7）。可能我的理解有偏差，但我的读法是：对于定位、跟踪、分割这类输出高度结构化的任务，推理链带来的格式噪声和 reward 稀疏，抵不上它提供的推理收益。CoT 不是免费的，它是拿训练效率和格式稳定性去赌推理深度。

![图7：性能-成本散点](https://www.mulanai.com/fs/files/0827_b245e6af_S4F7.png)

*图7：训练成本（横轴，每步秒数）对时序定位 mIoU（纵轴）。OraRL 在左上角：61.4 分、62.4 s/step；GRPO+CoT 在右下角：58.5 分、135.6 s/step。*

落到部署侧，收益更直接。10 分钟视频、2 fps、约 120K token 输入，单张 H20 上 Video-ORA-9B 和 backbone 的 TTFT 几乎一样（24.17 s 对 24.25 s），但生成 token 中位数是 13.5 个对 808.5 个——一个直接吐答案，一个要写八百个 token 的推理过程。TTFT 之后的解码是 0.13 s 对 4.78 s，P90 总延迟 25.15 s 对 62.67 s。对视频理解服务来说，这个延迟差异就是能不能上生产线的区别。

## 🤔 我的判断

这篇论文最打动我的不是某个单点创新，而是它对"监督信号经济学"的重新审视。RL 后训练的标注利用率低这件事，大家不是不知道，但主流解法要么是把答案蒸成 CoT 数据做 SFT，要么是搞 reward shaping，都在外围打转。OraRL 的选择是把答案直接放进优化目标，然后老老实实把因此引入的分布问题一个个解决掉——inversion 用解耦 baseline 解决，幅度失衡用 gain 和 cap 解决，成本用剪枝解决。每一步都不性感，但组合起来严丝合缝。

问题也不是没有。$g_q$、$w_q$、$A_{\mathrm{gt}}$、$\lambda_q$ 这一串 clip 边界（1 到 4、1.2 倍、0.05、0.25 到 1）一看就是调出来的，论文没给这些超参的敏感性分析，换个任务家族好不好迁移要打个问号。VSI-Bench 的 18 分领先有 backbone 红利，宣传口径上需要打折。还有，方法高度依赖"标注能序列化成模型响应格式"这个前提——视频感知任务天然满足，但换成开放式生成任务，oracle rollout 怎么构造就不显然了。

工程上的启发倒是立即可用：如果你在做任何有结构化标注的 RL 后训练（检测、定位、 grounding、选项式 QA），第一，检查你的组里有没有 advantage inversion 这类信号污染，统计一下翻转率，成本很低；第二，别再迷信 CoT，结构化输出任务上它可能纯亏；第三，外部指导信号混进 on-policy 组的时候，baseline 一定要解耦，这是这篇论文用 22.4% 翻转率换来的教训。

往大了说，随着标注数据形态越来越丰富，"标注即 rollout"这个思路大概率会被更多人捡起来。它把 SFT 和 RL 的边界模糊掉了——答案既是模仿对象，又是组内锚点，还顺手当了难度自适应的课程信号。一个信号三吃，这种抠门的美学，我很欣赏。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
