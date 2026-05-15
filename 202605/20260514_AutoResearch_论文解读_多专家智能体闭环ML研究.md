# 把"提想法-写代码-跑实验-改下一版"整个交给Agent：CMU这篇把ML自动研究跑成了一条可审计的轨迹

## 核心摘要

我们做ML研究的人都熟悉那条循环：改一段代码、起一个实验、看结果、决定下一步怎么动。这篇来自CMU的论文 *Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes* 把这条循环整段交给了语言Agent——人只在最开始把环境搭好、按下启动键，剩下的1197次正式trial加600次对照trial，全部由Agent自己提假设、改代码、提交实验、读评测器返回的分数和失败原因，再用这些反馈塑造下一次提案。

让我比较意外的是，三个环境同一套loop全都跑出了真东西：Parameter Golf的validation bpb从公开SOTA起点1.0810降到1.0722（-0.81%），NanoChat-D12的CORE从0.1618拉到0.2244（+38.7%），CIFAR-10 Airbench96在严格0.96准确率gate下把训练时间从26.36s压到25.15s（-4.59%）。更值钱的是它给出的不是一个"最终模型checkpoint"或者一篇"AI生成的论文"，而是一条完整可审计的trajectory——每个trial都带假设文本、代码diff、状态码、分数、崩溃日志，全部对外可审。

这是一篇典型的"组合式而非范式级"的工作。Agent没有发明新的Transformer，但是它真的在用工程师的方式做研究：把一个超过16MB cap的好idea想办法瘦下来、把attention backend的一次诊断换算成多塞的训练token、把一次0.9596的near-miss修成0.9601的keep。这件事能跑通本身就值得我们把它的架构掰开看看。

---

## 论文信息

- **标题**：Auto Research with Specialist Agents Develops Effective and Non-Trivial Training Recipes
- **作者**：Jingjie Ning, Xiaochuan Li, Ji Zeng, Hao Kang, Chenyan Xiong
- **机构**：Carnegie Mellon University, School of Computer Science
- **arXiv**：[https://arxiv.org/abs/2605.05724](https://arxiv.org/abs/2605.05724)
- **代码**：[https://github.com/cxcscmu/Auto-Research-Recipes](https://github.com/cxcscmu/Auto-Research-Recipes)

---

## 一、为什么需要这种"闭环"，而不是又一个"AI Scientist"

说实话，过去一年多"用Agent自动做ML研究"这个赛道挤进来不少东西——AI Scientist那一类生成想法和论文，MLE-bench、RE-Bench、MLGym这一类把Agent放进benchmark里打分，AlphaEvolve那类则是evaluator驱动的程序搜索。每一篇都把这件事的某一片做漂亮了。

但你真要问"现在的Agent能不能像研究员那样做ML研究"，事情就尴尬了：
- 生成假设的工作大多停在"写出来"，不真正提交GPU实验，没有**实测的失败反馈**回来打脸；
- benchmark那一类把Agent当被试，但**反馈通道是被赛道设计死的**（给一个题目让你做，做完打分），不是Agent自己挑工程方向；
- 程序搜索那条线（FunSearch、AlphaEvolve）做的是算法层面的演化，离"完整训练pipeline下的compute-budgeted recipe"还差一步。

CMU这篇的切入点其实很朴素：**让Agent干的事就是研究员每天干的事**——读一下当前best是什么，提一条假设，写代码改训练脚本，提交到真实的8×H100上跑，看评测器返回什么（包括崩溃、超时、超过artifact cap），把这些信息再喂给下一个agent去改。

整篇论文的判断我觉得是合理的：与其再搞一个"AI写论文"，不如先把这条最本质的闭环跑通——**一个不能被Agent自己改写的外部评测器、有限的submitted trial budget、能在合理时间内回收反馈的环境**。

![图1：闭环自动研究的trajectory示意。提案、可执行代码编辑、外部测量、反馈和下一步研究动作通过submitted trial连接。](https://arxiv.org/html/2605.05724v1/figures/overview_specialist_swarm_v5.png)

*图1：闭环自动研究trajectory。一个submitted trial = 一个假设 + 可执行code edit + evaluator-owned outcome + feedback signal。所有专家Agent共享同一份measured lineage，所以系统专家诊断出来的"attention backend是瓶颈"这件事，能立刻被schedule专家用来塞更多训练token。*

---

## 二、关键设计：submitted trial、specialist agent、shared lineage

理解这套方法只需要抓住三个名词，剩下都是工程实现细节。

### 2.1 Submitted trial：把"想法"逼到"可执行代码 + 外部评分"

整篇论文最关键的设定其实就一句：**研究的最小单位不是"一段思考"，而是一次submitted trial**。一个trial包含——

- 一段一句话的hypothesis；
- 一段对editable file的具体代码diff（比如`train_gpt.py`）；
- evaluator返回的status（keep / discard / crash / preflight_crash / size_blocked / train_budget_overrun / eval_budget_overrun / disqualified / harness_abort）；
- 分数、计时、崩溃日志（如果有）。

为什么这个设计这么硬？因为它把reward hacking堵死了。Agent能改训练代码，但**不能改评测器**——CIFAR用shell侧的计时sidecar，NanoChat用受保护的parser从log里抠CORE，Parameter Golf用官方评测路径。Agent就算在代码里print一个假分数也没用，评测器根本不读。

我之前看过几篇类似方向的工作，最容易翻车的就是这点：让Agent自己报分数，跑两轮以后score疯涨但生成质量肉眼可见在崩。这篇论文从一开始就把"editable recipe不能拥有evaluator"立成原则，我觉得这是它能跑通的前提。

### 2.2 Specialist agent：按recipe surface分工，不是按"前端后端"分工

这里的specialist不是软件工程意义上的"前端工程师/后端工程师"，而是按训练recipe的子表面来切：

| 任务 | 专家划分 |
|------|---------|
| Parameter Golf | 10个专家：architecture / optimization / quantization / regularization / loss / evaluation / curriculum / tokenizer / test-time training / meta search |
| NanoChat-D12 | 5个专家：architecture / optimization / data / schedule / systems |
| CIFAR-10 Airbench96 | 5个专家：architecture / optimization / augmentation / loss / regularization |

每个专家拿到的system prompt前面都拼一段**domain preamble**，明确写清"你owns什么、不owns什么、historically wins from什么级别的改动"。比如Architecture专家的preamble里直接告诉它："如果你的draft hypothesis是'change one number'，你大概率走错domain了，去找opt或meta"。这种anchoring非常有效——它不是靠modelsize去理解domain，而是靠prompt直接把边界画死。

比较有意思的是Meta-Search专家，定位是analyst：它先去读`results.tsv`和`KNOWLEDGE.md`，找一个"前面好几个trial都没动过、但应该有信号"的knob，再下小步tweak。论文里写得很直白：你这个角色"mostly read results"，不是要你脑爆idea，是要你做数据分析。

![图10：三个环境下的专家role partition和搜索行为。上面一行是固定的role分工，下面一行示意每种约束regime下产生的搜索pattern。](https://arxiv.org/html/2605.05724v1/figures/specialist_swarm_decomposition_v3.png)

*图10：Parameter Golf是宽recipe surface受artifact cap压制，所以10个专家广撒网；NanoChat-D12是固定预算的pretraining，所以5个专家走更集中；CIFAR-10是accuracy gate绑速度，所以集中在可能撞gate的几个面上。*

### 2.3 Shared lineage：闭环里最值钱的那块

如果只看上面两条，你可能觉得这就是个"prompt写得比较细的多Agent系统"。让这套架构真正活起来的是lineage——**一份所有专家共享的、append-only的trial记录，每次新session启动前都把最近活跃的几行喂回prompt里**。

具体rendering规则是：每次session start用户消息里带三块内容——

1. `LEADERBOARD.md`（当前best是哪个exp_id、什么分数）；
2. `KNOWLEDGE.md`（任务级别的静态priors）；
3. `Recent Activity`：最近10行results（包含别的专家的keep/discard/crash简报）。

这样一来，arch专家看到opt专家昨天试过Muon WSD warmdown 25%→30%是个keep，他就不会再去碰这个；data专家看到sys专家把attention backend从SSSL改成L，恢复出了大段wallclock，他就知道"啊这下我能塞更多token了"。

**这就是lineage的关键作用：它让"一次系统优化"在跨专家的搜索空间里被迅速变现成"另一个recipe方向的预算头ろ寸"。** 后面消融实验会证明，这个机制贡献了大头。

---

## 三、loop怎么打：四个level

我把方法部分整理成一张图，可能比照搬paper里的列表清楚一些：

| Level | 内容 | 谁拥有 |
|-------|------|--------|
| Task feedback | 度量、合法性检查、failure label | Environment |
| Submitted trial | hypothesis、code diff、status、score | Agent提交、Evaluator打分 |
| Shared lineage | append-only的全局记录 | Blackboard |
| Parallel iteration | N个submitter共享blackboard和GPU pool | Supervisor调度 |

并发跑这件事作者还专门给了个吞吐公式：

$$\tau_{e} = \tau^{\mathrm{run}}_{e} + \tau^{\mathrm{eval}}_{e} + \tau^{\mathrm{queue}}_{e} + \tau^{\mathrm{log}}_{e};\quad T_{e}(N) = \frac{N \cdot \eta_{\parallel,e}}{\tau_{e}};\quad \eta_{\parallel,e} = \frac{T_{e}(N)}{N \cdot T_{e}(1)} \in (0, 1]$$

说人话就是：把单个trial的wallclock拆成run/eval/queue/log四段，N个submitter共享一个blackboard，并行效率$\eta_{\parallel,e}$肯定小于1，因为大家抢GPU、抢filelock、抢cluster queue。在Parameter Golf上实测：单个generalist能跑2.26 trials/小时；10个specialist能跑18.15 trials/小时（$\eta \approx 0.80$）；10个generic agent能跑16.79 trials/小时（$\eta \approx 0.74$）。

这个数其实给我们一个挺重要的信号——**specialist swarm相对generic swarm在throughput上没有压倒性优势（18.15 vs 16.79）**，所以后面看到specialist效果更好的时候，那个差距并不是"跑了更多次"，而是真的"提案质量更高 / 边界纪律更好"。

### 3.1 安全护栏：不让Agent干蠢事

这里有几个细节值得拎出来说，因为这是我之前自己搭multi-agent系统时也踩过的坑：

- **Bash is read-only**：作者直接用SDK hook把`rm/mv/cp/sed/tar`这些destructive verb全block掉，Agent的Bash只能读，不能写。Write工具也根本没在allowed_tools里。
- **`size_project`/`syntax_check`是免费的，`submit_trial`是唯一的GPU-burning tool**：每次提交前Agent会本地跑syntax check和size projection，preflight挂掉的连GPU都不会烧。这点很关键，否则一次architecture重构跑40分钟才告诉你comma打错了，谁顶得住。
- **`block_bash_blackboard` hook**：no-lineage消融的时候不仅在prompt里去掉lineage section，连Bash读blackboard文件都直接拒掉。审计发现57.9%的Bash调用本来就在读`tree.tsv`这种东西——不堵这条路，prompt侧的消融基本是漏的。

这种"把作弊路径都堵死"的细节决定了消融结论是否可信，作者写得相当谨慎。

---

## 四、实验：三个环境，一套loop

主表如下，全部relative gain都是相对**search starting point**算的，不是相对外部的某个naive reference：

| 环境 | 起点 | 最终 | 相对变化 | Trials | 有效改进数 |
|------|------|------|---------|--------|-----------|
| Parameter Golf val_bpb（越低越好） | 1.0810（公开SOTA） | 1.0722 | **-0.81%** | 900 | 36 |
| NanoChat-D12 CORE（越高越好） | 0.1618（calibrated upstream） | 0.2244 | **+38.7%** | 200 | 5 |
| CIFAR-10 Airbench96 train_s（越低越好，要过0.96 gate） | 26.3560s | 25.1464s | **-4.59%** | 97 | 4 |

注意几个细节：
- Parameter Golf的1.0810是**OpenAI官方leaderboard的公开SOTA**，不是论文自己跑出来的baseline。能从公开SOTA再降0.81%，意义和"从random baseline降"完全不是一个量级。
- NanoChat-D12的0.1618是作者用本地8×H100对upstream recipe重新calibrate的结果。upstream自己在不同硬件上跑的数会不一样，所以重新校准是必要的——这点作者讲得很清楚，避免用stale denominator算relative gain。
- CIFAR的26.3560s是十seed cold-process aggregate，**绝对不是冷启动后第一次跑的single-shot数**——cold-process protocol保证不会被Python/CUDA cache带偏。

![图2：三个环境上best-so-far score随submitted trial index的变化。粗线是best-so-far，散点只显示valid measured trial。](https://arxiv.org/html/2605.05724v1/x1.png)

*图2：三条曲线的"楼梯式"形状是lineage反馈在起作用——每次keep落地之后，后续所有专家都从新的baseline出发。Parameter Golf花900 trial爬到1.0722，NanoChat在前几十个trial就拿到大头（attention path rewrite一次性放出大量wallclock），CIFAR则被accuracy gate反复打回票，最终是用"speed-up + warmup repair"的组合通关。*

### 4.1 三个代表性的"程序级"改动

作者特别强调loop提的不只是scalar tweak，是会真改代码结构的。我把审计出来的157个architecture-domain submission中的几个代表case放出来：

| 环境 | Trial id | 具体程序变化 |
|------|---------|-------------|
| Parameter Golf | 245, 475, 538 | Recurrent residual scaling；分开RoPE/NoPE的query gain；per-head data-dependent attention-output gate |
| NanoChat-D12 | 007 | SSSL → L attention path改写；masked SDPA层全部迁到Flash SDPA |
| CIFAR-10 Airbench96 | 040/044/053, 059/062 | Residual-preserved ConvGroup depth reduction；wider-shallower block under accuracy gate |

这些不是"调超参"。其中NanoChat的007号trial特别有意思——sys专家诊断出原recipe那个SSSL（short-short-short-long）的attention pattern在本地H100环境下是个runtime tax（masked SDPA走的是慢path），就把全部12层都改成Flash SDPA可吃的pattern。这次改动直接把wallclock从原本的cap上挤出一大块。

但你知道作者是怎么用这块wallclock的吗？**不是收割成"训得更快了"的指标，而是把省出来的时间塞更多training token进去。** 020号trial把token budget拉大，CORE一次性涨0.0334（论文里的"biggest NC jump"）。024号把data ratio调到12:100:130，025号landed在0.2241的schedule plateau，156号最后给lm_head后面加了一个zero-init的learnable logit-bias path，最终0.2244。

**这一连串移动里没有一步是"突破性"的，但每一步都是上一步的measured fact算出来的下一步。** 这才是作者说的lineage feedback的真正含义。

---

## 五、关键消融：lineage砍掉以后，loop彻底退化

这是整篇论文最值钱的一组对照。

### 5.1 四个对照组

在Parameter Golf的前200个trial窗口内做四组：

| 配置 | 最终val_bpb | 有效drop数 | 说明 |
|------|------------|----------|------|
| Role swarm + lineage（主方法） | 1.073142 | 16 | 10个specialist + 完整lineage |
| Generic-10（10个相同prompt的Agent） | 1.074495 | 10 | 砍掉role专精，只保留并发和lineage |
| Single generalist | 1.075384 | 14 | 单Agent + 完整lineage |
| Role swarm, NO lineage | 1.077413（exp_075卡住） | **3** | specialist还在，lineage没了 |

直接看最后一行——把lineage砍掉以后，**loop跑125个trial没产生任何新的improvement**。3个有效drop里两个还是头20个trial里被瞎蒙到的。eval-budget cap命中率从有lineage的19.0%飙到无lineage的61.5%，因为没有"current best已经贴着600s eval cap"这个动态信息，agent会一直试着提交吃满预算的方案。

而single generalist那一行也很值得看——14个drop比generic-10的10个drop还多，但最大的cluster占了200个submission中的35个（其中32个是polar-coefficient附近的preflight crash）。说人话：单Agent会陷入anchoring，反复试同一个改动方向。Specialist swarm相比之下，最大cluster只占7/200。

![图3：Parameter Golf前200个trial的对照实验。Y轴是delta validation bpb（越低越好）。Panel A对比不同agent organization，Panel B单独消融shared lineage。](https://arxiv.org/html/2605.05724v1/x2.png)

*图3：Panel B那条对照特别明显——有lineage（深色）能持续往下走，无lineage（浅色）在exp_075附近卡住，剩下的trial基本在原地踏步。这说明lineage不是"锦上添花"，而是闭环能持续学习的核心反馈通路。*

### 5.2 提案多样性审计

作者还干了件挺细的事——把每个trial的hypothesis文本拿出来，TF-IDF embed以后做online clustering（cosine ≥ 0.30并入最近centroid，否则开新cluster），再用Shannon熵算effective cluster数 $\exp(H)$。

挑几个数：

| Trace | 行数 | Effective clusters | 最大cluster占比 | Near-dup率 |
|-------|-----|-------------------|---------------|----------|
| Role swarm + lineage（200行） | 200 | 134.8 | 3.5% | 2.0% |
| Role swarm, no lineage（200行） | 200 | 121.7 | 2.5% | 2.0% |
| Generic 10-agent（200行） | 200 | 41.1 | 12.0% | 1.5% |
| Single generalist（200行） | 200 | 61.9 | 17.5% | 10.1% |
| Role swarm full run（900行） | 900 | 439.6 | 2.2% | 1.1% |

我看到generic-10那行才41.1个effective cluster的时候是真的有点意外——10个并发的Agent，因为prompt完全一样，提案流居然集中到只有41个有效cluster。这从反方向证明了**role partition不是个噱头**，而是真的把搜索空间撑开了。

但更关键的是cross-context keep rate这一列（论文Table 3里）：role swarm + lineage有10/12 (83.3%)的keep的parent来自其他role；role swarm no lineage是0/1。意思是**lineage开启时，专家之间真的在搭对方的台**，关掉就只剩自己玩自己的。

---

## 六、跑通三个case，每个case都在演同一件事

读到这里我才意识到，三个环境其实在demo一件相同的事——**外部测量到的边界失败，怎么经过feedback变成下一次有效编辑**。

### 6.1 Parameter Golf：超过artifact cap的好idea怎么救活

exp_587号trial提了TTT-only z-loss，val_bpb做到1.072431（比当时的best好），但artifact pack出来超过16MB cap 2056字节，被`size_blocked`挡住。

这种事在传统grid search里就直接丢了——score再好也没用，cap过不了。但lineage把"超过2056字节"这个具体数字带回了下一次proposal。exp_596换了同一套z-loss机制但recovery了byte headroom，artifact掉到15,995,930字节，score留在1.072251，变成legal的keep。后续746号、750号继续在这条线上refine，最终到1.072210。

![图12：Parameter Golf最终recipe的schematic。inherited部分（深色）是公开starting recipe的结构，rewritten部分（teal）是搜索过程中真正改写的component，包括score-first evaluation-time adaptation、TTT-only z-loss、separate RoPE/NoPE query gain、attention-output gate、GPTQ Hessian calibration等。](https://arxiv.org/html/2605.05724v1/figures/plot_J2_Parameter_Golf.png)

*图12：注意这张图标了"post hoc Claude Design schematic"——也就是搜索结束以后另外做的一张说明图，不是loop本身的产物。但teal部分确实是loop搜出来的代码改动，绝大多数都不是改超参那种小动作。*

### 6.2 NanoChat-D12：把"系统瓶颈"转化成"训练token"

前面已经讲了这条trace。再补一个细节：007号trial的诊断是怎么发生的？sys专家拿到Recent Activity时看到前几个trial都跑得很慢，他用Bash读训练log（在lineage开启的run里这是允许的），定位到attention backend走的是慢path，然后改成uniform L pattern。

这次trial的score本身没创纪录，但它通过lineage把"runtime headroom大概多少"这个信号曝光出去。然后data专家020号说"那我塞更多token"，CORE跳到0.1952（大概+0.0334）。024号把data ratio调到12:100:130，025号landed schedule plateau，156号最后给lm_head后面加zero-init的logit-bias path（一个非常小巧的改动，本质是给vocabulary加个learnable prior），到0.2244封顶。

![图13：NanoChat-D12最终recipe的schematic。SSSL→L attention path rewrite、data stage ratio expansion、zero-init logit-bias path，以及CORE从0.1618爬到0.2244的trajectory。](https://arxiv.org/html/2605.05724v1/figures/plot_J3_NanoChat.png)

*图13：这张图最值得关注的是中间那个"recovered wallclock → more training tokens"的反馈箭头——这是闭环里最具说服力的"反馈成为下一步edit"的实例。如果没有lineage，sys专家做完007号以后，data专家根本不知道时间空出来了。*

### 6.3 CIFAR-10 Airbench96：让accuracy gate的near-miss成为研究信号

这条case的妙处在于"失败"本身怎么用。

CIFAR的设定是：训练时间越短越好，但mean accuracy不到0.96直接`disqualified`，速度再快也是0分。exp_060号trial拿到了惊艳的25.1650s，但accuracy只有0.95956——差了0.00044。

传统setup里这就是个失败case，最多记一笔"太激进了下次保守点"。但这里的evaluator返回的是**精确的timing + 精确的accuracy deficit**，这个near-miss在lineage里就成了一份非常有用的信号：speed recipe结构本身没问题，差的就是accuracy margin。exp_070号保留了42 epoch、lr=11、跳过中间validation的整套speed recipe，**只把warmup从10%降到5%**，让schedule更早达到peak、训练主体段有更长的高lr窗口。结果：25.1464s @ 0.96008 accuracy，干净通关。

![图14：CIFAR-10 Airbench96最终recipe的schematic。schedule rewrite、四个code-level edit、严格的gate enforcement，以及"近miss → warmup repair"的闭环修复轨迹。](https://arxiv.org/html/2605.05724v1/figures/plot_J4_CIFAR.png)

*图14：右下角那个橙色的"near-miss feedback loop"特别能说明问题——一个被`disqualified`的trial不是垃圾，它返回的"差了多少accuracy"才是下一步该怎么改的最具体提示。*

---

## 七、一些专家级别的trace统计

![图11：三个环境下的专家outcome profile。每个堆叠柱按该专家的submitted trial归一化。绿色是有效改进，灰色是有效但非改进，橙色是不合规trial。柱顶数字是有效改进数。](https://arxiv.org/html/2605.05724v1/x3.png)

*图11：CIFAR的橙色比例最大（97个trial有81个miss了accuracy gate），完全符合"gate-dominated task"的画像。NanoChat的systems专家贡献了3个有效改进——和我们前面讲的SSSL→L attention rewrite完全对得上。Parameter Golf比较"散"，10个专家全都贡献过有效改进，没有任何一个专家垄断。*

按任务看专家分布：

- **Parameter Golf**："broad search"。10个专家都贡献过valid improvement，opt/TTT/quant各5个。最常撞到的边界是size gate，主要来自architecture/quantization/tokenizer。
- **NanoChat-D12**："concentrated"。systems专家贡献了3个有效改进，schedule和arch各1个。其他专家大多只产valid non-improvement。
- **CIFAR-10**："gate-dominated"。优化贡献2个，augmentation和regularization各1个。81/97的trial没过accuracy gate，accuracy gate就是这条任务的主要feedback来源。

我看到NanoChat是systems专家主导的时候有点意外——直觉上pretraining任务应该是data/arch专家主场。但回头一想，**当starting recipe是一个相对成熟的upstream code时，最大的提升空间其实是隐藏在implementation里的**。这跟我之前在做某个baseline优化时的经验是吻合的——纯靠改架构很难再涨，但把kernel换一遍马上多出一大块预算。

---

## 八、一些值得拎出来的细节和我的判断

### 8.1 Anti-anchoring + crash feedback：让Agent别在同一个坑里反复栽

每个specialist都有一份banlist——之前在自己role里被试过、要么失败要么noise内的pattern。下次新session启动时banlist会和lineage slice一起render进prompt。

crash反馈也走同一个通道：当一次trial崩溃，下次prompt里会带最深一层exception line + 训练脚本最深一帧。我觉得这个设计相比"只给一个crash status"是真有用的——LLM其实非常擅长从一段stack trace里推断哪段代码可疑，但如果你只告诉它"this trial crashed"，它能做的就只剩重试或者随机换个方向。

### 8.2 Compute accounting：诚实标注的upper bound

作者特意算了下upper bound：1500个Parameter Golf submitted trial（900 headline + 600 control），每个最多600s train + 600s eval × 8块H100 → 4000 H100-hours。NanoChat-D12 200个trial，每个最多90分钟 × 8块H100 → 2400 H100-hours。CIFAR小到可以忽略（<10 single-GPU-hours）。

这个数其实不算夸张——业界一次正经的pre-training run就能轻易吃掉几万H100-hour。但作者诚实标注"这是upper bound，不是summed per-job telemetry"，preflight挂掉的trial其实没烧到这么多。这种坦率是研究态度的好信号。

### 8.3 我对这篇论文的判断

**亮点**：
1. **闭环本身被设计成可审计的artifact**——releasable的不是model checkpoint，是整条trajectory（results.tsv、tree.tsv、best.json、code snapshot）。这件事的研究价值反而比"提了多少分"更长远。
2. **Lineage消融的对照非常硬**——3 vs 16 drop的差距，配合eval-cap命中率61.5% vs 19.0%的对照，没法用噪声解释。
3. **Specialist设计的颗粒度真的合理**——不是把"做研究"切成"前端/后端"这种nonsense，而是按recipe surface（architecture / optimizer / data / loss / quantization ...）切，每个专家的edit radius在preamble里被画死。

**问题和局限**：
1. **没有paradigm-level invention**。作者自己也承认了：观察到的边界是compositional——Agent能combine、transfer、repair已知技术，但不会发明新的Transformer。这个结论我同意，看完所有case study我没看到任何一个改动是"完全没人想过的"。
2. **三个任务都是compute-budgeted的"recipe优化"**，这是个非常特殊的setting。把这套loop搬到长horizon、需要新architecture invention的研究上能不能work，是开放问题。
3. **lineage的"信息传递"和"作弊倾向"之间有tension**。Agent能看到别人之前的keep和crash，理论上有可能学会"挑容易的knob刷"。论文里通过anti-anchoring banlist + role partition在控这件事，但这是个值得长期关注的方向。
4. **GPU hours成本不低**。4000 H100-hours换来Parameter Golf 0.81%的相对提升，这个性价比放在工业界并不一定划算。但作为研究证明，意义在于"loop能跑通"，不在于"loop更便宜"。

### 8.4 对工程实践的启发

如果你也在搭multi-agent的实验自动化系统，我觉得这篇至少有几个可直接复用的pattern：

- **第一层就把evaluator和editable code隔离**，不让Agent能print假分数。这是reward hacking的最基础防线。
- **prompt里塞current best + recent activity足够撑起90%的lineage价值**。Anthropic SDK的prompt cache刚好能把固定的system prompt部分cache住，不用每次都重算。
- **专家分工要按"editable surface"切，而不是按职级切**。Architecture专家、Optimizer专家、Systems专家这种划分自然能形成boundary discipline，比"manager Agent + worker Agent"的层级更工程化。
- **preflight check是免费的GPU救星**。`syntax_check`和`size_project`这种本地工具加上去之后，preflight crash不会烧GPU时间。这个看似工程小事，对长跑的throughput影响其实非常大。
- **failure feedback要带具体数字**——near-miss的0.95956是有用的，而"failed accuracy gate"是没用的。设计evaluator的时候多想一步"这个失败信号下个agent能拿来干什么"。

---

## 九、一个更本质的追问

读完这篇我有个一直在想的问题——**当lineage变得越来越长（比如跑10000个trial），prompt rendering怎么办？**

论文里现在的rendering规则是"current best + 最近10行 + 静态knowledge file"。900 trial的run里这个prompt窗口够用，但如果是10000 trial呢？挑选lineage slice的策略本身可能就是下一篇论文的题目。作者在future work里提到了这个方向，但没展开。

另一个我自己更关心的问题是：**这套loop能不能跑出真正的"反直觉发现"？** 现在所有的case study都是combine/transfer/repair——专家把已知技术按外部反馈组装起来。但人类研究员偶尔会有那种"等等，loss应该不收敛才对，为什么work了？"的时刻，然后顺着这个想法挖出一个新机制。当前这套loop里我没看到对"违反预期"的奖励——Agent的目标函数还是"分数往好的方向挪"。

不过我自己也在怀疑——也许这种"反直觉发现"对当前一代LLM Agent来说还是太奢侈了。先把"组合式自动研究"跑通、跑稳、跑便宜，再去想范式级的事，这个节奏其实挺合理的。

---

## 十、总结

这篇论文最值钱的地方不在于那0.81%、38.7%、4.59%的具体数字，而在于它把**"自动研究"这件事从一次性的generated artifact变成了continuously measured object**。

每个trial都是executable code edit，每个分数都来自evaluator-owned measurement，每个失败都被lineage记下来变成下一步的输入。整个搜索过程跑完以后，留下的不是一个最终模型，是一份能被reviewer逐行审计的研究trace。

我觉得这种设计哲学比任何一个具体的数据点都重要。它告诉我们一种可能性——agentic ML research可以不是"AI写论文"那种不可证伪的东西，而是和人类研究员的proceeding一样有据可查的工程过程。

就像作者在discussion里说的："The same feedback loop can make empirical research more scalable, inspectable, and powerful as models become more capable."

值不值得花时间细看？我的判断是：如果你在做multi-agent系统、AutoML、或者任何形式的"让LLM跑实验"的工作，强烈建议读完整篇并去clone他们的代码。如果你关心的是LLM能不能做"真研究"这种偏哲学的问题，至少把第3章和第4.3节的lineage消融读一遍——这是当下能给到的最实在的answer。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
