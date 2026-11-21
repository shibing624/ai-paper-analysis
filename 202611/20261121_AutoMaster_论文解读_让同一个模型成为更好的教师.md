# 老师还是那个老师，学生却多考了 9 分——AutoMaster 把「怎么教」变成了可以调的参数

你有没有碰到过这种事：从教师模型那儿筛出一批「做对了」的轨迹，拿去做 SFT，分数确实涨了，但涨得让人心里发虚，好像这个老师明明还能教得更多？

我之前调一个 agent 的 SFT 数据就卡在这儿。同一批任务、同一个 teacher，采样 512 条成功轨迹训完学生，Terminal 类任务从 25.47% 挪到 26.59%——1.12 个点，几乎在误差棒里打转。当时的第一反应是数据不够，或者该上 RL 了。

这篇论文给的答案不太一样：问题既不在学生，也不在数据量，而在**老师做对题时顺手带出来的那些小动作**。

---

## 🎯 核心摘要

一篇来自中国人民大学高瓴人工智能学院的工作，切的是 agent 后训练里一个被默认跳过的环节：大家都按「任务成功」筛轨迹，可一条成功的轨迹里可能塞满了没根据的假设、无意义的重复、走过场的验证，学生照单全收。作者提出 Skill-FT，把几段自然语言写的「技能」追加到 teacher 的 system prompt 里，先改老师的行为，再用改出来的轨迹训学生；外面再套一层 AutoMaster，让一个 LLM agent 自己分析轨迹、设计技能、挑数据、看评测反馈迭代。teacher、任务池、训练条数、SFT 配方全部锁死，只改「怎么生成和挑选示范」，在 Terminal、Search、SWE 三个域、4B 和 9B 两个尺寸上，六个 setting 的平均分比 Vanilla SFT 高出 1.42 到 8.99 个点。

我的判断：这不是底层突破，是一次很干净的「数据生成侧」工程整合。真正让我觉得值钱的甚至不是那 8.99 个点，而是后半段那个反直觉发现——**训练时把技能指令留在输入里，评测时不给，效果反而更好**。另外，4B 上有两组的置信区间是跨零的，读到宣传口径时心里得有数。

---

## 📖 论文信息

- 标题：*AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher*
- arXiv：2611.26510（v1，cs.CL），[摘要页](https://arxiv.org/abs/2611.26510) / [HTML 全文](https://arxiv.org/html/2611.26510v1)
- 提交时间：2026 年 11 月 19 日（v1 的 UTC 时间戳是 19 日 17:57，正文页标注的日期写的是 11 月 20 日）
- 作者：Guoxin Chen、Shuang Sun、Fanzhe Meng、Xinyu Tang、Ruihua Song、Wayne Xin Zhao（6 人，标注了 equal contributions 与通讯作者）
- 单位：中国人民大学高瓴人工智能学院（Gaoling School of Artificial Intelligence, Renmin University of China）；公开项目页、代码仓与模型集合分别挂在 aweai-team.github.io/AutoMaster、RUCAIBox/AutoMaster、Hugging Face 的 RUC-AIBOX 集合下
- 分类：Computation and Language (cs.CL)

元信息我按摘要页和正文首页核过了，作者顺序、人数、机构都照抄，没从搜索结果里拼。

---

## 🤔 「做对了」不等于「教得好」

后训练里最省事的一条路，是让模型自己做题、把做对的留下、再 SFT 回去。说到这个，做过推理训练的人应该都熟悉 STaR 那一套：采样一堆 chain-of-thought，用最终答案当 verifier 过滤，fine-tune，再采样，循环。ReST、RFT、DeepSeek-R1 冷启动阶段的 rejection sampling，骨子里都是这个「生成 → 按结果筛 → 模仿」的模板。它稳定、便宜、不需要 value model，工程上是首选。

但这条链有个从没被认真追问过的假设：**成功的轨迹一定是好的示范吗？**

作者把这个问题挑明了。一条轨迹可能踩着一堆没验证的假设蒙对了答案，可能反复读同一个文件做无用功，可能写了个「看起来验过了」的自测——最终结果是对的，可学生从中学到的是这些习惯，不是那些真正支撑正确性的动作。反过来，一些真正有用的行为（改代码前先定位失败源头、数值结论要独立复核）在无引导轨迹里可能根本就没出现过。你只按 outcome 过滤，这两种差异都看不见。

这就是论文那句 *task success does not specify which behaviors post-training should reinforce* 的分量。它不是说 outcome filter 错了，是说 outcome filter 的信息量不够——它把「对不对」这个标量，硬当成了「该学什么」的完整说明书。

这篇论文的赌注下在一个很少被动的地方：**不改权重、不换老师、不加数据，只改老师做事的方式**。

---

## 🏗️ Skill-FT：给老师的 system prompt 里塞几段话

核心 idea 一句话能讲完：既然学生学的是轨迹，那就别只挑轨迹，先动手把轨迹生成时的行为分布掰过来。

具体做法是把自然语言写的「技能」（skill）拼到基础系统提示后面。一个 skill 可以是一条行为原则、一段具体操作流程，或者某个领域的硬知识；可以跨任务复用，也可以几条叠在同一道题上。写成公式就是论文里的式 (1)：

$$\tau \sim \operatorname{Rollout}(\pi_T, x; p \oplus s)$$

$x$ 是任务，$p$ 是基础 system prompt，$s$ 是选中技能的合并文本，$\oplus$ 就是字符串拼接，$\pi_T$ 是 teacher。你看，权重一个字没动，变的只是条件输入。可就是这段 $s$，会让 teacher 在解题过程中多验一步、少猜一次。

关键在于：这个干预**在老师本来就会做的题上也生效**。传统思路里，老师已经能解出来的题没什么可优化的，所以数据都往难题上堆；作者反过来讲，恰恰是这些「已经会做」的题，示范里的坏习惯最容易被整批复制进训练集。

学生端则是常规 SFT。有个细节后面会变成全篇最反直觉的发现：每条轨迹的原始 system prompt 会**连同技能指令一起**留在训练输入里，学生在训练时看得到老师当时被叮嘱了什么；到了评测，按官方协议把这些生成的技能全部撤掉。

![图1：同一个老师，不同的教法](https://arxiv.org/html/2611.26510v1/automaster_teaser.png)

*图 1（论文 Figure 1）：左边是 teaser 案例，任务是修一个偶尔会卡死的并行管线。无引导的老师只测了默认配置（4 workers × 5 runs，一个配置）；加了技能的老师额外试了 1、2、4、8 四种 worker 数，验证强度完全不同，而两者都把 100 条记录处理完了。右边三组柱子是 Base / Vanilla SFT / AutoMaster 在 Terminal-Bench 2.1、BrowseComp、Doc2Repo 上的对比，柱顶标着相对 Vanilla SFT 的增益：5.62、8.99、2.88、4.68、1.42、4.74 个点，误差棒是三次评测的 ±1 SD。*

那个 teaser 我挺喜欢的，因为它把「行为」这东西具象化了。两条轨迹都完成了任务，都「成功」，可一条只验了默认配置，另一条把边界情况扫了一遍。如果按成功率筛，这两条一模一样；如果按学生该学什么筛，差着一个太平洋。

---

## 🔁 AutoMaster：两层循环，一个 agent 自己折腾

Skill-FT 只回答了「怎么改老师」。难的是另一个问题：你怎么知道该写哪条技能、该套到哪些任务上、改出来的轨迹对学生到底有没有用？

作者的答案是让一个 LLM agent 把这件事当成实验来做，内外两层循环套着跑。

![图2：AutoMaster 框架总览](https://arxiv.org/html/2611.26510v1/automaster.png)

*图 2（论文 Figure 2）：内循环负责技能设计与数据策展，依次是 ① 分析模式（标出 bad patterns / good patterns）、② 设计技能并分配到任务子集 D1/D2/D3、③ pilot teacher rollouts、菱形判断「Effective?」，没奏效就回到 ② 精炼，试通了才 scale up 到 ④ 批量 rollouts、⑤ review & select 形成训练集。外循环拿这个训练集做 SFT，学生评测时不给任何技能，分数和轨迹沿虚线反馈回顶部的 agent。注意左下角图例：节点是动作，红点是坏行为，绿点是好行为。*

内循环的四步我顺着图捋一遍。agent 先读训练任务和初始轨迹（无引导 teacher 的和初始学生的都看），找两类东西：该压掉的坏习惯、老师很少演示但有用的行为；后面的轮次里还会翻学生的评测轨迹，看意图中的行为有没有真的长出来。接着写技能、把技能挂到任务子集上，先跑小样本 pilot rollout 试探——这一步是验「技能有没有引出目标行为，同时有没有在执行时带出新毛病」。不行就改技能措辞或者收窄适用范围，再来一遍。试通了才放大到批量 rollout，挑出来的轨迹再进训练集，挑的时候同时看任务结果和展示出来的行为，也可以顺手收几条本来就很干净的无引导轨迹。

外循环就朴素得多，但有个设计我觉得挺关键。第 $r$ 轮内循环产出训练集 $\mathcal{D}_r$，SFT 每轮都**从同一个初始 checkpoint** $\theta_0$ 重新开始：

$$\theta_r = \operatorname{SFT}(\theta_0, \mathcal{D}_r)$$

不是接着上一轮的学生继续训。这么做把「上一轮训歪了」的累积效应切掉了，每轮都是一个干净的对照实验，agent 拿到的是分数加轨迹两份反馈，用来决定下一轮是改技能、换任务覆盖、还是换一批示范。实验历史往前传，学生权重不往前传。论文给的预算是 5 轮 outer loop。

用伪代码写，整个骨架大概是这样：

```python
for r in range(5):                                  # outer loop
    skills = agent.design(initial_traj, eval_traj)  # ① 分析 bad/good patterns
    pilot = teacher.rollout(task_subset, p + skills)  # ③ 小样本试探
    if not agent.judge_effective(pilot):             # 没试通就回去改技能
        skills = agent.refine(skills, pilot); continue
    batch = teacher.rollout(full_pool, p + skills)   # ④ 放大
    D_r = agent.select(batch, unguided_traj)         # ⑤ 挑数据，保留 skill 文本
    theta_r = sft(theta_0, D_r)                      # 每轮从 θ0 重训，不接上一轮
    scores, trajs = evaluate(theta_r, skills=None)   # 评测不给技能
    agent.update(skills, pilot, D_r, scores, trajs)  # 反馈进下一轮
```

---

## 🔧 基础设施，以及技能到底长什么样

![图3：AutoMaster 的执行基础设施](https://arxiv.org/html/2611.26510v1/execution_infrastructure.png)

*图 3（论文 Figure 3）：agent 跑在 Docker 容器里，通过 MCP 接口读数据、提交 rollout 和 SFT 任务。左侧是数据池与 rollout service（Teacher API + teacher harness），中间是 SFT service 与共享 checkpoint 存储，execution controller 负责调度和失败恢复，右侧独立的 GPU 集群上 SGLang workers 直接从共享存储加载权重做评测，底部 sandbox cluster 用隔离容器并发跑任务和 verifier。最上面一条 Linked experiment records 把 Skills、Rollout trajectories、Training set、Checkpoint、Evaluation 串起来，让每次评测都能倒查到当初是哪条技能、哪批数据生成的。*

这套基建里我最在意的是那条「linked experiment records」。它不是一个炫技的工程模块，而是这套方法能不能成立的前提——agent 想判断「学生没学会某个行为，到底是老师没演示、还是演示了没被选进训练集」，就必须能把评测结果一路回溯到技能文本和轨迹。没有这条链，所谓「自主迭代」就只剩盲试。

配置上也有几个数值得记：agent 是跑在 Codex 里的 GPT-6 Astra（xhigh reasoning effort），teacher 是 DeepSeek-V4-Pro-Preview（high reasoning effort），学生是 Qwen3.5-4B 和 Qwen3.5-9B。看到这一串我第一反应是——所谓「自主后训练」，其实是把两个强得离谱的闭源模型当发动机用了，这点后面批评环节还得拎出来说。

技能本身长什么样？附录 B 把四条原样贴了出来，我看完觉得比方法章节更说明问题。比如 `execute-numeric-decision` 里那段是要求把验收条件真的写进检查代码里：

```python
error = abs(actual - reference)
passed = math.isfinite(error) and error <= allowed_error
print(actual, reference, error, allowed_error, "PASS" if passed else "FAIL")
assert passed, "numerical requirement remains unresolved"
```

它还专门叮嘱：返回比较结果不算数，必须走到 assert 或失败分支；不能擅自放宽容差、不能把失败输入换成通过的输入；检查失败时要把那个 case 和随机种子留着再查。另外三条分别叫 `recover-search-anchor`（搜索卡住时丢掉没根据的限制、换一条能独立定位实体的关系去查）、`contract-checkpoint`（改代码前先读失败的 assertion 和输入，定位是解析、转换还是状态的问题）、`expression-grammar`（直接给一套前缀/幂运算的文法，要求先推分组规则再动 parser）。

你看，这些不是「请认真思考」这种正确的废话，是带适用条件、带具体动作、带领域知识的硬约束。这大概也是它们真能改行为的原因。

---

## 🧪 实验：数字摆出来，顺便说几处我皱眉的地方

设置先交代清楚，省得后面每个数都要回头查。

| 项目 | 取值 |
|------|------|
| 训练任务池 | 每个域从公开数据集随机采 1000 条，固定，所有方法共享 |
| Terminal | 训练源 CalibForge，评测 Terminal-Bench 2.1（89 题） |
| Search | 训练源 FORT-Searcher，评测 BrowseComp Lite-185（185 题） |
| SWE | 训练源 DeNovoSWE，评测 Doc2Repo（50 个仓库，禁用搜索工具） |
| 学生 | Qwen3.5-4B / Qwen3.5-9B |
| teacher | DeepSeek-V4-Pro-Preview，high reasoning effort，temperature 1.0 |
| agent | Codex + GPT-6 Astra，xhigh reasoning effort |
| baseline | Base（不微调）、Vanilla SFT（512 条随机采样的无引导成功轨迹） |
| SFT 配方 | 全参数，512 条轨迹，3 epochs，lr $10^{-5}$（cosine 衰减到 $10^{-6}$，warmup 0.05），global batch 32，最大序列 131072 tokens，Megatron Adam，bfloat16 |
| 评测 | 第 3 轮末的 checkpoint，撤掉所有生成技能，三次运行取 mean ± SD |

主实验表（Table 1）我整张搬过来，因为每个数后面都要用：

| 方法 | TB2.1 4B | TB2.1 9B | BC 4B | BC 9B | Doc2Repo 4B | Doc2Repo 9B |
|------|---------|---------|-------|-------|-------------|-------------|
| Base | 15.73 ± 2.25 | 25.47 ± 2.59 | 33.69 ± 0.82 | 35.50 ± 2.72 | 21.39 ± 2.61 | 33.42 ± 1.60 |
| Vanilla SFT | 22.47 ± 1.12 | 26.59 ± 1.30 | 43.24 ± 0.54 | 46.85 ± 2.98 | 32.30 ± 4.00 | 36.05 ± 3.89 |
| AutoMaster | 28.09 ± 2.25 | 35.58 ± 1.30 | 46.13 ± 2.98 | 51.53 ± 1.25 | 33.72 ± 3.58 | 40.79 ± 3.21 |
| 相对 Vanilla SFT | +5.62 | +8.99 | +2.88 | +4.68 | +1.42 | +4.74 |

先看最扎眼的那格：Terminal-Bench 2.1 的 9B。Vanilla SFT 把 25.47% 抬到 26.59%，1.12 个点，几乎贴着 SD；同样这个老师、同样 512 条、同样配方，AutoMaster 到 35.58%，比 baseline 高 8.99 个点。作者用这句 *successful-trajectory distillation can understate the post-training headroom available from a fixed teacher* 来收，我觉得是成立的一一至少在这个 setting 里，天花板从来不是老师的能力，是示范的生成方式。

但坦率的讲，把六格一起看，故事没那么齐整。9B 三域全涨且幅度都在 4.68 以上，4B 就参差：TB 涨 5.62，BC 只有 2.88，Doc2Repo 只剩 1.42。作者自己也承认 gain 随模型规模变大，还给了个解释——更大的目标模型容量可能更接得住这些干预。这条我读的时候愣了一下，因为它跟常见的期待是反的：通常我们更希望小模型从「更好的教法」里获益更大，结果这儿是强者恒强。

真正让我放下宣传口径的是附录 A.6 的 paired bootstrap（10000 次重采样，seed 0，95% percentile CI）：

| Benchmark | 模型 | 样本数 N | 增益 | 95% 置信区间 |
|-----------|------|---------|------|--------------|
| Terminal-Bench 2.1 | 4B | 89 | +5.62 | [1.12, 10.49] |
| Terminal-Bench 2.1 | 9B | 89 | +8.99 | [2.25, 16.10] |
| BrowseComp | 4B | 185 | +2.88 | [−1.08, 6.85] |
| BrowseComp | 9B | 185 | +4.68 | [0.72, 8.83] |
| Doc2Repo | 4B | 50 | +1.42 | [−1.94, 5.33] |
| Doc2Repo | 9B | 50 | +4.74 | [0.65, 8.93] |

4B 上 BrowseComp 和 Doc2Repo 两组的区间是**跨零**的。也就是说，摘要里「六个 setting 全部提升」说的是均值，统计上这两组并不能排除「没提升」。作者把这张表老老实实放在附录、还在正文写了「4B 上只有 Terminal-Bench 2.1 的区间完全在零以上」，这点我得给个好评——很多工作会把它塞进脚注或者干脆不报。但读的人心里要有数：小模型上的收益，别当成铁板钉钉。

另一处我想吐槽的是「挑最好 checkpoint」这件事。Figure 8 显示五轮里最高分出现在第 2 到 4 轮，比第一轮高 2.16 到 4.87 个点，轮与轮之间是波动的（BrowseComp-9B 前三轮一直在 49% 附近磨，第四轮才跳到 51.53%）。agent 会基于累积的评测反馈挑最终 checkpoint，再重新跑三次独立评测报数。三次独立评测缓解了一部分，但「在 5 轮 × 多组实验里选表现最好的那次」本身就有轻微的赢家诅咒。论文用 paired bootstrap 补了不确定性，没补这部分选择偏差，算个小遗憾。

---

## 🔬 三条轨迹，比一张表更能说明在改什么

![图5：无引导与技能引导的逐任务对比](https://arxiv.org/html/2611.26510v1/behavioral_shaping_cases.png)

*图 5（论文 Figure 5）：三列分别是 Terminal、Search、SWE 的同一道题，上排无引导老师、下排技能引导老师。(a) 修 Poisson solver：无引导的声称四阶精度，但实测收敛阶 p≈2，Pass 时误差 3.31×10⁻⁴；引导后的换用九点差分模板并独立复核，验证 p≈4，误差降到 2.55×10⁻⁷。(b) 找一集 2017 年的电视剧：无引导的反复猜奖项名（Preston Sturges Award、Frank Capra Achievement Award）；引导后的先检索剧集—导演记录（3x13, Meera Menon），再验证 Columbia 与 Nora Ephron Prize 两条线索，答案都是 This Land Is Your Land。(c) 实现 JSON 规范排序：无引导的为了迁就写错的自测把全局排序反转，32/38 通过；引导后的去查 UTF-16 编码、修正自测输入、保留规定的升序，38/38 通过。*

这三例的分量不一样，值得拆开看。

(a) 和 (b) 是同一类：**两边都成功，outcome filter 完全分辨不出差别**。差别全在过程里——一个是「嘴上说四阶、数据其实二阶」，一个是「先猜答案再找证据」和「先找证据再定答案」。学生如果照着上排学，学到的就是「结论可以先写、证据后补」。这正是论文最想证明的那件事：成功标签把行为差异给抹平了。

(c) 有点不同，它其实是**实现正确性的提升**（32/38 → 38/38），不完全是「同样成功但行为更好」。作者的解读是「改代码前先搞清楚一次失败的检查到底说明了什么」——为了迎合一个写错了的自测去反转全局排序规则，是把 bug 固化进了实现。这个案例更像是在说行为塑造顺带修了正确率。把三种情况混在一张图里讲，我觉得论证上稍微有点滑，不过不影响主线。

顺着这三个案例再往下想一层，这套东西其实碰的是个老问题：监督信号的粒度。SFT 的天花板等于示范质量，而「质量」从来不只是「答案对」，还包括「凭什么对、怎么验的、失败时先动哪儿」。以前提升示范质量靠的是采样更多、挑更好的，这篇换了个方向——**直接改示范生成时的行为分布**。这个转向我觉得比涨幅数字更值得记。

---

## 📊 消融：两个组件单独用，反而比 baseline 还差

为了排除「只是挑了好任务」这个解释，作者做了几组控制，Table A4 的完整数值：

| 方法 | TB 4B | TB 9B | BC 4B | BC 9B | D2R 4B | D2R 9B |
|------|-------|-------|-------|-------|--------|--------|
| AutoMaster | 28.09 | 35.58 | 46.13 | 51.53 | 33.72 | 40.79 |
| w/o Skill Guidance | 17.60 | 28.46 | 45.77 | 49.19 | 32.03 | 36.97 |
| w/o Data Selection | 20.22 | 34.08 | 43.78 | 50.27 | 31.43 | 38.29 |
| Task-Matched SFT | 17.98 | 27.72 | 44.32 | 50.45 | 32.37 | 36.93 |
| Vanilla SFT | 22.47 | 26.59 | 43.24 | 46.85 | 32.30 | 36.05 |

`w/o Skill Guidance` 是让 agent 从无引导轨迹里挑数据，`w/o Data Selection` 是保留技能但用固定随机任务子集、不做 agent 主导的轨迹挑选，两者都保持同样的 teacher、任务池、512 条和 SFT 配方。去掉任一个，六个 setting 的均值全掉。

有意思的是 Terminal-Bench 2.1 的 4B 那一列：单独用技能引导（20.22）和单独做数据选择（17.60）——**双双低于 Vanilla SFT 的 22.47**，只有两者合体才到 28.09。到了 BrowseComp 4B，顺序又反过来了：数据选择 45.77，技能引导 43.78。

这个现象我没完全看懂，说实话论文给的解释也偏轻。作者说「选择的价值取决于老师本来演示了什么：选择偏向已有有用行为的轨迹，引导偏向引出无引导轨迹里罕见的行为」。道理讲得通，但解释不了「为什么单用任一个会掉到 baseline 以下」。如果技能真的引出了好行为，只是没挑数据，怎么会比不干预还差？一种可能是技能把轨迹分布推偏了，没经过筛选就混进了不太干净的样本；另一种可能是 512 条这个量级下噪声本就不小。论文没给更细的分解，我只能猜，欢迎有复现过的同学纠正我。

`Task-Matched SFT` 这一组是实验设计里我最欣赏的一笔。它从**同一个老师**的无引导轨迹里，严格对齐 AutoMaster 最终训练集的 task IDs 和每任务采样数，连成功和失败轨迹都一并保留以维持任务构成。任务组合锁死之后，AutoMaster 在六个 setting 上仍高出 1.08 到 10.11 个点。配合 Table A5 的配对统计——四份 Terminal/Search 语料里，79.3% 到 91.2% 的配对是「两条轨迹都成功」——就把「只是挑了更容易的题」这条路堵得差不多了。既然成功率一样，差异只能来自轨迹里展示的行为。

配对数据里还有个细节：每行 512 对，Terminal-4B 有 106 对是「无引导失败、AutoMaster 成功」，100 对是 9B；Search 是 45 和 64 对；而「AutoMaster 成功反而变失败」是 0 对。SWE 按任务分数分档，4B 是 improved 43 / unchanged 433 / decreased 36，9B 是 24 / 483 / 5。技能引导确实也会把一部分原本做不出的题救回来，但主体还是「都成功、过程更好」的那批——这跟论文的主张是对得上的。

---

## 💡 最反直觉的一发现：训练时把技能留着，评测时撤掉，反而更好

这部分我认为是全篇最值钱的地方，甚至比主结果更值得单独记。

按常理，训练输入和评测输入越像越好。评测时是不给技能的，那把技能指令从训练输入里删掉（作者叫 `Remove`），让训练 prompt 贴近评测 prompt，应该更稳；保留技能（`Keep`）反而制造了 train-test mismatch。监督输出一模一样——推理、回复、工具调用一个字没改，只是输入里多/少几段指令。

结果反过来了：六个 setting 上，`Keep` 的均值比 `Remove` 高 **2.15 到 4.01 个点**，评测时两边都不给技能。

![图7：技能上下文在示范里是怎么被显式引用的](https://arxiv.org/html/2611.26510v1/internalization_case.png)

*图 7（论文 Figure 7）：SFT 语料里的两个例子。(a) 任务是用 `python3 /app/process_data.py` 运行一个没有执行位的脚本，技能 `check-discriminating-evidence` 给出条件规则——直接调用才补执行位，解释器调用保留文档命令；teacher 的推理里明写「Let me check the skill about this」「it's invoked through the interpreter, not directly」，于是没去改权限，输出逐字节对上参考。(b) 修一个 C 表达式求值器，技能 `expression-grammar` 给了 `prefix := sign prefix | power` 这套文法，teacher 直接引用「Actually, the skill says」推导出 −2^2 应解析为 −(2^2) = −4，并据此改 parser、跑测试。*

作者的解释我觉得说到点子上了：保留技能指令，等于把「行为规则 + 它的适用条件」显式摆在输入里。删掉之后，同样的推理和动作还在，但学生得自己从任务和轨迹里反推「这条规则什么时候该用」——监督信号没变，可学习难度上去了。图 7 那两例尤其明显：teacher 的推理是**直接引用**技能的（「让我查一下技能怎么说」「技能里写了」），你把技能从输入里抹了，这段推理就变成了一句没有前文的引用，输入与输出之间的对应关系断了。

说到这个，它跟 context distillation / 把指令「烧进」权重的那类工作有点像，但方向更微妙：这里不是想让学生背下技能文本，而是让技能文本充当**条件的标注**。规则不变、答案不变，多给一段「为什么这么做」的上下文，学生反而把行为内化得更好。这个发现对做 SFT 数据的人挺有直接用处——别急着把训练 prompt 洗得跟线上完全一致，有些「多余」的上下文可能是有用的监督。

当然，「减少训练—评测 prompt 不匹配会让同一批轨迹更不好学」这个说法，论文只给了案例和均值差，没做更细的机制实验（比如按轨迹是否显式引用技能分组看增益）。我倾向于相信方向，但对强度保留一点余地。

---

## ⚠️ 我的判断：值钱在哪，水分在哪

给这篇论文打个标签，我会说它是**数据生成侧的一次扎实工程整合**，不是底层突破。核心手法（自然语言提示改行为、按反馈迭代、SFT 时留上下文）拆开看都不新，把它组装成一个能自主跑五轮、且每个变量都锁死的闭环，是这篇的真功夫。

我觉得最值钱的三处：

一是把「成功轨迹 ≠ 好示范」这件事做成了可操作的实验，而不只是口号。Task-Matched SFT 那组控制做得干净，把「挑好题」和「改行为」拆开了。

二是 **Keep vs Remove** 这个反直觉发现。它独立于整套 AutoMaster 框架也成立，任何一个做轨迹 SFT 的团队都能直接拿去试，迁移成本几乎为零。

三是附录 B 把四条技能原文全贴了、A.6 把跨零的置信区间也报了。这种不藏拙的写法，比多刷两个点更让人愿意信。

水分和悬着的地方也得摆出来。

**成本被转移了，没被消掉。** agent 是 GPT-6 Astra 跑 xhigh，teacher 是 DeepSeek-V4-Pro-Preview，五轮 outer loop，每轮都要 pilot、批量 rollout、全参数 SFT、独立 GPU 集群评测。论文没给算力账、API 开销、墙钟时间。「自主」指的是不需要人写技能规则，可这套闭环的运转全靠两个强闭源模型加一整套 MCP + SGLang + sandbox 基建。对没有这套家底的团队，复现门槛比方法本身看起来高得多。

**只跟 SFT 比，没跟 RL 比。** 论文说的是「后训练 headroom」，可对照组只有 Base 和 Vanilla SFT。真要论证「同一个老师被低估的余量」，跟 GRPO / 在线 RLVR 摆一起才更有说服力——毕竟改行为分布这件事，RL 本来就在做，只是路径不同。现在这个对比范围，结论得收着说。

**规模偏小。** 4B / 9B 两个尺寸、512 条轨迹、3 个域、3 个 epoch。作者自己说容量越大收益越稳，那 70B 以上会怎样、跨到别的域还成不成立，一个字没测。9B 上三个 CI 下界为正，4B 上两个跨零——「小模型更该受益」的直觉在这儿是落空的。

**还有个更根本的问题没碰**：技能由 agent 自己写、自己判定好不好、自己挑数据，可「好不好」这把尺子到头来还是 benchmark 分数。会不会出现一种更隐蔽的 reward hacking——agent 慢慢摸索出一批专门讨好这三个 benchmark 的技能，而不是真正通用的好行为？五轮还看不出来，轮数再往上、换一批评测集，这个风险值得有人接着查。

---

## 🔗 跟同期工作摆在一起看

剥掉包装看本质，这篇踩在几条线的交叉点上：

| 路线 | 代表 | 改的是什么 | 跟 AutoMaster 的差别 |
|------|------|-----------|---------------------|
| 成功轨迹过滤 / 自举 | STaR、ReST、RFT、R1 冷启动 | 只按 outcome 筛，不动生成过程 | AutoMaster 的 baseline 祖宗；论文主张 outcome 信息不够，要改行为 |
| 轨迹选择 | Behavior Priming、PROOF-Gen | 从已有轨迹里挑带好行为的 / 从失败里恢复验证过的示范 | 选择发生在生成之后；AutoMaster 在生成之前就干预 |
| 技能 + 参数更新 | Skill-to-LoRA、SkillGym | 用技能引导示范训 adapter / 把人类技能内化 | 关注技能本身怎么进权重；这篇更在意「老师的行为」和「训练时留不留技能」 |
| on-policy 自蒸馏 | Skill-SD | 技能条件的 teacher 给学生自己的 rollout 打 token 分 | 走 on-policy 打分；这篇是改 teacher 轨迹后做 off-policy SFT，两条路 |
| 自主后训练系统 | FT-Agent、RSI-Master、autoresearch | agent 改数据/训练代码/实验编排，靠评测反馈迭代 | 优化对象是训练策略与研究探索；AutoMaster 把优化对象收窄到「teacher 展示的行为」 |

这么一摆，定位就清楚了：它没有去卷「更强的 optimizer」或「更聪明的实验 DAG」，而是把自主后训练的旋钮拧到了一个别人没太碰的位置——**示范里的行为**。跟 RSI-Master 那种「防 hacking、防策略锁死、用 reviewer DAG 组织探索」的系统比，这篇的问题更窄、控制变量更狠，代价是野心也小一圈。

坦率的讲，我更愿意把它读成一次「问题定义」上的贡献：它提醒做 agent 后训练的人，除了「数据够不够多、老师够不够强、算法够不够花」，还有一个一直被跳过的维度叫「老师做对题时的样子」。这个提醒比 8.99 个点耐放。

---

## 📝 工程启发与收尾

如果你也在做轨迹 SFT，有几件事我觉得可以马上试，不用等复现整套系统：

别只按成功率筛数据。抽一批「成功」轨迹人工看看有没有那种「结论先写、证据后补」「反复读同一文件」「为了过自测改规则」的模式，这几类是最容易整批复制进学生权重的。真要动手，先写三五条带适用条件、带具体动作的技能塞进 teacher 的 system prompt，小样本 pilot 验一下有没有引出目标行为、有没有带出新毛病，再放大——这套流程不依赖 AutoMaster 那套基建也能跑。

另一个几乎零成本的改动：**训练输入里别把生成时的指令上下文洗掉**。哪怕评测不给，留着它可能帮学生把「什么时候该用这条规则」一起学进去。Keep vs Remove 那 2.15 到 4.01 个点，是我读完全文最想在自己数据上验一遍的数。

再往深一层，这篇其实指向一个还没人认真回答的问题：我们到底该用什么来定义一条「好示范」？现在的答案基本是「答案对」。可一个学生如果学会了在四阶精度上撒谎、在改代码前不去定位失败、在验证时走过场，即便每题都做对，它学到的东西能叫能力吗？AutoMaster 给了一个在三个域上能跑通的近似解，但「行为质量」怎么度量、怎么不让它退化成对 benchmark 的另一种过拟合，这两问还敞着。

如果你手上有更大的模型或者别的域，我挺好奇这套「改教法」的余量还在不在。尤其是 4B 上那两个跨零的区间——小模型到底是接不住，还是这几条技能写得不够对路？这块的答案，可能比再刷一个 SOTA 有意思。

---

觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我
