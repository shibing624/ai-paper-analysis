# 窗口还没满，模型先把记忆删了：把上下文管理权交还给模型，到底走到哪一步了

你有没有碰到过这种情况：一个 Agent 跑得好好的，工具调用越堆越多，检索回来的文档也越堆越长，然后外层 harness 看一眼窗口快满了，啪一下把前面一大段历史压成摘要。模型刚才还在追某个表格里的脚注，下一轮就被喂了一份“之前干了些事”的概括。答案错在哪儿？常常不是模型不会推理，是关键证据被别人替它收走了。

我之前调长任务的时候也踩过类似的坑。日志看起来很干净，上下文 token 数也漂亮，可任务就是断在某个细节上。说实话，那时候我的第一反应是加检索、加缓存、加更长窗口；现在回头看，问题更像是控制权放错了地方。窗口该留什么、该扔什么，最懂任务状态的那位，偏偏没有发言权。

这篇 *Letting Models Manage Their Own Context: Where We Stand* 让我停了一下。它没有再造一个新模型，也没有丢出一个复杂训练算法，而是把上下文管理这层控制面从 harness 手里拿过来，递给模型：你自己的记忆，你自己整理。结果还挺硬。

---

## 🎯 核心摘要

痛点很具体：长任务里，harness 在窗口达到阈值时替模型摘要，容易把后面还要用的证据压没。论文的做法是在 NVIDIA 开源 Agent harness NOOA 里开放 context API，模型可以自己选一段事件做 `collapse`，也能按标签把归档原文取回，再配一个实时用量指示和压力警告。数据上，GPT-6.1 Sol 在 32K 预算的 OfficeQA Pro v2 从 21% 到 56%；Kimi K3 最多少用 41% 输入 token；服务侧复用编辑后未变后缀的 KV cache，让每次请求 prefill 计算少 4–15%，精度看不出可见变化。我的判断：这不是底层模型结构突破，更像一次把模型能力、Agent 控制面和推理引擎放到同一张桌子上的工程实测。值钱的地方在“协同设计”这四个字，不在单点技巧。

---

## 📖 论文信息

| 项目 | 内容 |
|---|---|
| 英文标题 | Letting Models Manage Their Own Context: Where We Stand |
| 中文话题 | 把上下文管理权从 harness 交还模型 |
| arXiv ID | 2611.34147，版本 v1 |
| 作者 | Alessio Devoto, Avinash Vem, Tara Safavi, Sanjay Kariyappa, Yu-Hsuan Tseng, Paul Furgale, Ricardo Silveira Cabral |
| 机构 | arXiv 摘要页未列作者机构；摘要中只写明 NOOA 是 NVIDIA 的开源 agent harness |
| 日期 | Submitted on 25 Nov 2026；v1 Wed, 25 Nov 2026 10:44:04 UTC |
| 学科 | Artificial Intelligence，cs.AI |
| 摘要页 | https://arxiv.org/abs/2611.34147 |
| HTML | https://arxiv.org/html/2611.34147v1 |

作者顺序和日期都按 arXiv 摘要页抄了下来。机构这块我保守一点：摘要页没有单独的 affiliation 字段，就把原文里能看到的项目归属写清楚，不替作者补单位。

标题里的 *Where We Stand* 挺诚实。它不像是在宣布“上下文问题解决了”，更像一份站位图：现役模型拿到这类工具后，谁会用，谁乱用，谁干脆不用；给推理引擎加一点配合，账又能省到什么程度。

---

## 🧠 问题动机：窗口满了，谁该做决定

先把两种策略摆在一起。传统写法里，harness 是管家。它盯着 token 数，到了 80% 预算就触发压缩，保留初始指令和最近 10 个事件，把更早的历史交给模型生成一段摘要。简单、稳定、好实现，也很粗暴。

模型驱动写法换了个方向。模型拿到 NOOA 的 context API、实时上下文用量指示、一条“到 85% 再考虑压缩”的时机提示，以及窗口接近上限时的 context-pressure warning。压缩不再由管家拍板，模型自己决定改哪段、怎么写摘要、什么时候把归档内容取回来。

| 维度 | harness 驱动 | 模型驱动 |
|---|---|---|
| 谁决定何时压缩 | harness 阈值，论文设为 80% | 模型参考实时用量，提示里建议 85% 后考虑 |
| 谁决定压缩哪段 | harness 的启发式规则 | 模型选择事件区间 |
| 谁写摘要 | 模型 | 模型 |
| 旧内容怎么处理 | 保留初始指令和最近 10 个事件，早期历史进摘要 | 事件被归档，可按原标签检索 |
| 额外信息 | 无压力提示，无编辑工具 | 用量指示、压力警告、编辑与取回工具 |
| 风险 | 丢证据，摘要时机不合任务节奏 | 模型不会用、过早压、压太狠、反复取回 |

这里有个容易被忽略的点：论文没有换模型权重，Section 4 明确写的是 without extra training。也就是说，NOOA 只是把工具递过去，看现役模型能不能在真实任务里自己用起来。我的第一反应是，这实验有点像把版本控制交给一个没受训的新同事。工具在那儿，不代表他会在合适的时候 commit，也不代表他记得以后去 `git show`。

任务选择也挺贴题。OfficeQA Pro v2 有 90 道关于美国联邦收支的问题，Agent 要在预解析的大文档库里搜、读、答；读文档会迅速塞满窗口，适合观察证据怎么被保留。MLE-bench Lite 有 22 个 Kaggle 比赛，Agent 要看数据、写代码、调试、评估、提交，状态多但上下文需求没那么夸张，刚好当对照。四个模型是 GPT-6.1 Sol、Kimi K3、DeepSeek V4 Pro、Claude Opus 5.5。预算从 8K 扫到 512K token，预算是硬上限；请求超限会触发兜底，删掉最老事件重试，三次失败就算错答。

论文把缩小预算称为制造 *context pressure*，用任务所需信息量与窗口容量的比值，模拟更长任务。这个代理很实用，也埋了一个后面要吐槽的坑：窗口被硬压小，和真的跑一个 12 小时 Agent，不是同一种压力。

---

## 🏗️ 方法核心：一段可回滚的记忆，而不是一锅端摘要

NOOA 的做法不绕弯：上下文是一串事件，模型可以调用 `collapse` 把某段事件替换成摘要，归档内容不删，之后还能按原标签取回。论文 Figure 1 的上半部分是 harness 在阈值处把早段历史糊成摘要；下半部分是模型在任务过程中挑 e3–e9 压成摘要，e1、e2、e10–e12 继续原样留着，归档区通过 tag 可查。

那个 API 小得有点可爱：

```python
summary_tag = self.events.collapse(
    "3", "17",
    summary_text="Explored the repo layout; tests live in tests/unit."
)
original = self.events["9"]
```

两行代码，把上下文管理拆成两个动作：把一段长历史换成短状态，以及在后悔时把某个原始事件拿回来。你可以把它想成带回收站的白板，不是拿橡皮直接擦掉。这个概念不复杂，难的是决策。

模型得同时判断：当前哪段信息已经完成使命，摘要要保留哪些数字、路径、失败原因，后面还可能问到哪个原始事件，什么时候该读取归档。harness 驱动只要求模型写摘要；模型驱动要求模型管理状态。难度不在一个量级。

用个记号写得更清楚。设当前上下文是事件序列 $C_t$，模型选择区间 $[i:j]$，写成摘要 $s_{i:j}$，编辑后上下文变成：

$$C_{t+1} = C_t[:i] \oplus s_{i:j} \oplus C_t[j:]$$

$\oplus$ 是拼接。真正费劲的不是这个式子，而是 $s_{i:j}$ 里该写什么。写短了，后面追不回线索；写长了，白压；写错了，模型会把自己的错误记成任务状态。论文后面那堆行为差异，几乎都从这儿长出来。

**模型自己写摘要不够，模型还得会回收证据**。

说到这个，我想起 Related Work 里一串名字：MemGPT 把记忆分层，Recursive Language Models 把长输入放进外部编程环境，MEMENTO、AgentFold、Context-Folding、SelfCompact 都在碰“保留、压缩、重组历史”；Context Language Models 把 live context 当可编辑文件，SideQuest 让模型管 KV cache，Cache Engineering for Agents 更靠近服务引擎。点子不是从零冒出来的。这篇论文的可贵之处，是把一个统一 API、四个现役模型、两个基准、多个预算和 serving 省算放在同一套测量里，看“现在到底站在哪儿”。

---

## 🔧 推理引擎这块：上下文变短，怎么反而没省钱

很多工程同学会下意识把“输入 token 变少”直接换算成“推理更便宜”。这篇论文专门泼了一盆冷水，泼得挺准。

标准 prefix caching 只复用从开头到第一个改动点之前的 KV cache。假设原上下文是 $A \oplus B \oplus C$，模型把 $B$ 压成摘要，得到 $A \oplus \text{summary} \oplus C$。$A$ 还能复用；摘要当然要重新 prefill；麻烦的是 $C$ 明明一个字没改，却因为位置落在编辑点之后，也被判失效。Figure 12 画的就是这件事：prefix reused，summary 和后面全部 fresh computation。

把“需要重新 prefill 的 token 数”粗略记作 $T$：

$$T_{\text{standard}} \approx |s_{i:j}| + |\text{suffix}|,\qquad T_{\text{suffix reuse}} \approx |s_{i:j}|$$

 suffix KV reuse 想保住后面那段没变文本的缓存状态。省下来的，就是原先被无辜重算的后缀。论文把 NOOA 的 context API 和一个跑在 SGLang 里的原型接起来，用 Qwen3.6-27B，在 MLE-bench Lite、OfficeQA Pro、DeepSWE 上测。结果是每次请求 prefill 计算少 4–15%，任务精度没有可见变化。OfficeQA Pro 省得最多，因为工具调用拉进来的长文档常常留在编辑点之后，平均后缀约 10K token，约占 prompt 的 60%。

**上下文变短不一定更便宜**。

这句得贴在 Agent 平台的墙上。只看 token 账单里的输入数，会把 prefix cache 失效、编辑后重算、缓存释放这些成本漏掉。坦率地讲，论文也承认 suffix KV reuse 是近似，不是严格重算；它测的是 prefill compute，不是端到端延迟或总费用。看到一个漂亮的 15%，别急着写进 OKR，先看 suffix 占比、位置编码处理、误差条和任务 reward delta。

---

## 🧪 实验主表：把散在图和正文里的硬数字收拢

原论文没有给一张传统意义的 Table 1，核心数值散在 Figure 3–13 和对应段落里。我把能核对的数字收成表，出处按 Section 或 Figure 标出来。论文里的 OfficeQA Pro v2 分数是 90 道 trial 的通过比例，错误也计入；MLE-bench Lite 用 leaderboard percentile，缺失分数按 0 算。

| 主题 | 设置 | 对比 | 原文数字 | 出处 |
|---|---|---|---|---|
| OfficeQA Pro v2 主结果 | GPT-6.1 Sol，32K 预算 | harness 驱动 vs 模型驱动 | 21% → 56%，接近 3 倍 | Abstract，Section 1/4，Figure 3 |
| OfficeQA Pro v2 趋势 | 四个模型，8K–512K | 两种策略 | 小预算增益最大，预算增大后收敛；Opus 在大预算才追上 | Section 4，Figure 3 |
| MLE-bench Lite | 四个模型，22 个任务 | 两种策略 | 增益更小，主要集中在最紧预算 | Section 4，Figure 4 |
| Kimi 输入 token | OfficeQA Pro v2，32K 预算 | harness 驱动 vs 模型驱动 | 36M vs 62M 累计输入 token，少 41%；分数 43% vs 41% | Section 6，Figure 9 |
| Kimi 输入 token | OfficeQA Pro v2，64K 预算 | harness 驱动 vs 模型驱动 | 少约 11% 累计输入 token；分数 57% vs 52% | Section 7，Figure 10 |
| 输入 token 上限 | 摘要与正文 | Kimi gains | 最多少 41% 输入 token | Abstract |
| KV cache 复用 | Qwen3.6-27B，32K，SGLang 原型 | 后缀复用 vs 标准 prefix caching | prefill 计算每次请求少 4–15%，精度无可见变化 | Section 7，Figure 13 |
| 后缀规模 | OfficeQA Pro | 编辑后保留的未变文本 | 平均约 10K token，约 60% prompt | Section 7 |
| KV 任务 reward | 三个基准 | 有复用减无复用 | 各个 delta 多数贴近 0，误差条覆盖零点 | Figure 13 |

看到 21% 到 56% 这个数，我确实愣了一下。32K 预算下，模型自己整理上下文，能把 OfficeQA Pro v2 的通过率抬 35 个点；这已经不是“日志更干净”那种小修小补。可翻到 MLE-bench Lite，增益立刻收窄。原因论文也给了：那类任务本身没那么吃上下文。想让管理权更值钱，得让任务真的被窗口卡住。

Figure 3 的 GPT-6.1 Sol 子图很直观：绿色模型驱动线在 16K、32K、64K 一截截拉开，到 128K 以后两条线基本贴住；Kimi K3 子图也有类似形状，只是两条线更接近。也就是说，模型驱动更像“紧张预算下的救生绳”。窗口够大时，harness 的粗暴摘要也来得及兜住局面。

Figure 9 还有一个我很喜欢的对照：同样是 32K 最大上下文，Kimi 模型驱动跑出 43% 分数、约 36M 累计输入 token；harness 驱动是 41% 分数、约 62M 输入 token。同样的预算，不等于同样的用法。你想想看，一个 Agent 如果懂得在合适时机把工具输出压成可检索状态，后面每一轮都不用重复拖着整堆旧文档走，省下来的不是一次请求，而是整条轨迹的反复 prefill。

---

## 🔬 消融与行为诊断：模型差异比方法差异还刺眼

这部分才是论文最耐读的地方。同样的 API、同样的指令，四个模型像四个性格完全不同的同事。

| 观察维度 | 原文结果 | 我读到的信号 |
|---|---|---|
| API 使用频率 | DeepSeek V4 Pro 53%，Kimi K3 45%，GPT-6.1 Sol 42%，Claude Opus 5.5 4% | Opus 几乎不碰工具，后面分数差异不能只归因于压缩算法 |
| 单次压缩力度，中位数 | GPT 3.7×，Kimi 2.1×，DeepSeek 1.8×，Opus 1.5× | GPT 敢下刀，且更常把上下文砍到很小 |
| 压缩后重新取回归档 | GPT 18.6%，Kimi 6.9%，DeepSeek 3.5%，Opus 14.1% | 敢压的人也更会补证；Opus 样本少，噪声大 |
| 64K 首次压力警告后的动作 | GPT 69% 先读 API 文档再压；DeepSeek 46% 先查事件历史；Kimi 49% 直接压；Opus 多数直接收尾作答 | 不同模型把“压力”解释成不同任务 |
| 时机提示消融 | 去掉 85% 指令后，Kimi 在 128K 从约 84% 降到 29% 才首次压缩；DeepSeek 在 256K 从 83% 降到 40% | 提示词能改压缩时机，能力并不稳 |
| Opus 时机提示消融 | 带指令时几乎不压；去掉后常态且很早压，128K 下中位数 16% | 一句提示能把行为从“不用”推到“过早用” |
| 早压代价 | 128K 及以上，无时机提示的 Kimi 和 DeepSeek 比 harness 基线低约 8 分；GPT 和 Opus 没有同量级变化 | 省 token 不是免费午餐，压太早会伤证据链 |

**时机比压缩率更脆**。

论文给这个现象起名 *context anxiety*：模型还剩很多空间，却急着把上下文变小，像考场里提前半小时交卷。Kimi K3 的公开技术说明里提到对过长 rollout 有 RL 惩罚，作者猜测这可能教会模型“省 token”，也可能让它在窗口还宽时误判预算。注意，作者自己把这写成 conjecture，不是验证过的训练结论。这个分寸我认可。

压力警告那组数据也挺有画面感。GPT 遇到警告，先去读 API 文档；DeepSeek 先翻事件历史；Kimi 常常直接压；Opus 更像“收到，那我答完吧”。这不是同一个算法跑出四种参数，是四个模型对“我快没空间了”这句话有四种理解。做 Agent 平台的人看到这里，应该会把“上下文管理策略”从 infra 配置里拎出来，当成模型能力的一项单独指标。

---

## 📊 论文原图与图注

下面的图来自 arXiv HTML 随论文发布的图形对象，转成 JPG 后用在线 URL 展示；图注按原文 Figure 编号整理，选取的是和主结论最贴的子图。图片我每张只看了一遍，确认坐标轴、标签和数值可读，下面的解释也按图里能看到的内容写。

![图 3：GPT-6.1 Sol 在 OfficeQA Pro v2 上，分数随上下文预算变化](https://www.mulanai.com/fs/files/1127_280b8fea_fig3_gpt.jpg)

*图 3（GPT-6.1 Sol 子图）：横轴是 8K 到 512K 的 context budget，纵轴是 task score。绿色实线为模型驱动，灰色虚线为 harness 驱动。32K 处绿色约 56%、灰色约 21%，到 128K 后两条线靠拢。原文图注说明分数是 90 个 trial 的通过比例，包含错误；点为重复实验等权均值，误差条为最小到最大范围。*

![图 3：Kimi K3 在 OfficeQA Pro v2 上，分数随上下文预算变化](https://www.mulanai.com/fs/files/1127_a7f45355_fig3_kim.jpg)

*图 3（Kimi K3 子图）：同样是 OfficeQA Pro v2 的 score 对 context budget。绿色模型驱动线在 8K 到 64K 略高于灰色 harness 基线，128K 以后两者基本交叠；图上还能看到 512K 时两条线都到 60% 出头。它把“预算紧张时收益大、预算宽裕时收敛”画得很清楚。*

![图 5：不同模型至少调用一次压缩的任务占比](https://www.mulanai.com/fs/files/1127_02b5aee6_fig5_ado.jpg)

*图 5：纵轴是带至少一次 compaction call 的任务比例。DeepSeek V4 Pro 为 53%，Kimi K3 为 45%，GPT-6.1 Sol 为 42%，Claude Opus 5.5 只有 4%。光看这一张图就知道，不能把“开放了 context API”当成“所有模型都会管理上下文”。*

![图 6 左：每次压缩后的上下文缩小倍数](https://www.mulanai.com/fs/files/1127_7ce943ac_fig6_com.jpg)

*图 6 左：纵轴是对数刻度的 compression factor，定义为发起压缩那轮输入 token 除以下一轮输入 token。箱线图中位数大约是 GPT 3.7×、Kimi 2.1×、DeepSeek 1.8×、Opus 1.5×；虚线 1× 表示没有变化。GPT 的箱体和须都更靠上，压缩动作最激进。*

![图 6 右：压缩后被重新打开的调用比例](https://www.mulanai.com/fs/files/1127_b389f0d2_fig6_reo.jpg)

*图 6 右：纵轴是 compaction calls reopened。图上数字为 GPT 18.6%、Kimi 6.9%、DeepSeek 3.5%、Opus 14.1%；原文提醒这是下界，而且 Opus 因压缩次数太少，估计噪声较大。敢压和会取回，最好成对出现。*

![图 8：Kimi K3 在有无时机指令下，首次压缩时的上下文占用](https://www.mulanai.com/fs/files/1127_b85a1f04_fig8_anx.jpg)

*图 8（Kimi K3 子图）：横轴是 8K 到 512K 预算，纵轴是首次压缩决策时的上下文占用比例。标着 85% instruction 的绿色线基本贴着 85% 虚线；橙色 no timing instruction 线从 8K 的约 85% 一路降到 512K 的十来个百分点。原文正文点名 128K 从约 84% 掉到 29%，这就是 context anxiety。*

![图 9：Kimi K3 分数与累计输入 token 的关系](https://www.mulanai.com/fs/files/1127_a5d2cbd3_fig9_kim.jpg)

*图 9：横轴是每次运行的累计输入 token，纵轴是 task score，点旁标预算。图上专门标注 32K budget：41% fewer input tokens，score 43% vs. 41%。绿色模型驱动点在约 36M token 处拿到约 43 分，灰色 harness 驱动点在约 62M token 处拿到约 41 分。原文说明累计输入包含缓存与非缓存的每一轮输入。*

![图 13 左：后缀 KV 复用带来的 prefill token 降幅](https://www.mulanai.com/fs/files/1127_ea093245_fig13_pr.jpg)

*图 13 左：横轴是相对标准 prefix caching 的 reduction in prefill tok/request，任务为 OfficeQA Pro、MLE-bench Lite、DeepSWE。可读数值为 OfficeQA Pro +15.1 ± 0.8 和 +12.6 ± 0.7；MLE-bench Lite +5.6 ± 1.1 和 +5.3 ± 1.2；DeepSWE +4.3 ± 0.3 和 +1.0 ± 0.2。两组条分别对应模型驱动与 harness 驱动，OfficeQA Pro 后缀最长，省得最多。*

![图 13 右：后缀 KV 复用相对无复用的任务 reward 变化](https://www.mulanai.com/fs/files/1127_a6bdca97_fig13_re.jpg)

*图 13 右：横轴是 task reward delta，零线居中。OfficeQA Pro 的 task score 两个 delta 都是 +0.000 ± 0.02；MLE-bench Lite 的 mean leaderboard percentile 是 -0.011 ± 0.03 和 +0.081 ± 0.05；DeepSWE 的 fail-to-pass rate 是 +0.013 ± 0.02 和 -0.043 ± 0.04。多数点贴近零线，误差条也不小，这比“精度无损”四个字更诚实。*

---

## 🤔 批判性判断：亮点很亮，边界也很硬

我喜欢这篇论文的地方，是它没有把“模型管理上下文”包装成一个新宗教。它把工具交给模型，然后老老实实记录：有人用得好，有人不用，有人一紧张就提前删，有人还会回头找归档。这个观察比一个排行榜第一名更有工程价值。

| 让我点头的点 | 让我皱眉的点 |
|---|---|
| 统一 API 下横评四个现役模型，能看到能力差异 | 模型驱动额外拿到用量指示、警告和编辑工具，和 harness 基线不是同样信息条件 |
| 同时看准确率、输入 token、压缩行为、取回行为和 serving 计算 | 90 道问答加 22 个 Kaggle 任务，样本不大，小差异可能不稳 |
| 把 prefill 重算这笔账算出来，戳破“短上下文等于低成本” | 用缩小预算模拟长任务，兜底删最老事件也会影响得分 |
| 主动列出训练未知、四个模型、单开源模型和单引擎限制 | GPT 更强的训练归因只是猜测，没做训练数据或消融验证 |
| 连接 MemGPT、MEMENTO、CLM、SideQuest 等前置路线 | suffix KV reuse 只测了 prefill compute，没测端到端延迟、显存和真实费用 |

回到公平性这块，我的第一反应是：这像能力评测，不像纯策略消融。harness 驱动只有 80% 阈值和 10 个最新事件；模型驱动有 API、实时用量、85% 指令、压力警告，甚至能读文档、查历史。多出来的不只是工具，还有信息。这样比出来的优势，回答的是“把 richer harness 交给模型会怎样”，不是“只换压缩触发方会不会更好”。论文并没有藏着这一点，实验设置写得清楚；读的人别把结论外推过头就行。

还有个问题我没完全看懂，也可能我的理解有偏差：suffix KV reuse 在编辑后保留未变后缀的缓存状态，位置变了、前文变了，注意力上下文也变了，论文把它称为 approximation。原型到底怎么处理位置编码、跨层依赖、混合注意力结构，HTML 正文没有展开到能复现的粒度。Figure 13 右的 reward delta 有正有负，虽然误差条大多覆盖零，但“no measurable change in accuracy”最好还是按“在这三组任务、一个模型、一个引擎下没测到可见变化”来读。

再说训练。论文把“训练模型管理自身上下文”放在结论里，方向我同意，可难度比听上去大很多。上下文编辑会打破 append-only 轨迹：最终上下文里不再保有早先动作产生时的输入。若拿最终状态直接做 RL，信用分配会拧巴。公开资料里，Context Language Models 的摘要提到训练要保留中间上下文并做分段策略梯度；SideQuest 用辅助推理任务决定 KV cache 清理；MEMENTO、ReSum、Sculptor 一类工作也在碰分段训练。说到底，工具调用只是外壳，真正要学的是“在哪个状态下留下哪段证据”的策略。

**工程实测加协同设计论证**，这是我对它的定位。不是新模型结构，不是新 attention，也不是把压缩公式推到多漂亮。它把现役模型的能力边界、harness 接口设计和推理引擎缓存行为放到一张图里，指出一条很实际的路线：模型训练要跟上，服务引擎也要跟上。

---

## 💡 工程启发：如果你也在做 Agent 平台

别急着把 compaction 全扔给模型。更稳的做法是先把控制权拆开：harness 负责硬上限、兜底、错误重试和安全边界；模型负责在允许区间内选择压缩对象、写摘要、按标签取回。两条路都保留，用灰度开关切。

监控指标也得换。除了任务分数和 token 数，至少记录 API 调用率、首次压缩时窗口占用、压缩倍数、归档重开率、压力警告后的动作分布、超限失败次数。论文最有信息量的几张图，几乎都不是主分数曲线，而是这些行为量。没有它们，模型早压、乱压、压完不取回，线上只会表现为“偶尔莫名答错”。

训练侧要保留编辑前后的中间上下文。轨迹被编辑过，最终 prompt 不再等于每一步动作的输入；把最终答案的奖励直接广播到整条轨迹，很容易把错误 credit 塞给已经不在场的状态。分段、掩码、按原输入评估各段动作，这些词听着琐碎，真做 RL 时会救命。

推理侧别只看“输入 token 降了”。把 prefix cache 命中区间、编辑点之后的 suffix 长度、重算 token、缓存释放、batch 影响、端到端 p50/p95 延迟一起记。suffix KV reuse 值得试，尤其长文档检索后还留在编辑点后面的场景；但它是近似，必须同时盯 reward delta，不只盯省了多少 prefill。

如果你只做应用，最可迁移的一条反而是产品观：上下文不是日志，是工作记忆。日志可以按时截断，工作记忆得允许模型自己整理，也允许它反悔。给用户一个“归档可检索”的语义，比给一个更长的窗口有用得多。

---

## 📝 收尾：走到哪儿了

我的判断挺明确：这篇论文值得读，尤其你正在做 Agent harness、长任务评测或者推理服务。它把一条以前散在 MemGPT、CLM、SideQuest、cache engineering 里的路线，压成了一个很具体的问题——现役模型现在能不能用 context API 管住自己的上下文，推理引擎能不能让这种编辑变得便宜。

答案不是漂亮的“能”，也不是扫兴的“不能”。是“有些模型能，紧预算下收益很大；但时机、频率、力度都漂，提示词一改就变；服务侧再配合一点，省算才落得下来”。这比一句口号有用。

标题里的 *Where We Stand* 其实已经把分寸说完了。下一站不是更花哨的摘要 prompt，而是 **模型训练 × 推理引擎** 的协同设计：一边教模型什么时候留、什么时候压、什么时候取回；一边让引擎识别编辑边界，复用未变后缀的 KV cache，及时释放不再需要的条目。两块拼上，上下文管理才可能从 harness 的小技巧，变成 Agent 的基础能力。

到收尾，我还是想留一个问号：如果模型真学会管理自己的上下文，评估基准也得跟着变。只看最终答案和总 token 数，可能刚好漏掉这项能力最值钱的部分——它在中途救回了哪条证据，又避开了哪次重算。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
