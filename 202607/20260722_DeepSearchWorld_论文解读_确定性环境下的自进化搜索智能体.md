---
title: DeepSearch-World：可验证环境里长出来的搜索智能体
date: 2026-07-22
arxiv: 2607.07820
authors: Xinyu Geng, Xuanhua He, Sixiang Chen, Yanjing Xiao, Fan Zhang, Shijue Huang, Haitao Mi, Zhenwen Liang, Tianqing Fang, Yi R. Fung
affiliations: HKUST, Tencent, HKUST(GZ)
tags: [LLM Agent, Self-Distillation, Web Search, Tool Use, Reinforcement Learning]
---

# DeepSearch-World：可验证环境里长出来的搜索智能体

你有没有被一个事困扰很久——让 LLM 真的学会"上网查资料然后给你写个靠谱的答案"？

SFT 路线简单暴力，找个强模型（比如 GPT-4o、DeepSeek-R1）去跑任务，采一堆成功轨迹，让学生模型去模仿。听上去很美，但有个要命的问题：**强模型蒸馏出来的天花板就在那，学生模型追到尽头还是追不上老师**。这条路线 2024 年很火，但天花板肉眼可见。

RL 路线（GRPO 那一波）呢？reward 终于从"模仿"变成了"做事"，但你给它一道多跳搜索题，等模型在搜索引擎里翻了好几十个网页，末了你只能告诉它"答对了+1，答错了-1"——**这种稀疏的 outcome reward 对长程任务几乎是个噪声信号**。你甚至不知道它到底是查询写错了、网页选错了、还是读完没理解。

那不用 outcome reward，训个 on-policy 的细粒度 teacher 让学生逐 token 跟着学？想法很美好，但 agent 任务里"每一步"的质量都很难稳定——老师自己在没见过的工具状态上也会犯迷糊，**那层蒸馏监督本身就不稳**。

这篇论文（[arXiv:2607.07820](https://arxiv.org/abs/2607.07820)）的作者想清楚了一件事：**既然问题在"环境"上——环境不可控、不可复现、不可验证，那为什么不先把环境做"对"，这是关键**。于是他们搭了个叫 DeepSearch-World 的离线维基百科沙盒，配合 420K 多跳问答任务，把"搜索-读网页"这件事完全变成可复现的确定性函数。模型每查一次、每读一页，系统都能精确告诉它"这一步有没有真的找到目标实体"。

然后在这个环境上跑 DeepSearch-Evolve——一个"自蒸馏"循环：学生模型自己跑、自己错、自己被环境打分，成功的轨迹转成 ReAct 训练数据再喂回给自己。11 轮下来，一个 9B 的开源小模型（基于 Qwen3.5-9B）在 BrowseComp 上从基线的 7.4% 干到了 31.2%，跟一堆用强模型轨迹训出来的开源 agent 打得有来有回。

**一句话判断**：这不是某个 RL trick 的微创新，而是一套"环境-监督-训练"三者打通的工程范式。对做 deep research / agent 的同行来说，值得细看它怎么把"环境确定性"做成 agent 训练的基础设施。

---

## 论文信息

- **标题**：DeepSearch-World: Self-Distillation for Deep Search Agents in a Verifiable Environment
- **作者**：Xinyu Geng, Xuanhua He, Sixiang Chen, Yanjing Xiao, Fan Zhang, Shijue Huang, Haitao Mi, Zhenwen Liang, Tianqing Fang, Yi R. Fung
- **机构**：HKUST、Tencent、HKUST(GZ)
- **arXiv**：[2607.07820](https://arxiv.org/abs/2607.07820)
- **v1**：2026 年 7 月 8 日
- **v2**：2026 年 7 月 13 日

---

## 核心摘要

- **痛点**：现有 web agent 自进化路线有三条死胡同——SFT 受限于 backbone + 轨迹多样性，RL 的稀疏 outcome reward 在长程任务上信号太弱，OPSD（on-policy self-distillation）的细粒度 teacher 监督在不可控的 web 环境里本身就不稳。
- **方案**：在确定性的离线维基百科环境（DeepSearch-World，420K QA + 1000 万实体）上，搭建"脚手架教师"（Plan-Act-End 三阶段 + 工作记忆 + 接地反思）生成高质量轨迹，筛选后转成 ReAct 训练数据，循环 11 轮 self-distill。
- **效果**：DeepSearch-World-9B 在 BrowseComp 31.2%、GAIA 61.5%、HotpotQA 93.4%，超过 Marco-DR、WebSailor、WebExplorer、ASearcher 等同期开源 9B–30B 级 agent，不依赖任何强模型蒸馏轨迹。
- **真实定位**：不是单点 RL 改进，是把"agent 训练环境"当成基础设施重做了一遍。9B 模型能追平 30B+ 量级的开源方案，关键不在参数规模，而在环境给的过程级监督信号。

---

## 问题：为什么 web agent 这么难训

在我自己折腾 web agent 的经历里，最让人挫败的就是：**模型明明能"答对"，但你看它的中间过程会发现很多无效搜索**——同义词翻来覆去、点开网页不读关键段落、点错了页面不肯回头。这些"过程级"的次优行为，单看 outcome reward 完全是"答对=好"，根本反馈不出来。

论文里把这件事拆得很清楚。我用一张图把三条路线的痛点对照看一下：

![图1：自进化范式对比](https://www.mulanai.com/fs/files/0722_602c59fc_opening.jpg)

*图1：三种自进化范式的对比。SFT 受限于 backbone 能力上限，RL 只能给稀疏 final reward，OPSD 在 web 这种非平稳环境里 teacher 自己都不稳。本文方案（绿色）：把环境做成可验证的，让"过程级"信号自然出现。*

对照来看，论文这条路的创新点其实可以一句话总结：**与其费劲在 reward 设计上做文章，不如把环境本身做对**。但说起来容易，做起来涉及三个问题：

1. 环境怎么搭才"够真"又"可控"？
2. 教师轨迹怎么生成才能既"质量高"又"可复用"？
3. 怎么把这些轨迹转成学生能直接学的 ReAct 数据？

下面分别看。

---

## 方法一：DeepSearch-World，把维基百科做成"可控沙盒"

### 数据怎么造

第一步是先有可验证的题。论文从维基百科的"超链接图"出发，按 entity-level random walks 采样 4-跳的实体链，比如 A→B→C→D，每跳是个真实的维基百科实体跳转。然后让 LLM（Gemini-3-Pro）围绕这条链生成多跳问题，并对显式实体做"特征模糊化"——把"Leonardo DiCaprio"这种直接的人名替换成间接描述，强迫 agent 必须查资料才能找到答案。

最终搞出 **420K 个多跳 QA**，外加一个 377 题的 DeepSearch-Val 验证集（每题至少三个专家标注答案，且能在离线语料里查得到）。

### 环境怎么"确定性"

光有题不够，还要让环境里的搜索/阅读行为可复现。DeepSearch-World 只暴露两个工具：

| 工具 | 接口 | 后端 |
|------|------|------|
| `web_search_wiki(query)` | 输入自然语言查询 → 返回 top-k（k=5）{title, snippet, URL} | Pyserini + Lucene BM25 索引 |
| `visit_wiki(url)` | 输入 URL → 返回完整文章正文 | SQLite 偏移索引 |

这两个工具的 schema 跟真实 web 搜索/浏览完全对齐（搜出来是标题+摘要+链接，点进去才是全文），所以训好的 agent 部署时直接换成真工具（SerpAPI + Jina）就完事了，不存在"训练-部署接口漂移"的问题。

### 接地反思：怎么给过程信号

这是我觉得最巧的部分。每道题在环境里都维护一个"目标实体集合" $T = \{e_1, ..., e_H\}$。agent 每调用一次工具，环境就用 entity-level matching 判定这次检索"是否真的命中了某个还没解决的子目标"——命中了就把这个实体从 todo 挪到 completed，失败了再分级给提示：

- 第一次失败：通用提示（"query 写得太宽/太窄"）
- 重复失败：揭示标准实体名或模糊描述作为更强引导

这样整个 rollout 轨迹里自然就出现了"search → fail → reflect → retry → succeed"的序列，而且反思信号是**基于 ground-truth 实体**的，不是 LLM judge 拍脑袋的，**完全可复现、零成本**。

整个环境和数据的关系看下面这张：

![图2：DeepSearch-World 和 DeepSearch-Evolve 总览](https://www.mulanai.com/fs/files/0722_0feac488_intro.jpg)

*图2：左边是 DeepSearch-World——维基百科知识图谱、420K 多跳 QA 任务、1000 万实体的离线语料，配合 search/visit 两个确定性工具。右边是 DeepSearch-Evolve 的四阶段循环：教师生成 → 拒采样+质量过滤 → scaffold 转 ReAct → 学生 SFT 更新，循环 11 轮。*

---

## 方法二：Scaffolded Teacher，三阶段生成高质量轨迹

有了可验证的环境，下一步是**谁来生成示范轨迹**。论文没用一个黑盒 LLM 直接出轨迹，而是搭了一个"三阶段脚手架教师"（Plan-Act-End），显式地把规划、记忆、纠错这些 agent 认知行为暴露出来：

![图3：Scaffolded Teacher Rollout 三阶段流程](https://www.mulanai.com/fs/files/0722_f5ba30b0_method1.jpg)

*图3：教师 rollout 的完整流程。Planning 阶段初始化 plan 和 working memory（completed/todo/information/experience 四栏）。Action Loop 阶段循环 think → tool call → observation → reflection → state update。Final Answer 阶段给最终答案。整个过程 working memory 在持续更新。*

展开说一下三个阶段：

**Plan**：拿到问题 $q$，先初始化进度状态 $s_0$，把任务拆成子目标，写进 todo 列表。

**Act**：每步选 tool call $a_t$，拿到环境 observation $o_t$，更新状态：
$$s_{t+1} = U(s_t, a_t, o_t, r_t)$$

其中 $r_t$ 是环境给的接地反思。成功时记录到 completed，失败时记到 experience 触发下轮重试。

**End**：当所有 $T$ 里实体都被找到或预算耗尽，生成最终答案——只允许基于已验证 evidence 写，不能瞎编。

下面是一个完整的轨迹例子，注意看它怎么从"模糊多跳问题"逐步搜出所有线索，并且中途失败了一次还能 recover：

![图4：Scaffold 教师的多跳推理轨迹示例](https://www.mulanai.com/fs/files/0722_2af8edeb_case1.jpg)

*图4：一个真实的多跳推理轨迹。题目是"找出某个坐标"，涉及 animal→flag→climate→capital→artwork→movement→place→coordinates 八个线索。教师按"国家→艺术运动→同运动艺术家→以其命名的地点→坐标"的链条逐步搜，每步都被环境验证。中间"Baroque crater"搜不到，触发接地反思后改搜"Rembrandt named place coordinates"，最终拿到答案 32.89°S, 272.13°W。整个 working memory 同步更新。*

看到这里你可能会问：**脚手架教师明明更"啰嗦"、更多 prompt 包装，训出来的学生怎么办**？这是下面要解决的。

---

## 方法三：Scaffold-to-ReAct Distillation，把脚手架塞回标准 ReAct

论文的解法很干脆——**脚手架只是教师用的训练轮，学生上线时要的是标准 ReAct**。所以每个成功的 scaffold 轨迹都要过一次"翻译"：

把教师轨迹里 $t$ 步对应的 think 块构造为：
$$\text{think}_t = P_t \oplus R_t \oplus A_t$$

- $P_t$：从当前进度状态 $s_t$ 改写（规划+记忆+已完成子目标）
- $R_t$：把环境反思 $r_t$ 改写成"自我纠错"语（mask 掉 ground-truth 实体名避免答案泄露，但保留反思结构）
- $A_t$：保留这一步的本地 rationale 和 tool call

转换后这条轨迹就是个标准 ReAct 序列（think → action → observation → next think → ...），可以直接喂给学生 SFT。

这一步消融数据显示得很硬：

| 配置 | DeepSearch-Val |
|------|---------------|
| Qwen3.5-9B-Instruct（基线） | 8.5 |
| 10K 转换轨迹的 vanilla SFT | 25.0 |
| 完整 pipeline | 31.9 |
| - 去掉 state internalization | 23.5 |
| - 去掉 reflection rewriting | 16.7 |
| - 两者都去掉 | 14.8 |

**看到没：reflection rewriting 是命门**。直接保留环境原始的 `[REFLECTION]` token 和消息会让 think 块分布严重畸变（毕竟这是给教师看的协议，不是给学生的语料）。这其实跟我们之前做 agent 训练的经验一致——**别图省事把"协议字段"直接当"自然语言"喂给学生，渲染器会把它当真**。

---

## 训练循环：Evolving SFT

整个训练是个异步循环：

1. 当前学生 $\pi_{\theta_R}$ 在 DeepSearch-World 里跑 10K 题的 scaffold rollout
2. 用拒采样（答案正确）+ 质量过滤（用 Qwen3.5-9B 做 LLM judge 去掉冗余/弱对齐/前后矛盾的轨迹）筛出 ~4K 高质量轨迹
3. 转换成 ReAct 训练数据
4. SFT 一轮 → 得到下一轮教师 $\pi_{\theta_{R+1}}$

论文一共跑了 11 轮，生成阶段用 1.6K 题的真实 web（SerpAPI + Jina）做 GRPO 微调弥补 offline-to-real 差距，importance sampling 衰减系数 $\gamma=0.5$ 平衡新旧数据。

这里有个**对比 SFT 和 OPSD** 的小细节值得拎出来说。SFT 是 offline + hard label：
$$L_{\text{SFT}} = \mathbb{E}_{\tilde{\tau}} \sum_{t=1}^T \text{KL}(\delta_{y_t} \| \pi_\theta(\cdot|x, y_{<t}))$$

OPSD 是 online + soft label：
$$L_{\text{OPSD}} = \mathbb{E}_{x,\hat{y}_{<t} \sim \pi_\theta} \sum_t \text{KL}(q_t \| \pi_{\theta}^t)$$

论文选了 SFT 路线，理由也很直接：**agent 任务里 student rollout 经常漂到 low-quality 工具状态，teacher 在这种状态上的 token 分布本身就是噪声，与其信它不如信自己跑出来的 verified hard label**。这是个挺反直觉的选择——大家都觉得 soft label 信息量更大，但前提是 teacher 得稳。

数据规模对最终效果的影响很直接（看图5）：

![图5：不同数据规模下 evolving SFT 的训练曲线](https://www.mulanai.com/fs/files/0722_7d83073d_training.jpg)

*图5：420K 大池子 vs 100K 小池子在 20 轮 evolving SFT 中的对比。左到右分别是验证集分数、格式错误率、工具成功率、实体命中率。420K 池子的目标分数更早爬到 ~40 平台，格式错误压到 1% 以下，工具成功率比 100K 高 8 个点。100K 因为题目多样性耗尽，10 轮以后就基本停滞了。*

这条曲线说明一件事：**self-distillation 不是"越多轮越好"，题目多样性耗尽就开始过拟合窄轨迹模式**。所以后面主实验选 11 轮就停了。

---

## 主实验：9B 干翻一群 30B+ 开源

主实验跑在 6 个 deep search 基准上，DeepSearch-World-9B（基于 Qwen3.5-9B 训 11 轮）的表现：

| 模型/Agent | BrowseComp | BrowseComp-ZH | HLE | GAIA | xbench | HotpotQA |
|------------|-----------|---------------|-----|------|--------|----------|
| OpenAI Deep Research | 51.5 | 42.9 | 26.6 | 67.4 | - | - |
| OpenAI-o3 | 49.7 | 58.1 | 20.2 | 70.5 | 65.0 | - |
| R1-Searcher | 1.0 | - | 5.4 | 8.3 | - | 62.4 |
| Search-R1 | 0.4 | - | 13.0 | 18.7 | - | 63.2 |
| ASearcher | 3.2 | - | 13.8 | 22.1 | - | 71.0 |
| WebSailor | 6.7 | 14.2 | 12.8 | 37.9 | 34.3 | 92.8 |
| WebExplorer | 15.7 | 32.0 | 17.3 | 50.0 | 53.7 | - |
| OffSeeker-DPO | 12.8 | 26.6 | 17.6 | 51.5 | 48.0 | - |
| Marco-DR | 31.4 | 47.1 | - | 69.9 | 42.0 | - |
| MiroThinker-v1.0 | 31.1 | 40.2 | 21.5 | 66.4 | 34.0 | - |
| **Qwen3.5-9B-Instruct** | 7.4 | 13.5 | 16.7 | 23.9 | 20.0 | 45.3 |
| **DeepSearch-World-9B** | **31.2** | **36.4** | **25.7** | **61.5** | **49.0** | **93.4** |
| Δ vs Qwen3.5-9B | 提升 **23.8** | 提升 **22.9** | 提升 **9.0** | 提升 **37.6** | 提升 **29.0** | 提升 **48.1** |

几个值得划重点的观察：

1. **9B 干到 BrowseComp 31.2%**：直接看齐 Marco-DR（31.4%）和 MiroThinker-v1.0（31.1%），后两个都是用了强模型合成轨迹或多智能体蒸馏的。DeepSearch-World-9B **完全用自己跑出来的 verified 轨迹**，在 9B 这个量级上做到这一步，数据-环境的功劳远大于模型本身。
2. **GAIA 61.5%**：从 23.9% 涨到 61.5%，+37.6 个点，这是六项里提升最大的。
3. **BrowseComp-ZH 比基线涨 22.9 但绝对值偏低**：论文也承认，训练只用英文轨迹，中文能力来自 Qwen3.5 backbone 的跨语言迁移，没专门训。**这是个真实的"数据集偏差"陷阱**，值得做中文场景的同行注意。
4. **HLE 25.7% 已经超过 OpenAI-o3 的 20.2%**：HLE 是专家级学术问题，纯工具调用是解不了的，能超过 o3 说明这套范式对推理-检索-综合全链路有真提升。

---

## 工具使用行为：不是分高，是"用得对"

光看分数涨不够说服力，更有意思的是训完之后模型的工具使用模式变了多少：

![图6：训练前后工具使用行为对比](https://www.mulanai.com/fs/files/0722_ab2acbb0_tool_usa.jpg)

*图6：Qwen3.5-9B-Instruct 和 DeepSearch-World-9B 在 DeepSearch-Val 上的工具使用行为对比。平均交互轮数从 4.7 → 18.0（接近 4 倍），search 调用从 3.8 → 12.6，visit 调用从 0.9 → 5.4（涨了 6 倍），LLM judge 打分的高级能力（规划/记忆/纠错/检索综合）从 19% → 70%。*

这个对比让我很有共鸣——**很多 9B 模型的"分数天花板"不是能力不够，是它**主动放弃**了**。你看 Qwen3.5-9B-Instruct 跑 4.7 轮就交卷了，visit 只调 0.9 次（基本没读网页），这是典型的"懒"——RL 训出来容易"偷懒"拿部分分就停。DeepSearch-World-9B 跑 18 轮，visit 5.4 次，是真的在多跳搜证据。

**反思信号在这里起了关键作用**：环境在每一步告诉它"你还没找到第 3 个目标实体"——这种过程级 grounding 让模型没法"差不多就交"。

---

## 消融实验

论文做了两类消融。

### 1. 拒采样 + 质量过滤的必要性（Table 2）

| 过滤策略 | SearchQA |
|---------|----------|
| 两者都去掉 | 46.4 |
| 仅 Quality Filter | 48.1 |
| 仅 Rejection Sampling | 54.9 |
| **RS + QF**（完整方案） | **58.2** |
| 对比 OPSD-style (SDAR) | 49.0 |
| 对比 Skill-SD | 47.8 |

拒采样（答案对不对）是最大头，单独贡献 +8.5 个点；质量过滤（轨迹是否"漂亮"）单独 +1.7 个点；两者合起来 +11.8 个点。**这个结果再次验证——"verifiable environment 的核心价值是过程级信号"，轨迹质量是锦上添花，答案正确才是命门**

### 2. Scaffold-to-ReAct 转换的细节（Table 3，前面贴过）

reflection rewriting 是核心，state internalization 是次要但有正向贡献。

---

## 我的判断：这套范式到底强在哪、坑在哪

### 亮点

1. **环境-数据-训练三件套打通了**：不是某一项创新，是工程范式。420K 题目 + 1000 万实体离线语料 + 确定性工具函数 + scaffold 教师 + evolving SFT——每一环都是"上一环出问题才这么设计"的。
2. **过程级监督来自环境，不来自 judge**：用实体级 matching 判定"这次检索有没有命中目标子实体"，比用 GPT-4o 判分稳定得多、便宜得多。这是它能在 9B 上追平 30B+ 的关键杠杆。
3. **数据多样性就是命门**：420K vs 100K 池子差了 10 个点，说明 self-distillation 的天花板不在训练轮数，而在"有没有没见过的题"。

### 限制与诚实交代

1. **离线环境的泛化性未知**：训练全在维基百科里跑，真部署靠的是 1.6K 真实 web 数据的 GRPO 微调"接缝"。**这意味着换成非英文维基以外的领域（金融、医疗、内部知识库），整套范式需要重新搭环境**，不可能直接复用。
2. **SFT 而非 RL 的代价**：作者自己承认"RL-style / OPSD-based updates may further improve generalization"。SFT 是"对 verified 数据 hard label 学习"，一旦遇到分布外问题，泛化能力受限于监督数据的多样性。Marco-DR、WebExplorer 这些用大规模 RL 训的 30B+ 模型在 xbench、GAIA 上仍然领先。
3. **BrowseComp-ZH 这种跨语言场景掉队**：单语训练对中文/小语种场景的天花板可见，做面向中文用户的 deep research 产品时这是个真问题。
4. **scaffold 教师的成本**：每个题要跑 plan + 多轮 act + state update + 反思，单题成本是普通 ReAct 的 2-3 倍。论文 11 轮 × 10K 题 = 11 万次 rollout + 1.6K 真实题 GRPO，**算力门槛不算低**。
5. **环境"真实性"的代价**：维基百科的世界是"静态、可枚举"的，跟真实 web 的"动态、含噪声"有 gap。这就是为什么论文末尾还要上真实工具做 GRPO 缝补。

### 跟同期工作的相对位置

- vs **WebSailor / WebExplorer**（同期用真实 web + 大量 RL）：DeepSearch-World 胜在"数据可验证、过程可监督"，但 RL 信号弱；
- vs **Marco-DR / MiroThinker**（同期用强模型合成轨迹）：DeepSearch-World 胜在"自给自足、不依赖强模型"，但绝对分数有上限；
- vs **Search-R1 / ZeroSearch**（同期 RL 路线）：DeepSearch-World 胜在"过程级信号"vs"outcome reward"；
- vs **OPSD 系列（SDAR、Skill-SD）**：在 9B 量级上 evolving SFT 优势明显（58.2 vs 49.0/47.8），但这是因为 OPSD 在"非平稳 agent 环境"里的 teacher 不稳——**这恰好是 DeepSearch-World 用"环境确定性"换来的红利**。

---

## 给做 agent 的同行：哪些思路可以照搬

如果你也在搭 deep research / web agent 系统，这篇论文有几个直接可用的工程启发：

1. **先评估你的环境——能不能做"可验证"，这是第一关**。如果你的工具调用是确定性的（同样的 query 永远返回同样结果），entity-level matching 几乎零成本；如果不是，考虑给关键中间步骤加个"轻量验证器"。
2. **教师和学生用不同 prompt 协议是正常的，关键是 conversion 步骤要把协议字段翻译成自然语言**。reflection rewriting 这条消融告诉我们——直接喂 `[REFLECTION]` token 进去会污染 think 分布。
3. **题目池大小比训练轮数更影响 self-distillation 上限**。420K vs 100K 差了 10 个点这件事比"训几轮"重要得多。
4. **GRPO 后处理补 offline-to-real gap**。这是务实做法，别指望离线环境能完全替代真实交互。

---

## 结语

DeepSearch-World 这篇最让我意外的不是某个具体数字，而是它把"环境确定性"做成了 agent 训练的基础设施层级——把维基百科这种"封闭、可枚举、零噪声"的语料变成 agent 训练场，配合 entity-level verification 给过程级监督，然后用 scaffold 教师 + ReAct 蒸馏把"显式规划"压回"隐式能力"。

对工程界来说，这意味着 9B 级别的模型在受控环境内完全可以追平 30B+——但代价是"环境本身"是核心资产，换场景就要重建。

对研究界来说，它给"verifiable environment + evolving SFT"这个范式开了一扇门。下一步有意思的方向我猜是：**把同样的思路搬到代码生成、SQL 生成、甚至 GUI agent 上**——这些场景的"中间步骤"也是可以验证的。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
