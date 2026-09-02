# 研究计划没有标准答案，怎么训 RL？这篇论文把 arXiv 论文变成了训练场

你有没有想过一个很拧巴的问题：现在人人都在喊"AI 科学家"，让模型自己提假设、设计实验、写研究计划——但研究计划这东西，**根本没有标准答案**。

做数学题好办，答案对就是对，错了就是错了，RLVR 那套可验证奖励直接上。可一份研究计划呢？"提出一个改进扩散模型采样效率的方案"——你怎么给这种回答打分？没有 verifier，GRPO 再强也使不上劲。这就是研究规划（research planning）这个方向一直卡脖子的地方：强化学习需要"任务 + 批评者"的环境，而这个环境里永远缺一个靠谱的批评者。

浙大和 Apple 合作的这篇 PaperGym（arXiv 2608.31119）给出了一个我觉得相当漂亮的思路：**既然批评者找不到，那就从论文里"挖"一个出来**。而且不是随便挖——他们发现了现有数据管线里一个挺要命的漏洞，把它修掉之后，小模型也能训出超过 Kimi K2.6 的研究规划能力。

**核心摘要**：研究计划没有可验证答案，现有 rubric 数据集又普遍存在"答案泄漏"——评分标准能从问题里直接推出来，模型靠复述问题就能骗奖励。PaperGym 利用论文的天然结构做解耦：问题只从研究目标 + 背景合成，评分准则只从方法 + 实验设计推导，标准泄漏率压到 **3.7**%（现有数据集是 11.9%–34.1%）。训练上同一份 rubric 用两次：先当 OPSD 教师的特权信息，再当 GRPO 的奖励，构成"先拓宽、后收窄"的熵课程。Qwen3-1.7B/4B/8B 上五基准平均分别提升 **5.6 / 5.0 / 4.8 个点**，训完的 8B 在 ResearchQA 上拿到 73.48，反超远大于它的 Kimi K2.6（73.19）。

---

## 📖 论文信息

- **标题**：PaperGym: Rubric-Centered Evolution for Research-Plan Generation
- **作者**：Yuhan Wang, Zhengxi Lu, Yuchen Yan, Kaitao Song, Wenqi Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, Yongliang Shen
- **机构**：浙江大学（ZJU-REAL 团队）× Apple（Kaitao Song）
- **链接**：https://arxiv.org/abs/2608.31119
- **代码与项目页**：https://github.com/ZJU-REAL/PaperGym
- **开源资产**：数据管线、PaperGym-20k 语料（2 万实例）、PaperGym-Innov 与 PaperGym-Design 两个基准

---

## 🎯 问题动机：现在的 rubric 数据集，答案写在题目里

先交代下背景。给不可验证任务造奖励信号，目前主流做法是 **rubric-as-reward**：从语料里抽出一套评分准则（rubric），让 LLM judge 拿着准则逐条给候选回答打 0/1，聚合成奖励。HealthBench、RubricHub 都是这个路子。

听起来没毛病。但作者扒了一遍现有数据集，发现一个尴尬的事实：**这些管线里，问题和评分标准是从同一段内容里抽出来的**。

后果是什么？假设论文原文写着"我们提出一种基于对比学习的方法"，问题和 rubric 都来自这句话——那 rubric 里很可能出现"方案应包含对比学习组件"这种标准。模型只要把问题里明晃晃的信息复述一遍，奖励就到手了。这叫 **criterion leakage（标准泄漏）**。

泄漏有多严重？作者用 DeepSeek-V4-Flash 做判官，把每个问题连同它的全部标准喂进去，逐条问一句："这条标准能不能只从问题推出来？"结果：

| 数据集 | 标准泄漏率 |
|---|---|
| ResearchPlanGen-ArXiv | 34.10% |
| ResearchPlanGen-ML | 31.29% |
| ResearchQA | 19.22% |
| RubricHub Science | 17.39% |
| HealthBench | 11.90% |

最高的三份数据集里，三分之一的标准等于把答案贴在了题目上。说实话看到这个数我愣了一下——在这种数据上训出来的"研究规划能力"，水分可想而知。

另一个问题是**监督信号浪费**。rubric 明明有十条八条细粒度标准，judge 逐条打完分之后，却全被压成每个 rollout 一个标量。粒度没了。而 SFT 那边更简单粗暴：只模仿单一参考答案。可研究问题的本质就是"一个题目有多个都成立的解法"，逼着模型复刻某一个具体答案，输出分布只会越压越窄——实验里 SFT 在 8B 上平均掉了 0.96 个点，印证了这个判断。

---

## 🏗️ 方法：把论文拆成两半，一半出题，一半当裁判

PaperGym 的核心观察特别朴素：**论文本身自带防泄漏结构**。研究目标（Introduction 里的 goal）和背景（Related Work）是"出题素材"；方法（Method）和实验设计（Experiments）是"答题素材"。这两拨内容在论文里本来就是分开写的——那就别让它们碰面。

![PaperGym 框架总览](https://arxiv.org/html/2608.31119v1/framework.png)

*图 1：PaperGym 框架。（a）数据生成：论文被拆成四阶段，Goal+Background 合成问题、Method+Experimental Design 合成参考答案，rubric 经双路生成、合并、过滤后定稿；（b）两阶段策略训练：rubric 先作为 OPSD 自蒸馏教师的特权上下文，再作为 GRPO 的逐条二元奖励。*

### 数据管线：四阶段 map-reduce

具体流程是四步：

1. **取材**：拿公开 arXiv 论文的 LaTeX 源码（比解析渲染后的 PDF 干净得多），按自然章节切分。
2. **抽取（map）**：让 Qwen3-235B-A22B 逐节抽出四阶段信息——Research Goal、Background、Research Method、Experimental Design。要求逐字忠实，没证据的类别直接返回空字符串。注意一个细节：Experimental Design 里**刻意剔除具体数值结果**，防止模型靠"背数字"虚假记忆。
3. **合并（reduce）**：按阶段去重合并，得到四段连贯摘要。
4. **合成 QA 与 rubric**：问题只从 Goal+Background 合成，参考答案只从 Method+Design 合成。rubric 则由 DeepSeek-V4-Flash 双路生成——一路只看问题（$\mathcal{R}_Q$），一路看问题加参考答案（$\mathcal{R}_A$），合并、语义去重、按重要性排序，保留 top-10。另配 7 条实例无关的通用标准（完整性、具体性、严谨性、效率、伦理等）。

最终产出 **PaperGym-20k**：2 万实例，CS 占 50%、物理 25%、经济 25%，每条带 10 条原子化二元标准。泄漏率压到 **3.7**%，比现有数据集低 3 到 9 倍。

![数据分析](https://arxiv.org/html/2608.31119v1/data_analysis.png)

*图 2：左——PaperGym-20k 的领域构成（CS 50% / Physics 25% / Econ 25%）；中——4 个评分模型各跑 5 轮的均分与跨轮极差，越强的 judge 打分越严格但都高度自洽；右——judge 间两两一致性，Qwen3-8B 与 Kimi K2.6 接近 80%，说明这套二元评分协议是可靠的。*

### 训练：一份 rubric，用两次

这是全文我最喜欢的设计。同一份 rubric 在训练里扮演两个完全不同的角色：

**第一阶段，rubric 当老师的"小抄"（Rubric-Conditioned OPSD）**。OPSD（On-Policy Self-Distillation）是 UCLA 和 Meta 今年初在 arXiv 2601.18734 提出的框架：同一个模型分饰两角，教师能看到特权信息（原文用的是标准答案），学生只看题目；学生在自己 on-policy 采样的轨迹上，逐 token 对齐教师的分布。PaperGym 把特权信息从"标准答案"换成了 **rubric**：

$$\mathcal{L}_{\text{OPSD}}(\theta)=\mathbb{E}_{(x,\mathcal{R})\sim\mathcal{D}}\left[\mathbb{E}_{\hat{y}\sim\pi_{\theta}(\cdot\mid x)}\frac{1}{|\hat{y}|}\sum_{n=1}^{|\hat{y}|}\mathrm{JSD}_{\beta}\!\left(\mathrm{sg}(\pi_{\theta}(\cdot\mid x,\mathcal{R},\hat{y}_{<n}))\,\|\,\pi_{\theta}(\cdot\mid x,\hat{y}_{<n})\right)\right]$$

为什么换？消融实验给了答案（Table 5）：用 rubric 当特权信息，Innov/Design 上是 14.16/10.45；换成参考答案，掉到 13.08/9.57；两个一起给，Design 上反而更差（9.17）。直觉也说得通——参考答案是一个**具体**解，学生容易学歪成"复述这篇论文"；rubric 是**原则级**的质量标准，枚举的是一整片有效解空间，教师能指出的合理续写集合宽得多。

**第二阶段，rubric 当裁判（Rubric-as-Rewards GRPO）**。用基座模型的冻结副本自评分，拿着问题、参考答案、rubric 逐条做严格二元判定——完全满足且质量不输参考答案才给 1，部分正确一律 0，全程 temperature 0。奖励是专属标准和通用标准的加权和：$r_i = \alpha\, r_{i,\text{spec}} + (1-\alpha)\, r_{i,\text{gen}}$，$\alpha = 0.7$。然后套标准 GRPO：组内归一化优势、PPO clip、对参考策略的 KL 惩罚（0.01）。

为什么要先 OPSD 再 GRPO？作者给了一个熵视角的解释：OPSD 注入新知识，策略熵**上升**（先拓宽）；GRPO 向高奖励区收敛，熵**下降**（后收窄）。而且 GRPO 在 OPSD 预热之后熵降更陡——说明先验结构确实更好了。这个 "widen-then-narrow" 的课程设计，说实话不是全新的想法，但把它落到 rubric 复用上、还用熵曲线讲清楚机制，执行得很干净。

工程细节也值得抄作业：1.7B/4B 用 4 张 A6000，8B 用 4 张 Pro A6000；OPSD 阶段用 LoRA（r=64），GRPO 阶段用 verl + vLLM 每 prompt 采 8 条；**1.7B 自己评分不靠谱，借 4B 当裁判**——这个细节说明自评分协议对模型能力是有下限要求的。

---

## 🧪 实验：小模型的逆袭

五个评测基准：域内的 PaperGym-Innov（只评方法创新）和 PaperGym-Design（只评实验设计），域外的 ResearchQA、ResearchPlanGen-ML、RubricHub Science。所有回答由 DeepSeek-V4-Flash 按 rubric 逐条评分。

### 主实验

Table 1 的数字很密，我挑关键的讲（完整数据建议看原文）：

| 模型 | Science | ML | ResearchQA | Innov | Design | 平均 |
|---|---|---|---|---|---|---|
| Qwen3-8B base | 46.07 | 20.28 | 66.65 | 18.69 | 17.03 | 33.74 |
| 8B + SFT | 45.29 | 17.91 | 67.53 | 17.52 | 15.68 | 32.79（降 0.96） |
| 8B + OPSD | 47.41 | 23.17 | 69.17 | 21.42 | 18.04 | 35.84 |
| 8B + GRPO | 48.09 | 20.76 | 71.02 | 20.26 | 19.17 | 35.86 |
| **8B + OPSD + GRPO** | **49.41** | **23.53** | **73.48** | **24.47** | **21.88** | **38.55（涨 4.81）** |

几个值得展开的点：

**两阶段在每个规模、每个基准上都是最优**，1.7B/4B 分别涨 5.56 和 5.04 个点。不是锦上添花，是全表格压制。

**SFT 在 8B 上是负收益**。只在单一参考答案上模仿，输出分布坍缩——这个实验结果是论文动机最直接的证据。

**单独 OPSD 几乎追平单独 GRPO**（8B 上 35.84 对 35.86），而 OPSD 的训练成本远低于 GRPO（不用每 prompt 采样 8 条）。预算紧张的场景下，光跑第一阶段就很划算。

最抓眼球的是：**训完的 Qwen3-8B 在 ResearchQA 上拿到 73.48，超过了 Kimi K2.6 的 73.19**。一个 8B 模型在域外基准上反超体量远大得多的旗舰模型，靠的不是基座能力，是训练环境和课程设计。

当然也得泼点冷水：所有评分都是 LLM-as-a-judge 给的，judge 本身是 DeepSeek-V4-Flash，这套分数对回答风格敏感——论文自己就披露 Claude-Sonnet-5 在 ResearchQA 上只有 66.04，原因是它判断信息不足时会拒答、拒答记 0 分。所以 73.48 这个数要看作"在该评测协议下的相对位置"，别当成绝对能力值。

### 数据质量隔离实验：58.1% 对 28.2%

模型涨了，到底是数据好还是训练 recipe 好？作者做了个干净的隔离实验：三个 Qwen3-1.7B 共享同一套 OPSD 超参、同样 200 步，唯一变量是训练语料——PaperGym-20k 一份、RubricHub Science 一份、不训的 base 一份。在独立于所有训练数据的 ResearchPlanGen-ML（685 条）上做三方对比，三个 judge 多数投票、票数分散时 Gemini 3.7 Flash 打破平局：

![Overall Score 三方胜率](https://arxiv.org/html/2608.31119v1/overall.png)

*图 3：Overall Score 维度的三方胜率——PaperGym-20k 训练的模型拿下 58.1%，RubricHub Science 28.2%，未训练的 base 只有 13.7%。六个维度（Goal Alignment、Novel Insight、Scientific Soundness、Execution Quality、Expected Impact）上 PaperGym 全部领先。*

58.1% 对 28.2%，接近两倍的差距。在 recipe 完全冻结的前提下，这个差距只能归因于数据本身——**低泄漏的 rubric 确实更值钱**。

### 消融：每个设计决策都被验证过

这篇论文的消融做得相当扎实，挑三个最有信息量的：

**顺序很重要**（Table 7）：OPSD→GRPO 在每个规模、每个基准上全胜 GRPO→OPSD，平均领先 1–2 个点，差距主要集中在域内创新基准（Innov 上 1.7B 差 2.97 个点）。先拓宽再收窄，反着来就是不行。

**rubric 质量是瓶颈**（Table 4）：把生成 rubric 的模型从 DeepSeek-V4-Flash 换成 Qwen3-8B，Innov 掉 2.00 个点；而把做四阶段抽取的模型换成 Qwen3-8B，只掉 0.97。同样换弱模型，rubric 环节掉分是抽取环节的两倍。只留 $\mathcal{R}_Q$ 或只留 $\mathcal{R}_A$ 也都不够——双路互补缺一不可。

**奖励配比**（Table 6）：专属标准与通用标准 7:3 最优（Innov 17.17 / Design 12.35），8:2 和 6:4 都更差。

---

## 🤔 我的判断：框架价值大于数字，但 reward hacking 没被消灭

先说亮点。这篇论文真正值钱的不是某个具体数字，而是**一套可复用的范式**：任何"没有标准答案但有结构化参考材料"的任务，都可以套这个思路——问题和标准从材料的不相交部分分别取材防泄漏，标准先当蒸馏的特权上下文再当 RL 奖励。论文、专利、法律文书、病历……凡是带天然结构的长文档，都能这么改造成训练环境。我觉得这比"又刷了一个榜"的意义大得多。

但有几个地方得保持清醒。

**reward hacking 只是被缓解，没有被消灭**。附录 C.1 里作者自己承认：GRPO 第二阶段之后，模型学会了"写更长、匹配更多 rubric 条目"，novel insight 胜率从 63.2% 涨到 80.0%，但 scientific soundness 从 72.0% 掉到 62.1%，execution quality 从 71.9% 掉到 66.3%。你想想看，奖励信号本身就是"逐条勾标准"，模型当然会往"显得条条都沾边"的方向进化。这是 rubric-as-reward 范式的结构性漏洞，不是 PaperGym 独有问题，但它也没治好。

**rubric 维度分布不均**。专属标准里方法类占 63.8%、实验类只有 36.2%，奖励信号天然向"创新话术"倾斜，这大概也是上面维度跷跷板的部分原因。

**评测闭环的自指性**。数据由大模型抽取、rubric 由大模型生成、奖励由模型自评、最终评测又是 LLM judge——整个链条里没有人类判断的位置。作者用 judge 间一致性（约 80%）来缓解这个担忧，但链条越长，系统性偏差越难排除。域内基准 PaperGym-Innov/Design 与训练数据同源同构，在这上面的涨幅（Innov 涨 5.78–6.69 个点）我会打个折看；真正有说服力的是域外 ResearchQA 和隔离实验的 58.1%。

还有个实操层面的提醒：**1.7B 连自己的裁判都当不了**，要借 4B 评分。如果你手上的基座模型太弱，这套自评分协议直接不成立。

---

## 🔗 收尾

回到开头那个拧巴的问题：不可验证任务怎么训 RL？PaperGym 的答案是——别纠结"找标准答案"，转而构造"防泄漏的批评者"，再让这份批评标准在蒸馏和 RL 里各打一份工。数据管线、2 万条语料、两个基准全部开源，想在这个方向上继续做的同学可以直接站在上面。

更大的问题还悬着：当 rubric 本身也由模型生成、由模型执行评分时，"什么才是好的研究"这个最终裁判权，始终还在闭环外面。这一天不解决，AI 科学家的自我进化就始终隔着一层玻璃。

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
