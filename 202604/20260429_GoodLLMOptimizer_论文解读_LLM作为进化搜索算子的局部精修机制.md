# 强LLM不一定是好"优化器"：决定胜负的不是参数量，而是"局部精修"的能力

## 核心摘要

把LLM塞进进化算法的循环里、让它当 mutation 算子来跑优化，这两年成了 AlphaEvolve、AI Scientist、PromptBreeder 这类系统的标准做法。但有个尴尬的现象一直没人系统讲清楚：**同样的 evolutionary loop、同样的 fitness function、同样的初始种群**，换个 LLM 进去跑，最终 fitness 能差出一截。这是 base model 能力的差异？还是更隐蔽的"搜索行为"差异？

这篇来自 Univ. Grenoble Alpes 的论文，扔了 15 个模型、8 个任务、72K+ 候选解进去做了个大规模 trajectory 分析，结论挺反直觉：**zero-shot 能力只能解释一部分性能差异**（R²=0.103），真正卡在哪里的，是模型作为"局部精修器（local refiner）"的稳定性。强 optimizer 不是那种偶尔来个大跃进的，而是**能持续不断小步改进、把搜索逐渐收敛到语义空间的局部高地**。更扎心的是——**Mistral-24B-Instruct 在性价比帕累托前沿上稳压 GPT-4o、DeepSeek-V3、Gemini-1.5-Pro**。

如果你正在选模型搭 evolutionary agent 系统，这篇值得花半小时读。

---

## 论文信息

- **标题**：What Makes an LLM a Good Optimizer? A Trajectory Analysis of LLM-Guided Evolutionary Search
- **作者**：Xinhao Zhang, Xi Chen, François Portet, Maxime Peyrard
- **机构**：Univ. Grenoble Alpes, CNRS, Grenoble INP, LIG, 38000 Grenoble, France
- **arXiv**：[2604.19440](https://arxiv.org/abs/2604.19440)
- **项目页**：[xinhao-zhang.github.io/traj_evo_search](https://xinhao-zhang.github.io/traj_evo_search/)
- **数据集**：[HuggingFace - LivevreXH/evo_llm_trajectories](https://huggingface.co/datasets/LivevreXH/evo_llm_trajectories)

---

## 为什么这个问题值得做

我之前在做 prompt optimization 的时候踩过一个坑——拿 GPT-4o 当 mutation 算子去 evolve prompt，跑 30 代之后 fitness 卡在某个值，怎么都上不去；换成 Mistral 7B，反而能多挤出几个点。当时下意识以为是 random seed 的事，重跑了一轮还是这样，就放下了，没深究。

后来看 AlphaEvolve、PromptBreeder、FunSearch 这一系列工作越铺越大，才意识到这个问题被绕过去了——**所有人都默认"更强的 LLM 当 optimizer 一定更好"，但其实没人去算过这个账**。

这篇论文做的事情就是把这个账算清楚：在严格控制的进化框架下（同样的 selection、同样的 mutation 接口、同样的 evaluation），15 个 LLM 在 8 个任务上跑 30 代，每代 10 个 offspring，重复两次，72,000+ API 调用、约 500 美刀的实验成本，把每条 trajectory 都存下来分析。

不是 benchmark 那种"谁分高谁牛"的玩法，而是看**搜索过程本身的几何形状**。

---

## 框架：LLM 当 mutation 算子的进化搜索

先把游戏规则讲清楚。整个 loop 走的是经典的进化算法：

![图1：LLM 驱动的进化搜索框架。左边是 across-generations 的种群迭代，右边是 within-generation 的"种群初始化 → LLM mutation → fitness 评估 → 选择"循环。底部是四类任务及其 genome 表示——Route Optimization 用 path、Equation Discovery 用 formula、Prompt Optimization 用 system prompt、Heuristic Design 用 priority function](https://arxiv.org/html/2604.19440v1/img/final_version_method.png)

*图1：方法总览。整个流程像一个"LLM 在不停看上一代的优胜方案、给出改进版"的循环。论文要回答的是：在这个循环里，什么样的 LLM 才是好算子？*

具体的设置：

- **Selection**：每代取 top-20% 作为 elite，按 fitness 加权采样作 parent
- **Mutation**：把 parent 当 context 喂给 LLM，让它生成 offspring（temperature=0.7）
- **Pool 更新**：去重后合并，超过容量 N 时只留 top-N

四个任务族覆盖了 LLM-guided 进化搜索常见的应用场景：

| 任务族 | 子任务 | Genome | Fitness |
|--------|--------|--------|---------|
| Route Optimization | TSP-30, TSP-60 | 城市排列 | $-L(\pi)$（路径长度的负值） |
| Prompt Optimization | SAMSum, ASSET | 自然语言指令 | ROUGE-L / SARI |
| Equation Discovery | Oscillator-1, Oscillator-2 | Python 函数 | $1 - \text{norm}(\text{MSE})$ |
| Heuristic Design | Bin Packing OR3, Weibull | priority function | $1/\text{bins}$ |

15 个模型横跨 6 个家族：OpenAI（GPT-4o, GPT-4o-mini, GPT-3.5-turbo）、Gemini（1.5-Pro, 1.5-Flash, Gemma-3n-4B）、Llama（3.1-70B/8B-Instruct, 3.2-3B/1B-Instruct）、DeepSeek-V3、Mistral 全家（7B, 24B, Large, Magistral-Small）。

这套实验设置算是相当扎实，没有"偷偷换 selection 规则"或者"挑对自己有利的 task"的嫌疑。

---

## 第一刀：zero-shot 能力到底解释了多少？

最朴素的假设：模型 zero-shot 能力越强，当 optimizer 也越强。这个相关性是存在的——

![图2：15 个 LLM 的 zero-shot 性能 vs. 进化后最终性能。线性拟合 r=0.860, p=3.95e-05](https://arxiv.org/html/2604.19440v1/img/zero_final_compare.png.png)

*图2：横轴是 zero-shot 平均 fitness，纵轴是 30 代进化后的最终 fitness。整体确实强相关，但散点离回归线的偏差不小——很多模型偏离回归线 5 个点以上。最显眼的是 Mistral-24B-Instruct——zero-shot 排在中间，进化后却跳到了 0.81 的最高点；而 GPT-4o、Gemini-1.5-Pro 这些 zero-shot 顶尖的，最终也就 0.75 左右。*

这就是论文要追的"残差"。同样在 zero-shot 0.4 附近聚成一堆的模型，进化完之后能差出一大截。zero-shot 解释力 R²=0.103，连 11% 都不到。

**说实话第一眼我有点意外**。我之前的直觉是 zero-shot 能力应该能解释 50% 以上——毕竟好的 mutation 至少得理解任务、得能产出 valid solution。结果连 1/4 都不到，剩下 90% 的方差是哪来的？

论文的回答是：**搜索过程本身的差异**。模型 A 和模型 B 在 zero-shot 上半斤八两，但 mutation 行为可以完全不一样——一个像精雕细琢的工匠，一个像撒网捞鱼。

举个具体例子：DeepSeek-V3 在 first-generation 的平均 fitness 是 56.9，全场最高，比 GPT-4o (46.5) 高了 10 个点；但跑完 30 代之后，DeepSeek-V3 只到 75.3，反而被 GPT-4o (77.4) 反超。**起跑领先 10 个点，全程被反超**——这就是"搜索行为差异"在做功的最清楚证据。Zero-shot 决定起点，但决定终点的是别的东西。

---

## 第二刀：novelty 真的是好东西吗？

这里就到了我个人最被打动的部分。

经典的进化算法理论里，novelty/diversity 是 exploration 的代理，被认为是好东西。如果搜索陷在局部最优，就需要更高的 novelty 来跳出去。LLM-guided evolution 的常识理解也是这样：mutation 不是随机的、而是被 LLM 的 prior 强烈塑形的，那 exploration 就更受限了，所以 **novelty 应该更重要**。

——论文的实验结果直接打脸。

![图3：OLS 回归各种 trajectory 描述符的预测能力。左边是标准化系数 + 95% CI，右边是 R²。Avg Novelty 和 Init Novelty 系数贴近 0、不显著；Breakthrough Rate (BR) 系数最大 (~0.5, p\lt0.001)；ZS+BR 联合解释力最强 (R²=0.246)](https://arxiv.org/html/2604.19440v1/img/nov_regre.png)

*图3：novelty 相关的两个指标（avg novelty、init novelty）系数是灰色的"ns"（not significant），贡献几乎为零。真正能预测最终 fitness 的，是 Breakthrough Rate——也就是"出现一次 best-so-far 改进"的频率。BR 单独的 R²=0.198，是 zero-shot 单独 R²=0.103 的近两倍。*

这个图我盯着看了好久。它其实在颠覆一个直觉：**做 LLM-guided 优化，不是要让模型"想得更野"，而是要让它"持续不断地小步前进"**。

什么是 breakthrough？论文定义得非常朴素：当前这一代里只要有任意一个 offspring 的 fitness 超过了**所有历史代**的最优，就算一次 breakthrough。Breakthrough Rate = breakthrough 数 / 总代数。

我的第一反应是怀疑：会不会是定义太宽？后面的混合效应回归（generation-level）进一步把这个故事讲实了——

![图4：generation-level 混合效应回归。左边是 concurrent（同代预测同代 breakthrough），右边是 lagged（t 代描述符预测 t+1 代 breakthrough）。H_fitness 显著负相关；Mean novelty × H_spatial 交互项也显著负相关](https://arxiv.org/html/2604.19440v1/img/mixed_effects_regression.png)

*图4：这是篇文章里我觉得最关键的统计证据。注意右图（lagged）：Mean novelty 单独的效应已经不显著了（系数贴近 0），但 **Mean novelty × H_spatial 的交互项**仍然显著负相关。翻译过来就是：novelty 本身没用，**只有当搜索保持局部化（low H_spatial）的时候，novelty 才会带来 breakthrough**。否则 novelty 只是在"瞎逛"。*

说到这块，我突然想起来之前看的一个 LAS landscape analysis 的工作（也在 related work 里被引了），跟这篇的结论是同一个方向——LLM-driven 的优化更像 "exploitation-heavy" 的过程，randomness 反而是负担。

这里也跟经典进化算法理论分歧得很彻底。Mitchell & Taylor (1999) 那本经典里，evolutionary search 的典型行为是"长 plateau + 偶发大跃进"——大部分时候在小修小补，偶尔一次成功的 mutation 把 fitness 推到新高。但在 LLM-guided 这套里，**好 trajectory 长得不像这样**——它是连续的小台阶累积，没有明显的"大跃进+长 plateau"模式。这是因为 LLM mutation 不是真正的随机，它本身就被 prior 引导着"往好的方向改"，所以"小步快走"是更自然的状态。

---

## 第三刀：trajectory 的几何长什么样？

光看数字不够，论文还在语义空间里把 trajectory 画了出来。这是我觉得最有说服力的可视化：

![图5：TSP-60 上两个模型 trajectory 的散点图。横轴是 normalized fitness，纵轴是 normalized novelty。颜色编码 generation。Gemini-1.5-Pro（左）的点云从高 novelty 逐步往右下角"沉降"，形成 sustained improvement；Mistral-7B-Instruct（右）的点云一直贴着上方 novelty=1 的线，没法把高 novelty 兑换成 fitness 提升](https://arxiv.org/html/2604.19440v1/img/explore_bad_good.png)

*图5：左边 Gemini-1.5-Pro 的 trajectory 像在"螺旋下降"——前几代 novelty 很高（在搜索空间游荡），随着代数推进逐渐 commit 到一个 fitness 高的 region。右边 Mistral-7B-Instruct 一直在搞高 novelty，但 fitness 涨不上去——典型的"为了多样性而多样性"。*

光看 fitness vs. novelty 还不够直观。再看 MDS 投影到二维语义空间——

![图6：TSP-60 上 Gemini-1.5-Pro vs. Mistral-7B-Instruct 的 MDS 投影。点的颜色编码 generation，大小编码 fitness](https://arxiv.org/html/2604.19440v1/img/mds_compare_v2.png)

*图6：Gemini-1.5-Pro（左）的 trajectory 形成一个明显的黄色聚团——"高 fitness 解都聚在某个 region"；Mistral-7B-Instruct（右）的点散满整个空间，黄色高 fitness 的点也是星星点点散开的。**前者是收敛、后者是扩散**——就这么简单。*

最后看 best-so-far fitness curve：

![图7：TSP-60 上 mean best-so-far fitness 随代数变化。Gemini-1.5-Pro 持续单调上升到接近 1.0；Mistral-7B-Instruct 在 0.6 附近震荡 30 代](https://arxiv.org/html/2604.19440v1/img/best_fitness_compare.png)

*图7：这就是 trajectory 几何差异的最终结果——同样跑 30 代，一个收敛到 fitness ≈ 1.0，一个还在 0.5-0.7 之间打转。*

到这里，"effective LLM optimizer = local refiner（局部精修器）"的故事基本闭环了。

---

## 第四刀：operator-level 验证——LRR 才是关键

前面都是 trajectory-level 的描述。但要把"局部精修"这个判断坐实，还得在更细的 parent → child mutation 层面做验证。

论文定义了两个 operator-level 指标：

- **LRR (Local Refinement Rate)**：offspring 严格优于其 prompted parent 的频率（在所有 valid offspring 里的比例）
- **PCD (Parent–Child Distance)**：offspring 跟 parent 在语义空间里的平均距离（"修改幅度"）

跑回归：

| 模型 | ZS + PCD | ZS + LRR + PCD |
|------|----------|----------------|
| Zero-shot Perf. (z) | 0.233* (0.028) | 0.144 (0.112) |
| Avg. Parent–Child Distance (z) | **−0.329\*\*** (0.001) | −0.024 (0.838) |
| Avg. Local Refinement Rate (z) | — | **0.528\*\*\*** (\lt0.001) |
| R² | 0.204 | **0.367** |

**这张表挺漂亮的**。

第一列：单独看 PCD（修改幅度），系数显著为负——改得越狠越差。一个直觉的解释是"魔改容易出 bug"。

第二列：把 LRR 加进去之后，PCD 的系数从 −0.329 直接掉到 −0.024，**完全失去显著性**。这说明 PCD 之所以负相关，是因为大幅修改往往降低了 refinement 的成功率；一旦控制了 refinement 行为，"修改幅度"本身没那么重要。

LRR 自己的系数是 0.528，p\lt0.001——**最强的预测因子**。R² 从 0.204 跳到 0.367，几乎翻倍。

这就是论文的核心 insight 浓缩成的一张表：**好的 LLM optimizer ≈ 高 LRR**。不是修改幅度大、不是 novelty 高、甚至不是 zero-shot 强——而是每次改动都能稳稳地比 parent 好一点点。

---

## 第五刀：扰动实验——把 refinement 行为掰断给你看

到这里还都是相关性。要做因果，论文跑了一个 model mixing 的扰动实验：在每代 mutation 里，混入一定比例由"弱 refiner"模型生成的 offspring，剩下由"强 refiner"生成。看 fitness 怎么变。

![图8：model mixing 扰动实验。横轴是 weak-refiner offspring 的比例（0% 到 100%），左轴是 normalized fitness（实线），右轴是 refinement rate（虚线）。三个 task：TSP-60（Pearson r=0.99）、PromptOpt-Sum（r=0.45）、Bin Packing OR3（r=0.96）](https://arxiv.org/html/2604.19440v1/img/model_mixing_dual_axis.png)

*图8：在 TSP-60 和 Bin Packing 上，weak-refiner 比例越高，fitness 几乎是单调下降，跟 refinement rate 同步坍塌（Pearson r 高达 0.96-0.99）。Prompt Optimization 上信号弱一些（r=0.45），但方向一致。这是相当强的因果证据——你可以直接通过"换掉一部分 mutation 算子"来摧毁系统的优化能力。*

我觉得这个实验设计很聪明。比起单纯换模型（很多变量都变了），混合策略只动 mutation 这一个环节，能更干净地隔离 refinement 行为的作用。

不过论文自己也在 Limitations 里坦诚——换模型还是会附带其他不可见的变化（比如 reasoning pattern、exploration tendency），完全 isolate refinement 还是做不到。这个 caveat 我觉得说得很到位。

---

## 第六刀：成本对比——Mistral-24B 是真正的赢家

最后一个 punchline 在这张图里：

![图9：所有模型在"成本-性能"空间的散布。横轴是估算成本（avg tokens × $/M output），纵轴是 evolution 平均 fitness 提升。蓝线是 Pareto 前沿](https://arxiv.org/html/2604.19440v1/img/cost_efficecny_analysis.png)

*图9：Pareto 前沿上从左到右是 Llama-3.2-1B → Llama-3.1-8B → Mistral-7B → **Mistral-24B-Instruct**。Mistral-24B 在 0.1-0.2 美刀的成本下达到了 0.6+ 的 fitness 提升——这是图里最高的提升幅度。再往右的 GPT-4o、DeepSeek-V3、Gemini-1.5-Pro，成本贵了 50-100 倍，fitness 提升反而更低。*

这张图其实是整篇论文最实用的一张。如果你正在搭一个 LLM-guided evolutionary 系统，看完这张图就能直接做选型决策——**别默认选最贵的模型，先验证它是不是好的 local refiner**。

---

## 完整实验结果：Table 2 摘要

论文 Appendix A 给了一个完整的 fitness 对比表（横跨 4 个 task family、最后一列是平均）。挑几个关键数字看：

| 模型 | Zero-Shot Avg | First Generation Avg | Last Generation Avg |
|------|---------------|----------------------|---------------------|
| GPT-4o | 53.3 | 46.5 | 77.4 |
| Gemini-1.5-Pro | 47.8 | 49.4 | 76.4 |
| DeepSeek-V3 | 45.8 | **56.9** | 75.3 |
| Mistral-Large | 45.0 | 49.5 | 75.7 |
| **Mistral-24B-Instruct** | 42.8 | 51.0 | **81.6** |
| Mistral-7B-Instruct | 34.7 | 45.5 | 70.0 |
| GPT-4o-mini | 38.4 | 47.1 | 70.1 |
| Gemini-1.5-Flash | 29.6 | 47.2 | 68.2 |
| Gemma-3n-4B | 19.6 | 46.8 | 60.8 |
| GPT-3.5-turbo | 18.0 | 45.8 | 59.3 |

注意几个有意思的现象：

1. **DeepSeek-V3 第一代最强**（first-gen avg 56.9），但最后一代只有 75.3——zero-shot 强、但优化能力没跟上。这就是论文里说的"early performance does not predict long-horizon outcomes"。
2. **Mistral-24B-Instruct 最终拿了第一名**（81.6），把 GPT-4o (77.4) 甩了 4 个点。它的 zero-shot 只有 42.8，比 GPT-4o 低 10 多个点。
3. **Llama-3.1-8B-Instruct (72.2) > Llama-3.1-70B-Instruct (69.2)**——同家族里 8B 反过来打 70B。这个现象单独抽出来，足够让所有"参数即正义"的信仰崩一块。
4. **Magistral-Small (70.4) 跟 Mistral-Large (75.7) 中间隔了 5 个点**，但 Magistral-Small 的 zero-shot 反而比 Mistral-Large 高（29.0 vs 19.5）。同家族里"reasoning 强化版"未必更适合当 search operator——这个观察单独拎出来，又是一个值得后续展开的研究点。

---

## 我的判断

**亮点**：

1. **大规模、多任务、严格 controlled** 的 setup 是真硬核，72K 个 candidate solution、500 美刀实验成本，结论的统计力度是足够的。把数据集开源到 HuggingFace 这一点也很专业，后续工作可以直接 build on top of。
2. **"Local Refiner"框架在概念上是对的**。它把"模型在 evolutionary loop 里到底在干什么"这个 black box 给打开了一个口子，给出了一个可量化的 operator-level 描述。LRR 是个非常实用的指标，可以直接拿来做模型筛选。
3. **几何分析框架（spatial entropy + fitness spatial entropy）有可迁移性**。不只能分析 evolutionary search，应该可以推广到任何 iterative agent loop 的 trajectory 分析（比如 ReAct、Reflection 这些）。
4. **批判性结论挺扎实**："novelty 不是好东西"、"参数大不一定好"这两点本身就是有冲击力的发现，配合扰动实验做了因果验证，让人不容易反驳。

**问题与局限**：

1. **"local refinement"的可操作性还需要更多工程经验**。论文给了 LRR 这个事后 metric，但没给"如何提前预测一个模型 LRR 高不高"的方法。除了实跑一遍，工程师拿到一个新模型时还是没法快速判断它适不适合当 optimizer。
2. **任务覆盖偏窄**。4 个 task family 里有 2 个是相对成熟的 LLM-guided evolution 应用（TSP、prompt optimization），equation discovery 和 bin packing 也算常见。但更接近真实 agent 场景的 web automation、coding agent 等没有覆盖。结论能不能外推到 long-horizon agent task 还要打个问号。
3. **"为什么 Mistral-24B 是最强的 local refiner？"这个机制问题没回答**。是它的 RLHF 数据里有更多 incremental refinement 的样本？还是某种 inductive bias？论文留给了"future work"——但这恰恰是工程上最关心的问题。
4. **跟 RL/SFT 训练的关系没展开**。论文最后提了一句"future work may benefit from training models as effective search operators"，引了 EvoTune (Šurina et al., 2025)，但没展开讲 LRR 这个目标如何融到 training loss 里。如果有"refine-as-RL-target"的实验，这篇论文的影响力会再上一个台阶。

**对工程实践的启发**：

- **选模型时别迷信"越强越好"**。如果你在搭 LLM-guided evolutionary 系统（PromptBreeder、AlphaEvolve 这类），先用一个小规模 trajectory 评估 LRR 和 trajectory localization，再决定主力模型。Mistral 24B 这个量级值得试。
- **prompting 要鼓励 incremental refinement**。论文 Appendix C 给的 evolution prompt 模板里都有"please return one BETTER child"这种引导。可以在 prompt 里强化 "modify minimally"、"keep what works"之类的约束，可能能提升 LRR。
- **不要盲目追求 diversity/novelty**。如果你在系统里加了 novelty bonus 来"鼓励探索"，先验证你的搜索是不是已经处于足够 localized 的状态——否则 novelty bonus 反而是噪声。
- **设计自己的 trajectory 监控指标**。LRR 和 spatial entropy 的计算成本不高，可以做成 evolution loop 的实时监控 dashboard。Spatial entropy 突然飙升可能是搜索"失控"的早期信号。
- **混合模型作为兜底策略**。论文的扰动实验是反向证明（混入弱 refiner 会拖垮性能），但反过来想——如果不同模型在不同任务/不同阶段的 LRR 不一样，那 ensemble 几个 strong refiner 来覆盖更广的 task distribution，理论上是更稳的方案。这个方向论文没展开，但工程上值得试。

---

## 一个开放问题

论文最后讨论里提到一个观察：**refinement behavior 不是 base model 单独的属性，而是整个 agentic system 的 emergent 性质**——模型、prompt、decoding 配置一起决定。

这其实暗示了一个更根本的问题——**LRR 是不是可以通过 prompt engineering 单独优化出来？**论文 Appendix D.1 提到 temperature 会影响 refinement rate，但没系统对比不同 prompting strategy 下同一模型的 LRR 变化。

如果答案是"可以"，那意味着我们不需要训练新模型，只要找对 prompt 就能把 GPT-4o 调成跟 Mistral-24B 一样好的 optimizer。如果答案是"不行"，那 RL/SFT 训练就成了必经之路（EvoTune 那条线）。

这两个方向的差别非常大。希望后续工作能给出明确答案。

---

## 收尾

LLM-guided 进化搜索这个赛道这一两年已经卷起来了——从 FunSearch 到 AlphaEvolve、从 PromptBreeder 到 EvoPrompt、从 EoH 到 ReEvo。但绝大部分工作都在比"哪个系统跑出来的最终 fitness 最高"，没几个人停下来问"为什么这个系统跑得高"。

这篇论文做的就是停下来问的那个工作。它告诉你：**这个游戏不是参数量的游戏、不是探索 vs. 利用的简单 trade-off，而是"能不能稳稳地小步前进"的游戏**。

这个 takeaway 让我想起做 RL 训练时的一个老经验——**reward 方差比 reward 均值更重要**。能稳定给出小奖励的 environment，比偶尔给大奖励的 environment 更容易学到好策略。LLM 当 optimizer 也是同一个道理。

如果你是在做 agent system 设计，这套 trajectory 分析框架应该会成为你工具箱里的一员。如果你是在做模型 post-training，"local refinement as objective"可能是一条值得探索的 fine-tuning 方向。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
