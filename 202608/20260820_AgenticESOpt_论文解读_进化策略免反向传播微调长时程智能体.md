# 不反向传播也能全参数微调 27B 智能体？进化策略在长时程任务上把 RL 反杀了

上周翻到一篇让我停下来想了一会儿的工作。做 Agent 训练的朋友应该都有体感：PPO/GRPO 那套在单轮推理上打得挺顺，可一旦把任务拉长到几十轮交互——浏览器操作、多轮工具调用——显存先爆，然后信用分配开始玄学，reward 涨了成功率不动。这篇论文给出的回答相当激进：**别修 RL 了，换进化策略（ES）**。

**核心摘要**：Agentic ESOpt 把全参数微调从"反向传播 + optimizer states"的重型栈里解放出来——只在参数上加高斯扰动、跑 rollout、按归一化奖励做加权更新，显存需求直接降到推理级别。4B 模型微调只占 8.41GB，比 GRPO 的 58.88GB 低 85.7%；在 15 轮的最短成功时程上成功率 53.13%，比最强的 GRPO 配置高 12.5 个点，而 PPO 直接崩到 0。更狠的是，它在 4 张 H100 上对 Qwen3.5-27B 做了全参数微调，WebArena-Lite 上从 29.47% 拉到 36.16%，反超 GPT-5.4 的 34.14%。这不是"穷人版 RL"，作者想论证的是：长时程、稀疏奖励场景下，ES 本来就是更对路的优化机制。我觉得这个论点值得认真对待。

---

## 📖 论文信息

- **标题**：Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Requirements
- **作者**：Zhi Zheng, Rongsheng Chen, Yunpeng Ba, Zhenkun Wang, Yee Whye Teh, Wee Sun Lee
- **机构**：新加坡国立大学、南方科技大学、牛津大学
- **发表日期**：2026 年 8 月 18 日
- **链接**：https://arxiv.org/abs/2608.17310
- **代码**：https://github.com/zz1358m/Agentic-ESOpt

---

## 🎯 问题动机：长时程 Agent 训练，RL 的两块天花板

先说我自己的体感。之前做多轮工具调用的 RL 训练时，最烦两件事：一是显存——activations、optimizer states、reference model，27B 这个量级全参数微调想都别想；二是时程一拉长，终端只有一个稀疏的 0/1 奖励，把这个标量摊回几十个动作上，梯度估计的方差大到没法看。

论文把这两个痛点画得很清楚。

![长时程 Agentic 推理的核心挑战](https://arxiv.org/html/2608.17310v1/Agent.png)

*图 1(a)：长时程 Agentic 推理的两大挑战——轨迹越来越长、分支越来越多，同时 Memory、Tool-Use、Skill 这些外部组件对行为的影响越来越大，光调参数已经不够了。*

![Agentic RL 微调的瓶颈](https://arxiv.org/html/2608.17310v1/RL.png)

*图 1(b)：Agentic RL（GRPO/PPO）的两处硬伤——反向传播需要存 activations 和 optimizer states，训练成本高、显存爆炸；turn-level 的信用分配在长分支轨迹上既混乱又次优，长时程可扩展性受限。*

论文里有一段对 PPO 的批评我觉得挺到位：PPO 需要 critic warm-up 才能学到靠谱的价值估计，但稀疏终端奖励下早期 advantage 根本不可信；就算 critic 练好了，策略梯度还是要对 H 个 action-level 的 score 项求和，方差照样随时程涨。GRPO 的组内相对优势在单轮场景好用，但多轮场景下那套"同一组 rollout 互相比较"的逻辑就开始打折扣。

作者顺手给了一个方差分析（附录 C.3），结论挺直白：在回报与单个动作弱相关、各步 score 项近似不相关的假设下，策略梯度估计器的方差随时程 H **近线性增长**：

$$\operatorname{Var}[\widehat{g}_{\mathrm{PG}}]\approx\operatorname{Var}[R(\bm{a})]\,\operatorname{Var}\!\left[\sum_{t=1}^{H}\nabla_{\theta}\log\pi_{\theta}(a_{t}\mid s_{t})\right]\propto H$$

而 ES 的估计器是 $\widehat{g}_{\mathrm{ES}} = (R(\bm{a}(\theta+\sigma\epsilon))-b)\cdot\epsilon/\sigma$，那个参数得分项 $\epsilon/\sigma$ **不对 H 求和**。一句话讲：ES 把终端回报直接归因于"这一整个连贯的策略扰动"，而不是强迫同一个标量去区分几十个时间步里谁贡献了多少。

信用分配这个老大难，ES 的选择是——不做。

---

## 🧠 方法核心：扰动、打分、加权，就这三步

Agentic ESOpt 的每一代流程简单到有点朴素：

1. 围绕当前参数 $\theta$ 采样 $G$ 个全参数扰动 $\epsilon_1,\dots,\epsilon_G$
2. 每个扰动后的 agent 去环境里跑轨迹，拿标量奖励 $R_i$
3. 奖励做群体內 z-score 归一化，然后按 reward-weighted 求和更新

公式长这样。高斯平滑目标：

$$J_{\sigma}(\theta;c)=\mathbb{E}_{\bm{\epsilon}\sim\mathcal{N}(0,I)}\left[J(\theta+\sigma\bm{\epsilon};c)\right]$$

实际更新（群体內归一化后的奖励 $\hat{R}_i$，实践中省略显式的 $1/\sigma$ 因子）：

$$\theta_{t+1}=\theta_{t}+\frac{\alpha}{G}\sum_{i=1}^{G}\hat{R}_{i}\bm{\epsilon}_{i}$$

工程上有个很讨巧的设计：只存每个扰动的**噪声种子**，推理时用原地加减恢复扰动参数。这样整个训练过程的显存占用和纯推理**完全一致**——8.41GB 微调 4B，这个数字就是这么来的。说实话我第一次看到"全参数微调 27B 只要推理显存"时是怀疑的，看完实现细节后服了，这确实是 ES 的结构性优势，不是花活。

![Agentic ESOpt 总览](https://arxiv.org/html/2608.17310v1/Agentic-ESOpt.png)

*图 1(c)：Agentic ESOpt 的思路总览——参数扰动跑轨迹、ES 直接更新参数，绕开 horizon-level 信用分配；同时 ES rollout 产生的 traces 还能顺手喂给 prompt 更新，实现 on-the-fly 的参数-提示协同进化。*

### σ 的余弦衰减：一个被理论撑起来的 schedule

以往单轮 LLM 的 ES 工作（比如论文引用的 [27]）全程用固定 σ。这篇论文给了一个 Lemma 说明为什么这不合理：对充分光滑的目标，

$$J_{\sigma}(\theta;c)=J(\theta;c)+\frac{\sigma^{2}}{2}\mathrm{Tr}\!\left(\nabla_{\theta}^{2}J(\theta;c)\right)+O(\sigma^{4})$$

注意第二项。σ 越大，平滑目标里那个 Hessian 迹的权重越大——对最大化问题，它惩罚尖锐局部最优、偏好平坦邻域，相当于一个正则项；但代价是相对原目标的偏差也越大。所以 σ 其实控制着"探索/正则"与"目标保真"的权衡，那这个量就不该从头到尾是常数。

于是有了余弦衰减：

$$\sigma_{t}=\sigma_{T}+(\sigma_{0}-\sigma_{T})\frac{1+\cos(\pi t/T)}{2}$$

训练时保留非零的终端 $\sigma_T$（还要泛化，正则不能丢）；测试时计算场景下 $\sigma_T$ 直接衰减到 0（只关心当前任务的无偏结果）。这个区分挺细的，后面的消融证明它不是摆设。

### 参数-提示协同进化：ES 的隐藏红利

这是我觉得这篇论文最值钱的一点。主流的测试时计算方法——prompt 优化、skill 蒸馏、启发式搜索——全程冻结模型参数，搜索空间被锁死在"冻结策略已经可达的行为"里。而 ESOpt 的黑盒接口让这件事变得自然：**参数更新和 prompt 更新可以共享同一批 rollout，交替进行**：

$$\theta_{t+1}=\mathcal{U}_{\mathrm{ES}}(\theta_{t};c_{t},\mathcal{D}_{t}),\quad c_{t+1}=\mathcal{U}_{c}(c_{t};\mathcal{D}_{t})$$

![Agentic ESOpt 工作流程](https://arxiv.org/html/2608.17310v1/main-process.png)

*图 2：完整工作流——采样 G 个参数扰动的 agent rollout，收集奖励做 z-score 归一化，reward-weighted 求和更新参数；右侧是 σ 的余弦 schedule（测试时衰减到 0，训练时保留终端值）；底部橙色环路表示同一批 rollout 同时驱动 prompt 更新，支持 LLM-based（Trace2Skill）和 heuristic-based（EoH）两种外部更新器。*

一批 rollout 两份用途，参数和 prompt 一起进化。这个 composability 是 RL 那套重型栈很难给的——你很难想象在 PPO 训练循环里顺手再插一个 prompt 进化外环，梯度图会把一切都搅在一起。

---

## 🧪 实验：时程越长，ES 越能打

### Agentic Sudoku：受控时程实验，结果出现了"排序反转"

作者设计了一个挺聪明的受控环境：多轮数独，每个有效动作最多填一格，只有终端奖励。遮挡 5/10/15 格，对应最短成功时程 H\* ∈ {5, 10, 15}，可以精确控制任务时程。4× H100，Qwen3.5-4B 骨干，对比 Agentic PPO、两种采样配置的 Agentic GRPO、Vanilla ES。

**表 1：Sudoku 最终评估成功率（%）与显存需求**

| 方法 | 显存 | H\*=5 | H\*=10 | H\*=15 |
|---|---|---|---|---|
| Qwen3.5-27B（不微调） | 51.75GB | 86.46 | 50.00 | 28.13 |
| Qwen3.5-4B（不微调） | 8.41GB | 63.54 | 31.25 | 10.42 |
| + Agentic PPO | 89.40GB | **90.63** | 56.25 | 0.00 |
| + Agentic GRPO† | 58.88GB | 80.21 | 44.79 | 30.21 |
| + Agentic GRPO‡ | 58.88GB | 85.42 | **67.71** | 40.63 |
| + Agentic ESOpt (G=32) | 8.41GB | 89.58 | 62.50 | **53.13** |
| w/o σ decay（Vanilla ES） | 8.41GB | 85.42 | 55.21 | 42.71 |
| w/o σ_T（衰减到 0） | 8.41GB | 85.42 | 54.17 | 28.13 |

看到这个表我愣了一下。不是说 ESOpt 全面碾压——H\*=5 时 PPO 还领先，H\*=10 时 GRPO 最强——而是**排序随时程在反转**：PPO → GRPO → ESOpt，依次在越来越长的时程上接管第一名。H\*=15 时 PPO 直接归零（critic 在稀疏终端奖励下完全学不到东西），GRPO 撑到 40.63%，ESOpt 53.13%，差了 12.5 个点。

说实话，这个"排序反转"比一致获胜更有信息量。它说明 ESOpt 不是全局更强的优化器，而是优势区间随时程移动——这恰好印证了前面那个方差 ∝ H 的分析。作者在讨论里也挺坦诚，没有把短时程的失利藏着。

效率账也算得清楚：同样 4× H100，H\*=15 时 ESOpt 训练 9.4 小时 / 9.4 EFLOPs，GRPO 要 19.0 小时 / 10.9 EFLOPs。ES 虽然群体大（G=32 vs 8-rollout），但省掉了反向传播和 reference model 评估，FLOPs 相当、墙钟时间减半、显存只有零头。

σ schedule 的消融同样在这张表里：去掉余弦衰减（Vanilla ES），H\*=15 从 53.13% 掉到 42.71%；把终端 σ 也衰减到 0，直接掉到 28.13%——过拟合了，快退回基座水平。Lemma 1 那个"终端正则不能丢"的理论判断，实验上是站得住的。

### Math 与 DocVQA：一半 FLOPs，反超 GRPO 8.3 个点

第二个场景是 ReAct 式工具使用，典型时程超过 10 轮。Math 在 400 道 DAPO 题上训练，DocVQA 在 50 题上微调。ESOpt 用 G=16，**微调 FLOPs 只有 GRPO 的大约一半**。

**表 2：Math 与 DocVQA 关键结果（Mean@4，%）**

| 方法 | DAPO | AIME 2026 | DocVQA Acc |
|---|---|---|---|
| Qwen3.5-27B No Skill | 65.8 | 76.7 | 51.8 |
| Qwen3.5-4B No Skill | 63.0 | 55.8 | 40.3 |
| 4B + Agentic GRPO | 68.8 | 58.3 | 48.0 |
| 4B + Agentic ESOpt | **76.8** | **70.8** | **52.5** |
| 4B + Trace2Skill | 64.8 | 50.8 | 47.3 |
| 4B + GRPO + Trace2Skill | 67.8 | 50.0 | 49.5 |
| 4B + ESOpt + Trace2Skill | **77.3** | **71.7** | **52.8** |

无技能时，ESOpt 让 4B 基座在 DAPO、AIME 2026、DocVQA 上分别涨 13.8、15.0、12.3 个点，三项平均比基座高 13.7 个点、比 Agentic GRPO 高 8.3 个点。微调后的 4B 甚至在 DAPO 和 DocVQA 上摸到了 27B 基座的水平。

有个细节值得玩味：GRPO + Trace2Skill 在 AIME 上反而比纯 GRPO 还差（50.0 vs 58.3），skill 和 RL 各练各的会打架；而 ESOpt + Trace2Skill 是协同进化的，AIME 上到 71.7。参数和 prompt 一起进化 vs 分开优化，差距就在这。

### WebArena-Lite：4 张 H100 全参数微调 27B，反超 GPT-5.4

这个实验是"Model Scalability"主张的直接证据。WebArena-Lite 是 165 任务的浏览器操作基准，骨干用 Qwen3.5-27B，4× H100 80GB。这个规模下全参数 Agentic RL 已经不现实了，而 ESOpt 还是推理级显存，群体只用 G=8。

**表 3：WebArena-Lite 成功率（%，3 次运行均值）**

| 模型 / 方法 | Reddit | GitLab | CMS | Map | OSS | 平均 |
|---|---|---|---|---|---|---|
| GPT-5.4 | 47.62 | 46.88 | 46.67 | 19.05 | 21.01 | 34.14 |
| GPT-5.4-mini | 39.68 | 29.17 | 30.48 | 13.10 | 13.77 | 23.23 |
| Qwen3.5-27B No Skill | 50.79 | 35.42 | 41.90 | 8.33 | 21.01 | 29.47 |
| + Agentic ESOpt | 49.21 | 43.75 | 49.52 | 14.29 | 30.43 | **36.16** |
| Qwen3.5-27B Trace2Skill | 49.21 | 39.58 | 46.67 | 13.10 | 28.26 | 33.94 |
| + Agentic ESOpt | 52.80 | 41.67 | 50.48 | 10.71 | 32.61 | **36.36** |

ESOpt 把 No Skill 基线从 29.47% 拉到 36.16%，涨了 **6.69 个点**，微调后的开源 27B 反超了 GPT-5.4 的 34.14%。分类别看，GitLab、CMS、Map、OSS 都在涨，只有 Reddit 微跌 1.58 个点——作者没回避这个，挺好。与 Trace2Skill 组合时再涨 2.42 个点到 36.36%，组合收益在收窄但方向一致。

### Automatic Heuristic Design：36 个对照里赢 28 个

最后一个场景把协同进化推到测试时计算：EoH 式的自动启发式设计——LLM 生成组合优化问题的启发式代码（TSP、KP、ASP、CVRP、BPP 这些 NP-hard 货），评估器执行后返回标量目标。ES 更新只挂在变异算子 m1、m2 上，奖励用"父代成本 − 扰动子代成本"。全部用 LLaMA-3.1-8B-Instruct。

结果：36 个匹配的方法-预算比较中改进 28 个。拆开看，ESOpt + EoH 在两种预算下改进了全部 6 个 constructive 测试集（12/12 全胜）；ESOpt + Sample 是 9 胜 1 平 2 负，输掉的两项都在 ASP N=15 上。TSP N=20 上最优性差距收窄 22.96%，ASP N=21 上提升 29.18%。不是全胜，但胜率分布很说明问题——跟越强的外部搜索器（EoH）组合，参数进化的收益越稳。

### 群体规模消融：大模型对 ES 更友好

15 轮数独上的 Vanilla ES 消融：4B 模型群体从 8 涨到 16，最佳测试成功率从 5.10% 飙到 35.42%；而 9B 模型从 30.21% 到 37.50%，对小群体远没那么敏感。

![大小模型的参数地形直觉图](https://arxiv.org/html/2608.17310v1/scaling_es_agent.png)

*图 5：群体规模的直觉解释——小模型所处的参数区域"平坦"，随机采样方向大多没用，需要更大群体才能碰到好方向；大模型预训练后处在能力更强的局部区域，邻近扰动更容易产生有信息量的行为，小群体就够用。*

这个发现工程意义不小：**模型越强，ES 微调越便宜**。WebArena 上 27B 只用 G=8 就能涨 6.69 个点，就是这个规律的实锤。顺着想下去，对更大的前沿模型做 ES 微调，成本可能比我们直觉预期的低得多。

### 参数地形可视化：奖励对比度随时程衰减

![15 轮数独的参数地形热力图](https://arxiv.org/html/2608.17310v1/qwen35_4b_mask15_centered_heatmap.png)

*图 6（H\*=15）：Qwen3.5-4B 在 15 轮数独上的局部参数地形——中心点奖励仅 0.0312，邻域内最好的扰动方向能到 0.1562，最差为 0。时程变长后奖励对比度整体下降、地形更破碎，但 ES 的参数得分不随时程求和，所以方差不跟着爆。*

---

## 🤔 我的判断

这篇论文最打动我的不是某个单点数字，而是它把一个被主流叙事边缘化的技术路线重新拉回了牌桌。过去两年 Agentic RL 的改进大多在信用分配内部打转——更细的 advantage、轮级奖励塑形、critic 设计——Agentic ESOpt 的选择是直接掀桌子：信用分配太难？那就不做，用轨迹级归因。显存太贵？不要反向传播，全参数照调。

批判性地说几点。其一，**它不是免费的午餐**：ESOpt 用更多的独立环境评估换掉了反向传播，在那些环境交互本身昂贵或缓慢的场景（真实机器人、生产系统），这个交换未必划算，作者在局限性里也承认了。其二，H\*=5 和 H\*=10 上它并不是赢家，"时程依赖的优势区间"这个定位要记牢——短任务上 GRPO/PPO 仍然是更优选择。其三，Sudoku 是合成的受控环境，排序反转现象能否在更杂乱的真实任务分布上复现，我持谨慎乐观。其四，ES 引入了新超参（σ 初值、终端值、群体规模、有效学习率 α），调参负担只是转移了，没有消失。

但即便如此，三个结构性优势是实打实的：推理级显存让 27B+ 的全参数微调从"不可能"变成"4 张卡搞定"；黑盒接口让参数进化和 prompt/skill 进化共享 rollout，这是梯度方法给不了的 composability；时程方差解耦有理论支撑也有 Sudoku 排序反转的实证。尤其"大模型对 ES 更友好、小群体够用"这个发现，如果在前沿模型上继续成立，ES 微调的成本曲线会跟 RL 完全不在一个量级上。

如果你在做长时程 Agent 的训练基础设施，我的建议是：不用急着换技术栈，但值得把 ES 作为对照组加进你的实验矩阵，特别是 20 轮以上交互、只有终端奖励的任务。这类"优化机制层面的换轨"工作，往往比又一种 RL 变体更值得花时间。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
