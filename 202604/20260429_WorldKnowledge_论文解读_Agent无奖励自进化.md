# 撕掉"自进化"的伪装：让Agent在没人喂奖励的时候，自己摸清楚一个陌生世界

## 一段开场白

最近一两年读 Agent 自进化相关的 paper，越读越觉得有件事很别扭。

每篇都说自己是 "self-evolving agent"，但你仔细看会发现：要么需要人精心设计一堆 task，要么需要一个 verifier 去算 reward，要么需要两个 agent 互相博弈互相出题——总之只要把人为设定的那一套抽掉，所谓的"自进化"立刻停摆。

这跟人类学一个新东西的方式完全不一样。你换个新城市、装个新软件，没人给你出考试题、也没人给你打分，你就是会主动溜达一圈，把这地方大概的格局摸清楚。**理解环境本身就是智能的前提**，跟有没有具体任务无关。

来自 Tencent 和 HKUST（广州）的这篇 paper 把这个矛盾摆到了台面上，并且认真给出了一个解法。核心想法很直接：把"探索环境、压缩成知识"这件事训练进模型本身，让 agent 在面对一个陌生网站/游戏/代码库时，**不需要任何任务、也不需要任何 reward，自己就能跑出一份"世界知识"**，回头干活的时候直接当上下文用。

效果上，Qwen3-30B 和 Seed-OSS-36B 在 WebVoyager / WebWalker 上分别涨了大约 20 个百分点的绝对值；最有意思的是，14B 的 Qwen3 拿着这套自己生成的"世界知识"，在两个测试域上居然干过了裸跑的 Gemini-2.5-Flash。

---

## 核心摘要（先看这段）

这篇 paper 在解决的问题是：**现有 self-evolving agent 都在装样子，说到底仍然是"被人指挥着进化"**。一旦把人类提供的任务集和 reward signal 抽掉，进化立刻停止。

作者提出了一个叫 **Meta-Learning-Driven Evolution** 的新范式，核心机制是：训练时用一个 outcome-based reward（生成的 World Knowledge 能让下游任务涨多少分）作为监督信号，但**这个 reward 只在训练阶段使用**。模型一旦训练好，部署时进入一个完全陌生的环境，就能自发地"探索 → 压缩 → 输出一份 markdown 格式的 World Knowledge"，整个过程不需要任何 task、reward 或 human prompt 介入。

技术栈是 SFT（用 Gemini-2.5-Pro 当 teacher 蒸馏）+ 两轮 Reinforcement Rejection Sampling（因为 trajectory 太长，约 374 步、单步 3322 token，纯在线 RL 跑不动）。

**我的看法**：思路确实漂亮，"训练阶段用 reward 教会模型怎么探索"这个 meta-learning 视角值得借鉴。但要注意几点——所谓"reward-free"只是在 inference 时成立，整个训练管道对 reward 的依赖一点都没少；headline 的"14B 干过 Gemini-2.5-Flash"，对比的是 Gemini 裸跑 vs 14B 拿 Seed-36B 的 WK，本质比的是"有 mental map vs 没 mental map"，而不是模型能力。它真正值钱的地方在于：**把 Anthropic 风格的 skill 文件变成 agent 自己生成的，而且生成出来的东西可以跨模型迁移**。

---

## 论文信息

- **标题**：Training LLM Agents for Spontaneous, Reward-Free Self-Evolution via World Knowledge Exploration
- **作者**：Qifan Zhang, Dongyang Ma, Tianqing Fang, Jia Li, Jing Tang, Nuo Chen, Haitao Mi, Yan Wang
- **机构**：Tencent；The Hong Kong University of Science and Technology (Guangzhou)
- **发布日期**：2026 年 4 月 20 日
- **arXiv**：[2604.18131](https://arxiv.org/abs/2604.18131)
- **模型**：[Bklight999/World-Knowledge on HuggingFace](https://huggingface.co/Bklight999/World-Knowledge)

---

## 为什么"自进化"在今天还是个伪命题

先别急着看方法。我想先把作者怒喷整个领域的那一段拎出来。

作者把现有的 self-evolving agent 分成两类——

**第一类，Experience-Driven Evolution**。这是绝大多数论文的范式：人定义一堆任务、人写一个 reward function，agent 反复尝试解题，把"轨迹—得分"对当作经验积累。然后用这些经验去更新 prompt、更新 memory、更新工具库、或者直接 fine-tune 模型参数。从 AgentS 到 ReasoningBank、Mem0，从 Darwin 到 SkillWeaver，本质都是这个套路。

**第二类，Adversarial Evolution**。把出题这件事也外包给一个 challenger agent，让 challenger 不断生成更难的任务，solver 不断学着解。SPICE、R-Zero、DR 之类。看起来"agent 自己出题自己解"很自动化，但实际上 challenger-solver 这套 workflow 本身就是人精心设计的。Agent 还是被困在做"作业本"。

这段批评其实挺到位的。我自己之前在做 RAG-based agent 评测的时候也碰到过类似的纠结：你说我的 agent 能"在线学习"，但只要把那个"标准答案数据集"抽掉，所有的 in-context learning 都没了着力点。

![图1：三种自进化范式对比 — 左：Experience-Driven 需要人定义任务和奖励；中：Adversarial 需要人设计博弈框架；右：本文的 Meta-Learning-Driven 真正实现了 task-free 和 reward-free 的进化](https://arxiv.org/html/2604.18131v1/x1.png)

*图1：三种自进化范式对比。前两种范式都标着 ❌ Task- and Reward-Free，只有 Meta-Learning-Driven 范式真正做到了 task- 和 reward-free，且对人工依赖最小。*

作者提出的第三条路是这样的：

> **训练时**用 reward 教会模型"怎么去探索"这件事本身；**部署时**模型自动地、主动地探索新环境，把观察到的东西压缩成一份结构化的 World Knowledge。

注意这里的关键转变——reward 不是用来评估"你这步做对了没"，而是用来评估"你产出的这份知识能不能让下游任务做得更好"。这是一个典型的 meta-learning 视角：用 outcome 反推 process。

---

## World Knowledge 是什么

这部分很重要，因为它决定了整套方案到底是什么形态。

作者把 World Knowledge $\mathcal{K}$ 定义成**一份 markdown 文档**，结构化地描述某个具体环境（比如某个网站、某个游戏世界、某个代码库）的"内在逻辑"。

我读到这里第一反应是：这不就是 Anthropic 的 [Claude Skills](https://github.com/anthropics/skills) 吗？作者也很坦诚，论文里直接 cite 了这个仓库，并给出了区分：

- **Skill** 通常是任务相关的（比如 webapp-testing 怎么做）
- **World Knowledge** 是环境实例相关的（比如 ACL 2025 这个具体网站长什么样、有哪些重要页面、关键日期是什么）

类比一下：Skill 是"我会用 Excel"，World Knowledge 是"我对这家公司的财务数据库结构很熟"。前者是能力，后者是地图。

整个 agent 的生命周期被拆成两个阶段：

**Native Evolution Phase**：进入新环境 $E$，自发地探索和总结，产出 $\mathcal{K} \leftarrow \pi_{\text{evolve}}(\mathcal{K}|E)$。这一步**没有 task、没有 reward**。

**Knowledge-Enhanced Execution Phase**：拿到具体任务后，agent 用 $\mathcal{K}$ 辅助决策：$a_t \sim \pi_{\text{task}}(a_t|o_t, \mathcal{K}, \text{Task})$。

这个解耦其实是很关键的工程设计——它意味着 World Knowledge 一次生成，可以被多个下游任务复用，分摊掉了探索成本。

---

## 那训练信号到底从哪来？

这是整篇论文最 tricky 的地方。如果探索阶段没有 task、没有 ground truth，怎么告诉模型"你这趟探索是好的还是糟的"？

作者的解法是 **outcome-based reward**：

$$R_{\text{evolve}}(\mathcal{K}) = \text{Success}(\mathcal{T}_E | \mathcal{K}) - \text{Success}(\mathcal{T}_E | \emptyset)$$

公式本身不复杂——给定一个环境 $E$，从这个环境里挑一组下游任务 $\mathcal{T}_E$。一份 $\mathcal{K}$ 的好坏，**由它能给下游任务带来多少 success rate 的提升来决定**。

具体实现里，作者构建了一个训练集：**20 个网站、600 道 deep search 题目**。对每个网站，让 agent 生成 World Knowledge，然后用这个 $\mathcal{K}$ 喂给基线模型 Qwen3-30B-A3B 去答题，对 600 题里属于该网站的子集求平均成功率，跟没有 $\mathcal{K}$ 时的 baseline 对比。

这里有个特别值得划重点的设计：**这个 reward 只在训练时使用**。

为什么这件事重要？因为它是整个范式区别于其他方法的核心。训练完成之后，模型已经把"怎么探索一个陌生环境才能产出高 utility 的知识"这件事内化进参数里了。部署到一个全新的环境时，根本没有 600 道题可参考，模型也根本不需要——它直接基于学到的 meta-policy 自发探索就行。

读到这里我的第一反应是：这跟人类学习驾驶有点像。你考驾照的时候，教练给你 reward（教练吼你了等于负 reward）；考完之后开自己的车去新城市，没人评分了，但你已经会开了。reward 只是 training-phase 的辅助信号。

---

## 整体架构：SFT 打底 + 两轮 Rejection Sampling 微调

![图2：方法整体流程 — 左：数据生成（teacher agent 用 plan-explore-refine-summarize 四阶段产出 World Knowledge）；中：多阶段训练（SFT 用 best trajectory 蒸馏，再做两轮 Reinforcement Rejection Sampling）；右：推理时 agent 自发执行 native evolution，产出 WK 再去解任务](https://arxiv.org/html/2604.18131v1/x2.png)

*图2：完整训练管道。左半部分是数据生成 pipeline，teacher agent 在 Web/Code/Game 环境里用四步循环（Planning → Exploring → Summarizing → Refining）生成 World Knowledge；中间是两阶段训练（SFT + 两轮 RFT），每轮都通过 reward calculator 挑出 best trajectory 喂给下一轮；右边是部署阶段，训练好的 agent 自动完成 Native Evolution 和 Knowledge-Enhanced Execution。*

具体训练管道分两步走。

### 第一步：SFT 蒸馏老师模型

作者用 **Gemini-2.5-Pro** 当 teacher，让它对每个环境生成 3 份候选 $\{\mathcal{K}_i\}_{i=1}^3$，然后用 reward function 挑出最好的那份 $\mathcal{K}^*$，连同对应的完整探索轨迹 $T^* = \{Q, o_1^*, a_1^*, ..., o_k^*, a_k^*\}$，作为 SFT 的训练数据。

这里有两个数字让我愣了一下：
- 平均轨迹长度 **374.8 步**
- 平均每步信息密度 **3322.4 tokens**（observation + action）

简单乘一下：**单条 trajectory 平均 1.25M tokens**。这是非常恐怖的长上下文样本，也直接解释了为什么后面不能做标准的在线 RL。

teacher 生成的知识本身效果就不错——给 Qwen3-30B-A3B 用，平均能涨 **10.72%** 的绝对准确率。这是个相当扎实的 SFT signal。

### 第二步：Reinforcement-based Rejection Sampling（RFT）

到 RL 这一步，作者很诚实地承认 GRPO 之类的标准在线 RL **跑不起来**：

1. **超长 horizon**：单 trajectory 几百步，sparse reward + backprop 显存炸掉
2. **reward 计算太重**：每评估一份 $\mathcal{K}$ 都要让另一个 agent 去跑下游任务，跟训练同步根本不现实

所以选了 **rejection sampling fine-tuning** 这个折中方案：把 trajectory 生成跟 policy update 解耦。

具体做法：
1. 用 SFT 后的 $\pi_{\theta_1}$ 自动生成 $C$ 个候选 World Knowledge
2. 用 $R_{\text{evolve}}$ 给每个打分
3. 挑最高分的轨迹作为下一轮 SFT 数据

这个过程做了两轮（rft1, rft2）。最终模型记作 $\pi_{\theta^*}$。

**坦率讲**，这种"rejection sampling 当 RL 用"的方案在 RLHF 圈不算新——LLaMA 2、Tülu 系列都用过类似套路。但用在这种超长 horizon、需要外部 agent 评估 reward 的 setting 里，确实是务实的工程取舍。如果硬上 PPO/GRPO 估计这个项目就跑不出来。

---

## 实验结果：数据怎么说

评测是在两个 benchmark 上做的：

- **WebWalker**：4 个领域（Conference / Game / Organization / Education），每个领域随机抽 10 个网站
- **WebVoyager**：4 个网站（Wolfram / Apple / Dictionary / Coursera）

总共 **1,427 道评测题**，过滤掉了模型靠 pretrained knowledge 就能直接答对的题目（这点很重要，避免 leak）。

### 主实验：World Knowledge 真的能让模型"开窍"

**Table 1：两个 backbone 在两个 benchmark 上的成功率（%）**

| 方法 | WebWalker Conf. | Game | Org. | Edu. | **Avg.** | WebVoyager Wolfram | Apple | Dict. | Coursera | **Avg.** |
|------|-----------------|------|------|------|----------|--------------------|-------|-------|----------|----------|
| **Qwen3-30B-A3B-Instruct-2507** | | | | | | | | | | |
| Without | 24.28 | 23.65 | 22.30 | 17.93 | 22.04 | 54.30 | 37.20 | 41.86 | 30.95 | 41.08 |
| Prompt-Only (Gemini) | 35.59 | 27.87 | 31.36 | 24.56 | 29.85 | 73.90 | 53.40 | 51.16 | 45.23 | 55.92 |
| Prompt-Only (Base) | 21.37 | 20.91 | 17.42 | 18.29 | **19.50** | 54.30 | 32.56 | 42.85 | 33.33 | 40.76 |
| Ours (SFT) | 45.05 | 37.35 | 37.98 | 32.31 | 38.17 | 60.87 | 41.86 | 62.79 | 40.48 | 51.50 |
| **Ours (RFT)** | 43.14 | **42.47** | **42.16** | **35.86** | **40.91** | 58.70 | 48.84 | **67.44** | **54.76** | **57.44** |
| **Seed-OSS-36B-Instruct** | | | | | | | | | | |
| Without | 19.37 | 10.75 | 21.80 | 13.11 | 16.26 | 54.30 | 48.84 | 23.26 | 33.33 | 39.93 |
| Prompt-Only (Gemini) | **53.50** | 24.37 | 23.96 | 23.48 | 31.33 | 58.60 | 53.49 | **62.79** | 52.38 | 56.82 |
| Prompt-Only (Base) | 20.51 | 12.20 | 17.42 | 16.46 | 16.65 | 47.82 | 46.34 | 30.23 | 22.50 | 36.72 |
| Ours (SFT) | 35.48 | 24.10 | 26.80 | 27.90 | 28.57 | **71.73** | 51.16 | 53.49 | 47.61 | 56.00 |
| **Ours (RFT)** | 45.07 | **34.29** | **38.41** | **32.22** | **37.50** | 63.04 | **55.81** | 51.16 | **57.14** | 56.79 |

我专门盯着这个表看了半天，挑几个值得划重点的现象：

**1. Prompt-Only (Base) 是最有意思的发现**

不训练，只让裸的 Qwen3-30B 用 expert prompt 去生成 WK，在 WebWalker 上拿了 **19.50%**——比直接答题（22.04%）还低 2.5 个点。

什么意思？**模型可以"假装"自己在按指令探索，但产出的东西反而是噪声，会污染下游任务**。这一点其实蛮反直觉的——你以为给个详细 prompt 模型就能做好探索，实际不行。这印证了作者的判断："standard LLMs lack the inherent instinct to explore for the sake of knowledge"。

**2. Ours (RFT) 超过了 Teacher**

Qwen3-30B 用 Gemini 生成的 WK 拿到 29.85%，但用自己 RFT 之后生成的 WK 拿到 **40.91%**——足足超过 teacher 11 个点。

这看起来很反直觉（学生超过老师？），但仔细想想合理：RFT 优化的目标本身就是"让 Qwen3-30B 这个具体模型用着舒服的 WK"。Gemini 写的 WK 可能更全面更精致，但不一定贴合 Qwen3-30B 的注意力分布和推理习惯。这个观察让我想起 [Shazeer 那句老话](https://arxiv.org/abs/1706.03762)的精神：要 evaluation aligned with deployment。

**3. SFT 已经吃掉了大部分增益**

Qwen3-30B：SFT 38.17 → RFT 40.91，RFT 只多了 2.7 个点。
Seed-36B：SFT 28.57 → RFT 37.50，RFT 多了 8.9 个点。

Seed-36B 上 RFT 收益更大，说明 RFT 的价值跟 base model 的 instruction following 能力有关——SFT 已经把容易学的部分学完了，剩下的需要 RL 信号去打磨。

### 效率：步数也少了 17%

**Table 2：Qwen3-30B 在 WebWalker 各域的平均执行步数对比**

| 配置 | Conference | Game | Organization | Education | **Avg.** |
|------|-----------|------|--------------|-----------|----------|
| Qwen3-30B（无 WK） | 25.65 | 23.26 | 17.96 | 30.25 | 24.28 |
| Qwen3-30B + $\mathcal{K}$ | 20.64 | 20.31 | 13.92 | 25.34 | 20.05 |
| **Improve Ratio** | 0.20 | 0.13 | 0.22 | 0.16 | **0.17** |

不光是答得更准，**还答得更快**——平均省了 17% 的步数。这其实比单纯涨准确率更有工程意义：在真实部署里，每多一步就是一次 LLM 调用 + 网页交互延迟，省 17% 步数对成本和用户体验的影响是直接的。

---

## 让我有点意外的发现：跨模型迁移居然这么强

![图3：跨模型 World Knowledge 迁移 — 在 Conference 和 Game 两个域上，将 Qwen3-30B(RFT) 和 Seed-36B(RFT) 生成的 WK 喂给 Qwen3-14B / GPT-OSS-120B / Kimi-K2-Turbo / Gemini-2.5-Flash，全都有大幅增益](https://arxiv.org/html/2604.18131v1/x3.png)

*图3：跨模型迁移结果。每组柱状图对比了"裸跑"和"加载来自 Qwen3-30B(RFT) 或 Seed-36B(RFT) 的 World Knowledge"。最有冲击力的现象在右两组：拿着 Seed-36B 生成的 WK，Kimi-K2-Turbo 在 Conference 域上从 34.2% 涨到 56.4%；Gemini-2.5-Flash 从 31.3% 涨到 51.9%——后者甚至超过了它更大版本 Gemini-2.5-Pro 的裸跑成绩 40.2%。*

这张图是整篇论文里最 striking 的部分。我盯着 Gemini-2.5-Flash 那一组看了很久——**用别人写的 mental map，能让一个模型干过它自己更大的版本**。

具体几个数据：
- **Qwen3-14B** + Seed-36B 的 WK：Conference 从 17.5 → 35.6（×2），Game 从 11.8 → 30.5（×2.5）
- **Kimi-K2-Turbo** + Seed-36B 的 WK 在 Conference 上拿到 **56.4%**，超过 Kimi-K2.5 裸跑的 39.9%
- **Gemini-2.5-Flash** + Seed-36B 的 WK 在 Conference 上拿到 **51.9%**，超过 Gemini-2.5-Pro 裸跑的 40.2%

最 headline 的现象：**14B Qwen3 + WK 在 Conference (35.6%) 和 Game (30.5%) 上都超过了 Gemini-2.5-Flash 裸跑（31.3% 和 25.7%）**。

作者把这个现象包装成 "Exploration over Parameters: The Knowledge Scaling"——说明对于 Agent 任务，**精确的环境知识可能比模型参数规模更关键**。

我对这个 framing 持谨慎态度。客观地讲，这个对比其实没那么 "shocking"——你给 14B 模型一份针对特定网站的 mental map，去答这个网站的问题；让 Gemini-2.5-Flash 没地图裸跑同样的问题。**这两个 setup 本身就不在一个维度**。更公平的对比是：让 Gemini-2.5-Flash 也用自己生成的 WK，看谁跑得更高。但即便如此，"WK 是模型无关的、可迁移的"这个发现本身就足够有价值——它意味着探索成本可以被高效模型一次性付掉，弱模型直接 free ride。

---

## 训练阶段的"边际收益曲线"

![图4：四个领域上，Qwen3-30B-A3B 和 Seed-OSS-36B 在 base / sft / rft1 / rft2 四个训练阶段的准确率变化趋势](https://arxiv.org/html/2604.18131v1/x4.png)

*图4：训练阶段消融。两个 backbone 在 4 个 WebWalker 域上随训练阶段的性能变化。普遍规律是：base → sft 的跨度最大，sft → rft1 还有可观增益，rft1 → rft2 普遍是边际收益甚至有些下滑（比如 Qwen3 的 Conference 从 rft1 的最高点回落到 rft2）。*

这张图坐实了我前面看 Table 1 时的判断：**SFT 是大头，rft1 还能再榨一点，rft2 基本就是修修补补**。

工程含义其实挺重要的：如果你想复现这套方法，**rft 一轮基本就够了，第二轮的投入产出比明显下降**。这跟 RLHF 圈里"PPO 跑太多步会过拟合 reward"的经验是一脉相承的。

---

## 多长的 World Knowledge 才合适？

![图5：在 Conference 和 Game 两个域上，World Knowledge token 长度从 0 / 4-8k / 8-16k / 16-32k / 32-64k 的准确率变化](https://arxiv.org/html/2604.18131v1/x5.png)

*图5：Token 长度敏感性分析。两条曲线呈现"先涨后平甚至略降"的形态：从 4-8k 扩到 8-16k 时 Game 域有跳变（30.74 → 39.71），但继续扩到 32-64k 时反而轻微下滑（41.56 → 40.72）。Conference 域的曲线在 16k 处达到 49.85%，之后趋平。*

这个发现也很扎实：**WK 不是越长越好，最佳长度在 16-32k 之间**。

物理直觉很清楚：
- 太短（<8k）→ 信息丢失，关键页面没覆盖到
- 中等（8-32k）→ 信息密度高，正好覆盖核心场景
- 太长（>32k）→ 噪声进来，分散 agent 注意力

这跟 long-context retrieval 圈最近这一两年在讨论的 "lost in the middle" 现象是同源的。你 dump 进 context 的信息越多，模型有效利用的比例越低，到某个 turning point 之后甚至是负收益。

工程启发：**生成 WK 的时候要内置 token budget control**，别让 agent 自由发挥写多长写多长。论文里也确实是这么做的——prompt 里硬编了 `{token_limit}` 和 `{min_token}` 两个约束，让 teacher agent 自己学会"该压就压、该扩就扩"。

---

## Case Study：一个具体的例子

![图6：ACL 2024 网站上的真实问答案例 — 左边是 Qwen3-30B-A3B 裸跑，7 步、答错；右边是 Qwen3-30B-A3B 加载 World Knowledge 后，2 步、答对](https://arxiv.org/html/2604.18131v1/x6.png)

*图6：真实案例对比。问题是：ACL 2024 的 Printing Order Service 注册截止日期，与会议主场地更新公告日期之间相差多少天？左侧裸跑 agent 因为找不到 venue 公告，第 7 步开始用"历史规律"瞎猜（猜成 190 天，红色标注为错误）；右侧带 WK 的 agent 第 1 步就在 WK 里看到 "printing service deadline is August 9, 2024"，第 2 步交叉对照找到 "venue update May 5, 2024"，正确答案 96 天。步数从 7 步降到 2 步，答案从错变对。*

这个 case 非常典型——**没有 mental map 的 agent 会在找不到信息的时候开始幻觉**（"历史规律显示 venue 公告通常在截止日 3-6 个月前"）。有了 mental map，agent 知道"venue update 信息可能藏在哪个页面"，能直接定位过去。

这跟我自己用 Cursor / Cline 写代码时的体感一致：在熟悉的 codebase 里 agent 表现飞起，丢到一个完全陌生的项目里就开始瞎猜文件名。给它一份 README + 项目结构图，效果立刻不一样。

---

## 几个值得展开的工程细节

### Input 预处理：把网站建模成图

直接把 homepage URL 丢给 agent 让它"自由探索"是不现实的——一个大型网站可能有几千个子页面，agent 跑几个小时都摸不清楚。

作者的预处理方案（详见 Appendix A）：

1. 把网站建模成有向图，每个页面是一个节点
2. 计算每个节点的 importance：$\text{Importance}(v) = 0.7 \cdot d_{\text{in}}(v) + 0.3 \cdot d_{\text{out}}(v)$
3. 按 URL 前缀聚类，把同前缀的页面分到一组

这套预处理把网站变成"分群 + 排序好"的输入 $\mathcal{G}(U)$，agent 只需要在这个结构化输入上做"先分配 token 预算 → 按 cluster 逐个抓 → 选高分页面 → 写 summary"。

我觉得这一步其实挺关键，但也是这个方法的一个软约束：**它假设环境本身可以被预处理成图结构**。对于网站、代码库这种天然有 URL 或 file tree 的场景，没问题；但对于一些更抽象的环境（比如 dialog system、IoT 设备网络），可能要重新设计预处理逻辑。

### 训练数据规模：600 道题、20 个网站

这个数字说大不大说小不小。跟 RLHF 动辄百万级对比数据相比，这个量级很少；但对于"教模型一种 meta-skill"来说，可能正好。

这里有个隐忧：**20 个网站够不够覆盖"探索新环境"这个能力的全部分布**？我倾向于认为不够，但作者在跨模型实验里用了 4 个 backbone × 4 个新域的设置，至少证明了不是过拟合在这 20 个网站上。

### 推理时的 hyperparameter

- World Knowledge 生成阶段：max steps **t=500**, max time **L=43,200 秒**（12 小时！）
- 下游任务回答阶段：max steps **t=100**, max time **L=3,600 秒**（1 小时）

这给我一个直观的 cost intuition：**生成一份 WK 需要的资源量级是个真实问题**。12 小时、上百万 token 的 LLM 调用——这是一笔不小的"前期投入"。当然，这笔投入是 amortize 在后续无数次任务上的，所以单次任务的边际成本可控。

---

## 我的判断：这论文到底怎么样

读完之后，几个观察：

**真正打动我的部分**

1. **"训练时给 reward，推理时不给 reward"这个 meta-learning 视角**确实漂亮。不是"在线学习"那种 hand-wavy 的伪自主，而是把"如何探索"这件事 distill 进参数里。
2. **Prompt-Only (Base) 比 Without 还差** 这个 finding 反直觉但很有价值——证明了"探索能力"不是任何一个 LLM 默认就有的，必须专门训练。
3. **跨模型迁移**确实有惊喜。WK 作为 markdown 文件这个设计，让它天然成为一个"开放协议"，任何模型都能消费。
4. **效率提升 17%** 是个被低估的数字。在 production agent 里，这个数字直接对应运行成本和用户感知延迟。

**让我皱眉的部分**

1. **"Reward-Free" 是个有水分的口号**。整套训练管道对 reward 的依赖丝毫没减少——600 道题 + 标准答案 + 一个 verifier agent 算分，这套东西比标准 RLHF 复杂多了。它准确的描述应该是 "inference-time reward-free"，但作者标题里就直接 "Reward-Free Self-Evolution"，有点 overclaim。
2. **跟 teacher 的对比设计偏向自己**。Ours (RFT) 比 Prompt-Only (Gemini) 高 11 个点，乍一看是把 teacher 干爆了。但更公平的对比应该加一组 "Gemini-2.5-Pro 用自己的 RFT 模型生成 WK"——这种对比缺位让 "学生超过老师" 看起来比实际更夸张。
3. **训练数据规模偏小**。600 道题、20 个网站，泛化到 ACL 2024 这种新 conference site 没问题（domain 接近），但泛化到一个 SaaS 后台、一个游戏世界、一个内部代码仓库的能力还有待验证。
4. **跟同期工作的对比不够充分**。SkillWeaver、AgentSquare、ReasoningBank 这些都是 self-evolving agent 方向的近作，论文里只是 cite 了一下，没有做横向 benchmark 对比。读者很难判断这个 method 是否真的全面优于这些 baselines。
5. **Native Evolution 这个名字起得有点炫**。说到底就是"用 SFT + rejection sampling 让模型学会写 markdown 文档"，没必要包装成新概念。但这是 paper 写作的常态，可以理解。

**对工程的启发**

如果你在做 web agent / browser agent / 任何需要在陌生环境工作的 agent 系统，这套方法值得试：

1. 在你的领域里收集一组带答案的"诊断题"（论文里是 600 道，类比可以理解为 SQL 中的 query plan 校验集）
2. 用一个强模型（Gemini-2.5-Pro / Claude / GPT-5）当 teacher 生成 World Knowledge
3. 用 reward = "WK 让基线模型涨多少分" 来筛选 best trajectory
4. 用筛出来的 trajectory SFT 你的部署模型
5. 一两轮 rejection sampling 就够，不用上昂贵的在线 RL
6. 部署时让模型先 spontaneously 探索，产出 WK 缓存，下游任务复用

**对 agent 设计哲学的启发**

最后想说一点 high-level 的观察。这篇论文背后有一个隐含的判断：**agent 性能的瓶颈正在从"模型能力"转向"上下文质量"**。14B 模型 + 一份精心 distilled 的环境知识 > Gemini-2.5-Flash 裸跑——这个等式如果在更多场景下成立，意味着我们应该把工程精力更多放在 "context engineering"（怎么帮 agent 把 task-relevant 的信息注入上下文）而不是只 chase 更大的模型。

这跟最近 [Anthropic skills](https://github.com/anthropics/skills) 和各家 codebase context (Cursor / Cline / Aider) 的产品方向其实是一致的——大家都在赌：未来的 agent 不是"更大的脑子"，而是"更快地把陌生环境内化"。

---

## 一段收尾

这篇论文的标题其实有点夸张，"Reward-Free Self-Evolution" 这个口号要打个折扣读。但去掉营销话术之后，**它给了一个让我觉得"这个工程问题被认真对待了"的解决方案**。

特别是 Prompt-Only (Base) 反而比 Without 差这个发现——它告诉我们一个反常识的事实：**给一个未经训练的 LLM 一个详细的 exploration prompt，并不能让它真的会探索；它只会更自信地胡说**。这个洞察本身就值整篇论文的票价。

如果你现在正在做需要 agent 在陌生环境工作的产品，无论是浏览器代理、代码助手还是企业内部知识库的 RAG，这套"先生成 mental map、再用 mental map 解任务"的范式都值得放进你的工具箱。它不是 silver bullet，但它是**对"长上下文 + 复用 + 跨模型迁移"这三件事的一个相当扎实的工程演绎**。

至于这是不是通往 AGI 的一步——作者在 conclusion 里写 "paving the way toward Artificial General Intelligence"，我觉得这话说得早了一点。但 "agent 应该能自己摸清陌生环境"这件事被严肃对待，本身就是 agent 领域往前走了一步。

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
