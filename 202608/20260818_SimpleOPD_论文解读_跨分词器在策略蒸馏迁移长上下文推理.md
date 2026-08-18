# SimpleOPD：分词器不同也能蒸，一招驯服长上下文蒸馏的"话痨失控"

> 论文：https://arxiv.org/abs/2608.14277
> 机构：SU-01 Team，上海人工智能实验室
> 日期：2026 年 8 月 14 日

## 核心摘要

把一个能连续写 10 万 token 证明过程的长上下文老师，蒸进一个短上下文学生，听起来很美好，实操却全是坑：两边分词器不一样、token 级监督对不上；学生学着学着回复越写越长、疯狂截断、甚至塌缩成单 token 死循环。SimpleOPD 给出的方案朴素得有点可爱——在共享文本空间上只做"边界完全对齐"的 token 监督，对齐不上的就不管；再用特殊终止符优势掩码和学生参考 KL 损失把生成长度摁住。结果相当能打：Intern-S2-Preview 在 ProofBench 上从 34.0 涨到 55.2，**涨了 21.2 个点**，反超 Gemini-2.5-Pro 和 GPT-5。这篇论文的价值不在理论多新，而在于把"跨家族、跨分词器 OPD"这条没人走通的路，用几个工程上可复用的 trick 趟平了。

---

## 🎯 一个真实的蒸馏困境

你有没有试过这种操作：手里有个超强的推理老师（比如能拿 IMO 金牌水平的模型），想把它"压缩"进一个便宜好用的学生里。

常规思路是 on-policy distillation（OPD）——学生自己采样轨迹，老师在学生生成的每个 token 上打分，给密集的 token 级监督。这比离线蒸（SFT 老师生成的固定数据）更少遗忘、泛化更好，Thinking Machines 那篇博客和 MiniLLM 之后，这条路基本成了共识。

但之前的工作几乎都在同一个模型家族、同一个词表里玩。一旦你换成"长上下文老师 + 短上下文学生 + 不同家族"，四个麻烦一起来：

- **分词器对不上**。老师是 Qwen 系的 byte-level BPE，学生可能是 SentencePiece（Gemma），同一段文本切出来的 token 序列完全不同，token 级 KL 根本没法算。
- **分布不匹配**。老师习惯写超长推理链，学生没这个习惯，硬学容易崩。
- **长度爆炸**。学生学着学着回复越写越长，截断率飙升。
- **训练不稳定**。论文附录里的 case 看得我头皮发麻：一个模型答对 AIME 题之后，把同一段十句自检话原样重复了 972 遍，撞满 160K token 上限；另一个直接塌缩成单 token 循环，"2^?"刷了 3.4 万次。

这篇论文就是来收拾这个烂摊子的。

## 📖 论文信息

- **标题**：SimpleOPD: Simple Tokenizer-Agnostic On-Policy Distillation for Long-Context Reasoning
- **作者**：Haonan He、Haodi Lei（共同一作）、Yun Luo（项目lead）、Haoran Zhang、Shunkai Zhang、Yizhuo Li、Shengji Tang、Zhilin Wang、Runzhe Zhan、Lei Bai、Ganqu Cui、Fangchen Yu、Yafu Li、Peng Ye、Ning Ding、Yu Cheng
- **机构**：SU-01 Team，上海人工智能实验室
- **老师模型**：SU-01，基于 Qwen3-30B-A3B 的 30B-A3B 推理模型，奥赛金牌水平，能对难题持续输出超过 100K token 的自然语言推理

## 🧠 方法核心：对齐不上的，就别硬凑

![SimpleOPD 方法总览](https://arxiv.org/html/2608.14277v1/simpleopd_intro.png)

*图 2：SimpleOPD 总览。学生用自己的分词器生成回复，老师在共享文本空间上对"边界完全对齐"的 token 计算优势（Eq.7），再用 PPO clipped 目标更新（Eq.9）；KL 约束让学生始终呆在参考策略附近的"信任域"里，不跑出界。*

核心思路一句话：**蒸馏发生在共享文本空间，只在两个分词器切分边界完全一致的 token 上做监督，其余位置不碰。**

### 跨分词器对齐：双指针扫描

形式化地说，定义学生和教师各自 token 之前的累计文本前缀 $P_\theta(t)$ 和 $P_\phi(i)$，一个对齐对 $(i, t)$ 要同时满足：

$$\mathcal{M}=\{(i,t): P_\phi(i)=P_\theta(t)\ \land\ \tau_\phi(z_i)=\tau_\theta(y_t)\}$$

也就是两边消耗的文本前缀相同、且当前 token 覆盖的文本片段完全相同。部分重叠的 token 直接放弃——因为一个老师 token 的 log 概率没法唯一地拆给多个学生 token，反之亦然。

实现上就是一个线性双指针扫描：维护两边的前缀，谁消耗得少谁就前进，碰上了就标记对齐。这个设计我挺喜欢的——它不要求任何分词器改造、不需要词表映射，天然是偏一对一映射，而且漏掉的监督信号比想象中少（后文 Figure 9 的 lexical overlap 曲线显示对齐 token 比例一开始就不低，训练中还在涨）。

### 目标函数：对齐位置上的 reverse-KL 替身

蒸馏目标定义在对齐位置上：

$$\mathcal{L}_{\mathrm{Distill}}(\theta)=\mathbb{E}_{y\sim\pi_\theta}\left[\sum_{t=1}^{n}\log\pi_\theta(y_t\mid c_\theta,y_{\lt t})-\widetilde{\ell}_{t}^{\phi}\right]$$

分词器相同时它退化为标准的 reverse KL $D_{\mathrm{KL}}(\pi_\theta \| \pi_\phi)$。为了能在同一批 rollout 上做多步更新，作者把优势写成固定形式 $\widehat{A}_t = \widetilde{\ell}_t^\phi - \log\pi_{\theta_{\text{old}}}(y_t \mid c_\theta, y_{\lt t})$，然后套 PPO 的 clipped 目标（clip 系数 0.2）。

![逐 token reverse KL 可视化](https://arxiv.org/html/2608.14277v1/token_kl_visualization.png)

*图 10：跨词表蒸馏下的逐 token reverse KL 可视化。黄色是未对齐 token（无教师信号），其余按 KL 值从蓝到红着色。可以看到大量数学符号 token（$、\omega、编号等）都能对齐，而未对齐的主要是 ". \n" 这类跨边界切分歧义的位置——损失的信号集中在格式性 token 上，推理内容基本保住了。*

### 驯服长度爆炸：两个小 trick

直接 OPD 的训练曲线相当难看：Intern-S2-Preview 的截断率和重复率随训练飙升，Qwen3.5-35B-A3B 更惨，性能直接退化。

作者的拆弹步骤是递进的，这个排查过程本身就值得读：

**第一招：特殊终止符优势掩码。** 把 `</think>`、`<|im_end|>` 这类终止 token 的优势直接 mask 掉。直觉很简单——如果老师在这些位置的打分被当成优势参与更新，学生会被诱导去扭曲"何时该停"这个关键决策。掩码之后长度相关的失稳缓解了，但单靠它解决不了长度膨胀。

**第二招：学生参考 KL 损失。** 加一个把学生策略往初始参考策略上拉的 KL 项，系数 0.5（Qwen 系和 Intern-S2）或 1.0（GLM/Gemma）。这一招下去，截断率直接压到接近零，AIME25 和 AnswerBench 还稳步上涨。

说实话，这两招单拎出来都不算新——KL 正则和特殊 token 处理都是 RL 后训练里的常见操作。但把它们用在"跨分词器 OPD 长度失控"这个具体病上，而且剂量给得恰到好处，这就是工程经验的价值。

## 🔬 实验设置

- **训练数据**：全是数学证明题，共 4528 道——OPC 63 道、AoPS 社区 2948 道、竞赛训练书 900 道、数之谜论坛和陈谊（Evan Chen）奥赛材料 617 道。
- **训练配置**：Slime 框架 + SGLang 推理，100 个 rollout 迭代，恒定学习率 1e-6，rollout batch 64，每 prompt 采 4 条，Qwen 系最长 32K token、GLM/Gemma 为 6K，每步 rollout 更新 4 次。
- **评测**：ProofBench（DeepSeek-V4-Flash 当评委，同一条证明评 4 次取平均）、AnswerBench、AIME25、AMOBench；规则验证失败时上 GPT-OSS-120B 兜底。生成配置 temperature 1.0、最长 160K token。
- **学生阵容**：同分词器的 Qwen3-4B / Qwen3-30B-A3B；跨分词器的 Qwen3.5-4B、Qwen3.5-35B-A3B、Intern-S2-Preview、GLM-4.7-Flash、Gemma-4-26B-A4B。

## 📊 主结果：证明能力几乎"无损搬运"

| 模型 | ProofBench@4 | AnswerBench@8 | AIME25@8 | AMOBench@8 |
|---|---|---|---|---|
| SU-01（老师） | 45.00 | 77.50 | 94.60 | 61.75 |
| Qwen3-4B | 11.42 | 47.50 | 71.25 | 23.00 |
| Qwen3-4B-OPD | 23.72（+12.30） | 64.50（+17.00） | 90.83（+19.58） | 35.00（+12.00） |
| Qwen3-30B-A3B | 13.80 | 59.13 | 88.33 | 36.50 |
| Qwen3-30B-A3B-OPD | 36.47（+22.67） | 74.46（+15.33） | 93.75（+5.42） | 52.75（+16.25） |
| Qwen3.5-35B-A3B | 26.78 | 73.16 | 94.60 | 57.25 |
| Qwen3.5-35B-A3B-OPD | 42.39（+15.61） | 80.15（+6.99） | 96.66（+2.06） | 61.25（+4.00） |
| Intern-S2-Preview | 21.70 | 76.03 | 88.33 | 58.00 |
| Intern-S2-OPD | 44.50（+22.80） | 80.10（+4.07） | 95.00（+6.67） | 59.50（+1.50） |

几个值得停一秒的数字：

**Intern-S2-OPD 的 ProofBench 从 21.70 涨到 44.50，离老师的 45.00 只差 0.5 分**——证明能力几乎是整个搬过去了。更夸张的是它的 AnswerBench 80.10 和 AIME25 95.00 双双反超老师。学生青出于蓝，说明 OPD 不只是复制，还保住了学生自己的底子。

用 Gemini-2.5-Pro 当评委重评 ProofBench（与 SU-01 论文同设置），Intern-S2-OPD 从 34.0 到 55.2，**反超 Gemini-2.5-Pro 和 GPT-5**，虽然还落后 SU-01 和 DeepSeek-V3.2-Speciale，但差距明显收窄。

跨家族那边，GLM-4.7-Flash 的 ProofBench 从 30.8 涨到 39.7、AnswerBench 69.6→72.0；Gemma-4-26B-A4B 的 ProofBench 25.5→34.2，但 AnswerBench 微跌 68.8→67.5。作者的解释很坦率：Gemma 用 SentencePiece，和老师的 byte-level BPE 差距更大，对齐上的监督信号更少。这个细节反而增加了可信度——他们没有回避"分词器差得越远越难蒸"这个事实。

## 🔍 分析部分的几颗珍珠

**对比 OPD 变体。** 跟 EOPD（高熵位置补 forward KL）和 G-OPD（泛化参考模型 + 奖励缩放）比，SimpleOPD 在四个基准里赢了三个，ProofBench 上优势最大；只在 AIME25 上比 EOPD 低 0.33 分，基本打平。

**蒸馏长度是硬约束。** 6K 蒸馏长度对证明推理来说不够用：Intern-S2 从 6K 加到 32K，ProofBench 从 38.80 涨到 44.50，AnswerBench 77.25→80.10。这也合理——老师动辄写几万 token 的推理链，你只给学生看 6K，相当于只让学开头。

**数据组成很挑剔。** 只喂证明题时 ProofBench 最高 44.50；混入可验证数学题后 AnswerBench 微涨到 81.10，但 ProofBench 掉到 38.50。想蒸证明能力，就得专注喂证明数据，贪多反而稀释。

**换个更大的老师也 work。** 用 158B 的 DeepSeek-V4-Flash 当老师、6K 长度蒸 Intern-S2，ProofBench 涨 18.01 分到 39.71，比同设置下 SU-01 当老师的 38.80 还高 0.91——说明方法不绑定自家老师，更强的老师收益更大。

**OOD 泛化。** 训练只用数学数据，但科学推理不掉反涨：HiPhO 38.6→41.1（还反超老师的 35.0），HLE 19.6→20.5，FrontierScience Research 1.7→5.0。蒸过去的不是"数学套路"，而是更通用的推理习惯。

**KL 系数的甜区。** GLM 上的消融：0.5 太松、1.2 太紧，1.0 最均衡（ProofBench +8.96、AnswerBench +2.38）。这提醒我们这个方法不是完全免调参的，跨家族蒸馏时这个系数得扫一下。

## 🤔 我的判断

这篇论文最值钱的地方，是把一个大家心知肚明但没人系统解决的工程问题——**跨家族 OPD 的分词器错配**——用一个近乎"偷懒"的方式解决了：对不齐就不监督。这比设计复杂的词表映射、token 合并拆分方案聪明得多，因为后者引入的近似误差未必比丢掉的信号小。

但也要有清醒认识。其一，"Simple"是双刃剑：partial alignment 意味着监督是稀疏的，图 10 里那些黄色未对齐 token 在格式敏感场景可能是隐患，论文没量化"丢掉的信号到底值多少分"。其二，长度爆炸的解决靠的是 KL 正则这个"缰绳"，说到底是在防止学生跑飞，而不是让学生真正学会老师的长推理节奏——32K 比 6K 好这个结论，某种程度上说明学生的长上下文能力天花板还在。其三，ProofBench 的评测依赖 LLM 评委（DeepSeek-V4-Flash / Gemini-2.5-Pro），评委偏好的影响没有充分消融，55.2 这个反超 Gemini-2.5-Pro 的数字，换个评委是否还成立，我持谨慎态度。

工程上如果你在搞模型压缩：这个方案的直接启发是——别再为跨家族蒸馏去对词表了，文本空间对齐 + PPO 框架就能跑；另外训练中盯紧截断率和重复率这两个指标，它们比 loss 更早暴露失控。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
