# 往 Prompt 前面拼一段 Lorem 乱码，GRPO 居然就训出来了——LoPE 是怎么靠"废话"破开零优势困境的

做过 GRPO 训练的朋友都知道一个让人头疼的现象：一道难题，rollout 八条全错，advantage 直接归零，这一题就白白浪费掉——既没贡献训练信号，又把推理预算烧得干干净净。题目越难，浪费越多。这就是俗称的 **zero-advantage problem**。

常见的处理方式有两类：一是多采样几次（adaptive rollout budget），二是把温度调高一点，多探索一些。但前者只是把同一个分布多摇几次，命中率被模型当前策略卡死；后者只是在 logit 空间里抖一抖，跳不出原有的 reasoning basin。

这篇来自 Washington University in St. Louis 的论文 **LoPE: Lorem Perturbation for Exploration** 给了一个让我看完愣了一下的方案——**在 prompt 前面拼一段毫无意义的 Lorem Ipsum 伪拉丁乱码，再去重新采样**。就这么个粗暴到像恶作剧的改动，居然在 Qwen3-1.7B / 4B、Qwen2.5-Math-7B 上分别拿到 +2.79 / +4.62 / +6.20 的平均提升。

更耐人寻味的是后面那一章分析——作者扫了一圈各种"乱码"扰动（随机 ASCII、随机 token、English Unigram、Latin Unigram、3-gram……），结论是**只有困惑度低的伪拉丁乱码能涨点，纯噪声反而砸盘**。这说明它真不是"加噪声"这么简单，背后有更细的机制。

下面把这篇论文掰开揉碎讲一讲。

---

## 论文信息

- **标题**：Nonsense Helps: Prompt Space Perturbation Broadens Reasoning Exploration
- **作者**：Langlin Huang, Chengsong Huang, Jinyuan Li, Donghong Cai, Yuyi Yang, Jiaxin Huang
- **机构**：Washington University in St. Louis
- **提交时间**：2026 年 5 月 7 日
- **arXiv**：[https://arxiv.org/abs/2605.05566](https://arxiv.org/abs/2605.05566)

---

## 一、先把"零优势"这个坑讲清楚

GRPO 的核心是用同一道题的多个 rollout 之间的相对正确率来算 advantage。具体说，对一个 query $q$ 和 prompt $p$，从旧策略 $\pi_{\theta_{\text{old}}}$ 采样 $G$ 条回答 $\{o_i\}_{i=1}^G$，每条算出一个 reward $r_i$，advantage 用组内归一化得到：

$$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}, \quad \mathbf{r} = [r_1, \dots, r_G]$$

这个设计很优雅——不用 value model，靠组内相对比较就能算策略梯度。但优雅的代价也很明显：**一旦 $G$ 条全错（或者全对），方差归零，advantage 全部塌成 0，这一题对梯度的贡献就是零**。

在数学推理这种本来就难的任务上，这个问题特别痛。Qwen3-1.7B-Base 在 OpenR1-Math 数据集上初始能解 500 道里的 148 道，剩下 352 道全部进入零优势黑洞。直接增大采样数 $G$ 是常见的处方，但作者点了一句很要害的话：**这些题就是难，多摇几次也是在同一个 reasoning basin 里打转，命中率提升非常有限**。

接下来一段顺手补一下背景。在 RLVR 这条线上，Yan et al. (2025) 的 LUFFY、还有几篇关注 rollout budget 自适应分配的工作（Liao、Li、Xiong 2025 那一波）思路都是"难题多采点样本"。但它们解决的是预算分配的问题，没解决**采样分布本身的探索面太窄**的问题。LoPE 正是从这里切进去的。

---

## 二、核心假设：在 prompt 空间扰动，比在 logit 空间扰动更能解锁正交推理路径

作者的假设其实挺干脆的——

> **logit 空间的高温采样只能在原有分布上轻微抖动，prompt 空间的扰动才能真正把模型推到一个不一样的输出分布上。**

为什么？因为模型在长上下文条件下的输出概率分布对 prompt 上下文极度敏感（这一点 In-Context Learning 那一系列工作已经反复验证过，比如 Xie et al. 2022、Dai et al. 2023）。改一改 prompt，相当于把模型放进一个不同的"初始信念"里，它在生成时走的链路自然就变了。

但问题来了——**怎么扰动 prompt，才能既改变输出分布、又不引入对任务的误导？**

直接加问题相关的提示？那等于偷偷喂答案。
加随机英文句子？英文是模型的"主语言"，会直接干扰对题目的理解。
加纯噪声 token？模型可能整个就懵了。

作者祭出的方案漂亮得有点反直觉——**用 Lorem Ipsum**。

Lorem Ipsum 是排版界的老朋友：一段伪拉丁占位文本，长得像自然语言（有词长、有句子结构、统计性质接近），但没有任何语义。Python 里有 `python-lorem` 包，从 63 个拉丁词里随机采样就能拼出一段。把这玩意儿拼到 prompt 前面，模型既看不懂它（语义上），但又不会被它带偏（它什么都没说），同时它确实改变了上下文。

这个 setup 真的挺精巧的。

---

## 三、Pilot Study：Venn 图告诉你 Lorem 扰动到底解锁了什么

光说不行，得看证据。作者先做了一个非常小但很有说服力的 pilot study。

在 Qwen3-1.7B-Base 上，从 OpenR1-Math 抽 500 道题，对比三种策略的 Pass@8：

1. **Naive Prompt（Base）**：原 prompt + 温度 0.6
2. **Naive Prompt（High-temp）**：原 prompt + 温度 1.2（logit 空间扰动）
3. **Lorem-perturbed Prompt**：前面拼随机 Lorem Ipsum + 温度 0.6（prompt 空间扰动）

然后画 Venn 图，看三种策略各自能解出哪些题、重合度如何。

![图 1：LoPE 框架总览。当 G=8 条 rollout 全错时，LoPE 在 prompt 前拼一段随机 Lorem Ipsum 序列，重新采样 G' 条回答；从中挑出成功的回答与原始失败回答一起重组成一个 size 为 G 的 mixed batch，进入策略更新。](https://arxiv.org/html/2605.05566v1/x1.png)

*图 1：LoPE 的整体流程示意。注意三个关键点：(1) 触发条件是"原 prompt 下 G 条全错"，不是无差别 resample；(2) 用扰动 prompt 重采 G'=24 条；(3) 重组 batch 时保留至少一个错误响应，保证组内 advantage 非零。*

下面是 Pilot Study 的核心结果——

![图 2：500 题子集 Venn 图。三种 prompting 策略（naive、Lorem 扰动、高温）各自的 Pass@8 成功题集合，颜色越深表示重叠度越高。Lorem 扰动解锁了一批 naive prompt 完全摸不到的难题。](https://arxiv.org/html/2605.05566v1/x2.png)

*图 2(a)：在 500 题全集上，Lorem 扰动新增解出的题目里有相当一部分是 naive prompt 和高温采样都完全错失的——也就是说 Lorem 把模型推到了一片它本来根本到不了的推理区域。*

![图 3：352 题难题子集（即 naive prompt 下 Pass@8 失败的题目）上的 Venn 图。Lorem 扰动独占的成功题数明显多于高温采样。](https://arxiv.org/html/2605.05566v1/x3.png)

*图 2(b)：在 naive 失败的 352 道难题上对比更明显——Lorem 扰动的"独占解出"区域显著大于高温采样。看到这张图我其实有点意外，本来以为高温和 Lorem 是程度差异，结果是路径差异。*

这两张图基本把作者的核心 claim 直接钉死了：**Lorem 扰动确实开辟了一条 logit 空间根本到不了的探索方向**。

更细的证据是 Figure 3，作者把三种策略下生成回答的 entropy 和 perplexity 分布画了出来——

![图 4：不同 prompt 形式下，生成回答的 entropy（左）和 perplexity（右）的分布对比。Lorem 扰动让模型在生成 token 时"更不确定"，但对应的 perplexity 没有炸掉，说明它扩大了探索面但没破坏语言流畅性。](https://arxiv.org/html/2605.05566v1/x4.png)

*图 3：高温采样让 entropy 整体右移（变更不确定）但 perplexity 也跟着右移（流畅度下降）；而 Lorem 扰动是 entropy 右移、perplexity 保持得很好——这就是"既扩展探索又不破坏生成质量"的视觉证据。*

我对这个观察最大的感受是：**它把 logit 扰动和 prompt 扰动的本质差异给画明白了**。logit 扰动是把分布拍平，所有方向都更随机；prompt 扰动是把分布平移到另一个 mode 附近，方向变了但锐度没变。前者是"乱"，后者是"换条路"。

---

## 四、LoPE 的训练流程：三个工程细节

讲清楚直觉，接下来看具体训练流程。LoPE 整体沿用 GRPO，但在 rollout 阶段插入三个改动。

### 1. 触发条件：只在零优势时启动

这点很关键。LoPE **不是无脑给每道题都加扰动**——只对那些"$G$ 条 rollout 全错"的难题，才用扰动 prompt $\delta \oplus p$ 重新采样 $G'=24$ 条回答。

实验里 $G=8$，$G'=24$。简单题不浪费预算，难题集中砸资源在"探索新路径"上。

### 2. 重组 batch：成功重采替换失败原采，保留至少一个失败响应

resample 拿到的成功响应记数 $c$，作者从中随机选 $N_s = \min(c, G-1)$ 条替换掉原 batch 里 $N_s$ 条失败响应。

注意 $G-1$ 这个上限——**它强制 batch 里至少有 1 条失败响应**。这是为了让组内 reward 标准差非零，advantage 才有意义。要是把 batch 替成全对，又退回零方差了。

### 3. Pseudo Rollout + Importance Sampling 修正

最容易被忽略但很要命的一点：**重采的响应来自带扰动的旧策略，但训练时 LoPE 把它当作不带扰动的样本来训**——也就是说，扰动 $\delta$ 只在采样时出现，训练时丢掉。形式上，采样分布是 $\pi_{\theta_{\text{old}}}(o' \mid \delta \oplus p, q)$，训练时却把 $o'$ 视为来自 $(p, q)$ 的样本。

这是 off-policy 训练。要做无偏估计就得加 importance sampling ratio：

$$\rho_{i,t} = \frac{\pi_\theta(o'_{i,t} \mid p, q, o'_{i,<t})}{\pi_{\theta_{\text{old}}}(o'_{i,t} \mid \delta \oplus p, q, o'_{i,<t})}$$

分子是没有扰动时的概率，分母是有扰动时的旧策略概率。这一步保证了"虽然采样时模型见过 Lorem，但训练时学的是不带 Lorem 的能力"。

实测里作者还顺手把 KL 正则 $\beta D_{\text{KL}}$ 也关掉了，原因是 resample 设置本来就和 reference policy 距离更远，强行拉回反而压死探索。

---

## 五、Training Signal Shaping：让 off-policy 训练不被稀释

光做 importance sampling 还不够。off-policy 训练有一个老问题——**对于在新策略下概率很低的 token，IS 比值 $\rho_{i,t}$ 极小，梯度被严重压制**。但这些 token 偏偏就是 resample 想引入的"罕见但正确"的关键步骤。

作者沿用 Yan et al. (2025) 的 **Policy Shaping**，把 IS 比值经过一个非线性函数：

$$f(\rho_{i,t}) = \frac{\rho_{i,t}}{\rho_{i,t} + \gamma}, \quad \gamma = 0.1$$

这个函数的好处是：
- 当 $\rho \to 0$ 时（即 $\pi_\theta$ 很小），$f$ 还是接近 0 但更平缓，不会把梯度压到完全消失
- 当 $\rho$ 较大时，$f$ 趋近 1，梯度被有界控制住

结果是把梯度峰值挪到了**低概率区域**——也就是模型当前不熟练的那些 token——而不再集中在已经掌握的 token 上。

![图 5：每 token 梯度权重对比。左：vanilla GRPO 梯度，低概率区域消失；中：GRPO clipping 后梯度，被硬截断；右：policy-shaped 梯度，峰值搬到低概率区域且峰值有界 1/4。](https://arxiv.org/html/2605.05566v1/x11.png)

*图 9（论文中编号）：三种梯度形态的对比。policy shaping 是把"学习重心"从已经掌握的 token 拨到模型不熟悉但被 reward 验证过的 token 上——这正是 off-policy resample 想要的。*

除此之外作者还做了个 **Advantage Shaping**——在算 advantage 时，把被丢弃的 $G'$ 条 rollout 也纳入均值/方差的计算，而不是只在保留的 $G$ 条里算。原因很直白：**被丢弃的几乎全是失败 rollout，把它们也算进去能更真实地反映"这道题有多难"**，从而给那些罕见的正确响应放大 advantage 信号。

![图 6：advantage shaping 的量化效果。左：correct 响应的绝对 advantage 值（蓝：vanilla，橙：shaped），c 越小（题越难），shaped advantage 越大；右：放大倍数。](https://arxiv.org/html/2605.05566v1/Figures/advantage_shaping_v2.png)

*图 10（论文中编号）：c 是 resample 出的正确响应数量。当 c=1（极难题，32 条只对一条）时，shaped advantage 几乎是 vanilla 的 2 倍。这一脚补在难题上是真的精准。*

数学上，shaped advantage 对正样本是：

$$\hat{A}^{+} = \sqrt{\frac{(G+G')-c}{c}}$$

可以验证，$c$ 越小，$\hat A^+$ 越大，符合"越难的题、越罕见的正确响应、越值得放大学习信号"的直觉。

---

## 六、主实验结果：在三个尺度上稳定上分

主实验在 Qwen3-1.7B-Base、Qwen3-4B-Base、Qwen2.5-Math-7B 上做，benchmark 覆盖 MATH-500、GSM8K、AMC、AIME 2024、AIME 2025。

| 模型 & 方法 | MATH-500 | GSM8K | AMC | AIME24 | AIME25 | **平均** |
|---|---|---|---|---|---|---|
| **Qwen3-1.7B-Base** | 63.40 | 76.92 | 26.87 | 5.33 | 2.00 | 34.90 |
| + GRPO | 64.20 | 82.71 | 27.61 | 6.15 | 4.47 | 37.03 |
| + Resample (Naive Prompt) | 67.00 | 82.18 | 28.36 | 8.70 | 4.58 | 38.16 |
| **+ LoPE（无 Shaping）** | 68.00 | 83.55 | 33.58 | 7.97 | 5.83 | 39.79 |
| **+ LoPE（有 Shaping）** | 68.80 | 82.94 | 32.84 | 8.80 | 5.73 | **39.82** |
| **Qwen3-4B-Base** | 65.80 | 82.71 | 32.84 | 9.38 | 7.24 | 39.59 |
| + GRPO | 77.80 | 91.74 | 47.76 | 16.41 | 13.12 | 49.37 |
| + Resample (Naive Prompt) | 79.80 | 92.87 | 45.52 | 14.90 | 11.67 | 48.95 |
| **+ LoPE（无 Shaping）** | 85.40 | 92.95 | 52.99 | 19.01 | 13.85 | 52.84 |
| **+ LoPE（有 Shaping）** | 82.60 | 92.95 | 58.21 | 19.90 | 16.27 | **53.99** |
| **Qwen2.5-Math-7B** | 52.80 | 65.50 | 35.40 | 12.90 | 7.90 | 34.90 |
| + GRPO | 78.00 | 85.06 | 47.76 | 17.66 | 9.90 | 47.68 |
| + Resample (Naive Prompt) | 78.20 | 83.02 | 50.00 | 17.19 | 9.17 | 47.52 |
| + LoPE (无 Shaping) | 77.40 | 86.35 | 47.01 | 15.31 | 10.52 | 47.32 |
| **+ LoPE（有 Shaping）** | 81.80 | 90.30 | 61.19 | 19.58 | 16.51 | **53.88** |

几个值得停下来盯一会儿的数：

- Qwen3-4B 上 **AMC 从 47.76 涨到 58.21（+10.45 个点）**，AIME25 从 13.12 涨到 16.27（+3.15）。这种幅度在已经训好 GRPO 的 baseline 上加的，含金量挺高。
- Qwen2.5-Math-7B 上，**LoPE 无 Shaping 居然不如 baseline**（47.32 vs 47.52），但加上 Training Signal Shaping 之后跳到 53.88（+6.20）。这说明 7B 这种参数更大的模型，off-policy 训练的 gradient suppression 问题更严重，**Shaping 不是可选项，是必选项**。
- naive prompt resample 自己也能涨点（如 1.7B 从 37.03 到 38.16），但和 LoPE 比还差一截。证明涨点不全是"多采几次"的功劳，**prompt 扰动本身的探索面才是关键贡献**。

训练曲线也很说明问题——

![图 7：Qwen3-1.7B-Base 训练过程中 resample 的成功率（左）和模型在评测上的准确率（右）。LoPE 的 resample 成功率从训练初期开始就显著高于 naive prompt，差距贯穿整个训练过程。](https://arxiv.org/html/2605.05566v1/x5.png)

*图 4：左图蓝线 (LoPE) 始终高于橙线 (naive resample)——说明 Lorem 扰动是稳定提升"难题救活率"的，不是偶发涨点。右图准确率曲线也持续保持领先。*

---

## 七、最有意思的一章：什么样的扰动才"好用"？

如果到这里你还没有疑问，那是不太够的。我看完前面六章心里的第一个反应是——**凭什么是 Lorem Ipsum？随便加点别的乱码不行么？**

作者也意识到这是必答题。第 7 章他做了一个完整的扰动对比实验，扫了 8 种不同的 perturbation 方案：

- **Random Fake English**：用 Faker 包生成假英文句子
- **Random ASCII**：随机可打印 ASCII 字符
- **Random Tokens**：从模型词表里均匀采 token
- **English Unigram Model**：从 C4 英文语料里前 50 高频词均匀采
- **Latin Unigram Model**：从 C4 拉丁语前 50 高频词均匀采
- **Latin 3-Gram Model**：在 C4 拉丁语料上训的 3-gram LM 生成的序列
- **Filtered Latin Natural Language**：真实的拉丁语自然文本

下面是两个核心图。

![图 8：各种扰动序列在 Qwen3-1.7B-Base 上的困惑度分布。前两行（Lorem、Filtered Latin、Latin Unigram、Latin 3-Gram）困惑度低，接近自然语言；最后一行（Random ASCII、Random Token）困惑度爆表。](https://arxiv.org/html/2605.05566v1/x6.png)

*图 5：困惑度高低是核心轴。Lorem Ipsum 的困惑度均值是 6.86，和真实拉丁自然语言（5.79）非常接近；而 Random Token 困惑度均值高达 11086，纯属 OOD 噪声。*

![图 9（左）：500 条问题 prompt 在各种扰动下的 entropy 分布。Random Token 明显右移，说明它把模型对问题的理解搞乱了；其他扰动基本保持原分布形状。](https://arxiv.org/html/2605.05566v1/x7.png)

*图 6(a)：Random Token 的 entropy 分布右移幅度极大——这就是为什么它会"砸盘"，模型连题目都看懵了。*

![图 9（右）：t-SNE 可视化不同扰动下问题表示的偏移。每种颜色代表一道题（8 个采样），Random Token 漂得很远，Lorem/Latin 类基本聚在原位附近。](https://arxiv.org/html/2605.05566v1/x8.png)

*图 6(b)：t-SNE 也佐证——Lorem 类扰动让模型对题目的表示发生"轻微偏移"，Random Token 则发生"剧烈漂移"。前者是换条路看题，后者是把题给毁了。*

把这些扰动放到训练里跑一圈，得到下面这张关键表——

| 方法 (Qwen3-1.7B-Base) | MATH-500 | GSM8K | AMC | AIME24 | AIME25 | **平均** |
|---|---|---|---|---|---|---|
| Base + GRPO | 64.20 | 82.71 | 27.61 | 6.15 | 4.47 | 37.03 |
| + Resample (Naive Prompt) | 67.00 | 82.18 | 28.36 | 8.70 | 4.58 | 38.16 |
| + Naive Prompt (Temp=1.2) | 64.40 | 82.87 | 31.34 | 8.65 | 4.48 | 38.35 |
| **+ LoPE** | 68.80 | 82.94 | 32.84 | 8.80 | 5.73 | **39.82** |
| + Random Fake English | 65.80 | 81.96 | 32.09 | 7.50 | 5.42 | 38.55 |
| + Random ASCII | 66.20 | 82.94 | 28.36 | 8.12 | 5.32 | 38.19 |
| + Random Token | 64.20 | 81.50 | 29.85 | 8.08 | 4.63 | **37.65（伤害训练！）** |
| + Filtered Latin Natural Language | 68.80 | 82.71 | 32.84 | 9.32 | 5.57 | **39.85** |
| + Latin Unigram Model | 69.40 | 83.32 | 32.09 | 7.19 | 6.35 | **39.67** |
| + Latin 3-Gram Model | 68.80 | 81.88 | 29.85 | 7.92 | 5.93 | 38.88 |
| + English Unigram Model | 67.00 | 83.32 | 28.36 | 8.49 | 5.42 | 38.52 |

读到这里我才完全理解作者想说什么。结论可以精炼成两条铁律：

1. **必须是"伪拉丁"——也就是模型主语言（英文）之外的语言**。Random Fake English / English Unigram 是英文乱码，效果都明显弱于拉丁系扰动。原因很可能是英文乱码会真的干扰模型对英文题目的语义理解，而拉丁语对当代 LLM 是"看着像但读不出意思"的状态，恰好达到"扰动分布但不破坏理解"的临界点。
2. **必须低困惑度**。Random Token 困惑度 10000+，直接把训练带跑偏；Random ASCII 也比较高，效果一般。三个困惑度最低的——Lorem、Filtered Latin、Latin Unigram——分别拿到 39.82 / 39.85 / 39.67，是 top 3。

归根到底这事儿不能太离谱也不能太规整。**轻度扰动 + 非主语言**才是 sweet spot。

我看到这里其实挺被打动的——这不是一个堆 trick 的论文，作者真的把"为什么 work"挖到了底。比起"加了个新模块涨了几个点"，这种"剥到只剩一个清晰原则"的工作更有价值。

---

## 八、几点我自己的判断

读完整篇我整理了几个观察：

**亮点**：

- **idea 漂亮**。在 prompt 空间做扰动这个方向其实早有人提（Xie et al. 2022 关于 ICL 的工作，von Oswald 2023 等），但把它具体落到"用 Lorem Ipsum 解 GRPO 零优势"这个 RLVR 实际痛点上，是这篇的新意。**思路简单但锋利**。
- **机理分析做得好**。perplexity 分布、t-SNE、entropy 分布、Venn 图，把"为什么 work"和"什么样的扰动才 work"分开讨论，论据链很硬。这一章是这篇论文的灵魂。
- **工程上易部署**。LoPE 不改模型结构、不改 GRPO loss 形式、不额外训其他组件，就是在 rollout 阶段加一段拼接。任何在跑 GRPO 的团队都可以一周内试出来效果。
- **Training Signal Shaping 是真功夫**。Qwen2.5-Math-7B 上的对照实验很说明问题——没有 Shaping，LoPE 反而比 baseline 差；有了 Shaping，立刻跳到全场最高。这一节的梯度分析挺扎实，没有糊弄。

**值得追问的几点**：

- **Lorem 序列长度 100–300 token，这个选择是经验值**。论文里没系统扫这个超参，我猜应该有一个甜区，但目前看不出对短序列的鲁棒性如何。
- **泛化到非数学任务怎么样？** 整篇实验全是数学推理，作者没在代码生成、open-ended QA 这种任务上验证。但我自己倾向于认为这套方法在依赖"verifiable reward"的任务上都有戏，因为它没用到任务相关信息。
- **boundary instruction 那个 trick 值得注意**。作者在扰动末尾会拼一句 "\nPlease reason step by step, and put your final answer within \boxed{}."，说是为了避免模型被扰动带歪输出乱码。这其实是个小补丁，没有这个 instruction，效果可能会打折——这点论文没单独消融，是个小遗憾。
- **算力开销其实不小**。$G'=24$ 意味着难题要额外多采 24 条 rollout——只在零优势触发，预算开销随训练进展而下降，但前期训练阶段开销显著。论文里没正面报告总 token 消耗对比。

**和工业界已有方案的关系**：

DAPO（Yu et al. 2025）那一批方法走的是 token-level loss、改裁剪边界的思路，跟 LoPE 是正交的；Yan et al. 2025 的 LUFFY 用了 policy shaping，LoPE 直接借用过来，这点作者也明确致谢了。所以 LoPE 不是从零造轮子，而是**把一个被忽略的角度（prompt 空间扰动）做透，并且和已有 off-policy 修正手段干净地拼起来**。我个人觉得这种工作的价值反而比一个全新但难复现的方法更高。

---

## 九、对实战的几点启发

如果你也在跑 GRPO 或者类似的 RLVR 训练，这篇论文里有几个 takeaway 可以直接用：

1. **零优势是浪费的元凶**。先去日志里查一下每个 step 里有多少题进入零优势——这个比例越高，LoPE 这类方法的边际收益越大。
2. **logit 空间扰动可能不是你想要的**。如果你之前在调温度、调 top-p 还不见效，换个思路——试试 prompt 空间。
3. **扰动序列要"非主语言 + 低困惑度"**。中文模型上跑可以试试拉丁、希腊、世界语之类的低困惑度异语言；不要用纯随机 token。
4. **off-policy 训练记得加 policy shaping**。如果你的训练涉及 importance sampling 修正，$f(x) = x/(x+\gamma)$ 这个简单变换非常便宜，效果显著。
5. **batch 重组时保留至少一个失败响应**。这是个看似小但很要命的工程细节，否则方差归零回到原点。

---

## 收尾

LoPE 这篇让我想到 Karpathy 之前说过的那句话——"Sometimes the simplest hack works." Lorem Ipsum 作为排版占位符存在了 500 年，没人想到有一天它会在 LLM RL 训练里救命。

但漂亮的不是 Lorem Ipsum 本身，而是作者把"为什么是它"挖到了底——伪拉丁、低困惑度、保留语言结构、不携带任务信息——这四个条件恰好构成一个理论上可被解释的扰动空间。我猜接下来会有一批工作沿着"prompt 空间扰动"这条线展开，比如自动学习扰动的策略、跨语言扰动的设计、动态扰动强度调节等。

如果你也在 RLVR 这块踩坑，这篇论文值得花一小时把方法和分析章节都过一遍——尤其是第 7 章的扰动对比实验，是这两年我看过的"为什么 work"分析里做得最干净的之一。

觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我
