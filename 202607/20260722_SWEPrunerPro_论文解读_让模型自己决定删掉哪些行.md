---
title: 让模型自己决定删掉哪些行：SWE-Pruner Pro 如何省掉那台外置剪刀
date: 2026-07-22
arxiv: 2607.18213
authors: Yuhang Wang, Yuling Shi, Shaoqiu Zhang, Jialiang Liang, Shilin He, Siyu Ye, Yuting Chen, Kai Cai, Xiaodong Gu
project: https://github.com/Ayanami1314/swe-pruner-pro
tags: [code agent, context pruning, hidden states, line-level, SWE-Bench]
---

# 让模型自己决定删掉哪些行：SWE-Pruner Pro 如何省掉那台外置剪刀

你有没有过这种感觉——让 AI 智能体去改一个真实的代码仓库，看着它一行一行 `cat`、一行一行 `grep`，每条工具回包动辄几百行，多翻几个文件就把上下文撑爆了。你想让它少看点东西吧，又怕它漏掉关键代码；让它照单全收吧，token 账单哗哗地涨，质量反而开始掉——注意力被无用的日志、重复的 import 拖垮，写出来的 patch 越来越糊。

这个"上下文墙"的问题过去大半年已经在 SWE-Agent / Claude Code / Cursor 这条赛道上被反复讨论过。解决的思路大家也摸过了：要么硬截断前缀，要么总结，要么学一个外部的剪枝模型。最出名的就是上海交大 LLMSE 实验室同一波人今年初发的 SWE-Pruner——训一个 0.6B 的小剪枝器，每轮让 agent 自己写一句 goal hint（"我要看错误处理"），剪枝器根据 hint 给工具输出打分，留线剔行。

听起来挺合理的。但 SWE-Pruner Pro 的作者自己站出来把这条路质疑了一遍：**你每轮让 agent 多写一句 goal hint、再多跑一个 0.6B 模型，这本身就是巨大的开销。** 而且更本质的问题——agent 在 prefill 工具响应的时候，主模型的隐藏状态里**已经**知道哪些行重要了，那为什么还要再外挂一台"剪刀"？

这篇文章就是讲怎么把这台外置剪刀拆掉。

## 核心摘要

SWE-Pruner Pro 的核心观察极简：在 coding agent 用 prefill 读工具响应的时候，把 backbone 最后一层 token 隐藏状态 mean-pool 到行级，然后丢一个两层 FFN 头上去（head 只有约 18M 参数），就能直接读出"这一行该留还是该删"的信号。

省掉了一台外挂剪枝模型，省掉了 goal hint 查询，**只多一个 head forward 的代价**。

效果是这样：在 SWE-Bench Verified 上用 MiMo-V2-Flash（小米开源 MoE），resolve rate 比不加剪枝多 **3.8 个点**，input token 多 7.4%；用 Qwen3-Coder-Next 跑同一个榜单，resolve 略掉 1.2 个点，但 input token 砍掉 **13.5%**——是所有剪枝器里压得最狠的。在 Oolong 长上下文聚合任务上，MiMo-V2-Flash 还顺便涨了 **2.2 个点**准确率。最重的一个 cell，Qwen3-Coder-Next 跑 SWE-QA-Pro，prompt + completion 总 token 砍了 **39.4%**。

更关键的——额外推理开销被压到了**聚合 wall time 的 15%**（中位 14.7%），通过把 head 塞进 SGLang 调度器里（in-engine colocation）实现。

一句话评价：**这是工程上"少即是多"的一次漂亮落地**。不是架构突破，是范式重构——把"读信号"的成本从"另起炉灶"降到"白嫖 backbone"。SWE-Pruner Pro 的姊妹篇 SWE-Pruner（v1，2026.01）拿到了同样的范式红利但选了更重的实现路径，这篇算是同一个团队对自己的方案做了一次"打补丁式迭代"——补丁补得相当彻底。

---

## 论文信息

| 字段 | 内容 |
|---|---|
| 标题 | SWE-Pruner Pro: The Coder LLM Already Knows What to Prune |
| 作者 | Yuhang Wang, Yuling Shi, Shaoqiu Zhang, Jialiang Liang, Shilin He, Siyu Ye, Yuting Chen, Kai Cai, Xiaodong Gu |
| 单位 | LLMSE Lab, Shanghai Jiao Tong University（同一波人） |
| arXiv | [2607.18213](https://arxiv.org/abs/2607.18213) |
| 提交时间 | 2026/07/20 |
| 代码 | https://github.com/Ayanami1314/swe-pruner-pro |
| 姊妹工作 | [SWE-Pruner (2601.16746)](https://arxiv.org/abs/2601.16746) |

---

## 问题背景：上下文墙有多严重

先说为什么这事值得做。SWE-Bench Verified 上跑 Mini-SWE-Agent（Claude Sonnet 4.5 后端），**文件读取类工具占了 70%+ 的 token 预算**——读一次 300 行的 `__init__.py`，这条内容会一直挂在 context 里直到任务结束。GLM-4.6 后端上观察到的模式也差不多。

这是个结构性矛盾：agent 必须先把文件读全才能开始推理，但每多读一次、每多挂一轮，attention 就在被稀释。一篇 SWE-Pruner Pro 引用了 Liu et al. 2023 的结论——长上下文里的关键信息容易被"lost in the middle"现象埋掉。所以减负不是省钱问题，是质量问题。

之前大家怎么做的？三派：

| 派别 | 代表 | 思路 | 痛点 |
|---|---|---|---|
| 通用压缩 | LLMLingua2、Selective Context | 用 PPL/自信息打 token 分，砍低分 token | 不知道 agent 当前关心什么，砍掉关键代码片段的风险高 |
| 检索/截断 | RAG、cursor 的硬截断 | 用 bge-reranker 召回或者掐头去尾 | 召回是粗粒度的，压缩比低；硬截断会丢上下文 |
| 任务感知剪枝 | SWE-Pruner、LongCodeZip | 训一个外部打分模型，按行裁剪 | 要外挂一个 0.6B 模型、还要 agent 每轮写 goal hint |

SWE-Pruner Pro 的吐槽是——**前两派根本没看 agent 想要什么，第三派虽然看了，但 view 用得太贵**：外挂 0.6B 模型 + 每轮多生成一段 goal hint，加起来可能比你要剪掉的 token 还多。

那么信号到底有没有？先把 backbone 的隐藏状态挖出来看看。

## 核心洞察：keep-or-prune 信号已经在那里了

第 2 节是整篇论文我最喜欢的一段——它没有直接讲方法，而是先做了一组"线性探针"（linear probe）实验来回答一个存在性问题：**agent 主模型的隐藏状态里，到底有没有"哪些行重要"这个信号？**

实验设计不复杂：从 6 个公开数据集里收 ~2,260 条工具响应、~155k 行，每行用 Claude Sonnet 4.6 当裁判打 keep/prune 二分类标签。然后把 backbone 冻结，取每行最后一层 token 隐藏状态做 mean-pool，丢一个 Logistic Regression 上去分类。

![图1：SWE-Pruner 与 SWE-Pruner Pro 范式对比](https://arxiv.org/html/2607.18213v1/x1.png)

*图 1：左边是 SWE-Pruner 的老办法——环境 → agent → 还要再过一个外挂 Pruner；右边是 SWE-Pruner Pro——直接在 agent 内部把隐藏状态喂给一个 head，剪枝信号在 agent 读工具响应的时候已经生成好了*

结果呢？在 LDA 判别轴上，kept 和 pruned 两类的分布确实有明显的均值偏移，**AUC 0.83，best-F₁ 0.63**。这个 F₁ 看起来不高，但要知道 majority-class baseline 在 keep rate ~30% 的情况下上界是 0.46——说明隐藏状态里**确实**有这份信号，但单靠一个线性分类器压不住中间那条重叠带。

![图 2：线性探针的可分性证据](https://arxiv.org/html/2607.18213v1/x2.png)
![图 2-右：探针分数的密度分布](https://arxiv.org/html/2607.18213v1/x3.png)

*图 2：左图是 LDA 投影空间里的散点——红（pruned）和绿（kept）分得开但有重叠；右图是分数密度分布——均值差清晰可见，但中间确实有一条阴影带。结论：信号是有的，但需要非线性 head 才能榨干*

这个探针实验是整篇论文的"敲门砖"——它把"信号在不在"从猜想变成事实，然后再用事实证明线性模型不够用，从而**顺理成章**地引出 SWE-Pruner Pro 的非线形 head 设计。**这是我觉得这篇论文写得最聪明的一笔。**

---

## 方法：藏在 backbone prefill 里的 18M head

### 每轮流水线

接下来讲怎么把探针思路工程化。SWE-Pruner Pro 的"魔法"在于——它**几乎不增加 backbone 的工作量**。

![图 3：SWE-Pruner Pro 的整体流水线](https://arxiv.org/html/2607.18213v1/x4.png)

*图 3：上半部分是 agent 轨迹。每一轮，agent 发 `cₜ` 工具调用，环境返回 `rₜ`（raw tool response），backbone 走 prefill 把它压进 KV cache。**前缀 [H_{t-1}, cₜ] 已经在缓存里，所以只有 rₜ 的新 token 被 forward**，最后得到最后一层 hidden states {h₁, ..., h_L}。下半部分是 head 的工作：每个 hᵢ 过一个 score 函数，binarize 后行内多数投票得到 keep/prune 决定，pruned 后的 r̃ₜ 替换 rₜ 进入下一轮历史*

关键点在图 3 的右上角那个小细节：head **直接挂在 prefill 上**，读 backbone 已经算出来的 hᵢ，**没有让 backbone 再跑一次 rₜ**。唯一的额外代价是下一轮 prefill 时换成了更短的 r̃ₜ——但因为你下一轮要重新 forward 的就是 r̃ₜ，**还省了事**。

整个 head 的工作可以理解为：**复用 backbone 现有的 prefill forward，把它产生的副产品（hidden states）捡起来用。**

### Pruning head 长什么样

head 不复杂。架构上是一个**小 FFN**（不是简单的线性层）：

![图 4：head 的架构](https://arxiv.org/html/2607.18213v1/x5.png)

*图 4：左边是 frozen backbone 给出的最后一层 hidden states（蓝色，冰冻不更新），右边是 trainable head（橙色，要训练）。两者之间通过一个 size embedding（按行数 N 查表）加性合并，然后过一个 LayerNorm + 两层 Linear-GELU-Dropout + 最后 Linear 投影到单个 keep logit。两条线颜色对比直接读懂——不动的蓝色和要更新的橙色*

具体公式是这么几步：

**长度感知嵌入**——工具响应可能 5 行也可能 300 行，"漏一行"的代价完全不同（5 行漏 1 行是 20%，300 行漏 1 行是 0.3%）。所以 head 拿到 hᵢ 之后，先加一个**按行数 N 查表**的 embedding：

$$\tilde{h}_i = h_i + \mathbf{e}(N) \quad (1)$$

`e(N)` 用 8 个 log-spaced 的行数桶（0-2, 3-5, ..., >200），零初始化让训练开始时 `e(N) = 0`，head 退化成 length-agnostic 版本。这个 trick 看着不起眼，但消融实验里**单这一项把 judge 分数从 6.86 拉到 7.08**（Table 3）——它让 head 学会了"长响应里可以激进剪，短响应里要保守"。

**Per-token 分类器**——经典的两层 FFN，hidden width 等于 backbone 隐藏大小 d，加 LayerNorm 和 Dropout（0.4）。最后 Linear 投到 1 维，sigmoid 拿 keep 概率。

**行级决策**——训练时用 per-token 标签（line label 展开到 token），推理时用行内 token 多数投票：

$$\hat{y}_\ell = \mathbb{1}\!\left[\frac{1}{|\ell|}\sum_{i \in \ell} \mathbb{1}[p_i > \tau] > \tfrac{1}{2}\right] \quad (3)$$

`τ = 0.5`，投票比例超过 50% 才保留——简单粗暴但有效。

### 训练：per-sample balanced focal loss

损失函数是这篇论文另一个设计亮点。一般人会用 BCE 或者全语料 focal，但这俩都不行——

问题在哪？**keep-or-prune 比例在每个样本里都不一致**。有的响应是 3 行里 keep 3 行（骨架类，约 17% 的训练集），有的响应是 100 行里 keep 30 行（典型类）。全局 BCE 会过度拟合"平均 keep rate ~30%"这个先验，把 3/3 这种"全保留"样本的信号压扁；而 batch 级 focal 也救不了——它看到的是全局分布。

作者用了一个相对简单但对路的招：**per-sample balanced focal**。先在每个样本内做 focal（γ=2），然后**单独对 keep 和 prune 两类 token 求平均**，最后 0.5/0.5 加权：

$$\mathcal{L}_s = \tfrac{1}{2}\underbrace{\frac{\sum_i y_i \mathcal{L}^{\text{tok}}_i}{\sum_i y_i}}_{\text{keep branch}} + \tfrac{1}{2}\underbrace{\frac{\sum_i (1-y_i) \mathcal{L}^{\text{tok}}_i}{\sum_i (1-y_i)}}_{\text{prune branch}} \quad (6)$$

直觉就是：**每个样本的少数类都被平等对待**。一个 3/3 全保留的样本，它的 prune 分支 loss 自动被 0 覆盖，但 keep 分支要扛全部责任，迫使 head 在这种极端情况下也得给出有意义的分数。

训练本身不贵——22,609 条样本在 8×H200 上 15 分钟训完，backbone 完全冻结，feature 提前抽好缓存好。

---

## 实验：质量没掉，token 真的省了

### 主实验：4 个 benchmark × 2 个 backbone

最大的表（Table 1）覆盖了三个 read-only 多轮 benchmark：SWE-QA、SWE-QA-Pro、Oolong。基线 7 个：No Pruning、LLMLingua2、Selective Context、RAG、Self-Prune、LongCodeZip、SWE-Pruner。下面是压缩比和质量的二维分布：

| Backbone | 方法 | SWE-QA Score | SWE-QA Tokens | SWE-QA-Pro Score | SWE-QA-Pro Tokens | Oolong Acc | Oolong Tokens |
|---|---|---|---|---|---|---|---|
| Qwen3-Coder-Next | No Pruning | 7.71 | 590K | 7.60 | 607K | 81.7 | 3.6K |
| Qwen3-Coder-Next | LLMLingua2 | 7.06 ↓0.65 | 363K ↓38.5% | 7.02 ↓0.58 | 381K ↓37.3% | 74.6 ↓7.1 | 9.5K ↑163.9% |
| Qwen3-Coder-Next | SWE-Pruner | 7.33 ↓0.38 | 397K ↓32.7% | 7.36 ↓0.24 | 433K ↓28.7% | 79.7 ↓2.0 | 3.6K |
| **Qwen3-Coder-Next** | **SWE-Pruner Pro** | **7.73 ↑0.02** | **385K ↓34.7%** | **7.84 ↑0.24** | **368K ↓39.4%** | **80.3 ↓1.4** | **3.1K ↓13.9%** |
| MiMo-V2-Flash | No Pruning | 8.02 | 321K | 7.97 | 438K | 92.4 | 58.9K |
| MiMo-V2-Flash | LLMLingua2 | 7.73 ↓0.29 | 422K ↑31.5% | 7.47 ↓0.50 | 423K ↓3.4% | 87.1 ↓5.3 | 170.7K ↑189.8% |
| MiMo-V2-Flash | SWE-Pruner | 8.20 ↑0.18 | 303K ↓5.6% | 7.94 ↓0.03 | 417K ↓4.8% | 92.1 ↓0.3 | 52.5K ↓10.9% |
| **MiMo-V2-Flash** | **SWE-Pruner Pro** | **7.98 ↓0.04** | **299K ↓6.9%** | **7.86 ↓0.11** | **339K ↓22.6%** | **94.6 ↑2.2** | **41.2K ↓30.1%** |

几个直接读表就能抓出来的点：

1. **SWE-Pruner Pro 是唯一一个在所有 cell 上都省 token 的方法。** 看 MiMo-V2-Flash 的 Oolong 列——LLMLingua2 把 token 从 58.9K 拉到了 170.7K（+189.8%！），Selective Context 涨了 233.3%，LongCodeZip 也有 RAG 这种"打分代价+2%"的零和游戏。SWE-Pruner Pro 一刀砍掉 30.1% token 还**额外涨了 2.2 个点准确率**。

2. **质量和压缩同时保住了。** Qwen3-Coder-Next 上 6 个对照剪枝器里，RAG 是质量不掉的唯一方法，但它压不下去（最多 6.9%）；SWE-Pruner Pro 砍掉 34.7% / 39.4% / 13.9% 的同时，judge 分数 ±0.3 以内浮动。

3. **"读信号免费"的红利真不是吹的。** LLMLingua2 这种通用 PPL 压缩器在 Oolong 长上下文任务上把 token 干到 9.5K（Qwen 上）、170.7K（MiMo 上），原因是它要给每个 token 算 PPL，外加它本身没有"agent 关心什么"的信号——结果就是既没压住，又把分数砸了 7 个点。SWE-Pruner Pro 在同样的 Oolong cell 上是 41.2K（-30.1%），分数还涨 2.2 点。

### SWE-Bench Verified：真刀真枪的 patch 生成

Table 2 跳到了重头戏——SWE-Bench Verified 上打补丁，500 个真实 issue：

| Backbone | 方法 | Resolved | Input Tokens | API Calls |
|---|---|---|---|---|
| MiMo-V2-Flash | No Pruning | 326/500 | 2,971K | 94.8 |
| MiMo-V2-Flash | LongCodeZip | 344/500 ↑3.6% | 3,166K ↑6.6% | 99.8 |
| MiMo-V2-Flash | RAG | 338/500 ↑2.4% | 3,391K ↑14.1% | 102.4 |
| MiMo-V2-Flash | SWE-Pruner | 347/500 ↑4.2% | 3,414K ↑14.9% | 103.8 |
| **MiMo-V2-Flash** | **SWE-Pruner Pro** | **345/500 ↑3.8%** | **3,190K ↑7.4%** | **111.8** |
| Qwen3-Coder-Next | No Pruning | 341/500 | 5,307K | 131.9 |
| Qwen3-Coder-Next | LongCodeZip | 288/500 ↓10.6% | 4,718K ↓11.1% | 124.7 |
| Qwen3-Coder-Next | RAG | 330/500 ↓2.2% | 4,805K ↓9.5% | 126.0 |
| Qwen3-Coder-Next | SWE-Pruner | 320/500 ↓4.2% | 4,881K ↓8.0% | 127.1 |
| **Qwen3-Coder-Next** | **SWE-Pruner Pro** | **335/500 ↓1.2%** | **4,590K ↓13.5%** | **139.8** |

**在 MiMo-V2-Flash 上，SWE-Pruner Pro 是唯一一个同时实现"省 token"和"涨 resolve"的方法**。SWE-Pruner 多涨 0.4 个点（347 vs 345），但代价是 14.9% 的 token 增长；SWE-Pruner Pro 只涨 7.4% token，resolve +3.8 个点——**单 resolve 性价比差不多是 SWE-Pruner 的 1.6 倍**。

**在 Qwen3-Coder-Next 上，所有剪枝器都掉 resolve**（这其实反过来印证了一个事实：剪枝本身有信息损失风险，Qwen3-Coder-Next 的 80B MoE 对 30% 的 token 减少**确实**更敏感）。但 SWE-Pruner Pro 只掉 1.2 个点，token 减 13.5%——**仍然是降损失最少的**。

API calls 那一列挺有意思——MiMo 上所有剪枝器都增加了调用次数（最长从 94.8 到 111.8），Qwen 上则普遍减少。**这说明剪枝不是"省 token 就省一切"**——压缩后的 context 让 agent 偶尔更迷茫，会多探几次。

### 消融：focal 和 length embedding 哪个更重要

Table 3 是教科书式的消融——固定 backbone、训练数据、head 架构，对比两个 design axis：

| Design axis | Variant | F₁ | Judge |
|---|---|---|---|
| Loss function | BCE | 0.475 | 5.95 |
| Loss function | Focal | 0.593 | 6.37 |
| Loss function | Dice | 0.591 | 5.30 |
| Loss function | Tversky | 0.591 | 3.03 |
| **Loss function** | **Per-sample balanced focal** ★ | **0.635** | **7.08** |
| Length embedding | Without length embedding | 0.636 | 6.86 |
| **Length embedding** | **With length embedding** ★ | 0.635 | 7.08 |

Loss 那块，**per-sample balanced focal 把 judge 从 5.95 拉到 7.08**（+1.13），比 BCE 强 16%——这说明"per-sample 重平衡"比单纯的 focal / Dice / Tversky 都有效。Dice 和 Tversky 都能拿到 0.591 的 F₁，但 judge 跌到 5.30 和 3.03，作者在 Appendix G 里给出了一个让人印象深刻的解释——F₁ 奖励"挑对了一小撮保留行"，但 judge 关心"剩下的骨架能不能用"。

**Length embedding** 那个维度 F₁ 几乎不动（0.636 vs 0.635），但 judge 从 6.86 涨到 7.08。意思是——**这个 trick 不改变"行决策对不对"，但改变"错的时候往哪里错"**：加了 length embedding，head 会把"误剪"的成本转移到长响应上，短响应里几乎不动。这个 ablation 设计挺巧妙的。

### 延迟：15% 是个什么水平

![图 5：每轮 prune 开销 vs 紧邻的 generation 步骤](https://arxiv.org/html/2607.18213v1/x6.png)

*图 5：横轴是按轮数排好序的 16 条 MiMo-V2-Flash 轨迹，绿色是 generation wall time，红色是额外的 prune 时间。prune 总量是 generation 的 ~15%（中位 14.7%，p95 34.8%）。可以看到 prune 几乎是一条水平线（约 20-25s）——不随响应长度爆炸，因为 head 共用 prefill 输出*

关键设计是**把 head 塞进 SGLang 调度器**（in-engine colocation）——head 在 scheduler 的 GPU 上直接对 hidden states 算 logits，跨进程传递的 hidden states 整个消失。结果：同样的 16 条轨迹，off-engine 路径要 19.3%，in-engine 路径只要 15.0%。

但 15% 不算"忽略不计"——作者也承认了。但要注意这个数字是"per-call overhead / per-call generation"的比例，而 prune 每个 assistant turn 发生一次，token 减少却是**后续每一轮**的 generation 都受益。所以净 wall-clock 是负的（论文里没明说，但 argument 是这个）。

---

## 我的判断

### 亮点

1. **范式切得很准。** "信号在 backbone 里"这件事不是这篇首次提，但把它系统化、给工程可落地方案、并且把 linear probe 当"敲门砖"先证存在性——这套路在 narrative 上很顺。论文的阅读体验是"探针实验看完我就信了，后面就是按图施工"。

2. **Engine colocation 的工程深度。** Appendix E 占了 ~6 页，专门讲怎么在 SGLang 里把 hidden state path 修对——batch alignment、chunked-prefill accumulation、prefix-cache exemption、payload 大小优化（fp16 + base64 替代 orjson nested list，**端到端省 20× 字节**）。这部分内容很多团队复现 SGLang 自定义 op 时都会踩，写出来是真正的工程贡献。

3. **per-sample balanced focal。** 这个 loss 设计不复杂，但 idea 漂亮——你 0/0 的样本就该被记 0 分，3/3 的样本就该被记 3 分。F₁ 和 judge 的 dissociation 暴露了传统 metric 的盲区，这个 case study 写得很到位（Appendix G）。

4. **效果数字真。** MiMo-V2-Flash 在 SWE-Bench Verified 上 73.4% 是开源 SOTA（[HF model card](https://huggingface.co/XiaomiMiMo/MiMo-V2-Flash)），在这个 baseline 上**还能再涨 3.8 个点**，不是小打小闹——大多数"调优型"工作在这种基线上拿 1 个点就要谢天谢地了。

### 没那么漂亮的地方

1. **Qwen3-Coder-Next 上掉 resolve。** SWE-Bench Verified 减 1.2 个点，6 个 patch 没修对。论文的辩护是"compression-reduction 性价比最高"，但这是实打实的精度损失。我的猜测是：Qwen3-Coder-Next 是 80B MoE 激活 3B，本身就读得"精"——head 的决策粒度和 backbone 的内在粒度不一定对齐，导致 head 看不到的信号 backbone 是看得到的。如果作者能跑一个"head 关掉一半 hidden dimension"或者"head 用最后 2 层 hidden state 而非最后一层"的消融，会更说明问题。

2. **API calls 涨了。** 尤其在 MiMo 上 94.8 → 111.8（+18%），意思是你虽然单轮 wall time 快了 7.4%，但 agent 跑了**更多**轮次才能修好——最终的总 wall time 谁赢还不好说。论文把 token 和 API calls 分开报是对的，但工业部署更关心的是**总 wall time**。

3. **Judge 评分体系的主观性。** 整个 Table 1 用的都是 GPT-5.4-mini 在 1-10 评分上打的 judge，训练数据用 Claude Sonnet 4.6 标的 label。**两个闭源模型在论文的 metric 里承担了双重角色**——这是不是有点循环依赖？Table 3 的 ablation 里 BCE vs per-sample balanced focal，judge 6.37 → 7.08，**这个 0.7 分的差距**到底反映"agent 真的能修好更多 issue"还是"judge 模型恰好偏好某种文本结构"？Appendix G 给的 qualitative case 偏向后者——一个"只保留参数列表"的高 F₁ head，judge 给 2 分（agent 啥也干不了），一个"保留函数体"的高 judge head，F₁ 还不如前者。**这说明 judge 才是 ground truth，F₁ 是 noisy proxy**——但论文里 judge 自身的可信度没有充分验证。

4. **跨语言泛化靠"推断"。** Appendix A 里说训练数据是 83% Python + CLI，剩下 17% 是 Multi-SWE-bench 6 个非 Python 语言。Oolong 是自然语言聚合任务，但跟"代码"还隔了一层。如果作者真要 claim "language-agnostic"，应该至少跑一个非 Python 的代码任务（如 Java/Go/Rust 的 SWE-Bench 衍生），目前没有。论文自己也说了"left to future work"。

5. **SWE-Bench Verified 上 SWE-Pruner Pro 比 SWE-Pruner resolve 少 2 个**（345 vs 347），但 token 少一半的代价。这笔账见仁见智——把 0.4 个 resolve 差换成 7.5% token 节省，部署到生产环境大概率是值的，但论文没把这条 trade-off 量化（比如"每 1% token 节省换多少 resolve"）。

### 给做 coding agent 工程的读者

1. **如果你现在用的是 SWE-Pruner 或者自己训了个外挂剪枝器，可以认真考虑换成 SWE-Pruner Pro**——head 只有 18M 参数，训练 15 分钟，代码开源，配合 SGLang 直接有 in-engine 实现。token 节省能力相当甚至更好，质量还更稳。

2. **hidden state + light head 这个范式**比 SWE-Pruner 的"外挂剪枝器"更具迁移性——任何能暴露 hidden state 的 backbone 都能用（论文 limitation 写明当前只 eval 了 open-weight 模型，但**结构上**对 Claude / Gemini 这种闭源模型也行，只要服务商愿意开 hidden state 接口）。

3. **Length embedding 这个 trick 通用**。如果你的剪枝 head 是 per-sample 训的，遇到"长响应 / 短响应"分布不齐的场景，**加一个 length-aware embedding 几乎没成本**——消融里 0.22 个 judge 点的提升是免费的。

4. **judge 评分 vs 真实任务 metric 的 dissociation 值得警惕**。你们公司如果有"用一个 LLM 当 judge 来评估另一个 LLM"的工作流，请严肃测试 judge 的校准——F1 不一定对，judge 也不一定对，**最终答案要落到具体业务 metric**（resolve rate、user satisfaction、rework rate）上。

---

## 收尾

回到开头那个问题——agent 在 prefill 工具响应的时候，主模型 hidden state 里**已经**知道哪些行重要了。SWE-Pruner Pro 给出的答案是：**那就别让 agent 再起一台"剪刀"了，直接从隐藏状态里读**。

技术上它没有发明新东西——一个两层 FFN，加个 length embedding，加个 per-sample balanced focal loss。**但它做对了一件关键的事：把已有范式推到极简**。不外挂模型、不让 agent 写 hint、不让 backbone 重复 forward——只让 head 复用 prefill 已经在算的 hidden state。

这种"用最少的算力把信号榨干"的思路，在 2026 年这个 coding agent 越来越卷成本的时点，是一条值得每个做 agent infra 的人记住的路径。

下次你写自己的 agent 中间件时，问一句：**我的主模型在那一刻已经在算的东西，有哪些是我可以白嫖的？**

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
