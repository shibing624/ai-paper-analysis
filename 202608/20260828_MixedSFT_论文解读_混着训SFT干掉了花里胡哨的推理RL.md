# Mixed SFT：把 no-CoT 和长 CoT 混着训一遍，就把花里胡哨的"推理 RL"干掉了

> 论文解读 | arXiv: 2608.23256 | 2026-08-24

你有没有遇到过这种纠结：手头有一大堆教辅书、题解、教科书推导——内容里全是推理干货，但就是没有那种长长的 `<think>` 思维链标注。想拿来训推理模型，怎么用最划算？

过去一年的潮流答案是：上 RL。RPT、RLP、RMT、RLPT、PretrainZero，一波接一波的工作都在说同一件事——让模型先自己生成一段"隐式推理"，再按"这段推理能不能帮助预测下一个 token / 下一句话"给奖励，用强化学习把 no-CoT 数据榨干。听起来很优雅，论文里的涨点也确实好看。

但中国科学技术大学和上海人工智能实验室的这篇论文泼了一盆冷水：**这些 next-chunk reasoning RL 赢的可能根本不是"RL 本身"，而只是"让模型多接触了 no-CoT 数据"这件事**。作者们找出了一个所有人都漏掉的 baseline——Mixed SFT，就是把 no-CoT 和长 CoT 数据混在一个 SFT 阶段里一起训，就这一步。结果呢？post-RLVR 性能天花板比 next-chunk reasoning RL 高一截，训练算力还省了 60 倍以上。

说实话，看到这个结论我的第一反应是"这也行？"，但看完实验设计和机制分析，我被打动了。这篇文章最值钱的不是提出了什么新方法，而是把一个被集体忽略的、简单到有点丢人的 baseline 摆上台面，然后用扎实的对照实验证明：潮水退去，RL 那套花哨机制并没有在裸泳的 SFT 面前占到便宜。

---

## 📖 论文信息

- **标题**：Is Next-Chunk Reasoning RL Really Better than SFT? Revisiting Training Strategies under no-CoT Data
- **作者**：Yinhao Tang, Youqing Fang, Yanan Sun, Jiangning Liu, Ziyi Wang, Xun Zhao, Weiming Zhang, Bin Liu, Kuikun Liu, Wenwei Zhang, Kai Chen
- **机构**：University of Science and Technology of China（中国科学技术大学）、Shanghai AI Laboratory（上海人工智能实验室）
- **发表**：2026 年 8 月 24 日，arXiv:2608.23256v1 [cs.AI]
- **链接**：https://arxiv.org/abs/2608.23256

---

## 🎯 背景：no-CoT 数据这块肥肉，到底该怎么吃

先交代一下战场。现在推理模型的后训练基本靠长 CoT 数据——从强教师模型 rejection sampling 出正确轨迹，贵得要死。但世界上还有多得多的文本是"有推理营养、没推理痕迹"的：标准题解、教科书推导、论文正文。这类数据作者叫 no-CoT 数据。

no-CoT 数据直接做 SFT 有个老问题：它没有 `<think>...</think>` 这种结构化推理格式，朴素地训上去会**破坏模型已有的长 CoT 输出格式**，训完的 checkpoint 到 RLVR 阶段根本没法用。于是一批工作转向 RL，统称 next-chunk reasoning（NCR），按预测粒度分两派：

| 派别 | 代表工作 | 监督粒度 | 奖励方式 |
|------|---------|---------|---------|
| Next-Token Reasoning（NTR） | RPT、RLP、RMT | 预测下一个 token | token 级前缀匹配 |
| Next-Sentence Reasoning（NSR） | RLPT、PretrainZero | 预测下一句话/文本段 | 生成式奖励模型判语义一致 |

这些工作报告的涨点都是相对"SFT 只训 no-CoT 数据"这个 baseline 来的。问题来了——这个 baseline 合理吗？作者指出，**no-CoT-only SFT 是个不合格的参照物**：它破坏了长 CoT 格式，训出来的 checkpoint 连结构化推理都产不出来，RLVR 无从谈起。跟它比，赢了也没什么说服力。

真正该比的 baseline 是 Mixed SFT：no-CoT 和长 CoT 混在一个 SFT 阶段联合训练。这样既保住了长 CoT 格式，又把 no-CoT 知识灌了进去。这个想法朴素到什么程度呢？朴素到 RPT、RLP、RMT、RLPT、PretrainZero 这五个工作没有一个拿它当过 baseline——作者专门做了张表（论文 Table 1）把这件事钉死了。

![图1：四种 no-CoT 数据利用策略的对比与 post-RLVR 性能](https://arxiv.org/html/2608.23256v1/teaser2_3.png)

*图1：上半部分是四种策略的流水线对比——NCR 系（A：token 级 NTR，B：句子级 NSR）从 Reasoning SFT 出发做推理重构 RL；C 是 Sequential SFT（先 no-CoT 再长 CoT 两段）；D 是 Mixed SFT（单阶段混合）。所有策略最后接同一个 RLVR。下半部分是四个代表 benchmark 的 post-RLVR 成绩，红色条（Mixed SFT）在 AIME25、HMMT26、IMO-Answer、GPQA-Diamond 上全面领先。*

---

## 🏗️ 实验设置：统一底座，只改"no-CoT 怎么用"

实验设计做得很干净，值得专门说一下。

基座统一用 **Qwen3-30B-A3B-Base**——注意是 base 模型，不是 instruction-tuned 版本，为的是排除先前后训练的干扰。数据全部来自 AoPS（一个数学竞赛社区）：长 CoT 数据是 152K 条轨迹（约 1.95B token），用 DeepSeek-V3.2 生成、只留答案正确的；no-CoT 数据是 421K 条原始题解（约 0.53B token），有推导但没思维链。后面的 RLVR 阶段用 DAPO-Math-17K，GRPO 算法配规则化精确匹配奖励。

五种策略同台竞技：Reasoning SFT（纯长 CoT 基线）、Sequential SFT、Mixed SFT、NTR（用 RPT 实例化）、NSR（用 RLPT 实例化）。NTR 和 NSR 都从 Reasoning SFT 初始化，这也符合它们原始论文的设定。RL 全程在 64 张 H200 上跑。

评测分 pre-RLVR 和 post-RLVR 两个时点，ID 用六个竞赛数学集（AIME 2024/2025/2026、HMMT 2025/2026、IMO-Answer），OOD 用 HLE、GPQA-Diamond、MMLU-Pro。主指标是 avg@32（AIME/HMMT/IMO）或 avg@4（GPQA-D），HLE 和 MMLU-Pro 报 pass@1。

---

## 📊 主结果：Mixed SFT 三线全胜

先看主表（论文 Table 2），我把关键数字整理一下，post-RLVR 是重点：

| 策略 | AIME25 | AIME26 | HMMT26 | IMO-Ans. | GPQA-Dia. | MMLU-Pro |
|------|--------|--------|--------|----------|-----------|----------|
| Reasoning SFT + RLVR | 76.67 | 65.10 | 39.39 | 47.25 | 56.94 | 73.95 |
| Sequential SFT + RLVR | 69.27 | 56.35 | 35.23 | 41.25 | 54.55 | 71.10 |
| **Mixed SFT + RLVR** | **85.73** | **70.42** | **51.52** | **54.00** | **60.98** | **75.84** |
| Reasoning SFT + NTR + RLVR | 84.38 | 69.38 | 47.16 | 46.50 | 57.70 | 74.03 |
| Reasoning SFT + NSR + RLVR | 80.92 | 72.19 | 46.02 | 48.25 | 56.31 | 74.89 |

三个结论，一个比一个扎心。

**结论一：Mixed SFT 在 ID 和 OOD 上都赢。** ID 六个 benchmark 平均 67.4，比次优的 NTR（64.2）高 3.1 个点，比 NSR（63.6）高 3.7 个点。OOD 上同样领先——GPQA-Diamond 60.98 vs NTR 的 57.70，HLE 9.24 vs 7.68，MMLU-Pro 75.84 vs 74.03。OOD 赢这一点挺关键的，说明 Mixed SFT 的优势不是数学域过拟合，泛化能力实打实更强。

**结论二：Mixed SFT 大幅赢 Sequential SFT。** 两者看到的数据完全一样（都是 no-CoT + 长 CoT），差别只在组织方式。AIME25 上 85.73 vs 69.27，差了 16 个点多。这直接说明：**no-CoT 数据怎么进模型，跟进不进一样重要**。

**结论三：算力差距夸张。** NTR 和 NSR 的 GPU 小时数是 SFT 的 60 倍以上——RL 要在线 rollout、算奖励、做策略优化，NSR 还得额外养一个生成式奖励模型（论文里用的是 gpt-oss-120b，reasoning_effort=high，这开销想想就肉疼）。性能落后、成本还贵两个数量级，这个买卖怎么算都不划算。

还有一个更反直觉的发现。看 pre-RLVR 分数：Mixed SFT 平均只有 27.5，比所有其他方法低约 20 个点——它是 pre-RLVR 阶段最差的那一个。但 post-RLVR 它平均 61.1，是最高的，从 pre 到 post 涨了 33.7 个点，是其他方法涨幅（3.3 到 10.3 个点）的三倍以上。

**pre-RLVR 分数根本预测不了 post-RLVR 天花板。**

这个发现的方法论意义可能超过主结果本身：评估 no-CoT 训练策略，不能看中间 checkpoint 的即时分数，必须跑完整个 RLVR 流水线再下结论。之前那些 NCR 论文拿 pre-RLVR 或者浅层对比说事，评估姿势本身就偏了。

---

## 🔬 机制分析：为什么 RL 那套没赢，为什么 Mixed 能赢

这部分是我个人觉得全文最扎实的。作者做了六个 Observation，环环相扣地把"为什么"讲清楚了。

**Observation 1：NTR 的熵过滤器选的根本不是"推理难"的 token。**

NTR 的核心假设是：高熵 token 是需要推理的硬目标，所以只监督 no-CoT 语料里熵最高的 top-20% token。作者直接抽查了 2048 个高熵 token，跟踪 NTR 训练过程中三个量的变化：token 熵、带显式推理的预测准确率、不推理直接预测的准确率。

结果是：推理准确率从 0.29 一路爬到 0.55，但同一批 token 的熵几乎纹丝不动。更直接的证据是 no-reasoning accuracy 全程稳在 0.48 左右——也就是说，**将近一半的"高熵难 token"，不推理也能直接猜对**。高熵反映的是原始语料的局部不确定性，不是推理难度。这个前提假设塌了。

**Observation 2：NTR 生成的推理轨迹退化成了局部补全。**

既然很多目标 token 局部就能预测，模型很快就找到了作弊路径：不用真推理，套个模板糊弄一下就能拿到重构奖励。训练过程中生成熵和回复长度持续下降，生成的轨迹收敛到几个重复模板——对不同前缀输出的都是类似的套路化短补全。NSR 在句子级别有一样的毛病。下游 RLVR 拿到的是一个已经被模板化策略锁死的初始化，没什么可放大的推理结构。

**Observation 3：把熵救回来也救不了天花板。**

这是最狠的一个对照。你自然会说：那是不是熵坍缩导致的？把探索保住，NTR 是不是就能学到真推理？作者做了两个干预——随机丢弃组内成功率高的 rollout 组，下调正优势的权重——成功把生成熵稳在 0.52 附近、回复长度稳在 640 token 左右。坍缩确实被压住了。

但 post-RLVR 天花板反而比原版 NTR 还低一点。每个 benchmark 都是。

这个实验的说服力在于它排除了一个最自然的"挽尊"解释：熵坍缩不是 NTR 平庸的原因，原版 NTR 恰恰是**靠收敛到那个模板化尖峰策略才到达它的天花板的**。阻止收敛只是拿走了一个能用的解，并没有提供更好的解。

**Observation 4：在 Mixed SFT 之上再叠 NTR/NSR，零增益。**

还有一个混淆变量要排除：RL 比 SFT 贵太多，覆盖的 no-CoT token 少得多，SFT 的优势会不会只是"看的 token 多"？作者从 Mixed SFT 出发接着训 NTR 或 NSR，再接同样的 RLVR。结果最终性能几乎不变，好几个任务还微降——AIME25 从 85.73 掉到 84.58（叠 NTR）和 84.67（叠 NSR）。模型一旦通过 SFT 充分吸收了 no-CoT 语料，next-chunk 目标就不再提供 RLVR 能放大的新东西了。

**Observation 5：Mixed SFT 的 pre-RLVR 低分是格式问题，不是能力问题。**

Mixed SFT pre-RLVR 为什么那么惨？两种数据的输出结构打架：长 CoT 有 `<think>` 标记和标准答案格式，no-CoT 就是普通题解写法。联合训练暂时搞乱了输出结构——有时直接跳过推理给答案，有时冒出多个 `<think>` 段。但知识没丢。RLVR 一开始，格式合规率快速回升，准确率跟着涨：可验证奖励重新立起输出格式，模型把 Mixed SFT 阶段吃进去的数学知识、推导模式、题型变体全部调动出来，冲到更高的天花板。

![图2：Mixed SFT 在 pre-RLVR 阶段的两种格式不稳定失败模式](https://arxiv.org/html/2608.23256v1/case_mixed_sft.png)

*图2：Mixed SFT pre-RLVR 的典型翻车现场——上面是多个 `<think>` 段挤在一起的混乱输出，下面是干脆跳过推理直接给答案。格式乱了，但内容是对的。RLVR 阶段格式合规率回升后，这些被压住的能力就释放出来了。*

**Observation 6：Sequential SFT 的第二阶段把第一阶段学的东西冲掉了。**

那 Sequential SFT 为什么差这么多？作者从 no-CoT 训练语料里随机抽了 100 道题做"记忆保持探针"，RLVR 之后测 avg@8：Mixed SFT 68.63，Sequential SFT 只有 59.19。Sequential SFT 的第二个长 CoT 阶段部分覆盖了第一阶段吸收的 no-CoT 信号，而且**RLVR 救不回开始前就被擦掉的知识**。跨阶段遗忘是真金白银的损失。

把六个 Observation 串起来，整个故事就通了：NCR 的收益不是来自它的推理重构机制（监督目标一半不需要推理、轨迹模板化、压住坍缩也没用、叠在 SFT 后零增益），而 Mixed SFT 避免了跨阶段遗忘，把 no-CoT 信号完整留给了 RLVR 去放大。

---

## 🤔 我的判断

这篇论文属于我最喜欢的类型：不发明新锤子，而是指出大家抡锤子的姿势不对。

**亮点很硬。** 对照实验设计干净（统一 base 模型、统一 RLVR 预算、统一数据），六个 Observation 层层递进，尤其 Observation 3 那个"熵救回来天花板反而更低"的实验，直接把最可能的反驳路径堵死了。Table 1 把五个前作"都没比 Mixed SFT"这件事列出来，也是很有勇气的做法。

**但也要泼点冷水。** 第一，所有实验都是数学单域——no-CoT 语料来自 AoPS，评测也全是数学加少量科学推理。代码、多语言、其他结构不同的 no-CoT 数据上结论是否成立，作者自己在 Limitations 里也承认没做。第二，Mixed SFT 只用了单一固定混合比例，比例扫描没做，而"no-CoT 占多少"很可能是实际落地时最敏感的旋钮。第三，基座只有 Qwen3-30B-A3B-Base 一个，跨模型规模的稳健性未知。

还有一个我想多嘴一句的点：这篇论文打的靶子主要是"在 no-CoT 上做 NCR 作为 RLVR 前置初始化"这个场景。像 RLP 那种把 NCR 往预训练阶段搬的工作，回答的是另一个问题（预训练算力怎么花），严格来说不完全被这篇的结论覆盖。不过 RLP 们确实也该补一个 Mixed SFT 式的 baseline 再说话。

**对工程的启发很直接**：如果你手头有推理味浓但没思维链的语料，别急着上 RL 基础设施。先试试把它跟长 CoT 数据混在一个 SFT 阶段里训，然后接 RLVR。60 倍算力差距，效果还更好。省下来的 GPU 小时拿去多刷几轮 RLVR，大概率收益更高。

另外那个方法论提醒值得所有做后训练的人贴在墙上：**中间 checkpoint 的分数会骗人**。Mixed SFT pre-RLVR 全场垫底、post-RLVR 全场第一——你要是中途看分就把这个方案砍了，就永远不知道自己错过了什么。

---

## 📝 总结

这篇论文回答了一个被整个行业带偏的问题。next-chunk reasoning RL 看起来精致的推理重构机制，在控制变量之后并没有提供超出"让模型接触 no-CoT 数据"本身的收益；而被所有人忽略的 Mixed SFT，用 1/60 的算力拿到了更高的 RLVR 天花板。机制上，NCR 的监督目标大半局部可预测、生成轨迹坍缩成模板；Mixed SFT 则靠单阶段联合训练避免跨阶段遗忘，把完整的 no-CoT 信号留给 RLVR 放大。

简单方案赢了，不是因为 RL 不行，而是因为对比实验少做了最关键的一组。做后训练数据配方的人，这篇值得细读。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
