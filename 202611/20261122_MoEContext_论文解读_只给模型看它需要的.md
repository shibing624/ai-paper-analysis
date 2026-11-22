# 只给模型看它需要的那一块：93/120 到 119/120，以及一个被作者自己拆穿的判断器

🎯 **核心摘要**

跑长程 Agent 的人都撞过这堵墙：每一步都把全部历史塞回上下文，聊到二三十轮，模型开始复读早就被证伪的旧值，眼神越来越散。这篇 MoE-Context 换了个思路——把上下文切成块，每步先跑一次极快的是/否判断，只有需要的块完整可见，其余缩成一行笔记随时能召回；在单一 KV 缓存里用 attention mask 把块**藏起来而不是删掉**。Qwen3-1.7B 上，只给需要的那一块把 **93/120** 抬到 **119/120**，token 从约 3450 降到约 650。听起来很爽对吧？但作者转头自己做了一组实验：给问题点名的文件做关键词匹配，**120/120 全中**，比那个精心设计的判断器还准。我的判断是，这不是底层突破，是一个干净的可逆上下文切换原型，加上一份我在近期论文里见过的最诚实的负面报告。值得花半小时读，别急着搬进生产。

---

📖 **论文信息**

| 项目 | 内容 |
|------|------|
| 标题 | MoE-Context: Show the Model Only What It Needs |
| 作者 | Vedant Navale（单作者） |
| 机构 | UpThink EduTech Services Pvt. Ltd. |
| 提交 | 2026 年 11 月 18 日（v1） |
| 篇幅 | 10 页，10 张图 |
| 分类 | cs.AI、cs.CL、cs.LG |
| 编号 | arXiv:2611.24379 |
| 链接 | https://arxiv.org/abs/2611.24379 |
| 许可 | CC BY 4.0 |

论文致谢里提到 Matthias Busch 的细致建议，直接催生了后面要聊的关键词基线、迁移测试和缓存顺序分析——这几块恰恰是全文最有意思的部分，等会儿细说。

---

🧠 **一、问题：Agent 把全部家当都背在身上**

上周调一个读代码仓库的 Agent，跑到第十几步它突然开始复读一个早就被推翻的配置值。我把上下文 dump 出来一看，三千多个 token 里堆着七八个文件的全量内容、每一轮的回复、几段测试日志，而当前这步真正要用的，只是其中一个文件里的一行。

论文的开头比我会写：你泡茶之前，不会把整个人生都装进脑子里。你手里只握着当下要用的东西，更早的事留成一段模糊的记忆，稍微使劲就能把细节调回来。今天的 Agent 恰好反过来——每条消息、每次回复、读过的每个文件，都在每一步里原封不动地杵着。

这就是"上下文腐烂"的一种具体形态。做过工程的人都知道，堆在窗口里的东西越多，模型越容易被无关内容分走注意力，越容易从一段旧的错误输出里抄答案。论文里那个撤销实验把这件事量化得很扎心：一次失败尝试还留在视野里时，模型 **26/30** 次照抄了它的旧错误值。

![图1：现状与 MoE-Context 的对比](https://www.mulanai.com/fs/files/1122_2e8a6a7e_fig1_ide.png)

*图 1：左边是现状，每步读全部（user msg、reply、read login.py、run tests、reply、read auth.ini、user msg 全在）；右边 MoE-Context 只把 read login.py 和 run tests 完整点亮（蓝色），其余缩成灰色的一行式笔记，随时能回来。*

顺着这张图往下，作者给这套机制起了个名字，来自 Mixture of Experts。说到这个，MoE 的老祖宗是 Jacobs、Jordan、Hinton 1991 年那篇 Adaptive Mixtures of Local Experts，现代形态则是 Shazeer 等人 2017 年的 sparsely-gated MoE 层：一个 gate 给所有 expert 打分，只留 top-k 激活，其余参数留在显存里但不参与计算。核心承诺一直没变——**容量做大到吓人，单次计算却几乎不变**。

MoE-Context 把这个"gate 选 expert"平移成了"decide pass 选 block"。

![图2：MoE 与 MoE-Context 的类比](https://www.mulanai.com/fs/files/1122_28f5e825_fig2_moe.png)

*图 2：上排 input → gate → expert 1~5（点亮 2、4）；下排 next step → decide pass → block 1~5（点亮 1、4）。对应关系一目了然，橙色那个 gate 换了个名字叫 decide pass。*

作者还顺手借了 Kahneman 的框架：decide pass 是快的 **System 1**，负责推理和写答案的模型是慢的 **System 2**。这个比喻我觉得贴，但也就只是个比喻，别顺着它去脑补什么认知科学机制。

现有方案怎么解决的？论文 Section 2 把相关工作按四个维度摊开了，我按原文 Table 1 复刻一版：

| 方法 | 保留原文 | 能召回 | 每步逐块判断 | 单模型既选又答 |
|------|:---:|:---:|:---:|:---:|
| 裁剪 / 摘要 |  |  |  |  |
| 模型自己管理上下文 | ✓ | ✓ |  |  |
| 固定规则 | ✓ | ✓ | 部分 |  |
| 学习型控制器（未实测） | ✓ | ✓ | ✓ |  |
| 决策模型，只判一次 | ✓ | ✓ |  |  |
| 读自身 hidden states，只判一次 | 部分 |  |  |  |
| 丢弃 KV 条目 | 部分 |  |  |  |
| **MoE-Context** | ✓ | ✓ | ✓ | ✓ |

几个挨得最近的工作值得点名：winnow 用一个决策模型给每段工具输出的 chunk 打一次分；SWE-Pruner Pro 从 coding agent 自己的 hidden states 里通过一个小训练头读 keep/prune 决策，然后**把行删掉**；Neural Paging 提了个学习型控制器，但没在真实模型上测；Cache Engineering 允许把 KV 条目从后续 attention 里丢掉，引擎能重算重建，却给不出"把丢掉的条目放回 attention"的路径。

看出缝隙了吗？大多数方法要么把原文切没了，要么让模型自己停下来管理上下文，要么只在开头判一次。MoE-Context 想占的是那个空着的右下角：**原文不丢、可召回、每步对每个块重新判一次、且由同一个模型一气呵成**。

坦率的讲，这个定位我挺买账的。它不是喊了个新口号，而是先在表格里把别人的格子画满，再指着自己那一行说"这四个我全要"。

---

🏗️ **二、方法：切块、策展，然后用掩码把块藏起来**

一句话讲清核心 idea：别在写上下文的时候决定留什么，改成**读的时候**决定这一步亮哪几块。

具体拆成四件事。上下文被切成块——一条消息、一次回复、一次带结果的工具调用、一次文件读取，各算一块。每步开始，一次 decide pass 对每个块问一句"这一步需不需要你"，回答是/否。需要的块全文展示，不需要的块缩成一行笔记，比如 "read login.py, 300 lines"。如果模型发现少了什么，可以 recall 把块调回来，**什么都不删**。像指令这类块会被 pinned，永远可见。

![图3：每一步的循环](https://www.mulanai.com/fs/files/1122_94249b0d_fig3_loo.png)

*图 3：new block arrives → one-line note written → decide: needed? → model sees kept + notes → model acts, may recall，虚线绕回开头。下方那行小字是关键：repeat every step: a block hidden now can come back later。*

有个细节我读的时候停了一下：decide pass 必须读**完整块**，不能只读笔记。作者给的理由特别实在——如果 bug 藏在 300 行文件的第 140 行，"read login.py" 这行笔记根本看不出来。笔记的作用不是给判断器省算力，是让**回答模型**知道"还有哪些东西可以要"。这个职责划分我觉得很干净，很多做上下文压缩的工作恰恰是在这里糊掉了。

### 最值钱的一块：单缓存 + 掩码

两种建法摆在那儿。两个模型的版本，让一个小决策模型（比如 Jev）给大模型挑块，作者明确说**没测**。真正实现并跑实验的是单模型版本：同一个模型先给每个块打分，再回答。

每个块只编码一次，进 KV 缓存。隐藏一个块，就是把它在 attention mask 里对应的位置置零；位置保留，缓存条目依然管用，什么都不用重读。每一步只是在切换"attention 能看见哪些块"。

![图4：单模型单缓存，三步切换](https://www.mulanai.com/fs/files/1122_b89c248b_fig4_cac.png)

*图 4：顶部写着 Blocks are read once into the cache. Each step only switches what attention can see. step 1 亮 b1、b2；step 2 只亮 b4；step 3 亮 b2、b4、b6；右侧每一步都标着 nothing re-read。灰色的小方块是 "bX (note)"，即缩成笔记的块。*

这就是全文我最欣赏的设计。传统做法里，"不看某块"通常靠改 prompt、重新拼接、重新 prefill；这里改成改一个 mask。改 mask 是 O(1) 的位操作，撤销也只要把位翻回来。后面两个"便宜的用法"全是从这一行长出来的。

附录 A 把块的物理布局写得很细，做复现的人用得上：每个块长成 `[block i | note]`，隐藏时插入一个只在隐藏态可见的 `(hidden)` 标签，然后是正文和 `[/block i]`；被隐藏的正文配合 explicit position ids 从 attention 里 mask 掉。笔记的生成提示是 "Describe this context block in under 15 words, one line. Name the tool and any file name it mentions."

不过作者自己先埋了个雷，叫 **Cache order**。块是按顺序在全序列上一次性编码的，靠后的块在编码时**已经 attend 过前面的块**，包括那些后来才被隐藏的。所以一个被 mask 的块，它的痕迹可能还残留在后面块的 hidden state 里。这一点在 Section 4 里被专门测了，而且测出一个挺微妙的结果，等会儿讲隐藏性实验时一并说。

### decide pass 到底怎么判

没有生成文本，没有 chain-of-thought。对块 $i$ 和问题 $q$，模型只看块 $i$ 的正文，加一句提示：

> Question: {q}. Is the content of block i needed to answer this question? Answer yes or no.

然后读 last position 的 output scores，不生成任何字。附录给的定义是：对 "yes" 与 "no" 相关 token 的分数各取 log-sum-exp，两者之差过 sigmoid，得到这个块的相关性分数。伪代码长这样：

```python
# 每步、每个块跑一次（论文里约每题 1 秒，作者明确不声称速度优势）
def decide(block_i, question):
    prompt = f"Question: {question}. Is the content of block {i} " \
             f"needed to answer this question? Answer yes or no."
    logits = model(prompt + block_i.body).last_position_logits
    score = sigmoid(logsumexp(logits[YES_TOKENS])
                    - logsumexp(logits[NO_TOKENS]))
    return score            # 没有生成文本

visible = [b for b in blocks if b.pinned or decide(b, question) > threshold]
context = render_full(visible) + render_notes(blocks - visible)
answer  = model(context, forbid_tokens=[PROBE_ROWS])   # 回答步禁用探针行
```

### 那个写进权重里的线性探针

小模型的原始 yes/no 分数分离得很差，于是作者训了一个监督线性探针。说是"训练"，其实没有梯度更新，是 Fisher 线性判别的一个闭式解：

$$w=(\Sigma+\lambda I)^{-1}(\mu_{\text{needed}}-\mu_{\text{not needed}})$$

$\mu$ 是两类样本的隐状态均值，$\Sigma$ 是共享协方差，$\lambda$ 是 ridge 项。拿到 $w$ 之后，去掉它沿平均隐状态的分量，再在留出数据上拟合一个 scale 和 offset。

真正让我觉得精巧的是**往哪儿写**。作者把探针写进两行 tokenizer 永远用不到的 spare output rows（第 151869、151870 行，即词表大小 +200 和 +201）：decide pass 去读这两行的分数，answer step 直接把这两行禁掉，于是正常回答完全不受污染。附录里连拟合参数都给了：$\lambda$ 取 $\Sigma$ 对角线均值的 0.1 倍，$a=4.691$、$c=-9.030$ 用 L-BFGS 在二元交叉熵上拟合，以 $\pm\tfrac{a}{2}w$ 写入。还有个容易被忽略的工程细节：这个模型的输入/输出 embedding 在配置里是 tied 的，作者是先确认两者在内存里是独立 tensor 才动的手。

他们一开始试过把探针写进真实的 "yes"/"no" 行，结果把普通回答搞坏了。这个失败记录我挺喜欢的——它解释了为什么非得绕到 spare rows 上去，而不是一上来就摆个聪明设计。

这里补一句背景：把行为当成隐状态里的一个方向读出来，再加回 activation 或写进权重，这条路有前置工作撑着（Arditi 等人关于 refusal direction、Panickssery 等人的 contrastive activation addition）。所以探针本身不算新，新在把它塞进 spare rows 当成一个不干扰生成的内部通道。

---

🔧 **三、实验：数字很漂亮，判断器很诚实**

**实验设置**：附录 A 给全了，Qwen3-1.7B，bf16，关掉 thinking mode，greedy decoding，最多生成 24 个新 token，跑在一张 8GB 显存的笔记本 GPU 上。每个 layout 有 16 个块：8 个 settings 文件（每个 30 行 key/value）+ 8 段 test log，合计约 3450 个 token。3 个 layout、每个 40 道题，凑成 120 道题；每题只问某个文件里的一个 value。问题模板是 "What is the value of {key} in config/{name}.ini? Answer with the value only."，改写版是 "What is the {key} setting for the {name} service?"。探针的练习数据用的文件名和 key 从不在测试里出现。种子、样本量也都列了：拟合用 seeds 101~110、2400 个样本（150 正例），校准用 seeds 201~204、960 个样本（60 正例）。

### 主实验：少给，反而对得多

![图5：主实验结果](https://www.mulanai.com/fs/files/1122_2aba2832_fig5_mai.png)

*图 5：Correct answers out of 120。四根柱子分别是 show everything（约 3450 tokens）93/120、model picks no probe（约 650）80/120、model picks with probe（约 650）110/120、perfect pick（约 650）119/120。*

| 展示方式 | 回答步可见 token | 答对 |
|------|:---:|:---:|
| 全给 | 约 3450 | 93/120 |
| 模型自选，无探针 | 约 650 | 80/120 |
| 模型自选，有探针 | 约 650 | 110/120 |
| 理想选块（oracle） | 约 650 | **119/120** |

把视野从 3450 砍到 650，约五分之一的 token，答对从 93 涨到 119，**多了 26 道题**。看到这个数我愣了一下，不是因为涨幅，是因为它反过来坐实了开头那个痛点：那 26 道错题里，有相当一部分不是模型不会，是被无关内容搅糊涂了。小模型尤其吃这个亏。

但真正该盯的是第二行。让模型自己选块，反而掉到 80/120，比啥都不做还差。这张图把全文的张力一次摆齐了：**少给这件事本身是对的，谁来选这件事还没解决**。

### 探针确实把选块拉起来了

![图6：探针前后的选块质量](https://www.mulanai.com/fs/files/1122_67c2bd40_fig6_edi.png)

*图 6：左半 Right block ranked first，no probe 81/120 → with probe 111/120；右半 Needed block wrongly hidden，39/120 → 9/120。两边加起来正好 120，说明这是同一批题的两个互补视角。*

把需要的块排到第一，从 81 涨到 111；把需要的块误藏起来，从 39 降到 9。取 top-1 块去回答，也从 80/120 涨到 110/120。探针在它自己那套练习题上，是实打实管用的。

### 然后作者自己把桌子掀了

第一刀是 reviewer 建议的关键词基线。每道题都点名了文件名，那就按"与问题词重叠最多的块"来选——**120/120 全中**，回答 119/120，跟 oracle 一模一样。

![图7：reviewer 的两组检查](https://www.mulanai.com/fs/files/1122_9d2e77f1_fig10_ch.png)

*图 7：左半 Right block picked (service named)，keyword search 120/120、model after probe 103/120、model no probe 76/120；右半 New task: test-log questions，model no probe 61/120、model after probe 16/120（橙色）。*

怕你说"那是因为文件名直接写在题里"，作者又做了一版改写题：用文字点名 service，不点文件名。关键词搜索还是 120/120，decide pass 无探针 76/120、加探针 103/120。作者自己在文中下的结论很不留情面：*On this task, simple retrieval is enough; the decide pass is not yet justified by accuracy alone.*

第二刀更狠，叫**探针不迁移**。换一类问题——不问配置文件，改问某段 test log 里某个测试是 passed 还是 skipped。没探针时 decide pass 把需要的块排第一 61/120；套上那个只在 settings-file 问题上拟合的探针之后，**掉到 16/120**，而且通常错选成 settings 文件。作者的判断是：探针学到的是训练任务，不是"相关性"本身。

说实话，读到这儿我是有点肃然起敬的。一篇 10 页的论文，完全可以把 93→119 那张图放头版、把探针 80→110 写成"我们的方法有效"，收工。它偏要花两整节证明自己最核心的那个智能部件不如 `grep`，还证明它换个任务就帮倒忙。致谢里那句"建议催生了关键词基线、迁移测试和缓存顺序分析"，现在看分量很重。

### 掩码到底藏得干不干净

![图8：掩码与删除的隐藏程度](https://www.mulanai.com/fs/files/1122_36f4fd84_fig8_lea.png)

*图 8：How hidden the answer is (minus log-probability; higher = more hidden)。block visible 11、block hidden by mask 54、block deleted 51。*

用"模型吐出那个被藏起来的答案"的负对数概率衡量：块可见时约 11，掩码隐藏时约 54，把块从输入里彻底删掉、重新编码时约 51。掩码甚至比删除藏得还干净一点点（正文里写的是 log-probability 约 −11 / −54 / −51）。

但作者紧接着自己划了条边界：这个 layout 里目标块排在大多数其他块**前面**，所以没覆盖到前面说的 cache order 隐患——后面块里可能残留的痕迹，这个实验测不到，得看下面的撤销测试。这种"我知道我这个实验的适用范围在哪"的写法，比多刷三个 benchmark 更让人放心。

### 一个没做成的功能：recall prompting

给提示词加上召回说明、给隐藏块打上 `(hidden)` 标签之后，小模型开始**疯狂要块**，哪怕答案就在眼前也要，全量可见时准确率直接塌到 **44/120**。有意思的是另一半：当只把需要的那个块藏起来、而且模型确实开口要了，它 **72/72** 次都点对了块。

所以问题不在"找不准"，在"什么时候该找"。作者说这需要更强的模型，或者一个学出来的技能。这块我没完全想明白为什么掉得这么惨——提示里同时出现"hidden"标签和"可以 RECALL"的许可，对 1.7B 来说可能直接变成了一种行为先验。如果我的理解有偏差，欢迎指正。

### 我对这组实验的几点保留

不吹不黑，几个地方我得皱眉：

任务太干净了是最明显的一条。"从 config 文件里取一个 value"，相关性几乎全写在词面上，关键词满分是必然的；作者自己也承认，公平的测试得换成"相关性在字面上看不出来"的任务，比如找出是哪个文件导致测试挂了——可那恰恰是这类方法最该证明、却没证明的场景。

120 道题没有 error bars，作者把它列进 Limits 了，我照抄一遍以示不是我挑刺：三个 seed、40 题一组，波动区间完全没给，26 道题的差值有多稳，心里得打个问号。再说只有一个模型、一个合成任务族，Qwen3-1.7B 是被"上下文太长会分心"坑得最狠的那一档，换个 70B，93→119 的幅度还剩多少，完全未知。

速度上也没占到便宜：decide pass 对每个块各跑一次 forward，论文里约每题 1 秒，作者明确写了 make no speed claim，省下的 token 是真省了，省下的时间不是。更现实的是落地——公开 API 和主流 serving 栈目前不暴露 attention masking，单缓存那套得自己改推理引擎。

---

🧪 **四、掩码不删除，白捡的两个用法**

前面说改 mask 是 O(1) 的位操作，红利在 Section 5 兑现了。

**撤销一次失败的尝试。** 把失败尝试对应的块关掉，换个干净视野重来，那段尝试仍然可召回。

![图9：两个便宜的用法](https://www.mulanai.com/fs/files/1122_008f5f31_fig7_pat.png)

*图 9：左半 Undo: repeated the old wrong answer，old try visible 26/30 → old try hidden 5/30；右半 Subagents: two-part questions right，one call everything 4/30 → subagents one block each 12/30。各 30 道题。*

数字是这样：旧尝试可见时，模型 26/30 次照抄旧错误值，只答对 3 次；隐藏后，照抄降到 5/30，答对升到 16 次。

真正精彩的是作者对那 5 次漏网之鱼的追查。30 个案例里有 13 个，模型自己写的那行笔记里**含着旧的错误值**；5 次照抄全部来自这 13 个案例（5/13，其余 17 个案例是 0/17）。把笔记连正文一起隐藏，照抄变成 **0/30**，答对保住 15/30。

结论一句话：泄漏源是笔记，不是被 mask 的缓存。所以失败尝试的笔记别复述它的错误输出。这条工程建议我觉得是全文最可以立刻拿走的东西——你在做 Agent 记忆摘要的时候，如果摘要里把 wrong answer 原样抄进去了，后面怎么 mask 都白搭。它顺带也回答了前面 cache order 那个悬着的问题：至少在这个测试里，后面块没把隐藏内容带出来，真正漏的是人写的那行字。

**不开新上下文跑子 Agent。** 碰到支线任务，只把需要的块打开，留下一句短结果，再把其余切回来。在两问题目上，一次看全部的调用只答对 4/30；每个 lookup 只看一个文件的子 Agent 答对 12/30，作者顺带注明了天花板——单次 lookup 准确率 38/60，子 Agent 已经被这个上限卡住了。相关工作中 Sculptor 把"抑制错误步骤"列为 future work，Pichay 报告共享子 Agent 状态是个麻烦，这两条正好对上。

12/30 看着不高，但对照 4/30 是三倍。而且它省掉的是重新拼上下文、重新 prefill 的开销——这正是"藏而不删"最直接的兑现。

---

💰 **五、算一笔账：省 token 不等于省钱**

Section 6 有段很容易被跳过的成本分析，我觉得挺清醒。

各家 API 对重复 prompt 前缀都给缓存折扣，工程上的铁律是"稳定在前、易变在后"，前缀得逐字节一致才能命中缓存。你每步切换展示哪些块，等于不停改动 prompt 中间的内容，前缀缓存就废了。

作者给了个不等式。设 $N$ 是全部块的 token 数，$A$ 是完整展示的，$S$ 是笔记的，$p_{\text{in}}$、$p_{\text{cache}}$、$p_c$ 分别是输入价、缓存输入价和 decide pass 的价：

$$\frac{A+S}{N}\lt\frac{p_{\text{cache}}-p_c}{p_{\text{in}}}$$

若缓存输入按正常输入的一折计，展示出来的部分就得压到总量的约十分之一以下，这套方案才比"乖乖全给"便宜。作者自己的结论是：*so the main argument is quality, not price*——主要卖点是质量，不是价格。而单缓存版本因为压根不改 prompt、只改 mask，绕开了这个问题。

这段我给满分。太多上下文管理的论文只算 token 数，把 token 当成钱；这篇把 serving 侧的缓存折扣一起算进去了，还老实承认在按量付费的 API 上自己未必划算。

---

⚠️ **六、作者自己列的局限**

Section 6 的 Limits 一小段，几乎不用我转述：一个小模型、一个合成任务族；120 道题没有误差棒；decide pass 每块一次 forward（这里约每题 1 秒），不作速度承诺；块是逐个判的，没有联合判断；笔记写得太差会让块找不到；serving 栈和公开 API 还没暴露 attention masking。

Discussion 里那句"什么还没成立"更值得抄下来：decide pass 在"问题点名目标"的任务上不比关键词搜索强，一个任务上拟合的探针在另一个任务上帮倒忙；公平的测试需要相关性不写在词面上的任务，以及一个在多种任务类型上训过的 decide pass。

我再加一条自己的：块与块被当成一袋互相独立的样本，逐个打 yes/no。可真实 Agent 历史明明是有结构的——一段 test log 属于它测的那个文件，一次 reply 属于它回答的那条消息。把结构扔掉只做逐块打分，会不会正是探针学不到"通用相关性"的原因之一？论文没验证，纯属我的猜测。

---

💡 **七、我的判断，以及那张指向未来的图**

坦率的讲，这篇论文我不会给它贴"突破"的标签。核心零件几乎都是现成的：MoE 的门控直觉、KV cache、attention mask、Fisher 判别、hidden state 里的方向、prompt caching 的经济学。它做的事是把它们拼成一个**可逆的、单缓存的、不重读的上下文切换原型**，然后老老实实测了一遍。

但"拼得干净"本身是稀缺的。我最认可的是那个因果链：**因为不删除、只改 mask，所以撤销和子 Agent 才便宜**。这两个用法不是硬凑出来的 demo，是从设计里自然长出来的。这比多刷几个点有说服力得多。

而它最值钱的贡献，可能反而是那两节"拆自己台"的实验。一篇论文愿意写明"我的智能选块器不如关键词、我的探针换个任务就变差"，这在一片 SOTA 通报里是股清流。你要是只想拿一个数去汇报，93→119 拿走即可；你要是想判断这套东西能不能用，请一定读到 Figure 7。

![图10：把块连成图](https://www.mulanai.com/fs/files/1122_3d7fd6e4_fig9_gra.png)

*图 10：Blocks are linked —— a test log links to the file it tested, a reply links to the message it answers。Routing：从当前步（橙色 task）出发沿链接扩散，每跳衰减，保留最强节点直到预算，图中是 3 of 10。蓝色高亮的是 read login.py、run tests。*

Section 7 把下一步指向了 **Blocks as a graph**：块之间本来就有链接，存下这些链接，选块就从"逐块打分"变成"在图上路由"——从当前步出发，沿链接往外走，每跳权重递减，按预算（正文举例留全部块的 30%）保留最强的节点。作者说这能避免留重复块，也能把单看毫不相关、连起来却关键的块拉进来。

我觉得这个方向比那个线性探针有戏得多。逐块 yes/no 天生看不见关系，而"哪个文件导致测试失败"这类真正难的相关性，本来就是图上的路径问题。Next steps 里还列了：换关键词会失效的更难任务、在多种任务类型上训 decide pass、跟 embedding retriever 正面对比、把 picker 和 model 成对训练、上更大模型、以及推动 serving 侧支持块切换。

如果你也在做长程 Agent，三条能马上带走的：

最便宜的一条修复是**笔记别复述错误输出**：撤销实验里 5/13 的照抄全来自笔记里那行旧值，改掉这一行就省事了。

再往下一层，**把隐藏做成掩码而不是删除**。可逆性带来的撤销、子 Agent、A/B 试错，价值往往超过省下的那点 token，代价是你得能改推理引擎。

还有一条心态上的：**别指望小模型自己选块**。词面线索明显的任务上老老实实上检索；把学习型的 decide pass 留给"相关性藏在结构里"的场景，而且一定要做跨任务迁移测试。

回到开头那个复读旧值的 Agent。这篇论文没给我一个可以直接 pip install 的判断器，但它让我把问题看清楚了：不是上下文太长，是**每一步都被迫看了不该看的东西**；而且"藏起来"和"删掉"是两件完全不同的事。至于到底该由谁来选，作者很诚实地说——还没做出来。

这大概就是我愿意为它写这么长的原因。一个把边界画得清清楚楚的半成品，比一个包装得无懈可击的结论有用。

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我*
