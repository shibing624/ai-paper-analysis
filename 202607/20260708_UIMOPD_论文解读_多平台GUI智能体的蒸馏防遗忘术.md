---
title: UI-MOPD：多平台GUI智能体的蒸馏防遗忘术
date: 2026-07-08
arxiv: 2607.04425
tags: [GUI Agent, On-Policy Distillation, Continual Learning, VLM]
---

# UI-MOPD：多平台GUI智能体的蒸馏防遗忘术

你有没有过这种崩溃时刻？明明是同一个“取消”操作，在 Windows 上要按 Esc，在手机上要点左上角箭头，在网页上要按浏览器后退。**同一个意图，三套动作语义。**

这个看似不起眼的差异，正是多平台 GUI 智能体训练里最让人头疼的拦路虎。一个模型如果想同时玩转桌面和移动端，最容易翻车的不是“做不对任务”，而是**做着做着就把另一边的特色动作给忘了**。

清华深研院、小米、哈工深、浙大、鹏城实验室联合提出的 **UI-MOPD**（arXiv:2607.04425），就是冲着这个问题来的。它的核心思路很直接：别把桌面和手机的训练信号搅成一锅粥，让**两个平台专属教师分别带一个 8B 小学生**，通过**平台条件路由的在线策略蒸馏**把平台特色“焊”在一个共享策略里。

最终效果：在 OSWorld 上拿到 **38.2%** 的任务成功率（比基线涨 12.7%），在 MobileWorld 上拿到 **12.0%**（比基线涨 **55.8%**）。更关键的是，**8B 小模型在 MobileWorld 上反超了 32B 同源底模**（9.4%），说明提升不是靠参数堆出来的。

---

## 论文信息

| 项 | 内容 |
|---|---|
| 标题 | UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning |
| 作者 | Niu Lian*, Alan Chen*, Zhehao Yu, Chengzhen Duan, Fazhan Liu, Hui Liu, Pei Fu, Jian Luan, Yaowei Wang, Shu-Tao Xia, Jinpeng Wang★ |
| 机构 | 1. 清华深研院；2. 小米；3. 哈工深；4. 浙大；5. 鹏城实验室 |
| 发表 | 2026-07-05（arXiv:2607.04425v1） |
| 链接 | https://arxiv.org/abs/2607.04425 |
| 项目页 | https://elispectre.github.io/UI-MOPD/ |

> 注：Niu Lian、Alan Chen 为共同一作；Jinpeng Wang 为通讯作者（xiaomi 作者 / 通讯单位归属看正文 Affiliations）。

---

## 问题动机：为什么“多平台”这么难？

论文把痛点拆成两块，我换个接地气的说法重新讲一下。

**痛点一：高质量的跨平台轨迹数据稀缺。** 公开的 GUI 数据集基本都偏单平台——OpenCUA 主打桌面，OpenMobile 主打移动。即使把两者堆一起，里面还夹杂着格式不统一、动作空间不兼容、任务粒度不一致的脏数据。你拿这种数据训模型，等于让一个厨师同时学川菜和法餐但只给他一本半成品食谱。

**痛点二：行为模式“互踩”。** 这是更隐蔽也更致命的问题。论文给了一个非常直观的例子：

> 在桌面上，“返回上一个上下文”= 关闭窗口
> 在移动端，“返回上一个上下文”= 按返回键
> 在网页上，“返回上一个上下文”= 浏览器后退

三个平台的“返回”在用户意图层面是同一件事，但底层动作序列完全不同。如果直接把多平台轨迹丢进去做 SFT 或 model merging，模型学到的会是一个**平均化策略**——啥都会一点，但哪个平台都不精。这就是论文里反复出现的 “behavioral pattern mixing” 和 “catastrophic forgetting”。

更要命的是，多平台训练天然带“持续学习”属性：先在桌面 SFT，再去手机上做 SFT，桌面能力几乎肯定掉。这一点在论文 Table 2 里被验证得极其尖锐——**只在 OSWorld 上做 SFT 的 8B 模型，MobileWorld 成绩直接归零**。

---

## 核心方案：平台条件路由 + 多教师在线蒸馏

UI-MOPD 的整体流水线分两阶段：

| 阶段 | 操作 | 输出 |
|---|---|---|
| Stage 1 | 在 Uni-GUI 上对 Qwen3-VL-32B-Thinking 做平台分开的 SFT | 桌面教师 π_ref^d + 移动教师 π_ref^m |
| Stage 2 | 用 RL + MOPD 训练 8B 学生（Qwen3-VL-8B-Thinking 初始化） | 一个共享策略 π_θ，桌面移动都能用 |

下面这张图把整体思路画得很清楚：

![Figure 1: UI-MOPD 的设计动机](https://arxiv.org/html/2607.04425v1/x2.png)

*图 1：UI-MOPD 的设计动机。三种“看起来能行其实不行”的方案 vs 论文的 MOPD 方案。左边 (a) 是模型参数空间硬合并 / Action-Space 冲突——直接把桌面教师和移动教师的权重相加，结果两边特色动作互相打架；中间 (b) 是 Mixed SFT——把桌面轨迹和移动轨迹混在一起训，policy 被压成一个平均分布；右边 (c) 是 UI-MOPD 提出的方案，用一个 Platform-Router 把不同环境的 rollout 路由到对应平台教师，再做 on-policy distillation。*

说实话看到 (a) 和 (b) 这俩反例的时候，我第一反应是“这不就是大家都在干的事吗”。**Model merge 和 Mixed SFT 是过去两年开源社区最常用的多能力整合手段，简洁到几乎没人质疑它有没有问题。** 这篇论文至少给出了一个清晰的反例：MobileWorld 上，TIES-Merging 直接 **0%**，Mixed SFT 也只有 6.4%，而 UI-MOPD 是 12.0%。

下面这张是训练 pipeline 全貌：

![Figure 2: UI-MOPD 训练 pipeline](https://arxiv.org/html/2607.04425v1/x3.png)

*图 2：UI-MOPD 训练 pipeline 全貌。左边 Stage-1 通过统一 Harness（Kimi-K2.6 采桌面 / Gemini-3.1-Pro 采移动）收集 Uni-GUI，再分别 SFT 出桌面 / 移动教师；右边 Stage-2 是 MOPD 主舞台——学生自己 on-policy rollout，平台 Router 根据数据来源标签把 token 送进对应教师取 logit，然后和 rule-based ORM 算出的 token 级 advantage 一起进入 GRPO-DAPO 目标。Reverse KL 那一支完全发生在学生真正访问的 state 分布上，不是离线模仿。*

---

## 方法细节：把公式翻译成“人话”

论文方法部分写了 5 个小节，我挑 3 个最关键的工程点展开。

### 1. On-Policy KL：蒸馏发生在学生自己的 state 分布上

这是 MOPD 和普通 SFT / 离线蒸馏最大的区别。普通蒸馏的逻辑是：拿教师自己的轨迹来教学生。但教师的轨迹分布和学生当前分布经常对不上，强行模仿只会让学生在教师擅长的状态上“被绑死”。

UI-MOPD 的做法是：让学生**自己采样** rollout（on-policy），然后在学生真正访问的每个 token 上，对学生的策略分布和对应平台教师的策略分布算一个**学生→教师的 KL**：

$$D_{\mathrm{KL}}^{(t,i)} = D_{\mathrm{KL}}\big(\pi_\theta(\cdot \mid h_t^{(i)})\,\|\,\pi_{\mathrm{ref}}^{(i)}(\cdot \mid h_t^{(i)})\big)$$

直觉上，**这个 KL 是“如果你想偏离教师当前行为，代价有多大”的衡量**。配合后面的 reward advantage 一起最大化，相当于告诉学生：“你想探索可以，但别把平台教师教的动作分布全扔了。”

实操里直接算全词表 KL 太贵，论文用了 **K3 estimator**：

$$\hat D_{\mathrm{KL}}^{(t,i)} = \rho_t^{(i)} - \delta_t^{(i)} - 1, \quad \rho_t^{(i)} = \exp(\delta_t^{(i)})$$

其中 $\delta_t^{(i)} = \log \pi_{\mathrm{ref}}(y_t \mid h_t) - \log \pi_\theta(y_t \mid h_t)$，δ 实现里会做 clamp 防数值溢出。**K3 estimator 只需要采样 token 处的 logprob，不用算完整词表分布**，这是它在大模型蒸馏里能跑起来的关键。

### 2. Platform-Conditioned Routing：路由是这套方案的灵魂

这是 UI-MOPD 和一般 MOPD（一个学生 vs 多个教师用同一种信号）的最大差异点。论文的路由规则极其简单：

$$\pi_{\mathrm{ref}}^{(i)} = \begin{cases} \pi_{\mathrm{ref}}^m, & s_i \in \mathcal{S}_{\mathrm{mobile}} \\ \pi_{\mathrm{ref}}^d, & s_i \in \mathcal{S}_{\mathrm{desktop}} \end{cases}$$

也就是说，**每条 rollout 在算 teacher logprob 时，只看它的数据来源平台标签**。桌面 rollout → 桌面教师；移动 rollout → 移动教师。**没有“教师投票”、没有 logits 平均、没有 ensemble**。

听起来粗暴，但工程上意义很大：
- 路由**只影响 teacher logprob 计算**，学生始终是单一共享策略，推理时只部署一个模型
- 平台教师的“行为锚点”被显式钉在共享参数空间里，不会被其他平台的梯度带跑
- 实现成本极低——只需要在 batch 维度按 platform 切分，分别 forward 一次对应教师，再 merge 回原顺序

### 3. Adaptive KL Mask：让蒸馏在“对的时候”才发力

这个细节我特别喜欢，论文里就一段话，但工程味道很对。KL 太强会把学生压死，太弱又压不住行为漂移。论文的做法是按 prompt group 的平均 reward 设了一个开关：

$$\mu^{(i)} = \begin{cases} 0, & \frac{1}{G}\sum_{k \in g(i)} R^{(k)} > \tau_{\mathrm{KL}} \\ 1, & \text{otherwise} \end{cases}$$

**意思就是：当某组 prompt 的平均 reward 已经够高，模型对这部分任务“已经学会了”，就别再让 KL 拽着它。** 反过来，在低 reward 区域，teacher 仍然要拉一把，避免学生在自己没把握的地方自由发挥。

这个机制其实就“**reward gating + KL masking**”，避免无差别蒸馏压制了高 reward rollout 的探索空间。

### 训练目标：RL + MOPD 的优雅拼装

最终学生要最大化的目标：

$$\mathcal{J}(\theta) = \mathbb{E}_{p, x, y \sim \pi_\theta}\Big[\sum_t m_t\big(\ell_{\mathrm{PG}}^{(t)}(\theta) - \beta \mu \hat D_{\mathrm{KL}}^{(t,p)}\big)\Big]$$

其中 $\ell_{\mathrm{PG}}^{(t)}$ 是带双侧 clip（low=0.2, high=0.28）的策略梯度，advantage $A_t$ 来自 DAPO 风格 GRPO。β 是 KL 系数，论文里设的 **0.01**——相当克制，主要靠 policy gradient 拿 reward，蒸馏只做“行为防漂移”。

**奖励设计**也值得一提，不是简单的 0/1，而是结构化的三档：
- 动作所有维度都匹配：+1.0
- 部分匹配：−0.5
- 解析不了 / 非法动作：−1.0

中间这档 −0.5 是关键，它给 group-relative advantage 留出了一个有用的间隔，让 GRPO 能区分“做对一半”和“完全错乱”两种情况。

---

## 数据：Uni-GUI 不是堆出来的

论文附带贡献是 **Uni-GUI** 数据集——一个统一跨平台数据采集 Harness 的产物。

| 平台 | 来源 | 步数（约） | 轨迹数（约） |
|---|---|---|---|
| 桌面 | 自采（Kimi-K2.6） | 95K | 7K |
| 桌面 | OpenCUA 清洗 | 13K | 0.8K |
| 移动 | 自采（Gemini-3.1-Pro） | 17K | 1K |
| 移动 | OpenMobile 清洗 | 35K | 2.7K |
| **合计** | — | **~160K** | **~11.5K** |

数据收集 Harness 的四个阶段：

1. **Query Generation**：让教师模型从可执行功能点反推用户 query，避免“指令看起来合理但环境里压根做不了”
2. **Trajectory Collection**：保持平台原生动作（不强行归一化）但用统一高层接口记录
3. **Trajectory Cleaning**：四道清洗——剔除结构损坏、剔除动作空间不兼容、剔除超过 40 步、剔除 query 和环境不匹配；再用 Gemini-3.1-Pro 作为裁判做**子任务级**判定，必须所有子任务都通过才保留
4. **Post-Processing**：把 Kimi/Gemini 异构的 reasoning 改写成 Qwen3-VL 风格的 structured CoT；重新标注 grounding bbox

![Figure 4: 统一跨平台数据采集 Harness](https://arxiv.org/html/2607.04425v1/x5.png)

*图 4：统一跨平台数据采集 Harness 全景。Query Generation 从环境里反推功能点→合成 query，Trajectory Collection 用 Kimi / Gemini 在桌面 / 移动环境里跑出原始轨迹，Trajectory Cleaning 用四种过滤器 + 自动裁判做质量控制，Post-Processing 做 CoT 改写和 bbox 重标注。*

我注意到一个工程细节挺讲究：**清洗阶段的“子任务级判定”而不是“整条轨迹一次性判定”**。这避免了“长 trajectory 一次性判断时裁判被上下文绕晕”的问题，也给失败执行归因提供了更清晰的颗粒度。**这种“先把任务拆开再判”的思路，和 Process Reward Model (PRM) 的精神其实是相通的。**

---

## 实验结果：数字看起来漂亮，但要分开看

### 主表：OSWorld + MobileWorld 对比

| 类别 | 方法 | OSWorld | MobileWorld |
|---|---|---:|---:|
| 通用模型 | SeedVL-1.5 | 34.1% | – |
| 通用模型 | Qwen3-VL-8B-Instruct | 33.9% | 9.4% |
| 通用模型 | Qwen3-VL-8B-Thinking | 33.9% | 7.7% |
| 通用模型 | Qwen3-VL-32B-Instruct | 32.6% | 9.0% |
| 通用模型 | Qwen3-VL-235B-A22B-Instruct | 31.6% | 9.5% |
| 通用模型 | Qwen3-VL-235B-A22B-Thinking | **38.1%** | – |
| GUI 单平台 | OpenCUA-7B | 28.2% | – |
| GUI 单平台 | OpenCUA-32B | 34.8% | – |
| GUI 单平台 | OpenAI CUA o3 | 31.3% | – |
| GUI 多平台 | UI-TARS-72B-DPO | 27.1% | – |
| GUI 多平台 | UI-TARS-1.5-7B | 27.4% | – |
| GUI 多平台 | GELab-Zero-4B | 31.9% | 10.9% |
| GUI 多平台 | GUI-Owl-7B | 34.9% | 4.5% |
| GUI 多平台 | GUI-Owl-32B | – | 5.5% |
| 集成策略 | Mixed-SFT | 35.0% | 6.4% |
| 集成策略 | Model Merge (Weight Averaging) | 36.5% | 6.8% |
| 集成策略 | Model Merge (TIES) | 36.8% | **0%** |
| **本文** | **UI-MOPD** | **38.2%** | **12.0%** |

几个我想拎出来说的点：

1. **OSWorld 上 38.2% 超过了 235B-A22B 的 38.1%。** 这个 0.1 个点的领先其实非常戏剧性——学生 8B 超过老师（不是这个老师，但同系列）235B 整整一个量级。在 MobileWorld 上更夸张，**8B 学生（12.0%）> 32B 老师（9.4%）> 8B 底模（7.7%）**，提升完全是 method 而不是 scale。

2. **TIES-Merging 在 MobileWorld 上是 0%。** 这个数字真的让人倒吸一口气——把两个平台专属模型直接做参数合并，移动能力直接归零。**这相当于论文用“硬合并 = 0”这件事，当面打了 model merging 一耳光。** 论文 Table 2 里那个“只在 MobileWorld 上 SFT 8B 模型，在 OSWorld 上 35.8% / 在 MobileWorld 上 12.8%”反过来也证明了这个“互踩”问题。

3. **GUI-Owl-7B / 32B 在 OSWorld 34.9% / MobileWorld 4.5% / 5.5%。** 同期多平台 GUI 模型在两个 benchmark 上严重不平衡——GUI-Owl-7B 桌面强但移动直接掉到 4.5%，UI-MOPD 是**目前唯一在两个平台上都拿得出手**的多平台模型。

不过说实话，**MobileWorld 12.0% 这个绝对值其实挺低的**。论文里给的相对提升 55.8% 听着很猛，但分母是 7.7%，也就是从 7.7% → 12.0%，绝对值也就 4.3 个点。我们可以理解成“移动端 GUI 任务本身就这么难”，但**这也是这篇论文最大的天花板**——它没有把移动端打到 20%+，而是在“比烂”的环境里拿到了第一。

### 跨平台能力分析（Table 2）

| 方法 | OSWorld | MobileWorld |
|---|---:|---:|
| Qwen3-VL-8B-Thinking（基线） | 33.9% | 7.7% |
| Qwen3-VL-32B-Thinking | 41.0% | 9.4% |
| 8B SFT on OSWorld | 35.8% | **0%** |
| 8B SFT on MobileWorld | 35.8% | 12.8% |
| Desktop Teacher, 32B | 46.3% | – |
| Mobile Teacher, 32B | – | 16.2% |
| **UI-MOPD** | **38.2%** | **12.0%** |

这张表非常关键。**32B 平台教师在单平台上分别拿到 46.3% / 16.2%，这就是 UI-MOPD 想“蒸馏进来”的目标能力。** 而 UI-MOPD 的 8B 学生分别拿到 38.2% / 12.0%——把桌面教师的 82%、移动教师的 74% 蒸馏到了 8B 学生里，且**两边都能 work**，没有出现单平台 SFT 那种“一个涨另一个归零”的灾难性遗忘。

### 静态能力保留（Table 3 / 7）

论文还做了 AndroidControl★、ScreenSpot-Pro、V2、OSWorld-G 的静态 GUI 评估。这部分是用来回答一个问题：**RL + MOPD 训下来，模型的“基础认知”会不会掉？**

| 模型 | AndroidControl★ | ScreenSpot-Pro | ScreenSpotV2 | OSWorld-G |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B-Thinking（基线） | 78.73% | 43.71% | 91.27% | 52.13% |
| Model Merge (TIES) | 74.01% | 37.13% | 88.60% | 47.16% |
| **UI-MOPD** | **80.05%** | 43.14% | 90.88% | **52.84%** |

**UI-MOPD 在所有四项静态评估上要么涨要么基本持平；Model Merge 在四项上全部掉档。** 这再次印证了“硬合并会破坏基础能力”这件事。AndroidControl★ 上 UI-MOPD 比基线涨了 1.32 个点（80.05% vs 78.73%），说明**MOPD 不仅没破坏 mobile GUI 理解，反而还有微弱提升**。

### 案例展示

下面是一个 mobile 任务执行示例：

![Figure 3: 移动端任务执行示例](https://arxiv.org/html/2607.04425v1/x4.png)

*图 3：UI-MOPD 在 mobile 端的一个长程任务示例。用户指令是“回复一个关于希腊菜 Moussaka 的帖子，回复内容是 'Nice sharing, I love it.'”。模型从首页开始，依次进入 Inbox、找到目标邮件、点击附件、添加 'waiver.jpg'（任务要求附件为同名图片）、进入 Compose 邮件、填写收件人 / 主题 / 正文、最终 finished 完成任务。中间展示了 structured CoT：[Observation] 看到什么、[Plan] 打算做什么、[Reasoning] 为什么这么做、[Expected] 预期结果。*

注意看 structured CoT 那一栏——**作者把“看到 → 计划 → 推理 → 预期”这四步显式拆开**。这不是论文的副产物，而是 **Uni-GUI 后处理阶段 CoT 改写的直接效果**：让模型的 reasoning 和 Qwen3-VL 本身的输出风格对齐，避免异构 reasoning 训练时把模型的输出分布带偏。

下面是一个 desktop 任务执行示例：

![Figure 5: 桌面端任务执行示例](https://arxiv.org/html/2607.04425v1/x6.png)

*图 5：UI-MOPD 在 desktop 端的任务示例。用户指令是“把当前 LibreOffice Calc 的数据转到 LibreOffice Writer 表格里，保持原始 calc 文件格式，保存为 'price.docx' 到桌面”。模型依次：moveTo → dragTo 选中区域 → 热键 ctrl+c 复制 → 切换窗口 → ctrl+v 粘贴 → 处理 File 菜单打开异常（structured CoT 里明确写了“File 菜单 still showing, 问题可能是 click 坐标和菜单 item 不匹配”）→ 改用 Save As → 选择 Desktop → 输入文件名 → 最终 DONE。*

这个 desktop 案例特别有意思的地方，是模型在 File 菜单 click 没打开时**自己发现并改用 Save As 路径**。structured CoT 里的“问题可能是 click 坐标和菜单 item 不匹配”这句话，**说明模型在推理时已经在做“动作执行效果反思”**——这其实有点 RL with self-correction 的味道，不是简单的 if-else 模式匹配。

---

## 我的判断

这篇论文我觉得有几个点值得多聊几句。

### 亮点 1：把“first to introduce MOPD into GUI agent”这件事做得很扎实

MOPD 在 LLM 后训练里已经不算新事物——MiMo-V2-Flash、GLM-5、Nemotron-Cascade 2、DeepSeek-V4 都用过。**UI-MOPD 的贡献不是发明 MOPD，而是把它在 GUI agent 场景下重新定义了一次**：

- **教师从“能力维度”变成“平台维度”**。原版 MOPD 是“数学用一个教师、写作用另一个教师”，UI-MOPD 是“桌面用一个教师、移动用另一个教师”。这种“按交互异质性切分教师”的视角在 GUI agent 圈是新的。
- **路由维度从“能力分类器”变成“数据来源标签”**。原版 MOPD 需要一个能力判别器决定当前 token 找谁蒸馏；UI-MOPD 直接用 trajectory 自己的 platform 标签——更便宜、更稳、不需要额外训练判别器。
- **从“任务 reward + 教师 KL”变成“行为锚点 + 教师 KL”**。原版 MOPD 蒸馏的是能力，UI-MOPD 蒸馏的是行为锚点（防止平台特征被覆盖），β 系数小到 0.01 也说明这点。

### 亮点 2：Table 2 的对照实验太关键了

单纯报告“我们训了一个 8B 模型在 OSWorld 上 38.2%”，说服力有限。**Table 2 把“单平台 SFT 会导致灾难性遗忘”这件事，用极其干净的对照实验直接摆出来**：

- 8B SFT on OSWorld → OSWorld 35.8% / MobileWorld **0%**
- 8B SFT on MobileWorld → OSWorld 35.8% / MobileWorld 12.8%
- UI-MOPD → OSWorld 38.2% / MobileWorld 12.0%

**MobileWorld 0% 这个数字比任何 ablation 都更有说服力**。它把“为什么需要 UI-MOPD”这个问题直接变成了视觉冲击。

### 我没那么满意的地方

批判性地看，这篇论文也有几个让我想皱眉的点：

1. **MobileWorld 12.0% 的绝对值确实偏低。** 我理解移动端任务天然难，但论文没有深入分析“为什么 12% 就是上限”。32B 移动教师自己也就 16.2%——**是不是 mobile 任务在 32B 模型上本身就没训透，限制了学生蒸馏的天花板**？论文没讨论。

2. **Uni-GUI 11.5K 轨迹的规模优势不明显。** 对比 OpenCUA（已经包含约 0.8K 桌面轨迹），11.5K 看着不算巨大。论文没在数据规模维度做 ablation——**多 1 倍数据会涨多少？数据里“自采 vs 公开清洗”哪个贡献更大**？这些都没说。

3. **β=0.01 是不是过小了？** 论文几乎所有消融都是在说“蒸馏比不蒸馏好”，但**蒸馏强度对最终效果的影响曲线**没做。这是个遗憾——如果 β 调到 0.05 / 0.1 结果会怎样？会不会反而更差（验证 KL gating 有效）或者更好（说明 β 还有调优空间）？

4. **“Continual learning” 名字用得有点勉强。** 论文训练流程是 Stage-1 SFT（已经分平台）+ Stage-2 RL+蒸馏，**没有显式的“先学桌面，再学移动，再回桌面”这种 incremental 序列**。它更像“joint training with routed teacher”。continual learning 的典型痛点（任务顺序敏感性、显存累积、单平台数据访问）在这篇工作里没有显式处理。

5. **没有和 GUI-Owl-32B（Qwen 系列同体量多平台模型）在 MobileWorld 上的对比表。** 论文只在 OSWorld 列了 GUI-Owl-7B / 32B 的部分数据，MobileWorld 数据看起来也不全。这种“挑数据列”在论文里有点危险。

### 同期工作坐标

UI-MOPD 这篇工作可以放到下面这条线索里看：

| 方向 | 代表工作 | 思路 |
|---|---|---|
| GUI Agent 通用底座 | UI-TARS-2、Mobile-Agent-v3.5、UI-Venus-1.5 | 大数据 + 大模型 + 多平台统一训练 |
| 桌面 / 移动专用 | OpenCUA、UI-R1、EvoCUA、ComputerRL | 单平台深耕 |
| 持续学习 / 能力整合 | Nemotron-Cascade 2、GLM-5、MOPD 系列 | 在 LLM 后训练里做 MOPD 防遗忘 |
| 跨平台 GUI 数据 | OpenCUA、OpenMobile、Aguvis | 多平台数据集 |
| **本文** | **UI-MOPD** | **首个把 MOPD 引入 GUI agent + 平台条件路由** |

**“首个”是稳的**——我做了一下相关检索，GUI agent 领域做 MOPD 的工作目前确实少见，UI-MOPD 在这个细分赛道里是开创性的。

---

## 收尾

回到开头那个“返回操作”的问题。UI-MOPD 给出的工程答案其实很朴素：**别用一个大模型试图同时学会三套“返回”动作的细节，而是让桌面 / 移动两个专家教师分别把“特色”通过 KL 锚点喂给同一个学生**。这个“特色”不是新能力，而是**平台专属的行为分布**——论文里叫 “platform-specific behavioral prior”。

这个思路能不能直接复用到其他多模态 Agent 场景？我感觉是能的。比如 web 智能体 + 桌面智能体、Excel 智能体 + PPT 智能体，**只要不同平台的动作语义有结构性差异，平台条件路由 + MOPD 都能套上去**。

但也得说一句：**如果两个平台的动作差异没那么大（比如都是 GUI 只是布局不同），MOPD 的边际收益可能就很小**。UI-MOPD 之所以能 work，很大程度上是因为桌面和移动的动作空间（mouse+key vs tap+swipe）天然就隔着一道墙。

64 张 H100、32B 教师 + 8B 学生、1 epoch、β=0.01——这套配置在工业界其实并不算贵。**问题是效果天花板被 32B 教师自己（16.2% on MobileWorld）卡住了**。如果未来能把教师进一步做强（比如用 100B+ GUI 专用教师），MOPD 的潜在收益才会真正释放。

---

## 参考文献

1. Lian, N., Chen, A., Yu, Z., et al. UI-MOPD: Multi-Platform On-Policy Distillation for Continual GUI Agent Learning. arXiv:2607.04425, 2026.
2. Xie, T., et al. OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments. 2024.
3. Liu, S., et al. MobileWorld: Benchmarking Multimodal Agent in Mobile Scenarios. 2025.
4. OpenCUA / OpenMobile / Aguvis 等开源 GUI 数据集
5. GLM-5、DeepSeek-V4、Nemotron-Cascade 2 等 MOPD 后训练工作

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
