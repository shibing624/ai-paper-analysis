# ActObs：把环境观测也纳入 SFT 监督，改变 Agent 在 RL 下的探索方式

> 论文：**Don't Mask the Environment: Observation Supervision Changes How Agents Explore Under RL**
> arXiv ID：**2609.20715**（2026-09-17 提交，29 页，9 图，11 表）
> 作者：Juzheng Zhang（University of Maryland，AWS AI Labs 实习期间）、Disha Makhija、Manoj Ghuhan Arivazhagan、Vinayshekhar Bannihatti Kumar、Rashmi Gangadharaiah（AWS AI Labs）
> 链接：https://arxiv.org/abs/2609.20715

---

## 一句话总结

标准 Agent SFT 只对 action token 算 loss，把环境 observation 当上下文却不做预测目标；本文提出 **ActObs**——只需在 label mask 上放开 observation token，让它们也成为预测目标——不改变数据、参数、序列长度、前向次数，也不改 RL 算法。SFT 后两者表现几乎相同，但 GRPO 后显著分化：Qwen3-4B 上 ActObs→GRPO 在每个采样预算都领先；Qwen3-8B 上以少许 pass@1 换取 pass@16 +3.4pp 并解决更多不同任务；跨域代码编辑（aider-polyglot，任务在 SFT 与 RL 均未见过）上 pass@1 +4.2pp。机制上，action 与 observation 梯度在 10–20 步内变为正交，action-only 训练留下巨大残余 observation 梯度并侵蚀基座的环境预测能力，而联合监督保留了"预测行动后果"的能力，使 RL 以更小策略移动保留更多熵，高 k 采样覆盖率更高。

## 核心问题

Agent 轨迹记录了"agent 做了什么、环境随后发生了什么"。标准 SFT 的惯例是：loss 只打在 agent 编写的 action token 上，环境 observation 仅作为上下文。问题是——**这个惯例是否为后续 RL 提供了最好的初始化？**

论文的反直觉回答是：应该训练策略去**预测它在部署时永远不会生成的 token**（环境观测）。部署时 agent 只生成 action，但学习预测 observation 等于强制策略建模"行动的后果"。

## 方法：ActObs

轨迹定义为 $\tau=(x,a_1,o_1,a_2,o_2,\dots,a_T,o_T)$，$x$ 为任务提示，$a_t$ 为 assistant turn（含 reasoning 与命令），$o_t$ 为环境响应。对 token 化轨迹 $y_{1:N}$，$\mathcal{A}$ 与 $\mathcal{O}$ 分别为 action / observation token 索引，目标函数：

$$\mathcal{L}_{\lambda}(\theta)=-\frac{\sum_{i\in\mathcal{A}}\log p_{\theta}(y_{i}\mid y_{<i})+\lambda\sum_{i\in\mathcal{O}}\log p_{\theta}(y_{i}\mid y_{<i})}{|\mathcal{A}|+\lambda|\mathcal{O}|}$$

- $\lambda=0$ 退化为 action-only SFT（ActionSFT）；ActObs 默认 $\lambda=1$（action 与 observation 等权）
- prompt 在两种目标中都保持 masked
- 分母使 per-example loss 尺度不随 $\lambda$ 变化
- 在 observation 位置，目标训练 $p_\theta(o_t\mid h_t,a_t)$，每条轨迹**既是 imitation 样本又是 transition（环境转移）样本**
- 实现上**只需改 label mask**：零额外数据、零额外参数、零额外序列 token、零额外前向传播，RL 算法完全不变

**对比变体**：
- **ActionSFT**：只对 action 算 loss
- **Obs→Act**：时序控制——先 1 epoch observation-only SFT，再 1 epoch action-only SFT（observation 监督总量与 ActObs 相同，但非联合）
- **ECHO**：在 GRPO 中对 policy 自己 rollout 的 observation 加权重 0.05 的 next-observation loss（RL 阶段持续监督的对照）

## 实验设置

| 项目 | 细节 |
|---|---|
| SFT 语料 | 50k 多轮 terminal 轨迹（0.71B tokens），Nemotron-Terminal-Corpus 合成部分，DeepSeek-V3.2 生成；observation 约占 45% token |
| 模型 | Qwen3-4B / Qwen3-8B |
| SFT 超参 | 1 epoch，781 步，batch 64，cosine，peak lr 1e-5 |
| RL | GRPO，135 步，2,392 个容器化任务（Endless Terminals，与 SFT 语料不相交），binary verifier，16 rollouts/task，batch 32，lr 1e-6 |
| 基准 | Terminal-Bench 2.0（89 个 OOD terminal 任务，terminus-2 agent，每 checkpoint 16 attempts）；aider-polyglot（225 个跨域代码编辑任务，6 种语言，单测判分，SFT/RL 均未见过，4 attempts） |
| 指标 | pass@k（Chen et al. 2021 无偏估计），bootstrap 标准误（20,000 次重采样） |
| 推理 | 单 vLLM，8×A100 张量并行，40,960 上下文，T=0.6，top-p 0.95，top-k 20 |

## 主要结果

### Terminal-Bench 2.0（Qwen3-4B）

| 模型 | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| Base | 1.2 | 3.3 | 4.7 | 5.6 |
| ActionSFT | **4.5** | **10.4** | **14.0** | 16.9 |
| ActObs | 4.4 | 9.9 | **14.0** | **18.0** |
| ActionSFT→GRPO | 5.6 | 11.1 | 14.0 | 18.0 |
| **ActObs→GRPO** | **7.2** | **12.5** | **15.5** | **19.1** |
| ActionSFT→ECHO | **5.9** | 11.6 | 14.7 | 18.0 |
| ActObs→ECHO | **5.9** | **12.7** | **16.5** | **20.2** |

4B 下 GRPO 后 ActObs 在每个采样预算领先（pass@1/4/8/16 分别 +1.6/+1.4/+1.5/+1.1pp，相对 +29%/+13%/+11%/+6%）。

### Terminal-Bench 2.0（Qwen3-8B）

| 模型 | pass@1 | pass@4 | pass@8 | pass@16 |
|---|---|---|---|---|
| ActionSFT→GRPO | **12.3** | 18.7 | 21.0 | 23.6 |
| **ActObs→GRPO** | 11.0 | **19.3** | **23.6** | **27.0** |
| Obs→Act→GRPO | 11.9 | **19.3** | 21.7 | 23.6 |

8B 下 ActionSFT 的 RL 增益集中在小 k，且 k≥8 时转为负（pass@8/16 分别 -1.0/-1.1）；而 ActObs 的 RL 增益随 k 增长（pass@16 **+4.5**）。ActObs 解决 24 个任务（vs 21），其中 3 个是任何 SFT 策略或 ActionSFT→GRPO 都解不出的。

### aider-polyglot（跨域代码编辑，任务完全未见过）

| 模型（4B） | pass@1 | pass@4 |
|---|---|---|
| ActionSFT→GRPO | 9.7 | 20.4 |
| **ActObs→GRPO** | **13.9** | **25.3** |

尽管 ActObs 的 SFT checkpoint 在代码编辑上反而更弱（1.0 vs 1.4），GRPO 后 pass@1 +4.2pp、pass@4 +4.9pp（相对 +43%/+24%）。GRPO 将 ActObs 的 pass@1 提升 12.9pp、pass@4 提升 21.7pp。

### RL 期间增益对比（相对各自 SFT 初始化的 Δpass@k）

| 初始化 | Δ@1 | Δ@4 | Δ@8 | Δ@16 |
|---|---|---|---|---|
| 8B ActionSFT | +3.1 | +0.5 | **-1.0** | **-1.1** |
| 8B ActObs | +3.1 | +3.2 | +3.9 | **+4.5** |
| 4B ActionSFT | +1.1 | +0.8 | +0.1 | +1.1 |
| 4B ActObs | **+2.7** | +2.6 | +1.5 | +1.1 |

SFT 目标不改变即时性能，但**塑造了下游 RL 的学习与探索能力**。

## 机制分析（论文精华）

### 1. RL 训练动态（Figure 3）

两个 run 起始 reward 与 entropy 相近，随后分化：ActObs reward 更平稳上升且终值更高；**training entropy 在 RL 后期持续上升**，而 ActionSFT 的 entropy 下降后仅部分恢复；ActObs 的 rollout 逐渐变短并保持更短——说明其高后期熵不是来自生成更长轨迹。8B 有相同熵分离现象。

### 2. 熵保留 + 小策略移动（Figure 4）

- **Self-entropy**：冻结 RL endpoint，在其自己的 200 条评估 rollout 上测 assistant token 平均 next-token entropy
- **Endpoint KL**：在相同 200 条固定 trace 的 action 位置测 $\mathrm{KL}(\pi_{\mathrm{init}}\|\pi_{\mathrm{final}})$
- 联合排序：**ActObs = 最小 endpoint 位移 + 更多保留熵 + 最高 pass@16**；Obs→Act 移动最远、保留熵最少
- **高熵本身不充分**：ECHO 产生最高 endpoint entropy 但 pass@16 不及 ActObs→GRPO
- **温度匹配控制**：把 ActionSFT→GRPO 采样温度提到 T*=0.64 使其 self-entropy 匹配 ActObs→GRPO，pass@k 变化 ≤0.7，未解决任何新任务——高 k 优势来自 **policy 学到的分布**，而非推理时注入的随机性

### 3. 梯度正交性（Figure 5，核心发现）

在相同 256 条 held-out 轨迹上分别计算 action / observation 梯度，分解：

$$g_{\mathrm{obs}}=r\lVert g_{\mathrm{act}}\rVert\left(c\,\hat{g}_{\mathrm{act}}+\sqrt{1-c^{2}}\,\hat{g}_{\mathrm{act}}^{\perp}\right)$$

其中 $c=\cos(g_{\mathrm{obs}},g_{\mathrm{act}})$，$r=\lVert g_{\mathrm{obs}}\rVert/\lVert g_{\mathrm{act}}\rVert$。

- **初始对齐，迅速正交**：预训练处 $c=0.83$，但 **10–20 个 SFT step 内 $c$ 降到经验噪声下限**——observation 梯度的平行分量消失，只剩几乎完全正交的信号
- **范数比分化**：ActionSFT 下 $r$ 升至约 **41**（拟合示范动作使 $g_{\mathrm{act}}$ 变小，未拟合的 $g_{\mathrm{obs}}$ 保持很大）；ActObs 下 $r$ 稳定在 **0.5** 附近
- **含义**：ActionSFT 对 action loss 接近平稳，但在正交的 observation 方向上仍陡峭——action-only 拟合**侵蚀了基座模型的环境预测能力**（teacher-forced observation cross-entropy 上比 base model 更差），而 ActObs 保留了它
- **熵分离时点**：on-policy entropy 曲线在**第 100 步（共 781 步）**即分离并贯穿 SFT 结束

### 4. Token 级效应（Figure 6）

- GRPO 后，ActObs 与 ActionSFT 在命令**前几个 token**同样 sharp，分离出现在**命令后部**——编码 arguments、flags、paths 的选择。ActObs 不是让 agent 在工具调用开头犹豫，而是**保留了命令实例化方式的变化**（案例：ActObs 逐个扫描 QEMU 启动参数找到可用内存配置）
- Teacher-forced observation cross-entropy：固定 wrapper（"New Terminal Output:"）带来最大早期增益，但 ActObs 在靠后位置及 shell prompts、error lines 等 payload 上也有改进；**ActionSFT 在大部分序列上比 base model 更差**

### 5. 消融（Table 2）

- **act2act**（第二轮 action-only SFT）≈ ActionSFT——更多优化不能恢复 ActObs 行为
- **ActObs-100→GRPO**（前 100 步联合后转 action-only）pass@8 仅 19.1（vs 全程联合 23.6）——observation 目标在早期更新后仍有用
- **Observation-only SFT** 与 **action→obs-only** 均解决 0 个任务——observation 预测是 action 学习的补充而非替代
- **Shuffled obs**（动作配错观测）pass@1 从 7.9 降至 3.7；而可区分的 wild logs 保持 7.9——**配对正确性重要**
- 完整 terminal 响应的监督强于仅选 salient 行
- **λ 扫描**：post-GRPO pass@8 优势随 λ 单调增长，pass@1 反向移动——是 **pass@1↔pass@8 权衡**，而非整体平移

## 局限与备注

- 仅在 terminal / 代码编辑两类环境验证；observation 占比 45% token 的设置不一定普适
- 8B 上 pass@1 有损失（-1.3pp），追求单次可靠性场景需权衡
- 依赖高质量配对轨迹；shuffled 消融表明错误配对有害

## 个人点评

这篇论文的价值在于把一个"零成本改动"（放开 label mask）讲成了一个完整的机制故事：SFT 目标 → 梯度正交性 → 环境预测能力保留 → RL 熵保留 + 小策略位移 → 高 k 覆盖。最有说服力的证据链是：温度匹配控制实验排除了"只是更随机"的解释；ActObs-100 消融排除了"只是早期正则化"；obs-only 全崩排除了"observation 监督本身有用"。对实践者的启示直接：做 agent SFT 时不要再默认 mask 掉环境观测——它几乎免费，却决定了后续 RLVR 是"pass@1 涨、覆盖收缩"还是"全 k 普涨"。
