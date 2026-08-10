# Fail-Fast, Restart-Smart：SWE 智能体任务的早期失败预测与重启

## 论文信息

- **标题**：Fail-Fast, Restart-Smart: Early Failure Prediction and Restart for SWE Agentic Tasks
- **作者**：Chenyu Wang, Yunbo Lyu, Junda He, Zhou Yang, Chenxing Zhong, Yaniv Harel, David Lo
- **发表**：arXiv:2608.03222（2026 年 8 月 4 日提交，v1）
- **链接**：https://arxiv.org/abs/2608.03222
- **领域**：Software Engineering (cs.SE); Artificial Intelligence (cs.AI)

## 一句话总结

针对 SWE 智能体长轨迹中"失败运行更长、更烧钱"的问题，作者提出两阶段控制器 **FailFast-RestartSmart**：用一个 0.6B 轻量监控器（FailFast）基于轨迹前缀预测失败并提前中止，再以同策略新 rollout 重启（RestartSmart），将被中止运行的仓库 diff 作为"可选 overlay"提供给新 agent 参考，在 SWE-bench Verified 上以 5% 误报率节省 14.6%–20.4% token，并在 25% 误报率下将 Qwen3.6-27B 解决率从 66.6% 提升至 71.8%。

## 研究动机

1. **Token 雪崩效应**：SWE agents 通过长轨迹解决仓库级 issue，随上下文累积，每步成本递增。
2. **失败轨迹代价更高**：失败运行往往更长，表现出冗余探索、重复或循环（looping），暗示部分失败可在结束前被检测出来。
3. **早停的两难**：
   - 过早终止是干预而非纯预测问题——误报会中止本会成功的轨迹，即使预测准确也可能降低端到端性能；
   - 但失败并非毫无价值——AgentLens 将 **54.9%** 的失败轨迹归类为 *Partial-fail*（结构上接近已知良好流程），说明失败尝试中可能含有可恢复的部分进展。
4. **研究空白**：先前工作将"早期终止"与"恢复"孤立研究。真正开放的问题不仅是**何时停止**，而是**什么信息应跨越重启边界**，使恢复收益超过误报损害与重试成本。

核心研究问题：*"Can we stop a likely-to-fail trajectory early and turn the saved compute and partial progress into a more successful retry?"*

## 方法：FailFast-RestartSmart

整体框架是针对**单条活跃轨迹**的两阶段控制器：(i) FailFast 决定**何时中止**；(ii) RestartSmart 决定**携带什么**进入重试。

### 3.1 FailFast 监控器

**问题形式化**：将早期终止建模为逐步（per-step）失败预测；部署决策为预算化警报——在固定误报率（FPR，被错误中止的*成功*运行比例）下最大化召回率（被捕获的*失败*运行比例）。

**训练数据**：
- 由被评估策略模型 Qwen3.6-27B-FP8 在 SWE-bench Verified 上、使用 mini-swe-agent 框架、以 seeds 0–10 采样产生的轨迹；
- 每个训练样本是截断于第 t 步的轨迹前缀。

**状态表示**（仅含严格在线信息），序列化为两部分：
- **[ISSUE]**：任务说明书（issue 描述）；
- **[WINDOW]**：最近 **8 步**的 thought、action、observation，外加一个持久钉选（pin）的最近产生补丁的步骤。

**监督信号**：
1. **终端标签** y_final ∈ {0,1}：最终是否成功（稀疏、延迟的监督）；
2. **密集进度目标** g_f2p, g_p2p ∈ [0,1]：在全新实例环境中**重放前缀的 bash 命令**提取中间补丁，再用官方终端评估器计算 fail-to-pass 与 pass-to-pass 通过率。初始值为 (0,1)，仅在有效编辑时更新；非编辑步骤或畸形补丁沿用上次有效分数；若 agent 还原工作树则重置为 (0,1)；
3. **Bradley–Terry 偏好对**：将成功与失败前缀配对，在每个 step-fraction 十分位桶内匹配，以控制实例难度与轨迹进度，分离出实例不变的排序信号。

**架构**：
- 冻结的 LM 主干（Qwen3-0.6B）+ LoRA（作用于 attention 与 MLP 投影）；
- 在池化的最后 token 隐藏状态上挂三个 fp32 线性头：value head s（预测最终解决）、F2P head、P2P head；
- 仅训练 LoRA 适配器与三个头。

**训练目标**（Eq. 1）：

```
L = BCE(s, y_final) + λ_f·BCE(â_f, g_f2p) + λ_p·BCE(â_p, g_p2p) + λ_r·[−log σ(s+ − s−)]
```

经验设定 λ_p = 0（F2P 信号有稳定提升，P2P 无）；部署配置 λ_f = 0.5, λ_r = 0.25。

**校准与部署策略**：
- **Platt scaling** 将原始 value logit 校准为概率；
- **meta score**：p_fail = σ(wᵀ[v, ĝ_f, φ] + b)，逻辑回归融合校准值 v、F2P 进度估计 ĝ_f 与步数比例 φ（优于纯 value 阈值，能有效惩罚低进度、后期状态）；
- **门控**：最小步数比例 φ ≥ φ_min（实验显示 φ_min = 20 步最佳），完全跳过早期步的监控推理；
- **警报规则**：累积投票达阈值 M（count-M）或连续 K 次投票（sustained-K）触发中止；
- 通过预算约束网格搜索选择工作点（在目标 FPR 下最大化召回）；
- 监控器完全 out-of-fold 训练；仅校准与阈值选择在评估折预测上拟合；嵌套实例级 cross-fit 产生几乎相同的工作点（排除测试标签过拟合）。

### 3.2 RestartSmart 重启策略

**设计理念**：当监控器在 t_f 步中止运行时，以全新预算启动**同策略**新 rollout——不携带先前 prompt 历史，但温启动以访问被中止仓库的编辑。作者发现文本摘要会诱发严重的 LLM **锚定效应（anchoring effects）**，使重启 agent 过早锁定先前错误推理路径，因此改为保留**物理代码编辑**作为干净的环境级 overlay。

**编辑提取**：
- 重放被中止轨迹的 bash 命令，每条命令后捕获 `git diff HEAD`，diff 变化时记录快照；
- 仅限任务基线时已存在的文件（排除 agent 新建文件）；
- **关键**：不在 t_f 处精确切割——定位 t_f 处或之后的第一个编辑（若无则回退到最近先前的编辑），并延伸切割点直至连续 **g 步**无进一步编辑，确保 overlay Δ 是"已安定"的完整补丁而非编辑中片段。

**Overlay 携带机制（offered, not forced）**：
- Δ 作为**可移除的、git-apply 支持的工具**挂载，而非拼入 prompt；
- agent 可执行 `overlay diff`（查看）、`overlay on`（应用）、`overlay off`（还原）、`overlay status`（查询）；
- overlay 初始为 **off**，prompt 将其框架为来自被停尝试的**未验证（unverified）线索**（附录 C 给出逐字 prompt：明确警告 "those changes are NOT applied — the source is pristine"、"do NOT assume they are right"，要求 agent 先复现 issue 再验证继承编辑）；
- 重启在实例预备的 test-bed 环境中执行；**不注入**自动生成的复现脚本（避免有缺陷的复现脚本成为 agent 过度信任的假 oracle）。

## 实验设置

### 数据集
- **SWE-bench Verified**：500 个实例，每个实例 11 次独立运行（seeds 0–10）；
- 划分：**350 训练 / 50 验证 / 100 测试**（instance-grouped、seeded、冻结为 JSON，同一实例的全部 11 次运行不跨集合泄漏）；
- 每条轨迹最多 **100 步**（遵循 SWE-bench Leaderboard）。

### 模型

| 角色 | 模型 | 原生 SWE-bench Verified 解决率 |
|---|---|---|
| 监控器主干 | Qwen3-0.6B（含上下文仅约 2GB VRAM） | — |
| 策略模型（开放权重） | Qwen3.5-9B | 48% |
| 策略模型（开放权重） | Qwen3.6-27B | 66% |
| 策略模型（开放权重） | Gemma4-31B | 62% |
| 策略模型（闭源 API） | Gemini3-Flash | 67% |

- FailFast 独立评估：**单个监控器仅在 Qwen3.6-27B 轨迹上训练**，零权重更新迁移至全部四个策略；
- RestartSmart 评估：使用**策略专属监控器**（在各模型自身轨迹上训练）。

### 基线
1. **AgentStop**（Pham et al. 2026）：将 token log-probabilities、token 计数、步骤重复输入梯度提升树；原设计在预定检查点做一次决策，作者从官方代码复现并扩展为逐步设置（显著提升其召回）；
2. **Duration 控制**：无监督基线，仅用步数打分，同样校准和阈值化（严格上界无信号停止）；
3. **SWE-PRM**（Gandhi et al. 2025）：推理时 Process Reward Model，通过分类法引导的 prompt 反馈纠错，使用其最强配置 Claude-Sonnet-4；
4. **Cold restart**：被 FailFast 中止的任务不带任何先前上下文重新运行。

### 训练超参数
- LoRA r=16，α=32；峰值 LR 1e-4，cosine 调度 1500 步；weight decay 0.01；LoRA dropout 0.05；
- 中止规则搜索空间：步数比例下限 {0.20,…,0.50}、分数阈值 0.30–0.96（步长 0.02）、M∈[1,12]、K∈[1,8]、校准正则 C∈{0.1,0.3,1,3}；
- 所有报告指标为 5 次独立运行平均；100 个留出实例 × 5 次 = 500 条轨迹。

## 主要结果

### FailFast 早停性能（Table 1，开放权重模型）

**Qwen3.6-27B（原生策略）**：

| FPR | 方法 | Recall | Prec. | Fired | Saved |
|---|---|---|---|---|---|
| 5% | **FailFast** | **30.5** | **76.1** | **13.4** | **20.4** |
| | AgentStop | 21.0 | 68.6 | 10.2 | 12.5 |
| | Duration | 19.8 | 67.3 | 9.8 | 11.4 |
| 25% | **FailFast** | **68.3** | **58.8** | **38.8** | **49.0** |
| | AgentStop | 54.5 | 52.3 | 34.8 | 37.2 |
| | Duration | 50.9 | 51.8 | 32.8 | 33.7 |

**Qwen3.5-9B**：5% FPR 时三家召回相近（34.7/35.1/34.7，弱策略的极长失败可被步数规则捕获），但 FailFast 节省 **15.7%** tokens vs. AgentStop 10.2%、Duration 10.9%；25% FPR 时 FailFast Recall 72.2、Saved **50.1%**。

**Gemma4-31B（跨家族迁移）**：5% FPR 时 Recall 27.2、Saved 14.6%（vs. AgentStop 23.8/10.3）；25% FPR 时 Recall 57.4 vs. AgentStop 51.5。

**核心数字**：5% FPR 下，单个 27B 训练的监控器在三个策略上稳定节省 **14.6%–20.4%** 总 token 成本；25% FPR 时在 Qwen3.6-27B 上拦截 68.3% 失败运行并回收 **49.0%** tokens，为重启提供算力缓冲。

### 闭源 API 迁移（Table 2，Gemini 3 Flash）

AgentStop 结构性不适用（专有 API 不暴露内部 token logprobs）。

| FPR | FailFast Recall | Prec. | Fired | Saved | Duration Saved |
|---|---|---|---|---|---|
| 5% | **22.7** | 69.8 | 10.6 | **16.0** | 12.0 |
| 25% | **44.8** | 46.8 | 31.2 | **37.1** | 30.2 |

召回略有下降，但 token 节省仍居首位。

### 能耗与碳排放估算
- 按 2N FLOPs/token 近似，0.6B 监控器每 token 比 9–31B 策略便宜 **15×–52×**；
- 监控器仅增加 **0.1%–0.6%** 估计计算开销；
- 扣除监控开销后，净推理能耗/运营碳节省约 **14.5%–20.3%**（三个开放权重策略）。

### RestartSmart 重启性能（Table 3）

| 策略（原生解决率） | FPR | 方法 | FP lost | TP rec. | Resolve rate | Token overhead |
|---|---|---|---|---|---|---|
| Qwen3.5-9B (48.2) | — | SWE-PRM | — | — | 52.4 (+4.2) | $0.15/t. |
| | 10% | **RestartSmart** | 45.8 | 21.7 | **52.2 (+4.0)** | +36.9% |
| | 10% | Cold Restart | 45.8 | 16.8 | 50.8 (+2.6) | +12.1% |
| | 25% | RestartSmart | 25.0 | 19.0 | 52.0 (+3.8) | +39.2% |
| Qwen3.6-27B (66.6) | — | SWE-PRM | — | — | 63.4 (**−3.2**) | $0.12/t. |
| | 10% | RestartSmart | 15.2 | 27.6 | 69.8 (+3.2) | +30.3% |
| | 25% | **RestartSmart** | **8.8** | **28.9** | **71.8 (+5.2)** | +43.8% |
| | 25% | Cold Restart | 27.5 | 20.2 | 66.8 (+0.2) | +18.2% |
| Gemma4-31B (62.2) | — | SWE-PRM | — | — | 63.0 (+0.8) | $0.12/t. |
| | 10% | RestartSmart | 12.9 | 24.4 | 65.2 (+3.0) | +20.5% |
| | 25% | RestartSmart | 9.1 | 17.5 | 65.2 (+3.0) | +36.9% |

**关键发现**：
1. **轨迹上下文的必要性**：RestartSmart 显著优于 cold restart（Qwen3.6-27B 在 25% FPR 下 +5.2% vs. +0.2%）；cold restart 会破坏多达 36.4% 的误报运行（27B、10% FPR），而 RestartSmart 仅损失 15.2%（25% FPR 下 8.8% vs. 27.5%）；
2. **更强模型更能利用 overlay**：Qwen3.6-27B 能批判性评估先前修改、保留可行编辑并将误报运行恢复至完成；
3. **SWE-PRM 的反转效应**：对弱模型 Qwen3.5-9B 有效（+4.2%），但对强模型 Qwen3.6-27B 反而**降低** 3.2%——侵入式 prompt 干预会扰乱强模型的连贯推理；且 SWE-PRM 无 FPR 预算控制机制；
4. **可控的算力-性能权衡**：27B 在 10% FPR 为高性价比点（+3.2% 解决率换 +30.3% 算力）；9B 的甜点位在 10% FPR（峰值 +4.0%，+36.9%）；总开销远低于一次完整额外 rollout 及 test-time scaling 预先承诺的采样预算。

## 消融实验

### 监控器主干容量（Table 4）
Qwen3-0.6B vs. Qwen3-4B：扩参**无意义提升**，0.6B 相当甚至更优（如 27B、10% FPR：45.5 vs. 41.3）——前缀级失败信号很早就饱和，与标量奖励建模的扩展瓶颈一致。

### 输入特征（Table 5）
- **+S**（加入全程结构化摘要）：无收益且多数情况有害（27B、15% FPR：58.1 → 55.1）——LM 监控器可直接分析原始执行文本时，粗粒度启发式摘要是干扰项；
- **−W**（去掉最近执行历史）：召回下降最剧烈（27B、15% FPR：58.1 → 44.9，**−13.2**）——FailFast 的决策紧密依赖动态轨迹行为；
- **−I**（去掉问题描述）：同样退化，说明任务意图提供必要的基线上下文。

### 编辑完成耐心度（Table 6，RestartSmart）

| FPR | 策略 | FP lost | TP rec. | Resolve |
|---|---|---|---|---|
| 25% | Immediate | 18.8 | 22.8 | 68.8 (+2.2) |
| 25% | **Wait (Ours)** | **8.8** | **28.9** | **71.8 (+5.2)** |

立即切割产生碎片化 overlay，错误破坏本可成功的运行；默认等待 5 步无编辑后再切割。

### 附录补充消融
- **观察窗口大小 k**（Table 9）：k∈{2,4,8,16}；k=2 上下文不足；k=8 与 k=16 在多数预算下处于评估噪声内，故选 **k=8**（窗口翻倍则监控输入成本翻倍而无可靠收益）；
- **进度目标头**（Table 10）：V Only vs. V+F2P vs. V+F2P+P2P；F2P 密集监督提供正向表示对齐（5% FPR 下 **+6.2%**），P2P 无额外收益，验证 *V+F2P* 为最优目标；
- **训练种子数 N**（Table 11）：N∈{3,6,11}；N=3 与 N=6 相近，**N=11** 在中等预算下带来最高约 10 点提升（暴露更丰富的失败模式多样性），轨迹收集为一次性前置成本；
- **策略专属监控器**（Table 12）：如 Qwen3.5-9B 在 5% FPR 时 Recall 39.8、Saved 27.4；Gemma4-31B 在 25% FPR 时 Recall 66.7、Saved 54.2。

### 统计显著性（附录 A）
- **Table 7**：McNemar 配对检验，FailFast 在 30 项比较中 25 项显著更优、从未显著更差；
- **Table 8**：RestartSmart 在每个策略与预算下收益均显著（p ≤ 9.4×10⁻³，最强 4.2×10⁻⁵）；对 cold restart 在 27B 上及三策略汇总显著；SWE-PRM 在任何策略上均无 α=0.05 显著的改进（9B p=0.050、27B p=0.060 净下降、Gemma4 p=0.73）。

## 图表说明

该论文**无插图（figures）**，全部实验证据以 12 个表格呈现：主文 Table 1–6（早停性能、闭源迁移、重启性能、主干规模、输入消融、编辑耐心度），附录 Table 7–12（统计检验、窗口大小、进度目标、种子数、策略专属监控器）。

## 局限性

1. **评估范围有限**：仅在 SWE-bench Verified + mini-swe-agent 上评估，结论可能无法推广到其他软件工程任务或 agent 框架；
2. **跨策略泛化**虽由多规模、多家族（含闭源 API）模型的一致结果支持，但闭源模型上检测召回有所下降（Gemini 3 Flash 5% FPR 时 22.7% vs. 开放权重 27.2%–30.5%）；
3. 5% FPR 工作点拦截轨迹过少，无法有效服务重启，重启需 10%/25% 的更宽松预算（以更多误报为代价）；
4. RestartSmart 需等待中止信号后的编辑安定，token 占用高于 cold restart。

## 结论与未来工作

**结论**：FailFast–RestartSmart 通过轻量级 0.6B 监控器提前终止疑似失败轨迹（5% FPR、即 ×0.95 基础性能下节省高达 20.4% 执行 tokens），并提取部分代码编辑 overlay 引导同模型重试，以 20.5%–43.8% 的适度 token 开销提升解决率（最高 **+5.2%**），无需昂贵的过采样。

**未来工作方向**：
1. 训练更高精度的监控器以获得更大的 token 节省；
2. 提取更广的非锚定（non-anchoring）工件，在保护被误分类运行成功率的同时，将更多失败轨迹转化为成功。

## 个人点评

- **问题切中要害**：agentic 长轨迹的成本控制是当前 LLM agent 落地的核心痛点，本文把"早停"与"重启恢复"统一到一个预算化框架中，FPR 预算的设定非常工程友好。
- **设计细节扎实**：dense F2P/P2P 重放监督、Bradley-Terry 偏好对、meta score 融合、count-M/sustained-K 警报规则、overlay "offered not forced" 机制、编辑安定等待——每个环节都有消融支撑。
- **意外发现有价值**：0.6B 监控器即可饱和（容量扩展无效）、SWE-PRM 对强模型反而有害、文本摘要诱发锚定效应——这些结论对后续 agent 监控研究有直接借鉴意义。
- **落地成本低**：监控器仅 2GB VRAM、0.1%–0.6% 计算开销，6 小时内可在 2 张 GPU 上训完，且能跨模型家族、跨闭源 API 迁移，实用性强。
