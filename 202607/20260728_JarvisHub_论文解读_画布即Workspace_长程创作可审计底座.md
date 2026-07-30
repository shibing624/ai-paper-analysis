# 把画布当 Workspace：JarvisHub 给长程多模态创作搭了一张可审计的底座

> 论文：JarvisHub: An Open Harness for Canvas-Native Multimodal Creative Agents
> arXiv: [2607.23588](https://arxiv.org/abs/2607.23588)
> 团队：JarvisX Team（共 26 位作者，核心贡献者 Yunlong Lin、Zixu Lin、Zhaohu Xing，学术顾问 Tianyu Pang、Xiangyu Yue）
> 项目页：https://www.jarvishub.site/ ｜ 代码：https://github.com/LYL1015/JarvisHub

---

## 核心摘要

如果你跟现在的 AI 创作 agent 协作过，大概率撞过这种墙：让它"拍一个赛博牛仔 + 机器人 + 拾荒的短剧"，前 3 个镜头很惊艳，到第 5 个角色就突然换脸，第 8 个镜头音频对不上嘴型，最后你想回头改第 2 个镜头的配色，发现所有上下文都淹没在聊天记录里，工具调用轨迹、失败重试、用户反馈全部丢失。

**这篇论文的判断很直接：长程多模态创作的痛点不是"模型不够强"，而是"没有合适的工作台"。** 现有 prompt-to-output 工具只产出最终资产，把中间过程丢光；聊天式 agent 把所有上下文塞进线性对话；节点式 workflow 又把流水线写死、用户和 agent 都改不动。JarvisHub 的解法是把**画布同时当 UI、当 memory、当 action space、当项目状态**——所有多模态资产、依赖、版本、反馈都以 typed canvas nodes 和 links 的形式存在，agent 通过一个"协议桥"在画布上读写，每次动作都留下可追溯的轨迹。

实验跑了 3 个典型长程任务（短剧、摄影网站、PPT 演示），覆盖数十步的规划-生成-反馈循环。它不是来刷榜的，是来**给这个领域定一套可被研究、被复现、被训练的开源底座**。

---

## 一、问题：为什么需要一个 "Canvas-Native" 的工作台？

作者在 Related Work 和 Introduction 里把现有三类系统挨个拎出来打，我觉得打得挺准的：

| 现有范式 | 强项 | 短板 |
|---|---|---|
| Prompt-to-Output 工具 | 单步生成质量高 | 中间决策、失败尝试、备选方案、修订历史全部丢弃 |
| Chat-based Creative Agent | 能多步调用工具 | 上下文是线性对话，表达不了空间布局、资产关系、版本分支 |
| Node-based Workflow | 执行步骤可见 | 流水线被手动定死，agent 没法 inspect / revise / extend |

**这三种系统都是短程思维**——完成单个任务就行。但现实创作是长程的：你要收集参考、规划布局、生成多个候选、对比、回退、修订、合并、接受反馈。中间产物本身就是项目的**状态**，不是噪声。

商业产品（Claude Design、Google Stitch、TapNow、LibTV 等）已经在往这个方向走，但都是闭源的——研究者能看到用户界面，但看不到**项目状态怎么表示、动作怎么校验、工具怎么调度、反馈怎么用、失败怎么修复**。这就让"长程创作 agent"这个领域没法做严肃的实验。

> 这是一个我之前没仔细想过的角度：开源 paper 越来越多，**但开源 harness 几乎没有**。结果就是大家都在 prompt 层面刷 SOTA，没人能在"agent 怎么管理一个长程项目"这件事上做对照实验。JarvisHub 想堵的就是这个缺口。

---

## 二、方法：三层架构 + 协议桥

JarvisHub 的设计分三块：**Canvas State、Protocol Bridge、Agent Runtime**。我先放总图，再逐层拆。

![Figure 2：JarvisHub 核心架构](https://arxiv.org/html/2607.23588v1/x2.png)

*图 1：JarvisHub 核心架构。Canvas state 存多模态产物、依赖、版本和用户选择；Protocol bridge 校验授权动作、控 canvas 读写；Agent runtime 负责规划、调用工具和技能、更新画布。底下的 trajectory 记录、评估器、人工编辑、checkpoint、长程数据，反过来支撑反馈、修复和后续分析。*

下面这张图把"画布到底怎么被改"讲清楚了——8 步循环，从用户请求到回写画布。

![Agent runtime 协议循环](https://arxiv.org/html/2607.23588v1/figs/agent-loop.png)

*图 2：Agent runtime 的执行循环。runtime 观察画布（observe canvas）→ 在当前 manifest 和 grant 下选动作（select action）→ 调用能力（invoke capability）→ 把观察值通过 protocol bridge 校验并 commit 到共享画布（return observation）→ 用户可以在画布上 review / steer。每一步都会写进 trajectory log。*

### 2.1 画布状态（Canvas State）：一切皆节点

论文把一个创作项目在第 $t$ 步的状态形式化成一个五元组：

$$\mathcal{C}_{t}=(\mathcal{G}_{t},\mathbf{X}_{t},\mathbf{M}_{t},\mathbf{U}_{t},\mathbf{L}_{t}),\quad\mathcal{G}_{t}=(\mathcal{V}_{t},\mathcal{E}_{t}).$$

简单说，$\mathcal{G}_t$ 是一个**有类型的产物图**，$\mathcal{V}_t$ 是 canvas 节点集，$\mathcal{E}_t \subseteq \mathcal{V}_t \times \mathcal{R} \times \mathcal{V}_t$ 是带类型的有向边——关系 $\mathcal{R}$ 覆盖了 reference use、version lineage、generation dependency、grouping、workflow continuation。剩下四个量分工也很干净：

- $\mathbf{X}_t$：节点的可编辑内容和产物句柄
- $\mathbf{M}_t$：provenance、runtime status、metadata
- $\mathbf{U}_t$：用户选择、编辑、反馈
- $\mathbf{L}_t$：空间位置、分组布局

每个节点进一步展开成：

$$v_{i}=(\mathrm{id}_{i},k_{i},\mathbf{p}_{i},\mathbf{x}_{i},\mathbf{y}_{i},\mathbf{m}_{i},s_{i}),\quad k_{i}\in\mathcal{K}_{\mathrm{node}}$$

作者强调这种表示的三个关键属性，我觉得很到位：

1. **可寻址（addressable）**——agent 可以引用"第 2 个候选图"这种具体节点，而不是在对话里描述"就是那个稍微蓝一点的图"。
2. **可复用（reusable）**——参考、草案、被拒绝的候选、半成品，下一步还能被引用。
3. **可审查（inspectable）**——用户和 agent 都能追溯"这个结果用了哪些材料、哪个版本被选中、哪些下游产物依赖它"。

**这其实是把 Git 的 working tree 思想搬到了多模态创作领域**——每个节点都有 id、status（planned / running / failed / selected），关系是 typed edge，commit 是带 observation 的写回。做过设计协作工具的人会立刻 get 到这种结构的价值。

### 2.2 协议桥（Protocol Bridge）：让每次画布修改都可审计

画布光"能改"还不够，长程创作要求**改得可恢复**。协议桥做了三件事：

1. **Capability manifest $\Gamma_t$**：列出当前项目可用的节点类型、修改操作、工具、产物句柄
2. **Execution grant $\Omega_t$**：从 manifest 派生出"这一轮允许 agent 干什么"
3. **Mutation validation**：任何 agent 提议的动作 $a_t$，必须能被编码成 checked tool call、canvas mutation、evaluation request、clarification request 或 user-facing response，否则就不执行

一次合法的状态转移可以写成：

$$\mathcal{C}_{t+1}=\mathcal{F}(\mathcal{C}_{t},a_{t},o_{t},f_{t},r_{t})$$

$a_t$ 是 action，$o_t$ 是 observation（工具返回值），$f_t$ 是 feedback（用户 / critic / 评估器），$r_t$ 是由 feedback 诱导的 repair 或 follow-up 决策。

> **坦白说**这个设计让我有点感慨。我自己做过几次 AI 创作 agent 的工程化，最痛的就是"agent 改了画布上某个东西但你不知道改了哪里、改的依据是什么"——最后只能靠人工 review 一遍所有节点。**协议桥的 grant 机制相当于在 action 层加了 RBAC**，这件事在大规模 agent 系统里早晚要做。

### 2.3 Agent Runtime：5 个工具族 + 3 个高层支撑

runtime 的核心契约是：

$$\mathcal{A}_{t}=\mathcal{A}(\Omega_{t},\Gamma_{t},\mathcal{C}_{t},q_{t}),\quad a_{t}\in\mathcal{A}_{t}.$$

每一个被接受的 action 必须同时满足四件事：基于观察到的画布、被 manifest 暴露、被当前 grant 允许、通过协议桥 commit。**这意味着 runtime 没有自己的隐藏状态**——所有上下文都锚在画布上。

工具族（Table 2）：

| 工具族 | 运行时角色 | 代表操作 |
|---|---|---|
| Canvas tools | 更新项目状态 | 读、建、改、连接、分组、选、分支；维护产物句柄和运行时状态 |
| Generation tools | 产出画布产物 | 图、视频、音频、复合媒体生成 |
| Native tools | 外部执行 | 浏览器、文件、代码、搜索、文档、演示文稿 |
| Recovery tools | 检查与修复 | 结构化反馈、校验、checkpoint、局部修复 |
| MCP tools | 扩展外部服务 | 同 manifest-grant 契约下的 MCP 能力 |

高层支撑（3 个）：

- **Skills**：把"分镜、参考引导生成、design-to-web 重建、视频 prompt 模板、deck 构建"这类**可复用创作流程**编码成结构，runtime 调用即可，不用每步都重新规划。
- **Memory**：跨 turn 保留用户偏好、之前的决策、过程性知识，避免对话被截断后上下文丢失。
- **Subagents**：当一个工作流能分支时（比如要并行探索多组镜头），子 agent 独立执行子任务，父 agent 选优并合回画布图。

**这套"工具族 + 高层支撑"的组合，让 runtime 在"调一个模型"和"做一个长程项目"之间有了清晰的中间层**。这其实和 LangGraph、CrewAI 这类框架的思路有重叠，但 JarvisHub 把状态层的"画布"做成了 first-class citizen——其它框架的 state 还在 message list 或 dict 里。

### 2.4 反馈与可追溯（Trajectory）

整个轨迹被定义为：

$$\tau=\{(q_{t},\mathcal{C}_{t},\Gamma_{t},\Omega_{t},a_{t},o_{t},f_{t},r_{t},\mathcal{C}_{t+1})\}_{t=1}^{T}$$

每一轮都把这 9 个量原样记录。这意味着：分析 agent 行为时，**你不需要靠日志去还原它当时看到的状态**——画布状态本身就被 snapshot 下来了。**这给长程 agent 训练留了一条活路**——trajectory 就是天然的训练数据，但作者也老实承认 raw trajectories 还需要质量过滤、consent、匿名化和版权审查。

---

## 三、三个长程创作任务

实验是**定性展示**而非 benchmark——作者在 Limitations 里明说了。但说实话，**对 harness 这类工作，定性反而比定量更诚实**：你没法用一个数字说"画布比对话好"。

| 任务 | 考察的能力 | 典型产物 |
|---|---|---|
| Narrative media generation | 叙事规划、身份保持、风格一致、跨镜头连贯 | 角色参考、场景设计、分镜面板、镜头规划、图像序列、视频片段、animatic |
| Interactive web development | 布局设计、交互逻辑、前端代码、预览与迭代 | 静态页、动态站、landing、交互原型、渲染预览、前端实现 |
| Presentation deck generation | 内容选择、叙事组织、版式、视觉综合、跨页一致 | 演示稿、学术报告、pitch deck、视觉摘要、说明图 |

模型后端：GPT-5.5 主 agent + GPT Image 2 图像 + Seedance 2.0 视频 + Gemini 3.1 Pro 多模态评估。下面三个案例每个我都贴工作区截图和最终交付物，方便你感受"画布上的长程协作到底是什么样子"。

### 案例 1：短剧《Cowboy Robot Zombie Scavenger》

**任务**："Generate a short drama about a 'cowboy robot zombie scavenger.'"

![工作区：画布上的分镜规划与依赖](https://arxiv.org/html/2607.23588v1/x3.png)

*图 3：短剧任务的工作区 trace。画布上能看到任务简报（左）、规划笔记、视觉参考、镜头候选、依赖连接线（中间密集的黑色折线就是节点之间的 dependency 边），以及右侧的任务进度面板和子任务 Todo。**注意左下方的 chat 区域其实在给 agent 喂具体的分镜参数**——比如 `video_clip_01_pool_aftermath` 的 taskid、cgt-xxxxx 标识、状态。*

![最终产物：跨镜头一致的多模态资产](https://arxiv.org/html/2607.23588v1/x4.png)

*图 4：短剧任务最终产物。9 张关键帧展示了机器人角色在海滩别墅场景中的连续动作——从发现金币、海水喷涌、拾荒勘察、与鸵鸟对峙，到头戴礼帽的招牌形象。**跨镜头的角色一致性和场景连续性肉眼可见**。作者提到这个 case 灵感来自 Mx-Shell 的原作《Zombie Sweeper》。*

### 案例 2：Awwwards 风格的摄影网站

**任务**："Create a personal photography website with a light, Awwwards-inspired design and rich animations."

![工作区：网页规划的画布状态](https://arxiv.org/html/2607.23588v1/x5.png)

*图 5：摄影网站任务的工作区。画布上分散放置了多个网页版本（黑色大字标题风格、淡雅留白风格、人物肖像风），右侧 chat 面板里能看到用户的中文 prompt + agent 把它拆成 Todo 后逐步推进（`canvas_create_webhero_node`、`canvas_web_style_reference_search` 等子任务）。*

![最终产物：4 个分页的摄影网站](https://arxiv.org/html/2607.23588v1/x6.png)

*图 6：摄影网站的最终交付。可以看到字体选择（粗体大字号）、留白、灰度滤镜的人物肖像、几何分割、底部项目卡片、黑色文字+黄色强调色的按钮——**整体风格高度统一**，而且人物形象在 4 个分页之间是同一套摄影主题。*

### 案例 3：决策树课件（Stanford Lecture 风）

**任务**："Create a PowerPoint presentation on decision trees for machine learning, styled like a Stanford lecture."

![工作区：PPT 课件的画布与进度](https://arxiv.org/html/2607.23588v1/x7.png)

*图 7：PPT 任务的工作区。画布上摆放了 10 张幻灯片的缩略图，**每张都是独立节点**，中间有黑色折线连接依赖关系。右侧 chat 面板显示已完成 10/10，包括 PPT deck 生成、Stanford-inspired 视觉风格、red accents、卡片风格布局、手绘 SVG 图嵌入、self-rendering check。*

![最终产物：决策树课件页面](https://arxiv.org/html/2607.23588v1/x8.png)

*图 8：PPT 最终交付。可以看到 Stanford 风格的极简学术风——Decision Trees / Learning Setup / Regression Trees / Overfitting and Tree Complexity 四张代表页，**重点用红色强调，配图统一用线条 + 数据点**。*

---

## 四、这张底座到底解决了什么、没解决什么？

### 亮点

1. **把"画布"提升为 first-class state**，不是 UI 装饰。agent 的 memory / action space / project state 全在画布上，这跟 Claude Artifacts、ChatGPT Canvas、ComfyUI 是**结构性差异**——后两者画布只是输出渲染区，不是 agent 的状态层。
2. **协议桥的 grant 机制**让"agent 改了什么、改的依据是什么"完全可审计。**对生产环境的 agent 系统来说，这等于内置了 audit log + RBAC**。
3. **trajectory 是一次性带走的**——每次跑都留下 $(q, C, \Gamma, \Omega, a, o, f, r, C')$ 完整元组。**这是给未来 creative agent 训练留的弹药**。
4. **5 个工具族 + 3 个高层支撑**的拆分在工程上很自然，Skills/Memory/Subagents 三件套跟 LangGraph 那一系思路兼容，迁移成本低。

### 局限（作者自己也写明了）

1. **没有量化 benchmark**——三个 case 是定性展示。**对 harness 工作来说这其实合理**，但你确实没法拿一个分数证明"它比 ComfyUI-flow 好"。
2. **最终质量受外部模型天花板制约**——JarvisHub 只管 orchestration 和 state management，生成质量靠 GPT-5.5、Seedance 2.0 等。**这是个 harness 的常见尴尬：上限看上游，锅还得自己背**。
3. **协议桥保证动作可恢复，但保证不了语义正确**——agent 可能在语法上合法地"改坏了"某个节点。
4. **trajectory 数据使用需要质量过滤、consent、匿名化、版权审查**——作者在 Limitations 里专门强调了这条，**这个提醒在 open data 时代越来越重要**。

### 跟商业系统比，位置在哪？

我的判断是：

- **vs Claude Design / Google Stitch**——它们闭源，JarvisHub 开源。**研究的可重复性上 JarvisHub 完胜**。但 UI 体验和产品打磨上，商业产品肯定领先。
- **vs ComfyUI / ComfyBench**——节点式工作流工具，已经有成熟生态。JarvisHub 的差异在"agent 自主调度 + canvas as state"，ComfyUI 更偏"用户拖节点"。
- **vs LangGraph / CrewAI / AutoGen**——这些是通用 agent 框架。JarvisHub 是**为多模态创作场景定制的垂直 harness**，state 层有 typed artifact graph 和 canvas 这种针对性抽象。**通用框架灵活，但创作场景下"画布作为 state"这个抽象比 dict 强太多**。

---

## 五、对工程实践的启发

最后说点我觉得真的值得借鉴的工程点：

1. **画布是状态的真正归宿**。如果你在做多模态 agent 工具，认真考虑一下把"画布 / 工作区"从 UI 提升为 state layer。**这不只是工程重构，是设计哲学的转变**。
2. **协议桥 + grant 机制值得抄**。给 agent 操作加权限范围 + audit log + validation，是任何 production-grade agent 系统早晚要补的课。**别等出事再加**。
3. **trajectory 元组设计很工程友好**。$(q, C, \Gamma, \Omega, a, o, f, r, C')$ 这种结构既能让 evaluator 跑过程性指标，又能在用户回放时清晰展示"当时 agent 看到了什么、做了什么、为什么"。**对调试长程 agent 来说，这比单纯保存 message log 强一个量级**。
4. **Skills / Memory / Subagents 三件套是必经之路**。无论你用什么框架，最终你都会发现需要"可复用流程 + 跨 turn 记忆 + 子任务并行"这三样东西。**早点在数据模型里给它们留位置，后期不会推倒重来**。

---

## 结语

JarvisHub 不是一篇"刷点论文"，它做了一件**更基础的事**——给"长程多模态创作 agent"这个新兴领域搭了一张可被研究、被复现、被训练的开源底座。短剧、网站、PPT 三个 case 展示的是"画布即工作台"这个抽象能撑住多复杂的真实任务。**对研究者来说，trajectory 数据的可获得性可能比模型本身更有长期价值**。

接下来值得关注的是：(1) JarvisHub 团队是否会在轨迹数据上放出公开 benchmark；(2) 是否有团队把 protocol bridge 的 grant 抽象扩展到通用 agent 框架；(3) canvas-as-state 范式是否会反过来影响 IDE 类的工具（Cursor、Claude Code 这一系）。

> arXiv: [2607.23588](https://arxiv.org/abs/2607.23588) ｜ 项目页：https://www.jarvishub.site/ ｜ 代码：https://github.com/LYL1015/JarvisHub

---

*觉得有启发的话，欢迎点赞、在看、转发。跟进最新 AI 前沿，关注我。*
