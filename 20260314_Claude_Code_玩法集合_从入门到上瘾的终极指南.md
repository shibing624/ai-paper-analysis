# Claude Code 玩法大全：从入门到上瘾的终极指南

> 这不是又一篇安装教程。网上的"从零开始配置 Claude Code"已经够多了，我不打算再写第 101 篇。这篇文章要解决的问题是：**装好之后，怎么把它用到飞起？**

你可能已经用 Claude Code 写过几段代码、修过几个 Bug。但我敢打赌，你大概率还停留在"高级搜索引擎"的阶段——问一句答一句，跟用 ChatGPT 没啥本质区别。

真正让 Claude Code 值回票价的，是把它从一个"问答机器人"调教成一个**24小时在线的数字员工**。怎么做？往下看。

---

## 🏗️ 第一招：CLAUDE.md —— 给你的 AI 写一份"员工手册"

每次开新会话，Claude Code 都像一个刚入职的实习生：啥也不知道，啥都要问。项目用什么框架？代码风格有什么要求？测试怎么跑？每次都要重新交代一遍，烦不烦？

**CLAUDE.md 就是解药。**

在项目根目录放一个 `CLAUDE.md` 文件，Claude 每次启动时会自动读取。你可以把它理解成写给 AI 的"员工手册"——项目约定、编码规范、常用命令、踩过的坑，全写进去。

```markdown
# CLAUDE.md

## 项目概述
这是一个基于 FastAPI + React 的全栈应用，后端 Python 3.12，前端 TypeScript。

## 编码规范
- 后端遵循 Google Python Style Guide
- 函数注释用英文，变量命名用 snake_case
- 所有 API 必须有 Pydantic model 做参数校验

## 常用命令
- 启动后端：`cd backend && uvicorn main:app --reload`
- 跑测试：`pytest tests/ -v`
- 数据库迁移：`alembic upgrade head`

## 已知坑点
- Redis 连接池在测试环境用 fakeredis，不要连真实 Redis
- 前端的 DatePicker 组件在 Safari 上有兼容问题，临时方案是用 native input
```

Boris Cherny（Claude Code 的创造者）在内部分享中提到一个操作：**每次 Claude 犯了错，就让它自己把教训写进 CLAUDE.md。** 随着项目迭代，这个文件越来越厚，Claude 的犯错率肉眼可见地下降。就像培养一个老员工——你不需要每天叮嘱"记得加分号"，因为他自己记住了。

**进阶技巧：多级 CLAUDE.md**

CLAUDE.md 不只能放在项目根目录。你可以在子目录、甚至用户主目录（`~/.claude/CLAUDE.md`）放不同层级的配置：

| 位置 | 作用 | 示例内容 |
|:-----|:-----|:---------|
| `~/.claude/CLAUDE.md` | 全局个人偏好 | "我习惯用 vim 键位"、"回复用中文" |
| 项目根目录 `/CLAUDE.md` | 项目级规范 | 框架选型、API 设计规范、测试策略 |
| 子目录 `/src/auth/CLAUDE.md` | 模块级上下文 | "这个模块负责 OAuth2 登录，依赖 jwt 库" |

Claude 会自动合并所有层级的 CLAUDE.md。模块级的上下文越精确，生成代码的质量越高。

---

## ⚡ 第二招：平行宇宙大法 —— 一个人当五个人用

这是 Boris 团队分享的**最炸裂**的效率技巧。

核心思路：用 `git worktree` 同时开 3-5 个工作目录，每个目录跑一个独立的 Claude 会话。就像游戏里开了五个分身，每个分身在不同的平行宇宙里帮你干活。

```bash
# 创建三个独立的工作目录
git worktree add ../project-refactor feature/refactor
git worktree add ../project-bugfix  fix/login-crash
git worktree add ../project-tests   feature/add-tests

# 分别在三个终端启动 Claude
cd ../project-refactor && claude  # 分身1：重构代码
cd ../project-bugfix && claude    # 分身2：修 Bug
cd ../project-tests && claude     # 分身3：补测试
```

三个任务齐头并进。你喝完一杯咖啡回来，三个分支都搞定了。这才是真正的"多线程人生"。

有人甚至设置了 shell 别名，按一个键就能在不同 worktree 之间瞬移：

```bash
alias za="cd ~/project-worktree-a && claude"
alias zb="cd ~/project-worktree-b && claude"
alias zc="cd ~/project-worktree-c && claude"
```

为什么这招这么猛？因为 Claude Code 的瓶颈从来不是算力——**瓶颈是你的注意力**。人类一次只能盯一个任务，但 AI 不受这个限制。把串行变并行，效率直接拉满。

---

## 🧠 第三招：Plan Mode —— 先画蓝图再动手

"先别急着写代码。"

任何复杂任务（超过 50 行代码的改动），都应该先进入 Plan Mode。用快捷键 `Shift+Tab` 切换，或者在提示词里写清楚：

```
请先制定详细的执行计划，列出所有要修改的文件和具体改动，
等我确认后再开始编码。
```

Plan Mode 的价值在于：**前期多花 10 分钟对齐需求，后期节省 1 小时返工。**

更狠的操作——开第二个 Claude 实例专门审查第一个 Claude 的计划。一个负责"做"，一个负责"挑刺"。对抗式思考，方案质量直线上升。

一位开发者分享的真实案例：接手一个 3000 行的遗留订单系统，需要加入预售功能。他没有让 Claude 直接动手重构，而是分了三步走：

1. **让 Claude 先分析**：识别出 6 个业务模块和依赖关系（15 分钟）
2. **制定渐进式方案**：只拆分预售相关的最小模块（10 分钟）
3. **逐步实施+验证**：每改一个模块跑一次测试（2 小时）

如果跳过前两步直接让 AI 重构，大概率半天时间全浪费在调试"连锁反应"的 Bug 上。

---

## 🔧 第四招：Skills —— 把重复劳动变成一键操作

每天都在做的事，为什么不封装成一个命令？

Skills 就是你给 Claude 定义的快捷指令。在 `~/.claude/skills/` 目录下创建 Markdown 文件，就能通过 `/` 命令触发。

**示例：一键技术债扫描**

创建文件 `~/.claude/skills/techdebt.md`：

```markdown
# /techdebt - 技术债扫描

请扫描当前项目，找出以下技术债：

1. **重复代码**：相似度超过 80% 的代码块
2. **过长函数**：超过 50 行的函数
3. **硬编码配置**：写在代码里的 URL、密钥、魔法数字
4. **缺失测试**：没有对应测试文件的核心模块
5. **过时依赖**：发布超过 2 年且有已知漏洞的包

输出格式：按严重程度排序的表格，包含文件位置、问题描述、建议修复方案。
```

以后在 Claude 里输入 `/techdebt`，一键完成扫描。一次编写，终身受益。

**更多 Skill 灵感：**

| Skill 名称 | 功能 | 触发命令 |
|:-----------|:-----|:---------|
| 代码审查 | 按团队规范审查最近一次 commit | `/review` |
| 日报生成 | 基于 git log 生成结构化日报 | `/daily` |
| API 文档 | 自动生成 OpenAPI 格式文档 | `/apidoc` |
| 性能分析 | 找出潜在的 N+1 查询和内存泄漏 | `/perf` |
| 依赖升级 | 检查可升级的依赖并评估影响 | `/upgrade` |

---

## 🔌 第五招：MCP —— 让 Claude 接上外部世界

MCP（Model Context Protocol，模型上下文协议）是 Claude Code 真正变成"全能助手"的关键。

如果说 Claude Code 本身只能读文件、写代码、跑命令，那 MCP 就是给它装上了手和脚——**连接数据库、调用 API、操作浏览器**，几乎无所不能。

**实战案例：用自然语言操作数据库**

安装一个 Database MCP Server，Claude 就能直接操作你的数据库：

```bash
# 安装数据库 MCP
claude mcp add database-server \
  --command npx \
  --args "-y" "@executeautomation/database-server" \
  --env DB_CONNECTION="postgresql://user:pass@localhost:5432/mydb"
```

然后你可以这样跟 Claude 对话：

```
你：帮我查一下过去7天注册用户数的变化趋势，按天统计
Claude：（自动生成 SQL，执行查询，返回结果表格和趋势分析）
```

不用写 SQL，Claude 帮你翻译、执行、解读。数据分析师看了直呼内行。

**常用 MCP 生态：**

| MCP 名称 | 功能 | 适用场景 |
|:---------|:-----|:---------|
| GitHub MCP | 操作 PR、Issue、Review | 代码协作自动化 |
| Filesystem MCP | 增强文件操作能力 | 批量文件处理 |
| Browser MCP | 浏览器自动化 | 网页测试、数据抓取 |
| Database MCP | 数据库查询与操作 | 数据分析、运维 |
| Slack/飞书 MCP | 消息通知和信息汇聚 | 团队协作 |
| Brave Search MCP | 联网搜索 | 实时信息获取 |

配置方式也很简单，在 `~/.claude/settings.json` 或项目级 `.claude/settings.json` 中声明：

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_TOKEN": "ghp_xxxx"
      }
    }
  }
}
```

**一个容易踩的坑**：别等 Claude "想起来"用 MCP 工具。很多时候你需要主动提醒它——"用 GitHub MCP 帮我创建一个 PR"——否则它可能用更笨的方式（比如调用 `gh` CLI）来完成同样的事。

---

## 🤖 第六招：SubAgent —— 用多个小助手互相配合

一个 Claude 不够用？那就多开几个。

SubAgent 是 Claude Code 的子代理机制：主 Claude 可以派出独立的子 Agent 去执行特定任务，子 Agent 有自己的上下文窗口，不会污染主会话。

```
你：这个 PR 的改动涉及三个模块，帮我做一次全面的代码审查。用 subagents 并行检查。
```

Claude 会自动拆分任务：
- SubAgent A → 检查 `auth/` 模块的安全性
- SubAgent B → 检查 `api/` 模块的接口一致性  
- SubAgent C → 检查 `tests/` 的覆盖率

三个子 Agent 并行工作，结果汇总到主 Agent。这是"分而治之"的经典模式，处理比单会话大得多的项目毫无压力。

**高级操作：对抗式 Code Review**

设置两个角色不同的 Agent——一个是"严格的安全审计师"，一个是"追求性能的优化师"。让它们同时审查同一段代码，从不同角度找问题。这种对抗式纠错，效果远超单一视角的审查。

---

## 📱 第七招：把 Claude 接入 IM —— 手机也能 Vibe Coding

Claude Code 只能在终端用？太亏了。

通过 Gateway 配置，你可以把 Claude Agent 接入微信、QQ、飞书、Slack 等 IM 工具。这意味着你**躺在沙发上用手机就能指挥 AI 干活**。

几个真实场景：

**场景 1：移动端 Vibe Coding**
在地铁上突然想到一个 feature，掏出手机在微信里发一句："在 user 模块加一个根据邮箱查用户的接口，要加参数校验和单元测试。" Claude 在服务器上默默执行，等你到公司打开电脑，PR 已经提好了。

**场景 2：定时任务管家**
配置定时 Skill，让 Claude 每天早上 8 点自动执行：
- 抓取行业新闻摘要
- 检查线上服务健康状态
- 汇总昨天的 Git 提交日报

结果推送到你的 IM，打开手机就能看。

**场景 3：Bug 自动修复**
把 Sentry 的报警接到 Claude，它自动分析错误堆栈、定位代码、生成修复 PR。你只需要在手机上点一下"Approve"。

---

## 🧬 第八招：Memory 机制 —— 让 AI 拥有"记忆力"

Claude Code 原生的会话是"无状态"的——关掉终端，它就忘了你们之前聊过什么。但通过 `memory.md` 机制，你可以给它装上持久化记忆。

具体做法：在项目目录下维护一个 `memory.md` 文件，记录重要的上下文信息：

```markdown
# Memory

## 项目决策记录
- 2026-03-01: 数据库从 MySQL 迁移到 PostgreSQL，原因是需要 JSONB 支持
- 2026-03-05: 放弃 Redux，改用 Zustand，因为状态逻辑不复杂
- 2026-03-10: API 限流策略定为 100次/分钟/用户

## 用户偏好
- 喜欢函数式编程风格
- 变量命名用 camelCase
- commit message 用 Conventional Commits 格式

## 已知 Bug 与临时方案
- #237: 文件上传在 Safari 上偶现失败，临时方案：重试 3 次
- #251: WebSocket 重连后偶发消息乱序，待排查
```

在 CLAUDE.md 里引用 memory.md，Claude 每次启动时就会加载这些记忆。日积月累，它对你项目的理解会越来越深——不再是每次都从零开始的"新实习生"，而是一个了解项目历史、清楚技术债务、知道你个人偏好的"老搭档"。

---

## 🚀 第九招：自动化流水线 —— 从"写代码"到"造工厂"

当你把前面的招数组合起来，Claude Code 就不再只是一个编程助手——它变成了一条**自动化流水线**。

一个真实案例：有人用 Claude Code 搭建了一套"AI 应用批量生产线"：

```
1. Claude 抓取社交平台热点话题 → 确定选题方向
2. 人工确认方向（5 分钟）
3. Claude 自动编码生成 Web 应用
4. 自动部署到服务器
5. Claude 生成推广文案
6. 自动发布到社交平台
7. Claude 接管客服，自动回复用户
```

从选题到上线到运营，整条链路几乎全自动。人类只需要在第 2 步拍板确认方向，剩下的全交给 AI。开发成本几乎为零（本地模型跑起来连 API 费都省了），一天能铺好几个轻量应用。

另一个案例：用 Claude Code 搭建**美股盘前量化分析 Agent**：

- 通过 MCP 接入金融数据 API，实时获取股价和新闻
- 编写量化指标分析脚本（波动率、趋势线、RSI 等），封装成 Skill
- 每天定时生成结构化盘前报告（含推荐标的、技术面分析、风险提示）
- 自动转换为 HTML 网页，邮件推送

从"写代码的工具"变成了"帮你赚钱的系统"——这才是 Claude Code 的终极形态。

---

## 💡 第十招：Hooks —— 事件驱动的自动化

Hooks 是 Claude Code 的"条件反射"机制。你可以配置在特定事件发生时自动执行脚本。

在 `.claude/settings.json` 中配置：

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "cat memory/context.md"
      }]
    }],
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "./scripts/error-detector.sh"
      }]
    }]
  }
}
```

| Hook 事件 | 触发时机 | 典型用法 |
|:----------|:---------|:---------|
| UserPromptSubmit | 每次你发送消息前 | 自动注入上下文（如当前分支、环境变量） |
| PostToolUse | Claude 执行完工具后 | 检测命令是否报错，自动记录到错误日志 |
| PreToolUse | Claude 执行工具前 | 权限校验、危险操作拦截 |

一个"自我进化"的玩法：配置 Hook，让 Claude 每次犯错时自动记录到 `.learnings/ERRORS.md`。当同一类错误出现 3 次以上，自动晋升为 CLAUDE.md 中的规则。Claude 在使用中不断学习、不断进化，越用越聪明。

---

## 🗣️ 第十一招：提示词的艺术 —— 少废话，多干货

跟 Claude Code 对话，别客气。它不是你同事，不需要寒暄。

**几个实战话术模板：**

| 场景 | 普通提示词 | 高手提示词 |
|:-----|:----------|:----------|
| 修 Bug | "帮我看看这个 Bug" | "修复 `auth.py` 第 42 行的 NoneType 错误，运行 `pytest tests/test_auth.py` 确认通过" |
| 重构 | "帮我优化一下代码" | "把 `order_service.py` 中超过 50 行的函数拆分为独立方法，保持所有测试通过" |
| 学习 | "解释一下这段代码" | "像给一个有 3 年经验的后端工程师解释，重点说清楚为什么用策略模式而不是 if-else" |
| 挑战 | "帮我写个方案" | "Grill me on these changes. 像严师一样拷问我的方案，发现所有漏洞后再帮我修正" |

Boris 说过一句话特别到位：**"模糊的需求导致模糊的结果，精确的需求导致精确的答案。"** 具体的文件路径、明确的验证方式、可度量的完成标准——给得越多，Claude 干得越好。

还有一个反直觉的技巧：**敢于推倒重来。** 如果 Claude 给的方案不够优雅，直接说："Knowing everything you know now, scrap this and implement the elegant solution."（你现在掌握了所有信息，推倒重来，给我一个优雅的方案。）很多时候第二版比第一版好得多，因为 Claude 在第一次尝试中已经深入理解了问题。

---

## 🎙️ 第十二招：语音输入 —— 说话比打字快三倍

这是一个被严重低估的技巧。

在 macOS 上双击 `fn` 键就能启动系统级语音输入。说话的速度是打字的 3 倍，而且思路更流畅——因为你不需要纠结"这句话怎么措辞"，直接说人话就行。

```
语音输入："帮我在用户注册接口加一个手机号格式校验，
         支持大陆和港澳台的号码格式，
         写好对应的单元测试，
         然后跑一遍确认没问题。"
```

Claude 完全能理解这种口语化的指令。比你敲键盘写一段精雕细琢的 prompt 效率高多了。

---

## 🛠️ 工具链推荐

最后附一份工具链清单，帮你把 Claude Code 的体验拉满：

| 工具 | 作用 | 推荐理由 |
|:-----|:-----|:---------|
| **Ghostty** / **iTerm2** | 终端 | 速度快、支持分屏，比默认 Terminal 好用十倍 |
| **tmux** | 终端复用 | 同时管理多个 Claude 会话，配合 worktree 使用 |
| **git worktree** | 并行开发 | 多分支同时工作，不互相干扰 |
| **Ollama** | 本地模型 | 跑 Qwen/DeepSeek 等开源模型，省 API 费 |
| **cc-switch** | 模型切换 | 一键切换 Claude Code 底层模型 |
| **jq** | JSON 处理 | Claude 输出 JSON 时快速格式化和过滤 |

---

## 📖 概念速查：这些术语到底是啥关系？

刚接触这套体系的人很容易被一堆名词绕晕。这里画一张"族谱"帮你理清关系：

```
LLM（大语言模型）
 └── 只会"文字接龙"，不能执行任何操作
      │
      ├── + Prompt/Context → 能理解指令了
      ├── + Memory → 能记住之前说过啥了
      ├── + Function Calling → 能调用工具了
      │     └── + MCP 协议 → 工具可以即插即用了
      ├── + Agent → 能自主决定下一步做什么了
      │     └── + SubAgent → 能把任务拆给"手下"了
      └── + Skill → 能执行预定义的复杂流程了
```

一句话概括：**Agent 是"大脑+手脚"，MCP 是"手脚的标准接口"，Skill 是"肌肉记忆"，CLAUDE.md 是"企业文化手册"，Memory 是"工作日志"。** 它们各司其职，组合在一起就构成了一个完整的"数字员工"。

这也解释了为什么 Claude Code 跟 ChatGPT 看起来都在"跟 AI 聊天"，体验却天差地别——ChatGPT 是纯 LLM 对话，而 Claude Code 是一个**完整的 Agent 系统**。差距就在那一层层的能力叠加上。

---

## 💰 省钱指南：怎么用 Claude Code 不心疼

说到这儿绕不开一个现实问题：Claude Code 烧 Token。

Claude Sonnet 4 的价格是输入 $3/百万 Token、输出 $15/百万 Token。日常使用一天烧掉 $5-10 很正常，重度用户一天 $30+ 也不稀奇。一个月下来几百美元，确实肉疼。

几个经过验证的省钱策略：

**策略 1：阶梯式模型调度**

不是所有任务都需要最强模型。简单的代码生成、格式化、文件操作，用便宜的模型就够了；只有复杂推理、架构设计、调试疑难 Bug 才需要上 Opus。

```bash
# 切换到便宜模型做日常任务
claude config set preferredModel claude-3.5-sonnet

# 遇到硬骨头再切换
claude config set preferredModel claude-opus-4.6
```

有人用 `cc-switch` 工具实现一键切换，甚至接入 DeepSeek、Qwen 等国产模型——简单任务的成本直接降到原来的十分之一。

**策略 2：善用 /compact 压缩上下文**

对话越长，消耗的 Token 越多。定期输入 `/compact` 让 Claude 压缩上下文，把冗余的历史对话精简掉，能省下不少钱。

**策略 3：本地模型兜底**

用 Ollama 跑一个 Qwen-3.5-27B 或 DeepSeek-V3 的量化版，处理不需要联网的简单任务。API 费用直接归零。M1 以上的 Mac、24G 内存就能流畅跑起来。

**策略 4：Plan Mode 省返工成本**

前面说过，Plan Mode 能省 1 小时返工时间。返工意味着额外的 Token 消耗。算下来，多花 10 分钟做计划，反而是最省钱的做法。

---

## 🔐 安全注意事项：别让 AI 帮你惹麻烦

Claude Code 有读写文件、执行命令的能力——这是双刃剑。

**几条铁律：**

1. **永远不要在有敏感数据的机器上裸跑 Claude Code。** 它可以读取任何本地文件，包括 SSH 密钥、.env 里的数据库密码、浏览器 Cookie。
2. **Skill 生态有"投毒"风险。** 社区分享的 Skill 本质上是 Prompt 注入 + 脚本执行。安装前一定要审查代码，别无脑 `install`。
3. **给 Claude 的权限越小越好。** 能用只读权限就别给写权限，能限制工作目录就别开放全盘访问。
4. **在 CI/CD 中使用时，用 `--dangerously-skip-permissions` 要三思。** 这个 flag 跳过所有权限确认，适合自动化流水线，但也意味着 Claude 可以不经确认执行任何命令。

一个真实的教训：有人让 Claude 帮忙调试登录功能，Claude 为了模拟登录流程，调用了真实的第三方 OAuth 接口，一番操作下来消耗了 $30 的 Token 还触发了风控。这种事靠 CLAUDE.md 里写清楚"禁止调用生产环境接口"就能避免。

---

## 写在最后

Claude Code 的本质是什么？

不是一个更智能的代码补全工具，也不是一个会写代码的聊天机器人。它是一个**Agent 运行时**——一个可以感知环境（读取项目文件）、制定计划（Plan Mode）、执行操作（编辑文件、运行命令）、验证结果（跑测试）的自主智能体。

你跟它的关系，不是"我问你答"，而是"我定方向，你来执行"。从问答模式切换到委托模式（Vibe Coding），才算真正入了门。

而当你把 CLAUDE.md + Skills + MCP + Hooks + SubAgent + Memory 组合起来，你构建的不再是一个工具，而是一个**系统**。这个系统可以自己学习（Memory + Self-improvement）、自己协作（SubAgent）、自己扩展（MCP + Skills），甚至可以 7x24 小时不间断运行。

回顾本文介绍的 12 招，它们构成了一个清晰的进阶路径：

| 阶段 | 招数 | 你的角色转变 |
|:-----|:-----|:------------|
| 入门 | CLAUDE.md + Plan Mode + 提示词 | 从"问答"到"委托" |
| 进阶 | Skills + MCP + Hooks | 从"委托"到"自动化" |
| 高手 | SubAgent + 并行开发 + Memory | 从"自动化"到"系统化" |
| 大神 | 自动化流水线 + IM 接入 | 从"系统化"到"无人值守" |

如果你现在还在第一阶段，别急。每掌握一招，效率就翻一倍。等你把 12 招融会贯通，你会发现自己的工作方式已经跟大多数人完全不一样了——不是你在写代码，而是你在管理一支 AI 团队。

这就是 AI Agent 时代真正的玩法：**不是让 AI 帮你写代码，而是让 AI 帮你建造一台不需要你的机器。**

---


*觉得有启发的话，欢迎点赞、在看、转发。跟进最新AI前沿，关注我的微信公众号：机器懂语言*

