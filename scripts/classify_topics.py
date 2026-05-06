"""用 Hunyuan LLM 把每篇论文解读分到固定 taxonomy 中的一个主题。

输出：scripts/topics.json，结构：
  {
    "202604/20260401_xxx.md": {
      "topic": "Agent 智能体",
      "hash": "<sha1 of head>"
    },
    ...
  }

设计：
- taxonomy 固定 10 个，prompt 里强约束 single-label，避免漂移
- 取每篇前 ~600 字（跳过 H1 标题行）作为分类依据，正文开头通常是摘要/导语，信号最强
- 用 head 的 sha1 做增量缓存：head 没变就跳过 LLM，重跑零成本
- 异步并发 10，174 篇约 1-2 分钟
- LLM 解析失败或返回非法标签 → 落 "其他"，不抛异常（外部 I/O 容错）

调用：
  python scripts/classify_topics.py            # 增量
  python scripts/classify_topics.py --force    # 全量重跑
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
MONTH_DIRS = ["202601", "202602", "202603", "202604"]
CACHE_FILE = ROOT / "scripts" / "topics.json"

TAXONOMY = [
    "Agent 智能体",
    "强化学习",
    "RAG 检索增强",
    "推理与思维链",
    "记忆系统",
    "评估与基准",
    "模型架构与训练",
    "AI 科研",
    "工程实践与 Coding Agent",
    "其他",
]

HEAD_CHARS = 600
CONCURRENCY = 10
MODEL = "hunyuan-turbos-latest"

API_KEY = os.environ.get("HUNYUAN_API_KEY")
if not API_KEY:
    sys.exit("[classify] 缺少环境变量 HUNYUAN_API_KEY，请先 `export HUNYUAN_API_KEY=...` 再运行")

client = AsyncOpenAI(
    api_key=API_KEY,
    base_url="http://hunyuanapi.woa.com/openapi/v1",
)

SYSTEM_PROMPT = f"""你是一个论文分类助手。给定一篇 AI 论文中文解读的开头片段，从下面 10 个主题中**选且只选一个**最贴切的：

{chr(10).join(f"- {t}" for t in TAXONOMY)}

分类规则：
1. "Agent 智能体"：单/多 Agent 架构、协作、Agent 框架、Agent 训练（非纯 RL）
2. "强化学习"：核心贡献是 RL/RLHF/RLVR/DPO/PPO/GRPO 等训练方法
3. "RAG 检索增强"：核心是检索、向量库、知识注入、search-augmented 生成
4. "推理与思维链"：CoT、ReAct、self-reflection、reasoning model 推理能力本身
5. "记忆系统"：长程记忆、memory bank、skill library、知识图谱记忆
6. "评估与基准"：benchmark、评测方法、新数据集
7. "模型架构与训练"：新模型权重、新架构、注意力变体、PEFT、量化、预训练
8. "AI 科研"：自动科研 Agent、Auto-Research、AI Scientist 类
9. "工程实践与 Coding Agent"：Claude Code、IDE Agent、源码精读、harness 工程
10. "其他"：实在不属于上述任何一类

输出：**只输出主题名本身**（如 "Agent 智能体"），不要解释、不要加标点、不要输出别的。"""


def read_head(md_path: Path, n_chars: int = HEAD_CHARS) -> str:
    """读文件，跳过首个 H1 标题行，取后续 n_chars 字符。"""
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    skip = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("# "):
            skip = i + 1
            break
    body = "\n".join(lines[skip:]).strip()
    return body[:n_chars]


def head_hash(head: str) -> str:
    return hashlib.sha1(head.encode("utf-8")).hexdigest()[:12]


async def classify_one(relpath: str, head: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    """返回 (relpath, topic)。LLM 失败/非法返回 → "其他"。"""
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": head},
                ],
                temperature=0.0,
                max_tokens=20,
            )
            raw = resp.choices[0].message.content.strip()
            for t in TAXONOMY:
                if t in raw:
                    return relpath, t
            print(f"[warn] {relpath}: LLM 返回未知标签 {raw!r}, 落 '其他'", file=sys.stderr)
            return relpath, "其他"
        except Exception as e:
            print(f"[warn] {relpath}: LLM 调用失败 {e}, 落 '其他'", file=sys.stderr)
            return relpath, "其他"


async def main_async(force: bool) -> None:
    cache: dict = {}
    if CACHE_FILE.exists() and not force:
        cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    todo: list[tuple[str, str]] = []
    skipped = 0
    for month in MONTH_DIRS:
        d = ROOT / month
        if not d.is_dir():
            continue
        for md in sorted(d.glob("*.md")):
            relpath = f"{month}/{md.name}"
            head = read_head(md)
            h = head_hash(head)
            cached = cache.get(relpath)
            if cached and cached.get("hash") == h and not force:
                skipped += 1
                continue
            todo.append((relpath, head))
            cache[relpath] = {"topic": cached["topic"] if cached else None, "hash": h}

    print(f"[classify] 待分类 {len(todo)} 篇，命中缓存 {skipped} 篇")
    if not todo:
        print("[classify] 无新内容，无需调 LLM")
        return

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [classify_one(rp, head, sem) for rp, head in todo]
    results = await asyncio.gather(*tasks)

    for relpath, topic in results:
        cache[relpath]["topic"] = topic

    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    counts: dict[str, int] = {}
    for v in cache.values():
        counts[v["topic"]] = counts.get(v["topic"], 0) + 1
    print(f"[classify] 完成，写入 {CACHE_FILE.relative_to(ROOT)}")
    for t in TAXONOMY:
        if counts.get(t):
            print(f"  {t:30s} {counts[t]}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="忽略缓存，全量重新分类")
    args = p.parse_args()
    asyncio.run(main_async(force=args.force))


if __name__ == "__main__":
    main()
