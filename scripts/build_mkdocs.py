"""扫描 202601~202604 月份目录，把所有论文解读 .md 拷到 docs/ 并生成 mkdocs.yml。

输出：
  docs/index.md                  ← 来自仓库根目录 README.md
  docs/{月份}/{原文件名}.md      ← 拷贝过来的解读文章
  mkdocs.yml                     ← Material 主题 + 按月份分组的 nav

调用：
  python scripts/build_mkdocs.py

设计取舍：
- 用脚本生成 docs/ 而不是直接 docs_dir=., 因为 mkdocs 要求 nav 文件必须在 docs_dir 内
- 月份目录用 02 表示 (e.g. 202604 → "2026 年 4 月"), 倒序展示, 让最新内容在最前
- 文章标题优先取 H1 (第一行 "# xxx"), 取不到则从文件名解析
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SITE_ASSETS = ROOT / "scripts" / "site_assets"
TOPICS_CACHE = ROOT / "scripts" / "topics.json"
MONTH_DIRS = ["202601", "202602", "202603", "202604"]

# 主题展示顺序（与 classify_topics.py 的 TAXONOMY 一致；nav 里按这个顺序排）
TOPIC_ORDER = [
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


def parse_title(md_path: Path) -> str:
    """优先从文件首个 H1 取标题；失败回退到文件名（去掉日期前缀和扩展名）。"""
    with md_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("# "):
                title = line[2:].strip()
                # 去掉 markdown 链接和粗体等噪音
                title = re.sub(r"[\*`]+", "", title)
                if title:
                    return title
    stem = md_path.stem
    stem = re.sub(r"^\d{8}_?", "", stem)
    stem = stem.replace("_", " ")
    return stem


def parse_date(filename: str) -> date:
    """从文件名前缀解析日期：20260429_xxx → 2026-04-29。失败则返回最早日期。"""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})", filename)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return date(y, mo, d)
    return date(1970, 1, 1)


def format_month_label(month_key: str) -> str:
    """202604 → 2026 年 4 月"""
    y, mo = month_key[:4], int(month_key[4:])
    return f"{y} 年 {mo} 月"


def main() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir()

    readme = ROOT / "README.md"
    if readme.exists():
        shutil.copy(readme, DOCS / "index.md")

    # 拷贝自定义 CSS/JS（拓宽正文 + 右侧浮动 TOC + scrollspy）。
    if SITE_ASSETS.is_dir():
        dst_assets = DOCS / "assets"
        dst_assets.mkdir(parents=True, exist_ok=True)
        for f in SITE_ASSETS.iterdir():
            if f.is_file():
                shutil.copy(f, dst_assets / f.name)

    topics_map: dict[str, str] = {}
    if TOPICS_CACHE.exists():
        raw = json.loads(TOPICS_CACHE.read_text(encoding="utf-8"))
        topics_map = {k: v["topic"] for k, v in raw.items() if v.get("topic")}
    else:
        print(f"[build_mkdocs] WARN: 未找到 {TOPICS_CACHE.name}，跳过 '按主题' nav；先跑 classify_topics.py")

    nav_by_month: dict[str, list[dict]] = defaultdict(list)
    nav_by_topic: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for month in MONTH_DIRS:
        src_dir = ROOT / month
        if not src_dir.is_dir():
            continue
        dst_dir = DOCS / month
        dst_dir.mkdir(parents=True, exist_ok=True)
        for md in sorted(src_dir.glob("*.md")):
            shutil.copy(md, dst_dir / md.name)
            relpath = f"{month}/{md.name}"
            entry = {
                "file": relpath,
                "title": parse_title(md),
                "date": parse_date(md.name),
            }
            nav_by_month[month].append(entry)
            topic = topics_map.get(relpath)
            if topic:
                nav_by_topic[topic].append(entry)
            total += 1

    nav: list = [{"首页": "index.md"}]

    # 按主题分组：放在月份之前，更醒目
    if nav_by_topic:
        topic_section: list = []
        for topic in TOPIC_ORDER:
            items = nav_by_topic.get(topic)
            if not items:
                continue
            items_sorted = sorted(items, key=lambda x: x["date"], reverse=True)
            label = f"{topic} ({len(items_sorted)})"
            topic_section.append(
                {label: [{item["title"]: item["file"]} for item in items_sorted]}
            )
        nav.append({"按主题": topic_section})

    for month in sorted(nav_by_month.keys(), reverse=True):
        items = sorted(nav_by_month[month], key=lambda x: x["date"], reverse=True)
        nav.append(
            {
                format_month_label(month): [
                    {item["title"]: item["file"]} for item in items
                ]
            }
        )

    config = {
        "site_name": "AI Paper Analysis",
        "site_description": "AI 前沿论文中文深度解读",
        "site_url": "https://shibing624.github.io/ai-paper-analysis/",
        "site_author": "shibing624",
        "repo_url": "https://github.com/shibing624/ai-paper-analysis",
        "repo_name": "shibing624/ai-paper-analysis",
        "edit_uri": "edit/main/",
        "docs_dir": "docs",
        "site_dir": "site",
        # mkdocs 自带 readthedocs 主题：灰白配色，朴素清爽，不是 Material 那种蓝色。
        # 底部原生支持 ← Previous / Next → 翻页，按 nav 顺序联动，无需额外插件。
        "theme": {
            "name": "readthedocs",
            "highlightjs": True,
            "hljs_languages": ["python", "yaml", "bash", "json", "go", "rust"],
            "navigation_depth": 2,
            "collapse_navigation": True,
            "titles_only": True,
            "sticky_navigation": True,
            "include_homepage_in_sidebar": True,
            "prev_next_buttons_location": "both",  # 顶部+底部都显示 prev/next
            "locale": "zh_CN",
        },
        "markdown_extensions": [
            "admonition",
            "footnotes",
            "tables",
            "attr_list",
            "md_in_html",
            "def_list",
            "toc",
            {"pymdownx.highlight": {"anchor_linenums": True, "line_spans": "__span"}},
            "pymdownx.inlinehilite",
            "pymdownx.snippets",
            "pymdownx.superfences",
            "pymdownx.details",
            "pymdownx.tabbed",
            "pymdownx.tasklist",
            # arithmatex 保留用于解析公式语法，但不再全站注入 MathJax，
            # 实际带公式的文章很少，按需引入即可。
            {"pymdownx.arithmatex": {"generic": True}},
        ],
        # search 只用中文分词；minify 插件压缩 HTML/CSS/JS。
        "plugins": [
            {"search": {"lang": "zh"}},
            {
                "minify": {
                    "minify_html": True,
                    "minify_js": True,
                    "minify_css": True,
                    "htmlmin_opts": {"remove_comments": True},
                }
            },
        ],
        # 自定义增强（仿 Material 体验）：正文加宽到 950px + 右侧浮动 TOC（>=1500px 屏幕显示）。
        # mathjax-on-demand.js 只在页面包含 .arithmatex 时拉 MathJax，不带公式的文章零开销。
        "extra_css": ["assets/extra.css"],
        "extra_javascript": [
            "assets/right-toc.js",
            "assets/mathjax-on-demand.js",
        ],
        "nav": nav,
    }

    with (ROOT / "mkdocs.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, width=200)

    print(f"[build_mkdocs] copied {total} articles into docs/")
    print(f"[build_mkdocs] mkdocs.yml written: {len(nav)} nav sections")


if __name__ == "__main__":
    main()
