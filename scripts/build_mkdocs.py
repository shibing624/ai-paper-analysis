"""扫描所有 YYYYMM 月份目录，把论文解读 .md 同步到 docs/ 并生成 mkdocs.yml。

输出：
  docs/index.md                  ← 来自仓库根目录 README.md
  docs/{月份}/{原文件名}.md      ← 同步过来的解读文章
  mkdocs.yml                     ← readthedocs 主题 + 按主题/月份分组的 nav

调用：
  python scripts/build_mkdocs.py

设计取舍：
- 用脚本生成 docs/ 而不是直接 docs_dir=., 因为 mkdocs 要求 nav 文件必须在 docs_dir 内
- 月份目录用 02 表示 (e.g. 202604 → "2026 年 4 月"), 倒序展示, 让最新内容在最前
- 文章标题优先取 H1 (第一行 "# xxx"), 取不到则从文件名解析
- **增量同步而非 rmtree 重建**：docs/ 有 500+ 文件，整目录删重建会触发 agent
  运行时的批量删除保护（阈值 50 文件），脚本被静默 kill、一行日志都不输出。
  现在只拷贝内容有变化的文件，并单独清理孤儿（源文件已删/改名的残留副本），
  日常增量场景删除数为 0。孤儿数超过 ORPHAN_ALERT 时只告警不删，避免误伤。
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
README_FILE = ROOT / "README.md"
MONTH_RE = re.compile(r"^\d{6}$")
README_LIST_START = "<!-- AUTO-LIST-START -->"
README_LIST_END = "<!-- AUTO-LIST-END -->"

# 孤儿文件（docs/ 里存在但源目录已没有的副本）一次删超过这个数就只告警不删。
# 正常增量只会有 0~几个孤儿；一旦爆量说明源目录被误删或挂载异常，不该跟着删。
ORPHAN_ALERT = 50


def sync_file(src: Path, dst: Path) -> bool:
    """按内容同步单个文件，内容一致则跳过。返回是否实际写入。"""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        if dst.read_bytes() == src.read_bytes():
            return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return True


def prune_orphans(expected: set[Path]) -> tuple[int, int]:
    """删除 docs/ 下不在 expected 中的 .md 文件（源已删除/改名的残留）。

    返回 (已删除数, 因超过 ORPHAN_ALERT 而跳过的数量)。
    assets/ 与 index.md 不在清理范围内。
    """
    if not DOCS.exists():
        return 0, 0
    orphans = [
        p
        for p in DOCS.rglob("*.md")
        if p not in expected and p != DOCS / "index.md" and "assets" not in p.parts
    ]
    if not orphans:
        return 0, 0
    if len(orphans) > ORPHAN_ALERT:
        print(
            f"[build_mkdocs] WARN: 检测到 {len(orphans)} 个孤儿文件，超过阈值 "
            f"{ORPHAN_ALERT}，已跳过清理。请确认月份目录是否被误删；"
            f"确认无误后手动 rm -rf docs/ 再重跑。"
        )
        return 0, len(orphans)
    for p in orphans:
        p.unlink()
        print(f"[build_mkdocs] 清理孤儿：{p.relative_to(DOCS)}")
    # 顺带清掉空目录（例如整个月份目录被移走）
    for d in sorted((d for d in DOCS.rglob("*") if d.is_dir()), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return len(orphans), 0


def discover_month_dirs(root: Path) -> list[str]:
    """扫描 root 下所有形如 YYYYMM 的目录名（按时间正序）。"""
    return sorted(d.name for d in root.iterdir() if d.is_dir() and MONTH_RE.fullmatch(d.name))

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


def update_readme_list(entries: list[dict]) -> None:
    """用所有解读文章重写 README.md 的文章列表段（sentinel 之间的内容）。

    sentinel 之外保持不动；如未找到 sentinel 则不改。entries 已按日期倒序。
    """
    if not README_FILE.exists():
        return
    text = README_FILE.read_text(encoding="utf-8")
    if README_LIST_START not in text or README_LIST_END not in text:
        print(f"[build_mkdocs] WARN: README 缺少 sentinel，跳过自动列表更新")
        return

    lines = ["", "| 日期 | 文章 |", "|------|------|"]
    for e in entries:
        lines.append(f"| {e['date'].isoformat()} | [{e['title']}]({e['file']}) |")
    block = "\n".join(lines)

    pattern = re.compile(
        re.escape(README_LIST_START) + r".*?" + re.escape(README_LIST_END),
        flags=re.DOTALL,
    )
    new = pattern.sub(f"{README_LIST_START}\n{block}\n{README_LIST_END}", text)
    if new != text:
        README_FILE.write_text(new, encoding="utf-8")
        print(f"[build_mkdocs] README 文章列表已刷新（{len(entries)} 篇）")


def main() -> None:
    # 增量同步：不再 rmtree 整个 docs/（500+ 文件会触发批量删除保护）。
    DOCS.mkdir(exist_ok=True)
    copied = 0

    # 同步自定义 CSS/JS（拓宽正文 + 右侧浮动 TOC + scrollspy）。
    if SITE_ASSETS.is_dir():
        dst_assets = DOCS / "assets"
        dst_assets.mkdir(parents=True, exist_ok=True)
        for f in SITE_ASSETS.iterdir():
            if f.is_file():
                copied += sync_file(f, dst_assets / f.name)

    topics_map: dict[str, str] = {}
    if TOPICS_CACHE.exists():
        raw = json.loads(TOPICS_CACHE.read_text(encoding="utf-8"))
        topics_map = {k: v["topic"] for k, v in raw.items() if v.get("topic")}
    else:
        print(f"[build_mkdocs] WARN: 未找到 {TOPICS_CACHE.name}，跳过 '按主题' nav；先跑 classify_topics.py")

    nav_by_month: dict[str, list[dict]] = defaultdict(list)
    nav_by_topic: dict[str, list[dict]] = defaultdict(list)
    all_entries: list[dict] = []
    expected: set[Path] = set()
    total = 0
    month_dirs = discover_month_dirs(ROOT)
    print(f"[build_mkdocs] 扫描到月份目录：{', '.join(month_dirs) or '(空)'}")
    for month in month_dirs:
        src_dir = ROOT / month
        dst_dir = DOCS / month
        dst_dir.mkdir(parents=True, exist_ok=True)
        for md in sorted(src_dir.glob("*.md")):
            dst = dst_dir / md.name
            copied += sync_file(md, dst)
            expected.add(dst)
            relpath = f"{month}/{md.name}"
            entry = {
                "file": relpath,
                "title": parse_title(md),
                "date": parse_date(md.name),
            }
            nav_by_month[month].append(entry)
            all_entries.append(entry)
            topic = topics_map.get(relpath)
            if topic:
                nav_by_topic[topic].append(entry)
            total += 1

    pruned, skipped = prune_orphans(expected)

    update_readme_list(sorted(all_entries, key=lambda x: (x["date"], x["file"]), reverse=True))

    if README_FILE.exists():
        copied += sync_file(README_FILE, DOCS / "index.md")

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

    print(
        f"[build_mkdocs] synced {total} articles into docs/ "
        f"(written {copied}, pruned {pruned}, orphan-alert {skipped})"
    )
    print(f"[build_mkdocs] mkdocs.yml written: {len(nav)} nav sections")


if __name__ == "__main__":
    main()
