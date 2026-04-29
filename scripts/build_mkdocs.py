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

import re
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MONTH_DIRS = ["202601", "202602", "202603", "202604"]


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

    nav_by_month: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for month in MONTH_DIRS:
        src_dir = ROOT / month
        if not src_dir.is_dir():
            continue
        dst_dir = DOCS / month
        dst_dir.mkdir(parents=True, exist_ok=True)
        for md in sorted(src_dir.glob("*.md")):
            shutil.copy(md, dst_dir / md.name)
            nav_by_month[month].append(
                {
                    "file": f"{month}/{md.name}",
                    "title": parse_title(md),
                    "date": parse_date(md.name),
                }
            )
            total += 1

    nav: list = [{"首页": "index.md"}]
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
        "theme": {
            "name": "material",
            "language": "zh",
            "features": [
                "navigation.instant",
                "navigation.tracking",
                "navigation.tabs",
                "navigation.tabs.sticky",
                "navigation.sections",
                "navigation.top",
                "navigation.indexes",
                "search.suggest",
                "search.highlight",
                "search.share",
                "content.code.copy",
                "content.action.edit",
                "toc.follow",
            ],
            "palette": [
                {
                    "media": "(prefers-color-scheme: light)",
                    "scheme": "default",
                    "primary": "indigo",
                    "accent": "indigo",
                    "toggle": {
                        "icon": "material/weather-night",
                        "name": "切换到深色模式",
                    },
                },
                {
                    "media": "(prefers-color-scheme: dark)",
                    "scheme": "slate",
                    "primary": "indigo",
                    "accent": "indigo",
                    "toggle": {
                        "icon": "material/weather-sunny",
                        "name": "切换到浅色模式",
                    },
                },
            ],
            "icon": {"repo": "fontawesome/brands/github"},
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
            {"pymdownx.arithmatex": {"generic": True}},
        ],
        "extra_javascript": [
            "https://polyfill.io/v3/polyfill.min.js?features=es6",
            "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
        ],
        "plugins": [{"search": {"lang": ["zh", "en"]}}],
        "nav": nav,
        "extra": {
            "social": [
                {
                    "icon": "fontawesome/brands/github",
                    "link": "https://github.com/shibing624",
                }
            ]
        },
    }

    with (ROOT / "mkdocs.yml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, width=200)

    print(f"[build_mkdocs] copied {total} articles into docs/")
    print(f"[build_mkdocs] mkdocs.yml written: {len(nav)} nav sections")


if __name__ == "__main__":
    main()
