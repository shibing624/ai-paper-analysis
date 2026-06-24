"""裁剪 mkdocs 生成的 search_index.json，使其满足 Cloudflare Pages 单文件 25 MiB 限制。

两个削减手段：
1. 用 UTF-8 重新编码（ensure_ascii=False），中文不再转义成 \\uXXXX。
   单字节数从 6 降到 3，整体体积直接砍半，且不丢任何内容。
2. 把每条记录的正文 text 截断到 MAX_CHARS 字。标题、标题层级全部保留，
   正文保留开头部分用于全文检索，体积随论文增长可控。

在 `mkdocs build` 之后、推送 gh-pages 之前执行。
"""

import json
from pathlib import Path

MAX_CHARS = 1000
INDEX_PATH = Path(__file__).resolve().parent.parent / "site" / "search" / "search_index.json"


def main() -> None:
    raw = INDEX_PATH.read_bytes()
    before = len(raw)

    data = json.loads(raw)
    truncated = 0
    for record in data["docs"]:
        text = record.get("text")
        if text and len(text) > MAX_CHARS:
            record["text"] = text[:MAX_CHARS]
            truncated += 1

    out = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    INDEX_PATH.write_bytes(out)
    after = len(out)

    print(
        f"[trim_search_index] {INDEX_PATH.name}: "
        f"{before / 1048576:.2f} MiB -> {after / 1048576:.2f} MiB, "
        f"truncated {truncated}/{len(data['docs'])} records (cap={MAX_CHARS} chars)"
    )


if __name__ == "__main__":
    main()
