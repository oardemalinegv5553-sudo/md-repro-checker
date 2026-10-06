# -*- coding: utf-8 -*-
"""
inspect_sections.py：查看某篇 e-print 里所有章节标题，用来核对方法段是否选对

调试工具：当 fetch_arxiv.py 选出的章节看起来不对（例如选到了子章节
而不是整段计算方法）时，用这个脚本把原文章节树打出来对照。

用法：
  python eval/real/inspect_sections.py 2207.13405
"""
from __future__ import annotations

import os
import re
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEC = re.compile(r"\\(sub)*section\*?\s*\{([^}]*)\}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    aid = sys.argv[1]
    cache = os.path.join(HERE, "_scratch", aid + ".tar.gz")
    if not os.path.exists(cache):
        hits = [p for p in os.listdir(os.path.join(HERE, "_scratch"))
                if p.startswith(aid)]
        if not hits:
            print(f"缓存中没有 {aid}")
            return 1
        cache = os.path.join(HERE, "_scratch", hits[0])

    tf = tarfile.open(cache)
    texs = [m for m in tf.getmembers()
            if m.isfile() and m.name.lower().endswith(".tex")]
    print(f"e-print 内含 {len(texs)} 个 .tex：")
    for m in texs:
        print(f"   {m.name}  ({m.size} B)")

    # 取带 \documentclass 的主文件，退化为最大的一份
    main = None
    for m in texs:
        try:
            s = tf.extractfile(m).read().decode("utf-8", "replace")
        except Exception:
            continue
        if "\\documentclass" in s:
            main = (m.name, s)
            break
    if main is None and texs:
        m = max(texs, key=lambda x: x.size)
        main = (m.name, tf.extractfile(m).read().decode("utf-8", "replace"))
    if main is None:
        print("没有可用 .tex")
        return 1

    name, s = main
    print(f"\n主文件: {name}  ({len(s)} 字符)")
    print("章节树：")
    for m in SEC.finditer(s):
        lvl = "  SUB" if m.group(1) else "  SEC"
        pad = "      " if m.group(1) else "  "
        print(f"{pad}{lvl} @{m.start():7d}  {m.group(2)[:78]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
