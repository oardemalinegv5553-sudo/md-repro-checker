# -*- coding: utf-8 -*-
"""
preview_picks.py：预览章节选择器在全部缓存 e-print 上会选哪一段

为什么要有这个：
  fetch_arxiv.py 的章节选择一旦选错，后面所有指标都是假的，
  但选错这件事本身不会报错，只会表现为"分数莫名很低"。
  本脚本把"选了什么"单独打出来，让选择质量可以和抽取质量分开看。

  它**只读 _scratch/ 缓存**，不下载、不写 sections/，
  因此可以在抓取任务运行期间安全地跑。

用法：
  python eval/real/preview_picks.py
  python eval/real/preview_picks.py --short 40    # 只显示前 40 个候选标题
  python eval/real/preview_picks.py --why         # 列出被领域闸门挡掉的论文及判据
"""
from __future__ import annotations

import argparse
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import fetch_arxiv as F          # noqa: E402


def ws_collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="预览章节选择结果")
    ap.add_argument("--short", type=int, default=52, help="标题截断长度")
    ap.add_argument("--why", action="store_true",
                    help="列出被领域闸门挡掉的论文，以及它命中了哪些判据词")
    args = ap.parse_args(argv)

    if not os.path.isdir(F.SCRATCH):
        print("没有 _scratch/，先跑 fetch_arxiv.py")
        return 2

    cached = sorted(f for f in os.listdir(F.SCRATCH)
                    if f.endswith(".tar.gz")
                    and os.path.getsize(os.path.join(F.SCRATCH, f)) > 0)
    print(f"缓存 {len(cached)} 个 e-print\n")
    print(f"{'arXiv ID':<20} {'L':>2} {'字数':>7}  选中章节")
    print("-" * 100)

    n_l1 = n_bad = n_err = n_oos = n_pdf = n_notex = 0
    rejected = []
    for fn in cached:
        aid = F.cache_id(fn)
        if not F.ARXIV_ID.match(aid):
            print(f"{aid:<20} {'':>2} {'':>7}  [--] 缓存文件名不是 arXiv 编号，跳过")
            n_oos += 1
            continue
        try:
            tex = F.fetch_tex(aid)                 # 命中缓存，不联网
        except F.NotLatexSource as e:
            print(f"{aid:<20} {'':>2} {'':>7}  [--] 非 LaTeX 源码：{e}")
            n_pdf += 1
            continue
        except Exception as e:
            print(f"{aid:<20} {'':>2} {'':>7}  [!!] 读缓存失败 {type(e).__name__}: {e}")
            n_err += 1
            continue
        if not tex:
            print(f"{aid:<20} {'':>2} {'':>7}  [--] e-print 内没有任何 .tex")
            n_notex += 1
            continue
        rep = F.md_scope_report(F.strip_latex(tex))
        if not rep["ok"]:
            print(f"{aid:<20} {'':>2} {'':>7}  [--] 领域闸门：非分子模拟论文"
                  f"（强证据 {len(rep['strong'])}）")
            rejected.append((aid, rep, tex))
            n_oos += 1
            continue
        try:
            title, body, level = F.pick_methods(tex)
            txt = F.strip_latex(body) if body else ""
        except Exception as e:
            print(f"{aid:<20} {'':>2} {'':>7}  [!!] 选段失败 {type(e).__name__}: {e}")
            n_err += 1
            continue
        if not txt:
            print(f"{aid:<20} {'':>2} {'':>7}  [--] 没有方法段")
            n_bad += 1
            continue
        if level == 1:
            n_l1 += 1
        mark = " " if level == 1 else "~"
        print(f"{aid:<20} {mark}{level} {len(txt):7d}  {title[:args.short]}")
        if len(txt) < 800:
            n_bad += 1

    n = len(cached)
    print("-" * 100)
    print(f"整章节(L1) {n_l1}/{n}   过短 {n_bad}   领域外 {n_oos}   "
          f"非 LaTeX {n_pdf}   无 .tex {n_notex}   出错 {n_err}")
    if n_pdf:
        print(f"注：{n_pdf} 篇是作者只交了 PDF，没有 LaTeX 源码可抓。")
        print("    本评估刻意不解 PDF（解析器误差不该混进工具质量指标），")
        print("    因此这些论文是**语料规模**的限制，不是工具的问题。")
    if n_l1 < n * 0.6:
        print("提示：L1 比例偏低，选中的多是子章节，方法段可能被截断，")
        print("      这会让'缺失'类指标虚高。用 inspect_sections.py 看具体某一篇。")

    # 领域闸门判错 = 静默删掉一篇真实论文，指标上完全看不出来。
    # 所以把它的判据全部摊开，让人可以逐条核对（--why）。
    if args.why and rejected:
        print("\n" + "=" * 100)
        print("被领域闸门挡掉的论文（请核对是否真的跑题）")
        print("=" * 100)
        for aid, rep, tex in rejected:
            print(f"\n{aid}   强证据 {len(rep['strong'])} 弱证据 {len(rep['weak'])}")
            ta = F.strip_latex(F.title_and_abstract(tex))
            print(f"  标题/摘要: {ws_collapse(ta)[:150]}")
            print(f"  命中强判据: {', '.join(rep['strong']) or '（无）'}")
            print(f"  命中弱判据: {', '.join(rep['weak']) or '（无）'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
