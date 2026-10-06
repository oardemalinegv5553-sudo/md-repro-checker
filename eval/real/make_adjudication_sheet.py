# -*- coding: utf-8 -*-
"""make_adjudication_sheet.py：重生成「分歧裁定工作表」与「分歧聚类」

为什么要单独成脚本
------------------
`_label_work/_adjudicate_24.md` 是逐条裁定用的清单，它**必须能重生成**。
工具一改版，分歧集合就变了：旧分歧可能已一致（对应裁定随之悬空），
新分歧可能出现。照着一张过期的清单逐条裁，会在已经解决的问题上白做功，
而且会漏掉新出现的分歧。

清单若是临时脚本生成、脚本又没留下，改版后就无法重跑。
这恰好是本项目专抓的那类缺陷：看着有（清单在），其实没有（生成不了）。

用法
----
    python eval/real/make_adjudication_sheet.py            # 写两个文件
    python eval/real/make_adjudication_sheet.py --check    # 只打印摘要

输出（都在 `_label_work/` 下，按 README 的打包规则**不得分发**）：
    _adjudicate_24.md   逐条裁定清单，按「字段 × 转移」聚类
    _cluster_24.txt     汇总视角：按转移类型 / 字段 / 篇目
"""
from __future__ import annotations

import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                          # pragma: no cover
    pass

import score_real as S                                     # noqa: E402

WORK = S.WORK
SHEET = os.path.join(WORK, "_adjudicate_24.md")
CLUSTER = os.path.join(WORK, "_cluster_24.txt")

# 第一批 12 篇（已有裁定）与第二批 12 篇（后补）。顺序不影响结果，
# 只决定汇总表里「前 12 篇 / 后 12 篇」两行的归属。
BATCH1 = [
    "1007.5359v2", "1208.6331v1", "1707.05453v1", "1812.03256v1",
    "2110.08914v2", "2312.07027v2", "2409.15478v1", "2506.04835v1",
    "2509.00937v1", "2602.17595v1", "2603.28012v1", "2605.16123v1",
]

_RULE = "=" * 72


def _batch(aid: str) -> int:
    return 1 if aid in BATCH1 else 2


def collect():
    """跑一遍当前工具，返回 (labels 数, 全部结果, 待裁定分歧)。"""
    labels = S.load_labels()
    adj = {}
    p = os.path.join(WORK, "adjudication.json")
    if os.path.exists(p):
        adj = json.load(io.open(p, encoding="utf-8"))
    r = S.run_all(labels, adjudication=adj)
    # 待裁定 = 无裁定意见的分歧。
    # 悬空裁定的那一对**已经一致**，因此不会出现在 disagreements 里，
    # 也就是说 disagreements 中 verdict 为空的就是真待裁定，不含悬空。
    pend = [d for d in r["disagreements"] if not d["verdict"]]
    return len(labels), r, pend


def render_sheet(n_papers: int, r, pend) -> str:
    dis = r["disagreements"]
    inert = r.get("inert_verdicts") or []
    L = []

    per = {1: {"dis": 0, "adj": 0, "pend": 0}, 2: {"dis": 0, "adj": 0, "pend": 0}}
    for d in dis:
        b = _batch(d["id"])
        per[b]["dis"] += 1
        per[b]["adj" if d["verdict"] else "pend"] += 1

    L.append(f"# {n_papers} 篇标注集，分歧裁定工作表")
    L.append("")
    L.append(f"共 **{len(dis)}** 处分歧（{r['n_judgments']} 个判定）。"
             "本文件由 `eval/real/make_adjudication_sheet.py` 生成，工具改版后可随时重跑。")
    L.append("")
    L.append("| 范围 | 分歧数 | 已裁定 | 待裁定 |")
    L.append("|---|---|---|---|")
    L.append(f"| 前 12 篇 | {per[1]['dis']} | {per[1]['adj']} | **{per[1]['pend']}** |")
    L.append(f"| 后 12 篇 | {per[2]['dis']} | {per[2]['adj']} | **{per[2]['pend']}** |")
    L.append(f"| 合计 | {len(dis)} | {len(dis) - len(pend)} | **{len(pend)}** |")
    L.append("")
    if inert:
        L.append(f"> **悬空裁定 {len(inert)} 条**：`adjudication.json` 里的这些裁定，"
                 "其对应的一对在当前工具版本下**已经一致**，裁定作废，"
                 "`score_real.py` 会列出来但不计入。它们不计入上表的任何一列。")
        L.append("")
    L.append("裁定的写法：在 `adjudication.json` 里加一条 "
             "`\"<arXiv-id>|<字段>\": \"tool_wrong\"` 或 `\"annotator_wrong\"`，然后跑：")
    L.append("")
    L.append("```bash")
    L.append("python eval/real/score_real.py --adjudicated")
    L.append("```")
    L.append("")
    L.append("> **裁定会过期**：裁定是针对*某一次具体分歧*的判，不是对字段的永久判决。"
             "工具一改版，这一对可能已经一致，旧裁定会被 `score_real.py` 显式作废"
             "并计数上报（见 README「裁定会过期」）。")
    L.append("")
    L.append("> **裁定者是自己**，给自家工具判分天然偏松，两个数都要报，"
             "不能只报裁定后那个。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 一、待裁定分歧，按「字段 × 转移」聚类")
    L.append("")
    L.append("同一类里的例子常同因，但**不能整类一刀切**，同一类里工具可能对"
             "某几篇是对的。逐例看备注。")
    L.append("")

    groups = collections.OrderedDict()
    for d in pend:
        groups.setdefault((d["field"], d["gold"], d["got"]), []).append(d)
    # most_common 的并列顺序 = 首次出现顺序（稳定排序），与旧版清单一致
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))

    for (field, gold, got), items in ranked:
        L.append(f"### `{field}`  {gold} → {got}   （{len(items)} 例）")
        L.append("")
        for d in items:
            head = f"- **{d['id']}**：工具判 `{d['got']}`"
            if d.get("got_value") is not None:
                head += "  工具值 = " + json.dumps(d["got_value"], ensure_ascii=False)
            L.append(head)
            if d.get("evidence"):
                L.append(f"    - 标注引文：{d['evidence']}")
            if d.get("annotator_note"):
                L.append(f"    - 标注备注：{d['annotator_note']}")
        L.append("")
        L.append("")

    return "\n".join(L).rstrip("\n") + "\n"


def render_cluster(n_papers: int, r, pend) -> str:
    dis = r["disagreements"]
    L = []
    L.append(f"判定总数 {r['n_judgments']}，分歧 {len(dis)}"
             f"（{len(dis) / max(1, r['n_judgments']):.1%}），"
             f"其中待裁定 {len(pend)}")
    L.append(f"第一批（有裁定）{len(BATCH1)} 篇，第二批（后补）{n_papers - len(BATCH1)} 篇")
    L.append("")

    L.append(_RULE)
    L.append("一、按转移类型（全部分歧）")
    c = collections.Counter((d["gold"], d["got"]) for d in dis)
    for (g, t), n in c.most_common():
        L.append("   %-16s-> %-17s%d" % (g, t, n))

    L.append(_RULE)
    L.append("二、按字段 × 转移类型（只列 >= 4 的）")
    c2 = collections.Counter((d["field"], d["gold"], d["got"]) for d in dis)
    for (f, g, t), n in c2.most_common():
        if n >= 4:
            L.append("   %-23s%-16s-> %-17s%d" % (f, g, t, n))

    L.append(_RULE)
    L.append("三、按篇目（降序）")
    c3 = collections.Counter(d["id"] for d in dis)
    for aid, n in c3.most_common():
        L.append("   %-23s[批%s] %d" % (aid, "一" if _batch(aid) == 1 else "二", n))

    L.append(_RULE)
    L.append("四、not_applicable -> missing 这一类：按字段")
    L.append("   （本类是本工具最大的单一假阳性来源：被问的对象在体系里根本不存在）")
    c4 = collections.Counter(d["field"] for d in dis
                             if d["gold"] == "not_applicable" and d["got"] == "missing")
    for f, n in c4.most_common():
        L.append("   %-23s%d" % (f, n))

    L.append(_RULE)
    L.append("五、ambiguous -> missing / ambiguous -> present：按字段")
    c5 = collections.Counter((d["field"], d["got"]) for d in dis
                             if d["gold"] == "ambiguous")
    for (f, t), n in c5.most_common():
        L.append("   %-23s-> %-17s%d" % (f, t, n))

    return "\n".join(L).rstrip("\n") + "\n"


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    check = "--check" in argv

    n_papers, r, pend = collect()
    if not n_papers:
        print("没有标注：_label_work/out 为空。")
        return 2

    sheet = render_sheet(n_papers, r, pend)
    cluster = render_cluster(n_papers, r, pend)

    if check:
        print(cluster)
        print(f"[--check] 未写文件。工作表将含 {len(pend)} 条待裁定。")
        return 0

    io.open(SHEET, "w", encoding="utf-8").write(sheet)
    io.open(CLUSTER, "w", encoding="utf-8").write(cluster)
    print(f"已写 {SHEET}")
    print(f"已写 {CLUSTER}")
    print(f"\n共 {len(r['disagreements'])} 处分歧，"
          f"待裁定 {len(pend)}，悬空裁定 {len(r.get('inert_verdicts') or [])}。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
