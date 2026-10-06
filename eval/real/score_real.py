# -*- coding: utf-8 -*-
"""
score_real.py：在**真实文献**的独立标注上度量抽取质量

与 eval/score.py 的分工：
  score.py 跑的是 case-01（我们自己写的合成方法段，n=2），用来暴露错误类型。
  本文件跑的是 arXiv 真实方法段（n=24 篇 × 18 字段 = 432 个判定），
  用来给出**真实的**状态/取值指标，这件事在 case-01 上做不到：
  规则在自写文本上全对，只证明规则符合我们自己的预期。

关于标注员的性质，见 eval/real/README.md「二轮」一节：两批都是**模型标注**
（每篇一个独立上下文的子代理，看不到 src/、看不到彼此），不是人工标注。
协议能隔断的都隔断了，但"同一个模型既写工具又当标注员"这一层隔不断。

三个刻意的设计决定：

1. **池化（micro）而不是按篇宏平均。**
   432 个判定放进同一张混淆矩阵。按篇宏平均会让"没有 missing 样本的篇目"
   把该类的分数凭空抬到 1.0；池化则如实反映每一类到底判对多少。
   两种都打印，但结论看池化那一张。

2. **四态一起算，包括 ambiguous。**
   ambiguous 是最容易被忽略的一类：把它并进 present 或 missing 都会
   掩盖"说了但没说清"这种真实存在的失败模式。

3. **分歧不自动算作工具的错。**
   标注员也会错。本脚本只**列出**分歧（含原文片段 + 标注员的 note），
   由人逐条裁定后写进 adjudication.json；
   `--adjudicated` 模式才按裁定结果重算。默认输出的是"原始一致率"，
   不是"准确率"，这两者不是一回事，混为一谈就是自欺。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src import validate                                  # noqa: E402
from src.extract import extract as run_extract            # noqa: E402

WORK = os.path.join(HERE, "_label_work")
OUT = os.path.join(WORK, "out")
STATES = ["present", "missing", "ambiguous", "not_applicable"]


# -----------------------------------------------------------------------------
def load_labels() -> List[Dict[str, Any]]:
    """读取 _label_work/out/*.json，按 manifest 顺序排列。"""
    if not os.path.isdir(OUT):
        return []
    labels = []
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".json"):
            continue
        d = json.load(io.open(os.path.join(OUT, fn), encoding="utf-8"))
        labels.append(d)
    return labels


def section_meta(aid: str) -> Dict[str, Any]:
    """从我们自己加的头注释里取章节标题与层级。

    层级很要紧：level 2 是**子章节**，它天然可能漏掉正文别处写过的内容。
    对子章节而言，missing 未必是"作者没写"，也可能只是"不在这一段里"。
    所以指标必须按层级分开报，否则会把"截取范围的限制"记成
    "工具读不出来"，或反过来把两者混在一起自我表扬。
    """
    p = os.path.join(HERE, "sections", aid.replace("/", "_") + ".txt")
    if not os.path.exists(p):
        p = os.path.join(HERE, "sections", aid + ".txt")
    raw = io.open(p, encoding="utf-8").read()
    meta: Dict[str, Any] = {"section": "", "level": None}
    for line in raw.splitlines():
        if not line.startswith("#"):
            break
        m = re.match(r"#\s*(section|level)\s*:\s*(.*)$", line, re.I)
        if m:
            k, v = m.group(1).lower(), m.group(2).strip()
            if k == "level":
                lm = re.match(r"(\d+)", v)
                meta["level"] = int(lm.group(1)) if lm else None
            else:
                meta["section"] = v
    body = "\n".join(l for l in raw.splitlines() if not l.startswith("#"))
    meta["text"] = body
    return meta


def section_text(aid: str) -> str:
    return section_meta(aid)["text"]


# -----------------------------------------------------------------------------
# 取值比较
#
# 标注员写的是自由文本（可能带括号说明），工具给的是规范化名字或 dict。
# 直接字符串相等会把大量**格式差异**算成取值错误，那是在虚报问题。
# 这里只做保守归一化，并且把每一处分歧都留给人看。
# -----------------------------------------------------------------------------
_PAREN = re.compile(r"[（(][^）)]*[）)]")
_NONWORD = re.compile(r"[^a-z0-9./+-]+")


def _dump(v: Any, keep_keys: bool) -> str:
    """dict -> 可比较的字符串。

    keep_keys=False 时**只取值、丢掉键名**。这一点很要紧：原实现把 dict
    摊成 "method=pme rcoulomb-nm=1.0" 再比，于是 **dict 型字段与自由文本
    标注永远不可能匹配**，键名 "method=" 当然不会出现在标注里。
    后果是取值准确率被系统性低估：一篇写对了 "PME + 1.0 nm" 的抽取，
    只因为标注写得更长就算错。这里问的是"工具报的这个值，标注支持吗"。
    """
    if v is None:
        return ""
    if isinstance(v, dict):
        if keep_keys:
            parts = [str(v.get("type", ""))]
            for k in sorted(v):
                if k != "type" and not isinstance(v[k], (dict, list)):
                    parts.append(f"{k}={v[k]}")
        else:
            parts = [str(x) for k in sorted(v)
                     if not isinstance(x := v[k], (dict, list))]
        v = " ".join(parts)
    s = str(v).lower()
    s = _PAREN.sub(" ", s)              # 去掉括号内的说明性文字
    s = s.replace("_", "-")
    s = _NONWORD.sub(" ", s).strip()
    s = re.sub(r"\s+", " ", s)
    # 10.0 与 10 是同一个数，不是取值差异
    return " ".join(re.sub(r"^(\d+)\.0+$", r"\1", t) for t in s.split())


def value_match(gold: Any, got: Any) -> bool:
    """判定：工具报的值与标注**相容**。

    判据是词元集合的包含关系（任一方向）。刻意宽松，因为取值这一侧
    格式差异太多（"1.0 nm" / "1 nm" / "rcoulomb=1.0"）。

    这条指标读作"**工具报出来的值被标注支持吗**"，
    而**不是**"工具把标注里所有细节都抓全了吗"。后者是另一回事：
    标注常常写成一句话描述多个子项，工具抓到其中一个正确子项即算相容。
    真正要抓的"取值错误"（报了个标注里根本没有的数）依然会被抓出来，
    错的数字不在标注里，子集关系不成立。
    """
    if gold is None or got is None:
        return False
    a, b = _dump(gold, False), _dump(got, False)
    if not a or not b:
        return False
    if a == b:
        return True
    ta, tb = set(a.split()), set(b.split())
    return ta <= tb or tb <= ta


# -----------------------------------------------------------------------------
def run_all(labels: List[Dict[str, Any]], adjudication: Optional[Dict[str, str]] = None
            ) -> Dict[str, Any]:
    paths = validate.default_paths()
    schema = validate.load_yaml(paths["schema"])
    sensitivity = validate.load_yaml(paths["sensitivity"])

    cm = {a: {b: 0 for b in STATES + ["absent"]} for a in STATES}
    rows: List[Dict[str, Any]] = []
    disagreements: List[Dict[str, Any]] = []
    value_rows: List[Dict[str, Any]] = []
    per_section: List[Dict[str, Any]] = []
    missing_labels: List[str] = []
    inert_verdicts: List[Dict[str, str]] = []

    for lab in labels:
        aid = lab["id"]
        try:
            sm = section_meta(aid)
            text = sm["text"]
        except Exception as e:                             # noqa: BLE001
            missing_labels.append(f"{aid}: {type(e).__name__}")
            continue
        extraction = run_extract(text, schema, backend="rule")
        validation = validate.validate(extraction, schema, sensitivity, raw_text=text)
        got_fields = extraction["fields"]

        n_ok = 0
        n_tot = 0
        for key, g in lab["fields"].items():
            gs = g.get("status")
            if gs not in STATES:
                continue
            got = got_fields.get(key) or {}
            ps = got.get("status", "absent")
            n_tot += 1

            # 裁定：如果这条分歧被人工判定为"标注员错"，则按工具的结果算对
            verdict = None
            if adjudication is not None:
                verdict = adjudication.get(f"{aid}|{key}")
                # 裁定是对**某一次具体分歧**的判断，不是对该字段的永久判决。
                # 工具一改版，被裁定的那一对可能已经一致了；此时若还按
                # "tool_wrong" 扣分，就是**对一个已不存在的分歧罚分**。
                # 方向不重要（这里恰好是低估），错的是逻辑。
                # 实测：恒压器适用性判据放宽又改回，留下 5 条悬空裁定，
                # 全是 tool_wrong，不处理就会把 5 个已一致的判定算成错。
                # 这里显式作废并**计数上报**，不静默丢弃，静默丢弃与
                # 静默罚分一样，都会让指标对不上账。
                if verdict and gs == ps:
                    inert_verdicts.append({"id": aid, "field": key,
                                           "verdict": verdict})
                    verdict = None
            effective_ok = (gs == ps)
            if verdict == "annotator_wrong":
                effective_ok = True
                ps_eff = gs
            else:
                ps_eff = ps                                  # 分歧一律先算工具的错
            if verdict == "tool_wrong":
                effective_ok = False

            if effective_ok:
                n_ok += 1
            if cm.get(gs) and ps_eff in cm[gs]:
                cm[gs][ps_eff] += 1

            if gs != ps:
                disagreements.append({
                    "id": aid, "field": key, "gold": gs, "got": ps,
                    "gold_value": g.get("value"),
                    "got_value": got.get("value"),
                    "evidence": g.get("evidence") or "",
                    "annotator_note": g.get("note") or "",
                    "verdict": verdict or "",
                })

            # 取值只在双方都给出具体值时比较
            if gs == "present" and g.get("value") is not None and ps == "present":
                ok = value_match(g["value"], got.get("value"))
                value_rows.append({"id": aid, "field": key, "ok": ok,
                                   "gold": g["value"], "got": got.get("value")})

        per_section.append({
            "id": aid, "n_ok": n_ok, "n_total": n_tot,
            "acc": (n_ok / n_tot) if n_tot else 0.0,
            "section": sm["section"], "level": sm["level"],
            "status": {k: (got_fields.get(k) or {}).get("status")
                       for k in lab["fields"]},
            # 逐格取值，供 status_diff.py 对账用。它要钉住一条约束：
            # 状态被改动的格子**不得同时获得取值**，报 ambiguous 却带个值，
            # 就是"说了但没说清"变成"说了"，两回事被悄悄合并。
            "values": {k: (got_fields.get(k) or {}).get("value")
                       for k in lab["fields"]},
            "rules": sorted({f.get("rule_id") for f in validation["findings"]
                             if f.get("rule_id")}),
            "score": validation["score"], "grade": validation["grade"],
            "ctx": {k: v for k, v in validation["context"].items() if v is True},
            # 完整的上下文，供 status_diff.py 钉"ctx 未漂移"用。
            # 上面那个 ctx 只留值为 True 的键，是给报告当"场景"显示的；
            # 但它**看不见 ctx["ensemble"]**，那是字符串（"NPT"/"NVE"），
            # 从来不是 True。而 ensemble 恰恰是恒压器适用性闸门唯一的输入
            # （_APPLICABILITY["barostat"] == `ctx["ensemble"] == "NPT"`）。
            # 于是原样的"ctx 未漂移"检查对这条闸门是瞎的：闸门翻了个面，
            # 它照样报 OK。加这个键，让对账看得到闸门输入本身，
            # 而不是只看闸门翻面之后在字段层留下的痕迹。
            "ctx_all": dict(validation["context"]),
        })

    # ---- 池化指标 ----
    def _pool(cmat):
        pc: Dict[str, Dict[str, float]] = {}
        for c in STATES:
            tp = cmat[c][c]
            fp = sum(cmat[o][c] for o in STATES if o != c)
            fn = sum(cmat[c][o] for o in cmat[c] if o != c)
            support = sum(cmat[c].values())
            p = tp / (tp + fp) if (tp + fp) else 0.0
            rr = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * p * rr / (p + rr) if (p + rr) else 0.0
            pc[c] = {"precision": p, "recall": rr, "f1": f1, "support": support,
                     "tp": tp, "fp": fp, "fn": fn}
        scored_ = [v["f1"] for v in pc.values() if v["support"] > 0]
        tot = sum(sum(cmat[a].values()) for a in STATES)
        cor = sum(cmat[c][c] for c in STATES)
        return {
            "per_class": pc,
            "macro_f1": (sum(scored_) / len(scored_)) if scored_ else 0.0,
            "status_acc": (cor / tot) if tot else 0.0,
            "n_judgments": tot,
            "n_correct": cor,
        }

    pooled = _pool(cm)

    # 按章节层级分层：level 1 是完整章节，level 2 是子章节。
    # 子章节天然可能漏掉正文别处写过的内容，两者不能混在一起报。
    by_level: Dict[str, Any] = {}
    for lv in (1, 2):
        sub = {a: {b: 0 for b in STATES + ["absent"]} for a in STATES}
        # 重算该层的混淆矩阵
        for lab in labels:
            aid = lab["id"]
            sec = next((s for s in per_section if s["id"] == aid), None)
            if sec is None or sec["level"] != lv:
                continue
            for key, g in lab["fields"].items():
                gs = g.get("status")
                if gs not in STATES:
                    continue
                ps = sec["status"].get(key, "absent")
                verdict = adjudication.get(f"{aid}|{key}") if adjudication else None
                if verdict == "annotator_wrong":
                    ps = gs
                if ps in sub[gs]:
                    sub[gs][ps] += 1
        r_ = _pool(sub)
        r_["n_sections"] = sum(1 for s in per_section if s["level"] == lv)
        by_level[f"level{lv}"] = r_

    return {
        "per_class": pooled["per_class"],
        "confusion": cm,
        "macro_f1": pooled["macro_f1"],
        "status_acc": pooled["status_acc"],
        "n_judgments": pooled["n_judgments"],
        "by_level": by_level,
        "per_section": per_section,
        "disagreements": disagreements,
        "value_rows": value_rows,
        "value_acc": (sum(1 for r in value_rows if r["ok"]) / len(value_rows)
                      if value_rows else None),
        "missing_labels": missing_labels,
        "inert_verdicts": inert_verdicts,
    }


# -----------------------------------------------------------------------------
def render(r: Dict[str, Any], adjudicated: bool) -> str:
    L: List[str] = []
    L.append("=" * 78)
    L.append("真实方法段上的抽取质量（arXiv 语料）")
    L.append("=" * 78)
    L.append(f"判定总数 {r['n_judgments']}（{len(r['per_section'])} 篇 × 18 字段）")
    L.append(f"模式     : {'按裁定结果' if adjudicated else '原始一致率（未裁定）'}")
    L.append("")
    L.append(f"状态一致率      {r['status_acc']:.1%}")
    L.append(f"四态宏平均 F1   {r['macro_f1']:.3f}")
    L.append("")
    L.append("逐类（池化，n 为标注中该类出现的次数）:")
    for c in STATES:
        v = r["per_class"]
        L.append(f"    {c:16s} P {v[c]['precision']:.2f}  R {v[c]['recall']:.2f}  "
                 f"F1 {v[c]['f1']:.2f}   (n={v[c]['support']}, "
                 f"tp={v[c]['tp']} fp={v[c]['fp']} fn={v[c]['fn']})")
    L.append("")
    L.append("混淆矩阵（行=标注，列=工具）:")
    L.append(f"    {'':16s}" + "".join(f"{c[:9]:>11s}" for c in STATES) + f"{'absent':>11s}")
    for a in STATES:
        L.append(f"    {a:16s}" + "".join(f"{r['confusion'][a][b]:>11d}"
                                          for b in STATES + ["absent"]))
    L.append("")
    # 没有可比的取值时如实说"无"，不能报成 0%，0% 会被读成"取值全错"
    if r["value_acc"] is None:
        L.append("取值一致率      -（无可比取值）")
    else:
        L.append(f"取值一致率      {r['value_acc']:.1%}  (n={len(r['value_rows'])})")
    bad = [x for x in r["value_rows"] if not x["ok"]]
    for x in bad:
        L.append(f"    {x['id']} {x['field']}: 标注={x['gold']!r} 工具={x['got']!r}")

    L.append("")
    L.append("-" * 78)
    L.append("按章节层级分层（子章节天然可能漏掉正文别处写过的内容）")
    L.append("-" * 78)
    for lv, name in ((1, "level 1 完整章节"), (2, "level 2 子章节")):
        b = r["by_level"].get(f"level{lv}") or {}
        if not b.get("n_sections"):
            continue
        L.append(f"  {name}：{b['n_sections']} 篇 / {b['n_judgments']} 判定，"
                 f"一致率 {b['status_acc']:.1%}，四态宏 F1 {b['macro_f1']:.3f}")
        for c in STATES:
            v = b["per_class"][c]
            L.append(f"      {c:16s} P {v['precision']:.2f}  R {v['recall']:.2f}  "
                     f"F1 {v['f1']:.2f}   (n={v['support']})")

    L.append("")
    L.append("-" * 78)
    L.append("逐篇")
    L.append("-" * 78)
    for s in r["per_section"]:
        L.append(f"  {s['id']:20s} L{s['level']} 一致 {s['n_ok']:2d}/18 ({s['acc']:4.0%})  "
                 f"评分 {s['score']:3d}[{s['grade']}]  规则 {','.join(s['rules']) or '-'}")
        L.append(f"      {s['section'][:66]}")
        if s["ctx"]:
            L.append(f"      场景: {', '.join(s['ctx'])}")

    L.append("")
    L.append("-" * 78)
    L.append(f"分歧清单 {len(r['disagreements'])} 条（未裁定前，不得当作工具错误率）")
    L.append("-" * 78)
    for d in r["disagreements"]:
        v = f"  [裁定: {d['verdict']}]" if d["verdict"] else ""
        L.append(f"  {d['id']} {d['field']}: 标注={d['gold']} 工具={d['got']}{v}")
        if d["evidence"]:
            L.append(f"      原文: {d['evidence']}")
        if d["annotator_note"]:
            L.append(f"      标注备注: {d['annotator_note']}")
    if r["missing_labels"]:
        L.append("")
        L.append("!! 缺标注: " + "; ".join(r["missing_labels"]))

    # 悬空裁定必须出现在报告里。它意味着 adjudication.json 与当前工具版本
    # 已经对不上号，指标仍然算得出来，但"裁定"这一层的前提已经不成立。
    inert = r.get("inert_verdicts") or []
    if inert:
        L.append("")
        L.append("-" * 78)
        L.append(f"!! 悬空裁定 {len(inert)} 条（该对已一致，裁定作废未计入）")
        L.append("-" * 78)
        for x in inert:
            L.append(f"  {x['id']} {x['field']}: 原裁定 {x['verdict']}")
        L.append("  说明：裁定针对的是当时那一处分歧；工具改版后该处已一致，")
        L.append("        留着的旧裁定会把一个不存在的分歧算成错。请重跑裁定或删除。")
    L.append("=" * 78)
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="真实语料上的抽取质量评估")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--adjudicated", action="store_true",
                    help="按 _label_work/adjudication.json 的裁定结果重算")
    args = ap.parse_args(argv)

    labels = load_labels()
    if not labels:
        print(f"没有标注：{OUT} 为空。先跑标注。")
        return 2

    adj = None
    adj_path = os.path.join(WORK, "adjudication.json")
    if args.adjudicated:
        if not os.path.exists(adj_path):
            print(f"没有裁定文件 {adj_path}。先逐条复核分歧清单再重跑。")
            return 2
        adj = json.load(io.open(adj_path, encoding="utf-8"))

    r = run_all(labels, adjudication=adj)

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0

    print(render(r, args.adjudicated))
    return 0


if __name__ == "__main__":
    sys.exit(main())
