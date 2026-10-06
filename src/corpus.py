# -*- coding: utf-8 -*-
"""
corpus.py：跨文献可比性分析（流水线第五段）

    python -m src.corpus <方法段文件或目录>...  [--flag heat_transport] [--json]

前四段回答的是**一篇**文献的问题："这份描述够不够复现。"
本段把分析单元从「一篇」换成「一组」，回答另一个问题：

    这一组文献报告的同一个物理量，**彼此之间**可比吗？

这不是新的领域知识。sensitivity.yaml 里那些判断早就写好了：
"未报告恒温器 = 该 κ 值不可与其他文献横向比较"、"Green-Kubo / NEMD /
RNEMD 三者数值不可直接比较"。只是它们一直只被施加在单篇上。
本段做的是**换一个分析单元，把同一套知识重新施加一遍**。

三个刻意的设计决定
------------------

1. **只检查知识层写了影响规则的字段。**
   一组文献在 timestep 上有 4 种取值，但 sensitivity.yaml 没为它写
   conditional_impact，那就不报。有差异就报，等于又造一个劝退的
   假阳性工具。「宁可漏报不可误报」在这里的具体形态是：**没有依据的
   差异不构成发现。**

2. **规则的 when 条件必须对该组每一篇都成立，才纳入比较。**
   保守取交集，不做过度推断。

3. **字典取值按「公共键交集」比较。**
   {type: Langevin, coupling: 0.1} 与 {type: Langevin} 的交集是
   {type}，两边同为 Langevin，这只是**报告不全**，不构成协议分歧。
   而同为 {atoms: N} 但 1728 ≠ 8640，交集是 {atoms}，值不同，
   这是**真分歧**。同一条规则同时覆盖两种情形，无需为每个字段
   单写比较逻辑。

   （这里的两个数换了：原先举的是 1728 vs 54840，其中 54840 后来查明是
   `2207.13405` 的 NEP **训练集**规模，被抽取器当成了体系原子数——
   拿一个已知的错值当"真分歧"的范例，本身就是文档层面的静默失效。
   现取 1728 与 8640，两者都来自 `1805.00269v1` 真正模拟的体系。）
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import extract as extract_mod
from . import validate as V


# -----------------------------------------------------------------------------
# 取值比较
# -----------------------------------------------------------------------------
def _common_keys(values: Sequence[Any]) -> Optional[set]:
    """字典取值的公共键。全是标量时返回 None。"""
    dicts = [v for v in values if isinstance(v, dict)]
    if not dicts:
        return None
    common = set(dicts[0])
    for d in dicts[1:]:
        common &= set(d)
    return common


def comparison_key(value: Any, common: Optional[set]) -> str:
    """把取值规范化成一个可比较的字符串。

    common 为 None  -> 标量，直接转字符串
    common 为集合   -> 只取公共键，按键名排序后序列化
    """
    if common is None:
        return str(value)
    if not isinstance(value, dict):
        # 同一字段上标量与字典混用：保留原样，交给人去判断
        return str(value)
    sub = {k: value[k] for k in sorted(common) if k in value}
    return json.dumps(sub, ensure_ascii=False, sort_keys=True)


def _fmt(key: str, papers: Sequence[str]) -> str:
    return f"{key}  ({len(papers)} 篇)"


# -----------------------------------------------------------------------------
# 主分析
# -----------------------------------------------------------------------------
def analyse(papers: Sequence[Tuple[str, Dict[str, Any]]],
            schema: Dict[str, Any],
            sensitivity: Dict[str, Any],
            flag: str) -> Dict[str, Any]:
    """papers: [(论文 id, extract() 的输出), ...]，已按 flag 过滤。"""
    n_total = len(papers)
    fieldsets = [p[1].get("fields") or {} for p in papers]
    ids = [p[0] for p in papers]

    # 组级上下文：只有对该组**每一篇**都成立的 when 条件才纳入
    ctxs = [V.build_effective_context(p[1]) for p in papers]

    all_fields: List[Dict[str, Any]] = list(schema.get("fields", []))
    all_fields += list(schema.get("heat_transport_extension", []))
    by_key = {f["key"]: f for f in all_fields}

    conflicts: List[Dict[str, Any]] = []
    checked: List[str] = []

    for rule in sensitivity.get("conditional_impacts", []):
        cond = rule.get("when") or {}
        # 条件必须对该组每一篇都成立
        if cond and not all(all(c.get(k) == v for k, v in cond.items()) for c in ctxs):
            continue

        fkey = rule.get("field")
        if not fkey:
            continue
        checked.append(fkey)

        statuses = [(f.get(fkey) or {}) for f in fieldsets]
        present = [(i, s.get("value")) for i, s in zip(ids, statuses)
                   if s.get("status") == "present" and s.get("value") is not None]
        n_present, n_missing = len(present), n_total - len(present)

        spec = by_key.get(fkey, {})
        field_cn = spec.get("name_cn", fkey)

        if n_present == 0:
            # 全组都没有报告，单篇工具永远看不到，只有成组才显现
            conflicts.append({
                "field": fkey, "field_cn": field_cn,
                "kind": "collective_absence",
                "severity": rule.get("severity", "medium"),
                "variants": [],
                "n_present": 0, "n_missing": n_missing, "n_total": n_total,
                "impact": (rule.get("impact") or "").strip(),
                "action": (rule.get("action") or "").strip(),
            })
            continue

        common = _common_keys([v for _, v in present])
        buckets: Dict[str, List[str]] = {}
        for i, v in present:
            buckets.setdefault(comparison_key(v, common), []).append(i)

        if len(buckets) >= 2:
            conflicts.append({
                "field": fkey, "field_cn": field_cn,
                "kind": "disagreement",
                "severity": rule.get("severity", "medium"),
                "variants": [{"key": k, "papers": sorted(v)}
                             for k, v in sorted(buckets.items())],
                "compared_on": sorted(common) if common else None,
                "n_present": n_present, "n_missing": n_missing, "n_total": n_total,
                "impact": (rule.get("impact") or "").strip(),
                "action": (rule.get("action") or "").strip(),
            })
        elif n_missing:
            # 取值一致，但不是每篇都报了，沉默的那几篇无从判断
            conflicts.append({
                "field": fkey, "field_cn": field_cn,
                "kind": "incomplete_report",
                "severity": rule.get("severity", "medium"),
                "variants": [{"key": k, "papers": sorted(v)}
                             for k, v in sorted(buckets.items())],
                "compared_on": sorted(common) if common else None,
                "n_present": n_present, "n_missing": n_missing, "n_total": n_total,
                "impact": (rule.get("impact") or "").strip(),
                "action": (rule.get("action") or "").strip(),
            })

    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    conflicts.sort(key=lambda c: (order.get(c["severity"], 9), c["field"]))

    blockers = [c for c in conflicts if c["severity"] in ("critical", "high")]
    return {
        "flag": flag,
        "n_papers": n_total,
        "papers": sorted(ids),
        "fields_checked": sorted(set(checked)),
        "conflicts": conflicts,
        "summary": {
            "n_conflicts": len(conflicts),
            "n_blocking": len(blockers),
            "comparable": not blockers,
            "by_kind": {k: sum(1 for c in conflicts if c["kind"] == k)
                        for k in ("disagreement", "collective_absence",
                                  "incomplete_report")},
        },
    }


def group(papers: Sequence[Tuple[str, Dict[str, Any]]], flag: str
          ) -> List[Tuple[str, Dict[str, Any]]]:
    return [(i, e) for i, e in papers
            if (V.build_effective_context(e).get(flag) is True)]


# -----------------------------------------------------------------------------
# 渲染
# -----------------------------------------------------------------------------
KIND_CN = {
    "disagreement":       "取值分歧",
    "collective_absence": "集体缺失",
    "incomplete_report":  "报告不全",
}


def render(res: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("=" * 74)
    L.append("跨文献可比性分析")
    L.append("=" * 74)
    L.append(f"分组依据: {res['flag']}      论文数: {res['n_papers']}")
    # 不足 2 篇时**不能**给出"可比"的结论：那是把"没有数据"说成"没有问题"，
    # 正是本项目要抓的那种静默误导。没有可比较的对象，就没有结论。
    if res["n_papers"] < 2:
        L.append(f"该分组只有 {res['n_papers']} 篇，不足以做跨文献比较。")
        L.append("跨文献比较的分析单元是「一组」，至少需要 2 篇。")
        L.append("=" * 74)
        return "\n".join(L)

    L.append(f"参与比较的字段（知识层为其写了影响规则、且对该组每篇都适用）:")
    L.append("    " + (", ".join(res["fields_checked"]) or "（无）"))
    L.append("")

    if not res["conflicts"]:
        L.append("未发现影响可比性的协议差异。")
    for c in res["conflicts"]:
        tag = V.SEV_TAG.get(c["severity"], "  ")
        L.append("-" * 74)
        L.append(f"{tag} [{KIND_CN[c['kind']]}] {c['field_cn']} ({c['field']})")
        L.append(f"    报告 {c['n_present']}/{c['n_total']} 篇"
                 + (f"，缺失 {c['n_missing']} 篇" if c["n_missing"] else ""))
        if c.get("compared_on"):
            L.append(f"    比较依据的公共键: {', '.join(c['compared_on'])}")
        for v in c["variants"]:
            L.append(f"      · {_fmt(v['key'], v['papers'])}")
        if c["impact"]:
            L.append("")
            for line in V._wrap(c["impact"], 62):
                L.append("      " + line)
        if c["action"]:
            L.append(f"      → {c['action']}")
        L.append("")

    s = res["summary"]
    L.append("=" * 74)
    verdict = "可比" if s["comparable"] else "不可横向比较"
    L.append(f"结论: {verdict}")
    L.append(f"      冲突 {s['n_conflicts']} 项"
             f"（取值分歧 {s['by_kind']['disagreement']} · "
             f"集体缺失 {s['by_kind']['collective_absence']} · "
             f"报告不全 {s['by_kind']['incomplete_report']}）"
             f"，其中阻断可比性 {s['n_blocking']} 项")
    if not s["comparable"]:
        L.append("      即：这一组文献即使报告了同一个物理量，其数值也不构成")
        L.append("      一组可相互比较的数据点。把它们并列或做元分析前需先对齐协议。")
    L.append("=" * 74)
    return "\n".join(L)


# -----------------------------------------------------------------------------
# 演示（合成数据，不依赖任何真实语料；语料受版权保护，不分发）
# -----------------------------------------------------------------------------
DEMO = [
    ("demo-01", "Green-Kubo，恒温器给了类型和耦合常数，给了关联窗口"),
    ("demo-02", "RNEMD，恒温器完全没写，没给关联窗口"),
    ("demo-03", "Green-Kubo，恒温器只给了类型没给耦合常数"),
    ("demo-04", "Direct NEMD，恒温器没写，体系尺寸比别的篇大 30 倍"),
    ("demo-05", "Green-Kubo，恒温器给了类型和耦合常数，没给体系尺寸"),
]


def _demo_papers() -> List[Tuple[str, Dict[str, Any]]]:
    """构造一组合成的抽取结果。

    刻意做成「同一个物理量、协议却对不齐」的样子，这正是真实语料里
    最常见的情形。全部为合成值，不含任何真实文献数据。
    """
    def mk(written: bool, ff: str, flux: str, thermo: Optional[Dict[str, str]],
           size: Optional[int], window: Optional[Dict[str, float]]) -> Dict[str, Any]:
        def f(v, status="present"):
            return {"value": v, "status": status, "confidence": "high",
                    "evidence": {"quote": "", "offset": -1}}
        def na():
            return {"value": None, "status": "not_applicable", "confidence": "high",
                    "evidence": {"quote": "", "offset": -1}}
        def miss():
            return {"value": None, "status": "missing", "confidence": "high",
                    "evidence": {"quote": "", "offset": -1}}
        fields = {
            "force_field":     f(ff) if written else miss(),
            "heat_flux_method": f(flux),
            "thermostat":      f(thermo) if thermo else miss(),
            "system_size":     f({"atoms": size}) if size else miss(),
            "correlation_window": f(window) if window else miss(),
            "temperature_bath": miss(),
            "timestep":        f(1.0),
            "ensemble":        f("NVT"),
            "simulation_length": f({"ns": 20.0, "replicas": 3}),
            "water_model":     na(),
            "electrostatics":  f({"method": "PME", "rcoulomb_nm": 1.2}),
            "dispersion_correction": f("EnerPres"),
            "initial_structure": f("crystal"),
            "barostat":        na(),
            "constraints":     f("LINCS"),
        }
        return {"fields": fields,
                "context": {"heat_transport": True, "explicit_solvent": False}}

    # 大体系取 51840 = 1728 × 30，与 DEMO 说明里的"大 30 倍"严格对上。
    # 早先用的是 54840——那恰好是真实语料里 2207.13405 的 NEP **训练集**
    # 规模，也是抽取器曾误当成体系原子数的那个错值。演示里摆一个
    # 已知错值当"另一个体系"，读者会以为它就是 §3.7 讲的那个数，
    # 而这个演示恰恰是在说明"这两篇真的不一样大"。合成数据就该长得像合成数据。
    return [
        ("demo-01", mk(True,  "GAFF2", "Green-Kubo", {"type": "Nose-Hoover", "coupling": "0.1"}, 1728,  {"ps": 2.0})),
        ("demo-02", mk(False, "",     "RNEMD (Muller-Plathe)", None, 51840, None)),
        ("demo-03", mk(True,  "GAFF2", "Green-Kubo", {"type": "Nose-Hoover"}, 1728, {"ps": 2.0})),
        ("demo-04", mk(False, "",     "Direct NEMD", None, 51840, None)),
        ("demo-05", mk(True,  "OPLS-AA", "Green-Kubo", {"type": "Langevin", "coupling": "0.5"}, None, {"ps": 5.0})),
    ]


# -----------------------------------------------------------------------------
def load_papers(paths: Sequence[str]) -> List[Tuple[str, Dict[str, Any]]]:
    schema = V.load_yaml(V.default_paths()["schema"])
    files: List[str] = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "*.txt")))
        else:
            files.append(p)
    out: List[Tuple[str, Dict[str, Any]]] = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError as e:
            print(f"  !! 读不到 {f}: {e}", file=sys.stderr)
            continue
        try:
            out.append((os.path.basename(f).replace(".txt", ""),
                        extract_mod.extract(text, schema, backend="rule")))
        except Exception as e:                            # noqa: BLE001
            print(f"  !! 抽取失败 {f}: {e}", file=sys.stderr)
    return out


def run(papers: Sequence[Tuple[str, Dict[str, Any]]],
        flag: Optional[str] = None) -> Dict[str, Any]:
    paths = V.default_paths()
    schema = V.load_yaml(paths["schema"])
    sensitivity = V.load_yaml(paths["sensitivity"])

    if flag is None:
        # 自动挑最大的组
        best, best_n = None, 0
        for cand in ("heat_transport", "protein_present", "ligand_present",
                     "explicit_solvent"):
            n = len(group(papers, cand))
            if n > best_n:
                best, best_n = cand, n
        flag = best or "heat_transport"

    grp = group(papers, flag)
    if len(grp) < 2:
        # comparable 用 None 而非 True：缺数据不等于通过。
        return {"flag": flag, "n_papers": len(grp), "papers": [i for i, _ in grp],
                "fields_checked": [], "conflicts": [],
                "summary": {"n_conflicts": 0, "n_blocking": 0, "comparable": None,
                            "by_kind": {"disagreement": 0, "collective_absence": 0,
                                        "incomplete_report": 0}},
                "error": f"分组 {flag} 只有 {len(grp)} 篇，不足以做跨文献比较"}
    return analyse(grp, schema, sensitivity, flag)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="跨文献可比性分析")
    ap.add_argument("paths", nargs="*", help="方法段文件或目录")
    ap.add_argument("--flag", help="按哪个场景标志分组（默认自动挑最大的组）")
    ap.add_argument("--demo", action="store_true", help="用合成分组演示")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    if args.demo:
        print("（演示使用**合成的**抽取结果，不含任何真实文献数据）\n")
        papers = _demo_papers()
        for pid, note in DEMO:
            print(f"  {pid}  {note}")
        print()
        res = run(papers, args.flag or "heat_transport")
    else:
        if not args.paths:
            ap.error("需要给出方法段文件或目录（或用 --demo）")
        papers = load_papers(args.paths)
        if not papers:
            print("没有可分析的输入。", file=sys.stderr)
            return 2
        res = run(papers, args.flag)

    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(render(res))
    if res.get("error"):
        print(f"\n注: {res['error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
