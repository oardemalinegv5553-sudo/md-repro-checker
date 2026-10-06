# -*- coding: utf-8 -*-
"""
score.py：在人工标注上度量抽取与校验的质量

为什么必须有这个文件：
  "能跑" 与 "跑得对" 是两件事。一个字段抽取器给出的结果 90% 是
  "missing"，它看起来也在正常工作。没有标注集，无法区分
  "这个工具确实发现了问题" 与 "这个工具什么都发现不了"。

四个指标，各自对应一类失败模式：

  1. 状态 F1（三态）
     把 present / missing / not_applicable 都当独立类别算。
     为什么不分组成二分类：missing 与 not_applicable 混淆正是
     假阳性的来源，合并计算会把这个错误掩盖掉。

  2. 取值准确率（仅统计 gold 给了值的字段）
     状态判对但值抓错，比状态判错更危险：错值不会被看见。
     cold_K=30、replicas=30 都属于这一类，必须单独度量。

  3. 一致性规则 F1 + 假阳性率（forbidden 命中数）
     假阳性率单独列出，因为它是使用者放弃工具的头号原因。

  4. 可执行率
     生成的全部输入文件中，通过语法检查且不含 unresolved 字段的比例。
     注意这一项横跨三套引擎：4 个 .mdp + 4 个 .lmp + 1 个 OpenMM 骨架，
     判定标准各不相同（.mdp 查参数名与重复键，.lmp 查命令名、重复 ID、
     冲突恒温、空转 run，OpenMM 走 Python 语法解析）。

用法：
    python eval/score.py                 # 跑两个案例并打印报告
    python eval/score.py --json          # 输出 JSON
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src import build_inputs, validate                    # noqa: E402
from src.extract import extract as run_extract            # noqa: E402


# -----------------------------------------------------------------------------
def _norm_value(v: Any) -> Any:
    """把值归一化到可比较形式。dict 只比 gold 里出现的键，允许抽取结果多带信息。"""
    if isinstance(v, str):
        return v.strip().lower()
    if isinstance(v, float):
        return round(v, 4)
    return v


def compare_value(gold: Any, got: Any) -> Tuple[bool, str]:
    if isinstance(gold, dict):
        if not isinstance(got, dict):
            return False, f"expected object, got {type(got).__name__}"
        for k, gv in gold.items():
            if k not in got:
                return False, f"missing key {k}"
            if _norm_value(gv) != _norm_value(got[k]):
                return False, f"{k}: gold={gv} got={got[k]}"
        return True, ""
    if _norm_value(gold) != _norm_value(got):
        return False, f"gold={gold} got={got}"
    return True, ""


# -----------------------------------------------------------------------------
def score_case(case: str, labels: Dict[str, Any]) -> Dict[str, Any]:
    gold = labels[case]
    path = os.path.join(ROOT, labels["_files"][case])
    text = io.open(path, encoding="utf-8").read()

    paths = validate.default_paths()
    schema = validate.load_yaml(paths["schema"])
    sensitivity = validate.load_yaml(paths["sensitivity"])

    extraction = run_extract(text, schema, backend="rule")
    validation = validate.validate(extraction, schema, sensitivity, raw_text=text)
    got_fields = extraction["fields"]

    # ---- 1. 状态三态 ----
    status_rows: List[Dict[str, Any]] = []
    for key, g in gold["fields"].items():
        gs = g["status"]
        got = got_fields.get(key) or {}
        ps = got.get("status", "absent")
        status_rows.append({"field": key, "gold": gs, "got": ps, "ok": gs == ps})

    # 三态混淆矩阵
    classes = ["present", "missing", "not_applicable"]
    cm = {a: {b: 0 for b in classes + ["absent"]} for a in classes}
    for r in status_rows:
        if r["got"] in cm[r["gold"]]:
            cm[r["gold"]][r["got"]] += 1

    # 宏平均 F1（三态各自算，再平均；避免 present 数量多而掩盖 N/A 的错误）
    #
    # 零样本类别必须排除在平均之外。若某版本标注中不含 missing 样本，
    # 该类的 0/0 会被算成 F1=0，把完美抽取压成 0.667：指标本身在撒谎。
    # 这与"某类召回确实为 0"是两回事，必须区分。
    per_class: Dict[str, Dict[str, float]] = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[o][c] for o in classes if o != c)
        fn = sum(cm[c][o] for o in cm[c] if o != c)
        support = sum(cm[c].values())
        p = tp / (tp + fp) if (tp + fp) else (1.0 if support == 0 else 0.0)
        rr = tp / (tp + fn) if (tp + fn) else (1.0 if support == 0 else 0.0)
        f1 = 2 * p * rr / (p + rr) if (p + rr) else 0.0
        per_class[c] = {"precision": p, "recall": rr, "f1": f1,
                        "support": support}
    scored = [v["f1"] for v in per_class.values() if v["support"] > 0]
    macro_f1 = sum(scored) / len(scored) if scored else 1.0
    status_acc = sum(1 for r in status_rows if r["ok"]) / len(status_rows)

    # ---- 2. 取值准确率 ----
    value_rows: List[Dict[str, Any]] = []
    for key, g in gold["fields"].items():
        if "value" not in g:
            continue
        got = got_fields.get(key) or {}
        ok, why = compare_value(g["value"], got.get("value"))
        value_rows.append({"field": key, "ok": ok, "why": why,
                           "gold": g["value"], "got": got.get("value")})
    value_acc = (sum(1 for r in value_rows if r["ok"]) / len(value_rows)
                 if value_rows else 1.0)

    # ---- 3. 规则 ----
    fired = {f["rule_id"] for f in validation["findings"]
             if f["kind"] == "consistency" and f.get("rule_id")}
    exp = set(gold["rules_expected"])
    forb = set(gold["rules_forbidden"])
    tp = len(fired & exp)
    fp = len(fired - exp)
    fn = len(exp - fired)
    # 无预期规则且无误报 = 完全正确，应记 1.0 而不是 0.0。
    # v2 是补全后的版本，本就不该触发任何规则，把它记为不及格是荒谬的。
    if not exp and not fired:
        p = rr = rule_f1 = 1.0
    else:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rr = tp / (tp + fn) if (tp + fn) else 0.0
        rule_f1 = 2 * p * rr / (p + rr) if (p + rr) else 0.0
    false_positives = sorted(fired & forb)
    missed = sorted(exp - fired)
    spurious = sorted(fired - exp)

    # ---- 4. 可执行率 ----
    b = build_inputs.build_all(extraction["fields"], os.path.join(
        ROOT, "eval", "_tmp", case), blocking=[])
    syntax_ok = sum(1 for c in b["checks"].values() if c["ok"])
    syntax_total = len(b["checks"])
    exec_ok = (syntax_ok == syntax_total) and not b["unresolved"]

    return {
        "case": case,
        "status_accuracy": status_acc,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "confusion": cm,
        "status_rows": status_rows,
        "value_accuracy": value_acc,
        "value_rows": value_rows,
        "rule_precision": p,
        "rule_recall": rr,
        "rule_f1": rule_f1,
        "false_positives": false_positives,
        "missed_rules": missed,
        "spurious_rules": spurious,
        "fired_rules": sorted(fired),
        "executability": {"syntax_ok": syntax_ok, "syntax_total": syntax_total,
                          "n_unresolved": len(b["unresolved"]),
                          "pass": exec_ok},
        "score": validation["score"],
        "grade": validation["grade"],
    }


# -----------------------------------------------------------------------------
def render(results: List[Dict[str, Any]]) -> str:
    L: List[str] = []
    L.append("=" * 74)
    L.append("抽取与校验质量评估")
    L.append("=" * 74)
    L.append("标注集: eval/labels/case-01.json（人工标注，18 字段 × 2 版本）")
    L.append("")

    for r in results:
        L.append("-" * 74)
        L.append(f"案例 {r['case']}")
        L.append("-" * 74)
        L.append(f"状态准确率      {r['status_accuracy']:.1%}"
                 f"    三态宏平均 F1  {r['macro_f1']:.3f}")
        for c in ("present", "missing", "not_applicable"):
            v = r["per_class"][c]
            L.append(f"    {c:16s} P {v['precision']:.2f}  R {v['recall']:.2f}  "
                     f"F1 {v['f1']:.2f}   (n={v['support']})")
        wrong = [x for x in r["status_rows"] if not x["ok"]]
        if wrong:
            L.append("    错误明细:")
            for x in wrong:
                L.append(f"      {x['field']}: gold={x['gold']} got={x['got']}")

        L.append(f"取值准确率      {r['value_accuracy']:.1%}"
                 f"    (仅统计标注给出值的字段)")
        bad = [x for x in r["value_rows"] if not x["ok"]]
        if bad:
            for x in bad:
                L.append(f"      {x['field']}: {x['why']}")

        L.append(f"一致性规则      P {r['rule_precision']:.2f}  R {r['rule_recall']:.2f}  "
                 f"F1 {r['rule_f1']:.2f}")
        L.append(f"    命中: {', '.join(r['fired_rules']) or '(无)'}")
        if r["missed_rules"]:
            L.append(f"    漏报: {', '.join(r['missed_rules'])}")
        if r["false_positives"]:
            L.append(f"    !! 假阳性: {', '.join(r['false_positives'])}")
        if r["spurious_rules"]:
            L.append(f"    未标注但触发: {', '.join(r['spurious_rules'])}")

        e = r["executability"]
        L.append(f"可执行率        {e['syntax_ok']}/{e['syntax_total']} 个生成文件通过语法检查 "
                 f"(.mdp/.lmp/OpenMM), "
                 f"unresolved {e['n_unresolved']} → {'通过' if e['pass'] else '未通过'}")
        L.append(f"评分            {r['score']} / 100  [{r['grade']}]")
        L.append("")

    # 汇总
    L.append("=" * 74)
    L.append("汇总")
    L.append("=" * 74)
    L.append(f"平均状态准确率  {sum(r['status_accuracy'] for r in results)/len(results):.1%}")
    L.append(f"平均三态宏 F1   {sum(r['macro_f1'] for r in results)/len(results):.3f}")
    L.append(f"平均取值准确率  {sum(r['value_accuracy'] for r in results)/len(results):.1%}")
    L.append(f"平均规则 F1     {sum(r['rule_f1'] for r in results)/len(results):.3f}")
    total_fp = sum(len(r['false_positives']) for r in results)
    L.append(f"假阳性总数      {total_fp}"
             f"{'  ← 假阳性是使用者放弃工具的首要原因，须优先消除' if total_fp else ''}")
    L.append("")
    L.append("说明：这是**合成集**的冒烟测试，不是准确率的证据：n=2 只能暴露")
    L.append("     错误类型；而且本集的标注与抽取出自同一作者，两边一致更多说明")
    L.append("     规则符合自己的预期，不说明它对真实文献管用。")
    L.append("")
    L.append("真实语料上的评估另有一轮，见 eval/real/README.md：24 篇真实方法段 /")
    L.append("     432 个判定，标注员只看得到方法段原文与 schema 生成的字段说明，")
    L.append("     看不到 src/。状态一致率 75.5%（原始）/ 78.7%（裁定后），")
    L.append("     四态宏平均 F1 0.706 / 0.762。")
    L.append("     该语料受版权限制不随包分发，因此本命令不重算这些数字。")
    L.append("=" * 74)
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="在人工标注上评估抽取质量")
    ap.add_argument("--labels", default=os.path.join(HERE, "labels", "case-01.json"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    labels = json.load(io.open(args.labels, encoding="utf-8"))
    results = [score_case(c, labels) for c in ("v1", "v2")]

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(render(results))

    # 有假阳性或漏报即返回非零，便于 CI 卡住回归
    bad = any(r["false_positives"] or r["missed_rules"] for r in results)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
