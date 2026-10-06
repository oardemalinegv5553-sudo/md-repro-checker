# -*- coding: utf-8 -*-
"""
fp_hunt.py：在真实方法段上排查假阳性

为什么要有这个脚本：
  case-01 的两段方法段是**我们写的**，规则在那上面全对，只证明规则符合
  我们自己的预期，不证明它能读真实文献。真实文献的写法千奇百怪，
  假阳性（把"本文体系不存在的问题"报成"缺失"）是使用者放弃工具的首要原因，
  而假阳性只有在真实文本上才会暴露。

核心难点：没有标注就没有 ground truth，怎么自动找出可疑项？
  答案是找一个**独立于被评估模块**的判据，词汇证据。
  例：一段方法段里如果从头到尾没出现过任何"水/溶剂/水合"词，
  那么"水模型缺失"这条就一定是假阳性，因为该体系根本不存在水模型。
  这个判据不看 extract.py 的 context 判定结果，只看向量和文本本身，
  因此可以用来**证伪** extract.py 的结论。

  三个语境字段（water_model / ligand_parameters / protonation_state）
  各自适用性明确：显式溶剂、含配体、含蛋白。词汇证据足以在
  "明确不适用"这一侧给出可靠判定，而假阳性恰好全部落在这一侧。

用法：
  python eval/real/fp_hunt.py                  # 全部方法段
  python eval/real/fp_hunt.py --only 2207.13405
  python eval/real/fp_hunt.py --json           # 机器可读
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.cli import run_pipeline          # noqa: E402

SECTIONS = os.path.join(HERE, "sections")

# -----------------------------------------------------------------------------
# 独立词汇判据
#
# 注意"隐式溶剂"要单独处理：MM/PBSA、GB 这类方法**没有**水模型，
# 把 "solvent" 一词的出现当成"存在水模型"会把隐式溶剂体系判反，
# 这正是下面必须把词表分正负两侧的原因。
# -----------------------------------------------------------------------------
VOCAB = {
    "water_explicit": [
        r"tip3p", r"tip4p", r"tip5p", r"spc/e?", r"\bopc\b", r"water model",
        r"explicit(?:ly)? solvat", r"explicit water", r"water box",
        r"solvat(?:ed|ion) (?:in|with) water", r"aqueous solution",
        r"sPC/E", r"three-site water", r"four-site water",
    ],
    "water_implicit": [
        r"implicit solvat", r"\bmm/?pbsa\b", r"\bmm/?gbsa\b",
        r"generalized born", r"poisson[- ]boltzmann", r"continuum solvent",
        r"\bgb\s*model\b", r"solvent accessible",
    ],
    "any_solvent_word": [
        r"\bwater\b", r"solvent", r"solvat", r"hydrat", r"aqueous",
    ],
    "protein": [
        r"\bprotein", r"\bpeptide", r"\bresidue", r"amino acid",
        r"\bhis\b", r"histidine", r"\basp\b", r"\bglu\b", r"\blys\b",
        r"\barg\b", r"\bcys\b", r"\bbackbone\b", r"secondary structure",
        r"\bpdb\b", r"alpha[- ]helix", r"beta[- ]sheet", r"\bunfold",
        r"\bwild[- ]type\b", r"\bmutant\b",
    ],
    "ligand": [
        # 刻意不收 \bsubstrate\b：材料学里指固体基底（silica substrate），
        # 收进来会让一批无机/界面论文被判成"有配体"。
        # 同样不收 gaff/cgenff/antechamber/am1-bcc，聚合物也用它们，
        # 它们说明"电荷怎么来的"，不说明"体系里有配体"。
        r"\bligand", r"\bdrug\b", r"\binhibitor", r"small molecule",
        r"\bdock", r"binding free energy",
    ],
    # 明确"不存在水模型"的体系描述
    "solvent_free_marker": [
        r"\bvacuum\b", r"gas phase", r"\bmelt\b", r"bulk polymer",
        r"\bcrystal\b", r"\bsolid\b", r"nanoparticle", r"graphene",
        r"\bcn[t]?\b", r"silica", r"\bslab\b", r"\bfilm\b",
    ],
}

# -----------------------------------------------------------------------------
# 语境字段 -> (适用性词汇, 不适用时该报什么)
# 判据方向：只有当"必然不适用"的证据充分时才算假阳性。
# 反过来（"没看到水，所以一定没有水"）不成立，
# 作者可能只是没写，那就是真的 missing。所以这里只做单侧判定。
# -----------------------------------------------------------------------------
GATED = [
    ("water_model", "water_explicit",
     "无任何显式水模型词汇，且上下文为无溶剂体系"),
    ("ligand_parameters", "ligand",
     "无任何配体/小分子词汇"),
    ("protonation_state", "protein",
     "无任何蛋白/残基词汇"),
]


def hits(text: str, key: str) -> List[str]:
    low = text.lower()
    return [p for p in VOCAB[key] if re.search(p, low)]


def triage(rec: Dict[str, Any]) -> Dict[str, Any]:
    path = os.path.join(HERE, rec["file"].replace("/", os.sep))
    text = io.open(path, encoding="utf-8").read()
    # 去掉我们自己加的头注释，只留正文，避免注释里的"水"字干扰判据
    body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))

    r = run_pipeline(body)
    fields = r["extraction"]["fields"]
    ctx = r["extraction"]["context"]
    findings = r["validation"]["findings"]

    status = {k: v.get("status") for k, v in fields.items()}
    fired = sorted({f.get("rule") for f in findings if f.get("rule")})

    suspects: List[str] = []

    # 假阳性判据 1：语境字段在不适用体系里被报 missing，
    for fld, vkey, why in GATED:
        if status.get(fld) != "missing":
            continue
        applicable_evidence = hits(body, vkey)
        if applicable_evidence:
            continue                    # 有适用证据，missing 是合理的
        # 水模型额外要求"确实是无溶剂体系"，否则可能是作者漏写
        if fld == "water_model":
            if hits(body, "water_implicit"):
                suspects.append(
                    "water_model=missing 但体系为**隐式溶剂**"
                    "（MM/PBSA/GB 类），隐式溶剂没有水模型，应判 not_applicable")
                continue
            if not ctx.get("solvent_free"):
                continue
        suspects.append(f"{fld}=missing 但全文无任何适用词汇（{why}）")

    # 关于 correlation_window 的一条**已撤回**的判据，
    # 曾经在这里判过："非 Green-Kubo 方法不应要求关联时间窗口"。
    # 在真实语料上核对后撤回了：规则 R-03 的原文是
    #   "热输运计算未报告**统计窗口**/关联时间，无法判断 κ 是否取自平台区"
    # 它本来就把 NEMD/RNEMD 的"稳态平均窗口"算在内，报告并无问题。
    # 换句话说，错的是这条评估判据，不是被评估的工具，
    # 假阳性排查工具自己也会假设错，所以每条判据都要在真数据上过一遍。

    if status.get("temperature_bath") == "missing" and not ctx.get("heat_transport"):
        suspects.append("temperature_bath=missing 但本文无热输运场景，该字段不适用")

    return {
        "id": rec["id"],
        "chars": rec.get("chars"),
        "section": rec.get("section"),
        "context": ctx,
        "status": status,
        "n_present": sum(1 for v in status.values() if v == "present"),
        "n_missing": sum(1 for v in status.values() if v == "missing"),
        "n_ambiguous": sum(1 for v in status.values() if v == "ambiguous"),
        "n_na": sum(1 for v in status.values() if v == "not_applicable"),
        "score": r["validation"]["score"],
        "grade": r["validation"]["grade"],
        "rules": fired,
        "suspects": suspects,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="真实方法段上的假阳性排查")
    ap.add_argument("--only", help="只跑指定 arXiv ID")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    man = json.load(open(os.path.join(HERE, "manifest.json"), encoding="utf-8"))
    recs = [r for r in man["records"] if r.get("ok")]
    if args.only:
        recs = [r for r in recs if r["id"] == args.only]
    if not recs:
        print("没有可用的方法段。先跑 fetch_arxiv.py。")
        return 2

    out = []
    for rec in recs:
        try:
            out.append(triage(rec))
        except Exception as e:
            print(f"  [!!] {rec['id']}: {type(e).__name__}: {e}")

    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    if not out:
        # 全部篇目都抛异常时，若无此判，下面会在算百分比时抛
        # ZeroDivisionError，真正的报错信息反而被它盖掉了。
        print("没有任何一篇跑成功，上面 [!] 行是真实原因。")
        return 1

    # 逐篇，
    print("=" * 100)
    print("逐篇结果")
    print("=" * 100)
    for t in out:
        flag = "  <<< 可疑" if t["suspects"] else ""
        print(f"\n{t['id']}  [{t['grade']}] {t['score']:3d}  "
              f"present {t['n_present']:2d} / missing {t['n_missing']:2d} / "
              f"amb {t['n_ambiguous']} / n-a {t['n_na']:2d}{flag}")
        print(f"   章节: {(t['section'] or '')[:70]}")
        ctx = t["context"]
        tags = [k for k, v in ctx.items() if v is True]
        if tags:
            print(f"   场景: {', '.join(tags)}")
        if t["rules"]:
            print(f"   规则: {', '.join(t['rules'])}")
        for s in t["suspects"]:
            print(f"   !! {s}")

    # 聚合，
    print("\n" + "=" * 100)
    print("聚合")
    print("=" * 100)
    n = len(out)
    print(f"篇数 {n}")

    rule_ct: Dict[str, int] = {}
    for t in out:
        for rr in t["rules"]:
            rule_ct[rr] = rule_ct.get(rr, 0) + 1
    print("\n规则触发次数（按次数降序）:")
    for rr, c in sorted(rule_ct.items(), key=lambda kv: -kv[1]):
        print(f"   {rr:6s} {c:3d} / {n}  ({c/n:.0%})")

    st_ct: Dict[str, Dict[str, int]] = {}
    for t in out:
        for k, v in t["status"].items():
            st_ct.setdefault(k, {}).setdefault(v, 0)
            st_ct[k][v] += 1
    print("\n各字段状态分布:")
    for k in sorted(st_ct):
        d = st_ct[k]
        print(f"   {k:24s} present {d.get('present',0):2d} "
              f"missing {d.get('missing',0):2d} "
              f"amb {d.get('ambiguous',0):2d} "
              f"n-a {d.get('not_applicable',0):2d}")

    ns = sum(1 for t in out if t["suspects"])
    print(f"\n**可疑项：{ns} / {n} 篇（{ns/n:.0%}）**")
    for t in out:
        for s in t["suspects"]:
            print(f"   {t['id']}: {s}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
