# -*- coding: utf-8 -*-
"""
validate.py：缺失/歧义标注、跨字段一致性检查、参数敏感性归因、可复现性评分

第三段（extract -> validate -> build_inputs）。
功能列表：

  1. 跨字段一致性检查（R-01..R-12）
     交叉检查字段之间相矛盾的错误。

  2. 条件敏感性归因
     分析字段缺失的严重程度。

  3. 可复现性评分
     把 flag 汇总成可横向比较的数字。

  4. 证据留存
     每条 finding 都带原文引用或"未找到"的明确说明，便于人工复核。
"""
from __future__ import annotations

import os
from typing import Any, Dict, List
from .extract import has_titratable_groups

try:
    import yaml
except ImportError:                                   # pragma: no cover
    yaml = None


# 一致性规则实现
#   每条规则：(id, 触发条件(ctx, fields) -> bool, 检查(fields) -> bool, 消息, severity)
#   返回 True 表示"发现问题"
def _val(fields: Dict[str, Any], key: str) -> Any:
    f = fields.get(key) or {}
    return f.get("value")


def _present(fields: Dict[str, Any], key: str) -> bool:
    return (fields.get(key) or {}).get("status") == "present"


def _rule_r01(ctx, f):
    if _val(f, "ensemble") != "NPT":
        return False
    return not _present(f, "barostat")


def _rule_r02(ctx, f):
    if _val(f, "ensemble") != "NPT":
        return False
    b = _val(f, "barostat")
    return isinstance(b, dict) and not b.get("coupling")


def _rule_r12(ctx, f):

    th = _val(f, "thermostat")
    if not isinstance(th, dict):
        return False
    return bool(th.get("type")) and not th.get("coupling")


def _rule_r03(ctx, f):
    if not ctx.get("heat_transport"):
        return False
    return not _present(f, "correlation_window")


def _rule_r04(ctx, f):
    ts = _val(f, "timestep")
    if not isinstance(ts, (int, float)) or ts < 4.0:
        return False
    # 方法段是否提及氢质量重分配
    return not ctx.get("h_mass_repartitioning", False)


def _rule_r05(ctx, f):
    if not ctx.get("heat_transport"):
        return False
    return not _present(f, "system_size")


def _rule_r06(ctx, f):
    if not ctx.get("explicit_solvent"):
        return False
    return not _present(f, "water_model")


def _rule_r07(ctx, f):
    if not ctx.get("ligand_present"):
        return False
    return not _present(f, "ligand_parameters")


def _rule_r08(ctx, f):
    sl = _val(f, "simulation_length")
    if not isinstance(sl, dict):
        return False
    return not sl.get("replicas")


def _rule_r09(ctx, f):
    # 与字段闸门共用 extract.has_titratable_groups 这一个定义处，
    # 否则会出现"字段判不适用、规则却报缺失"的自相矛盾输出。
    if not has_titratable_groups(ctx):
        return False
    return not _present(f, "protonation_state")


def _rule_r10(ctx, f):
    es = _val(f, "electrostatics")
    if not isinstance(es, dict) or es.get("method") != "PME":
        return False
    return es.get("rcoulomb_nm") is None


def _rule_r11(ctx, f):
    """
    NEMD 线性响应：ΔT 相对平均温度过大时，κ 依赖于 ΔT 本身，失去材料常数意义。
    """
    if not ctx.get("heat_transport"):
        return False
    tb = _val(f, "temperature_bath")
    if not isinstance(tb, dict):
        return False
    hot, cold = tb.get("hot_K"), tb.get("cold_K")
    if hot is None or cold is None:
        return False
    t_avg = 0.5 * (hot + cold)
    if t_avg <= 0:
        return False
    return (hot - cold) / t_avg > 0.10          # 经验阈值：>10% 通常已偏离线性区


RULES = [
    ("R-01", _rule_r01, "声明 NPT 系综但未报告恒压器：体系状态未定义，无法复现",
     "critical"),
    ("R-02", _rule_r02, "报告了恒压器类型但未给耦合常数 tau_p：压力耦合强度不可复现",
     "important"),
    ("R-03", _rule_r03, "热输运计算未报告统计窗口/关联时间：无法判断 κ 是否取自平台区",
     "critical"),
    ("R-04", _rule_r04, "时间步 ≥ 4 fs 但未提及氢质量重分配，内部不一致，需人工确认",
     "important"),
    ("R-05", _rule_r05, "热输运计算未报告体系尺寸：无法评估有限尺寸效应",
     "important"),
    ("R-06", _rule_r06, "显式溶剂体系未报告水模型",
     "critical"),
    ("R-07", _rule_r07, "体系含配体但未报告参数化方法与电荷来源",
     "critical"),
    ("R-08", _rule_r08, "报告了模拟时长但未说明重复次数：统计误差无法评估",
     "important"),
    ("R-09", _rule_r09, "含蛋白体系未说明可滴定残基（尤其 HIS）的质子化态",
     "critical"),
    ("R-10", _rule_r10, "使用 PME 但未给出实空间截断半径 rcoulomb",
     "important"),
    ("R-11", _rule_r11, "NEMD 冷热浴温差相对平均温度超过 10%：可能已偏离线性响应区，"
                        "此时 κ 依赖于 ΔT 本身，不再是材料常数",
     "important"),
    ("R-12", _rule_r12, "报告了恒温器类型但未给耦合常数 tau_t：恒温强度不可复现，"
                        "对扩散/热导/粘度等输运性质可造成 10-30% 偏差",
     "important"),
]


# 场景增强：把抽取结果并进 context，供 sensitivity.yaml 的 when 条件使用
def build_effective_context(extraction: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
    ctx = dict(extraction.get("context") or {})
    fields = extraction.get("fields") or {}

    ens = _val(fields, "ensemble")
    if ens:
        ctx["ensemble"] = ens

    es = _val(fields, "electrostatics")
    if isinstance(es, dict):
        ctx["electrostatics_method"] = es.get("method")

    # 氢质量重分配：常见写法 h-mass repartitioning / 4 fs with H-mass
    import re
    if re.search(r"mass\s+repartition|h[- ]?mass|氢质量|heavy[- ]hydrogen|4\s*fs", raw_text, re.I):
        ctx["h_mass_repartitioning"] = True

    return ctx


# 主校验函数
def validate(extraction: Dict[str, Any], schema: Dict[str, Any],
             sensitivity: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
    fields = extraction.get("fields") or {}
    ctx = build_effective_context(extraction, raw_text)

    all_fields: List[Dict[str, Any]] = list(schema.get("fields", []))
    all_fields += list(schema.get("heat_transport_extension", []))
    by_key = {f["key"]: f for f in all_fields}

    findings: List[Dict[str, Any]] = []

    # --- 1. 缺失 / 歧义 ---
    for key, res in fields.items():
        if key.startswith("_"):
            continue
        status = res.get("status")
        spec = by_key.get(key, {})
        sev = spec.get("severity", "important")
        if status == "missing":
            findings.append({
                "kind": "missing",
                "field": key,
                "field_cn": spec.get("name_cn", key),
                "severity": sev,
                "message": f"未报告 {spec.get('name_cn', key)}",
                "why": (spec.get("why") or "").strip(),
                "action": "",
                "evidence": {"quote": "", "offset": -1},
            })
        elif status == "ambiguous":
            findings.append({
                "kind": "ambiguous",
                "field": key,
                "field_cn": spec.get("name_cn", key),
                "severity": sev,
                "message": f"{spec.get('name_cn', key)} 描述模糊，存在多义",
                "why": (spec.get("why") or "").strip(),
                "action": "",
                "evidence": res.get("evidence") or {},
            })

    # --- 2. 跨字段一致性 ---
    for rid, fn, msg, sev in RULES:
        try:
            if fn(ctx, fields):
                findings.append({
                    "kind": "consistency",
                    "rule_id": rid,
                    "field": "",
                    "field_cn": "",
                    "severity": sev,
                    "message": msg,
                    "why": "",
                    "action": "",
                    "evidence": {},
                })
        except Exception as e:                        # noqa: BLE001
            findings.append({"kind": "error", "rule_id": rid, "message": str(e),
                             "severity": "info"})

    # --- 3. 条件敏感性归因 ---
    for rule in sensitivity.get("conditional_impacts", []):
        cond = rule.get("when") or {}
        if cond and not all(ctx.get(k) == v for k, v in cond.items()):
            continue
        fkey = rule.get("field")
        if _present(fields, fkey):
            continue                                   # 有值就不归因
        spec = by_key.get(fkey, {})
        findings.append({
            "kind": "impact",
            "field": fkey,
            "field_cn": spec.get("name_cn", fkey),
            "severity": rule.get("severity", "medium"),
            "message": f"{spec.get('name_cn', fkey)} 在本场景下是高影响字段",
            "why": (rule.get("impact") or "").strip(),
            "action": (rule.get("action") or "").strip(),
            "evidence": {},
        })

    # --- 4. 评分 ---
    sc = sensitivity.get("scoring", {})
    pen = sc.get("penalty", {})
    score = sc.get("base", 100)
    for f in findings:
        k = f.get("kind")
        sev = f.get("severity")
        if k == "missing":
            score -= pen.get(f"missing_{sev}", 0)
        elif k == "ambiguous":
            score -= pen.get("ambiguous", 0)
        elif k == "consistency":
            score -= pen.get("consistency_violation", 0)
    score = max(0, min(100, score))

    grade = "D"
    for g, thr in sorted(sc.get("grade_cutoffs", {}).items(),
                         key=lambda kv: kv[1], reverse=True):
        if score >= thr:
            grade = g
            break

    return {
        "context": ctx,
        "score": score,
        "grade": grade,
        "findings": findings,
        "summary": {
            "n_missing": sum(1 for f in findings if f["kind"] == "missing"),
            "n_ambiguous": sum(1 for f in findings if f["kind"] == "ambiguous"),
            "n_consistency": sum(1 for f in findings if f["kind"] == "consistency"),
            "n_impact": sum(1 for f in findings if f["kind"] == "impact"),
            "n_critical": sum(1 for f in findings if f["severity"] == "critical"),
        },
    }


# 输出：可读的检查单
SEV_TAG = {"critical": "!!", "high": "! ", "important": "! ", "medium": "- ", "info": "  "}


def render_report(extraction: Dict[str, Any], validation: Dict[str, Any]) -> str:
    L: List[str] = []
    meta = extraction.get("meta", {})
    s = validation["summary"]

    L.append("=" * 74)
    L.append("MD 方法段可复现性检查单")
    L.append("=" * 74)
    # 请求的后端与实际生效的后端可能不是一回事：llm/hybrid 层整体失败时，
    # 结果全部来自规则层。只报请求值等于谎报来源，必须两条都报。
    _req = meta.get("backend")
    _eff = meta.get("backend_effective", _req)
    L.append(f"抽取方式      : {_req}")
    if _eff and _req and _eff != _req:
        L.append(f"!! 该后端本次未生效，以下结果全部来自 {_eff} 层"
                 f"（请求 {_req}，实际 {_eff}）")
        _err = (meta.get("llm_error") or "").strip()
        if _err:
            for _ln in _wrap(f"原因: {_err}", 68):
                L.append(f"     {_ln}")
    L.append(f"方法段长度    : {meta.get('methods_chars')} 字符")
    L.append(f"字段命中      : present {meta.get('n_present')} / "
             f"missing {meta.get('n_missing')} / ambiguous {meta.get('n_ambiguous')} / "
             f"n-a {meta.get('n_na')}")
    ctx = validation.get("context", {})
    on = [k for k, v in ctx.items() if v is True]
    L.append(f"识别场景      : {', '.join(on) if on else '(无特殊场景)'}")
    L.append("")
    L.append(f"可复现性评分  : {validation['score']} / 100   [等级 {validation['grade']}]")
    L.append(f"  严重 {s['n_critical']} · 一致性 {s['n_consistency']} · "
             f"缺失 {s['n_missing']} · 歧义 {s['n_ambiguous']} · 影响归因 {s['n_impact']}")
    L.append("")

    by_kind = {"consistency": "一致性冲突", "missing": "字段缺失",
               "ambiguous": "字段歧义", "impact": "场景影响归因"}
    for kind, title in by_kind.items():
        items = [f for f in validation["findings"] if f["kind"] == kind]
        if not items:
            continue
        L.append("-" * 74)
        L.append(f"【{title}】")
        L.append("-" * 74)
        for f in items:
            tag = SEV_TAG.get(f["severity"], "  ")
            head = f"[{f['rule_id']}] " if f.get("rule_id") else ""
            L.append(f"{tag} {head}{f['message']}")
            if f.get("why"):
                for line in _wrap(f["why"], 70):
                    L.append(f"     {line}")
            if f.get("action"):
                L.append(f"     → {f['action']}")
            q = (f.get("evidence") or {}).get("quote")
            if q:
                L.append(f"     原文: “{q[:110]}”")
            L.append("")

    L.append("=" * 74)
    L.append("说明：评分为启发式指标，用于同一批文献内部的相对比较，非绝对标准。")
    L.append("=" * 74)
    return "\n".join(L)


def _wrap(text: str, width: int) -> List[str]:
    text = " ".join(text.split())
    out, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= width and ch in "。；，、 ":
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out or [""]


# -----------------------------------------------------------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("需要 pyyaml：pip install pyyaml")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def default_paths() -> Dict[str, str]:
    here = os.path.dirname(os.path.abspath(__file__))
    schema_dir = os.path.join(os.path.dirname(here), "schema")
    return {
        "schema": os.path.join(schema_dir, "md_method.yaml"),
        "sensitivity": os.path.join(schema_dir, "sensitivity.yaml"),
    }
