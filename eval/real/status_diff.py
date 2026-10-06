# -*- coding: utf-8 -*-
"""
status_diff.py：逐格对账，改动前后的状态差异**对着标注**分类

为什么要有这个脚本（而不是改完直接看 score_real 的总分）：
  总分会把两类完全不同的变化压成同一个数：
    · 改对了
    · 换了一种错法（旧错新错）：总分可能纹丝不动，甚至正好抵消
  修掉一个假阳性不等于修对了：可能只是把它换成了另一种错，
  而总分对这两种情况给出同一个数字。
  所以本脚本按**每一格**分类，并把 neutral（旧错新错）显式打印出来：
  它最容易被总分掩盖。

三类：
  fixed    gold == new and gold != old     改对了
  broken   gold == old and gold != new     原来是对的，改坏了  ← 唯一的红线
  neutral  其余（旧错新错，含 gold=present → ambiguous）

为什么"对着标注"而不是"对着上一次运行"：
  对着上一次运行只能回答"变了没有"，回答不了"变好了没有"。
  同一格从 missing 变成 ambiguous：标注是 ambiguous 就是修复，
  标注是 missing 就是改坏。同样一次变化，方向相反。

裁定优先于标注（关键）：
  有 14 格的裁定结论是 annotator_wrong：那几格**标注本身是错的**，
  工具当时是对的。若还拿标注当基准，把工具改成"迎合错标注"会被记成
  fixed，而真正的回归反而看着没问题。所以这几格改用
  "上一次工具自己的答案"当基准：任何改动都记 broken。

用法：
  python eval/real/status_diff.py --snapshot --tag baseline
  python eval/real/status_diff.py --diff baseline fix-ens
  python eval/real/status_diff.py --diff baseline fix-ens --field ensemble

退出码：
  0  可接受（broken == 0，硬约束全绿）
  1  broken > 0
  2  硬约束失败（ctx 漂移 / 规则漂移 / 变动格子获得了取值 / 不确定 / 快照不存在）
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                          # noqa: BLE001
    pass

import score_real as S                                     # noqa: E402

SNAP_DIR = os.path.join(HERE, "_scratch", "status")
ADJ_PATH = os.path.join(HERE, "_label_work", "adjudication.json")
EXIT_NAMES = ("left_present_or_na", "ctx_drift", "ctx_derived", "ctx_declared",
              "rule_drift", "rule_drift_explained", "value_state",
              "nondeterministic", "stale_snapshot")
CATS = ("broken", "disputed", "neutral", "fixed")

# 哪些 ctx 键是**字段值的镜像**：validate.build_effective_context 会把
# 字段层的取值抄进 ctx（ensemble 取字段值，electrostatics_method 取
# electronstatics.method）。这类搬动是纯函数，不引入新信息，
# 所以可以把"ctx 未漂移"放宽成"ctx 只能是其字段的镜像"：
# 键的旧值与字段旧值一致、新值与字段新值一致，才算镜像。
# 任何别的键（solvent_free / protein_present / has_bonds …）动了仍是致命的：
# 那 7 条闸门读的是它们，而且它们由 detect_context 扫原文得出，
# 与字段层无关，动了就说明文本判读变了。
_CTX_FROM_FIELD = {
    "ensemble": ("ensemble", lambda v: v if v else None),
    "electrostatics_method": (
        "electrostatics",
        lambda v: v.get("method") if isinstance(v, dict) else None),
}

# 已声明并复核过的 ctx 搬动。每条的准入条件都是可当场复核的事实，不是
# "我保证看过"：表里写死了**旧值、新值、理由**三件，三者全中才认；
# 差一个（键以别的方式变了、或值不是这个方向）一律照旧报致命漂移。
# 与镜像表的区别：镜像表说的是"这个键是字段的影子，动是被字段带着动"，
# 这张表说的是"这个键的判读本身被我**有意**改了，理由是实测出来的"。
# 两种都不致命，但都必须逐条打印，因为"改判读"和"改错了"在快照层面
# 长得完全一样，只有让眼睛过一遍才能分开。
#
# new = None 表示该键被**退役**（从 detect_context 摘掉）。
#
# 形状是**一张列表**而不是"键 -> 一次搬动"：同一个键可以按不同方式动
# （新增的键在不同篇里取不同的值），每次搬动各自是一条、各自带理由。
# 每条仍然是 (键, 旧值, 新值, 理由) 四件，四件全中才认。
# 旧值为 None 有两种含义，看新值就知道是哪种：新值非 None = 键**新增**，
# 新值为 None = 键**退役**。
DECLARED_CTX = [
    ("has_temperature_grade", False, None,
     "退役：只写不读。全仓唯一出现处是 detect_context 的写点；"
     "温度浴闸门读的是 nemd，不是它。留着的唯一效果是任何一次对"
     "_extract_temperature_bath 的改动都报成 ctx 漂移。"),
    ("nemd", False, True,
     "闸门词表补上「两端热浴」写法（left/right heat baths applied to "
     "regions，1007.5359v2）。该写法全语料只有 1 篇，且语义上"
     "只可能是 NEMD：热浴加在两端区域正是为了建立温度梯度。"),
    ("has_bonds", True, False,
     "反应族 pattern 收了散文写法 reactive MD（1812.03256v1 全文 6 处，"
     "一次也没写 ReaxFF）。键随配位生成断裂 ⇒ 没有固定拓扑可约束，"
     "constraints 由 missing 转为 not_applicable。兜底未动：真写了 "
     "SHAKE/LINCS/SETTLE/RATTLE 的篇章照旧由 _POS_BONDS 判有键。"),
    ("has_dispersion", True, False,
     "色散的正向证据分了强/弱两档，LJ/Lennard-Jones 这类**母名**不再"
     "推翻 WCA 族（WCA 本就是截断到势极小点的 LJ）。2312.07027v2 全文 "
     "12 处 LJ 指的都是那个 WCA 体系，dispersion_correction 由 missing "
     "转为 not_applicable。强证据那条（2506.04835v1 的 12-6 LJ）未动。"),
    ("titratable_residues_present", None, True,
     "新增键。为真的条件：体系含蛋白/肽，且文本没把可滴定残基的存在"
     "排除掉（蛋白几乎必然含，作者没写不等于没有）。全语料只有 "
     "1707.05453v1 落在这里（蛋白 + charged termini）。"),
    ("titratable_residues_present", None, False,
     "新增键。为假有两种情形，行为影响不同：(a) 体系里根本没有蛋白/肽"
     "（22 篇），与新增前完全等价，闸门原本就由 protein_present 判假；"
     "(b) 含肽但文本明确两端封端且全文无可滴定残基命名，2602.11098v1 "
     "的 ACE-ALA-NME 二肽，**只有这一篇行为变了**：protonation_state "
     "由 missing 转 not_applicable，R-09 随之不再触发（同一判据，"
     "见 extract.has_titratable_groups）。"),
]

# 两张表必须互斥。同一个键既「是字段的影子」、又「被我改过判读」是自相
# 矛盾的说法：先查谁就由谁决定结果，这种"静默谁先赢"是配置错误，不是
# 数据问题。表都是手写的、加起来十几个键，所以在导入时直接拒绝。
_CTX_TABLE_OVERLAP = sorted({k for k, _, _, _ in DECLARED_CTX}
                            & set(_CTX_FROM_FIELD))
if _CTX_TABLE_OVERLAP:
    raise RuntimeError(
        "DECLARED_CTX 与 _CTX_FROM_FIELD 有重键，先决定谁是权威："
        + ", ".join(_CTX_TABLE_OVERLAP))


# -----------------------------------------------------------------------------
def _capture() -> Dict[str, Any]:
    """跑一遍全量，留下逐格状态 + 取值 + 上下文 + 规则编号。

    只走 score_real.run_all 这一条路径，自己重算一套指标等于在量
    两件不同的东西，两边"都对了"也可能只是因为量错了同一处。
    """
    labels = S.load_labels()
    r = S.run_all(labels)
    papers: Dict[str, Any] = {}
    for s in r["per_section"]:
        papers[s["id"]] = {
            "status": s["status"],
            "values": s["values"],
            "ctx": sorted(s["ctx"]),
            # 完整上下文（含 ensemble 这类字符串值）。见 score_real 里的说明：
            # 只留 True 键的那个 ctx 看不见闸门输入本身。
            "ctx_all": s.get("ctx_all") or {},
            "rules": s["rules"],
        }
    return {
        "papers": papers,
        "n_papers": len(papers),
        "status_acc": r["status_acc"],
        "n_judgments": r["n_judgments"],
        "per_class": {k: {"p": v["precision"], "r": v["recall"], "f1": v["f1"]}
                      for k, v in r["per_class"].items()},
    }


def _canonical(cap: Dict[str, Any]) -> str:
    return json.dumps(cap, ensure_ascii=False, sort_keys=True, default=str)


def _load_labels_by_id() -> Dict[str, Dict[str, Any]]:
    return {lab["id"]: lab["fields"] for lab in S.load_labels()}


def _load_adj() -> Dict[str, str]:
    if not os.path.exists(ADJ_PATH):
        return {}
    return json.load(io.open(ADJ_PATH, encoding="utf-8"))


# -----------------------------------------------------------------------------
def cmd_snapshot(args) -> int:
    path = os.path.join(SNAP_DIR, args.tag + ".json")
    if os.path.exists(path) and not args.force:
        print(f"快照 {args.tag} 已存在：{path}")
        print("覆盖会把基准线换掉（对账从此对着一个不同的起点）。"
              "确实要覆盖请加 --force。")
        return 2

    first = _capture()
    second = _capture()
    if _canonical(first) != _canonical(second):
        print("!! 不确定：同一棵树上连跑两次，结果不一致。")
        print("   抽取必须是纯函数。先修这个，再谈对账，"
              "否则所有 fixed/broken 都可能是抖动。")
        return 2

    first["tag"] = args.tag
    first["hash"] = _hash(_canonical(first))
    os.makedirs(SNAP_DIR, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        json.dump(first, fh, ensure_ascii=False, indent=1, default=str)

    pc = first["per_class"]
    print(f"快照 {args.tag} -> {path}")
    print(f"  篇数 {first['n_papers']}  判定 {first['n_judgments']}  "
          f"一致率 {first['status_acc']:.1%}  hash {first['hash']}")
    print("  " + "  ".join(f"{k} P{v['p']:.2f}/R{v['r']:.2f}"
                           for k, v in pc.items()))
    return 0


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _load_snap(tag: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(SNAP_DIR, tag + ".json")
    if not os.path.exists(path):
        return None
    return json.load(io.open(path, encoding="utf-8"))


# -----------------------------------------------------------------------------
def cmd_diff(args) -> int:
    a, b = _load_snap(args.diff[0]), _load_snap(args.diff[1])
    for tag, snap in ((args.diff[0], a), (args.diff[1], b)):
        if snap is None:
            print(f"没有快照 {tag}。先跑 --snapshot --tag {tag}。")
            return 2

    labels = _load_labels_by_id()
    adj = _load_adj()

    cells: List[Dict[str, Any]] = []
    problems: Dict[str, List[str]] = {k: [] for k in EXIT_NAMES}
    entered: List[Dict[str, Any]] = []
    rule_moves: Dict[str, Any] = {}
    status_changed: set = set()

    # 硬约束 2：状态与取值自洽（**全量**查，不只查变动格子）
    # present 必须给出取值；missing / not_applicable 必须没有取值。
    # 反过来（present 却无值）是"声称有、却拿不出"，比缺失更难发现。
    # ambiguous **允许**带值：本项目的既有约定是"报出你匹配到的那个东西"
    # （裸 AMBER 报 ambiguous 且带值 "AMBER"），值告诉人到底提到了什么。
    # 至于"乙层的 ambiguous 不得凭空造值"，那是乙自己的不变量，
    # 由 test_regressions 第 [18] 组直接钉住实现，不靠总量对账。
    for aid, pb in b["papers"].items():
        for f, st in pb["status"].items():
            val = pb["values"].get(f)
            if st == "present" and val is None:
                problems["value_state"].append(f"{aid}|{f}: present 却没有取值")
            elif st in ("missing", "not_applicable") and val is not None:
                problems["value_state"].append(
                    f"{aid}|{f}: {st} 却带着取值 {val!r}")

    for aid, pb in b["papers"].items():
        pa = a["papers"].get(aid)
        if pa is None:
            problems["rule_drift"].append(f"{aid}: 快照里没有这一篇")
            continue

        # 硬约束 1：ctx 与规则编号不得漂移，
        # 比较完整上下文（ctx_all），不是那个只留 True 键的显示用 ctx。
        # 后者看不见 ensemble，恒压器闸门的唯一输入。
        ka, kb = pa.get("ctx_all"), pb.get("ctx_all")
        if ka and kb:
            diff = {k: (ka.get(k), kb.get(k))
                    for k in set(ka) | set(kb) if ka.get(k) != kb.get(k)}
            for k, (ov, nv) in sorted(diff.items()):
                shown = f"{aid}: {k}: {ov!r} -> {nv!r}"
                decls = [d for d in DECLARED_CTX if d[0] == k]
                if decls:
                    # 表里写死的旧值、新值必须**同时**对上才算数。
                    # 只对上旧值就放行等于给这个键开了永久豁免，
                    # 那正是这张表要避免的东西。
                    hit = [d for d in decls if (d[1], d[2]) == (ov, nv)]
                    if hit:
                        # 存元组而非拼好的句子：一次声明多半是"一个键 × 全语料"，
                        # 逐篇重复理由会把信号淹掉，渲染时按键分组。
                        problems["ctx_declared"].append(
                            (aid, k, ov, nv, hit[0][3]))
                    else:
                        declared = "；".join(f"{d[1]!r} -> {d[2]!r}" for d in decls)
                        problems["ctx_drift"].append(
                            shown + f"（{k} 已声明，但声明的是 {declared}，"
                                    f"与实测不符）")
                    continue
                src = _CTX_FROM_FIELD.get(k)
                if src is None:
                    problems["ctx_drift"].append(shown)
                    continue
                # 镜像判定：旧值对旧字段、新值对新字段，两边都一致才算。
                # 只查一边会漏掉"ctx 本来就与字段不一致"的情形，
                # 那种不一致本身就是缺陷，不该被这次搬动顺手掩盖。
                fld, get = src
                ov_f = (pa.get("values") or {}).get(fld)
                nv_f = (pb.get("values") or {}).get(fld)
                if ka.get(k) == get(ov_f) and kb.get(k) == get(nv_f):
                    problems["ctx_derived"].append(
                        shown + f"（{fld} 字段值的镜像，可解释）")
                else:
                    problems["ctx_drift"].append(
                        shown + f"（声称是 {fld} 的镜像，但对不上："
                                f"{get(ov_f)!r} / {get(nv_f)!r}）")
        elif pa["ctx"] != pb["ctx"]:
            # 旧快照（jia3 及以前）没有 ctx_all，只剩"哪些键为真"的集合。
            # 弱一些，但好过静默不查。**仍然要过 DECLARED_CTX**：键集合看不见
            # 真值，但"出现"就等于 False->True、"消失"就等于 True->False，
            # 正好是声明表记的那两种形态。不过这一关的话，任何一次全轮对账
            # （baseline 恰是旧格式）都会因这几条**已声明**的搬动直接退出码 2，
            # 把真正的漂移淹在噪音里。
            sa, sb = set(pa["ctx"]), set(pb["ctx"])
            for k in sorted(sa ^ sb):
                if k in sb:                                # 出现：旧假新真
                    hit = [d for d in DECLARED_CTX
                           if d[0] == k and not d[1] and d[2]]
                    shown_v = "False -> True"
                else:                                      # 消失：旧真新假
                    hit = [d for d in DECLARED_CTX
                           if d[0] == k and d[1] and not d[2]]
                    shown_v = "True -> False"
                if hit:
                    problems["ctx_declared"].append(
                        (aid, k, hit[0][1], hit[0][2], hit[0][3]))
                else:
                    problems["ctx_drift"].append(
                        f"{aid}: {k} {shown_v}"
                        f"（旧快照 {pa['ctx']} -> {pb['ctx']}，无 ctx_all，"
                        f"只能用键集合判断，且不在声明表里）")
        if pa["rules"] != pb["rules"]:
            rule_moves[aid] = (pa["rules"], pb["rules"])

        for f in sorted(pb["status"]):
            new = pb["status"][f]
            old = pa["status"].get(f)
            if old == new:
                continue
            # 注意：--field 只过滤**显示**，不过滤约束检查，
            # 只盯一个字段时，别的字段违反硬约束同样要拦下来。
            status_changed.add(aid)

            gold = (labels.get(aid, {}).get(f) or {}).get("status")
            verdict = adj.get(f"{aid}|{f}", "")
            # 裁定为 annotator_wrong 的格子**不自动归类**。
            # 这种格子的含义是"标注错了、工具当时是对的"，也就是基准只能是
            # 工具自己的旧答案；可工具一旦改动，就没有任何可信基准了，
            # 拿旧答案当基准的话，任何改动都必然记成 broken，包括本来正确的修复，
            # 那是**循环论证**：用裁定本身去判定裁定所豁免的那一格。
            # 所以单列 disputed，交人复核，既不算 fixed 也不算 broken。
            if verdict == "annotator_wrong":
                cat = "disputed"
                ref = None
            else:
                ref = gold
                if new == ref:
                    cat = "fixed"
                elif old == ref:
                    cat = "broken"
                else:
                    cat = "neutral"

            # 硬约束 3：present / not_applicable 不得被离开，
            if old in ("present", "not_applicable"):
                problems["left_present_or_na"].append(
                    f"{aid}|{f}: {old} -> {new}（gold={gold}）")
            if new in ("present", "not_applicable") and old not in (
                    "present", "not_applicable"):
                entered.append({"id": aid, "field": f, "old": old, "new": new,
                                "gold": gold})

            cells.append({"id": aid, "field": f, "old": old, "new": new,
                          "gold": gold, "ref": ref, "cat": cat,
                          "verdict": verdict})

    # 规则漂移
    # 原以为"规则只读 present，状态翻转不该移动它们"。**实测证伪**：
    # 甲-2 修好重音切分后，恒温器由 missing 变 present，R-12（报告了恒温器
    # 类型却没给耦合常数 tau_t）就正确地开始触发了，规则本来就依赖
    # "某字段是否 present"，这是设计使然，不是 bug。
    # 所以真正该拦的是**无法解释的漂移**：某篇的规则变了，
    # 但该篇没有任何状态变动，说明规则与状态脱钩（确定性或耦合出了问题）。
    for aid, (orl, nrl) in sorted(rule_moves.items()):
        added = sorted(set(nrl) - set(orl))
        removed = sorted(set(orl) - set(nrl))
        desc = (f"{aid}: "
                + (f"新触发 {','.join(added)}" if added else "")
                + ("；" if added and removed else "")
                + (f"消失 {','.join(removed)}" if removed else ""))
        if aid in status_changed:
            problems["rule_drift_explained"].append(desc)
        else:
            problems["rule_drift"].append(
                desc + "（该篇没有任何状态变动，无法解释）")

    # 硬约束 4：确定性（同一棵树上连跑两次必须一致），
    # 这里测**两件不同的事**，原先混成了一件、报错话术还说反了：
    #   (a) 真·确定性：当前树连跑两次，两次结果必须一致。与快照无关。
    #   (b) 快照新鲜度：磁盘上的快照 b 是否等于当前树。它**不是**在测确定性，
    #       b 只是个文件，树改了它就过期，那说明"你该重拍快照了"，
    #       不说明抽取不纯。原先 (b) 的失败被报成"连跑两次不一致：抽取不是纯函数"，
    #       会把人引向错误的排查方向（实测在 yb2e→yb2f 上就是这么误导的：
    #       真正原因是当时树已经走到 yb3，yb2f 快照过期了）。
    first, second = _capture(), _capture()
    if _canonical(first) != _canonical(second):
        problems["nondeterministic"].append(
            "当前树连跑两次结果不一致：抽取不是纯函数")
    base = {k: v for k, v in b.items() if k not in ("tag", "hash")}
    if _canonical(base) != _canonical(first):
        problems["stale_snapshot"].append(
            f"快照 {b.get('tag') or '?'} 与当前树不一致（快照已过期，"
            f"不是确定性失败），结论仍成立，但请重拍该快照")
    _render(a, b, cells, entered, problems, args)
    return _exit_code(cells, problems, args)


def _exit_code(cells, problems, args) -> int:
    fatal = (problems["ctx_drift"] or problems["rule_drift"]
             or problems["value_state"] or problems["nondeterministic"])
    if fatal:
        return 2
    if problems["ctx_derived"] and not args.allow_ctx_derived:
        return 2
    if args.strict_containment and problems["left_present_or_na"]:
        return 2
    if any(c["cat"] == "broken" for c in cells):
        return 1
    # disputed 默认也拦下来：它是"基准不可信、没人知道对错"的格子，
    # 静默放过等于把裁定本身的缺陷藏起来。看过之后用 --allow-disputed 放行。
    if any(c["cat"] == "disputed" for c in cells) and not args.allow_disputed:
        return 2
    # 可解释的规则漂移同样要人过目：新触发的规则**可能是对的**（甲-2 之后
    # R-12 开始报"有类型没耦合常数"就是对的），也可能是错的。
    # 只有看过每一条新触发才能判，所以默认拦下。
    if problems["rule_drift_explained"] and not args.allow_rule_drift:
        return 2
    return 0


def _render(a, b, cells, entered, problems, args) -> None:
    W = 78
    shown = [c for c in cells if not args.field or c["field"] == args.field]
    print("=" * W)
    print(f"状态对账  {args.diff[0]} → {args.diff[1]}")
    if args.field:
        print(f"（只显示字段 {args.field}；约束检查仍是全量的）")
    print("=" * W)
    print(f"{'':16s}{'一致率':>10s}{'missing P':>12s}{'ambiguous R':>14s}")
    for tag, s in ((args.diff[0], a), (args.diff[1], b)):
        pc = s["per_class"]
        print(f"  {tag:14s}{s['status_acc']:>9.1%}"
              f"{pc['missing']['p']:>12.2f}{pc['ambiguous']['r']:>14.2f}")
    print()

    counts = {c: sum(1 for x in shown if x["cat"] == c) for c in CATS}
    print(f"变动格子 {len(shown)} / {a['n_judgments']}    "
          + "   ".join(f"{c} {counts[c]}" for c in CATS))
    if args.field and len(shown) != len(cells):
        print(f"  （另有 {len(cells) - len(shown)} 格变动在其他字段，已隐藏）")
    print()

    print("-" * W)
    print("硬约束")
    print("-" * W)
    checks = [
        ("ctx 未漂移", problems["ctx_drift"]),
        ("规则编号未漂移", problems["rule_drift"]),
        ("状态与取值自洽", problems["value_state"]),
        ("确定性（连拍两次一致）", problems["nondeterministic"]),
        ("快照未过期", problems["stale_snapshot"]),
        ("未离开 present / not_applicable", problems["left_present_or_na"]),
    ]
    for name, bad in checks:
        tag = "OK" if not bad else f"!! {len(bad)} 处"
        if name.startswith("未离开") and bad and not args.strict_containment:
            tag += "（本层允许，未计入退出码）"
        print(f"  {name:34s}{tag}")
        for x in bad[:20]:
            print(f"      {x}")
    cdr = problems["ctx_derived"]
    if cdr:
        print(f"  {'ctx 漂移（字段值的镜像）':34s}!! {len(cdr)} 处，需逐条确认")
        for x in cdr[:20]:
            print(f"      {x}")
    cdc = problems["ctx_declared"]
    if cdc:
        print(f"  {'ctx 已声明的搬动':34s}OK {len(cdc)} 处，不致命，需过目")
        by_key: Dict[str, List[Any]] = {}
        for aid, k, ov, nv, why in cdc:
            by_key.setdefault(k, []).append((aid, ov, nv, why))
        for k in sorted(by_key):
            rows = by_key[k]
            for move in sorted({(ov, nv) for _, ov, nv, _ in rows}):
                same = [r for r in rows if (r[1], r[2]) == move]
                print(f"      {k}  {len(same)} 篇: {move[0]!r} -> {move[1]!r}")
                print(f"        （{same[0][3]}）")
                aids = sorted(a for a, _, _, _ in same)
                for i in range(0, len(aids), 6):
                    print(f"        {', '.join(aids[i:i + 6])}")
    drift = problems["rule_drift_explained"]
    if drift:
        print(f"  {'规则漂移（有状态变动可解释）':34s}!! {len(drift)} 篇，需逐条确认")
        for x in drift[:20]:
            print(f"      {x}")
    if entered:
        print(f"  （另有 {len(entered)} 格新进入 present/not_applicable，"
              f"属预期方向）")

    print()
    print("-" * W)
    print("逐格")
    print("-" * W)
    if not shown:
        print("  （无变动）")
    for c in sorted(shown, key=lambda x: (CATS.index(x["cat"]), x["id"],
                                          x["field"])):
        if c["cat"] == "broken":
            why = f"!! 原答案与基准一致（基准={c['ref']}）"
        elif c["cat"] == "disputed":
            why = "该格裁定为 annotator_wrong，基准已不可信，需人工复核"
        elif c["cat"] == "neutral":
            why = f"gold={c['gold']}（旧错新错，非修复）"
        else:
            why = ""
        print(f"  {c['cat']:9s}{c['id']:18s}{c['field']:22s}"
              f"{c['old']:14s}→ {c['new']:14s} {why}".rstrip())

    if counts["disputed"]:
        print()
        print("  disputed：这些格子的标注被裁定为错（annotator_wrong），也就是")
        print("            '工具当时是对的'。但工具已经改了，基准不复存在，")
        print("            算 fixed 是自我表扬，算 broken 是循环论证。必须人工看：")
        print("            要么这格本来就该改（裁定本身有问题），要么改错了。")
    if counts["neutral"]:
        print()
        print("  neutral：旧错新错，**不算修复**。总分可能因此不动，")
        print("           但那不代表没发生变化，它代表换了一种错法。")
    print("=" * W)


# -----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="逐格状态对账")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--snapshot", action="store_true")
    g.add_argument("--diff", nargs=2, metavar=("OLD", "NEW"))
    ap.add_argument("--tag")
    ap.add_argument("--force", action="store_true", help="覆盖同名快照")
    ap.add_argument("--field", help="只显示该字段的变动")
    ap.add_argument("--strict-containment", action="store_true",
                    help="离开 present/not_applicable 即失败（乙/丙 层用）")
    ap.add_argument("--allow-disputed", action="store_true",
                    help="放过 disputed 格子（看过并确认之后才用）")
    ap.add_argument("--allow-rule-drift", action="store_true",
                    help="放过可解释的规则漂移（逐条确认新触发的规则是对的之后才用）")
    ap.add_argument("--allow-ctx-derived", action="store_true",
                    help="放过「字段值镜像」式的 ctx 搬动（逐条确认后再说）")
    args = ap.parse_args(argv)

    if args.snapshot:
        if not args.tag:
            print("--snapshot 需要 --tag。")
            return 2
        return cmd_snapshot(args)
    return cmd_diff(args)


if __name__ == "__main__":
    sys.exit(main())
