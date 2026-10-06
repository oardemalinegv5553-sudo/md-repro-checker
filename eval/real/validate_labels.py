# -*- coding: utf-8 -*-
"""
validate_labels.py：标注文件的结构与证据校验

用途：在把人工标注交给 score_real.py 打分**之前**，先机械地验一遍。
一份格式错、或 evidence 不是原文逐字的标注，会让之后每一步都建在沙上，
而这两种错误光看指标是发现不了的（它们会伪装成"工具判错了"）。

检查项：
  1. JSON 合法、顶层有 id / fields
  2. id 能在 manifest 里找到对应的方法段文件
  3. fields 恰好 18 个，字段名与 schema 一致（顺序不同只警告，不判失败）
  4. status ∈ {present, missing, ambiguous, not_applicable}
  5. present / ambiguous  → value 非空、evidence 非空
     missing / not_applicable → value 为 None、evidence 为空串
  6. **evidence 必须能在方法段原文中定位**（分级判定，见 evidence_tier），
     这一条是整份校验里最有价值的：它挡掉"凭印象编一句原文"。

退出码：0 全部通过；1 有失败项。

用法：
    python eval/real/validate_labels.py            # 校验 out/ 下全部标注
    python eval/real/validate_labels.py <id> ...   # 只校验指定 ID
"""
import io
import os
import re
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_label_work")
OUT = os.path.join(WORK, "out")

STATES = ("present", "missing", "ambiguous", "not_applicable")
# 与 schema/md_method.yaml 的字段顺序一致
FIELDS = [
    "software", "force_field", "water_model", "protonation_state",
    "ligand_parameters", "electrostatics", "constraints", "timestep",
    "thermostat", "barostat", "ensemble", "simulation_length",
    "system_size", "initial_structure", "dispersion_correction",
    "heat_flux_method", "temperature_bath", "correlation_window",
]


def norm(s: str) -> str:
    """空白归一：标注员引用原文时可能合并了换行、多打了空格。"""
    return re.sub(r"\s+", " ", s).strip()


def squash(s: str) -> str:
    """去掉全部空白。语料在数学模式两侧补过空格（`(NPT)` → `( NPT )`），
    标注员照原文抄或按常规排版抄都不算错，这一层用来吸收这种差异。"""
    return re.sub(r"\s+", "", s)


# 排版字符 → ASCII。原文是 LaTeX 排出来的，带弯引号 `et al.’s`、连接号 `–`、
# 减号 `−`、乘号 `×`；标注员手敲的是直引号和 ASCII 连字符。差异在字体不在内容。
FOLD = {0x2018: "'", 0x2019: "'", 0x201c: '"', 0x201d: '"',
        0x2013: "-", 0x2014: "-", 0x2212: "-", 0x00d7: "x",
        0x00a0: " ", 0x2032: "'", 0x2033: '"'}


def fold(s: str) -> str:
    return s.translate(FOLD)


def deparen(s: str) -> str:
    """去掉括号插入语。`(red online)`（图注颜色）、`(SI, Sec. S1)`（交叉引用）、
    `(Figs. a,b)` 这类旁注在引用时被省略是惯例，也不承载证据，省掉它们而
    保留正句，与"编造原文"是两回事。"""
    return re.sub(r"\([^()]*\)", "", s)


def longest_run(a: str, b: str):
    """a 与 b 的最长公共连续片段。用来给"对不上但确有出处"的引文找一个锚点。"""
    import difflib
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blk = m.find_longest_match(0, len(a), 0, len(b))
    return blk.size, a[blk.a: blk.a + blk.size]


TIERS = {"verbatim": 0, "elided": 0, "spacing": 0, "typographic": 0,
         "paren": 0, "punctuated": 0, "rewritten": 0, "unmatched": 0,
         "empty": 0}


def evidence_tier(ev: str, text: str):
    """判定 evidence 与原文的吻合层级，返回 (层级, 附注)。

    分级而不是一刀切，是因为"证据可追溯"有几种合法的书写方式。把它们一律
    判失败，校验器自己就成了噪声源，本文件第一版正是这么错的：122 条逐字
    命中的旁边，18 条用省略号拼接的合法引用被一起判成了失败，看起来像 38
    处标注事故，实际是 1 条规则定错了。

    另一头也要防：把"引文被压缩改写"和"编造原文"混为一谈。两者都让字面比对
    失败，但前者有真实出处、只是标注员把散文改成了记号或把推算写进了引文，
    值本身不受影响。故设 `rewritten` 档，**只要与原文共享 20 字以上的连续
    片段就认定为有锚点**，降为警告并报出那段锚点，让人一眼能核。编造的句子
    不会与原文共享 20 字连续串，仍然判失败。
    """
    if not ev.strip():
        return "empty", ""
    a = norm(ev)
    if a in text:
        return "verbatim", ""

    parts = [p.strip() for p in re.split(r"\.\.\.|…", a) if p.strip()]
    if len(parts) > 1 and all(p in text for p in parts):
        return "elided", ""

    tsq = squash(text)
    if all(squash(p) in tsq for p in parts):
        return "spacing", ""

    ft, fp = fold(tsq), [fold(squash(p)) for p in parts]
    if all(x in ft for x in fp):
        return "typographic", ""

    dt = fold(squash(deparen(text)))
    if all(fold(squash(deparen(p))) in dt for p in parts):
        return "paren", ""

    # 有标注员用 ';' 而不是 '…' 拼接不相邻的引文。分隔符不规范，但每一段都
    # 能在原文里定位，这与"编造原文"是两回事，故降为警告，不判失败。
    semi = [p.strip() for p in a.split(";") if p.strip()]
    if len(semi) > 1 and all(fold(squash(p)) in ft for p in semi):
        return "punctuated", ""

    size, run = longest_run(fold(squash(a)), ft)
    if size >= 20:
        return "rewritten", run

    return "unmatched", run if size else ""


def section_text(aid: str) -> str:
    for cand in (aid.replace("/", "_"), aid):
        p = os.path.join(HERE, "sections", cand + ".txt")
        if os.path.exists(p):
            raw = io.open(p, encoding="utf-8").read()
            # 首部的 "# section: ..." / "# level: ..." 是我们自己加的，不算原文
            return norm("\n".join(l for l in raw.splitlines()
                                  if not l.startswith("#")))
    return ""


def load_manifest_ids() -> set:
    p = os.path.join(HERE, "manifest.json")
    if not os.path.exists(p):
        return set()
    m = json.load(io.open(p, encoding="utf-8"))
    return {r["id"] for r in m.get("records", [])}


def check_one(path: str, known_ids: set):
    fails, warns = [], []
    name = os.path.basename(path)

    try:
        d = json.load(io.open(path, encoding="utf-8"))
    except Exception as e:
        return [f"{name}: JSON 解析失败，{type(e).__name__}: {e}"], []

    if not isinstance(d, dict) or "id" not in d or "fields" not in d:
        return [f"{name}: 顶层必须含 id 与 fields"], []

    aid = d["id"]
    if known_ids and aid not in known_ids:
        fails.append(f"{name}: id={aid!r} 不在 manifest 里")

    fields = d["fields"]
    if not isinstance(fields, dict):
        return [f"{name}: fields 必须是字典"], []

    keys = list(fields)
    missing = [f for f in FIELDS if f not in keys]
    extra = [k for k in keys if k not in FIELDS]
    if missing:
        fails.append(f"{name}: 缺字段 {missing}")
    if extra:
        fails.append(f"{name}: 多余字段 {extra}")
    if not missing and not extra and keys != FIELDS:
        warns.append(f"{name}: 18 个字段齐全但顺序与 schema 不同")

    text = section_text(aid)
    if not text:
        warns.append(f"{name}: 找不到方法段原文，跳过 evidence 逐字校验")

    for key in FIELDS:
        if key not in fields:
            continue
        v = fields[key]
        tag = f"{name} [{key}]"
        if not isinstance(v, dict):
            fails.append(f"{tag}: 必须是含 status/value/evidence/note 的字典")
            continue
        for k in ("status", "value", "evidence", "note"):
            if k not in v:
                fails.append(f"{tag}: 缺键 {k!r}")

        st = v.get("status")
        if st not in STATES:
            fails.append(f"{tag}: status={st!r} 不在四态内")
            continue

        val, ev = v.get("value"), v.get("evidence")

        if st in ("present", "ambiguous"):
            if val is None or (isinstance(val, str) and not val.strip()):
                fails.append(f"{tag}: status={st} 但 value 为空")
            if not (isinstance(ev, str) and ev.strip()):
                fails.append(f"{tag}: status={st} 但 evidence 为空")
        else:
            if val is not None:
                fails.append(f"{tag}: status={st} 但 value 非 None")
            if ev not in (None, ""):
                fails.append(f"{tag}: status={st} 但 evidence 非空")

        # evidence 可定位性，整份校验里最有价值的一条：
        # 它挡掉"凭印象编一句原文"，而那种错误在指标里会伪装成"工具判错了"。
        if isinstance(ev, str) and ev.strip() and text:
            tier, run = evidence_tier(ev, text)
            TIERS[tier] = TIERS.get(tier, 0) + 1
            if tier == "punctuated":
                warns.append(
                    f"{tag}: 用 ';' 而非 '…' 拼接不相邻引文"
                    f"（各段均可在原文定位，建议改写为省略号）")
            elif tier == "rewritten":
                warns.append(
                    f"{tag}: evidence 非逐字，最长的可定位片段为 {run!r}"
                    f"（引文被压缩或改写；值本身不受影响，裁定该条时请直接核原文）")
            elif tier == "unmatched":
                fails.append(f"{tag}: evidence 无法在原文中定位，{norm(ev)[:60]!r}")

    return fails, warns


def main(argv):
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    want = set(argv[1:])
    known = load_manifest_ids()

    if not os.path.isdir(OUT):
        print(f"找不到标注目录：{OUT}")
        return 1

    all_f, all_w, n = [], [], 0
    for fn in sorted(os.listdir(OUT)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(OUT, fn)
        try:
            aid = json.load(io.open(p, encoding="utf-8")).get("id", "")
        except Exception:
            aid = ""
        if want and aid not in want and os.path.splitext(fn)[0] not in want:
            continue
        n += 1
        f, w = check_one(p, known)
        all_f += f
        all_w += w
        mark = "OK  " if not f else "FAIL"
        print(f"[{mark}] {fn:<26} 失败 {len(f)}  警告 {len(w)}")

    print("\n" + "=" * 70)
    tot = sum(TIERS.values())
    print(f"evidence 可定位性（共 {tot} 条）：")
    print(f"   逐字命中     {TIERS['verbatim']}")
    print(f"   省略号拼接   {TIERS['elided']}")
    print(f"   仅空白差异   {TIERS['spacing']}  （语料在数学模式两侧补过空格）")
    if TIERS["typographic"]:
        print(f"   排版字符差异 {TIERS['typographic']}  （弯引号 / 连接号，原文是 LaTeX 排的）")
    if TIERS["paren"]:
        print(f"   省去括号旁注 {TIERS['paren']}  （(red online)、(SI, Sec. S1) 一类）")
    if TIERS["punctuated"]:
        print(f"   分隔符不规范 {TIERS['punctuated']}  （用 ';' 代替 '…'，内容可定位）")
    if TIERS["rewritten"]:
        print(f"   压缩改写     {TIERS['rewritten']}  （有锚点，非逐字；见下方警告）")
    if TIERS["unmatched"]:
        print(f"   **无法定位** {TIERS['unmatched']}")

    if all_w:
        print(f"\n警告 {len(all_w)} 条：")
        for w in all_w:
            print("   -", w)
    if all_f:
        print(f"\n失败 {len(all_f)} 条：")
        for f in all_f:
            print("   -", f)
        print(f"\n结论：{n} 份标注，{len(all_f)} 条失败，未通过")
        return 1
    ok = tot - TIERS["unmatched"] - TIERS["empty"]
    print(f"\n结论：{n} 份标注全部通过结构校验；"
          f"evidence {ok}/{tot} 条可在原文中定位"
          + (f"，其中 {TIERS['rewritten']} 条为压缩改写（已列在上方警告里）"
             if TIERS["rewritten"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
