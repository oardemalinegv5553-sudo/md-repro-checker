# -*- coding: utf-8 -*-
"""逐字段做大小写敏感性审计，系统性找出"少写 re.I"这一整类漏检。

起因：在 _extract_software 上发现正则少写 re.I，改为手写大小写变体兜底
（`GROMACS|Gromacs|gromacs|...`）。后果比"偶尔漏一个"严重得多：
**只有当年被手动补过的那几个拼写能识别**，实测 "we used openmm 8.0"
与 "namd 2.14 was employed" 都返回 None。

这类漏检的可怕之处在于**症状是"字段缺失"**：使用者看到的是
`software: missing`，只会以为作者没写软件名，不会想到是抽取器的锅。
单看输出永远发现不了，所以必须主动构造等价输入去逼它现形。

做法：每个字段配一句该领域真实会出现的句子，分别以 原样 / 全大写 /
全小写 三种形式跑同一个抽取器，输出不同即报可疑。

一次扫出三个字段（software、ensemble、temperature_bath）的缺失 re.I。
三者的连锁后果值得记下：ensemble 漏掉 -> context["ensemble"] 为 None
-> 恒压器适用性判据拿不到值 -> 整条规则链失效。即**一个 re.I 能让
下游若干条规则静默失声**。

用法：
  python eval/real/audit_case.py

局限（须与结果一起理解）：
  只覆盖"大小写"这一种等价变换。同义写法（"timestep"/"time step"）、
  语序变化、被动语态改写都不在覆盖范围内，本脚本证明不了抽取器
  对这些变换也稳健。它是一把**只量一个维度**的尺子；量出来是 0
  不代表抽取器整体没有漏检。

**第二轮补充（为什么下面多了一段上下文审计）**：
  上面这段局限原本还漏了更要命的一条，第一版只遍历 PATTERNS，
  也就是只量**字段抽取器**。而 `detect_context` 是另一套独立代码，
  它算出的场景标志（ensemble / explicit_solvent / ligand_present ...）
  同样拿去做判断，却完全不在扫描范围内。

  后果实测到了：`_APPLICABILITY` 里 barostat 的闸门写作
  `ctx["ensemble"] == "NPT"`，而 `_extract_ensemble` 把匹配到的原文
  原样返回，于是 "in the **npt** ensemble" 得到 'npt' ≠ "NPT"，
  恒压器被判"本体系不需要"，字段直接消失成 not_applicable。
  同一句话换成大写 "NPT" 就正常。

  这是最难发现的一类：症状是**字段不该消失却消失了**，而且现场
  还留着一个看似合理的 not_applicable，不像 missing 那样扎眼。
  教训是审计必须覆盖**所有会参与判断的产出**，不能只覆盖叫
  PATTERNS 的那一个字典。
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "eval/real")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.extract import PATTERNS, detect_context

# 每句话都按"文献里常见的原始写法"设计，然后整体大写/小写各跑一遍
PROBES = {
    "force_field": "the system was described by the CHARMM36m force field",
    "water_model": "water was represented by the TIP3P model",
    "thermostat": "temperature was controlled by a Nose-Hoover thermostat",
    "barostat": "pressure was maintained by a Parrinello-Rahman barostat",
    "ensemble": "the simulation was run in the NPT ensemble",
    "timestep": "a timestep of 2 fs was used",
    "electrostatics": "electrostatics were handled with PME and a cutoff of 1.0 nm",
    "constraints": "bond lengths were constrained with LINCS",
    "dispersion_correction": "dispcorr = EnerPres was used",
    "simulation_length": "the production run was 200 ns",
    "system_size": "the system contained 52000 atoms",
    "heat_flux_method": "kappa was computed by the Green-Kubo method",
    "temperature_bath": "the cold bath was 300 K and the hot bath 500 K",
    "correlation_window": "the correlation time was 20 ps",
    "software": "simulations used the LAMMPS package",
    "initial_structure": "the initial structure was taken from PDB 1ABC",
}

print("=" * 92)
print("大小写敏感性审计：同一句话三种写法，结果不同即可疑")
print("=" * 92)
print(f"{'字段':22s} {'原样':>10s} {'全大写':>10s} {'全小写':>10s}   结论")
print("-" * 92)

susp = []
for key, sent in sorted(PROBES.items()):
    fn = PATTERNS.get(key)
    if fn is None:
        continue
    outs = []
    for s in (sent, sent.upper(), sent.lower()):
        try:
            h = fn(s)
        except Exception as e:                              # noqa: BLE001
            h = f"ERR:{type(e).__name__}"
        outs.append(h[0] if isinstance(h, tuple) else h)
    ok = [bool(o) for o in outs]
    if len(set(ok)) > 1:
        verdict = "!! 大小写敏感（漏检）"
        susp.append(key)
    elif all(ok):
        verdict = "一致"
    else:
        verdict = "一致（都未命中）"
    print(f"{key:22s} {str(outs[0])[:10]:>10s} {str(outs[1])[:10]:>10s} "
          f"{str(outs[2])[:10]:>10s}   {verdict}")

print()
print(f"可疑字段 {len(susp)}: {', '.join(susp) if susp else '(无)'}")

# -----------------------------------------------------------------------------
# 第二部分：上下文标志审计
#
# 字段抽取器只是产出的一半。detect_context 另有一套判断，算出的标志
# 被 _APPLICABILITY 的闸门、规则触发条件、敏感度归因三处消费。
# 它同样可能"只认某一种写法"，而症状是**场景标志静默为假**，
# 比字段缺失更难看出，因为下游只是"少报了几条规则"，看上去很正常。
#
# 这里对每个标志构造一句该标志**必须为真**的句子，再整体大写/小写各跑一遍。
# 布尔标志只查真值是否稳定；字符串标志（ensemble）还要查**取值是否规范化**，
# barostat 的闸门比的正是这个字符串。
# -----------------------------------------------------------------------------
CTX_PROBES = {
    "ensemble": ("the simulation was run in the npt ensemble", "NPT"),
    "explicit_solvent": ("the protein was solvated in water", True),
    "protein_present": ("the protein was solvated in water", True),
    "ligand_present": ("the ligand was docked into the pocket", True),
    "solvent_free": ("the bulk polymer melt was simulated in vacuum", True),
    "heat_transport": ("the thermal conductivity was computed", True),
    "nemd": ("a temperature gradient was imposed across the slab", True),
}

print()
print("=" * 92)
print("上下文标志审计：detect_context 的产出（闸门/规则/归因都读它）")
print("=" * 92)
print(f"{'标志':22s} {'原样':>14s} {'全大写':>14s} {'全小写':>14s}   结论")
print("-" * 92)

ctx_susp = []
for flag, (sent, want) in sorted(CTX_PROBES.items()):
    outs = []
    for s in (sent, sent.upper(), sent.lower()):
        try:
            outs.append(detect_context(s).get(flag))
        except Exception as e:                              # noqa: BLE001
            outs.append(f"ERR:{type(e).__name__}")
    # 三条都必须等于期望值。"都非空"不够，ensemble 的坑正是
    # 值非空但**大小写不规范化**，闸门比较随之失败。
    bad = [o for o in outs if o != want]
    if bad:
        verdict = f"!! 不稳定（期望 {want!r}）"
        ctx_susp.append(flag)
    else:
        verdict = "一致"
    print(f"{flag:22s} {str(outs[0])[:14]:>14s} {str(outs[1])[:14]:>14s} "
          f"{str(outs[2])[:14]:>14s}   {verdict}")
    if bad and outs[0] == want:
        print(f"{'':22s} └ 原样写法正确、改写后失效，属"
              f"{'取值未规范化' if isinstance(want, str) else '漏检'}类")

print()
print(f"可疑标志 {len(ctx_susp)}: {', '.join(ctx_susp) if ctx_susp else '(无)'}")
print()
print("注：字段层与上下文层是两套独立代码，必须分别量。")
print("    第一版审计只量了字段层，于是 `ensemble` 在 detect_context 侧")
print("    的大小写缺陷整轮没被发现，这是本脚本扩充的直接原因。")
