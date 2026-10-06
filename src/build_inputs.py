# -*- coding: utf-8 -*-
"""
build_inputs.py：从结构化字段生成可执行的 MD 输入文件


本模块的核心是**来源标注（provenance）**：

 文献写了什么    -> 用文献值，标记 source="paper"
 文献没写但可默认 -> 用推荐默认值，标记 source="default"  ← 关键
 文献没写且不能默认 -> 拒绝生成，标记 source="unresolved"

第三类必须让流水线**显式失败**。理由：
一个静默填了默认值的输入文件，跑出来的结果看起来完全正常，
这正是"跑出一堆不可信的数"的成因。把默认值标出来，
使用者才知道哪些结论其实建立在假设之上。

输出：GROMACS .mdp（em / nvt / npt / prod）+ OpenMM 的最小 Python 骨架
      + LAMMPS 输入脚本（em / nvt / npt / prod）
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# GROMACS .mdp 参数名白名单（用于语法与拼写校验，不依赖安装 GROMACS）
# 覆盖本项目会用到的全部键；完整列表见 GROMACS 手册 mdp 选项。
# -----------------------------------------------------------------------------
MDP_KEYS = {
    # Run control
    "integrator", "tinit", "dt", "nsteps", "init-step", "simulation-part",
    "comm-mode", "nstcomm", "comm-grps",
    # Langevin / BD
    "bd-fric", "ld-seed",
    # Energy minimization
    "emtol", "emstep",
    # Shell molecular dynamics
    "niter", "fcstep", "cgstep",
    # Test particle insertion
    "nstlist", "rtpi",
    # Output control
    "nstxout", "nstvout", "nstfout", "nstlog", "nstcalcenergy", "nstenergy",
    "nstxout-compressed", "compressed-x-precision", "compressed-x-grps",
    "energygrps",
    # Neighbor searching
    "cutoff-scheme", "nstlist", "ns-type", "pbc", "periodic-molecules",
    "verlet-buffer-tolerance", "rlist", "rlistlong",
    # Electrostatics
    "coulombtype", "coulomb-modifier", "rcoulomb-switch", "rcoulomb",
    "epsilon-r", "epsilon-rf", "vdwtype", "vdw-modifier", "rvdw-switch",
    "rvdw", "DispCorr", "table-extension", "fourierspacing",
    "fourier-nx", "fourier-ny", "fourier-nz", "pme-order", "ewald-rtol",
    "ewald-rtol-lj", "lj-pme-comb-rule", "ewald-geometry", "epsilon-surface",
    # Temperature coupling
    "tcoupl", "nsttcouple", "nh-chain-length", "print-nose-hoover-chain-variables",
    "tc-grps", "tau-t", "ref-t",
    # Pressure coupling
    "pcoupl", "pcoupltype", "nstpcouple", "tau-p", "compressibility",
    "ref-p", "refcoord-scaling",
    # QM/MM
    "QMMM", "QMMM-grps", "QMmethod", "QMbasis", "QMMM-scale",
    # Bonds
    "constraints", "constraint-algorithm", "continuation", "shake-tol",
    "lincs-order", "lincs-iter", "lincs-warnangle", "morse",
    # Energy group exclusions
    "energygrp-excl",
    # Walls
    "nwall", "wall-type", "wall-r-linpot", "wall-atomtype", "wall-density",
    "wall-ewald-zfac",
    # COM pulling
    "pull", "rotation", "interactiveMD",
    # AWH
    "awh",
    # Distributed topology
    "constraints",
    # Free energy
    "free-energy", "init-lambda", "init-lambda-state", "calc-lambda-neighbors",
    "sc-alpha", "sc-coul", "sc-power", "sc-sigma", "nstdhdl",
    "dhdl-derivatives", "dhdl-print-energy", "separate-dhdl-file",
    "couple-moltype", "couple-lambda0", "couple-lambda1", "couple-intramol",
    # Non-equilibrium MD
    "acc-grps", "accelerate", "freezegrps", "freezedim", "cos-acceleration",
    "deform",
    # Electric fields
    "E-x", "E-xt", "E-y", "E-yt", "E-z", "E-zt",
    # Random seed
    "ld-seed",
    # Velocity generation
    "gen-vel", "gen-temp", "gen-seed",
    # Simulated annealing
    "annealing", "annealing-npoints", "annealing-time", "annealing-temp",
}

# 白名单里重复出现的键（mdp 中同一键只应出现一次）
_SEEN = set()
_DUP = [k for k in MDP_KEYS if k in _SEEN or _SEEN.add(k)]
MDP_KEYS = _SEEN

# -----------------------------------------------------------------------------
# 常见力场的推荐默认（仅在文献未报告时使用，且必须被标注）
# -----------------------------------------------------------------------------
DEFAULTS = {
    "coulombtype": ("PME", "GROMACS 对周期性体系的常规选择"),
    "rcoulomb": ("1.0", "多数力场（AMBER/CHARMM/OPLS）的常规截断；"
                        "CHARMM36 推荐 1.2"),
    "rvdw": ("1.0", "与 rcoulomb 保持一致，避免能量不连续"),
    "DispCorr": ("EnerPres", "截断 LJ 时补偿长程色散，避免密度系统偏低"),
    "constraint-algorithm": ("LINCS", "GROMACS 默认；与 SETTLE 处理水共同使用"),
    "lincs-order": ("4", "常规取值"),
    "lincs-iter": ("1", "常规取值"),
    "cutoff-scheme": ("Verlet", "GROMACS 4.6 之后的唯一推荐方案"),
    "nstlist": ("20", "与 Verlet 缓冲配合的常规取值"),
    "tcoupl": ("V-rescale", "产生正确正则系综涨落；输运性质计算**不可**用 Berendsen"),
    "tau-t": ("0.1", "常规弱耦合常数"),
    "pcoupl": ("C-rescale", "产生正确系综涨落，优于 Berendsen"),
    "tau-p": ("2.0", "常规弱耦合常数"),
    "compressibility": ("4.5e-5", "水的压缩系数"),
}

# 无法默认、必须由文献提供的字段 -> 缺失即拒绝生成
UNRESOLVED_IF_MISSING = ["force_field", "water_model", "ensemble",
                         "simulation_length", "initial_structure"]

# 力场名 -> mdp 相关片段（GROMACS 用 include 的 itp，这里只记录以便溯源）
FF_TO_ITP = {
    "AMBER99SB-ILDN": "amber99sb-ildn.ff",
    "AMBER14SB": "amber14sb.ff",
    "CHARMM36m": "charmm36m.ff",
    "CHARMM36": "charmm36.ff",
    "OPLS-AA": "oplsaa.ff",
    "GAFF2": "gaff2 (需用 acpype/antechamber 生成 itp)",
}


def _lookup(fields: Dict[str, Any], key: str) -> Tuple[Optional[Any], str]:
    """返回 (value, source)。source ∈ {paper, default, unresolved, not_applicable}"""
    f = fields.get(key) or {}
    st = f.get("status")
    if st == "present":
        return f.get("value"), "paper"
    if st == "not_applicable":
        return None, "not_applicable"
    return None, "unresolved"


def _mdp_line(key: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    return f"{key:<24}= {value}"


def build_gromacs(fields: Dict[str, Any], stage: str = "prod",
                  blocking: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    stage ∈ {em, nvt, npt, prod}
    blocking: 由敏感性归因层判定为"本场景下缺失即不可复现"的字段
    返回 {"mdp": str, "provenance": {...}, "unresolved": [...], "warnings": [...]}
    """
    lines: List[str] = []
    prov: Dict[str, str] = {}
    warnings: List[str] = []

    def add(key: str, value: Any, source: str) -> None:
        lines.append(_mdp_line(key, value))
        prov[key] = source

    # ---- 运行控制 ----
    if stage == "em":
        add("integrator", "steep", "builtin")
        add("emtol", "1000.0", "builtin")
        add("emstep", "0.01", "builtin")
        add("nsteps", "50000", "builtin")
    else:
        ts, ts_src = _lookup(fields, "timestep")
        if ts is None:
            ts, ts_src = 2.0, "default"
            warnings.append("文献未报告时间步，默认 2 fs（需配合氢键约束）")
        add("integrator", "md", "builtin")
        # 单位换算：文献惯例用 fs 报告时间步，GROMACS 的 dt 单位是 ps。
        # 直接把 2 写进 dt 会得到 2 ps 的步长，模拟会瞬间炸开，
        # 而且报错信息不会指向这里。
        add("dt", f"{ts / 1000.0:g}", ts_src)
        # 时长
        sl, sl_src = _lookup(fields, "simulation_length")
        if isinstance(sl, dict) and sl.get("length"):
            L, unit = sl["length"], sl.get("unit", "ns")
            ns = L * (1000 if unit == "us" else (1 if unit == "ns" else 0.001))
            add("nsteps", int(ns * 1e6 / ts), sl_src)
            if not sl.get("replicas"):
                warnings.append("文献未报告重复次数，nsteps 仅对应单条轨迹")
        else:
            add("nsteps", 50000000, "default")
            warnings.append("文献未报告模拟时长，默认 100 ns")

    lines.append("")

    # ---- 输出 ----
    if stage != "em":
        add("nstxout-compressed", 5000, "builtin")
        add("nstenergy", 5000, "builtin")
        add("nstlog", 5000, "builtin")
        add("compressed-x-precision", 1000, "builtin")
        lines.append("")

    # ---- 近邻与静电 ----
    add("cutoff-scheme", DEFAULTS["cutoff-scheme"][0], "default")
    add("nstlist", DEFAULTS["nstlist"][0], "default")
    add("pbc", "xyz", "builtin")

    es, es_src = _lookup(fields, "electrostatics")
    if isinstance(es, dict):
        ct = {"PME": "PME", "reaction-field": "Reaction-Field",
              "Ewald": "Ewald"}.get(es.get("method", ""), "PME")
        add("coulombtype", ct, es_src)
        if es.get("rcoulomb_nm"):
            add("rcoulomb", f"{es['rcoulomb_nm']:g}", es_src)
            add("rvdw", f"{es['rcoulomb_nm']:g}", es_src)
        else:
            add("rcoulomb", DEFAULTS["rcoulomb"][0], "default")
            add("rvdw", DEFAULTS["rvdw"][0], "default")
            warnings.append("文献未报告截断半径，默认 1.0 nm；"
                            "若原研究用力场为 CHARMM36 系，真实值多为 1.2 nm，"
                            "此处会引入系统偏差")
    else:
        add("coulombtype", DEFAULTS["coulombtype"][0], "default")
        add("rcoulomb", DEFAULTS["rcoulomb"][0], "default")
        add("rvdw", DEFAULTS["rvdw"][0], "default")

    dc, dc_src = _lookup(fields, "dispersion_correction")
    if dc:
        add("DispCorr", dc, dc_src)
    else:
        add("DispCorr", DEFAULTS["DispCorr"][0], "default")
    lines.append("")

    # ---- 键约束 ----
    cons, cons_src = _lookup(fields, "constraints")
    add("constraints", "h-bonds" if cons in ("LINCS", "SHAKE", "h-bonds only", None) else "all-bonds",
        cons_src if cons else "default")
    if cons in ("LINCS", None):
        add("constraint-algorithm", "LINCS", cons_src if cons else "default")
        add("lincs-order", DEFAULTS["lincs-order"][0], "default")
        add("lincs-iter", DEFAULTS["lincs-iter"][0], "default")
    elif cons == "SHAKE":
        add("constraint-algorithm", "SHAKE", "paper")
        add("shake-tol", "0.0001", "default")
    lines.append("")

    # ---- 温度耦合 ----
    tc, tc_src = _lookup(fields, "thermostat")
    if isinstance(tc, dict):
        mapping = {"V-rescale": "V-rescale", "Berendsen": "berendsen",
                   "Nose-Hoover": "nose-hoover", "Langevin": "langevin",
                   "Andersen": "andersen", "CSVR": "v-rescale"}
        add("tcoupl", mapping.get(tc.get("type"), "V-rescale"), tc_src)
        if tc.get("coupling"):
            add("tau-t", tc["coupling"], tc_src)
        else:
            add("tau-t", DEFAULTS["tau-t"][0], "default")
        if tc.get("type") == "Berendsen":
            warnings.append("文献使用 Berendsen 恒温器。该恒温器压制涨落，"
                            "若原研究涉及扩散系数/热导率等输运性质，"
                            "其结果存在已知的系统偏差，复现时需注意")
    else:
        add("tcoupl", DEFAULTS["tcoupl"][0], "default")
        add("tau-t", DEFAULTS["tau-t"][0], "default")
        warnings.append("文献未报告恒温器，默认 V-rescale")

    # 目标温度：有冷热浴信息时取其平均（NEMD 的生产温度通常即平均值），
    # 否则退回 300 K。硬编码 300 K 会在文献实际用 350 K 时悄悄写错。
    tb, tb_src = _lookup(fields, "temperature_bath")
    if isinstance(tb, dict) and tb.get("hot_K") and tb.get("cold_K"):
        t_target = 0.5 * (tb["hot_K"] + tb["cold_K"])
        t_note = f"取冷热浴平均 ({tb['hot_K']}+{tb['cold_K']})/2"
    else:
        t_target, t_note = 300, "文献未报告温度，默认 300 K"

    if stage in ("nvt", "npt"):
        add("ref-t", f"{t_target:g}", "builtin")
        add("gen-vel", "yes", "builtin")
        add("gen-temp", f"{t_target:g}", "builtin")
        add("gen-seed", "-1", "builtin")
        warnings.append(f"目标温度 {t_target:g} K（{t_note}）")
    else:
        add("ref-t", f"{t_target:g}", "builtin")

    lines.append("")

    # ---- 压力耦合 ----
    # 判据取文献报告的系综，与阶段名无关。
    # 阶段名 em/nvt/npt/prod 描述的是**协议**（平衡流程），
    # 而生产段用哪个系综是文献决定的：本文献的 production 就是 NPT。
    ens, _ = _lookup(fields, "ensemble")
    needs_barostat = (ens == "NPT") or (ens is None and stage == "npt")

    if needs_barostat and stage != "em":
        bc, bc_src = _lookup(fields, "barostat")
        if isinstance(bc, dict):
            mapping = {"Parrinello-Rahman": "Parrinello-Rahman", "C-rescale": "C-rescale",
                       "Berendsen": "berendsen", "MTTK": "MTTK",
                       "Monte Carlo": "berendsen"}
            add("pcoupl", mapping.get(bc.get("type"), "C-rescale"), bc_src)
            if bc.get("coupling"):
                add("tau-p", bc["coupling"], bc_src)
            else:
                add("tau-p", DEFAULTS["tau-p"][0], "default")
                warnings.append("文献报告恒压器但未给 tau_p，"
                                "默认 2.0 ps：压力耦合强度影响体积涨落")
        else:
            add("pcoupl", DEFAULTS["pcoupl"][0], "default")
            add("tau-p", DEFAULTS["tau-p"][0], "default")
            warnings.append("文献未报告恒压器，默认 C-rescale；"
                            "若原研究用 Parrinello-Rahman，体积涨落谱会不同")
        add("pcoupltype", "isotropic", "default")
        add("compressibility", DEFAULTS["compressibility"][0], "default")
        add("ref-p", "1.0", "builtin")
    else:
        add("pcoupl", "no", "builtin")
    lines.append("")

    # ---- 未解决字段 ----
    # 静态清单 + 调用方传入的场景性阻断字段。
    # 后者由 validate 的敏感性归因层产出：对热输运研究，system_size 缺失
    # 是致命的；对一般体相性质则否。生成器不自己判断这件事：
    # 那是知识层（schema/sensitivity.yaml）的职责，这里只负责执行结论。
    unresolved = [k for k in list(UNRESOLVED_IF_MISSING) + list(blocking or [])
                  if _lookup(fields, k)[1] == "unresolved"]
    unresolved = sorted(set(unresolved))

    return {
        "stage": stage,
        "mdp": "\n".join(lines) + "\n",
        "provenance": prov,
        "unresolved": unresolved,
        "warnings": warnings,
    }


# -----------------------------------------------------------------------------
# 可执行性检查（不依赖 GROMACS 安装）
#   1. 参数名是否都在白名单内（拼写/非法键）
#   2. 是否有重复键（GROMACS 会静默取最后一个，是隐蔽错误源）
#   3. 未解决字段是否为空
# 若 PATH 中存在 gmx，可额外执行非交互的语法检查。
# -----------------------------------------------------------------------------
def check_mdp(mdp: str) -> Dict[str, Any]:
    errors: List[str] = []
    unknown: List[str] = []
    seen: Dict[str, int] = {}
    n_params = 0

    for raw in mdp.splitlines():
        line = raw.split(";")[0].strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        n_params += 1
        if key in seen:
            errors.append(f"重复参数 {key}（第 {seen[key]} 行与当前行）："
                          f"GROMACS 会静默取最后一个值")
        seen[key] = n_params
        if key not in MDP_KEYS:
            unknown.append(key)

    if unknown:
        errors.append(f"未识别的参数名: {', '.join(sorted(set(unknown)))}")

    return {
        "ok": not errors,
        "n_params": n_params,
        "errors": errors,
        "unknown_keys": sorted(set(unknown)),
    }


def check_grompp_available() -> Optional[str]:
    """若系统装有 GROMACS，返回可执行文件路径，供可选的真实 dry-run 使用。"""
    import shutil
    return shutil.which("gmx") or shutil.which("gmx_mpi")


def check_python_source(src: str) -> Dict[str, Any]:
    """
    用 ast.parse 校验生成的 Python 骨架语法。

    为什么值得单独做：骨架里大量使用 f-string 拼注释与参数，
    缩进或引号写错时**只有到运行时才暴露**，而且报错位置会指向
    生成器的模板而非使用者的输入。在生成端就拦住，成本几乎为零。
    """
    import ast
    try:
        ast.parse(src)
        return {"ok": True, "errors": []}
    except SyntaxError as e:
        return {"ok": False,
                "errors": [f"第 {e.lineno} 行: {e.msg}"]}


# OpenMM 自带 XML 的力场族。不在此列的（GAFF / CGenFF / PCFF / COMPASS /
# DREIDING / ReaxFF 等）需要先经 antechamber / openmmforcefields 转换，
# 直接写成 ForceField("GAFF2.xml") 会在运行时报错找不到文件。
OPENMM_XML_FF = {"AMBER99SB-ILDN", "AMBER14SB", "CHARMM36m", "CHARMM36",
                 "CHARMM27", "Martini"}


def build_openmm_skeleton(fields: Dict[str, Any]) -> str:
    """
    生成 OpenMM 最小骨架。

    与 .mdp 生成器遵守同一条原则：**不静默填默认值**。
    给干态熔体加上溶剂、把水模型写成 "TODO.xml"、把温度与时长硬编码，
    都会让骨架看起来完整、实际与原研究处处不符。
    这正是本模块声称要避免的事，
    所以这里逐项按字段状态分支，缺失项一律以 TODO 显式标出。
    """
    ff, ff_src = _lookup(fields, "force_field")
    wm, wm_src = _lookup(fields, "water_model")
    es, es_src = _lookup(fields, "electrostatics")
    cons, _ = _lookup(fields, "constraints")
    ts, _ = _lookup(fields, "timestep")
    ens, _ = _lookup(fields, "ensemble")
    bc, _ = _lookup(fields, "barostat")
    sl, _ = _lookup(fields, "simulation_length")
    tb, _ = _lookup(fields, "temperature_bath")

    # 温度与时长：能取到就用文献值，取不到才用默认值，且必须标注
    if isinstance(tb, dict) and tb.get("hot_K") and tb.get("cold_K"):
        t_target, t_src = 0.5 * (tb["hot_K"] + tb["cold_K"]), "文献冷热浴平均"
    else:
        t_target, t_src = 300.0, "默认值：文献未报告温度"

    cut = es.get("rcoulomb_nm") if isinstance(es, dict) else None
    cut_src = ("文献报告" if cut else "默认值：文献未报告截断半径")
    cut = cut or 1.0

    ts = ts or 2.0
    ts_src = "文献报告" if _lookup(fields, "timestep")[0] else "默认值：文献未报告时间步"

    # 溶剂：由水模型的适用性决定
    if wm_src == "not_applicable":
        solvent_mode = "none"
    elif wm:
        solvent_mode = "explicit"
    else:
        solvent_mode = "unknown"

    # 体系规模：有盒子尺寸就用来建盒子，没有就 TODO
    box = None
    ss, _ = _lookup(fields, "system_size")
    if isinstance(ss, dict):
        box = ss.get("box_nm")

    # ---- 力场 XML 可用性 ----
    # 注释保持单行：多行注释会插进 ForceField(...) 的实参列表里，
    # 虽然语法合法，但缩进会断掉，读起来像是出错了。
    if ff in OPENMM_XML_FF:
        ff_xml = f'"{ff}.xml"'
        ff_note = "# 文献报告的力场，OpenMM 自带"
    elif ff:
        ff_xml = '"TODO.xml"'
        ff_note = (f'# 文献报告 {ff}，OpenMM 不自带其 XML；'
                   f'需先经 antechamber / openmmforcefields 生成参数文件')
    else:
        ff_xml = '"TODO.xml"'
        ff_note = "# 文献未报告力场：必须补齐后才能运行"

    # ---- 溶剂相关代码块 ----
    if solvent_mode == "none":
        solvent_code = (
            "# 文献为无溶剂体系（真空 / 隐式溶剂 / 干态熔体），不添加溶剂。\n"
            "# 注意：不添加溶剂时不存在水模型，createSystem 不应传入水模型 XML。")
        ff_args = f"    {ff_xml},   {ff_note}"
        add_solvent = "# （无溶剂体系，无 addSolvent 步骤）"
    elif solvent_mode == "explicit":
        solvent_code = (
            f"# 水模型: {wm}（文献报告）\n"
            f"modeller.addSolvent(forcefield, padding=1.0*nanometer,\n"
            f"                     ionicStrength=0.15*molar)  "
            f"# TODO: 离子强度按文献报告值确认")
        ff_args = (f"    {ff_xml},   {ff_note}\n"
                   f'    "{wm}.xml",   # 文献报告的水模型')
        add_solvent = ""
    else:
        solvent_code = (
            "# !! 文献未报告水模型：显式溶剂体系必须补齐，否则无法复现\n"
            "# 在确认之前，本骨架不可运行。")
        ff_args = (f"    {ff_xml},   {ff_note}\n"
                   f'    "TODO.xml",   # 水模型：文献未报告')
        add_solvent = ""

    # ---- 盒子 ----
    if box:
        box_line = f"# 文献报告的盒子边长 {box} nm"
        add_box = (f"modeller.addSolvent(forcefield, boxShape=Vec3({box}, {box}, {box})"
                   f"*nanometer)  # 按文献盒子尺寸" if solvent_mode == "explicit"
                   else f"# 盒子: {box} nm 立方（文献报告）")
    else:
        box_line = "# !! 文献未报告体系尺寸：盒子大小未定义，无法开始复现"
        add_box = "# TODO: 补齐体系尺寸"

    # ---- 恒压器 ----
    if ens == "NPT":
        if isinstance(bc, dict):
            bc_line = (f"system.addForce(MonteCarloBarostat(1*bar, {t_target:g}*kelvin))"
                       f"  # 文献报告 {bc.get('type')}，OpenMM 中以 MonteCarloBarostat 等价实现")
        else:
            bc_line = ("# !! 文献声明 NPT 但未报告恒压器：压力耦合未定义\n"
                       "# system.addForce(MonteCarloBarostat(1*bar, ...))")
    else:
        bc_line = f"# 系综 {ens or '未报告'}：无压力耦合，不添加 barostat"

    # ---- 恒温器说明：按文献实际类型给，不给通用套话 ----
    th, _ = _lookup(fields, "thermostat")
    th_type = th.get("type") if isinstance(th, dict) else None
    if th_type in ("Nose-Hoover", "V-rescale", "CSVR"):
        th_note = (f"# 文献用 {th_type}。OpenMM 无严格对应实现；Langevin 类在弱耦合极限下\n"
                   f"# 与 Berendsen 接近，但与 {th_type} 的涨落谱不同，\n"
                   f"# 跨软件复现本身即存在系统差异，需在结果中说明。")
    elif th_type == "Berendsen":
        th_note = ("# 文献用 Berendsen 恒温器：该恒温器压制涨落，\n"
                   "# 若原研究涉及扩散系数 / 热导率等输运性质，其结果存在已知系统偏差。\n"
                   "# OpenMM 中以 Langevin 实现，涨落行为与原研究不同。")
    elif th_type == "Langevin":
        th_note = "# 文献用 Langevin 恒温器，与下方积分器一致。"
    else:
        th_note = "# !! 文献未报告恒温器：恒温器类型直接决定涨落谱，必须补齐。"

    # ---- 时长 ----
    if isinstance(sl, dict) and sl.get("length"):
        L, unit = sl["length"], sl.get("unit", "ns")
        ns = L * (1000 if unit == "us" else (1 if unit == "ns" else 0.001))
        nsteps = int(ns * 1e6 / ts)
        dur_note = f"# {L} {unit}（文献报告）"
        if not sl.get("replicas"):
            dur_note += "；!! 文献未报告重复次数，以下仅为单条轨迹"
    else:
        nsteps = int(100 * 1e6 / ts)
        dur_note = "# !! 文献未报告时长，此处为 100 ns 占位：必须替换"

    return f'''# -*- coding: utf-8 -*-
"""
由 md-repro-checker 自动生成：OpenMM 最小骨架

逐项来源（与 .mdp 生成器同一套标注规则）：
  力场    : {ff or "【文献未报告】"}   [{ff_src}]
  水模型  : {wm or "【不适用】"}   [{wm_src}]
  系综    : {ens or "【文献未报告】"}
  温度    : {t_target:g} K      [{t_src}]
  时间步  : {ts} fs           [{ts_src}]
  截断    : {cut} nm          [{cut_src}]
  约束    : {cons or "h-bonds（默认）"}

凡标 TODO 或 !! 处，均不得在未确认的情况下运行。
"""
from openmm.app import *
from openmm import *
from openmm.unit import *

# TODO: 用文献报告的初始结构替换（来源: 需人工确认）
pdb = PDBFile("input.pdb")
{box_line}

# 力场选择：必须与文献完全一致，否则结果不可比
forcefield = ForceField(
{ff_args}
)

modeller = Modeller(pdb.topology, pdb.positions)
{add_box}
{solvent_code}

system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff={cut}*nanometer,
    constraints={"HBonds" if cons in (None, "LINCS", "SHAKE", "h-bonds only") else "AllBonds"},
    ewaldErrorTolerance=0.0005,
)

{bc_line}

{th_note}
integrator = LangevinMiddleIntegrator({t_target:g}*kelvin, 1/picosecond, {ts}*femtoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()

# 生产段
{dur_note}
simulation.step({nsteps})
'''


# =============================================================================
# LAMMPS 输入脚本生成
#
# units real：距离 Å · 时间 fs · 能量 kcal/mol · 温度 K · 压强 atm
#
# 与 .mdp 生成器的三点关键差异，每一条都是"看起来等价其实不等价"的地方：
#
#   1. timestep 在 real 单位下**就是 fs**，与文献报告单位一致，无需换算。
#      （GROMACS 的 dt 单位是 ps，必须做 fs->ps 换算；写错时模拟会瞬间炸开，
#        而报错信息不会指向单位那一行。）
#
#   2. LAMMPS 不读取力场库。pair_coeff / bond_coeff / angle_coeff ... 全部
#      由力场参数文件提供，生成器无法凭空补齐。因此输出 include TODO.forcefield，
#      不伪造一套参数：伪造的力场跑得起来，但跑的不是原来那个体系。
#
#   3. special_bonds（1-4 非键缩放）由**力场族**决定，不由使用者随手写。
#      它必须在所有阶段保持一致：平衡段用 AMBER 约定、生产段改用 OPLS 约定，
#      等于"被平衡的体系和被模拟的体系不是同一个体系"。
#      生成器把它挂在力场上，从源头消除这一类不一致。
#
#   4. pair_modify mix（LJ 交叉项混合规则）同样由力场族决定，且**有默认值**。
#      默认是 geometric（σ 取几何平均），而 AMBER/GAFF/CHARMM 要的是
#      Lorentz-Berthelot（σ 取算术平均），COMPASS/PCFF 要的是六次方规则。
#      **OPLS-AA 是例外：它用几何平均，与默认值一致**（详见 MIXING_BY_FF
#      上方注释）。所以"漏写 mix"这件事对哪个力场有害，取决于力场族。
#      漏写这一行时 LAMMPS 不报错、不警告，只是所有 I≠J 交叉对的作用距离
#      系统性偏移。注意 ε 在 geometric 与 arithmetic 下**是同一个值**
#      （源码 src/pair.cpp 的 mix_energy() 两种规则都返回 sqrt(eps1*eps2)），
#      只有 σ 不同，所以这个错误从能量项上完全看不出来。
#      日志里会打印 "Generated N of M mixed pair_coeff terms from X mixing rule"，
#      可用于自查实际生效的规则。
# =============================================================================
LAMMPS_COMMANDS = {
    "units", "dimension", "atom_style", "boundary", "newton", "processors",
    "include", "read_data", "read_restart", "read_dump",
    "pair_style", "pair_coeff", "pair_modify",
    "bond_style", "bond_coeff", "angle_style", "angle_coeff",
    "dihedral_style", "dihedral_coeff", "improper_style", "improper_coeff",
    "special_bonds", "kspace_style", "kspace_modify",
    "neighbor", "neigh_modify", "comm_modify",
    "group", "region", "variable", "compute", "fix", "fix_modify", "unfix",
    "timestep", "velocity", "mass", "set", "delete_atoms",
    "thermo", "thermo_style", "thermo_modify",
    "run", "minimize", "min_style", "min_modify", "reset_timestep",
    "write_data", "write_restart", "write_dump", "dump", "dump_modify",
    "log", "print", "shell", "jump", "if", "label", "next", "clear", "info",
}

# 力场族 -> special_bonds 约定。1-4 非键缩放是力场定义的一部分，
# 各阶段必须一致。
SPECIAL_BONDS_BY_FF = [
    (r"^(AMBER|GAFF|ff99|ff14)", "amber",
     "AMBER/GAFF 约定：1-4 库仑缩放 5/6，1-4 LJ 缩放 1/2"),
    (r"^CHARMM", "charmm",
     "CHARMM 约定：1-4 库仑与 LJ 均不缩放"),
    (r"^OPLS", "lj/coul 0.0 0.0 0.5",
     "OPLS 约定：1-4 库仑缩放 1/2，1-4 LJ 缩放 1/2"),
    (r"^(COMPASS|PCFF|CFF|DREIDING)", "lj/coul 0.0 0.0 0.0",
     "COMPASS/PCFF 约定：1-4 非键完全不参与"),
]

# 力场族 -> LJ 交叉项混合规则（pair_modify mix）。
#
# 这是一条纯粹的"静默失效"：LAMMPS 的 mix 有默认值 geometric，
# 漏写不报错、不警告，但所有 I≠J 交叉对的 σ 都偏了。
# ε 不受影响（geometric 与 arithmetic 下 mix_energy() 都返回 sqrt(eps1*eps2)），
# 所以从能量项上看不出任何异常。
#
# **OPLS 是这条规律里的例外，必须单独记住。** 常见的说法是
# "生物力场用 Lorentz-Berthelot"，但那是 AMBER / GAFF / CHARMM 的情况；
# OPLS-AA 用的是**几何平均**（σ 与 ε 都取几何平均），恰好与 LAMMPS
# 的默认值一致。于是同一个"漏写 mix"的动作，对 AMBER 家族是错的、
# 对 OPLS 反而是对的。判据因此不能按"这一行在不在"来定，只能按力场族来定
# ——这正是本表存在、且 `--check-lammps` 只能报警不能报错的原因。
#
# DREIDING **刻意不收**：它的混合规则我未能在合理成本内坐实，
# 与其猜一个值写进生成器（伪造参数的危害见本文件第 2 条），
# 不如让它落到 TODO 分支：与"拒绝伪造力场参数"是同一条原则。
# 注意这与 SPECIAL_BONDS_BY_FF 里 DREIDING 归在 COMPASS 组不冲突：
# 那是 1-4 缩放，与此处是两件事。
MIXING_BY_FF = [
    (r"^(AMBER|GAFF|ff99|ff14)", "arithmetic",
     "AMBER/GAFF 用 Lorentz-Berthelot：ε 几何平均、σ 算术平均"),
    (r"^CHARMM", "arithmetic",
     "CHARMM 用 Lorentz-Berthelot：ε 几何平均、σ 算术平均"),
    (r"^OPLS", "geometric",
     "OPLS-AA 用几何平均（σ 与 ε 均取几何平均），与 LAMMPS 默认值一致，仍显式写出"),
    (r"^(COMPASS|PCFF|CFF)", "sixthpower",
     "COMPASS/PCFF 用六次方组合规则：σ=(σi⁶+σj⁶)/2 开六次方，ε 亦按该规则缩放"),
]

# 力场族 -> LAMMPS 需要显式声明的键合 style。
# 只在力场已知时输出：写错一个 dihedral_style 会静默改变势能面，
# 比留 TODO 危险得多。
LAMMPS_FF_STYLES = {
    "GAFF2": {"bond": "harmonic", "angle": "harmonic",
              "dihedral": "harmonic", "improper": "cvff"},
    "GAFF": {"bond": "harmonic", "angle": "harmonic",
             "dihedral": "harmonic", "improper": "cvff"},
    "AMBER99SB-ILDN": {"bond": "harmonic", "angle": "harmonic",
                       "dihedral": "harmonic", "improper": "cvff"},
    "AMBER14SB": {"bond": "harmonic", "angle": "harmonic",
                  "dihedral": "harmonic", "improper": "cvff"},
    "CHARMM36m": {"bond": "harmonic", "angle": "charmm",
                  "dihedral": "charmm", "improper": "harmonic"},
    "CHARMM36": {"bond": "harmonic", "angle": "charmm",
                 "dihedral": "charmm", "improper": "harmonic"},
    "OPLS-AA": {"bond": "harmonic", "angle": "harmonic",
                "dihedral": "opls", "improper": "harmonic"},
}


def check_lammps(script: str) -> Dict[str, Any]:
    """
    LAMMPS 脚本的静态检查（不依赖安装 LAMMPS）：

      1. 每条命令的首 token 是否在白名单内（拼写/非法命令）
      2. fix / compute / group / region / variable 的 ID 是否重复
         LAMMPS 对重复 ID 直接报错，但生成器的模板拼接出错时
            往往到运行才暴露，且报错位置指向脚本而非模板
      3. 恒温 fix 的 group 冲突
         fix langevin / fix nvt 等的受力范围由 **fix 自己的 group** 决定，
            与 fix_modify temp 指定的 compute 的 group 无关。两个恒温器
            都写 group = all 时，全体原子会被两个不同温度的浴同时作用，
            温度梯度被抹平：脚本照常运行，结果无意义。
      4. 有 run 步数却无积分器：原子不动，空转到底（见下方注释）
      5. 有 lj/cut 系 pair_style 却未声明混合规则：静默采用 mix geometric

    1-4 进 errors（脚本一定有问题），5 只进 warnings：无法从脚本反推力场族，
    而"所有交叉对都由 pair_coeff 显式给定"的脚本确实不需要那一行。
    """
    errors: List[str] = []
    warnings: List[str] = []
    unknown: List[str] = []
    ids: Dict[str, Dict[str, int]] = {}
    thermostats: List[Tuple[str, str]] = []
    n_commands = 0
    has_integrator = False
    has_minimize = False
    has_positive_run = False
    has_lj_cut = False
    declares_mix = False

    # LAMMPS 用行尾 & 续行。必须先合并成逻辑行再解析，
    # 否则续行会被当成独立命令（如 fix ... &\n  file x.dat 里的 file）。
    logical: List[Tuple[int, str]] = []
    buf, start = "", 0
    for i, raw in enumerate(script.splitlines(), 1):
        line = raw.split("#")[0].rstrip()
        if not buf:
            start = i
        if line.endswith("&"):
            buf += line[:-1] + " "
            continue
        buf += line
        if buf.strip():
            logical.append((start, buf.strip()))
        buf = ""
    if buf.strip():
        logical.append((start, buf.strip()))

    for lineno, line in logical:
        parts = line.split()
        cmd = parts[0]
        n_commands += 1

        if cmd not in LAMMPS_COMMANDS:
            unknown.append(cmd)
            continue

        if cmd in ("fix", "compute", "group", "region", "variable") and len(parts) >= 2:
            ids.setdefault(cmd, {})
            if parts[1] in ids[cmd]:
                errors.append(f"{cmd} ID 重复: {parts[1]}"
                              f"（第 {ids[cmd][parts[1]]} 行与第 {lineno} 行）")
            else:
                ids[cmd][parts[1]] = lineno

        # fix 的 style 在第四列
        if cmd == "fix" and len(parts) >= 4:
            style = parts[3]
            if style in ("langevin", "nvt", "npt", "temp/berendsen",
                         "temp/rescale", "temp/csvr", "nvt/asphere"):
                thermostats.append((style, parts[2]))
            # nve/nvt/npt/nph 及其变体（nve/asphere、nvt/sphere…）才是积分器。
            # fix langevin 只加力不推进时间，故不在其中。
            if style.split("/")[0] in ("nve", "nvt", "npt", "nph"):
                has_integrator = True

        if cmd == "minimize":
            has_minimize = True

        # 注释已在上面按 # 剥掉，因此被注释掉的 mix 行不会误判为"已声明"。
        if cmd == "pair_style" and len(parts) >= 2 and parts[1].startswith("lj/cut"):
            has_lj_cut = True
        if cmd == "pair_modify" and "mix" in parts[1:]:
            declares_mix = True

        # run 0 只建邻接表/触发一次力计算，不推进时间
        if cmd == "run" and len(parts) >= 2:
            try:
                if float(parts[1]) > 0:
                    has_positive_run = True
            except ValueError:
                has_positive_run = True      # 写成变量时按推进处理

    if unknown:
        errors.append(f"未识别的命令: {', '.join(sorted(set(unknown)))}")

    # 两个恒温器都作用于 all -> 它们作用的原子集合完全重合
    on_all = [s for s, g in thermostats if g == "all"]
    if len(on_all) >= 2:
        errors.append(
            f"有 {len(on_all)} 个恒温 fix 的 group 都是 all"
            f"（{', '.join(on_all)}）：全体原子被多个浴同时控温，"
            f"温度梯度会被抹平。恒温 fix 的 group 应写实际控温的区域，"
            f"如 hot / cold，不要写 all")

    # 有实际推进的 run，却没有任何积分器 -> 原子永远不动。
    # 这是最安静的一类错误：LAMMPS 不报错、跑满全部步数、
    # 输出文件齐全且数值稳定，只是物理上什么都没发生。
    if has_positive_run and not has_integrator and not has_minimize:
        errors.append(
            "脚本有 run 步数 > 0，但没有任何积分器 fix"
            "（nve/nvt/npt/nph）：原子不会移动，"
            "模拟会跑完并输出一份看似正常、实则空转的结果")

    # LJ 混合规则。这是警告而非错误：脚本能跑，也可能完全正确
    # （交叉对若由 pair_coeff 逐条显式给出，就不需要这一行）。
    # 但只要存在自动生成的交叉对，漏写就会让它们的 σ 系统性偏移，
    # 而 LAMMPS 不会给出任何提示。
    if has_lj_cut and not declares_mix:
        warnings.append(
            "有 pair_style lj/cut* 但未声明 pair_modify mix：LAMMPS 会静默采用"
            "默认的 mix geometric（σ 取几何平均）。若这份脚本用的是 "
            "AMBER/GAFF/CHARMM，它们要的是 Lorentz-Berthelot（σ 取算术平均），"
            "交叉对的作用距离就会偏移；COMPASS/PCFF 要六次方规则，同样对不上。"
            "但 OPLS-AA 用的就是几何平均，对它是正确的——"
            "本检查无法从脚本反推力场族，因此只能提示，不能判定为错误。"
            "（注意 ε 在 geometric 与 arithmetic 下相同，能量项上看不出差别）")

    return {
        "ok": not errors,
        "n_commands": n_commands,
        "errors": errors,
        "warnings": warnings,
        "unknown_commands": sorted(set(unknown)),
        "thermostats": [f"{s}@{g}" for s, g in thermostats],
    }


def build_lammps(fields: Dict[str, Any], stage: str = "prod",
                 blocking: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    stage ∈ {em, nvt, npt, prod}，与 .mdp 生成器同构。
    返回 {"script", "provenance", "unresolved", "warnings"}
    """
    lines: List[str] = []
    prov: Dict[str, str] = {}
    warnings: List[str] = []

    def add(key: str, text: str, source: str) -> None:
        lines.append(text)
        prov[key] = source

    def comment(text: str = "") -> None:
        lines.append(("# " + text).rstrip() if text else "#")

    ff, ff_src = _lookup(fields, "force_field")
    ens, _ = _lookup(fields, "ensemble")
    ts, ts_src = _lookup(fields, "timestep")
    es, es_src = _lookup(fields, "electrostatics")
    dc, dc_src = _lookup(fields, "dispersion_correction")
    cons, cons_src = _lookup(fields, "constraints")
    tc, tc_src = _lookup(fields, "thermostat")
    bc, _ = _lookup(fields, "barostat")
    tb, _ = _lookup(fields, "temperature_bath")
    sl, sl_src = _lookup(fields, "simulation_length")
    hf, hf_src = _lookup(fields, "heat_flux_method")
    init, init_src = _lookup(fields, "initial_structure")

    # ---- 基本设置 ----
    add("units", "units           real", "builtin")
    comment("units real: 距离 A / 时间 fs / 能量 kcal/mol / 温度 K / 压强 atm")
    add("atom_style", "atom_style      full", "builtin")
    comment("full: 带电荷与分子 ID；聚合物/生物体系的标准选择")
    add("boundary", "boundary        p p p", "builtin")
    lines.append("")

    # ---- 力场 ----
    # LAMMPS 不从力场库读参数，这里只能给出声明与占位，不能伪造系数。
    if ff:
        comment(f"力场: {ff}   [{ff_src}]")
        styles = LAMMPS_FF_STYLES.get(ff)
        if styles:
            for k in ("bond", "angle", "dihedral", "improper"):
                add(f"{k}_style", f"{k + '_style':<16}{styles[k]}", "default")
            warnings.append(
                f"键合 style 由力场名 {ff} 推得，属工具假设："
                f"请与力场参数文件核对；写错 style 会静默改变势能面")
        else:
            comment(f"!! 力场 {ff} 对应的 LAMMPS style 未知，须人工填写")
            for k in ("bond", "angle", "dihedral", "improper"):
                add(f"{k}_style", f"# {k + '_style':<15}TODO", "unresolved")

        sb = None
        for pat, val, why in SPECIAL_BONDS_BY_FF:
            if re.match(pat, ff):
                sb = (val, why)
                break
        if sb:
            add("special_bonds", f"special_bonds   {sb[0]}", ff_src)
            comment(f"{sb[1]}；此值由力场决定，所有阶段必须一致")
        else:
            add("special_bonds", "# special_bonds   TODO  # 按力场族确定 1-4 缩放",
                "unresolved")
    else:
        comment("!! 文献未报告力场：无参数则 LAMMPS 无法开始")
        add("force_field", "# include         TODO.forcefield", "unresolved")

    add("forcefield_include", "include         TODO.forcefield", ff_src if ff else "unresolved")
    comment("↑ 须含 pair_coeff / bond_coeff / angle_coeff / dihedral_coeff / improper_coeff")
    comment("  LAMMPS 不读取力场库；用 fftool / msi2lmp / Moltemplate / 自备参数文件生成")
    lines.append("")

    # ---- 初始结构 ----
    if init:
        comment(f"初始结构: {init}   [{init_src}]")
    else:
        comment("!! 文献未报告初始结构来源：无结构文件则无法开始")
        warnings.append("文献未报告初始结构，read_data 指向 TODO.data，需人工提供")
    add("read_data", "read_data       TODO.data", init_src if init else "unresolved")
    lines.append("")

    # ---- 非键与静电 ----
    cut_nm = es.get("rcoulomb_nm") if isinstance(es, dict) else None
    method = es.get("method") if isinstance(es, dict) else None
    cut_a = f"{cut_nm * 10:g}" if cut_nm else "10.0"

    if cut_nm:
        cut_src = es_src
    else:
        cut_src = "default"
        warnings.append("文献未报告截断半径，默认 10.0 A；"
                        "若原研究用力场为 CHARMM36 系，真实值常为 12.0 A")

    if method == "PME":
        # 长程静电在 LAMMPS 里必须成对出现：pair_style 用 coul/long，
        # 且必须有 kspace_style。只写 coul/cut 会静默退化为纯截断库仑。
        add("pair_style", f"pair_style      lj/cut/coul/long {cut_a}", es_src)
        add("kspace_style", "kspace_style    pppm 1.0e-4", "default")
        comment("!! PME 必须成对出现：pair_style 用 coul/long + kspace_style pppm。")
        comment("   写成 lj/cut/coul/cut 且不给 kspace_style 时，LAMMPS 不会报错，")
        comment("   只是静默退化为纯截断库仑：结果与原文不可比。")
    elif method:
        add("pair_style", f"pair_style      lj/cut/coul/cut {cut_a}", es_src)
        comment(f"静电方法 {method}：LAMMPS 中以截断库仑实现，无 kspace")
    else:
        add("pair_style", f"pair_style      lj/cut/coul/long {cut_a}", "default")
        add("kspace_style", "kspace_style    pppm 1.0e-4", "default")
        warnings.append("文献未报告静电处理方法，默认 PME(pppm)")

    if dc and dc != "none":
        add("pair_modify", "pair_modify     tail yes", dc_src)
        comment("tail yes: 补偿截断 LJ 的长程部分（对应 GROMACS 的 DispCorr EnerPres）")
    else:
        add("pair_modify", "pair_modify     tail yes", "default")
        warnings.append("文献未报告长程色散校正，默认启用 tail correction")

    # LJ 交叉项混合规则。与上面这些不同，它**有默认值**：漏写时 LAMMPS
    # 静默采用 geometric，而多数生物力场要的是 arithmetic：见文件头第 4 条。
    # 注意 provenance key 不能复用 "pair_modify"，否则会覆盖 tail 行的来源。
    mix = None
    for pat, val, why in MIXING_BY_FF:
        if ff and re.match(pat, ff):
            mix = (val, why)
            break
    if mix:
        add("pair_modify_mix", f"pair_modify     mix {mix[0]}", ff_src)
        comment(f"{mix[1]}")
        comment("LAMMPS 默认是 geometric；漏写不报错，只让所有 I≠J 交叉对的 σ 偏移")
    else:
        add("pair_modify_mix",
            "# pair_modify     mix TODO  # 按力场族确定 σ 混合规则", "unresolved")
        comment("  AMBER/GAFF/CHARMM 需 arithmetic，COMPASS/PCFF 需 sixthpower；")
        comment("  OPLS-AA 用 geometric，与默认值一致，但显式写出更稳；")
        comment("  漏写时 LAMMPS 不报错、不警告，只有 σ 静默偏移")
    lines.append("")

    # ---- 近邻 ----
    add("neighbor", "neighbor        2.0 bin", "default")
    add("neigh_modify", "neigh_modify    every 1 delay 0 check yes", "builtin")
    lines.append("")

    # ---- 积分与控温 ----
    if ts is None:
        ts, ts_src = 1.0, "default"
        warnings.append("文献未报告时间步，默认 1.0 fs")
    # units real 下 timestep 的单位就是 fs，与文献报告一致，不做换算。
    add("timestep", f"timestep        {ts:g}", ts_src)
    comment("units real 下 timestep 单位为 fs，与文献报告单位一致")

    if stage == "em":
        add("min_style", "min_style       cg", "builtin")
        add("minimize", "minimize        1.0e-4 1.0e-6 1000 10000", "builtin")

    # 目标温度
    if isinstance(tb, dict) and tb.get("hot_K") and tb.get("cold_K"):
        t_target = 0.5 * (tb["hot_K"] + tb["cold_K"])
        t_note = f"取冷热浴平均 ({tb['hot_K']}+{tb['cold_K']})/2"
    else:
        t_target, t_note = 300, "文献未报告温度，默认 300 K"

    seed = 12345
    add("seed", f"variable        seed equal {seed}", "builtin")
    comment("固定随机种子：换种子须重跑并重新记录，否则轨迹不可复现")

    # 直接 NEMD 的生产段**不能**再给全体原子加恒温器：
    # 控温由 hot / cold 两个浴负责，中间区域必须是 NVE。
    # 若这里再补一个 fix nvt all，温度梯度会被第二个恒温器抹平：
    # 脚本照常运行，κ 无意义。所以整个全局恒温段在这里让位。
    nemd_direct = stage == "prod" and hf in ("Direct NEMD", "NEMD")
    if nemd_direct and ens in ("NPT", "NVT"):
        warnings.append(f"NEMD 生产段本不应有全局 {ens} 恒温器（控温由冷热浴承担）；"
                        f"已按 NEMD 要求省略全局控温与控压，"
                        f"原文若确实如此设置，需人工核对")

    if stage != "em" and not nemd_direct:
        warm = _lammps_thermostat(ens, tc, t_target, warnings)
        if stage in ("nvt", "npt"):
            add("velocity_create",
                f"velocity        all create {t_target:g} ${{seed}} dist gaussian",
                "builtin")
            warnings.append(f"目标温度 {t_target:g} K（{t_note}）")

        if ens == "NPT" and stage != "em":
            pdamp = "1000.0"
            if isinstance(bc, dict) and bc.get("coupling"):
                pdamp = f"{float(bc['coupling']) * 1000:g}"   # ps -> fs
                pdamp_src = "paper"
            else:
                pdamp_src = "default"
                warnings.append("文献报告恒压器但未给 tau_p，默认 1000 fs (1 ps)")
            # 与 .mdp 生成器保持一致：文献报的是 NPT，则各阶段都按 NPT 建模
            # （nvt.mdp 同样带 pcoupl）。此处不能按阶段名分别处理：
            # 那会让 nvt 阶段既拿不到 nvt fix、也拿不到 npt fix，
            # 生成一个**没有任何积分器**的脚本：LAMMPS 不报错，
            # 只是原子永远不动，跑满 1000 万步后得到一个空结果。
            add("fix_integrate",
                f"fix             integrate all npt temp {t_target:g} {t_target:g} "
                f"{warm} iso 1.0 1.0 {pdamp}", warm_src(ens, tc))
            add("Pdamp", f"# Pdamp = {pdamp} fs", pdamp_src)
        elif warm == "langevin":
            add("fix_integrate",
                f"fix             integrate all langevin {t_target:g} {t_target:g} "
                f"100.0 ${{seed}}", warm_src(ens, tc))
            comment("!! fix langevin 不做时间积分，必须另配 fix nve")
            add("fix_nve", "fix             nve_all all nve", "builtin")
        elif warm is None:
            add("fix_integrate", "fix             integrate all nve", "builtin")
        else:
            damp = "100.0"
            if isinstance(tc, dict) and tc.get("coupling"):
                damp = f"{float(tc['coupling']) * 1000:g}"    # ps -> fs
            add("fix_integrate",
                f"fix             integrate all nvt temp {t_target:g} {t_target:g} "
                f"{damp}", warm_src(ens, tc))
            if tc.get("type") == "Berendsen" if isinstance(tc, dict) else False:
                warnings.append("文献使用 Berendsen 恒温器。该恒温器压制涨落，"
                                "若原研究涉及扩散系数/热导率等输运性质，"
                                "其结果存在已知的系统偏差")

    # ---- 生产段：热输运专项 ----
    if stage == "prod" and hf:
        lines.append("")
        lines.extend(_lammps_heat_transport(hf, tb, add, warnings))

    # ---- 输出与运行 ----
    lines.append("")
    add("thermo", "thermo          1000", "builtin")
    add("thermo_style",
        "thermo_style    custom step temp pe ke etotal press vol", "builtin")
    add("thermo_modify", "thermo_modify   flush yes", "builtin")

    if stage == "em":
        add("run", "run             0", "builtin")
    else:
        if isinstance(sl, dict) and sl.get("length"):
            L, unit = sl["length"], sl.get("unit", "ns")
            fs = L * (1e9 if unit == "us" else (1e6 if unit == "ns" else 1e3))
            nsteps = int(fs / ts)
            add("run", f"run             {nsteps}", sl_src)
            comment(f"{L} {unit} / {ts:g} fs = {nsteps} 步")
            if not sl.get("replicas"):
                warnings.append("文献未报告重复次数，run 仅对应单条轨迹")
        else:
            add("run", "run             50000000", "default")
            comment("100 ns / 1 fs")
            warnings.append("文献未报告模拟时长，默认 100 ns")

    # ---- 阶段交接 ----
    lines.append("")
    add("reset_timestep", "reset_timestep  0", "builtin")
    if stage == "em":
        add("write_data", "write_data      relaxed.data nocoeff", "builtin")
    elif stage in ("nvt", "npt"):
        add("write_data", f"write_data      equil_{stage}.data nocoeff", "builtin")
    else:
        add("write_data", "write_data      final.data nocoeff", "builtin")

    unresolved = [k for k in list(UNRESOLVED_IF_MISSING) + list(blocking or [])
                  if _lookup(fields, k)[1] == "unresolved"]
    unresolved = sorted(set(unresolved))

    return {
        "stage": stage,
        "script": "\n".join(lines) + "\n",
        "provenance": prov,
        "unresolved": unresolved,
        "warnings": warnings,
    }


def warm_src(ens: Any, tc: Any) -> str:
    """控温 fix 的来源：文献报了恒温器就是 paper，没报就是工具默认。"""
    return "paper" if isinstance(tc, dict) and tc.get("type") else "default"


def _lammps_thermostat(ens: Any, tc: Any, t_target: float,
                       warnings: List[str]) -> Optional[str]:
    """
    返回需要插入 nvt/npt 行的恒温器名：nose-hoover 返回 "nose-hoover" 以外的
    占位（fix nvt 已经隐含 Nose-Hoover），Langevin 返回 "langevin"，
    NVE 返回 None。
    """
    if ens == "NVE":
        return None
    if isinstance(tc, dict):
        t = tc.get("type")
        if t == "Langevin":
            return "langevin"
        if t == "Nose-Hoover":
            return "nose-hoover"
        if t in ("V-rescale", "CSVR", "Andersen"):
            warnings.append(f"文献使用 {t} 恒温器，LAMMPS 无对应实现："
                            f"已用 Nose-Hoover (fix nvt) 替代，"
                            f"涨落行为与原研究不同")
            return "nose-hoover"
        if t == "Berendsen":
            warnings.append("文献使用 Berendsen 恒温器，LAMMPS 无对应实现："
                            "已用 Nose-Hoover (fix nvt) 替代")
            return "nose-hoover"
    else:
        warnings.append("文献未报告恒温器，默认 Nose-Hoover (fix nvt)")
    return "nose-hoover"


def _lammps_heat_transport(hf: str, tb: Any,
                           add, warnings: List[str]) -> List[str]:
    """
    热输运专项段。三个实现分支：Green-Kubo / 直接 NEMD / RNEMD。

    这一段的每条命令都对应一个"写错了不报错、只是数不对"的坑：
      · 热流用 centroid/stress/atom 而非 stress/atom（手册明确警告）
      · 归一化体积必须是**测量组所在**的体积
      · 恒温 fix 的 group 必须是冷热区，不能是 all
      · 区域厚度必须取**温度梯度方向**的长度
    """
    L: List[str] = []
    L.append("# " + "=" * 70)
    L.append(f"# 热输运：{hf}   [paper]")
    L.append("# " + "=" * 70)

    hot_K = tb.get("hot_K") if isinstance(tb, dict) else None
    cold_K = tb.get("cold_K") if isinstance(tb, dict) else None

    # 只有直接 NEMD 是"用冷热浴施加 ΔT、再测响应"。
    # Green-Kubo 是平衡态方法，RNEMD 的梯度来自动能交换：
    # 对这两者，文献里的冷热浴数值不是梯度来源，套用 ΔT/T_avg 判据没有意义。
    imposes_dT = hf in ("Direct NEMD", "NEMD")
    if hot_K and cold_K and imposes_dT:
        t_avg = 0.5 * (hot_K + cold_K)
        ratio = (hot_K - cold_K) / t_avg if t_avg else 0.0
        L.append(f"# 冷热浴 {hot_K} K / {cold_K} K，ΔT/T_avg = {ratio:.1%}")
        if ratio > 0.10:
            L.append("# !! 温差已超过平均温度的 10%，可能偏离线性响应区：")
            L.append("#    此时 κ 依赖于 ΔT 本身，不再是材料常数。")
            L.append("#    验证方式：做 ΔT 扫描（如 0.5ΔT / 1.0ΔT / 1.5ΔT），")
            L.append("#    确认 κ 不随 ΔT 变；否则须减小 ΔT 重跑。")
            warnings.append(f"NEMD 冷热浴温差占平均温度 {ratio:.0%}，"
                            f"超过 10%：可能已偏离线性响应区，"
                            f"κ 将依赖于 ΔT 本身")
    elif hot_K and cold_K:
        L.append(f"# 文献报告了冷热浴 {hot_K} K / {cold_K} K，但 {hf} 不靠浴施加 ΔT，")
        L.append(f"# 故此处不套用 ΔT/T_avg 判据；这些数值在本文中不参与梯度设置。")
        warnings.append(f"文献报告冷热浴温度，但热流方法为 {hf}："
                        f"该方法的梯度不来自浴，温度浴字段按不适用处理；"
                        f"若原文实为 NEMD，说明方法抽取有误")

    if hf == "Green-Kubo":
        L.append("")
        L.append("# Green-Kubo：平衡态下测量热流自关联，κ 由关联函数积分得到。")
        L.append("# 关键在统计：κ(t) 必须取自平台区，取在未收敛区会系统性低估。")
        L.append("compute         ke_atom   all ke/atom")
        L.append("compute         pe_atom   all pe/atom")
        L.append("compute         strs_atom all centroid/stress/atom NULL virial")
        L.append("#   ↑ 用 centroid/stress/atom，不要用 stress/atom：")
        L.append("#     LAMMPS 手册明确警告 stress/atom 对 angle / dihedral /")
        L.append("#     improper / constraint 的贡献会给出非物理值")
        L.append("#     （Surblys2019, Boone, Surblys2021）。")
        L.append("compute         jflux   all heat/flux ke_atom pe_atom strs_atom")
        L.append("")
        L.append("# 归一化体积：group 为 all 时用整盒体积")
        L.append("variable        Jx equal c_jflux[1]/vol")
        L.append("variable        Jy equal c_jflux[2]/vol")
        L.append("variable        Jz equal c_jflux[3]/vol")
        L.append("")
        L.append("# 关联长度 p、采样间隔 s、输出间隔 d 需按体系调整；")
        L.append("# 这三个值决定了积分上限与统计精度，属必须报告的参数。")
        L.append("variable        p equal 2000     # 关联长度 (步)")
        L.append("variable        s equal 10       # 采样间隔 (步)")
        L.append("variable        d equal ${p}*${s}")
        L.append("fix             JJ all ave/correlate ${s} ${p} ${d} &")
        L.append("                c_jflux[1] c_jflux[2] c_jflux[3] type auto &")
        L.append("                file J0Jt.dat ave running")
        L.append("")
        L.append("# κ = V/(kB T^2) * ∫<J(0)J(t)>dt ；单位换算见 LAMMPS 手册")
        L.append("#   heat/flux 页面的固体 Ar 示例。")
        L.append("")
        L.append("# !! 统计窗口与关联时间（correlation_window）未报告时，")
        L.append("#    上面三个 variable 只是工具的建议值，不构成对原研究的复现。")

    elif hf in ("Direct NEMD", "NEMD"):
        L.append("")
        L.append("# 直接 NEMD：两端设固定壁，紧邻固定壁设冷/热浴，")
        L.append("# 中间为 NVE 区，在**盒子中央**测量热流（避开恒温浴的扰动）。")
        L.append("")
        L.append("# 几何：温度梯度沿 x。区域厚度必须取梯度方向的长度 Lx，")
        L.append("# 取 Lz 会让切片厚度错误：脚本照常运行，梯度却是错的。")
        L.append("variable        Lx equal (xhi-xlo)")
        L.append("variable        w_wall  equal 0.05*${Lx}")
        L.append("variable        w_bath  equal 0.10*${Lx}")
        L.append("variable        w_slab  equal 0.10*${Lx}")
        L.append("")
        L.append("variable        wallL_lo equal xlo")
        L.append("variable        wallL_hi equal xlo+${w_wall}")
        L.append("variable        bathH_lo equal ${wallL_hi}")
        L.append("variable        bathH_hi equal ${bathH_lo}+${w_bath}")
        L.append("variable        wallR_hi equal xhi")
        L.append("variable        wallR_lo equal xhi-${w_wall}")
        L.append("variable        bathC_hi equal ${wallR_lo}")
        L.append("variable        bathC_lo equal ${bathC_hi}-${w_bath}")
        L.append("variable        slab_c   equal (xlo+xhi)/2.0")
        L.append("variable        slab_lo  equal ${slab_c}-${w_slab}/2.0")
        L.append("variable        slab_hi  equal ${slab_c}+${w_slab}/2.0")
        L.append("")
        L.append("#   units box 必须显式写出：region 默认按 lattice 单位解释边界，")
        L.append("#   而 read_data 读入的体系没有 lattice 定义。")
        L.append("region          wallL block ${wallL_lo} ${wallL_hi} INF INF INF INF units box")
        L.append("region          bathH block ${bathH_lo} ${bathH_hi} INF INF INF INF units box")
        L.append("region          bathC block ${bathC_lo} ${bathC_hi} INF INF INF INF units box")
        L.append("region          wallR block ${wallR_lo} ${wallR_hi} INF INF INF INF units box")
        L.append("region          slab  block ${slab_lo}  ${slab_hi}  INF INF INF INF units box")
        L.append("")
        L.append("group           wallL region wallL")
        L.append("group           bathH region bathH")
        L.append("group           bathC region bathC")
        L.append("group           wallR region wallR")
        L.append("group           slab  region slab")
        L.append("")
        if hot_K and cold_K:
            L.append(f"variable        T_src equal {hot_K:g}")
            L.append(f"variable        T_snk equal {cold_K:g}")
        else:
            L.append("# !! 文献未报告冷热浴温度：无温度则 NEMD 无法建立梯度")
            L.append("variable        T_src equal 350.0")
            L.append("variable        T_snk equal 250.0")
        L.append("variable        Tdamp equal 100.0    # 单位 fs")
        L.append("")
        L.append("compute         TbathH all temp/region bathH")
        L.append("compute         TbathC all temp/region bathC")
        L.append("")
        L.append(f"fix             fH bathH langevin ${{T_src}} ${{T_src}} "
                 f"${{Tdamp}} ${{seed}}")
        L.append(f"fix             fC bathC langevin ${{T_snk}} ${{T_snk}} "
                 f"${{Tdamp}} ${{seed}}")
        L.append("fix_modify      fH temp TbathH")
        L.append("fix_modify      fC temp TbathC")
        L.append("#   ↑↑ 恒温 fix 的 group 必须是 bathH / bathC，**不能写 all**。")
        L.append("#      fix langevin 的受力范围由 fix 自己的 group 决定")
        L.append("#      （源码 fix_langevin.cpp 的 `if (mask[i] & groupbit)`），")
        L.append("#      与 fix_modify temp 指定的 compute 的 group 无关。")
        L.append("#      两个恒温器都写 all 时，全体原子被两个不同温度的浴同时")
        L.append("#      作用，温度梯度被抹平：脚本正常运行，κ 无意义。")
        L.append("")
        L.append("# 两端固定壁：冻结受力，并清零初速度")
        L.append("fix             freezeL wallL setforce 0.0 0.0 0.0")
        L.append("fix             freezeR wallR setforce 0.0 0.0 0.0")
        L.append("velocity        wallL set 0.0 0.0 0.0")
        L.append("velocity        wallR set 0.0 0.0 0.0")
        L.append("")
        L.append("# 中间区域 NVE：热流必须在一段能量守恒的区域里测量")
        L.append("group           mobile subtract all wallL wallR")
        L.append("fix             nve_mobile mobile nve")
        L.append("")
        L.append("# 热流：只在中央测量组上计算")
        L.append("compute         ke_atom   slab ke/atom")
        L.append("compute         pe_atom   slab pe/atom")
        L.append("compute         strs_atom slab centroid/stress/atom NULL virial")
        L.append("#   ↑ 见 Green-Kubo 段的说明：手册明确推荐 centroid/stress/atom")
        L.append("compute         jflux   slab heat/flux ke_atom pe_atom strs_atom")
        L.append("")
        L.append("# 归一化体积 = **测量组所在**的体积 = Ly * Lz * 切片厚度。")
        L.append("#   heat/flux 返回的是 extensive 值（energy*velocity），")
        L.append("#   手册要求除以「包含该 group 原子的体积」：")
        L.append("#   group 是中央切片时用整盒体积会算错，且不报错。")
        L.append("variable        V_slab equal (yhi-ylo)*(zhi-zlo)*v_w_slab")
        L.append("variable        J_slab equal c_jflux[1]/v_V_slab")
        L.append("")
        L.append("fix             flx_avg slab ave/time 10 100 1000 v_J_slab ave one &")
        L.append("                file flux.profile")
        L.append("compute         layers all chunk/atom bin/1d x lower 0.5 units box")
        L.append("fix             tmp_avg all ave/chunk 10 100 1000 layers temp &")
        L.append("                norm sample file temp.profile")
        L.append("")
        L.append("# 判据：T(x) 剖面须**线性且单调**（V 形分布的线性拟合 R² 也不低），")
        L.append("# 且 κ 在多个统计窗口内不漂移。见 src/convergence.py。")

    elif hf and "RNEMD" in hf:
        L.append("")
        L.append("# 反 NEMD（Müller-Plathe）：交换两个薄层中原子的动能，")
        L.append("# 由产生的温度梯度反推 κ。与直接 NEMD 不同，这里**不加**冷热浴，")
        L.append("# 整盒仍由全局恒温器控温；梯度由速度交换自然形成。")
        L.append("")
        L.append("# 注意：本方法不需要手写 hot / cold 区域：")
        L.append("#   fix thermal/conductivity 自己沿指定轴等分成 Nbin 层，")
        L.append("#   在冷热两端层之间交换动能。多写一组 region/group 反而危险：")
        L.append("#   它们是**未被任何命令引用**的死代码，会让人误以为控温范围是它们。")
        L.append("")
        L.append("# 交换间隔 N 与层数 Nbin 直接决定梯度大小，均须报告；")
        L.append("#   Nbin 必须为偶数（两端各取一半）。")
        L.append("fix             mp all thermal/conductivity 20 z 20")
        L.append("#   ↑ 每 20 步沿 z 交换一次；层数 20")
        L.append("#   该 fix 的 tally 给出累计交换能量，")
        L.append("#   κ = J / (2 A t dT/dz)：A 为垂直 z 的截面积，")
        L.append("#   因子 2 来自能量沿 z 单向传递。")
        L.append("")
        L.append("# 温度剖面按 z 分层统计，与交换方向一致")
        L.append("compute         layers all chunk/atom bin/1d z lower 0.5 units box")
        L.append("fix             tmp_avg all ave/chunk 20 100 2000 layers temp &")
        L.append("                norm sample file temp.profile")
        L.append("")
        L.append("# !! 交换间隔与层数取错时，梯度可能太陡（偏离线性响应区）")
        L.append("#    或太缓（信噪比不足）。须做 N 的扫描确认 κ 不随 N 变。")

    else:
        L.append("")
        L.append(f"# !! 热流方法 {hf} 尚无生成模板，须人工补全。")
        L.append("#    已识别的分支：Green-Kubo / Direct NEMD / RNEMD (Muller-Plathe)")

    L.append("# " + "=" * 70)
    return L


# -----------------------------------------------------------------------------
def build_all(fields: Dict[str, Any], outdir: str,
              stages: Tuple[str, ...] = ("em", "nvt", "npt", "prod"),
              blocking: Optional[List[str]] = None) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    result: Dict[str, Any] = {"files": [], "checks": {}, "warnings": [],
                             "unresolved": [], "provenance": {},
                             "provenance_lammps": {}}
    # 同一参数在不同阶段可能来源不同（prod 取自文献、nvt 用默认值）。
    # 合并时保留更"弱"的那个：报告必须反映最坏情况，
    # 不能因为某一阶段命中了文献值就掩盖另一阶段的默认值。
    rank = {"paper": 0, "builtin": 1, "default": 2}

    for st in stages:
        b = build_gromacs(fields, st, blocking=blocking)
        path = os.path.join(outdir, f"{st}.mdp")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"; generated by md-repro-checker: stage={st}\n")
            fh.write("; 逐参数来源见同目录 provenance.json\n\n")
            fh.write(b["mdp"])
        chk = check_mdp(b["mdp"])
        for k, v in b["provenance"].items():
            old = result["provenance"].get(k)
            if old is None or rank.get(v, 3) > rank.get(old, 3):
                result["provenance"][k] = v
        result["files"].append(path)
        result["checks"][st] = chk
        result["warnings"] += [f"[{st}] {w}" for w in b["warnings"]]
        result["unresolved"] += b["unresolved"]

    # LAMMPS 与 GROMACS 并行产出。两份来源表分开合并、分开汇报：
    # 同一个物理量在两套引擎里的来源可能不同（如 special_bonds 由力场决定，
    # 而 rcoulomb 由文献决定），混在一张表里会让"哪个默认值属于哪个引擎"
    # 变得不可判断。
    for st in stages:
        lb = build_lammps(fields, st, blocking=blocking)
        lpath = os.path.join(outdir, f"{st}.lmp")
        with open(lpath, "w", encoding="utf-8") as fh:
            fh.write(f"# generated by md-repro-checker: stage={st}\n")
            fh.write("# 逐参数来源见同目录 provenance.json 的 lammps 段\n\n")
            fh.write(lb["script"])
        result["files"].append(lpath)
        result["checks"][f"{st}.lmp"] = check_lammps(lb["script"])
        result["warnings"] += [f"[lmp:{st}] {w}" for w in lb["warnings"]]
        result["unresolved"] += lb["unresolved"]
        for k, v in lb["provenance"].items():
            old = result["provenance_lammps"].get(k)
            if old is None or rank.get(v, 3) > rank.get(old, 3):
                result["provenance_lammps"][k] = v

    skeleton = build_openmm_skeleton(fields)
    with open(os.path.join(outdir, "openmm_skeleton.py"), "w", encoding="utf-8") as fh:
        fh.write(skeleton)
    result["files"].append(os.path.join(outdir, "openmm_skeleton.py"))
    result["checks"]["openmm"] = check_python_source(skeleton)

    result["unresolved"] = sorted(set(result["unresolved"]))

    # 落盘一份机器可读的来源表：使用者（或下游 agent）据此判断
    # 哪些参数可以直接用于复现，哪些只是工具的假设。
    with open(os.path.join(outdir, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump({"parameters": result["provenance"],
                   "lammps": result["provenance_lammps"],
                   "unresolved": result["unresolved"],
                   "warnings": result["warnings"]},
                  fh, ensure_ascii=False, indent=2)
    result["files"].append(os.path.join(outdir, "provenance.json"))
    return result
