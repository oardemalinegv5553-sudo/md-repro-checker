# -*- coding: utf-8 -*-
"""
test_regressions.py：真实文献上踩出来的坑，逐条钉成回归测试

为什么这个文件值得单独存在：

  case-01 的黄金标注集是**我们自己写的**方法段，规则在那上面全对，
  只证明规则符合我们自己的预期。真实文献的写法不是我们写的那种，
  真正的缺陷全部是在 arXiv 真实方法段上跑出来的。

  本文件把那一轮回排查中确认的每一条缺陷，都固化成一条断言。
  没有这些断言，下次改正则时同样的坑会再踩一遍，而且症状是
  "字段缺失"或"取值不对"，从输出上极难看出来。

  每个用例都注明了它来自哪篇（arXiv ID）以及当时观察到的错误现象。

设计上刻意**不依赖 eval/real/sections/**：
  那是受版权保护的论文正文，不分发。这里只引用**极短的技术短语**
  （最长十余词）作为夹具，足以触发对应的正则，且构不成对原文的复制。
  因此本文件可以随工具一起交付，并在没有语料的情况下独立运行。

用法：
  python eval/real/test_regressions.py          # 全部
  python eval/real/test_regressions.py -v       # 打印每个用例的细节
"""
from __future__ import annotations

import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.cli import run_pipeline            # noqa: E402
from src import extract as E                # noqa: E402

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


# -----------------------------------------------------------------------------
# 断言小工具
# -----------------------------------------------------------------------------
_FAILS = []
_N = [0]


def check(name, got, want, why=""):
    _N[0] += 1
    ok = got == want
    if VERBOSE or not ok:
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {name}")
        if not ok:
            print(f"         期望 {want!r}")
            print(f"         实得 {got!r}")
            if why:
                print(f"         背景 {why}")
    if not ok:
        _FAILS.append((name, got, want, why))


def field(text, key):
    """跑完整管道，取某字段的 status。"""
    return run_pipeline(text)["extraction"]["fields"].get(key, {}).get("status")


def value(text, key):
    return run_pipeline(text)["extraction"]["fields"].get(key, {}).get("value")


def ctx(text):
    return run_pipeline(text)["extraction"]["context"]


def _has_linearity_warning(text):
    """R-11 是否触发。

    findings 里不带规则编号（只有 kind/field/message），所以按措辞认，
    这比按编号认更稳：编号是内部记账，措辞才是使用者看到的东西。
    """
    return any("偏离线性响应区" in (f.get("message") or "")
               for f in run_pipeline(text)["validation"]["findings"])


# -----------------------------------------------------------------------------
# 1. 取值错误：短键的子串匹配
# -----------------------------------------------------------------------------
def test_short_key_word_boundaries():
    # 2404.14338v1 附近的膜体系论文里 "dopc"（一种磷脂）内含 "opc"（一种水模型）。
    # 后果：一篇只提过 DOPC 的膜论文报出"水模型 = OPC"。
    # 取值错误比字段缺失危险：错的值会被下游当成事实写进输入文件。
    check("'dopc' 不得被读成水模型 OPC",
          value("the 1,2-dioleoyl-sn-glycero-3-phosphocholine (dopc) bilayer "
                "was constructed using the united-atom berger force field", "water_model"),
          None,
          "2404.14338v1 类：磷脂名内含 opc")
    # 正向对照：SPC 仍然要认出来，且不能因为加边界而漏掉带斜杠的 SPC/E。
    check("'spc' 仍能识别",
          value("molecules were treated using the rigid simple point charge (spc) model",
                "water_model"), "SPC")
    check("'spc/e' 仍能识别（带斜杠键不受词边界影响）",
          value("solvated in spc/e water", "water_model"), "SPC/E")
    check("'tip4p/2005' 优先于 'tip4p'",
          value("water was described by the tip4p/2005 model", "water_model"),
          "TIP4P/2005")
    check("'gaff2' 优先于 'gaff'",
          value("parameters were taken from gaff2", "force_field"), "GAFF2")


# -----------------------------------------------------------------------------
# 2. 假阳性之王：裸子串匹配把 "this" 读成组氨酸
# -----------------------------------------------------------------------------
def test_protein_keywords_need_word_boundary():
    # 原实现用 `"his" in low`，于是 "this" 命中。24 篇真实方法段里
    # 20 篇（83%）因此被判成"含蛋白"，随即报"未报告质子化态"。
    # 这是一篇石墨烯热导论文里再普通不过的一句话：
    check("'reported in this article' 不得判成含蛋白",
          field("the thermal conductivity reported in this article was computed "
                "at 300 k for a graphene sheet", "protonation_state"),
          "not_applicable",
          "裸 'his' 命中 'this'，24 篇里 20 篇栽在这条")
    check("真正的组氨酸残基仍要认出来",
          field("the his64 residue was protonated", "protonation_state"),
          "missing")
    check("'residues' 仍要认出来",
          field("all residues were protonated according to ph 7", "protonation_state"),
          "missing")


# -----------------------------------------------------------------------------
# 3. 适用性必须由正向证据支撑
# -----------------------------------------------------------------------------
def test_solvent_requires_positive_evidence():
    # 原实现 explicit_solvent = not (vacuum or solvent_free)，默认 True，
    # 等于把"没说"当成"有"。24 篇里 14 篇（58%）是石墨烯/二氧化硅/金属
    # 这类压根没有溶剂的体系，却被报"未报告水模型"。
    check("无溶剂体系：水模型不适用",
          field("we performed molecular dynamics simulations of a graphene sheet "
                "using the lennard-jones potential with a timestep of 1 fs", "water_model"),
          "not_applicable")
    check("隐式溶剂：水模型不适用",
          field("binding free energies were computed with mm/pbsa in implicit solvent",
                "water_model"), "not_applicable")
    check("干态熔体：水模型不适用",
          field("the bulk polymer melt was equilibrated at 500 k with a timestep of 2 fs",
                "water_model"), "not_applicable")
    # 现象词 ≠ 溶剂。溶剂词修饰的是另一个名词，而不是"体系泡在什么里"。
    check("'hydrated surface'（表面羟基化）不得判成显式溶剂",
          ctx("the (hydrated) surface and (hydrated) edge structures were selected "
              "for the training set")["explicit_solvent"], False,
          "2404.14338v1：吸附/羟基化，不是溶剂")
    check("'ion solvation'（电解液现象）不得判成显式溶剂",
          ctx("ion solvation in the electrolyte was described by a continuum model"
              )["explicit_solvent"], False,
          "2602.17595v1：溶剂化现象，不是溶剂模型")
    check("'water molecules adsorbed on the surface' 不得判成显式溶剂",
          ctx("structures are included with an increasing number of water molecules "
              "adsorbed on the surface")["explicit_solvent"], False,
          "吸附质，不是溶剂")
    # 但**不能**收紧成"必须写出水模型名"，那会让本字段永不可能报 missing
    # （自证循环）。"solvated in water" 是独立于模型名的适用证据。
    check("'solvated in water' 但没写模型名 -> 仍要报缺失",
          field("the protein was solvated in water and neutralized with ions",
                "water_model"), "missing",
          "这正是 water_model 缺失这条要抓的东西")
    check("写了水模型名 -> present",
          field("the system was solvated in water using the tip3p model",
                "water_model"), "present")


# -----------------------------------------------------------------------------
# 4. ambiguous 状态：文档里写了四态，代码里必须真的会产生
# -----------------------------------------------------------------------------
def test_ambiguous_is_actually_produced():
    # 抽取器原先的注释就写着"只写了族名 -> ambiguous"，但代码返回的
    # (canon, "medium", frag) 被调用方无条件写成 present。
    # 后果：一个评审者拿"只写了 AMBER 没写版本"去测，看到的会是 present。
    check("只写 'AMBER' 不写版本 -> ambiguous",
          field("we used the amber force field for the protein", "force_field"),
          "ambiguous")
    check("ambiguous 时仍保留族名作为取值（便于人读）",
          value("we used the amber force field for the protein", "force_field"),
          "AMBER")
    check("写了具体版本 -> present",
          field("we used the amber99sb-ildn force field", "force_field"), "present")
    check("'GAFF' 不写 2 -> ambiguous（GAFF 与 GAFF2 是两套参数）",
          field("small molecules were described by gaff", "force_field"), "ambiguous")
    # 反向：UFF/DREIDING 没有通行版本号，族名就是完整答案。
    # 把它们也报 ambiguous 等于凭空制造歧义，同样是假阳性。
    check("'UFF' 没有版本号可说 -> present，不得报 ambiguous",
          field("the metal slab was described by the uff force field", "force_field"),
          "present")
    check("'DREIDING' -> present",
          field("the polymer was modeled with the dreiding force field", "force_field"),
          "present")


# -----------------------------------------------------------------------------
# 5. 色散校正：把过泛的键撤掉
# -----------------------------------------------------------------------------
def test_dispersion_context_required():
    # DISPERSION_CANON 里曾有一个独立键 "energy and pressure" -> EnerPres。
    # 它太泛：任何"能量与压力"的句子都会被映射成 EnerPres，取值错误。
    check("'the error in energy and pressure' 不得读成 EnerPres",
          value("the error in energy and pressure was analyzed statistically",
                "dispersion_correction"), None,
          "撤掉的过泛键")
    check("'long-range correction to energy and pressure' -> EnerPres",
          value("we applied the standard long-range correction to energy and pressure",
                "dispersion_correction"), "EnerPres")
    check("'EnerPres' 字面量 -> EnerPres",
          value("dispcorr = enerpres was used", "dispersion_correction"), "EnerPres")
    check("'force-switch' 字面量 -> force-switch",
          value("a force-switch was applied", "dispersion_correction"), "force-switch")


# -----------------------------------------------------------------------------
# 6. 双义词与工具名：配体词表的边界
# -----------------------------------------------------------------------------
def test_ligand_vocabulary_boundary():
    # 踩过的坑：把 gaff/cgenff/antechamber/am1-bcc 收进配体词表后，
    # 一个用 GAFF2 + AM1-BCC 参数化的**聚合物熔体**被判成"含配体"，
    # 随即报"未报告配体参数"。这些词说明"电荷怎么来的"，不说明"体系里有配体"。
    check("GAFF2 + AM1-BCC 的聚合物熔体不含配体",
          field("the polymer was parameterized with gaff2 and am1-bcc charges",
                "ligand_parameters"), "not_applicable")
    # substrate 是双义词：材料学指固体基底，酶学指底物。
    check("'silica substrate'（固体基底）不含配体",
          ctx("a silica substrate was used")["ligand_present"], False)
    check("'protein substrate'（酶学底物）算配体",
          ctx("the protein substrate complex was simulated")["ligand_present"], True)
    check("'ligand' 仍然算配体",
          ctx("the ligand was docked")["ligand_present"], True)


def test_extractors_are_case_insensitive():
    # 这一条是**系统审计**发现的：在 _extract_software 上发现正则少写 re.I
    # 之后，把每个抽取器都拿同一句话的三种写法（原样/全大写/全小写）跑一遍，
    # 结果不同即为漏检。一次扫出三个字段。
    # 共同点：漏检表现为"字段缺失"，从输出上完全看不出来，
    # 使用者只会以为作者没写。
    from src.extract import PATTERNS

    CASES = [
        # (字段, 原样句, 小写句)：小写句是文献里同样常见的写法
        ("software", "simulations used the LAMMPS package",
         "simulations used the lammps package"),
        ("software", "the openmm 8.0 package was used",
         "the openmm 8.0 package was used"),
        ("software", "namd 2.14 was employed",
         "namd 2.14 was employed"),
        ("ensemble", "the simulation was run in the NPT ensemble",
         "the simulation was run in the npt ensemble"),
        ("temperature_bath", "the cold bath was 300 K and the hot bath 500 K",
         "the cold bath was 300 k and the hot bath 500 k"),
    ]
    for key, upper, lower in CASES:
        fn = PATTERNS[key]
        check(f"{key}: {upper!r} 必须命中", bool(fn(upper)), True,
              "原样写法")
        check(f"{key}: 小写写法也必须命中", bool(fn(lower)), True,
              "少了 re.I 会整类漏掉（表现为'字段缺失'）")

    # 反向：加了 re.I 不能把别的单位吃进来
    tb = PATTERNS["temperature_bath"]
    for bad in ("the energy was 500 kJ/mol and the distance 100 km",
                "using kB = 1.38e-23"):
        check(f"temperature_bath 不得误吃 {bad[:34]!r}", tb(bad), None,
              "\\b 挡住了 km / kPa / k_B")


def rules(text):
    """跑完整管道，取命中的一致性规则编号。"""
    v = run_pipeline(text)["validation"]
    return sorted({f.get("rule_id") for f in v["findings"] if f.get("rule_id")})


# -----------------------------------------------------------------------------
# 8. 召回类：关键词与数字隔着一段话，或语序反过来
# -----------------------------------------------------------------------------
def test_prose_phrasings_are_recalled():
    # 时间步：原实现把间隔写死成 \D{0,12}，"time step was chosen to be 0.5 fs"
    # 里关键词与数字隔了 18 个字符，直接漏掉。真实文献很少写紧邻形式。
    check("'time step was chosen to be 0.5 fs' -> present 0.5",
          value("the time step was chosen to be 0.5 fs for all runs", "timestep"), 0.5)
    # 语序反过来：数字在前、关键词在后。
    check("'t=0.2 fs as time step' -> present 0.2",
          value("we used t=0.2 fs as time step throughout", "timestep"), 0.2)
    # 无单位写法（LJ / Martini 自然单位）**绝不能**当成 fs。
    # "0.001" 在 LJ 单位下约合 2 fs，写成 "0.001 fs" 差三个量级，
    # 所以只报 ambiguous，不给一个会被当真的数。
    check("LJ 自然单位的时间步 -> ambiguous 而非 present",
          field("a time step size of t=0.001 was employed in lj units", "timestep"),
          "ambiguous", "无单位不能当 fs")
    check("该情形仍保留原数（值为 0.001，供人判断）",
          value("a time step size of t=0.001 was employed in lj units", "timestep"),
          0.001)

    # 模拟时长：采样间隔不是模拟时长。实测 "500 ps long trajectories,
    # with velocity re-selections at 100 ps intervals" 曾抓成 100 ps，
    # 取值错误，且错的那个数看起来完全合理。
    check("'at 100 ps intervals' 不得当成长度 100 ps",
          value("500 ps long trajectories, with velocity re-selections at 100 ps intervals",
                "simulation_length"), {"length": 500.0, "unit": "ps"})
    check("'correlation time was 20 ps' 不得当成长度",
          field("the correlation time was 20 ps", "simulation_length"), "missing")
    # 只给步数时只报步数，不折算成 ns：折算依赖单位制，折了就是编数。
    check("只给步数 -> 报步数，不折算",
          value("the production run consisted of 4 x 10^6 time steps",
                "simulation_length"), {"steps": 4000000})


# -----------------------------------------------------------------------------
# 9. 同一算法在不同软件里叫法不同
# -----------------------------------------------------------------------------
def test_algorithm_aliases_across_codes():
    check("'pppm'（LAMMPS 的叫法）-> PME",
          value("long-range electrostatics were treated with the pppm solver",
                "electrostatics"), {"method": "PME"},
          "GROMACS 叫 PME，LAMMPS 叫 PPPM，是同一个算法")


# -----------------------------------------------------------------------------
# 10. 软件名与版本的三种位置
# -----------------------------------------------------------------------------
def test_software_version_positions():
    check("'gromacs (version 2023.3)' -> present",
          field("simulations were performed with gromacs (version 2023.3)", "software"),
          "present")
    check("'(october 29, 2020) version of lammps' -> present（版本在名前）",
          field("we used the (october 29, 2020) version of lammps", "software"),
          "present", "日期式版本号")
    # 只写软件名不写版本 -> ambiguous，不是 present。
    # 这一条是两批独立标注员都指出的同一处分歧（先例见 [4]）。
    check("只写软件名不写版本 -> ambiguous",
          field("all runs used the openmm package", "software"), "ambiguous")


# -----------------------------------------------------------------------------
# 11. 子串陷阱（第二类）：后缀相同的词
# -----------------------------------------------------------------------------
def test_substring_traps_beyond_word_boundary():
    # `t?leap` 会把 "temperature leaps at the boundary" 里的 leaps
    # 读成 Amber 的建模工具 LEaP，与 `opc` 命中 `dopc` 同类：取值错误。
    check("'temperature leaps' 不得读成 LEaP",
          value("the temperature leaps at the boundary between regions",
                "initial_structure"), None)
    check("'tleap' 仍要认出来",
          value("the structure was built with tleap", "initial_structure"), "tleap")

    # `RESP(?:2)?` 不带词边界时会在 "respectively"、"corresponding" 里命中子串，
    # 还会把 r-RESPA 积分器读成 RESP 电荷，实测让 1002.0354v1、2404.14338v1、
    # 2604.01940v3 三篇各多报一条配体参数化，其中第三篇因 charge 与 ff 同时
    # 命中而被判成 present（标注是 ambiguous）。同 `leaps`/`LEaP` 一类。
    check("'respectively' 不得读成 RESP 电荷",
          E._extract_ligand_params("filled with 58 and 63 water molecules, respectively"), None)
    check("'corresponding' 不得读成 RESP 电荷",
          E._extract_ligand_params("the corresponding pair parameters were used"), None)
    check("r-RESPA 积分器不得读成 RESP 电荷",
          E._extract_ligand_params("a scheme based on the r-RESPA algorithm"), None)
    # 词边界不能矫枉过正。中文分支里的 `RESP` 刻意**不加** `\b`，
    # 它前面是 `电荷`（汉字属 `\w`），加了反而失配，这条断言把该例外钉住。
    check("真正的 RESP 仍要认出来",
          E._extract_ligand_params("charges were derived with RESP")[0], "RESP")
    check("RESP2 的数字不能被词边界切掉",
          E._extract_ligand_params("RESP2 charges were used")[0], "RESP2")
    check("中文分支里的 RESP 不受词边界影响",
          E._extract_ligand_params("电荷由RESP拟合")[0], "电荷由RESP拟合")


# -----------------------------------------------------------------------------
# 12. 上下文标志：大小写必须规范化
# -----------------------------------------------------------------------------
def test_context_flags_are_canonicalised():
    # 这一条是**审计脚本扩充后才量到**的：第一版 audit_case.py 只遍历
    # PATTERNS（字段抽取器），而 detect_context 是另一套独立代码。
    # `_APPLICABILITY` 里 barostat 的闸门写作 ctx["ensemble"] == "NPT"，
    # 而 _extract_ensemble 把匹配到的原文原样返回，于是小写 "npt" 的
    # 文献被判"本体系不需要恒压器"，恒压器字段**直接消失成 not_applicable**。
    # 现场留着一个看似合理的理由，比 missing 更难发现。
    check("小写 'npt' 也要认：恒压器必须照常出现",
          field("the simulation was run in the npt ensemble with a berendsen barostat",
                "barostat"), "present")
    check("系综取值须规范化成大写（闸门比的就是它）",
          value("the simulation was run in the npt ensemble", "ensemble"), "NPT")
    check("NVT 下恒压器仍判不适用",
          field("the simulation was run in the nvt ensemble with a berendsen barostat",
                "barostat"), "not_applicable")


# -----------------------------------------------------------------------------
# 13. 散文式耦合常数：不能漏，也不能截出半个数
# -----------------------------------------------------------------------------
def test_coupling_prose_and_number_boundaries():
    # 符号式（tau_t = 0.1 ps）主要在方法学论文里出现；真实文献更常写
    # "with a coupling constant of 0.1 ps"。原实现只认符号式，于是
    # **明明给了耦合常数**的文献被报"未给耦合常数"，正是最忌讳的假阳性。
    check("散文式耦合常数 -> 不再误报 R-12",
          rules("temperature was maintained at 320 k using a nose-hoover "
                "thermostat with a coupling constant of 0.1 ps"), [])
    check("确实没给耦合常数 -> R-12 仍要报",
          rules("temperature was maintained at 320 k using a nose-hoover thermostat"),
          ["R-12"])
    # 数字边界：没有边界时 `[0-9]*\.?[0-9]+` 会从 "200" 里截出 "20"
    # （`[0-9]*`吃"2"、`[0-9]+`吃"0"），把 200 fs 报成 20 fs，取值错误。
    # 句子必须写明 NPT 系综，否则恒压器按适用性判据就是 not_applicable，
    # 取值为 None，测不到数字边界这一层。
    bc = value("the simulation was run in the npt ensemble using a berendsen "
               "barostat with a coupling constant of 200 fs", "barostat")
    check("恒压器耦合常数取到的是完整的 200，而不是 20",
          (bc or {}).get("coupling"), "200", "数字边界")
    # "X and Y ... set to A and B, respectively" 句型里两个数按器件顺序
    # 一一对应，而两个抽取器都会抓到同一个（第一个）数，恒温器会拿到
    # 恒压器的 200 fs。这是取值错误，比漏抓更糟：正则分不清归属，整句放弃。
    check("respectively 句型 -> 不猜，宁可报'没给耦合'",
          "R-12" in rules(
              "a nose-hoover thermostat and a parrinello-rahman barostat were used; "
              "the coupling parameters were set to 20 fs and 200 fs, respectively"),
          True, "猜错归属 = 取值错误")
    # 物理里的 coupling 不是恒温器耦合（2312.07027v2）。
    # 这里只断言 R-12 不被误触发，同句里的 "thermal conductivity" 会
    # 正常触发 heat_transport，于是 R-03/R-05 照报，那是另一回事。
    check("'coupling of energy and momentum fluxes' 不得触发 R-12",
          "R-12" in rules("the effect of coupling of the energy and momentum "
                          "fluxes leads to an increase in the thermal conductivity"),
          False, "物理耦合不是恒温器耦合")


def test_lammps_mixing_rule():
    # 这一组不来自 arXiv 语料，来自 LAMMPS 源码与手册：
    # pair_modify mix 有默认值 geometric（σ 取几何平均），而 AMBER/GAFF/
    # CHARMM 用的是 Lorentz-Berthelot（σ 取算术平均）。漏写时 LAMMPS
    # 不报错、不警告，只是所有 I≠J 交叉对的 σ 偏移；且因为
    # src/pair.cpp 的 mix_energy() 在两种规则下返回同一个 sqrt(eps1*eps2)，
    # **ε 不受影响，能量项上看不出任何异常**。
    #
    # **OPLS-AA 不在这条规律里**：它用几何平均（σ 与 ε 都取几何平均，
    # GROMACS 手册 comb-rule 3 原文 "This last rule is used by the OPLS
    # force field"），与 LAMMPS 默认值一致。本表早期把 OPLS 与 AMBER 归成
    # 同一组，等于给 OPLS 文献填了一个不会被看见的错值——正是这个工具
    # 自己要防的那类错误。下面单独钉住"OPLS 与 AMBER 不同组"，
    # 因为这一条一旦回归，症状恰好是静默的。
    #
    # 夹具全部为合成字符串，不取自任何真实脚本。
    import re

    from src import build_inputs as B

    # --- 映射表本身 ---
    for ff, want in (("GAFF2", "arithmetic"),
                     ("AMBER99SB-ILDN", "arithmetic"),
                     ("CHARMM36m", "arithmetic"),
                     ("OPLS-AA", "geometric"),
                     ("OPLS-AA/L", "geometric"),
                     ("COMPASS III", "sixthpower"),
                     ("PCFF+", "sixthpower")):
        got = None
        for pat, val, _why in B.MIXING_BY_FF:
            if re.match(pat, ff):
                got = val
                break
        check(f"力场 {ff} -> mix {want}", got, want, "映射表")

    # OPLS 必须与 AMBER/GAFF/CHARMM 分属不同规则。合并回一组就是回归。
    def _mix_of(name):
        for pat, val, _why in B.MIXING_BY_FF:
            if re.match(pat, name):
                return val
        return None

    check("OPLS 与 AMBER 不是同一规则（几何 vs 算术）",
          (_mix_of("OPLS-AA") == _mix_of("AMBER99SB-ILDN")), False,
          "OPLS 几何平均 / AMBER 算术平均")
    check("OPLS 与 CHARMM 不是同一规则（几何 vs 算术）",
          (_mix_of("OPLS-AA") == _mix_of("CHARMM36m")), False,
          "OPLS 几何平均 / CHARMM 算术平均")

    # DREIDING 刻意不收：混合规则未能逐族坐实，宁可留 TODO 也不猜一个值，
    # 与"拒绝伪造力场参数"是同一条原则。
    got = None
    for pat, val, _why in B.MIXING_BY_FF:
        if re.match(pat, "DREIDING"):
            got = val
            break
    check("DREIDING 不在映射表里（留 TODO 而非猜一个值）", got, None)

    # --- 生成器：已知力场 -> 真值，且来源非 unresolved ---
    lb = B.build_lammps({"force_field": {"status": "present", "value": "GAFF2"}})
    check("GAFF2 生成 mix arithmetic",
          "pair_modify     mix arithmetic" in lb["script"], True, "生成器")
    check("mix 行的来源是 paper（由力场名推得，非文献原文字符串）",
          lb["provenance"].get("pair_modify_mix"), "paper")
    chk = B.check_lammps(lb["script"])
    check("自己生成的脚本 ok 且无警告", (chk["ok"], chk["warnings"]), (True, []))
    # tail 行与 mix 行必须各占一个 provenance key。复用同一个 key 会静默覆盖：
    # 报告里"这一行是默认值"的标注会消失，而那是生成器存在的理由。
    check("tail 与 mix 的 provenance key 不互相覆盖",
          lb["provenance"].get("pair_modify"), "default")

    # --- 生成器：未知力场 -> 注释掉的 TODO，且不污染 unresolved ---
    lu = B.build_lammps({"force_field": {"status": "present", "value": "DREIDING"}})
    check("未知力场留 TODO 且被注释掉（不会真的生效）",
          "# pair_modify     mix TODO" in lu["script"], True)
    check("TODO 行的来源标为 unresolved",
          lu["provenance"].get("pair_modify_mix"), "unresolved")
    # build_lammps 的 unresolved 是从**字段名**算的，不读 provenance。
    # 这个不变量值得钉住：否则给新行打 unresolved 标签会连带拖垮
    # eval/score.py 的 exec_ok（它要求 not b["unresolved"]）。
    check("provenance 标 unresolved 不会推进 unresolved 字段列表",
          "pair_modify_mix" in lu["unresolved"], False, "exec_ok 不受影响")

    # --- check_lammps：漏写只警告不报错 ---
    bare = ("units real\n"
            "atom_style full\n"
            "pair_style lj/cut/coul/long 10.0\n"
            "kspace_style pppm 1.0e-4\n"
            "fix 1 all nvt temp 300.0 300.0 100.0\n"
            "run 1000\n")
    c = B.check_lammps(bare)
    check("有 lj/cut 无 mix -> 出警告", len(c["warnings"]) > 0, True)
    # 关键：警告绝不能变成错误。判据不足以断定脚本有问题，
    # 无法从脚本反推力场族，而交叉对全由 pair_coeff 显式给定的脚本
    # 本来就不需要这一行。
    check("但 ok 仍为 True、errors 仍为空", (c["ok"], c["errors"]), (True, []))

    c2 = B.check_lammps(bare.replace("fix 1 all nvt",
                                     "pair_modify     mix arithmetic\nfix 1 all nvt"))
    check("补上 mix 后警告消失", c2["warnings"], [])

    # 注释在解析前已按 # 剥掉：被注释掉的 mix 不能算"已声明"。
    # 否则生成器留的 TODO 反而会把检查器哄住，那正是它要抓的情况。
    c3 = B.check_lammps(bare.replace("fix 1 all nvt",
                                     "# pair_modify     mix arithmetic\nfix 1 all nvt"))
    check("注释掉的 mix 不算已声明", len(c3["warnings"]) > 0, True)

    # 没有 lj/cut 系 pair_style 时不提这条，适用性需要正面证据。
    c4 = B.check_lammps("units real\npair_style  zero 10.0\n"
                        "fix 1 all nvt temp 300.0 300.0 100.0\nrun 1000\n")
    check("非 lj/cut 体系不报这条（适用性需正面证据）", c4["warnings"], [])


def test_corpus_comparability():
    # 前 14 组问的都是"**这一篇**够不够复现"。这一组问另一个问题：
    # "**这一组**文献报告的同一个物理量，彼此之间可比吗"。
    #
    # 它没有引入新的领域知识，sensitivity.yaml 里"未报告恒温器 = 该 κ 值
    # 不可与其他文献横向比较"早就写好了，只是一直被施加在单篇上。
    # 所以这里要钉的是**换个分析单元之后新出现的判断**：
    # 哪些差异算发现、哪些不算、以及条件不成立时该怎么退。
    #
    # 夹具全部为合成的抽取结果结构，不含任何真实文献数据。
    from src import corpus as C

    def val(v):
        return {"value": v, "status": "present", "confidence": "high",
                "evidence": {"quote": "", "offset": -1}}

    def miss():
        return {"value": None, "status": "missing", "confidence": "high",
                "evidence": {"quote": "", "offset": -1}}

    def paper(thermo=None, size=None, window=None, timestep=1.0,
              dispersion="EnerPres", explicit_solvent=False):
        fields = {
            "heat_flux_method":      val("Green-Kubo"),
            "thermostat":            val(thermo) if thermo else miss(),
            "system_size":           val({"atoms": size}) if size else miss(),
            "correlation_window":    val(window) if window else miss(),
            "dispersion_correction": val(dispersion) if dispersion else miss(),
            "temperature_bath":      miss(),
            "initial_structure":     val("crystal"),
            "timestep":              val(timestep),
        }
        return {"fields": fields,
                "context": {"heat_transport": True,
                            "explicit_solvent": explicit_solvent}}

    papers = [
        # 恒温器：01 给了类型+耦合常数，02 只给类型，03 是另一种类型
        ("syn-01", paper(thermo={"type": "Nose-Hoover", "coupling": "0.1"},
                         size=1728,  window={"ps": 2.0}, timestep=1.0)),
        ("syn-02", paper(thermo={"type": "Nose-Hoover"},
                         size=1728,  window={"ps": 2.0}, timestep=2.0)),
        ("syn-03", paper(thermo={"type": "Langevin", "coupling": "0.5"},
                         size=8640,  window={"ps": 5.0}, timestep=0.5)),
        # 04 什么都不报，用来触发"报告不全"与"集体缺失"
        ("syn-04", paper(timestep=4.0, dispersion=None)),
    ]
    res = C.run(papers)
    by = {c["field"]: c for c in res["conflicts"]}

    check("不指定分组时自动挑最大的组", res["flag"], "heat_transport")
    check("组内论文数", res["n_papers"], 4)

    # --- 1. 公共键交集：报告不全 ≠ 协议分歧 ---
    # syn-01 有 coupling、syn-02 没有。若按"取值字符串不同"比较，
    # 二者会被算成两种协议，于是每多一篇少写一个可选键就多一条假分歧。
    th = by.get("thermostat") or {}
    check("恒温器判为取值分歧", th.get("kind"), "disagreement")
    check("比较依据的公共键只有 type（coupling 不是每篇都有，不参与比较）",
          th.get("compared_on"), ["type"],
          "共同拥有的键才构成比较基础")
    check("只给类型的那篇与给了类型+耦合常数的那篇归为同一协议",
          sorted(tuple(v["papers"]) for v in th.get("variants") or []),
          [("syn-01", "syn-02"), ("syn-03",)],
          "把可选键的缺失当成另一种协议，会凭空造出分歧")

    # --- 2. 真正的取值分歧 ---
    # 夹具里的两个尺寸取 1728 与 8640，不取 54840：后者是 2207.13405 的
    # 训练集规模，正是本文件第 25 组要挡下的那类错值。拿一个已知错值
    # 当"真分歧"的范例，测试本身就成了文档层面的静默失效。
    check("体系规模 1728 vs 8640 原子 -> 取值分歧",
          (by.get("system_size") or {}).get("kind"), "disagreement")

    # --- 3. 报告不全：取值一致，但有篇没报 ---
    check("三篇取值一致、一篇没报 -> 报告不全（不是分歧）",
          (by.get("dispersion_correction") or {}).get("kind"), "incomplete_report")

    # --- 4. 集体缺失：单篇工具原理上看不见的一类 ---
    tb = by.get("temperature_bath") or {}
    check("全组 0 篇报告冷热浴温度 -> 集体缺失",
          tb.get("kind"), "collective_absence",
          "单篇工具只能逐篇各报一次 missing；"
          "「这一组文献全都没报」只有成组才显现")
    check("集体缺失的篇数覆盖全组",
          (tb.get("n_present"), tb.get("n_missing")), (0, 4))

    # --- 5. 精确率闸门：没有依据的差异不构成发现 ---
    # timestep 在本组有 4 种取值（1.0 / 2.0 / 0.5 / 4.0），
    # 但 sensitivity.yaml 没为它写 conditional_impact，那就不报。
    check("四种取值但知识层没写影响规则的字段不纳入比较",
          "timestep" in res["fields_checked"], False,
          "有差异就报，等于再造一个劝退的假阳性工具")
    check("它也不出现在任何冲突里", "timestep" in by, False)

    # --- 6. 取值一致且无缺失 -> 保持安静 ---
    check("全组取值一致的字段不产生发现", "initial_structure" in by, False)

    # --- 7. 结论与阻断项 ---
    check("存在 high 级冲突 -> 判为不可横向比较",
          res["summary"]["comparable"], False)
    check("阻断项计数 = severity 为 critical/high 的冲突数",
          res["summary"]["n_blocking"],
          sum(1 for c in res["conflicts"]
              if c["severity"] in ("critical", "high")))

    # --- 8. 组级 when：条件必须对该组每一篇都成立 ---
    # dispersion_correction 的影响规则带 when: {explicit_solvent: false}。
    # 组内只要有一篇是显式溶剂，这条规则对该组就不成立：退，不猜。
    mixed = [("syn-01", paper(thermo={"type": "Langevin"},
                              explicit_solvent=False)),
             ("syn-02", paper(thermo={"type": "Langevin"},
                              explicit_solvent=True))]
    rb = C.run(mixed, "heat_transport")
    check("对照组：全组都无显式溶剂时才纳入该字段",
          "dispersion_correction" in res["fields_checked"], True)
    check("组内有一篇是显式溶剂 -> 该字段整体退出比较",
          "dispersion_correction" in rb["fields_checked"], False,
          "保守取交集，不做过度推断")

    # --- 9. comparison_key 本身 ---
    check("没有公共字典键时退化为标量比较",
          C.comparison_key("NVT", None), "NVT")
    check("键序不影响比较键",
          C.comparison_key({"coupling": "0.1", "type": "L"}, {"type"}),
          C.comparison_key({"type": "L", "coupling": "0.1"}, {"type"}))
    check("同一字段上标量与字典混用不崩（交给人判断）",
          C.comparison_key("NVT", {"type"}), "NVT")

    # --- 10. 不足 2 篇：缺数据不等于通过 ---
    one = C.run([("syn-01", paper(thermo={"type": "Langevin"}))], "heat_transport")
    check("组内不足 2 篇 -> 报 error，且 comparable 为 None（不是 True）",
          (bool(one.get("error")), one["conflicts"], one["summary"]["comparable"]),
          (True, [], None),
          "把「没有可比较的对象」渲染成「可比」正是本项目要抓的静默误导")
    check("不足 2 篇时渲染文本不出现「可比」结论",
          "结论: 可比" in C.render(one), False)

    # --- 11. --demo 必须只用合成数据（语料受版权保护，不分发）---
    demo = C.run(C._demo_papers(), "heat_transport")
    check("演示论文 id 全为 demo- 前缀",
          all(i.startswith("demo-") for i in demo["papers"]), True)
    check("演示本身要能跑出发现，否则 --demo 形同虚设",
          demo["summary"]["n_conflicts"] > 0, True)


def test_schema_rules_match_implementation():
    # 本项目专抓"看着有、其实没有"的静默失效，它不该出现在自己的 schema 里。
    #
    # schema/md_method.yaml 的 consistency_rules 块**不参与执行**：真正跑的是
    # src/validate.py 的 RULES 表。而这个块曾经与实现漂移且无人察觉，
    # 12 条 id 对得上，但 9 条文案不同，when/require 里引用过 16 个
    # schema 中并不存在的标识符（solvent、replica_count、barostat_coupling…），
    # 因为没有任何代码读它：改它不会有反应，不改它也不会有反应。
    #
    # 这组断言把"声明与实现漂移"从静默变成失败。它钉的是
    # **知识资产的自洽性**：一个读 schema 的人看到的必须是工具真正做的事。
    from src import validate as V

    schema = V.load_yaml(V.default_paths()["schema"])
    declared = schema.get("consistency_rules") or []
    impl = V.RULES

    check("声明条数 = 实现条数", len(declared), len(impl))
    check("id 与顺序完全一致",
          [r["id"] for r in declared], [r[0] for r in impl])
    check("severity 与实现一致",
          [r.get("severity") for r in declared], [r[3] for r in impl])
    for d, (rid, _fn, msg, _sev) in zip(declared, impl):
        check(f"{rid} 的 message 与实现一致", d.get("message"), msg,
              "文案不一致 = 读 schema 的人看到的是过时说明")
    # 摘要块里不许再出现"引用 schema 中不存在的字段"这种情况。
    # 断言的形式是：每条规则的 when/require 必须非空（写清楚），
    # 且 id 集合与实现严格相等（上面已查）。空值会让读者以为无条件。
    check("每条规则都写明了触发条件与判据",
          [r["id"] for r in declared if not r.get("when") or not r.get("require")],
          [])


def test_force_field_capability_table():
    # 这一组不来自 arXiv 语料，来自势函数的原始文献：
    #   Tersoff, Phys. Rev. B 37, 6991 (1988)
    #   Stillinger-Weber, Phys. Rev. B 31, 5262 (1985)
    #   Brenner, Phys. Rev. B 42, 9458 (1990)
    #   Weeks-Chandler-Andersen, J. Chem. Phys. 54, 5237 (1971)
    #
    # constraints / electrostatics / dispersion_correction 问的是"有没有用
    # SHAKE 类约束、有没有选 Ewald/PME、有没有做长程色散校正"。在共价多体势、
    # 机器学习势、平面波从头动力学、纯排斥 WCA 这些体系里，被问的对象
    # **在势函数的定义里就不存在**，报 missing 是假阳性。
    # 24 篇语料上这一类占 40 条，是最大的单一假阳性来源。
    #
    # 夹具全部为合成短句，不取自任何真实文献或真实脚本。
    import re

    OFF = {"bonds": False, "coulomb": False, "dispersion": False}
    ON = {"bonds": True, "coulomb": True, "dispersion": True}

    # --- 各族必须命中自己的规范拼写 ---
    for text, want in (
            ("We use the Tersoff potential.", OFF),
            ("a second-generation Brenner potential", OFF),
            ("the Stillinger-Weber potential", OFF),
            ("a neuroevolution potential (NEP) was trained", OFF),
            ("deep potential molecular dynamics was used", OFF),
            ("a Gaussian approximation potential", OFF),
            ("All runs used VASP with plane-wave basis sets.", OFF),
            ("ab initio molecular dynamics at the PBE level", OFF),
            ("a purely repulsive WCA model", OFF),
            ("the ReaxFF reactive force field",
             {"bonds": False, "coulomb": True, "dispersion": True})):
        check(f"能力判定 {text[:40]!r}", E._ff_capability(text), want)

    # --- 陷阱：普通英文词，一律不得触发 ---
    # 这类坑本项目已栽过两次：`his` 命中 "this"（24 篇里 20 篇误报"含蛋白"）、
    # `opc` 命中 "dopc"（膜论文报出水模型 OPC）。误命中的代价在这里不对称，
    # 多翻一个 NA，就把一个真实的缺失静静压掉了，而且从输出上看不出来。
    for t in ("The band gap of the semiconductor is 1.1 eV.",
              "a honeycomb lattice of silicon",
              "software for molecular dynamics",
              "DFT was used to fit the charges while CHARMM36 ran the MD.",
              "NVT simulations were performed for 10 ns."):
        check(f"不得触发 {t[:34]!r}", E._ff_capability(t), ON)

    # --- 反向证据优先于族能力 ---
    # 族能力说的是"这个势通常有没有"，反向证据说的是"这个体系里到底有没有"。
    # 2506.04835v1 实测：Tersoff 描述 Si、同时用 12-6 LJ 描述 Au。
    check("Tersoff + 12-6 LJ：色散不得判成不存在",
          E._ff_capability("Tersoff for Si and a 12-6 Lennard-Jones for Au"),
          {"bonds": False, "coulomb": False, "dispersion": True})
    check("Tersoff + PME：静电不得判成不存在",
          E._ff_capability("the Tersoff potential; PME was used for the ions"),
          {"bonds": False, "coulomb": True, "dispersion": False})
    check("Tersoff + SHAKE：键约束不得判成不存在",
          E._ff_capability("the Tersoff potential; SHAKE was applied to the O-H bonds"),
          {"bonds": True, "coulomb": False, "dispersion": False})
    check("识别不出势函数时必须退回原行为（默认适用）",
          E._ff_capability("We performed molecular dynamics simulations."), ON)

    # --- 表本身要完整 ---
    check("每族都写全三项能力且取布尔值",
          [f["name"] for f in E.FF_FAMILIES
           if any(not isinstance(f.get(k), bool)
                  for k in ("bonds", "coulomb", "dispersion"))], [])
    check("每族都写明了判据来源",
          [f["name"] for f in E.FF_FAMILIES if not f.get("why")], [])
    compiled = []
    for f in E.FF_FAMILIES:
        try:
            re.compile(f["pattern"])
            compiled.append(f["name"])
        except re.error:
            pass
    check("每族正则都能编译", compiled, [f["name"] for f in E.FF_FAMILIES])

    # --- 匹配纪律：不撞普通词的裸缩写 ---
    # 先把正则转义剥掉再找词：`\bnep\b` 里的 `\b` 是**字面上的反斜杠加 b**，
    # 后面那个 `b` 是词字符，直接在原文里搜 \bnep\b 永远搜不到。
    joined = re.sub(r"\\[bBAZz]|\\.", " ", " ".join(f["pattern"]
                                                    for f in E.FF_FAMILIES))
    check("表中只允许 nep 这一个裸缩写（它在方法段里没有第二个常见义）",
          [w for w in ("gap", "snap", "comb", "sw", "nep")
           if re.search(r"\b%s\b" % w, joined)], ["nep"],
          "band gap / snap / comb 都会命中普通词")
    check("表中不得出现裸 dft（经典力场论文常顺带提 DFT）",
          bool(re.search(r"\bdft\b", joined)), False,
          "收了会造出方向相反的错：把真实的缺失压成不适用")

    # --- 表必须真的接到闸门上，不能只是躺在模块里 ---
    for k in ("constraints", "electrostatics", "dispersion_correction"):
        check(f"_APPLICABILITY 装了 {k} 的闸门", k in E._APPLICABILITY, True)

    # --- 端到端：闸门没接通的话，下面三条会退回 missing ---
    md = ("We performed molecular dynamics simulations of silicon using the "
          "Tersoff potential. A timestep of 1 fs was used and the temperature "
          "was held at 300 K by a Nose-Hoover thermostat.")
    for k in ("constraints", "electrostatics", "dispersion_correction"):
        check(f"Tersoff 体系：{k} 应判 not_applicable", field(md, k),
              "not_applicable")

    # 反方向：经典力场体系不得被误伤。这一条防的是"真实语料改对了、
    # 合成黄金集却坏了"，eval/score.py 必须仍是满分。
    cls = ("CHARMM36m was used with the TIP3P water model. Electrostatics "
           "were treated with PME and a 12 A cutoff.")
    check("经典力场体系：constraints 仍应判 missing",
          field(cls, "constraints"), "missing")

    # 未识别出力场时**绝不能**收紧：那会把真实的缺失压掉。
    # 这一条与上面"识别不出势函数时退回原行为"是同一个不变量，
    # 只是走完整管道再确认一次。
    check("未写力场名的石墨烯体系：静电仍应判 missing",
          field("we simulated a graphene sheet with a timestep of 1 fs and a "
                "Nose-Hoover thermostat at 300 K", "electrostatics"), "missing")


def test_partial_evidence_layer():
    """[18] 部分证据层：提到了、但不够具体 -> ambiguous。

    这一层是四态里唯一一条"匹配不成功也可以是 ambiguous"的路径。
    它新开了一条从 missing 出去的路，所以下面每条断言都是在钉它的边界，
    尤其是**它不许越过谁**：抽取器命中与适用性闸门都必须排在它前面。
    """
    # 主行为：指名了恒温器、通篇没声明系综
    # 合成夹具（不含任何真实论文正文）。
    named = ("We used the Nose-Hoover thermostat to hold the temperature at "
             "300 K during the run.")
    check("只写恒温器、未写系综 -> ambiguous", field(named, "ensemble"),
          "ambiguous",
          "全语料 3 篇同类情形标注均为 ambiguous，且裁定均为工具错")

    # 明确写了系综名就不能再降级，抽取器命中赢过 partial。
    declared = named + " The production run was performed in the NVT ensemble."
    check("写了 NVT -> present，不得被 partial 降级",
          field(declared, "ensemble"), "present")

    # 不变式 2：partial 不产生取值
    got = run_pipeline(named)["extraction"]["fields"]["ensemble"]
    check("partial 不产生取值", got.get("value"), None,
          "报了 ambiguous 却顺手给个数，就是编数")
    check("partial 必须说明理由", bool(got.get("note")), True)
    check("partial 必须留下可核验的证据", bool(got.get("evidence")), True,
          "报'提到了但不够具体'时，必须能指出提到的是哪句")

    # 不变式 1：闸门赢过 partial（顺序即判据）
    # correlation_window 同时有闸门（heat_transport）与 partial 探针，
    # 正好用来钉顺序。同一句话，加不加"thermal conductivity"结果相反：
    #   不加热输运语境 -> 闸门判 not_applicable
    #   加热输运语境   -> 闸门放行，partial 才轮到出手
    acf = "We computed the HCACF from the stored trajectories."
    check("非热输运语境：闸门先赢，判 not_applicable",
          field(acf, "correlation_window"), "not_applicable",
          "partial 若排在闸门之前，17 篇无系综语境的恒压器会整体翻面")
    check("热输运语境：闸门放行后才轮到 partial",
          field(acf + " The thermal conductivity was obtained from it.",
                "correlation_window"), "ambiguous")

    # 窄版本才是量出来的：只给盒子几何算 ambiguous，粒子数不算
    box = ("The simulation cell was 5 x 5 x 5 nm and periodic boundary "
           "conditions were applied.")
    check("只给盒子几何、不给原子数 -> ambiguous",
          field(box, "system_size"), "ambiguous",
          "spec 原句：只给盒子尺寸不给原子数，属 ambiguity")
    # 直接问探针，不问字段状态：字段状态还掺着抽取器是否命中，
    # 量不出"这条探针的边界在哪"。
    check("给了粒子数（不是盒子尺寸）不触发该探针",
          E._partial_evidence(
              "system_size", "The system contained 63 water molecules."), None,
          "宽版本实测 6 fixed / 3 broken，三篇标注均为 missing")

    # 反方向：探针不得被当成"没提就是不存在"
    # 本层只在**热流自关联被点名写出**时触发。它说的不是
    # "没提关联函数 -> not_applicable"（那条已实测否决，见 fp_hunt.py）。
    check("全文无热流词汇时，不得由本层断言'不存在'",
          field("We ran an NVE simulation for 1 ns.", "correlation_window"),
          "not_applicable",
          "闸门本身就会判 N/A，但那是闸门判的，不是 partial 判的")

    # 表自身的卫生
    bad = [k for k in E._PARTIAL if k not in E.PATTERNS]
    check("_PARTIAL 的键都必须有对应抽取器", bad, [],
          "没有抽取器的字段挂 partial，等于用 ambiguous 盖住'根本没写抽取器'")
    thin = [k for k, v in E._PARTIAL.items()
            if not v or any(not why.strip() for _p, why in v)]
    check("_PARTIAL 每条都必须写理由", thin, [],
          "理由必须来自 spec.md；语料只用来量'正则是否精确实现了那条判据'")


def test_corpus_damage_and_ensemble_names():
    """[19] 语料损伤修复（甲-2）与系综全称写法（甲-4）。

    两件事放一组，因为它们都属"**说法不同 ≠ 没写**"：
    重音字母被文本管线切开、系综写成统计力学的标准名，
    原实现都当成"没报告"，而 missing 的含义是"文本没有给出"。
    """
    # 甲-2：重音分裂
    # LaTeX -> 文本的转换会把 Nosé-Hoover 切两半。两种切法各命中一篇。
    for broken, tag in (("N 'ose-Hoover", "N 'ose-Hoover 型"),
                        ("Nos ' e -Hoover", "Nos ' e -Hoover 型")):
        t = f"The temperature was controlled by a {broken} thermostat."
        check(f"{tag}分裂写法仍应认出恒温器", field(t, "thermostat"), "present")
        check(f"{tag}应规范成 Nose-Hoover",
              (value(t, "thermostat") or {}).get("type"), "Nose-Hoover")

    # 反方向：修复不能碰正常撇号。宽度必须窄到只认这两条已知切法，
    # 否则 "system's" / "Fourier's" 会被改坏。
    for ok in ("the system's potential energy", "Fourier's law was applied"):
        check(f"正常撇号不得被改写：{ok[:22]}",
              E.normalize_ws(ok), ok)

    # 甲-4：全称写法
    for phrase, want in (("isothermal ensemble", "NVT"),
                         ("canonical ensemble", "NVT"),
                         ("microcanonical ensemble", "NVE"),
                         ("isothermal-isobaric ensemble", "NPT"),
                         ("constant pressure ensemble", "NPT")):
        check(f"{phrase} -> {want}",
              value(f"The simulation used the {phrase} throughout.", "ensemble"),
              want)

    # grand canonical 是 µVT，不在本字段的 NPT/NVT/NVE 词表里，
    # 不能因为字符串里含 "canonical" 就报成 NVT。"不知道"要报成不知道。
    check("grand canonical（µVT）不得被报成 NVT",
          value("We sampled the grand canonical ensemble.", "ensemble"), None)

    # 修饰语必须与 ensemble 紧邻：isothermal 单独出现多是物理量，不是系综。
    check("isothermal compressibility 不是系综",
          value("We measured the isothermal compressibility.", "ensemble"),
          None)

    # 缩写优先于全称：两条都在时，显式缩写是更具体的证据。
    check("同时出现时，显式缩写优先",
          value("The isothermal ensemble was used for equilibration, then we "
                "switched to NPT production.", "ensemble"), "NPT")


def test_temperature_bath_recall():
    """[20] 冷热浴召回（乙-2）与退役的死 ctx 键。

    三件事都是"说法不同 ≠ 没写"的变体，但各自的边界不同：
      · source/sink 是**同一个东西的另一种叫法**，该给值；
      · 浴温是符号式、只有 ΔT 有数，该给**那个数**，不许替符号编一个开尔文值；
      · 点名了源汇却一个温度都没给，是 ambiguous 而不是 missing。

    夹具全部是构造句（数值与真实论文不同），只保留触发正则所必需的构型。
    """
    # (a) source/sink 写法：另一个名字，同样是冷热浴
    # 真语料里 NEMD 的"热端/冷端"有一半写成 heat source / heat sink。
    # 这里把两个温度都写成数字，读出来就该是 hot/cold。
    ss = ("We performed NEMD simulations to obtain the thermal conductivity. "
          "A steady-state heat flux was generated by coupling a source region "
          "to a thermostat with a higher temperature of 350 K and a sink "
          "region to a thermostat with a lower temperature of 290 K.")
    check("source/sink 写法应读出冷热浴温度", field(ss, "temperature_bath"),
          "present", "1805.00269v1 类：同句、两个数字、差 60 K，只差 hot/cold 字面")
    val = value(ss, "temperature_bath") or {}
    check("冷热两端各自的温度都要报出来",
          (val.get("hot_K"), val.get("cold_K")), (350, 290))
    check("温差要算对", val.get("delta_K"), 60)

    # ΔT 相对平均温度 18.75% > 10%，R-11 该响。这条同时钉住
    # "规则漂移已逐条确认"：规则一直装着，只是以前没值可算。
    check("ΔT/T_avg 超阈值时 R-11 应触发", _has_linearity_warning(ss), True,
          "1805.00269v1：ΔT/T=20%，且正文自认源汇附近有非线性温区")

    # (b) 只有 ΔT 有数字、浴温是符号式
    # 符号式的"±"在 LaTeX→文本时会被吃掉（2207.13405 只剩 "T+ T/2"）。
    # 这时唯一带单位的数是 ΔT：spec 这个字段叫"冷热浴温度**与梯度**"，
    # 梯度有数就该报出来；浴温是符号，**不替它编一个数**。
    # 夹具要写明 NEMD，那既是这类文献的常态，也是 nemd 闸门的输入。
    dt = ("We performed NEMD simulations. In the x direction there are two "
          "thermostatted regions, serving as a heat source with a higher "
          "temperature T+ T/2 and a heat sink with a lower temperature T- T/2. "
          "With a temperature difference of T=25 K, a typical temperature "
          "profile is shown in Fig. 1.")
    check("只有 ΔT 有数值时仍应判 present", field(dt, "temperature_bath"),
          "present")
    dval = value(dt, "temperature_bath") or {}
    check("只报 ΔT，不许凭空造出开尔文浴温",
          (dval.get("hot_K"), dval.get("cold_K"), dval.get("delta_K")),
          (None, None, 25),
          "符号式浴温补成数字就是取值错误（缺失会被看见，错值不会）")
    check("只有 ΔT 时 R-11 不得触发（它要冷热两端同时在场）",
          _has_linearity_warning(dt), False,
          "否则会在明确保持小 ΔT 的论文上误报'偏离线性区'")

    # (c) 点名了源汇、一个温度都没给 -> ambiguous，不是 missing
    only = ("A temperature gradient was imposed between the heat source "
            "region and the heat sink region.")
    check("点名源汇却无温度 -> ambiguous", field(only, "temperature_bath"),
          "ambiguous", "spec：只给方法不给温度设置，无法判断是否在线性区")
    got = run_pipeline(only)["extraction"]["fields"]["temperature_bath"]
    check("partial 路径不得产生取值", got.get("value"), None)
    check("partial 路径必须留下证据", bool(got.get("evidence")), True)

    # (d) 反方向：Green-Kubo 论文不得被这条探针拖进冷热浴
    # 闸门（nemd）排在 partial 之前，所以平衡态方法压根轮不到探针出手。
    gk = ("The thermal conductivity was obtained from the Green-Kubo formula "
          "using the heat current autocorrelation function.")
    check("Green-Kubo 论文：冷热浴应判 not_applicable",
          field(gk, "temperature_bath"), "not_applicable",
          "GK 是平衡态方法、没有浴；报'未报告冷热浴'就是假阳性")

    # (e) 只有比较级、没有第二个温度时不得取值
    # "higher temperature" 是相对说法，单独出现不构成冷热两端的数值。
    check("只有比较级和一个温度时不取值",
          E._extract_temperature_bath("The surface was held at a higher "
                                      "temperature of 400 K."), None)

    # (f) 表自身的卫生
    entry = E._PARTIAL.get("temperature_bath") or []
    check("temperature_bath 挂上了 partial 探针", len(entry), 1)
    check("该探针的理由必须引 spec 原句",
          "只给方法不给温度设置" in (entry[0][1] if entry else ""), True)

    # (g) 退役的死 ctx 键不得被重新加回来
    # has_temperature_grade 曾是只写不读的死键：任何一次对
    # _extract_temperature_bath 的改动都会在账本里报成 ctx 漂移，
    # 而温度浴闸门读的是 nemd。加回来就会重新制造假警报。
    c = ctx("We ran a NEMD simulation with a temperature gradient.")
    check("detect_context 不得再有 has_temperature_grade",
          "has_temperature_grade" in c, False,
          "只写不读；账本对它的处置见 status_diff.py 的 RETIRED_CTX")


def test_recall_vocabulary_batch():
    """[21] 乙-2 召回批：同一件事的**另一种说法**（放宽后各语料只多 1-2 处）。

    这一层最容易犯的错是**宽**：一条"顺便也认"的正则会把别篇
    正确的 missing 变成错误的 present。所以每条放宽都在注释里写明了
    实测可达范围，下面则钉住"不许越界"的那一侧。
    """
    # Green-Kubo 的等价说法（2110.08914v2）
    gk = ("The heat current time correlation functions (HCCFs) were obtained "
          "from 30 independent NVE trajectories. The thermal conductivity was "
          "computed from them.")
    check("HCCF 写法应认出 Green-Kubo", value(gk, "heat_flux_method"),
          "Green-Kubo")

    # NEMD 的两端热浴（1007.5359v2）
    # 夹具必须带热输运语境：heat_flux_method 的闸门是 heat_transport，
    # 少了它连闸门都过不去，量到的就不是这条正则的效果。
    nemd = ("The left/right heat baths are applied to regions near the ends in "
            "order to obtain a large thermal current. The thermal conductivity "
            "was obtained from the resulting flux.")
    check("两端热浴写法应认出 NEMD", value(nemd, "heat_flux_method"),
          "Direct NEMD")

    # microcanonical **system**（2108.03424v1）：少了 ensemble 一词
    check("microcanonical system -> NVE",
          value("In the EMD runs the system is evolved in a microcanonical "
                "system (isolated ring).", "ensemble"), "NVE")
    # 反方向：这个口**只**给 microcanonical 开。描述体系性质的
    # "isothermal system" 到处都是，宽开会把 missing 变成错误的 present。
    check("isothermal system 不得被当成系综声明",
          value("The isothermal system was kept at 300 K throughout.",
                "ensemble"), None)

    # fully flexible（2110.08914v2）：直接回答了"没有上约束"
    check("fully flexible 应认出（没有键约束）",
          value("Simulations used the fully flexible CHARMM36 potential.",
                "constraints"), "fully flexible")
    # 顺序不变式：算法名必须赢。反了就是**取值错误**，一篇约束了水的体系
    # 会被报成"没有约束"。
    check("同时写了 SHAKE 时算法名优先",
          value("SHAKE was applied to the water bonds, while the solute was "
                "fully flexible.", "constraints"), "SHAKE")

    # dDsC（2404.14338v1）
    check("dDsC 应认出为色散校正",
          value("CHARMM36 was used, and an a posteriori dispersion correction "
                "dDsC was applied.", "dispersion_correction"), "dDsC")

    # 作者命名的势（2110.08914v2）
    # 势函数以作者姓氏命名是本领域常态（Tersoff / Brenner / WCA），
    # 所以报出姓氏对与既有各表是同一约定。
    check("作者命名的势应报出姓氏对",
          value("the molecular potential for nitramines proposed by Smith and "
                "Bharadwaj", "force_field"), "Smith and Bharadwaj")

    # 端基质子化态（1707.05453v1）
    check("charged termini 应认出质子化态",
          value("A peptide with charged termini (NH_3^+ and COO^-) was built.",
                "protonation_state"), "charged termini")

    # 配体：OPLS-AA + 电荷缩放（2602.17595v1）
    lig = ("Ligand parameters were taken from the OPLS-AA force field, and the "
           "ion charges were scaled with a factor of 0.75.")
    check("OPLS-AA + 电荷缩放 -> present", field(lig, "ligand_parameters"),
          "present")
    # 反方向：只给力场、没有任何电荷信息，spec 明写属 ambiguity。
    # 电荷**缩放**是真信息（它是电荷怎么定的），光有力场名不是。
    check("只给力场、无电荷信息 -> ambiguous",
          field("Ligand parameters were taken from the OPLS-AA force field.",
                "ligand_parameters"), "ambiguous")


def test_nemd_gate_for_end_heat_baths():
    """[22] 两端热浴 = NEMD 闸门（1007.5359v2）。

    这一格和别的召回不同：它要修的是**闸门**，不是抽取。原文说
    "The left/right heat baths are applied to regions …"，而闸门词表里
    只有热/冷（hot/cold）与温度梯度，没有"两端"这种空间写法，于是整篇
    被判成"没有冷热浴"→ temperature_bath 判 not_applicable。标注是
    ambiguous：**提到了冷热浴，只是没给两端的温度**。

    所以它同时钉两件事：闸门认得这种写法；闸门判"适用"之后，
    部分证据探针（"点名了冷热浴却没给温度"）接着判 ambiguous。
    """
    # 正向：两端热浴，闸门开 + 部分证据 -> ambiguous
    ends = ("The left/right heat baths are applied to regions one column away "
            "from the ends in order to obtain a large thermal current.")
    check("两端热浴应开 NEMD 闸门", ctx(ends).get("nemd"), True)
    check("两端热浴：无温度设置 -> ambiguous",
          field(ends, "temperature_bath"), "ambiguous",
          "闸门先赢、部分证据后到：这正是 1007.5359v2 那一格")
    # ambiguous 不产生取值，本项目的反复准则：缺失会被看见，错值不会。
    check("两端热浴的 ambiguous 不产生取值",
          value(ends, "temperature_bath"), None)
    # R-11（线性响应区判据）需要冷热两端的温度，没有温度就不许触发。
    check("缺冷热温度时 R-11 不得触发", _has_linearity_warning(ends), False)

    # 分隔符的另一种写法
    check("left and right heat baths 同样开闸门",
          ctx("Both the left and right heat baths were thermostatted "
              "separately.").get("nemd"), True)

    # 反向一：恒温器意义上的"热浴"不是冷热浴
    # 1208.6331v1 的 "coupling of the system to a heat bath at temperature T"
    # 说的是 Nosé-Hoover 的浴，与温度梯度无关。宽开会把该篇正确的
    # not_applicable 变成错误的 ambiguous。
    thermo = ("The system was coupled to a heat bath at temperature T through "
              "a Nosé-Hoover thermostat.")
    check("恒温器的 heat bath 不得开 NEMD 闸门",
          ctx(thermo).get("nemd"), False,
          "1208.6331v1 类：Nose-Hoover 的浴，与两端梯度无关")
    # 闸门必须赢过部分证据：这一篇探针也是命中的（同样出现 "heat bath"），
    # 若闸门晚于探针生效，这一格就会错判成 ambiguous。
    check("闸门赢过部分证据（恒温器浴 -> not_applicable）",
          field(thermo, "temperature_bath"), "not_applicable")

    # 反向二：跨句不许串味
    # "left" 与 "heat bath" 分属两句时是两件事，正则必须要求同一名词短语。
    check("跨句的 left … heat bath 不得开闸门",
          ctx("The atoms on the left end are held fixed. The heat bath is "
              "governed by a Langevin equation.").get("nemd"), False)

    # 反向三：boundary heat baths（2108.03424v1）
    # 那篇是真 NEMD，但它的 nemd 早已由别的词命中；"boundary" 这种写法
    # 没有被收进词表，所以这里钉的是"它不因这条新正则而改变"。
    check("boundary heat baths 不由这条正则命中",
          ctx("Finite-size effects arising from the use of boundary heat "
              "baths were examined.").get("nemd"), False)


def test_applicability_gates_batch():
    """[23] 适用性闸门（丙层）：问的对象在体系里根本不存在时，不该问作者报没报。

    三条判据都是从被问对象的定义里推出来的：
      · WCA 就是截断到势极小点的 LJ（Weeks-Chandler-Anderson, 1971），
        所以 "LJ/Lennard-Jones" 是它的**母名**，不构成"体系里有色散"的证据；
      · 反应力场的键随配位生成断裂、没有固定拓扑可约束，能力表自己的
        why 里就是这么写的，缺的只是散文写法没被 pattern 收进去；
      · spec 的字段名是「**可滴定残基**的质子化态」，闸门该问的就是
        "有没有可滴定基团"，而不是"有没有蛋白"。

    每条都同时钉正反两面：闸门该关时关，该开时必须还开着。夹具全部为
    合成短句，不取自任何真实文献正文。
    """
    # --- A. 色散证据分强弱两档 ---
    for t, want in (
            # 强证据：声明的是一个色散**处理**，或一个明确带吸引尾的势
            ("a dispersion correction was applied", True),
            ("van der Waals interactions were included", True),
            ("a 12-6 Lennard-Jones potential described the gold slab", True),
            # 弱证据（裸的势名）在别的族面前照样算数
            ("a Lennard-Jones fluid was simulated", True),
            # 但在 WCA 面前不算数：那多半是同一个体系的另一个叫法
            ("a purely repulsive WCA fluid reported in LJ units", False),
            ("Weeks-Chandler-Andersen particles were simulated", False),
            # 姓氏的两种拼法都要认：本语料唯一的全名写法是 Anderson
            ("Weeks-Chandler-Anderson particles were simulated", False),
            # 而 WCA 体系里真出现强证据时，仍然算数
            ("a WCA fluid confined by a 12-6 Lennard-Jones wall", True),
            ("a WCA fluid with a van der Waals attraction added", True)):
        check(f"色散证据 {t[:40]!r} -> {want}",
              E._ff_capability(t)["dispersion"], want)

    check("lj_derived 只标在由 LJ 截断而来的那一族上",
          [f["name"] for f in E.FF_FAMILIES if f.get("lj_derived")],
          ["纯排斥模型势"],
          "标记一旦被复制到别的族，那些体系里的 LJ 就会被误判成不存在")
    check("单档的 _POS_DISPERSION 确已删除",
          hasattr(E, "_POS_DISPERSION"), False,
          "分档之后它成了死状态（写进模块、没人读）")

    # --- B. 反应力场的散文写法 ---
    check("reactive MD 应判无固定拓扑",
          E._ff_capability("we ran reactive MD simulations of the interface"
                           )["bonds"], False)
    check("reactive molecular dynamics 同判",
          E._ff_capability("reactive molecular dynamics was employed"
                           )["bonds"], False)
    check("ReaxFF 照旧判无固定拓扑",
          E._ff_capability("the ReaxFF reactive force field")["bonds"], False)
    check("普通 MD 不得被这条正则沾上",
          E._ff_capability("we ran molecular dynamics simulations"
                           )["bonds"], True)
    check("reactive 修饰别的名词不得触发",
          E._ff_capability("the reactive site was solvated")["bonds"], True,
          "pattern 要求 reactive 后面紧跟 MD / molecular dynamics")
    check("兜底：反应体系里明写了约束算法就仍算有键",
          E._ff_capability("reactive MD with SHAKE applied to the O-H bonds"
                           )["bonds"], True)

    # --- C. 闸门的端到端（能力表要真的接到闸门上） ---
    react = ("We performed reactive MD simulations of the silica surface at "
             "300 K with a timestep of 0.5 fs.")
    check("反应体系：has_bonds 为假", ctx(react).get("has_bonds"), False)
    check("反应体系：constraints 判 not_applicable",
          field(react, "constraints"), "not_applicable")
    shaken = ("We performed reactive MD simulations. SHAKE was applied to "
              "all bonds involving hydrogen.")
    check("反应体系 + SHAKE：has_bonds 翻回真（兜底赢）",
          ctx(shaken).get("has_bonds"), True,
          "明写了算法就说明真有可约束的键")
    check("反应体系 + SHAKE：闸门不得关",
          field(shaken, "constraints") != "not_applicable", True,
          "这里抽取器本来就会命中约束值（present），关键是别被闸门压成不适用")
    check("WCA 体系：dispersion_correction 判 not_applicable",
          field("We simulated a purely repulsive WCA fluid in LJ units at a "
                "reduced temperature of 1.0.", "dispersion_correction"),
          "not_applicable")

    # --- D. 可滴定残基闸门 ---
    capped = ("The capped peptide ACE-ALA-NME was placed in a box of "
              "explicit water.")
    check("封端肽：蛋白仍判为含（protein_present 未动）",
          ctx(capped).get("protein_present"), True)
    check("封端肽：可滴定残基判为无",
          ctx(capped).get("titratable_residues_present"), False)
    check("封端肽：protonation_state 判 not_applicable",
          field(capped, "protonation_state"), "not_applicable")
    check("封端肽：R-09 不得触发",
          "R-09" in rules(capped), False,
          "字段判不适用、规则却报'含蛋白体系未说明质子化态'是自相矛盾输出")
    check("封端写法的全称拼写同样判无",
          ctx("We studied the capped peptide acetyl-alanine-N-methylamide in "
              "water.").get("titratable_residues_present"), False)

    # 反向一：普通蛋白，"作者没写"不是"不存在"的理由
    lyso = "Hen egg-white lysozyme was solvated in TIP3P water."
    check("普通蛋白：可滴定残基判为有",
          ctx(lyso).get("titratable_residues_present"), True)
    check("普通蛋白：protonation_state 仍判 missing",
          field(lyso, "protonation_state"), "missing")
    check("普通蛋白：R-09 仍触发", "R-09" in rules(lyso), True)

    # 反向二：封端**且**点名了可滴定基团，两个条件缺一不可
    both = "The capped peptide ACE-HIS-NME was simulated at neutral pH."
    check("封端但点了 HIS：可滴定残基仍判有",
          ctx(both).get("titratable_residues_present"), True,
          "排除只在'两端封端 且 全文无可滴定命名'时成立")
    check("封端但点了 HIS：闸门不得关",
          field(both, "protonation_state") != "not_applicable", True)

    # --- E. 单一判据：字段闸门与 R-09 必须是同一个定义处 ---
    check("闸门与 R-09 共用同一个函数对象",
          E._APPLICABILITY["protonation_state"] is E.has_titratable_groups,
          True)
    check("手工 ctx（无该键）退回旧判据：有蛋白即真",
          E.has_titratable_groups({"protein_present": True}), True,
          "审计脚本与旧夹具手工构造 ctx，不能因此报错或翻转")
    check("手工 ctx（无该键）退回旧判据：无蛋白即假",
          E.has_titratable_groups({"protein_present": False}), False)
    check("ctx 里有显式键时以它为唯一判据",
          E.has_titratable_groups({"protein_present": True,
                                   "titratable_residues_present": False}),
          False)


def test_backend_never_overstates_itself():
    # 这一组钉的是工具**对自己**的诚实性，不来自任何一篇文献。
    #
    # 缺陷原形：`--backend hybrid` 把 LLM_ONLY_FIELDS 交给模型，而该表当时
    # 唯一一项 missing_residues_note 不在 schema 的 18 字段内，返回值被
    # `if k not in by_key: continue` 整批丢弃。于是：客户端调用了、prompt
    # 发出去了、token 花了、结果**一格没进**，报告却写"抽取方式: hybrid"。
    # 也就是说，一份 100% 由规则层产出的检查单，把自己的来源标成了混合。
    #
    # 这与本工具存在的理由是同一件事的两面：它整份报告都在追问
    # "'报告了'和'真的做了'之间有没有缝"，它自己不能留这条缝。
    #
    # 因此 llm_error 的语义被收紧为"**本层零贡献**"，而不是"本层报错"：
    # 抛异常、没给客户端、没有输出通道、跑通了但 0 格进结果，四种都算。
    # backend_effective 随之落到 rule，检查单显式警告并附原因。
    #
    # 夹具全部为合成字符串。
    text = ("We simulated the protein in explicit water.\n"
            "The temperature was kept at 300 K using a Nose-Hoover thermostat.\n"
            "A 12 A cutoff was used for Lennard-Jones interactions.\n")

    calls = []

    def stub(prompt):
        calls.append(prompt)
        return ('{"thermostat": {"value": "Nose-Hoover", "status": "present",'
                ' "confidence": "medium", "evidence": "synthetic", "note": ""}}')

    def run(backend, client):
        return run_pipeline(text, backend=backend, llm_client=client)

    # --- A. 无输出通道 = 零贡献，必须自己说出来 ---
    calls.clear()
    h = run("hybrid", stub)
    meta = h["extraction"]["meta"]
    check("hybrid 目标字段无输出通道时，实际生效后端落回 rule",
          meta["backend_effective"], "rule",
          "请求 hybrid 而产出全来自规则层，就不许写 hybrid")
    check("零贡献时 llm_applied 为 0（与 backend_effective 互为佐证）",
          meta["llm_applied"], 0)
    check("没有输出通道的字段不去问模型（省一次调用，也免得谎报）",
          calls, [],
          "调用了却必然丢结果，等于花钱买一个假标签")
    check("llm_error 说明了零贡献的原因",
          "没有输出通道" in meta["llm_error"], True)
    check("请求的后端仍如实保留在 backend 里",
          meta["backend"], "hybrid",
          "backend 是请求，backend_effective 是事实，两者都要留")

    # --- B. 报告上必须看得见，不能只躺在 JSON 里 ---
    check("检查单警告该后端本次未生效", "未生效" in h["report"], True)
    check("检查单给出原因", "没有输出通道" in h["report"], True)

    # --- C. 0 贡献 ⇒ 结果必须与纯 rule 逐格相同（这是"等价"的可检验形式）---
    def cells(res):
        return {k: (v.get("status"), v.get("value"))
                for k, v in res["extraction"]["fields"].items()
                if not k.startswith("_")}

    check("hybrid 零贡献时，18 字段与 rule 逐格相同",
          cells(h) == cells(run("rule", None)), True,
          "自称 hybrid 却与 rule 一字不差，那就该自称 rule")

    # --- D. 有输出通道时必须真的生效，别修成"永远报 rule" ---
    # 把目标字段临时换成一个**在 schema 内**的键，走通正常路径。
    saved = E.LLM_ONLY_FIELDS
    try:
        E.LLM_ONLY_FIELDS = ["barostat"]
        calls.clear()
        ok = run("hybrid", stub)
    finally:
        E.LLM_ONLY_FIELDS = saved
    m2 = ok["extraction"]["meta"]
    check("有输出通道时 LLM 确实被调用", len(calls), 1)
    check("LLM 有产出时 backend_effective 保持 hybrid",
          m2["backend_effective"], "hybrid",
          "修复不得退化成'凡 hybrid 皆报 rule'")
    check("llm_applied 如实计数", m2["llm_applied"], 1)
    check("有产出时 llm_error 为空", m2["llm_error"], "")

    # --- E. 前两种零贡献（无客户端、抛异常）不得被这次修改碰坏 ---
    m3 = run("hybrid", None)["extraction"]["meta"]
    check("无客户端仍记为未运行且落回 rule",
          (m3["backend_effective"], "未提供" in m3["llm_error"]), ("rule", True))

    def boom(_prompt):
        raise RuntimeError("synthetic upstream failure")

    saved = E.LLM_ONLY_FIELDS
    try:
        E.LLM_ONLY_FIELDS = ["barostat"]
        m4 = run("hybrid", boom)["extraction"]["meta"]
    finally:
        E.LLM_ONLY_FIELDS = saved
    check("客户端抛异常仍记为错误且落回 rule",
          (m4["backend_effective"], "synthetic upstream failure" in m4["llm_error"]),
          ("rule", True))

    # --- F. 纯 rule 不该被误伤：它本来就不跑 LLM 层 ---
    r = run("rule", stub)["extraction"]["meta"]
    check("rule 后端无 llm_error（没跑不等于跑了没生效）",
          (r["backend_effective"], r["llm_error"]), ("rule", ""))


def test_size_count_must_belong_to_the_simulated_system():
    # 这一组钉的是"取值来自错误的指称对象"——本工具最危险的一类缺陷，
    # 因为它的症状不是缺失，而是**一个 high 置信度的错值**：
    # 检查单上看起来"这篇给了原子数"，读者据此横向比较各文献的体系规模，
    # 而那个数根本不是体系规模。
    #
    # 真实出处（两条都已在 eval/real/apply_adjudication_batch2.py 里
    # 裁定为 tool_wrong）：
    #   2207.13405   "914 structures with 54840 atoms in total"
    #                → NEP **训练集**，被模拟的硅烯全文未给原子数
    #   2602.17595v1 "...SMILES representation ... each polymer has
    #                 around 200 atoms" → 单条链的化学表示，不是模拟胞
    #
    # 夹具是合成句，只保留触发否决所需的结构，不引原文。
    from src import extract as E

    def size(t):
        r = E._extract_system_size(t)
        return r[0] if r else None

    # --- A. 训练集规模不得被当成体系规模 ---
    check("训练集条目数 + atoms 不得被当成体系规模",
          size("The training data cover 100 to 900 K. "
               "There are 914 structures with 54840 atoms in total."),
          None)
    check("数据集/数据库同理",
          size("Our database holds 12000 atoms across 300 snapshots."), None)

    # --- B. 化学表示的规模不得被当成体系规模 ---
    check("SMILES 表示的原子数不得被当成体系规模",
          size("The polymers were encoded into a SMILES representation "
               "such that each polymer has around 200 atoms."), None)
    check("单条链的规模（each chain）同理",
          size("We built a melt where each chain contains 1000 atoms "
               "in a periodic box."), None)

    # --- C. 真值不得被误伤：这三条是本语料里**正确**的那几个 ---
    # 1805.00269v1 的原句结构：晶体学的 cells 是真体系，前面那个
    # "6x6x6" 不是条目数，所以 "cells with a total of" 不构成否决。
    check("晶体学 cell 的原子数仍要报（1805.00269v1 的形状）",
          size("A simulation cell consisting of N_x x N_y x N_z = 6x6x6 "
               "conventional cubic cells with a total of N=1728 atoms "
               "was used for bulk silicon."),
          {"atoms": 1728})
    check("普通陈述句里的体系原子数仍要报",
          size("The simulation cell contained 8000 atoms."), {"atoms": 8000})
    check("含 representation 一词但不指化学表示的句子不应被否",
          size("The system consists of 64000 atoms in total."),
          {"atoms": 64000})

    # --- D. 否决掉第一个候选后，后面的候选仍要试 ---
    # 只查第一个命中就放弃，会把"开头提训练集、后文才给体系"的论文
    # 整篇判成缺失——那是把修错值换成了造缺失，不是修好了。
    check("前一处是训练集时，继续找后一处的真体系",
          size("Training used 914 structures with 54840 atoms. "
               "Separately, the simulated cell held 4096 atoms."),
          {"atoms": 4096})

    # --- E. 否决后若只剩盒子几何，落在 schema 的 ambiguous，而不是缺失 ---
    # 二维材料只写 W= / L= 的尺寸（没有立方盒子可写），上面三条要求
    # box/cubic/cell 的模式全部落空，必须有第四种写法兜住，
    # 否则 2207.13405 会从"错值"变成"缺失"，而标注是 ambiguous。
    r = E._extract_system_size(
        "The width in the transverse direction is chosen as W=13.3 nm. "
        "The training set had 914 structures with 54840 atoms in total.")
    check("符号式尺寸 W=13.3 nm 要认出来", r[0], {"box_nm": 13.3})
    check("只剩盒子尺寸时判 ambiguous（schema 原文）",
          r[3] if len(r) > 3 else "present", "ambiguous")
    check("符号式尺寸连列表写法一起收（L = a, b, c nm）",
          E._extract_system_size("Domain lengths: L = 7.7, 15.4, 30.8 nm.")[0],
          {"box_nm": 7.7})

    # --- F. 已有原子数时，不得再去别处凑一个尺寸 ---
    # 凑来的那个多半属于**另一个子体系**（1805.00269v1 的 1728 是体硅的，
    # 而 nm 尺寸是硅烯的 NEMD 宽度），把两份指称不同的数拼进同一个 dict，
    # 正是这一组要修的那类错误。
    check("有原子数时不再用符号式尺寸去补 box_nm",
          size("The cell has 1728 atoms. In the NEMD runs, L_y=10 nm."),
          {"atoms": 1728})

    # --- G. 端到端：语料里这两个字段的最终状态 ---
    def status_of(t, key):
        return run_pipeline(t)["extraction"]["fields"][key]["status"]

    train = ("The training data were obtained by active learning. "
             "There are 914 structures with 54840 atoms in total. "
             "The width in the transverse direction is W=13.3 nm.")
    check("端到端：训练集原子数不进 system_size 的取值",
          str(run_pipeline(train)["extraction"]["fields"]["system_size"]["value"])
          .find("54840"), -1)
    check("端到端：该字段最终落在 ambiguous", status_of(train, "system_size"),
          "ambiguous")

    ent = ("The polymers were encoded into a SMILES representation such "
           "that each polymer has around 200 atoms.")
    _v = run_pipeline(ent)["extraction"]["fields"]["system_size"]["value"]
    check("端到端：化学表示的原子数同样不进取值",
          (isinstance(_v, dict) and "atoms" in _v), False,
          f"实得 {_v!r}；200 若进了 atoms 就是回归")


def main() -> int:
    print("=" * 78)
    print("真实文献回归测试：每条都来自一轮 arXiv 语料排查")
    print("=" * 78)
    # 组标题集中在这里，不散在各函数体内：靠函数各自 print 会数漏组数，
    # 这份输出就会**低报自己的覆盖面**。
    groups = [
        (test_short_key_word_boundaries, "[1] 短键必须带词边界（取值错误类）"),
        (test_protein_keywords_need_word_boundary,
         "[2] 蛋白关键词必须带词边界（本项目最大的假阳性来源）"),
        (test_solvent_requires_positive_evidence,
         "[3] explicit_solvent：缺证据要判不适用，而不是判缺失"),
        (test_ambiguous_is_actually_produced,
         "[4] ambiguous 必须真的会被产生（文档与实现一致）"),
        (test_dispersion_context_required,
         "[5] 色散校正：'energy and pressure' 必须与 correction 同处一个短语"),
        (test_ligand_vocabulary_boundary,
         "[6] 配体词表：不收参数化工具名，substrate 双义词要分清"),
        (test_extractors_are_case_insensitive,
         "[7] 抽取器必须大小写不敏感（同一类漏检，一次性扫出来）"),
        (test_prose_phrasings_are_recalled,
         "[8] 散文写法必须能召回（漏检表现为'字段缺失'）"),
        (test_algorithm_aliases_across_codes,
         "[9] 同一算法在不同软件里的叫法都要认（PME / PPPM）"),
        (test_software_version_positions,
         "[10] 版本号可以在软件名之前或之后（原先只认一种）"),
        (test_substring_traps_beyond_word_boundary,
         "[11] 子串陷阱（续）：'leaps' 不是 Amber 的 LEaP"),
        (test_context_flags_are_canonicalised,
         "[12] 上下文标志必须规范化大小写（闸门读的是它）"),
        (test_coupling_prose_and_number_boundaries,
         "[13] 散文式耦合常数（R-02/R-12 的假阳性来源）"),
        (test_lammps_mixing_rule,
         "[14] LAMMPS 混合规则：漏写 pair_modify mix 是静默失效"),
        (test_corpus_comparability,
         "[15] 跨文献可比性：把分析单元从「一篇」换成「一组」"),
        (test_schema_rules_match_implementation,
         "[16] schema 里的一致性规则声明必须与实现一致"),
        (test_force_field_capability_table,
         "[17] 势函数能力表：势函数的定义里没有的东西，不该问作者报没报"),
        (test_partial_evidence_layer,
         "[18] 部分证据层：提到了、但不够具体 → ambiguous"),
        (test_corpus_damage_and_ensemble_names,
         "[19] 语料损伤修复与系综全称写法（甲-2 / 甲-4）"),
        (test_temperature_bath_recall,
         "[20] 冷热浴召回与退役的死 ctx 键（乙-2）"),
        (test_recall_vocabulary_batch,
         "[21] 召回批：同一件事的另一种说法（乙-2）"),
        (test_nemd_gate_for_end_heat_baths,
         "[22] 两端热浴 = NEMD 闸门（1007.5359v2）"),
        (test_applicability_gates_batch,
         "[23] 适用性闸门（丙层）：问的对象在体系里根本不存在"),
        (test_backend_never_overstates_itself,
         "[24] 后端不得虚报自己：0 贡献就落回 rule 并说明原因"),
        (test_size_count_must_belong_to_the_simulated_system,
         "[25] 取值必须来自正确的指称对象：训练集/化学表示的原子数不是体系规模"),
    ]
    for fn, title in groups:
        print(f"\n{title}")
        fn()

    print("\n" + "=" * 78)
    if _FAILS:
        print(f"失败 {len(_FAILS)} / {_N[0]}")
        for name, got, want, why in _FAILS:
            print(f"  FAIL {name}: 期望 {want!r} 实得 {got!r}")
        print("=" * 78)
        return 1
    print(f"全部通过：{_N[0]} / {_N[0]}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
