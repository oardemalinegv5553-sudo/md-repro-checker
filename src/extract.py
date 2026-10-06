# -*- coding: utf-8 -*-
"""
extract.py：从文献方法段抽取 MD 关键参数

设计原则（这是本项目的核心论点，也是与"端到端训一个模型"的分野）：

  规则负责"高精度、可枚举"的部分，力场名、水模型、时间步、截断半径。
  这些是封闭词表 + 固定书写范式，正则的精确率远高于 LLM，且零成本、可解释。

  LLM 负责"开放表达"的部分，初始结构来源、缺失残基处理、体系构建叙述。
  这些没有词表可枚举，规则召回率会崩。

  两者输出同一份 schema，下游的 validate.py 和 build_inputs.py 不关心
  参数是哪种方式抽出来的，这也是"可替换 backend"这个设计的意义。

输出统一为：
  {key: {"value":..., "status":..., "confidence":..., "evidence":..., "note":...}}

status: present | missing | ambiguous | not_applicable
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# 高精度正则表
#   key -> [(编译后的正则, 归一化函数)]
#   顺序有意义：先匹配到的先赢（更具体的写在前面）
# -----------------------------------------------------------------------------

FF_CANON = {
    "amber99sb-ildn": "AMBER99SB-ILDN",
    "amber99sb": "AMBER99SB",
    "amber ff99sb-ildn": "AMBER99SB-ILDN",
    "ff99sb-ildn": "AMBER99SB-ILDN",
    "ff14sb": "AMBER14SB",
    "amber14sb": "AMBER14SB",
    "charmm36m": "CHARMM36m",
    "charmm36": "CHARMM36",
    "charmm27": "CHARMM27",
    "opls-aa": "OPLS-AA",
    "opls_aa": "OPLS-AA",
    "oplsaa": "OPLS-AA",
    "compass iii": "COMPASS III",
    "pcff": "PCFF",
    "gaff2": "GAFF2",
    "gaff": "GAFF",
    "cgenff": "CGenFF",
    "openff": "OpenFF",
    "reaxff": "ReaxFF",
    "dreiding": "DREIDING",
    "uff": "UFF",
    "martini": "Martini",
    # 只写族名、不写版本号的写法（"we used the AMBER force field"）。
    # 必须收进来：否则这类句子一个字都匹配不上，字段被报成"未报告力场"，
    # 而作者其实报告了力场，只是没给版本。那是**假阳性**。
    # 收进来之后由 _FF_FAMILY_ONLY 把它们判成 ambiguous（见下）。
    "amber": "AMBER",
    "charmm": "CHARMM",
    "opls": "OPLS",
    "compass": "COMPASS",
}

# 族名之外还有通行版本号的力场。判据很干净：
# **本表里本来就存在这个族名的更具体变体**，那裸族名就是有歧义的。
#   AMBER  -> AMBER99SB / AMBER14SB / ...   差一个版本，结果差很多
#   CHARMM -> CHARMM27 / 36 / 36m           同上
#   GAFF   -> GAFF2                          GAFF 与 GAFF2 是两套参数
#   OPLS   -> OPLS-AA / OPLS-AA/L
#   COMPASS-> COMPASS III
# 反过来 UFF / DREIDING / PCFF / CGenFF 没有通行的版本号，族名就是完整答案，
# 把它们也报 ambiguous 等于**凭空制造歧义**，同样是假阳性。
_FF_FAMILY_ONLY = {"AMBER", "CHARMM", "GAFF", "OPLS", "COMPASS"}

# 族名的**报告侧**子表：FF_FAMILIES 认得、而 FF_CANON 报不出的势函数。
#
# 为什么需要它：同一模块里两张表互不知道对方，FF_CANON 决定"报出什么"，
# FF_FAMILIES 决定"这个势函数的形式里有没有库仑项/色散项"。Tersoff、Brenner、
# WCA、NEP 只出现在后者。后果是这几个势函数**在工具内部已经被识别，对外却
# 报成"未报告力场"**。实测这一类在 24 篇里占 6 格。
#
# 每项 = (变体, 规范名, 裸名是否不足以据以行动)。第三位沿用 _FF_FAMILY_ONLY
# 的判据：**同一个名字下有多个通行的参数化**，裸名就是有歧义的。
#   Tersoff  -> 原始参数化 / Munetoh 等，多套            裸名有歧义
#   Brenner  -> 第一代 / 第二代                          裸名有歧义
#   NEP      -> 由具体训练集定义，光有缩写不足以复现      裸名有歧义
#   WCA      -> 只有一套定义（LJ 截断到势极小点并平移）   裸名即完整答案
# 与 _FF_FAMILY_ONLY 同一条纪律：给只有一套定义的势函数判歧义 = **凭空制造歧义**。
#
# **只收量过的名字。** 每加一个词，都可能把某篇正确的 missing 翻成 present。
# 下面三张表就是实测修正的那 6 格所需的全部；FF_FAMILIES 的 pattern 里还有
# stillinger-weber / rebo / edip / meam 等，暂不接上来，等量过再说。
#
# 两类**刻意不收**，各有实测依据：
#   · 平面波族（VASP/CP2K/AIMD…）整族不接，把 "AIMD" 当力场值报出来是
#     **取值错误**：它不是力场，也不含泛函信息（而泛函才是这一类里
#     "差一个版本结果就不同"的那个东西）。相关篇目标注均为 missing。
#   · deepmd / DeePMD-kit 不接，那是软件包名（"LAMMPS interfaced with
#     DeePMD-kit"），接上来会把一篇正确的 missing 变成错误的 present。
_FF_NAMES_BONDORDER = [
    ("tersoff", "Tersoff", True),
    ("brenner", "Brenner", True),
]
_FF_NAMES_ML = [
    # 长变体必须与缩写同表：查表按变体长度降序，全称优先命中。
    ("neuroevolution potential", "NEP", True),
    ("nep", "NEP", True),
]
_FF_NAMES_WCA = [
    ("weeks-chandler-andersen", "WCA", False),
    ("wca", "WCA", False),
]

_FF_NAME_CANON: Dict[str, str] = {}
_FF_NAME_BARE_AMBIGUOUS: Dict[str, bool] = {}
for _names in (_FF_NAMES_BONDORDER, _FF_NAMES_ML, _FF_NAMES_WCA):
    for _variant, _canon, _bare_ambiguous in _names:
        _FF_NAME_CANON[_variant] = _canon
        _FF_NAME_BARE_AMBIGUOUS[_canon] = _bare_ambiguous
del _names, _variant, _canon, _bare_ambiguous

WATER_CANON = {
    "tip3p": "TIP3P",
    "tip4p/2005": "TIP4P/2005",
    "tip4p/ice": "TIP4P/Ice",
    "tip4p": "TIP4P",
    "tip5p": "TIP5P",
    "spc/e": "SPC/E",
    "spce": "SPC/E",
    "spc": "SPC",
    "opc": "OPC",
    "opc3": "OPC3",
}

THERMOSTAT_CANON = {
    "v-rescale": "V-rescale",
    "vrescale": "V-rescale",
    "berendsen": "Berendsen",
    "nose-hoover": "Nose-Hoover",
    "nosé-hoover": "Nose-Hoover",
    "nosehoover": "Nose-Hoover",
    "langevin": "Langevin",
    "andersen": "Andersen",
    "csvr": "CSVR",
    "bussi": "CSVR (Bussi)",
}

BAROSTAT_CANON = {
    "parrinello-rahman": "Parrinello-Rahman",
    "parrinellorahman": "Parrinello-Rahman",
    "c-rescale": "C-rescale",
    "crescale": "C-rescale",
    "berendsen": "Berendsen",
    "mttk": "MTTK",
    "monte carlo": "Monte Carlo",
}

HEATFLUX_CANON = {
    "green-kubo": "Green-Kubo",
    "green kubo": "Green-Kubo",
    "green–kubo": "Green-Kubo",  # en dash 是原文排版写法，键必须是 en dash 才匹配
    # 全称写法常与缩写混用，必须都收：文献里"direct NEMD"与
    # "direct non-equilibrium molecular dynamics"出现频率相当。
    # 注意 "non-equilibrium molecular dynamics" 内含
    # "equilibrium molecular dynamics"，查表按变体长度降序匹配，
    # 长变体优先命中，因此不会误判为 EMD。
    "non-equilibrium molecular dynamics": "Direct NEMD",
    "nonequilibrium molecular dynamics": "Direct NEMD",
    "non equilibrium molecular dynamics": "Direct NEMD",
    "equilibrium molecular dynamics": "EMD",
    "rnemd": "RNEMD (Muller-Plathe)",
    "muller-plathe": "RNEMD (Muller-Plathe)",
    "müller-plathe": "RNEMD (Muller-Plathe)",
    "direct nemd": "Direct NEMD",
    "nemd": "Direct NEMD",
    "emd": "EMD",
    # Green-Kubo 的**等价说法**：说的是同一件事（平衡态下热流的自关联），
    # 只是全句没写 "Green-Kubo" 这个词。归一到 GK 是安全的，因为这些短语
    # 就是 GK 定义式里的那个量：
    #   · 2110.08914v2 "The heat current time correlation functions (HCCFs),
    #     defined as C(t)=<J(0)J(t)>, were obtained from 30 independent …
    #     (NVE) trajectories"（该格 gold=present，工具报 missing）
    #   · 2207.13405 "heat current autocorrelation function (HCACF)"
    # 都归到同一个规范名，所以已命中的篇（1805/2207）取值不变，
    # 只是证据片段可能换成更长的那条，状态不移动。
    "hccf": "Green-Kubo",
    "heat current time correlation": "Green-Kubo",
    "heat current autocorrelation": "Green-Kubo",
    "heat current correlation": "Green-Kubo",
}

# NEMD 的"两端热浴"写法：热浴加在两端区域上以建立热流。这是直接法 NEMD 的
# 标准描述，但整句没有 "NEMD" 三个字母，查表查不到。
# 全语料只有 1007.5359v2 一处："The left/right heat baths are applied to
# regions one column away from the left/right ends in order to obtain a large
# thermal current across the system."（该格 gold=present，工具报 missing）
# 形式刻意与量过的那条一字不差：放宽到 `heat bath … region` 会把 1208 的
# "the system to a heat bath"（那是平衡段恒温器）一并吃进来。
_HEATFLUX_NEMD_RE = re.compile(r"\b(?:left|right)\s+heat\s+baths?\b", re.I)

CONSTRAINT_CANON = {
    "lincs": "LINCS",
    "shake": "SHAKE",
    "settle": "SETTLE",
    "h-bonds": "h-bonds only",
    "hbonds": "h-bonds only",
}

DISPERSION_CANON = {
    "enerpres": "EnerPres",
    "ener-pres": "EnerPres",
    "force-switch": "force-switch",
    "forceswitch": "force-switch",
    # dDsC = density-dependent dispersion correction（Steinmann & Corminboeuf）。
    # 它是**带名字的**色散校正方法，与 EnerPres / force-switch 同类，
    # 作者点名了用哪一种校正，就不该报"未报告"。
    # 全语料只出现 1 处：2404.14338v1 "An additional a posteriori density-dependent
    # dispersion correction dDsC is applied."（该格 gold=present，工具报 missing）
    "ddsc": "dDsC",
}

# "energy and pressure" 曾经作为独立键放在上表里，已撤掉：
# 它太泛，"the error in energy and pressure"、"energy and pressure
# fluctuations" 之类句子都会被映射成 EnerPres，这是**取值错误**。
# 正确的做法是要求 "correction" 与它同处一个短语。
_DISP_CTX = re.compile(
    r"\b(?:long[- ]range|tail|dispersion|standard)\s+corrections?\b[^.]{0,40}?"
    r"\benergy and pressure\b", re.I)


# 语料损伤修复：LaTeX -> 文本的转换管线会把重音字母切开。
#   Nosé-Hoover  ->  "N 'ose-Hoover"（1007.5359v2）
#                ->  "Nos ' e -Hoover"（1805.00269v1）
# README 已把"重音字母被切断"列为语料四个已知缺陷之一。THERMOSTAT_CANON 里
# 本来就有 "nosé-hoover" 变体，但查表走的是子串 find，切开之后永远找不到，
# 于是这两篇的恒温器被报成"未报告"。
#
# 这是工具对**文本管线副产物**的鲁棒性，不是语料修复：语料保持原样，
# 因为 24 篇的标注本来就是对着损坏文本做的。
#
# 两条约束，都是为了不碰坏别的东西：
#   1. 必须连着 "Hoover" 一起匹配，表里的键是 "nose-hoover"，
#      只把 "Nos ' e" 修成 "Nose" 会留下 "Nose -Hoover"（中间有空格），
#      照样匹配不上。也就是说，**光修名字不修连字符是没用的**。
#   2. 必须窄到只认这两个专名。语料里有 17 处正常撇号（"system's"、
#      "Fourier's"），一律不许碰。
_ACCENT_SPLIT = [
    (re.compile(r"\bN\s*'\s*ose\s*-\s*Hoover", re.I), "Nose-Hoover"),
    (re.compile(r"\bNos\s*'\s*e\s*-\s*Hoover", re.I), "Nose-Hoover"),
]


def normalize_ws(text: str) -> str:
    """
    把任意空白串（含换行、制表、连续空格）压成单个空格，并修掉语料损伤。

    这一步不是美化，是匹配正确性的前提：文献 PDF 转文本后的硬换行会让
    多词短语在行尾折断，"non-equilibrium molecular\\ndynamics" 用
    "non-equilibrium molecular dynamics" 去 find 是找不到的。
    实测中这一类失败占多词字段漏检的大部分，且极难从输出上看出来
    （表现为"字段缺失"，而不是报错）。

    归一化后所有匹配都基于同一份文本，证据引用也随之统一；
    重音切分的修复（见 _ACCENT_SPLIT）也在这里做，因为
    extract / _ff_capability / detect_context 三条路径都先过这里。
    """
    text = re.sub(r"\s+", " ", text)
    for pat, repl in _ACCENT_SPLIT:
        text = pat.sub(repl, text)
    return text


def _variant_lookup(table: Dict[str, str], text: str) -> Optional[Tuple[str, str]]:
    """在 text 中查找表中的任一变体（大小写不敏感），返回 (规范名, 命中片段)。"""
    low = text.lower()
    # 长变体优先，避免 "gaff" 抢先命中 "gaff2"
    for variant in sorted(table, key=len, reverse=True):
        # 短的纯字母数字键必须带词边界。实测 `opc`（一种水模型）会命中
        # "dopc"（一种磷脂），于是一篇只提过 DOPC 的膜论文报出
        # "水模型 = OPC"。这是**取值错误**：错的值会被下游当成事实读走，
        # 比"字段缺失"危险得多（缺失至少是诚实的）。
        # 带符号的键（"spc/e"、"tip4p/2005"、"ener-pres"）不受影响，
        # 继续走下面的子串匹配。
        if len(variant) <= 6 and variant.isalnum():
            m = re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(variant), low)
            if m:
                return table[variant], text[m.start(): m.end()]
            continue
        idx = low.find(variant)
        if idx >= 0:
            return table[variant], text[idx: idx + len(variant)]
    return None


# 每个字段的抽取器：输入方法段全文，输出 (value, confidence, evidence_span) 或 None。
# 可选第 4 位是**状态覆盖**，用于需要报 ambiguous 的场合；省略即 present。
def _extract_force_field(t: str):
    hit = _variant_lookup(FF_CANON, t)
    if hit:
        canon, frag = hit
        if canon in _FF_FAMILY_ONLY:
            # "we used the AMBER force field"，说了力场，没说哪个版本。
            # 报 present 是错的：下游会拿 "AMBER" 去写 mdp / 比对文献，
            # 而 AMBER99SB 与 AMBER14SB 结果不同；报 missing 也错，因为作者确实报告了。
            # 这正是四态里 ambiguous 的用武之地，语义上"有，但不足以据以行动"。
            return canon, "medium", frag, "ambiguous"
        return canon, "high", frag

    # FF_CANON 全落空，再看族名子表（Tersoff / Brenner / WCA / NEP，它们
    # 原先只出现在 FF_FAMILIES 里，于是被整个漏报成"未报告力场"）。
    # 顺序不能反：一篇同时写了 GAFF2 与 Tersoff 的文献应当报出更具体的 GAFF2。
    hit = _variant_lookup(_FF_NAME_CANON, t)
    if hit:
        canon, frag = hit
        if _FF_NAME_BARE_AMBIGUOUS.get(canon) and not _ff_param_near(t, frag):
            return canon, "medium", frag, "ambiguous"
        return canon, "high", frag

    # 最后一道：**作者命名的势**。"本领域的势函数大量以作者姓氏命名"
    # （Tersoff、Brenner、Stillinger-Weber、Weeks-Chandler-Anderson 都是人名），
    # 所以"the molecular potential for X proposed by A and B"报出姓氏对，
    # 与既有各表是同一约定，作者点名了是**哪一个**势，就不是歧义。
    # 全语料只命中 2110.08914v2 一处（gold=present，工具报 missing）。
    # 兜底位置刻意放在最后：任何表内命中的名字都更规范、更该优先。
    m = _FF_AUTHOR_RE.search(t)
    if not m:
        return None
    return re.sub(r"\s+", " ", m.group(1)).strip(), "medium", m.group(0).strip()


# 参数化/代次标记：命中处附近出现这些词，说明作者点名了**是哪一个**参数化，
# 裸名歧义因而不成立。这条是 spec.md 那句「只写 AMBER 或 CHARMM 而未给版本，
# 一律标为 ambiguity」的**反面**，说了参数化就不是歧义。
# 实测四种写法：'second-generation Brenner'（1007.5359v2）、'Tersoff with the
# original parameterization'（1805.00269v1）、'Tersoff ... with Munetoh et al.'s
# parameters'（2506.04835v1）该判 present；'the Tersoff bond-order potential'
# 附近没有任何标记（1208.6331v1）该判 ambiguous。
_FF_PARAM_NEAR = re.compile(
    r"second[- ]generation|first[- ]generation|"
    r"parameteri[sz]\w*|re-?parametri[sz]\w*|\bparameters?\b", re.I)


_FF_AUTHOR_RE = re.compile(
    r"\bpotential\b[^.]{0,50}?\bproposed\s+by\s+"
    r"([A-Z][A-Za-z\-]+(?:\s+and\s+[A-Z][A-Za-z\-]+)?)")


def _ff_param_near(text: str, frag: str, window: int = 80) -> bool:
    """命中片段 ±window 字符内是否点名了参数化。

    取片段的**首次**出现处：势函数一般是在首次提及处被定义的。
    """
    idx = text.find(frag)
    if idx < 0:
        return False
    lo = max(0, idx - window)
    hi = min(len(text), idx + len(frag) + window)
    return _FF_PARAM_NEAR.search(text[lo:hi]) is not None


def _extract_water(t: str):
    hit = _variant_lookup(WATER_CANON, t)
    if not hit:
        return None
    canon, frag = hit
    return canon, "high", frag


_COUPLING_NEAR = re.compile(
    r"\b(?:thermostat|barostat|bath|piston|langevin|manostat|pressure)\b", re.I)


def _coupling_prose(t: str):
    """散文式耦合常数："with a coupling constant of 0.1 ps"。

    为什么必须补：符号式（`tau_t = 0.1 ps`）主要出现在方法学论文里，
    真实文献更常写 "with a coupling time of 0.5 ps"。原实现只认符号式，
    于是这些**明明给了耦合常数**的文献被 R-02/R-12 判成"未给耦合常数"，
    正是本项目最忌讳的假阳性（实测语料里 2207.13405 与 cond-mat_9910397v1
    两篇中招，后者写的是 "weak-coupling scheme of Berendsen with a coupling
    time of t_coup* = 4…"）。

    两层保守：
      1. **coupling/relaxation 必须与数值紧邻**，中间只允许白名单连接词
         （of/was/were/set/to/=…）。白名单外的词立刻截断，所以它不会跨过
         半个句子去够一个不相干的数。"coupling of energy and momentum
         fluxes"、"relaxation time of the polymer conformation"（都在
         2312.07027v2 里）因此都不会命中。
      2. **relaxation 还要求器件词在附近**（thermostat/barostat/piston/
         Langevin…）。否则聚合物的弛豫时间会被读成恒温器耦合常数。

    返回值第三项 unit 可以为空，表示"给了数、没给单位"。这种情形
    必须与"没给耦合"区分开（否则 R-02/R-12 会假阳性），但**只降 conf、
    不改状态**：调用方保留这个数，状态仍是 present。为什么不是 ambiguous，
    见 _extract_thermostat 里那段实测。
    """
    # 数值前缀：允许 LaTeX 风格的角标（"t_ coup ^ * = 4.66"）。
    _SYM = r"(?:t\s*[_^]?\s*\w+\s*[_^]?\s*\*?\s*=\s*)?"
    # 连接词允许**一串**（"parameters were set to 200 fs" 是三个词），
    # 但每个词都在白名单里。
    _CONN = r"(?:\s*(?:of|was|were|is|are|set|to|equals?|=|＝|：|:))*\s*"
    _UNIT = r"(fs|ps|ns|飞秒|皮秒)"
    # 无单位数值。两道数字边界是必须的，缺一个就会**截出半个数**：
    # 实测 "set to 200 fs and 20 fs" 里 `[0-9]*\.?[0-9]+` 会从 "200" 中
    # 匹配出 "20"（`[0-9]*`吃"2"、`[0-9]+`吃"0"），留下 "0 fs"，
    # 于是耦合常数被报成 20 fs 而不是 200 fs。这是**取值错误**，
    # 会被下游当成事实。加边界后该处直接不匹配。
    # 后面只排除**时间单位**而不是所有字母：否则 "50 and an oscillation
    # period of 100" 这种（数后面跟 and）会被一起误杀。
    _NUM0 = (r"(?<![\d])([0-9]*\.?[0-9]+)(?![\d])"
             r"(?!\s*(?:fs|ps|ns|飞秒|皮秒)\b)")

    tries = [
        # ① coupling + 限定词 + 数值 + 单位（`s?` 收复数）
        (r"\bcoupling\s+(?:constant|time|parameter|strength|coefficient)s?"
         + _CONN + _SYM + r"([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b", True),
        # ② 无单位：给了数但没写单位（cond-mat_9910397v1 的 "= 4.66"、
        #    2604.01940v3 的 "relaxation time of 50" 都是这种）。
        #    **不能当成"没写"**，那会让 R-02/R-12 报出"未给耦合常数"，
        #    而文献明明给了。但也不能把这个数当真值用：它是自然单位，
        #    4.66 直接当 ps 读就是编数。所以照抓，但标成无单位，
        #    由调用方降级为 ambiguous（与时间步的自然单位处理一致）。
        (r"\bcoupling\s+(?:constant|time|parameter|strength|coefficient)s?"
         + _CONN + _SYM + _NUM0, False),
    ]
    for pat, need_unit in tries:
        m = re.search(pat, t, re.I)
        if not m:
            continue
        # "barostat and thermostat coupling parameters were set to 200 fs
        # and 20 fs, respectively"，两个数按器件顺序一一对应，而两个
        # 抽取器都会抓到**同一个**（第一个）数，恒温器会拿到恒压器的
        # 200 fs。这是取值错误，比漏抓更糟。这种句型正则分不清归属，
        # 整句放弃。
        if re.match(r"\s*(?:and|、|和)\s*[0-9]", t[m.end():m.end() + 12]):
            continue
        return m.group(1), (m.group(2) if need_unit else ""), m.group(0).strip()

    # ③④ relaxation 变体，必须邻近器件词
    for pat, need_unit in (
            (r"\brelaxation\s+(?:time|constant)s?" + _CONN + _SYM +
             r"([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b", True),
            (r"\brelaxation\s+(?:time|constant)s?" + _CONN + _SYM +
             _NUM0, False)):
        for cand in re.finditer(pat, t, re.I):
            win = t[max(0, cand.start() - 90): cand.end() + 40]
            if not _COUPLING_NEAR.search(win):
                continue
            if re.match(r"\s*(?:and|、|和)\s*[0-9]",
                        t[cand.end():cand.end() + 12]):
                continue
            return (cand.group(1), cand.group(2) if need_unit else "",
                    cand.group(0).strip())
    return None


def _extract_thermostat(t: str):
    hit = _variant_lookup(THERMOSTAT_CANON, t)
    if not hit:
        return None
    canon, frag = hit
    # 尝试抓耦合常数
    m = re.search(
        r"(tau[_ ]?t|tau|gamma|friction(?: coefficient)?)\s*[=：:]?\s*"
        r"([0-9]*\.?[0-9]+)\s*(ps|fs|ps\^-1|ps-1)?",
        t, re.I,
    )
    value: Dict[str, Any] = {"type": canon}
    conf = "high"
    if m:
        value["coupling"] = m.group(2)
        value["unit"] = m.group(3) or ""
        frag = t[max(0, m.start() - 20): m.end() + 5].strip()
    else:
        prose = _coupling_prose(t)
        if prose:
            value["coupling"], value["unit"], frag = prose
            if not value["unit"]:
                # 给了数、没给单位。
                #
                # 这里**曾经**打算把状态降级为 ambiguous（原注释就是这么写的），
                # 实测否决：全语料走到这一支的只有两格，标注**都是 present**，
                #   · cond-mat_9910397v1："t_coup^* = 4.66"，LJ 约化单位；
                #   · 2604.01940v3：Martini 单位下的 "relaxation time of 50"。
                # 两篇都在方法段别处定义了整套单位制，数本身是够用的，
                # 按 spec 对 ambiguous 的定义（"不足以据以行动"）这两格不成立。
                # 真照注释改成 ambiguous 得到的是 0 fixed / 2 broken：
                # 一格没修好，改坏两格。这是本项目"注释里的道理要在语料上过一遍"
                # 的又一个例子。
                #
                # 但 conf 仍降级、unit 仍是空串，因为不确定在**取值**层面而不在
                # 状态层面：unit 空意味着下游不能把这个数直接当 ps 用
                # （build_inputs 会把它写进输入文件）。两个层面分开表达，
                # 正是本项目反复强调的那条线。
                conf = "medium"
        else:
            conf = "medium"      # 有类型没常数
    return value, conf, frag


def _extract_barostat(t: str):
    hit = _variant_lookup(BAROSTAT_CANON, t)
    if not hit:
        return None
    canon, frag = hit
    m = re.search(r"tau[_ ]?p\s*[=：:]?\s*([0-9]*\.?[0-9]+)\s*(ps)?", t, re.I)
    value: Dict[str, Any] = {"type": canon}
    conf = "high"
    if m:
        value["coupling"] = m.group(1)
        value["unit"] = m.group(2) or ""
        frag = t[max(0, m.start() - 20): m.end() + 5].strip()
    else:
        prose = _coupling_prose(t)
        if prose:
            value["coupling"], value["unit"], frag = prose
            if not value["unit"]:
                conf = "medium"      # 只降 conf，状态仍 present，理由见恒温器那段
        else:
            conf = "medium"
    return value, conf, frag


_ENS_CN = {"等温等压": "NPT", "等温等容": "NVT", "微正则": "NVE"}

# 系综的**全称写法**。除缩写（NPT/NVT/NVE）与中文词外，文献也写统计力学的
# 标准名。原实现只认前两类，于是 "isothermal ensemble"、"microcanonical
# ensemble" 一字不匹配，整个系综被判成"未报告"（实测 2207.13405 就这么漏的，
# 标注为 present）。这与甲-1 的力场族名是同一类缺陷：**说法不同 ≠ 没写**。
#
# 三条纪律，每条都有理由：
#
#   1. 修饰语必须与字面 "ensemble" 相邻。单独出现的 "isothermal" 多是
#      "isothermal compressibility"（一个物理量，不是系综）；不要求紧邻，
#      就会把物性读成系综，那是取值错误，比漏检更难发现。
#
#   2. **只在缩写与中文词全落空时才启用**（与 _extract_force_field 查族名
#      子表同构）。这不只是保守：`detect_context` 直接调用本函数，
#      ctx["ensemble"] 又是恒压器闸门的唯一输入。让全称写法去和显式缩写
#      争"生产段优先"，就会顺带移动闸门，把恒压器从 not_applicable 翻出去。
#      作为补充分支，它的效果是"只把 None 变成值"，这是可审计的包含性。
#
#   3. grand canonical 是 µVT，不在本字段的 NPT/NVT/NVE 词表里，明示放弃
#      （canon 记 None，直接跳过），不能因为里面含 "canonical" 就报成 NVT。
#      "不知道"要报成不知道，不能报成最近的那个已知值。
_ENS_MOD = [
    ("isothermal-isobaric", "NPT"),
    ("constant pressure", "NPT"),
    ("microcanonical", "NVE"),
    ("constant energy", "NVE"),
    ("grand canonical", None),
    ("canonical", "NVT"),
    ("isothermal", "NVT"),
    ("constant temperature", "NVT"),
    ("constant volume", "NVT"),
]
_ENS_MOD_RE = re.compile(
    r"\b(" + "|".join(re.escape(p).replace(r"\ ", r"[\s-]") for p, _ in _ENS_MOD)
    + r")\s+ensembles?\b", re.I)
# 键按**归一化**后的形式存：文献写 "Isothermal-Isobaric"、"isothermal isobaric"
# 都该映射到同一条。不做归一化的话，带连字符的那几条会静默匹配不上，
# 静默漏掉比报错更糟，因为没有任何地方会提示你它漏了。
_ENS_MOD_CANON = {re.sub(r"[\s-]+", " ", p.lower()): c for p, c in _ENS_MOD}
_ENS_MICRO_RE = re.compile(
    r"\bmicrocanonical\s+(?:ensembles?|systems?|simulations?)\b", re.I)


def _extract_ensemble_general(t: str):
    """全称写法 -> 系综。只在缩写/中文词都没命中时作为补充调用。"""
    for m in _ENS_MOD_RE.finditer(t):
        canon = _ENS_MOD_CANON.get(re.sub(r"[\s-]+", " ", m.group(1).lower()))
        if canon:
            return canon, "high", m.group(0)
    # "microcanonical system"，短了一截的写法（少了 ensemble 一词）。
    # microcanonical 只有一个含义（NVE），不存在"哪个系综"的歧义，
    # 所以把 system / simulation 也算作系综声明是安全的。
    # **只给 microcanonical 开这个口**："isothermal system"、"NPT system"
    # 这类构式在描述体系性质时到处都是，宽开会把 missing 变成错误的 present。
    # 位置在循环**之后**：2207.13405 同时写了 microcanonical 与 isothermal，
    # 循环先命中，取值与 甲-4 时逐位一致；否则它的系综会被这次改动悄悄换掉
    # （present → present，账本看不见，正是最该防的那种改动）。
    # 全语料只有 2108.03424v1 落到这里（gold=present，工具报 missing）。
    m = _ENS_MICRO_RE.search(t)
    if m:
        return "NVE", "high", m.group(0)
    return None


def _extract_ensemble(t: str):
    """
    取"生产段"的系综，而不是文中第一个出现的系综。

    方法段通常成串描述流程：能量最小化 -> NVT 退火 -> NPT 平衡 -> 生产。
    取首个匹配会把退火段的 NVT 当成生产系综，进而把 barostat 判为
    "不适用"，而文献其实明确报告了恒压器。这个错误会连锁：系综错 ->
    恒压器被判 N/A -> 恒压器相关的一致性规则全部失效。
    """
    cn_hits = list(re.finditer(r"(等温等压|等温等容|微正则)", t))
    # 词表来自正则，但**文献里小写写法很常见**（"in the npt ensemble"）。
    # 少了 re.I 会让小写系综整个漏掉，而系综漏掉会连锁：
    # ctx["ensemble"] 为 None -> barostat 的适用性闸门拿不到值 ->
    # 恒压器连带整条规则链失效（见 detect_context 里那段注释）。
    en_hits = list(re.finditer(r"\b(NPT|NVT|NVE|NPAT|NPH)\b", t, re.I))
    hits = en_hits or cn_hits
    if not hits:
        # 缩写与中文词都没写，才轮到全称写法（见 _ENS_MOD 那段）。
        return _extract_ensemble_general(t)

    def canon(m):
        g = m.group(1)
        # ASCII 系综必须**规范化成大写**，不能原样吐出文献里的写法。
        # `_APPLICABILITY` 里 barostat 的闸门是 `ctx["ensemble"] == "NPT"`，
        # 于是把系综写成小写 "in the npt ensemble" 的文献会被判"不需要恒压器"，
        # 恒压器字段**直接消失成 not_applicable**，不报错、不留痕，
        # 相关的一致性规则（R-01/R-02）连带失效。
        #
        # 这与 `_extract_software` 少写 re.I 是同一类：**大小写差异伪装成
        # "作者没写"**。但它比字段层的漏检更隐蔽，上下文检测器
        # `detect_context` 不在 audit_case.py 的扫描范围内（那个脚本只遍历
        # PATTERNS），所以这一类缺陷用那把尺子**量不到**。
        # 实测：同一句话里 "npt" 小写给 not_applicable，"NPT" 大写给 present。
        return _ENS_CN.get(g, g.upper())

    # 优先取生产段附近的系综
    prod = re.search(r"production|生产|成品|成品段", t, re.I)
    if prod:
        after = [m for m in hits if m.start() >= prod.start()]
        if after:
            m = after[0]
            return canon(m), "high", m.group(0)
        before = [m for m in hits if m.end() <= prod.start()]
        if before:
            m = before[-1]
            return canon(m), "high", m.group(0)

    m = hits[0]
    return canon(m), "high", m.group(0)


def _extract_timestep(t: str):
    # 间隔曾写死为 \D{0,12}，于是 "time step was chosen to be 0.5 fs"
    # （关键词与数字相隔 18 个字符）直接漏掉。真实文献很少写
    # "a timestep of 2 fs" 这种紧邻形式。
    m = re.search(r"(?:time\s*step|timestep|dt|时间步)\D{0,30}([0-9]*\.?[0-9]+)\s*(fs|飞秒)", t, re.I)
    if not m:
        # 语序反过来："using t=0.2 fs as time step"，数字在前、关键词在后。
        m = re.search(
            r"([0-9]*\.?[0-9]+)\s*(?:fs|飞秒)\s*(?:\w+\s+){0,3}?"
            r"(?:time\s*step|timestep|时间步)", t, re.I)
    if m:
        return float(m.group(1)), "high", m.group(0).strip()

    # 无单位写法：LJ 单位、Martini 自然单位（"a time step size of t=0.001"）。
    # **绝不能**当成 fs 返回。0.001 在 LJ 单位下约合 2 fs，写成 "0.001 fs"
    # 差三个量级，而下游会把取值当真写进输入文件（build_inputs 直接把
    # timestep 写进 dt）。这正是本项目最防的**取值错误**：
    # 缺失会被看见，错值不会。所以只报 ambiguous，不给可被当真的数。
    m = re.search(
        r"(?:time\s*step|timestep|dt|时间步)[^.]{0,30}?[=＝]?\s*"
        r"([0-9]*\.?[0-9]+)(?!\s*(?:fs|飞秒))", t, re.I)
    if m:
        return float(m.group(1)), "low", m.group(0).strip(), "ambiguous"
    return None


def _extract_electrostatics(t: str):
    value: Dict[str, Any] = {}
    conf = "high"
    # PPPM 是 LAMMPS 对同一算法的叫法（particle-particle particle-mesh）。
    # 只认 "PME" 会漏掉所有 LAMMPS 论文，实测一篇写
    # "particle-particle particle-mesh (PPPM) algorithm" 的方法段被判"未报告静电处理"，
    # 而静电是 critical 字段，于是整篇评分被拉低。同一算法三个名字：
    # GROMACS/NAMD 叫 PME，LAMMPS 叫 PPPM，文献里也写 particle-mesh Ewald。
    if re.search(
        r"\bPME\b|\bP3M\b|\bPPPM\b|particle[- ]particle\s+particle[- ]mesh|"
        r"(?:smooth\s+)?particle[- ]mesh\s+ewald",
        t, re.I,
    ):
        value["method"] = "PME"
    elif re.search(r"reaction[- ]field|\bRF\b", t, re.I):
        value["method"] = "reaction-field"
    elif re.search(r"\bEwald\b", t, re.I):
        value["method"] = "Ewald"
    else:
        return None
    # 实空间截断
    m = re.search(
        r"(rcoulomb|r_?c(?:oulomb)?|cut[- ]?off|截断)\D{0,15}([0-9]*\.?[0-9]+)\s*(nm|å|angstrom)",
        t, re.I,
    )
    if m:
        value["rcoulomb_nm"] = float(m.group(2)) if m.group(3) == "nm" else round(float(m.group(2)) / 10.0, 3)
    else:
        conf = "medium"     # 有方法没截断
    return value, conf, m.group(0).strip() if m else "PME"


# "fully flexible" = 体系里**没有上键约束**。这是对"约束算法是什么"的直接
# 回答（答案是"没有"），而不是"没写"。
# 必须排在算法名**之后**：一篇若同时写了 SHAKE 与 fully flexible（前者约束水、
# 后者是描述势的柔性），算法名必须赢，顺序反了就是取值错误。
# 全语料只有 2110.08914v2 一处（"The nonreactive, fully flexible molecular
# potential for nitramines …"，该格 gold=present，工具报 missing）。
_FLEXIBLE_RE = re.compile(r"\bfully\s+flexible\b|\ball\s+bonds\s+(?:were\s+)?flexible\b",
                          re.I)


def _extract_constraints(t: str):
    hit = _variant_lookup(CONSTRAINT_CANON, t)
    if hit:
        canon, frag = hit
        return canon, "high", frag
    m = _FLEXIBLE_RE.search(t)
    if m:
        return m.group(0).strip(), "medium", m.group(0).strip()
    return None


def _extract_dispersion(t: str):
    hit = _variant_lookup(DISPERSION_CANON, t)
    if not hit:
        m = _DISP_CTX.search(t)
        if not m:
            return None
        # 长程校正的意译写法（"long-range correction to energy and pressure"）。
        # 归一化到 EnerPres 是有依据的：这句话描述的正是 GROMACS 里
        # DispCorr = EnerPres 在做的事（同时校正能量与压力）。
        return "EnerPres", "medium", m.group(0)
    canon, frag = hit
    return canon, "medium", frag


_LEN_UNIT = r"(ns|μs|us|ps)"

# 时长锚点词表。原实现要求时长附近**必须**出现 "production"，那是照着
# case-01 自己的写法写的。真实文献极少这么写："Each trajectory was 4 ns long"、
# "simulations of 2 ns duration" 全部漏判。症状同样是"字段缺失"，
# 使用者只会以为作者没写时长。
_LEN_ANCHOR = (r"(?:simulations?|trajector(?:y|ies)|runs?|production|"
               r"equilibrat\w*|sampling|模拟|轨迹|生产|平衡)")

# 与时长无关、但同样以时间单位出现的量。命中这些词附近的 "N ns/ps"
# 关联时间 / 寿命 / 阻尼常数，报成模拟时长就是一个
# **取值错误**（缺失看得见，错值看不见），所以宁可弃用该候选。
_LEN_DECOY = re.compile(
    r"correlat\w*|autocorrelat\w*|relax\w*|lifetime|residence|"
    r"decay|window|damping|\btau\b|关联|弛豫|寿命|"
    # 采样间隔 / 重复周期也不是模拟时长。
    # 实测："500 ps long ... trajectories, with velocity re-selections at
    # 100 ps intervals"，工具抓了 100 ps 当模拟时长，而真正的时长是 500 ps。
    # 这是**取值错误**（缺失看得见，错值看不见），必须挡住。
    r"interval|spacing|\bevery\b|frequency|间隔|每[0-9]|\bper\b", re.I)

# 最小化步数不是模拟时长。
_MIN_DECOY = re.compile(r"minimi[sz]\w*|最小化|能量优化", re.I)


def _len_ok(text: str, m) -> bool:
    """匹配点邻近若出现"另一种时间量"的词，弃用该候选。"""
    lo = max(0, m.start() - 45)
    hi = min(len(text), m.end() + 45)
    return not _LEN_DECOY.search(text[lo:hi])


# 动力学动词 + "for <时长>"：把"演化了多久"这种**叙述式**的时长也认下来。
#   "we evolved the system for another 20 ns in the NVE ensemble"（1805.00269v1）
#   "we evolve the system in the microcanonical ensemble for 5 ns"（2207.13405）
# 上面 pats 里的四条都要求 production / 锚点名词 / duration|long|period，
# 这两句一个都不占，时长是作为**动作的持续时间**说出来的。
#
# 为什么锚在动词上，而不是写成通用的 `for <n> <unit>`：后者会命中
#   "the HCl evolution on the (11 2 0)-(a) surface continues for 10 ps"
#   （1812.03256v1）
# 那是**化学过程的时长**（该表面模型上 HCl 持续析出多久），不是模拟时长；
# 报成模拟时长是取值错误。实测通用版 2 fixed / 0 broken / **1 neutral**，
# 收窄到动词版后 2 fixed / 0 broken / 0 neutral。
# `evolv(?:e|ed|es|ing)` 刻意不写成 `evolv\w*`，后者会把 "evolution"
# 这个**名词**当成动力学动词，1812 那句正是栽在这里。
# 窗口 60 是量出来的：40 够不到 2207（动词与 "for" 之间隔了
# "the system in the microcanonical ensemble"），60 与 80 结果逐格相同，
# 取小的。
_LEN_VERB = (r"(?:evolv(?:e|ed|es|ing)|propagat(?:e|ed|es|ing)|"
             r"run|ran|running|perform(?:ed|ing)?|carry\s+out|"
             r"simulat(?:e|ed|es|ing)|integrat(?:e|ed|es|ing))")
_LEN_RUN_RE = re.compile(
    r"\b" + _LEN_VERB + r"\b[^.]{0,60}?\bfor\s+"
    r"(?:another\s+|a\s+further\s+|an\s+additional\s+|at\s+least\s+)?"
    r"([0-9]*\.?[0-9]+)\s*" + _LEN_UNIT + r"\b", re.I)


def _attach_replicas(value: Dict[str, Any], t: str) -> None:
    """把重复次数并进 simulation_length 的取值（找不到就不并）。

    重复次数。两个坑：一是 "repeat" 会命中 "repeat units"（体系构成，
    不是重复次数），二是 "3 runs of 20 ns" 这类写法里数字在 runs 之前。
    因此只认语义明确的词，并显式排除 repeat unit / repeating unit。

    抽成函数是因为**两条取值路径都要用它**：主路径（pats）与下面的叙述式
    分支。叙述式分支漏掉这一步时，2207.13405 的 "3 independent runs"
    就不进取值，R-08（报了时长却没报重复次数）随之误触发，同一句话
    走哪条路得到不同的取值，是缺陷，不是风格。
    """
    # `independent simulations` 是**必需**的：裸 `simulations?` 会命中
    # "4x 10^6 simulation steps are performed"（1007.5359v2），那是步数，
    # 加进去会给该篇安上 replicas=6，是取值错误。实测全语料只有
    # 1805.00269v1 的 "50 independent simulations for each material"
    # 命中这个必需形式，而 R-08 正是拿这一条对它误报"未说明重复次数"。
    rep = re.search(
        r"([0-9]+)\s*(?:independent\s+|separate\s+|different\s+)?"
        r"(?:replicas?|independent\s+runs?|independent\s+simulations?|"
        r"repeats?|trajectories|"
        r"条(?:独立)?(?:轨迹|重复)|个独立(?:轨迹|重复))"
        r"(?!\s*(?:unit|单元))",
        t, re.I,
    )
    if rep and not re.search(r"repeat(?:ing)?\s+units?", rep.group(0), re.I):
        value["replicas"] = int(rep.group(1))


def _extract_simulation_length(t: str):
    pats = [
        # 1) production 附近（原路径，保留）
        r"([0-9]*\.?[0-9]+)\s*%s\b[^.]{0,40}?(?:production|生产|成品)" % _LEN_UNIT,
        r"(?:production|生产|成品)[^.]{0,40}?([0-9]*\.?[0-9]+)\s*%s" % _LEN_UNIT,
        # 2) 锚点词 + 时长（"simulations of 2 ns"）。
        # 窗口从 40 收到 20：窗口开得越大，"trajectories, with velocity
        # re-selections at 100 ps" 这种"锚点词在很远的地方、时长属于另一件事"
        # 的句子越容易被误抓。
        r"%s[^.]{0,20}?([0-9]*\.?[0-9]+)\s*%s\b" % (_LEN_ANCHOR, _LEN_UNIT),
        # 3) 时长 + 说明其身份的词（"4 ns long"、"2 ns duration"）。
        # 边界必须有：否则 "prolonged" 里的 "long" 会被当成身份词。
        r"([0-9]*\.?[0-9]+)\s*%s\b[^.]{0,25}?\b(?:duration|long|period)\b" % _LEN_UNIT,
        r"duration\s+of\s+([0-9]*\.?[0-9]+)\s*%s\b" % _LEN_UNIT,
    ]
    m = None
    for pat in pats:
        for mm in re.finditer(pat, t, re.I | re.S):
            if _len_ok(t, mm):
                m = mm
                break
        if m:
            break

    if not m:
        # 叙述式时长："we evolved the system for another 20 ns"。
        # 排在步数分支**之前**：真给出时长时，时长比步数有用
        # （步数在 LJ / Martini 自然单位下折不成 fs，见下）。
        for mm in _LEN_RUN_RE.finditer(t):
            if not _len_ok(t, mm):
                continue
            val: Dict[str, Any] = {
                "length": float(mm.group(1)),
                "unit": mm.group(2).lower().replace("μ", "u")}
            _attach_replicas(val, t)
            return val, "medium", mm.group(0).strip()

    if not m:
        # 只有步数、没有时长单位（"10^8 time steps"）。
        # 这是**可直接使用**的信息：build_inputs 要的正是 nsteps。
        # 但物理时长取决于单位制（LJ / Martini 的自然单位无法折算成 fs），
        # 所以只给步数，**不**折算成 ns，折算就是在编数。
        for pat, conv in (
            (r"([0-9]+(?:\.[0-9]+)?)\s*[x×]\s*10\s*\^\s*([0-9]+)"
             r"\s*(?:time\s+|simulation\s+)?steps?\b",
             lambda g: int(float(g[0]) * 10 ** int(g[1]))),
            (r"10\s*\^\s*([0-9]+)\s*(?:time\s+|simulation\s+)?steps?\b",
             lambda g: 10 ** int(g[0])),
            (r"([0-9][0-9,]{3,})\s*(?:time\s+|simulation\s+)?steps?\b",
             lambda g: int(g[0].replace(",", ""))),
        ):
            sm = re.search(pat, t, re.I)
            if sm and not _MIN_DECOY.search(
                    t[max(0, sm.start() - 30): sm.end() + 30]):
                return ({"steps": conv(sm.groups())}, "medium",
                        sm.group(0).strip())
        return None

    length = float(m.group(1))
    unit = m.group(2).lower().replace("μ", "u")
    value: Dict[str, Any] = {"length": length, "unit": unit}
    _attach_replicas(value, t)
    return value, "high", m.group(0).strip()[:80]


# 原子数必须**归属于被模拟的体系**，而裸模式 "<数字> atoms" 做不到这一点。
# 那句话里的"原子数"至少还指着三样别的东西，本语料实测到两处：
#
#   2207.13405    "914 structures with 54840 atoms in total"
#                 → NEP **训练集**的规模；被模拟的硅烯全文没给原子数
#   2602.17595v1  "...SMILES representation ... each polymer has around
#                  200 atoms" → 单条链的**化学表示**，不是模拟胞
#
# 这两处的症状与"静默失效"正相反：字段不缺失，而是报出一个 **high 置信度的
# 错值**。检查单上看起来"这篇给了原子数"，读者据此横向比较各文献的体系规模，
# 而那个数根本不是体系规模。缺失会被看见，错值不会——这是本工具反复回到的
# 那条线，抽取器自己也不能例外。
#
# 表里只收**本语料有实例**的语境。没有实例的猜测不收：这工具在别处已经因为
# "替作者猜一个值"吃过亏（见 build_inputs 的 TODO 分支），这里同理。
_SIZE_OTHER_REFERENT = (
    (r"train\w*\s+(?:data|set|structures?|configurations?|samples?|frames?)",
     "训练集/验证集的规模"),
    (r"data\s?sets?|databases?|数据集|训练集|数据库",
     "数据集规模"),
    # "914 structures with 54840 atoms"：数字**紧跟在条目数之后**，
    # 说明它描述的是数据集里的条目，不是体系。
    # 前面那个 [0-9] 不能省：晶体学里 "6x6x6 conventional cubic cells with
    # a total of N=1728 atoms" 的 cells 是**真体系**（1805.00269v1），
    # 没有条目数领着的 "cells/structures with" 一律不算。
    (r"[0-9][0-9,]*\s+(?:structures?|configurations?|frames?|snapshots?|"
     r"samples?|molecules?|chains?)\s+with\s+(?:a\s+total\s+of\s+)?$",
     "「N 个结构，共 M 个原子」：M 是数据集条目的属性"),
    (r"\bSMILES\b|\brepresentation\b|\bencoded\b|\bfingerprints?\b|"
     r"\bdescriptors?\b",
     "化学表示/描述符的规模"),
    (r"\beach\s+(?:chain|polymer|molecule|monomer|repeat\s+unit|CRU)\b",
     "单条链、单个分子的规模"),
)


def _sentence_head(t: str, pos: int, back: int = 240) -> str:
    """取 pos 之前、同一句内的文本，供否决模式匹配。

    只取数字**之前**的部分：表里的模式都是"这个数属于 XXX"的引导语，
    落在数字之后的，多半已经在说下一件事了。

    两种切错的代价不对称，所以 windows 宁可**切长**：
    切短了会把该否决的漏过去（→ 错值），切长了会多收一个否决（→ 缺失）。
    因此句末标点只作首选边界，另有下限 60 字符兜底——句末标点本身会假切
    （"i.e." / "e.g." / "Fig."），而"紧贴在数字前的引导语"必须落在窗口里。
    """
    start = max(0, pos - back)
    cut = max(t.rfind(c, start, pos) for c in ".;。；\n")
    beg = cut if cut >= 0 else start
    return t[max(start, min(beg, pos - 60)):pos]


def _size_count_other_referent(t: str, pos: int) -> str:
    """原子数若被上下文指派给了别的对象，返回那个对象；否则返回空串。"""
    head = _sentence_head(t, pos)
    for pat, why in _SIZE_OTHER_REFERENT:
        if re.search(pat, head, re.I):
            return why
    return ""


def _extract_system_size(t: str):
    value: Dict[str, Any] = {}
    # 逐个候选往下找：被否决的那个不算数，但**后面的候选仍要试**。
    # 只检查第一个命中就放弃，会把"开头提了训练集、后文才给体系"的论文
    # 整篇判成缺失——那是把修错值换成了造缺失，不是修好了。
    m = None
    for cand in re.finditer(r"([0-9][0-9,]{2,})\s*(?:atoms|particles|原子)", t, re.I):
        if _size_count_other_referent(t, cand.start()):
            continue
        m = cand
        break
    if m:
        value["atoms"] = int(m.group(1).replace(",", ""))
    # 两种语序都要收："8.0 nm cubic box" 与 "cubic box of 8.0 nm"
    b = re.search(r"([0-9]*\.?[0-9]+)\s*(nm|å|angstrom)[^.]{0,25}(?:box|cubic|cell|盒子)", t, re.I)
    if not b:
        b = re.search(r"(?:box|cubic|cell|盒子)[^.]{0,25}?([0-9]*\.?[0-9]+)\s*(nm|å|angstrom)", t, re.I)
    if not b:
        # 只报尺寸、不写 "box/cell" 的写法："the length of the system is 123 A"。
        # 要求同时出现"尺寸词 + of the system/box/cell"才认，单靠一个
        # 裸 "A" 太容易撞上别的含义（安培、编号、单位标签），
        # 而这里判错就是**取值错误**，宁可漏。
        b = re.search(
            r"(?:length|width|height|thickness|dimension|长|宽|高)[^.]{0,15}?"
            r"(?:of\s+the\s+)?(?:system|box|cell|simulation\s+cell)[^.]{0,20}?"
            r"([0-9]*\.?[0-9]+)\s*(nm|å|angstrom|a)\b", t, re.I)
    if not b and not m:
        # 只报尺寸、不写 "box/cell" 的写法之三：符号赋值。
        # "W=13.3 nm"、"L_y = 6 nm"、"L = 7.7, 15.4, ... nm"。
        # 二维材料（硅烯、石墨烯）几乎只写这一种——它们的"盒子"就是
        # 宽度 × 输运方向长度，本来也没有立方盒子可写，
        # 上面三条要求 box/cubic/cell 的模式在它们身上全部落空。
        # 要求**等号**：没有等号就没有"这是尺寸"的断言，裸的 "500 nm"
        # 在正文里可能是波长、是厚度、是截断半径。
        #
        # `not m` 这个条件是判据，不是排序：本分支存在的**唯一目的**是喂饱
        # 下面"只给盒子尺寸不给原子数 → ambiguous"那条 schema 规则。
        # 已经有原子数时再去别处凑一个尺寸，凑来的那个多半属于**另一个
        # 子体系**（1805.00269v1 的原子数是体硅的 1728，而 nm 尺寸是硅烯的
        # NEMD 宽度），把两份指称不同的数拼进同一个 dict，正是这次要修的
        # 那类错误——取值来自错误的指称对象。
        b = re.search(
            # 数字子模式写成 [0-9]+(?:\.[0-9]+)? 而不是 [0-9][0-9.,]*：
            # 后者会把列表里的分隔逗号吃进数字（"7.7," -> float() 直接抛），
            # 前者在 "7.7, 15.4" 上停在 7.7。列表整体收下，
            # 但**取值给第一个**——它就是要找的那个尺度。
            r"\b[WL](?:_?[a-z])?\s*=\s*([0-9]+(?:\.[0-9]+)?)"
            r"(?:\s*,?\s*(?:and\s+)?[0-9]+(?:\.[0-9]+)?)*\s*(nm|å|angstrom)\b",
            t, re.I)
    if b:
        value["box_nm"] = float(b.group(1)) if b.group(2).lower() == "nm" else round(float(b.group(1)) / 10.0, 3)
    if not value:
        return None
    span = m.group(0) if m else b.group(0)
    if "atoms" not in value:
        # 只给盒子尺寸不给原子数。schema 对这一条的原文是
        # "只给盒子尺寸不给原子数，属 ambiguity"，尺寸给了，但
        # 有限尺寸外推要的是粒子数，缺了它这组数据不足以据以行动。
        # 原实现返回 present + medium，与 schema 不符。
        return value, "medium", span, "ambiguous"
    return value, "high", span


def _extract_heat_flux(t: str):
    hit = _variant_lookup(HEATFLUX_CANON, t)
    if hit:
        canon, frag = hit
        return canon, "high", frag
    m = _HEATFLUX_NEMD_RE.search(t)
    if m:
        return "Direct NEMD", "medium", m.group(0).strip()
    return None


def _extract_temperature_bath(t: str):
    """
    冷热浴温度。

    不能简单地取全文温度的 max/min，方法段里还会出现别的温度：
    退火温区、ΔT 线性度扫描点、实验对照温度。取极值会把"扫描用的小 ΔT"
    当成冷浴，得到一个看似合理、实则完全错误的 ΔT，进而污染
    R-11 线性响应判据。这类错误比"字段缺失"危险得多：
    缺失会被看见，错值不会。

    策略：只在含 hot/cold 语义的句子里取温度，并要求两个温度同句出现。
    """
    # "Delta T = 30 K" 里的 30 K 是温差本身，不是任何一个浴的温度。
    # 不排除它就会把 ΔT 当成冷浴温度。
    dt_ctx = re.compile(r"(?:delta\s*T|ΔT|温差|temperature\s+difference)\s*[=≈:]?\s*$", re.I)

    def temps_in(s: str) -> List[Tuple[int, int]]:
        """返回 [(温度值, 位置)]，已剔除 ΔT 自身的数值。"""
        out: List[Tuple[int, int]] = []
        # re.I 是必需的：小写 "300 k" 同样常见。带 \b 所以不会误吃
        # "500 km"、"100 kPa"（k 后面还有字母，不构成词边界），
        # 也不会吃 "k_B"（下划线是词字符）。这两个反例都实测过。
        for m in re.finditer(r"([0-9]{2,4})\s*K\b", s, re.I):
            if dt_ctx.search(s[max(0, m.start() - 16):m.start()]):
                continue
            out.append((int(m.group(1)), m.start()))
        return out

    # 按句切分（同时处理中英文句号）。
    # 注意 "i.e."、"e.g." 会造成假切分，因此切完还要按"含 hot/cold 双关键词"
    # 重新聚合，只要一个片段同时含两个关键词就够用，不强求整句完整。
    sentences = re.split(r"(?<=[.;。；])\s+", t)
    # 源汇写法：NEMD 文献里"热端/冷端"有一半写成 source/sink 或
    # higher/lower temperature，原词表只认 hot/cold 字面。
    # **实测放宽后新触达的句子全语料只有 3 句 / 2 篇**：
    #   · 1805.00269v1 "coupling a source region … with a higher temperature of
    #     330 K and a sink region … with a lower temperature of 270 K"
    #     两个温度都是数字、同句、差 60 K，只差"hot/cold"这两个词。
    #   · 2207.13405 两句的温度是符号式（T±ΔT/2，`\pm` 在 LaTeX→文本时被吃掉，
    #     只剩 "T+ T/2"），temps_in 抓不到，落到下面的 ΔT 分支。
    hot_kw = re.compile(
        r"\bhot\b|热浴|高温浴|热端|\bT_?hot\b|heated|"
        r"\bheat\s+source\b|\bsource\s+(?:region|reservoir)\b|"
        r"higher\s+temperature", re.I)
    cold_kw = re.compile(
        r"\bcold\b|冷浴|低温浴|冷端|\bT_?cold\b|cooled|"
        r"\bheat\s+sink\b|\bsink\s+(?:region|reservoir)\b|"
        r"lower\s+temperature", re.I)

    for s in sentences:
        mh, mc = hot_kw.search(s), cold_kw.search(s)
        if not (mh and mc):
            continue
        ts = temps_in(s)
        if len(ts) < 2:
            continue
        # 次序配对：文献的标准写法是"热端...X K，冷端...Y K"，
        # 关键词的先后与温度的先后一致。按关键词顺序取前两个温度，
        # 比取 max/min 稳健得多，max/min 会被句子里其它温度带偏。
        ordered = sorted(ts, key=lambda x: x[1])[:2]
        if mh.start() <= mc.start():
            hot, cold = ordered[0][0], ordered[1][0]
        else:
            cold, hot = ordered[0][0], ordered[1][0]
        if hot - cold < 20:
            continue
        return ({"hot_K": hot, "cold_K": cold, "delta_K": hot - cold},
                "high", s.strip()[:120])

    # 回退：整段里找梯度语义，且明确出现"a K 和 b K"的并列结构
    seg = t
    if re.search(r"gradient|Delta\s*T|ΔT|温差|温度梯度", seg, re.I):
        pair = re.search(
            r"[^.]{0,60}?([0-9]{2,4})\s*K\s*(?:and|和|,)\s*([0-9]{2,4})\s*K",
            seg, re.I)
        if pair:
            a, b = int(pair.group(1)), int(pair.group(2))
            hot, cold = max(a, b), min(a, b)
            if hot - cold >= 20:
                return ({"hot_K": hot, "cold_K": cold, "delta_K": hot - cold},
                        "medium", pair.group(0).strip()[:120])

    # 只有 ΔT、没有两个绝对浴温：浴温写成 T±ΔT/2 这类符号式，而符号在
    # LaTeX→文本时被吃掉（2207.13405 只剩 "T+ T/2" / "T- T/2"），全文
    # 唯一带单位的数是 "a temperature difference of T=20 K"。
    # spec 这个字段的名字是"冷热浴**温度与梯度**"，"为什么重要"写的是
    # 「ΔT 过大将进入非线性区……只给方法不给温度设置，无法判断是否在线性区」
    # ΔT 本身正是判据要的量，有明确数值就该报。
    # 浴温是符号式，**不替它编一个数**（缺失会被看见，错值不会）。
    # R-11 要 hot_K 与 cold_K 同时在场才触发，只报 delta_K 不会误报线性区。
    # 实测 "temperature difference" 这个字面全语料 5 处，只有这一处满足
    # "of <数字> K" 的形状（另 4 处在 1208、2108，都是"between the hot and
    # cold slabs"这类描述，没有数值）。
    if hot_kw.search(seg) and cold_kw.search(seg):
        d = re.search(r"temperature\s+difference\s+of\s+\S{0,6}?([0-9]{2,4})\s*K",
                      seg, re.I)
        if d:
            return ({"delta_K": int(d.group(1))}, "medium",
                    d.group(0).strip()[:120])
    return None


def _extract_correlation_window(t: str):
    m = re.search(
        r"(correlation\s+time|correlation\s+window|关联时间|统计窗口)\D{0,15}"
        r"([0-9]*\.?[0-9]+)\s*(ps|ns)",
        t, re.I,
    )
    if not m:
        m = re.search(r"([0-9]+)\s*(?:independent\s+)?(?:windows|blocks|窗口)\b", t, re.I)
        if m:
            return {"n_windows": int(m.group(1))}, "medium", m.group(0)
        return None
    return {"correlation_time": float(m.group(2)), "unit": m.group(3)}, "high", m.group(0)


_SW_NAMES = (r"GROMACS|AMBER|AmberTools|NAMD|LAMMPS|OpenMM|CHARMM|Desmond|HOOMD|"
             r"GROMOS|Tinker|ESPResSo|"
             r"Materials\s+Studio|Discovery\s+Studio|Forcite|DL_POLY|CP2K")
_SW_VER = r"[0-9]{4}(?:\.[0-9]+)*|[0-9]+\.[0-9]+(?:\.[0-9]+)*"


def _extract_software(t: str):
    """
    软件 + 版本号。版本号是复现的硬约束：
    同一力场在 GROMACS 2016 与 2022 上可能因默认参数变更给出不同结果。
    """
    # 必须带 re.I。原先靠手写大小写变体（列表里同时列出 "LAMMPS" 与
    # "Gromacs|gromacs"）来兜底，结果是**只有被手动补过的那几个**能识别：
    # 实测 "we used openmm 8.0" 与 "namd 2.14 was employed" 都返回 None，
    # 小写的 "lammps" 同样漏，字段被报成"未报告软件"，而作者明明写了。
    # 漏的是哪些软件取决于谁当年被补进过列表，这种漏检从输出上看不出来。
    m = re.search(
        r"\b(" + _SW_NAMES + r")\b\s*[\(\[]?\s*"
        r"(?:version|release|v|/)?\s*(" + _SW_VER + r")",
        t, re.I,
    )
    if m:
        return f"{m.group(1)} {m.group(2)}", "high", m.group(0).strip()
    # 版本信息在名字**前面**："The most recent (October 29, 2020) version of LAMMPS"
    m = re.search(
        r"[\(\[]([^)\]]{3,40})[\)\]]\s*version\s+of\s+(" + _SW_NAMES + r")\b",
        t, re.I,
    )
    if m:
        return f"{m.group(2)} {m.group(1)}", "high", m.group(0).strip()
    m = re.search(r"\b(" + _SW_NAMES + r")\b", t, re.I)
    if not m:
        return None
    name = m.group(1)
    # 只写软件不写版本。schema 对这一条的原文是"属于**看起来写了其实没写**"
    # 即"提到了但不足据以行动"，按四态定义正是 ambiguous，
    # 不是 present。原实现返回 present + 低置信度，等于把"说了但没说清"
    # 折价成"说了"；前 12 篇真实方法段上有 3 篇栽在这里
    # （"using LAMMPS interfaced with DeePMD-kit"，两个软件都没版本）。
    return f"{name}（未给版本）", "low", m.group(0).strip(), "ambiguous"


# spec 对 initial_structure 的原文是「来源（晶体结构 / 预测 / 自建）与缺失残基、
# 缺失原子的处理方式必须写明」。下面的模式只补**来源**这一类，分三种直述方式，
# 每种对应语料里一个真实句子：
#
#   (a) 起点式   "We start with a bulk α-quartz crystalline substrate"（0706.3514v1）
#                "we always start from crystal slab structures"（1002.0354v1）
#   (b) 取用式   "The reference structures were taken from previous publications"
#                （2404.14338v1）
#   (c) 自建式   "which is obtained by joining the two ends of the nanoribbon"
#                "MGS can be constructed in such a way as to half-twist…"（1007.5359v2）
#
# 三条一起实测 4 fixed / 0 broken / 0 neutral（恰为上面四格）。
#
# 两个**试过并收窄掉**的宽版本，各有实测代价：
#   · (b) 若把 snapshot / coordinate 也当名词，"the snapshots were taken when
#     the solute first reached the desired distance"（1707.05453v1，伞形采样取帧）
#     会被读成结构来源，该格 gold=ambiguous，变 neutral；
#   · (c) 若不要求系动词，"an ensemble of configurations … **as** generated by
#     the WT-metaD" 同样误命中（1707.05453v1）。要求 `is/are/was/were/be` 紧邻，
#     留下的就都是"这个几何是怎么拼出来的"这类谓语句。
# 名词表里没有 system / model / box，"Starting from an equilibrated system"
# （2108.03424v1，gold=ambiguous：怎么制备、以什么能量密度平衡都没写）不是
# "结构来源"，而"从某个结构出发"才是。
_INIT_START_RE = re.compile(
    r"\b(?:we\s+)?(?:start|begin|initiali[sz]e)\w*\s+(?:with|from)\s+"
    r"(?:an?\s+|the\s+)?(?:[\w\-]+\s+){0,3}"
    r"(structures?|configurations?|geometr\w+|cells?|"
    r"supercells?|slabs?|substrates?|crystals?|lattice|nanoribbons?|"
    r"nanotubes?|chains?|sheets?|grains?)\b",
    re.I)
_INIT_FROM_RE = re.compile(
    r"\b(structures?|configurations?|geometr(?:y|ies)|cells?|supercells?|"
    r"slabs?|substrates?|crystals?|lattices?|coordinates?)\b"
    r"[^.]{0,30}?\b(?:were|was|are|is)\s+"
    r"(?:taken|obtained|drawn|generated|built|constructed|created|prepared)\b",
    re.I)
_INIT_BUILD_RE = re.compile(
    r"\b(?:is|are|was|were|be)\s+"
    r"(?:obtained|constructed|built|created|formed|generated|made)\s+"
    r"(?:by|in\s+such\s+a\s+way\s+as\s+to|through)\b", re.I)


def _initial_structure_narrative(t: str):
    """(a)/(b)/(c) 三种来源叙述。原模式全落空后调用。"""
    for pat in (_INIT_START_RE, _INIT_FROM_RE, _INIT_BUILD_RE):
        m = pat.search(t)
        if m:
            frag = m.group(0).strip()
            return frag, "medium", frag
    return None


def _extract_initial_structure(t: str):
    """
    初始结构来源。可正则化的常见情形：
      Packmol 自建 / CHARMM-GUI / 实验晶体(PDB) / 商业软件建模 / 上一段轨迹续跑
    """
    # `t?leap` 曾把 "temperature **leaps** at the boundary" 里的 leaps
    # 读成 Amber 的建模工具 LEaP，于是那篇论文的初始结构被报成
    # "tleap 生成"。与 `opc` 命中 `dopc` 是同一类：**取值错误**。
    # LEaP 用 (?-i:) 限定大小写，只有真写成那个驼峰名才算。
    m = re.search(
        r"(Packmol|CHARMM-GUI|charmm-gui|AmberTools|tleap|xleap|(?-i:LEaP)|Avogadro|"
        r"Materials\s+Studio|"
        r"Protein\s+Data\s+Bank|PDB\s+(?:ID|entry|code|file)|"
        r"crystallographic\s+structure|X-?ray\s+structure|NMR\s+structure|"
        r"crystal\s+structure|experimental\s+structure|"
        r"晶体结构|实验结构|从头建模|自建|同源建模|homology\s+model)",
        t, re.I,
    )
    if not m:
        return _initial_structure_narrative(t)
    return m.group(0).strip(), "medium", m.group(0).strip()


def _extract_protonation(t: str):
    """质子化态：pKa 工具、显式 tautomer 命名、恒定 pH 方法都可判。"""
    m = re.search(
        r"(PROPKA|H\+\+|Hpp|PDB2PQR|pKa\s+(?:calculation|prediction|values?)|"
        r"\b(HID|HIE|HIP|HSP|HSD|HSE)\b|"
        r"protonation\s+state|protonation\s+states|tautomer|"
        # 端基的带电/中性写法。"charged termini (NH_3^+ and COO^-)" 与
        # "neutral termini (NH_2 and COOH)" 是**直接**说明了可滴定基团的
        # 质子化态，spec 该字段的取值示例正是这一类。
        # 全语料只命中 1707.05453v1（gold=present，工具报 missing）。
        r"charged\s+termini|neutral\s+termini|\bzwitterionic\b|"
        r"constant[- ]pH|恒定\s*pH|质子化态|质子化状态)",
        t, re.I,
    )
    if not m:
        return None
    # 只提"考虑了质子化态"但不说依据，信息量不足
    weak = re.fullmatch(r"protonation\s+states?|质子化态|质子化状态", m.group(0).strip(), re.I)
    return (m.group(0).strip(), "low" if weak else "high", m.group(0).strip())


def _extract_ligand_params(t: str):
    """配体参数化：电荷方法（AM1-BCC/RESP）+ 力场（GAFF/GAFF2/CGenFF）。"""
    # 短模式一律加词边界。`RESP` 不加 `\b` 会命中 "respectively"、"corresponding"
    # 的子串，还会把 r-RESPA 积分器读成 RESP 电荷，实测让 1002.0354v1、
    # 2404.14338v1、2604.01940v3 三篇各多报一条配体参数化。
    # 末一个分支里的 `RESP` 刻意不加：它前面是 `电荷`（`\w`），加了 `\b` 反而失配。
    charge = re.search(r"AM1[- ]BCC|\bRESP(?:2)?\b|ESP\s+charges?|Mulliken|"
                       r"HF/6-31G\*|电荷.{0,8}(?:AM1|RESP|拟合)", t, re.I)
    # 电荷**缩放**：离子液体 / 有机电解液文献的标配做法
    # （"the ion charges were scaled with a factor of 0.75"）。它是电荷处理的
    # 决定本身，与 AM1-BCC/RESP 属同一类信息，只是给的是缩放因子，
    # 而不是电荷来源。全语料只出现 1 处（2602.17595v1，gold=present）。
    charge_label = charge.group(0).strip() if charge else None
    if not charge_label:
        m = re.search(r"charges?\s+(?:were\s+|are\s+|was\s+)?scal\w*"
                      r"[^.]{0,40}?([0-9]*\.?[0-9]+)", t, re.I)
        if m:
            charge_label = f"charge scaling {m.group(1)}"
    ff = re.search(r"\bGAFF2?\b|\bCGenFF\b|\bOpenFF\b|SMIRNOFF|\bOPLS[- ]?AA\b|"
                   r"general\s+amber\s+force\s+field|通用力场", t, re.I)
    if not (charge_label or ff):
        return None
    parts = []
    if ff:
        parts.append(ff.group(0).strip())
    if charge_label:
        parts.append(charge_label)
    span = " + ".join(parts)
    if not (charge_label and ff):
        # 只给了力场没给电荷方法（或反之）。schema 对这一条的原文是
        # "写'配体用 GAFF 处理'而未写电荷方法，属 ambiguity"，
        # 电荷来源（AM1-BCC vs RESP）直接决定小分子水合自由能，只给力场名
        # 不足以据以行动。原实现返回 present + medium，与 schema 不符。
        # 注意判据用的是 charge_label 而不是 charge：电荷缩放同样是"电荷怎么
        # 定的"，只认来源正则会让 2602.17595v1（OPLS-AA + 缩放 0.75）被判成
        # ambiguous，而它把两件事都写清楚了。
        return span, "medium", span, "ambiguous"
    return span, "high", span


# -----------------------------------------------------------------------------
# 势函数族 → "这个势的定义里存在什么"的能力表
# -----------------------------------------------------------------------------
# FF_CANON 认得的是**名字**（AMBER99SB、CHARMM36m、GAFF2…），清一色是
# 生物分子/聚合物力场，外加 ReaxFF、Martini、DREIDING、UFF、PCFF、COMPASS。
# 共价多体势与机器学习势一个都没有。后果有两层：
#
#   ①  `force_field` 抽不出来。17 篇"非经典势"论文里抽取器只认出 3 篇，
#       Tersoff / Brenner / Stillinger-Weber / NEP / DeePMD 全部漏检；
#       `force_field` 也因此是真实语料上 present→missing 最多的字段。
#   ②  更要紧的一层：这些势的解析形式里**不存在**若干对象，而工具不知道，
#       于是照问"你用了哪种静电方法""做了色散校正吗"。在一个没有那件
#       东西的体系里问，得到的 missing 是**假阳性**（实测 40 条）。
#
# 每一行给出该族的三项能力，问的是"势函数的定义里有没有这个东西"，
# 不是"作者写没写"：
#   bonds       有无**固定**键表，决定 SHAKE/LINCS/SETTLE 有没有作用对象
#   coulomb     有无点电荷库仑项，决定 Ewald/PME/reaction-field 有无对象
#   dispersion  有无可截断的 1/r^6 吸引项，决定长程色散校正有无对象
#
# **判据全部来自势函数的原始定义，不来自本项目语料。** 语料只告诉我们
# "该给哪几个字段装闸门"，装不装得上由这些定义决定。这条界线必须守住，
# 从案例反推规则就是拿测试集调参，而本项目的核心主张正是"不这么做"。
#
# 匹配纪律比 FF_CANON 更严：**只收拼写完整、或不会撞车的缩写。**
# 裸 `gap` / `snap` / `comb` / `sw` 一律不收（band gap、snap、comb 都会
# 命中普通词）。这类坑本项目已经栽过两次：`his` 命中 "this"（24 篇里
# 20 篇误报"含蛋白"）、`opc` 命中 "dopc"（膜论文报出水模型 OPC）。
# 误命中的代价在这里特别不对称：多翻一个 NA，就把一个真实的缺失静静压掉了，
# 而且看不出来，静默失效正是本项目要抓的东西，不能自己再造一个。
FF_FAMILIES = [
    {
        "name": "键序/多体势",
        # Tersoff PRB 37, 6991 (1988)；Stillinger-Weber PRB 31, 5262 (1985)；
        # Brenner PRB 42, 9458 (1990) 及 J. Phys. Condens. Matter 14, 783 (2002)；
        # EDIP、MEAM 同属按局域配位构造的短程多体势。
        "pattern": r"tersoff|brenner|\brebo\b|airebo|\blcbop\b|"
                   r"stillinger[- ]weber|\bedip\b|\bmeam\b|erhart[- ]albe",
        # 报告侧：本族里已接上队的名词。pattern 认得 ≠ 报得出来，
        # 两边不同步正是本轮要修的结构性问题，故在此显式指向同一张表。
        "names": _FF_NAMES_BONDORDER,
        "bonds": False, "coulomb": False, "dispersion": False,
        "why": "能量由局域配位（键级/三体项）构造，形式中不含点电荷项与 "
               "1/r^6 吸引项；键随配位变化而非固定拓扑",
    },
    {
        "name": "机器学习势",
        # GAP、SNAP、MTP、NEP、DeePMD/DeePPM 等：以局域环境描述符回归参考
        # 数据。**只收全称与不会撞车的缩写**，裸 gap / snap 不收。
        "pattern": r"gaussian approximation potential|"
                   r"spectral neighbor analysis|moment tensor potential|"
                   r"neuroevolution potential|\bnep\b|"
                   r"deep[- ]?potential|deepmd|deeppmd|deeppotential|"
                   r"atomic cluster expansion|"
                   r"machine[- ]learn(?:ing|ed) potential",
        "names": _FF_NAMES_ML,
        "bonds": False, "coulomb": False, "dispersion": False,
        "why": "势由描述符加回归给出，作为势函数使用时不含解析库仑项或 LJ 项，"
               "亦无固定键表",
    },
    {
        "name": "平面波电子结构动力学",
        # VASP / CP2K / Quantum ESPRESSO / CASTEP / CPMD / ABINIT 等。
        # **刻意不收裸 `dft`**：经典力场论文也常提 DFT（拟合参数、或顺带算
        # 个性质），那类体系里静电与色散都是适用的。收进来会造出方向相反的
        # 错，把真实的缺失压成"不适用"，比多报一条缺失更坏。
        "pattern": r"\bvasp\b|\bcp2k\b|quantum espresso|\bcastep\b|"
                   r"\babinit\b|\bcpmd\b|plane[- ]wave|"
                   r"born[- ]oppenheimer|ab initio molecular dynamics|\baimd\b",
        "bonds": False, "coulomb": False, "dispersion": False,
        "why": "库仑作用由平面波基组隐式处理，不存在 Ewald/PME/reaction-field "
               "的选择；能量来自电子结构泛函，无 LJ 截断",
    },
    {
        "name": "反应力场",
        # 除商品名外还收**散文写法** "reactive MD"：说的是同一件事，
        # 键不断生成与断裂，所以没有固定拓扑可约束。只认 reaxff 会漏掉
        # 1812.03256v1（全文 6 处 "reactive MD simulation"，一次也没写
        # ReaxFF）。散文写法必须带上 MD/molecular dynamics，否则
        # "reactive surface"、"reactive site" 这类化学描述会撞进来。
        # 兜底仍在 _POS_BONDS：真写了 SHAKE/LINCS 的篇章照旧算有键可约束。
        # COMB 没进来：`comb` 是普通英文词，撞车风险不值当。
        "pattern": r"reaxff|reactive\s+(?:MD|molecular\s+dynamics)",
        "bonds": False, "coulomb": True, "dispersion": True,
        "why": "键级动态计算，无固定拓扑，固定键长约束无对象；"
               "但含 EEM 电荷与 vdW 项，静电与色散两项仍然存在",
    },
    {
        "name": "纯排斥模型势",
        # Weeks-Chandler-Andersen, J. Chem. Phys. 54, 5237 (1971)：
        # LJ 截断到势极小点并平移。
        # 姓氏两种拼法都收：原文是 Andersen，但英语世界普遍写成 Anderson，
        # 本语料里**唯一**的全名写法就是后者（2312.07027v2 一处），正字法
        # 一次都没出现（那一篇是靠 \bwca\b 才命中本族的）。要求三节全名
        # 连写，所以 "Anderson localization" 这类同名不同物不会误命中。
        "pattern": r"\bwca\b|weeks[- ]chandler[- ]anders[eo]n",
        "names": _FF_NAMES_WCA,
        "bonds": False, "coulomb": False, "dispersion": False,
        # 本族由 LJ 截断而来，所以文本里的 "LJ"/"Lennard-Jones" 是**母名**、
        # 不是反证据。只有本族带这个标记，别的族（键序、机器学习、平面波）
        # 面前出现的 LJ 仍然算数，那些体系里出现 LJ 意味着真有第二个
        # 带吸引尾的物种。见 _POS_DISPERSION_WEAK 处的说明。
        "lj_derived": True,
        "why": "WCA 是截断到势极小点并平移的 LJ，纯排斥，没有吸引尾可做长程校正",
    },
]

# 反向证据：文本里**明确出现了该对象的语汇**时，不再判"不存在"。
# 这条覆盖优先于族能力，因为族能力是"这个势通常有没有"，而这里是
# "这个体系里到底有没有"。实测反例：2506.04835v1 用 Tersoff 描述 Si、
# 同时用 12-6 LJ 描述 Au，只看势函数族会判"没有色散项"，那是错的。
# 宁可漏报不可误报：反向证据只要出现就退回适用。
_POS_COULOMB = re.compile(
    r"\bcoulomb\w*|\belectrostat\w*|\bewald\b|\bpme\b|\bpppm\b|\bspme\b|"
    r"\bpartial charges?\b|\bcations?\b|\banions?\b|"
    r"库仑|静电|电荷", re.I)
# 色散的正向证据**分两档**（原先是下面这两条合成一条 `_POS_DISPERSION`，
# 分档后那条已删除，别再把它加回来），因为"声明做了色散处理"与
# "报了势的名字"不是一回事：
#   · 强证据（下面这条）：dispersion / van der Waals / vdW / 12-6，
#     声明的是一个**色散处理**，或一个明确带吸引尾的势，足可推翻族能力。
#   · 弱证据：LJ / Lennard-Jones，只是势的**名字**。
# 弱证据在 WCA 族面前不作数：**WCA 就是截断到势极小点的 LJ**（见该族 why），
# 所以 WCA 论文里的 "LJ fluid" 多半是同一个体系的叫法，不是第二个物种。
# 实测：2312.07027v2 全文 12 处 LJ/Lennard-Jones 指的都是那个 WCA 体系；
# 而 2506.04835v1（Tersoff 描述 Si 加 12-6 LJ 描述 Au）走的是强证据那条，
# 不受影响，那条正是本机制当初被加进来的原因。
_POS_DISPERSION_STRONG = re.compile(
    r"\bdispersion\b|\bvan der waals\b|\bvdw\b|\b12[- ]6\b|色散|范德", re.I)
_POS_DISPERSION_WEAK = re.compile(r"lennard[- ]jones|\blj\b", re.I)
# 键约束只认**算法名**，不认 `bond` 一词（"chemical bonds" 到处都是）。
# 出现 SHAKE/LINCS/SETTLE/RATTLE，说明确实有可约束的键。
_POS_BONDS = re.compile(r"\bshake\b|\blincs\b|\bsettle\b|\brattle\b", re.I)


def _ff_capability(text: str) -> Dict[str, bool]:
    """识别文本里的势函数族，返回 {bonds, coulomb, dispersion} 三项能力。

    极性是刻意的：**默认全是 True**，True 就是改动前的行为（适用 → 该问）。
    只有明确识别出势函数族、且该族的形式里没有这个东西时才翻成 False，
    同时还要没有反向证据。

    识别不出势函数时绝不能推断"不存在"，没写力场名不等于体系里没有库仑项。
    这一条是本表最重要的一行：它把改动限制成"只在证据充分时才收紧"。
    """
    low = normalize_ws(text).lower()
    cap = {"bonds": True, "coulomb": True, "dispersion": True}
    matched = [fam for fam in FF_FAMILIES
               if re.search(fam["pattern"], low, re.I)]
    for fam in matched:
        for k in ("bonds", "coulomb", "dispersion"):
            # 族之间取并集：只要有一个匹配到的族具备该能力，就算存在。
            # 两个族都缺，才算不存在。
            cap[k] = cap[k] and fam[k]
    if _POS_COULOMB.search(low):
        cap["coulomb"] = True
    # 色散的正向证据分两档：强证据直接推翻族能力；弱证据（LJ 这个词本身）
    # 在"由 LJ 截断而来"的族面前不算数，因为那时它多半是同一个势的名字。
    if _POS_DISPERSION_STRONG.search(low):
        cap["dispersion"] = True
    elif _POS_DISPERSION_WEAK.search(low) and not any(
            fam.get("lj_derived") for fam in matched):
        cap["dispersion"] = True
    if _POS_BONDS.search(low):
        cap["bonds"] = True
    return cap


# 可滴定基团。spec 把 protonation_state 的对象写成「**可滴定残基**的
# 质子化态」，取值示例点的是 His 的命名约定（HID/HIE/HIP）、Asp/Glu 是否
# 质子化、Cys 是否成二硫键，所以判据是"有没有可滴定基团"，而不是
# "有没有蛋白"。词表直接照 spec 的示例列：His/Asp/Glu/Cys/Lys/Arg 的
# 各种写法，加上游离末端与质子化/滴定语汇。
# 短名（his/asp/glu…）的边界照 PROTEIN_RE 的规矩：残基编号形式要带数字
# （His64），光秃秃的 `his` 会命中英文代词，这里保留了整词形式，因为
# spec 的取值示例本身就是"HID/HIE/HIP 命名约定"这种不带编号的写法。
# 这张词表**只会把闸门开得更大**（命中越多，越不会掉进"封端肽"那条排除），
# 所以过宽是安全方向、过窄才是危险方向，与别的词表相反。裸的 `\bhis\b`
# 因此可以收：三字母残基码 HIS 在残基列表里就是这么写的，而 spec 对本字段的
# 说明里点名要求关注 HIS。全语料实测出现 0 次（唯一的封端篇 2602.11098v1
# 一个都没有），所以这一条是防真名漏判，不改动任何现有格子。
_TITRATABLE_RE = re.compile(
    r"\bhistidine\b|\bhis\d+\b|\bhis\b|\bhid\b|\bhie\b|\bhip\b|\bhsd\b|\bhse\b|"
    r"\bhsp\b|\baspart\w*|\bglutam\w*|\bcystein\w*|\blysin\w*|"
    r"\barginin\w*|\basp\d+\b|\bglu\d+\b|\blys\d+\b|\barg\d+\b|\bcys\d+\b|"
    r"\btermin(?:us|i|al)\b|\bprotonat\w*|\bdeprotonat\w*|\btitrat\w*|"
    r"质子化|可滴定|末端", re.I)

# 两端封端的**显式写法**：ACE-…-NME（乙酰基 … N-甲酰胺）是肽化学的通用
# 记号，不是哪一篇论文的自造词。要求两端**同时**写出来，是为了不把
# 泛指的 "capped"（封端纳米颗粒、加帽试剂）也算进来。
_BLOCKED_PEPTIDE_RE = re.compile(
    r"\bace[- ][a-z]{2,6}[- ]nme\b|"
    r"\bacetyla?l?[- ][a-z]{2,8}[- ]n[- ]methylamide\b", re.I)


def has_titratable_groups(ctx: Dict[str, Any]) -> bool:
    """体系里是否存在可滴定残基，protonation_state 的适用性判据。

    **只有一个定义处**：适用性表与 R-09 都读它。两处各判一次迟早会漂移，
    届时会出现"字段判不适用、规则却报'含蛋白体系未说明质子化态'"这种
    自相矛盾的输出。

    判为真的条件：体系含蛋白/肽，且文本没有把可滴定残基的存在排除掉。
    **"文本没点名"不是排除的理由**，蛋白几乎必然含可滴定残基，作者
    没写不等于不存在（本项目反复记下的那条：从沉默推断不存在，只在
    文本完备时才成立）。排除只发生在 _BLOCKED_PEPTIDE_RE 那一情形：
    文本明确写出两端都被封端，且全文没有任何可滴定基团的命名，
    两端无游离末端、侧链也没有，这个自由度才真不存在
    （2602.11098v1 的 ACE-ALA-NME 二肽）。

    已知会漏报的地方：封端肽若用本表以外的写法命名其可滴定残基
    （只给残基序号、或写突变体），本判据会误判"无"，属漏报，
    按本项目既定的优先级（假阳性比漏报更致命）可以接受。
    """
    if "titratable_residues_present" in ctx:
        return bool(ctx["titratable_residues_present"])
    # 手工构造的 ctx（测试夹具、审计脚本）没有这个键时退回旧判据。
    return bool(ctx.get("protein_present", False))


# 适用性表：返回 False 表示该字段在本体系下不存在。
# 这张表被两处使用，必须只有一份定义：
#   1. 抽取阶段的闸门，阻止正则误命中（见 extract 中 "适用性闸门"）
#   2. 补全阶段，把"文献没写"与"本文不存在"区分开（见 _na_or_missing）
# 两处若各自维护一份，迟早会漂移，届时同一字段在不同阶段得到矛盾结论。
_APPLICABILITY = {
    # 表语义：返回 False = 该字段在本体系下不存在。
    # （曾把这条写反成 `not in ("NVT","NVE")`，于是 ensemble=NPT 反而被判
    # "不需要恒压器"，R-01 在黄金集上连报两次假阳性。适用性表的极性写错
    # 是**静默**的：不崩、不报错，只是结论全反。）
    #
    # 这条判据**试过改成更"讲道理"的版本，又改了回来**，过程值得记下：
    #
    # 道理上，恒压器的适用性取决于**协议**（跑没跑 NPT），不取决于体系里有什么。
    # 所以"没写系综"≠"不需要恒压器"：作者跑了 NPT 却没声明，恒压器照样在。
    # 于是改成"只有写明 NVT/NVE 才判不适用，系综未知时判缺失"。
    #
    # 在前 12 篇标注语料上，这一改**净亏**：改对了 3 篇（2506/2509/2602，
    # 液相或体相无定形体系，NPT 确实很可能），却改错了 5 篇
    # （1007/1208/1812/2312/2603）。后 5 篇标注员给的理由是一致的：
    # 固定晶格固体、表面板、定容 NEMD、盒长与数密度都写死，
    # 这些体系里恒压器**物理上就不存在**，不需要作者再声明什么。
    #
    # 关键在于：这个理由来自**体系的物理性质**，不是来自文本。
    # 标注员是从"纳米管没有体相压强"推出来的，而这句话文本里没有。
    # 工具要复现这个判断，得先能识别"这是固定晶格固体/定容 NEMD 装置"，
    # 那是另一层领域知识建模，不是改一行 lambda 能做到的。
    #
    # 取舍依据是本项目的既定优先级：假阳性是使用者放弃工具的首要原因，
    # 所以宁可漏报也不误报。3 处漏报优于 5 处误报。
    # 这条限制已记入 README 的局限一节。
    "barostat": lambda c: c.get("ensemble") == "NPT",
    "ligand_parameters": lambda c: c.get("ligand_present", False),
    # 问的是**可滴定基团**，不是"有没有蛋白"：spec 的字段名就是「可滴定
    # 残基的质子化态」。判据只有一处定义，见 has_titratable_groups。
    "protonation_state": has_titratable_groups,
    "heat_flux_method": lambda c: c.get("heat_transport", False),
    # 冷热浴只存在于 NEMD。Green-Kubo 是平衡态方法，压根没有浴，
    # 按 heat_transport 判适用会在 GK 论文上误报"未报告冷热浴温度"。
    # 实测 2110（GK）与 2605（GK）两篇都栽在这条上。
    "temperature_bath": lambda c: c.get("nemd", False),
    "correlation_window": lambda c: c.get("heat_transport", False),
    # 无溶剂体系（真空/隐式/干态熔体）没有"水模型"可言。
    # 与 severity 无关：水模型对湿体系是 critical，对干态是 N/A。
    "water_model": lambda c: c.get("explicit_solvent", True),
    # 以下三条问的是"这个势函数的定义里有没有这个
    # 东西"。判据与依据见 FF_FAMILIES，全部来自势函数的原始文献，
    # 不来自本项目语料。
    # 默认 True = 改动前的行为：识别不出势函数就照常问，绝不推断"不存在"。
    "constraints": lambda c: c.get("has_bonds", True),
    "electrostatics": lambda c: c.get("has_coulomb", True),
    "dispersion_correction": lambda c: c.get("has_dispersion", True),
}


PATTERNS = {
    "force_field": _extract_force_field,
    "water_model": _extract_water,
    "thermostat": _extract_thermostat,
    "barostat": _extract_barostat,
    "ensemble": _extract_ensemble,
    "timestep": _extract_timestep,
    "electrostatics": _extract_electrostatics,
    "constraints": _extract_constraints,
    "dispersion_correction": _extract_dispersion,
    "simulation_length": _extract_simulation_length,
    "system_size": _extract_system_size,
    "heat_flux_method": _extract_heat_flux,
    "temperature_bath": _extract_temperature_bath,
    "correlation_window": _extract_correlation_window,
    "software": _extract_software,
    "initial_structure": _extract_initial_structure,
    "protonation_state": _extract_protonation,
    "ligand_parameters": _extract_ligand_params,
}

# 规则覆盖不了的字段才交给 LLM。
# 早期版本把 software / initial_structure 也归入此类，实测完全可以正则化，
# 规则精确率高于 LLM 且零成本，能规则化的绝不下放。
# 目前仅保留真正需要读句子语义的字段：缺失残基/建模步骤这类开放描述。
LLM_ONLY_FIELDS = ["missing_residues_note"]


# -----------------------------------------------------------------------------
# 场景判定：contextual 字段是否"适用"取决于体系与方法
# -----------------------------------------------------------------------------
# 蛋白/配体语境词表。
# 短词一律带 \b：`his` 必须写成 `\bhis\d+\b`（残基编号如 His64），
# 否则会命中 "this"；`pdb` 必须带边界，否则会命中别的标识符。
PROTEIN_RE = re.compile(
    r"\bproteins?\b|\bresidues?\b|\bpeptides?\b|\bhistidine\b|\bhis\d+\b|"
    r"\basp\d+\b|\bglu\d+\b|\blys\d+\b|\barg\d+\b|\bcys\d+\b|"
    r"\bpdb\b|\blysozyme\b|\bbackbone\b|\balpha[- ]helix\b|"
    r"\bbeta[- ]sheet\b|\bwild[- ]type\b|\bunfold\w*\b|"
    r"蛋白|残基|组氨酸|多肽|氨基酸")

# 配体词表：只收**几乎不会**出现在无配体语境里的词。
#   · 不收 substrate，材料学里指固体基底（见 detect_context 的双义词处理）
#   · 不收 gaff / cgenff / antechamber / am1-bcc / resp，
#     这些是"小分子力场参数化工具"，聚合物同样用它们。
#     实测教训：把它们收进来后，一个用 GAFF2 + AM1-BCC 参数化的
#     聚合物熔体被判成"含配体"，随即报"未报告配体参数"。
#     它们说明的是"电荷怎么来的"，不是"体系里有配体"。
LIGAND_RE = re.compile(
    r"\bligands?\b|\binhibitors?\b|\bsmall[- ]molecules?\b|\bdrugs?\b|"
    r"\bbinding free energy\b|\bdock\w*\b|"
    r"配体|小分子|抑制剂|药物")

def detect_context(text: str) -> Dict[str, bool]:
    low = normalize_ws(text).lower()
    # 关键词必须用**词边界**匹配，绝不能用 `k in low`。
    # 实测代价：`"his" in low` 会命中 "this"，于是
    # "reported in this article" 被判成"含蛋白"，"未报告质子化态"
    # 随之报出。在 24 篇真实方法段上，20 篇（83%）栽在这一条，
    # 是本工具最大的假阳性来源；而假阳性正是使用者放弃工具的首要原因。
    protein_hit = PROTEIN_RE.search(low)
    ligand_hit = LIGAND_RE.search(low)
    # "substrate" 是双义词：酶学里指配体，材料学里指固体基底
    # （silica substrate / graphene substrate）。后者在本语料里极常见，
    # 单凭它判"有配体"会稳定误报，因此只在同时出现蛋白语境时才采信。
    if not ligand_hit and protein_hit and re.search(r"\bsubstrates?\b", low):
        ligand_hit = True
    # 无溶剂体系有三类，判定必须分开：
    #   真空/气相、隐式溶剂  -> 明确写了
    #   干态熔体/无溶剂      -> 通常不写"vacuum"，只写 melt / solvent-free
    # 早期版本只认前两类，导致任何聚合物熔体研究都被误报"未报告水模型"。
    # 这类假阳性会直接摧毁使用者的信任，所以单列一类。
    implicit = re.search(
        r"implicit solvent|隐式溶剂|\bmm/?pbsa\b|\bmm/?gbsa\b|generalized born|"
        r"poisson[- ]boltzmann|continuum solvent|solvent[- ]accessible|"
        r"\bgb\s+model\b", low)
    vacuum = re.search(r"vacuum|真空|gas phase|气相|in\s+vacuo", low)
    solvent_free = re.search(
        r"solvent[- ]free|without\s+(?:any\s+)?solvent|no\s+solvent|dry\s+(?:melt|polymer|state)|"
        r"无溶剂|不含溶剂|干态|熔体|melt\b|bulk\s+polymer", low)

    # explicit_solvent 必须由**正向证据**支撑，不能"没写真空就算有水"。
    # 原实现是 `not (vacuum or solvent_free)`，默认 True，等于把
    # "没说"当成"有"。后果在真实语料上很严重：24 篇里 14 篇（58%）
    # 是石墨烯/二氧化硅/金属这类压根没有溶剂的体系，却被报
    # "未报告水模型"。判定适用性时，缺证据要判**不适用**，
    # 而不是判"缺失"，这正是本工具四态设计要区分的那件事。
    # 反向证据（写着有溶剂）在文献里几乎是必然出现的：
    # 真用了显式溶剂，方法段不可能不出现 water / solvated / aqueous 之类字样。
    #
    # 但"出现溶剂词"还不够，必须出现的是**溶剂作为介质**的构式。
    # 第一版拿了裸词 \bwater\b|solvat|hydrat，结果两类**现象词**混了进来：
    #   · "hydroxylated (hydrated) surface / (hydrated) edge structures"
    #     表面羟基化，说的是表面化学，不是溶剂
    #   · "ion solvation"，电解液里的溶剂化现象，不是溶剂模型
    #   · "water molecules adsorbed on the surface"，吸附质，不是溶剂
    #   第三类：\bwater molecules?\b（曾收进来）会命中表面吸附水。
    # 这两类的共同点是：溶剂词修饰的是**另一个名词**，而不是"体系泡在什么里"。
    # 所以判据收紧为下面这些构式，其中每一条都要求溶剂是介质或专门的模型：
    # 裸 \bwater\b / hydrat / solvat / 溶剂 / 水分子 一律不再单独采信。
    #
    # 注意这里没有变成"必须写出水模型名才算适用"，那会让本字段永不可能
    # 报 missing（自证循环）。"solvated in water" 这类构式是**独立于模型名**
    # 的适用证据：作者说了体系泡在水里、却没说哪个水模型，正是要抓的缺失。
    solvent_present = re.search(
        r"\bwater box\b|\bwater model\b|\bwater shell\b|"
        r"\bexplicit(?:ly)? water\b|\bexplicit(?:ly)? solvat\w*\b|"
        r"\baqueous solution\b|\bsolvated\s+(?:in|with|by)\b|"
        r"\bsolvent box\b|\bsolvation shell\b|\bin water\b|\bin aqueous\b|"
        r"\bexplicit solvent\b|"
        # 具体水模型名即"用了这个水模型"的直接证据，无需再要求构式
        r"\btip[3-5]p\b|\bspc/?e?\b|\bopc3?\b|"
        r"水盒|水溶液|显式溶剂", low)
    explicit_solvent = bool(solvent_present) and not (implicit or vacuum or solvent_free)

    # 系综必须在这里就定下来：_APPLICABILITY 中 barostat 的闸门依赖它。
    # 早先版本只把 ensemble 留在字段层、由 validate.build_effective_context
    # 补进 ctx，导致抽取阶段的闸门永远拿到 None，恒压器连带整条规则链失效。
    ens_hit = _extract_ensemble(text)
    ens = ens_hit[0] if ens_hit else None

    # 势函数能力：见 FF_FAMILIES。注意这里传的是**全文** text，
    # 不是切出来的方法段 body。
    cap = _ff_capability(text)

    # 可滴定残基（protonation_state 的适用性判据，理由见 has_titratable_groups）。
    # 默认跟着"有没有蛋白"走；只有在**文本明确两端封端、全文又没点名任何
    # 可滴定基团**时才把它从真翻成假，那一篇确实是"有肽但没有可滴定残基"。
    titratable = bool(protein_hit)
    if titratable and (_BLOCKED_PEPTIDE_RE.search(low)
                       and not _TITRATABLE_RE.search(low)):
        titratable = False

    return {
        "ensemble": ens,
        "solvent_free": bool(solvent_free),
        "protein_present": bool(protein_hit),
        "ligand_present": bool(ligand_hit),
        "titratable_residues_present": titratable,
        "explicit_solvent": explicit_solvent,
        "heat_transport": bool(re.search(
            r"thermal conductivity|heat flux|热导率|导热|green-kubo|nemd", low)),
        # 冷热浴 / 温度梯度只属于 NEMD。GK 是平衡态方法，没有浴。
        # 这条与 heat_transport 必须分开：后者是"这个研究算不算热输运"，
        # 前者是"用的哪一类热输运方法"。两者混同会让 GK 论文被报
        # "未报告冷热浴温度"，那是把不适用的字段报成缺失。
        "nemd": bool(re.search(
            r"\bnemd\b|non[- ]equilibrium\s+molecular\s+dynamics|\brnemd\b|"
            r"cold\s+(?:bath|reservoir|slab)|hot\s+(?:bath|reservoir|slab)|"
            # 「两端热浴」的空间写法：left/right heat baths。
            # 它和上面 hot/cold 那一支是同一件事的两种说法，热浴加在
            # 两端区域，目的就是建立温度梯度，所以只可能出现在 NEMD 里。
            # 单写 \bleft\b ... heat bath 是不行的：语料里 "left/right ends
            # are fixed"、"the heat bath is governed by" 这样的句子各说各的，
            # 中间夹一个句号就会串味，所以要求两者**紧邻**（同一名词短语）。
            # 实测：分隔符允许 / and or 时全语料仍只有 1007.5359v2 一处
            # （"The left/right heat baths are applied to regions …"）；
            # 1208.6331v1 的 "coupling of the system to a heat bath at
            # temperature T" 与 2108.03424v1 的 "boundary heat baths"
            # 都不是两端写法，不受影响。
            r"(?:left|right)(?:\s*(?:/|and|or)\s*(?:left|right))?\s+heat\s+baths?\b|"
            r"temperature\s+gradient|imposed\s+flux|swap\s+frequency|"
            r"冷浴|热浴|温度梯度", low)),
        # 这里原先还有一个 "has_temperature_grade" = bool(_extract_temperature_bath(text))。
        # 它是**只写不读的死键**：全仓 grep 只有这一处出现，src/ 与 eval/ 都没有读者，
        # temperature_bath 的闸门读的是 nemd、不是它。留着的唯一效果是：任何一次
        # 对 _extract_temperature_bath 的改动都会在账本里报成"ctx 漂移"，
        # 而那恰恰是召回层要做的事，于是它只制造假警报。
        # 退役它不改变任何行为：validate.build_effective_context 只**增加**键、
        # 不遍历 ctx（src/validate.py:182-199），所以下游拿到的字典除了少一个
        # 没人读的键之外逐位相同。已实测：24 篇的状态与规则编号逐格不变。
        # 势函数能力三项。由 FF_FAMILIES 判定，见该表的说明。
        # 在 text 上判，不在 body（切出的方法段）上判，势函数名常出现在
        # 摘要或引言，切段会把证据切掉；这是从 `protein_present` 那次
        # 误报里学到的：判据要在全文里找，抽取才在方法段里做。
        "has_bonds": cap["bonds"],
        "has_coulomb": cap["coulomb"],
        "has_dispersion": cap["dispersion"],
    }


# -----------------------------------------------------------------------------
# 方法段定位
# -----------------------------------------------------------------------------
SECTION_HEADS = re.compile(
    r"^\s*(?:#{1,4}\s*)?(?:\d+\.?\d*\.?\s*)?"
    r"(methods?|computational\s+details?|simulation\s+details?|materials?\s+and\s+methods?|"
    r"方法|计算方法|模拟方法|计算细节)\s*$",
    re.I,
)


def segment_methods(text: str) -> str:
    """切出方法段。切不出就返回原文（宁可多给，不可漏给）。"""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        if SECTION_HEADS.match(ln):
            start = i + 1
            break
    if start is None:
        return text
    # 到下一个同级或更高级标题为止
    end = len(lines)
    for j in range(start, len(lines)):
        if re.match(r"^\s*(?:#{1,4}\s*)?\d+\.?\s+[A-Z一-鿿]", lines[j]):
            end = j
            break
        if re.match(r"^\s*#{1,3}\s+\S", lines[j]) and j > start:
            end = j
            break
    return "\n".join(lines[start:end]).strip() or text


# -----------------------------------------------------------------------------
# 主抽取函数
# -----------------------------------------------------------------------------
def extract(text: str, schema: Dict[str, Any], backend: str = "rule",
           llm_client=None) -> Dict[str, Any]:
    """
    backend="rule"，纯规则，离线可跑
    backend="llm"，调用 llm_client(prompt) -> str，返回 JSON 字符串
    backend="hybrid"，规则打底 + LLM 补 LLM_ONLY_FIELDS（推荐）
    """
    # 先归一化空白，再分段与抽取，顺序不能反，
    # 否则跨行折断的短语在分段后已经定型，救不回来。
    text = normalize_ws(text)
    body = segment_methods(text)
    ctx = detect_context(text)

    all_fields: List[Dict[str, Any]] = list(schema.get("fields", []))
    all_fields += list(schema.get("heat_transport_extension", []))
    by_key = {f["key"]: f for f in all_fields}

    result: Dict[str, Any] = {}

    # --- 规则层 ---
    for key, fn in PATTERNS.items():
        if key not in by_key:
            continue
        # 适用性闸门：体系里不存在的东西，正则也会误命中。
        # 例：聚合物熔体用 GAFF2 描述主链，GAFF2 会被 ligand_parameters 的
        # 正则捕获，但这不是"配体参数"，记录它等于制造一个假阳性。
        gate = _APPLICABILITY.get(key)
        if gate is not None and not gate(ctx):
            continue
        hit = fn(body)
        if hit:
            # 抽取器一般返回 3 元组；第 4 位是可选的状态覆盖（目前只有
            # force_field 用它来报 ambiguous）。默认仍是 present。
            value, conf, span = hit[:3]
            status = hit[3] if len(hit) > 3 else "present"
            result[key] = {
                "value": value,
                "status": status,
                "confidence": conf,
                "evidence": _locate(text, span),
                "note": "",
            }

    # --- LLM 层 ---
    # llm_error 非空 ⇒ 这一层**一格都没生效**，下面的结果全部来自规则层。
    # 它必须进 meta.backend_effective：请求了 --backend llm 而实际跑的是规则，
    # 报告上还写"抽取方式: llm"，就是谎报自己的来源，而"别让'报告了'和
    # '真的做了'之间有缝"正是这个工具存在的理由，它自己不能留这条缝。
    #
    # 零贡献有四种，先前只堵了前两种（抛异常、没给客户端）：
    #   3. **目标字段没有输出通道**。hybrid 只把 LLM_ONLY_FIELDS 交给模型，
    #      而该表当前唯一一项 missing_residues_note 不在 schema 的 18 字段内
    #      （不进 by_key），于是返回值会被下面 `if k not in by_key: continue`
    #      整批丢弃：调用发生了、prompt 发出去了、结果一格没进来，
    #      报告仍写"抽取方式: hybrid"。这与静默降级是同一类错误。
    #   4. **层跑通了但 0 格进入结果**（返回值全被高置信规则挡下、或本就为空）。
    # 因此 targets 先与 by_key 求交——没有输出通道的字段不去问；
    # 四种情况一律记入 llm_error，让 backend_effective 落到 rule。
    llm_error = ""
    llm_applied = 0
    if backend in ("llm", "hybrid"):
        if llm_client is None:
            llm_error = "未提供 LLM 客户端（llm_client=None），本层未运行"
        else:
            _want = LLM_ONLY_FIELDS if backend == "hybrid" else list(by_key)
            targets = [k for k in _want if k in by_key]
            _no_channel = [k for k in _want if k not in by_key]
            if not targets:
                llm_error = (
                    "目标字段均不在 schema 的 18 字段内、没有输出通道："
                    + "、".join(_no_channel)
                    + "。该字段不进检查单，其返回值会被丢弃，"
                      "故本层本次零贡献——结果与纯 rule 完全等价")
            else:
                try:
                    llm_out = _call_llm(llm_client, body, by_key, targets, ctx)
                    for k, v in llm_out.items():
                        if k not in by_key:
                            continue
                        if v in (None, "", [], {}):
                            continue
                        # 规则层已命中且高置信，则保留规则结果（规则精确率更高）
                        if result.get(k, {}).get("confidence") == "high":
                            continue
                        result[k] = {
                            "value": v.get("value") if isinstance(v, dict) else v,
                            "status": v.get("status", "present") if isinstance(v, dict) else "present",
                            "confidence": v.get("confidence", "medium") if isinstance(v, dict) else "medium",
                            "evidence": v.get("evidence", "") if isinstance(v, dict) else "",
                            "note": v.get("note", "") if isinstance(v, dict) else "",
                        }
                        llm_applied += 1
                except Exception as e:               # noqa: BLE001
                    llm_error = f"{type(e).__name__}: {e}"
        # 第四种零贡献：层跑通了，但 0 格进入结果（返回值全被高置信规则
        # 挡下、或本来就是空）。产出仍是 100% 来自规则层，同样不能报 hybrid。
        if not llm_error and not llm_applied:
            llm_error = ("LLM 层已调用，但 0 格进入结果"
                         "（返回值被高置信规则挡下或为空），本次产出全部来自规则层")
        if llm_error:
            result["_llm_error"] = {"status": "error", "note": llm_error}

    # --- 补全：未命中的字段标 missing / not_applicable ---
    for key, f in by_key.items():
        if key in result:
            continue
        status, span, why = _na_or_missing(key, f, ctx, body)
        result[key] = {
            # 刻意不产生取值：本层只表达"提到了但不够具体"，
            # 顺手给个数就是编数（见 _PARTIAL 的不变式 2）。
            "value": None,
            "status": status,
            # n/a 意为"没去看"，与"看了、证据不足"是两回事，不能共用。
            "confidence": "low" if status == "ambiguous" else "n/a",
            "evidence": _locate(text, span) if span else "",
            "note": why,
        }

    return {
        "context": ctx,
        "fields": result,
        "meta": {
            "backend": backend,
            # 实际生效的后端：只要 LLM 层 0 格进入结果（四种零贡献见上），
            # 产出就 100% 出自规则层，此处写 "rule" 而不是 backend。
            # 检查单会显式警告并附原因。
            "backend_effective": "rule" if llm_error else backend,
            "llm_error": llm_error,
            # LLM 层实际写进结果的字段格数。为 0 而 backend 非 rule ⇒ 谎报来源，
            # 故它与 backend_effective 必须一起看。
            "llm_applied": llm_applied,
            "methods_chars": len(body),
            "n_present": sum(1 for v in result.values() if v.get("status") == "present"),
            "n_missing": sum(1 for v in result.values() if v.get("status") == "missing"),
            "n_ambiguous": sum(1 for v in result.values() if v.get("status") == "ambiguous"),
            "n_na": sum(1 for v in result.values() if v.get("status") == "not_applicable"),
        },
    }


# -----------------------------------------------------------------------------
# 部分证据：提到了，但说得不够具体 -> ambiguous
# -----------------------------------------------------------------------------
# 四态里 ambiguous 的定义是"文本**提到了**这个字段，但说得不够具体、不足以
# 据以行动"（spec.md）。但在这张表之前，这条路径根本不存在：_na_or_missing
# 只能返回 not_applicable 或 missing，ambiguous 仅能从"已经匹配到具体东西"
# 的抽取器内部返回。于是"有证据、但证据不足"，恰恰是匹配不成功的典型情形，
# 一律被判成 missing。145 处分歧里 103 处是"工具报 missing 而标注不是"，
# 其中目标域内 32 处标注为 ambiguous，全部卡在这条缺失的路径上。
#
# 四条不变式（第 [18] 组回归直接钉住）：
#   1. **闸门先赢**：适用性判定排在部分证据之前。24 篇里 17 篇
#      ctx["ensemble"] 为 None，恒压器全靠闸门判 N/A；顺序反了，这 17 篇
#      会整体从 not_applicable 变成 ambiguous，比要修的错误大一个量级。
#   2. **不产生取值**：value 保持 None。本项目的反复准则是"缺失会被看见，
#      错值不会"；"提到了但没给数"绝不能顺手给一个数。
#   3. **只在抽取器未命中之后生效**：命中即 present，不降级。所以这张表
#      只可能让格子**离开 missing**，不可能移动 present 或 not_applicable。
#      这是可审计的包含性，与甲-1、甲-4 的补充分支同一形状。
#   4. **探针必须是文本可判的**：判据得是"作者写了什么"，不是"这个体系
#      物理上应该怎样"。后者是 [extract.py:963-982] 那次回退的教训，
#      那条更"讲道理"的恒压器闸门实测 3 fixed / 5 broken。
#
# 每条 why 都从 spec.md 抄下来，不许自己发明理由，语料只用来量
# "这条正则是否精确实现了规范里的那条判据"，不用来发明判据。


def _probe_re(pat):
    """把正则包成探针：返回命中片段（供 _locate 定位），没命中返回 None。"""
    def probe(t: str):
        m = pat.search(t)
        return m.group(0) if m else None
    return probe


def _probe_thermostat_named(t: str):
    """指名了恒温器型号，却没声明系综。

    复用 THERMOSTAT_CANON 而不是另写 `nose|berendsen|langevin`：这些名字
    已经是本模块的封闭词表，再写一份迟早漂移。
    """
    hit = _variant_lookup(THERMOSTAT_CANON, t)
    return hit[1] if hit else None


# 盒子几何。spec 的原句是"只给盒子尺寸不给原子数，属 ambiguity"，
# 所以探针**只认盒子几何**，不认 "63 water"、"number density" 这类
# 粒子数/密度写法，那不是"只给了盒子"，是给了另一件东西。
# 宽版本实测 6 fixed / 3 broken（1603.06347v2 / 2404.14338v1 / 2604.01940v3
# 标注全是 missing）。**"宁可窄"是量出来的，不是猜的。**
_SYS_BOX_RE = re.compile(
    r"\bL[_ ]?[xyz]\s*=|"
    r"\d+(?:\.\d+)?\s*(?:[x×]|by)\s*\d+(?:\.\d+)?\s*(?:[x×]|by)\s*\d+(?:\.\d+)?|"
    r"\bsupercell\b|\bsystem\s+sizes?\b", re.I)

# 给了**单条链/单个分子**的规模，却没给被模拟体系的规模。
# 2602.17595v1 是这条的实例："the representation of each polymer has
# around 200 atoms" —— 那个 200 是单条链的化学表示，不是模拟胞；
# 抽取器已按"取值必须归属于被模拟的体系"把它否决（见 _SIZE_OTHER_REFERENT），
# 否则报出去就是一个 high 置信度的错值。
# 但也不能就此停在 missing：作者确实给了规模，只是给的是另一个层级，
# 属"有证据但不足"，正是 ambiguous 的定义。
# 词表与 _SIZE_OTHER_REFERENT 的末条刻意保持一致——同一批词，那边用来
# **否决取值**，这边用来**判歧义**，两处若漂移就会一边否一边不认。
# 本语料实测只命中这 1 篇，其标注正是 ambiguous（1 fixed / 0 broken）。
_SYS_PER_UNIT_RE = re.compile(
    r"(?:each|per|every)\s+(?:chain|polymer|molecule|monomer|repeat[ -]unit|"
    r"CRU|bead)s?\b[^.]{0,70}?[0-9][0-9,]*\s*(?:atoms|particles|原子)", re.I)

# 热流的自关联分析被点名写出，但没给窗口/关联时间。
# 注意它**不是**"没提关联函数 -> 判 not_applicable"：后者已实测否决
# （见 README 缺陷表与 fp_hunt.py 里那条撤回的判据），词汇数为 0 的
# 真实含义是"我们手上这段节选没用这些词"，不是"没做这个计算"。
# 两者形似而方向相反：一个是"有证据但不足"，一个是"断言不存在"，
# 只有前者能在只读文本的层面上成立。
_CORR_RE = re.compile(
    r"\bHCACF\b|"
    r"(?:heat|thermal)[- ](?:flux|current)[^.]{0,40}?(?:auto)?correlation|"
    r"(?:auto)?correlation\s+function\s+of\s+the\s+(?:heat|thermal)", re.I)

# 冷热浴被点名写出（source/sink 区、thermostatted regions、温度梯度），
# 但没给出冷热两端的具体温度。
# 它**不是**"没提冷热浴 -> 判 not_applicable"，那是丙节里已实测否决的
# 方向（2506.04835v1：词汇数为 0 的含义是"手上这段节选没用这些词"，
# 不是"没做这个计算"）。本探针只在闸门（nemd）已经判"这篇就是 NEMD"
# 之后才可能生效，落在"有证据但不足"那一侧。
# 2108.03424v1 是这条探针的已知边界：它的冷热 slab 温度只有约化单位
# （"T_1 = 1.1 and T_2 = 0.9"，无 K），报成 {hot_K, cold_K} 就是**取值错误**
# （谎称开尔文），还会让 R-11 在一篇明确保持小 ΔT 的论文上误报"偏离线性区"。
# 所以那格只能停在 ambiguous，属召回层**做不到**，不是没做。
_TB_MENTION_RE = re.compile(
    r"\b(?:hot|cold)\s+(?:slab|bath|reservoir|end|region)s?\b|"
    r"\bheat\s+(?:source|sink|bath)s?\b|"
    r"\b(?:source|sink)\s+(?:region|reservoir)s?\b|"
    r"\bthermostatted\s+regions?\b|"
    r"temperature\s+gradient|linear\s+response\s+regime|"
    r"temperature\s+difference|温度梯度|冷热浴", re.I)

# initial_structure 曾有一条 `initial configuration/structure/coordinates` 探针，
# **已撤掉**：实测它一格也动不了。全语料有 4 篇出现该短语（1812 / 2409 /
# 2602.11098 / 2604），而这 4 篇的抽取器**全都已经命中**，partial 恒被遮蔽。
# 一条在本语料上无法过账的规则，留着只是不可验证的风险面；
# 属于它的那份 recall 缺口归 乙-2（2602.11098 是抽取器命中而标注 missing，
# 那是"多报"方向的问题，更不能拿 ambiguous 去盖）。
_PARTIAL: Dict[str, List[Tuple[Any, str]]] = {
    "temperature_bath": [
        (_probe_re(_TB_MENTION_RE),
         "点名了热源/热汇或温度梯度，却没给出冷热两端的温度设置，spec 原文"
         "「只给方法不给温度设置，无法判断是否在线性区」"),
    ],
    "ensemble": [
        (_probe_thermostat_named,
         "指名了恒温器型号，却没声明系综，系综决定哪些量守恒、哪些由浴固定；"
         "写了恒温器说明温度确实被控制，只是没说清是 NVT 还是 NPT"),
    ],
    "correlation_window": [
        (_probe_re(_CORR_RE),
         "点名了热流自关联分析（HCACF），却没给统计窗口与关联时间，"
         "κ(t) 要在平台上取值，缺窗口无法判断取的是不是收敛区"),
    ],
    "system_size": [
        (_probe_re(_SYS_BOX_RE),
         "只给了盒子几何、没给原子数，spec 明写「只给盒子尺寸不给原子数，"
         "属 ambiguity」"),
        (_probe_re(_SYS_PER_UNIT_RE),
         "只给了单条链/单个分子的原子数，没给被模拟体系的规模。"
         "这个数不是体系规模（所以没有进取值），但作者确实报告了规模，"
         "只是报在另一个层级上——有限尺寸外推要的是体系粒子数，"
         "给链的规模仍不足以据以行动"),
    ],
}


def _partial_evidence(key: str, text: str) -> Optional[Tuple[str, str]]:
    """返回 (证据片段, 理由)，没有则 None。"""
    for probe, why in _PARTIAL.get(key, ()):
        try:
            span = probe(text)
        except Exception:                             # noqa: BLE001
            continue                                  # 探针失败按"没证据"处理
        if span:
            return span, why
    return None


def _na_or_missing(key: str, field: Dict[str, Any], ctx: Dict[str, bool],
                   text: str = "") -> Tuple[str, str, str]:
    """
    区分 "文献没写" 与 "这个问题在本文体系里不存在"。

    两者都表现为字段为空，但含义相反：前者是缺陷，后者报出来就是假阳性。
    假阳性比漏报更伤信任，使用者看到"聚合物熔体未报告水模型"这种
    提示后，会连正确的提示一起不再相信。

    判定顺序**就是判据**，不能换：
      1. 适用性闸门   -> not_applicable
      2. 部分证据     -> ambiguous
      3. 否则         -> missing
    闸门必须在部分证据之前（见 _PARTIAL 的不变式 1）。

    返回 (status, span, why)：span 给 _locate 定位用，why 进 note。
    这两样是 ambiguous 最需要的东西，报"提到了但不够具体"时，
    必须能指出**提到的到底是哪句**，否则这句判断无法复核。
    """
    fn = _APPLICABILITY.get(key)
    if fn is not None:
        try:
            if not fn(ctx):
                return "not_applicable", "", ""
        except Exception:                             # noqa: BLE001
            pass                                      # 判定失败则保守按 missing 处理

    if text:
        got = _partial_evidence(key, text)
        if got:
            return "ambiguous", got[0], got[1]

    return "missing", "", ""


def _locate(full_text: str, span: str) -> Dict[str, str]:
    """把命中片段映射回原文位置，便于人工核验。"""
    if not span:
        return {"quote": "", "offset": -1}
    span = span.strip()
    idx = full_text.find(span)
    return {"quote": span[:160], "offset": idx}


def _call_llm(llm_client, body: str, by_key: Dict[str, Any],
              targets: List[str], ctx: Dict[str, bool]) -> Dict[str, Any]:
    """把 schema 里每个字段的 why 一并喂给模型，这是'强约束'的落地方式。

    why 与 state_rule 都要喂：why 讲理由（帮助判断），state_rule 讲
    "什么情况算哪一态"（帮助定状态）。schema 里把两者拆开是为了让
    why 能安全地印进人类可读的检查单（见 schema 头部说明），
    但拆开不等于模型不需要后者，这里补回去。
    """
    spec = []
    for k in targets:
        f = by_key.get(k)
        if not f:
            continue
        line = (f"- {k} ({f.get('name_cn')}): {f.get('why', '').strip()}\n"
                f"    合法取值示例: {f.get('examples', f.get('enum', ''))}")
        rule = (f.get("state_rule") or "").strip()
        if rule:
            line += f"\n    状态规则: {rule}"
        spec.append(line)
    prompt = (
        "You are extracting molecular-dynamics method parameters from a paper.\n"
        "Return STRICT JSON only. For each field below, output:\n"
        '  {"value": <value or null>, "status": "present|missing|ambiguous|not_applicable",'
        ' "confidence": "high|medium|low", "evidence": "<verbatim quote from the text>",'
        ' "note": "<one short sentence>"}\n'
        "Rules:\n"
        "  * Only use information present in the text. Never guess a force field name.\n"
        "  * If a value is described only vaguely (e.g. 'AMBER' without version),"
        " set status=ambiguous, not present.\n"
        "  * If the field does not apply to this system/method,"
        " set status=not_applicable.\n\n"
        f"Detected context (for deciding applicability): {json.dumps(ctx, ensure_ascii=False)}\n\n"
        "Fields to extract:\n" + "\n".join(spec) +
        "\n\nTEXT:\n<<<\n" + body + "\n>>>\n"
    )
    raw = llm_client(prompt)
    m = re.search(r"\{.*\}", raw, re.S)
    if m is None:
        # 不要把 json.loads 的原始报错直接抛上去，它只会说
        # "Expecting value: line 1 column 1 (char 0)"，看的人无从知道
        # 是模型没吐 JSON、还是网络/鉴权出了问题。把响应内容带进错误里。
        raise ValueError(
            "LLM 返回的内容里没有 JSON 对象（前 120 字符：%r）"
            % (raw or "")[:120])
    return json.loads(m.group(0))


def load_llm_client(provider: str = "anthropic", model: Optional[str] = None):
    """
    返回一个 callable(prompt) -> str。
    需要用户自备 API key；不配置时整个项目用 backend="rule" 也能跑通。
    """
    import os

    if provider == "anthropic":
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic 或改用 --backend rule") from e

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        mdl = model or "claude-sonnet-5"
        # 2000 曾经够，现在不够了：这个模型会先输出 thinking 块，而
        # **thinking 也计入 max_tokens**。18 个字段的说明 + 整段方法段的
        # 提示词下，思考没结束预算就用完，响应里一个 text 块都没有，
        # 表现为"模型返回空串"，且错误被 json.loads 报成
        # "Expecting value: line 1 column 1"，看不出真正原因。
        max_tok = int(os.environ.get("MD_REPRO_LLM_MAX_TOKENS", "8000"))

        def _call(prompt: str) -> str:
            r = client.messages.create(
                model=mdl, max_tokens=max_tok,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = "".join(b.text for b in r.content if b.type == "text")
            if not txt and r.stop_reason == "max_tokens":
                raise RuntimeError(
                    f"响应在 max_tokens={max_tok} 处被截断（thinking 计入该预算），"
                    f"没有可用的 text 块；调大环境变量 MD_REPRO_LLM_MAX_TOKENS")
            return txt

        return _call

    if provider == "openai":     # 兼容任何 OpenAI-format 端点（含 DeepSeek/Qwen 等）
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("pip install openai 或改用 --backend rule") from e

        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("OPENAI_BASE_URL"),
        )
        mdl = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        def _call(prompt: str) -> str:
            r = client.chat.completions.create(
                model=mdl, messages=[{"role": "user", "content": prompt}],
            )
            return r.choices[0].message.content

        return _call

    raise ValueError(f"unknown provider: {provider}")
