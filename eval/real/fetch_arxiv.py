# -*- coding: utf-8 -*-
"""
fetch_arxiv.py：从 arXiv 抓取真实文献的方法段，用于假阳性排查与抽取评估

为什么要走 arXiv 的 e-print（LaTeX 源码）而不是 PDF：
  PDF 抽出来的文本会把上下标、公式、连字符断行揉成噪声，
  正则抽取器的错误率里会混进大量"PDF 解析错误"，
  而这些错误与工具本身的质量无关，评估必须只测工具，不测解析器。
  LaTeX 源码是作者写的原文，结构干净，是更公平的输入。

输出（全部落在 eval/real/）：
  sections/<id>.txt     抽取出的方法段纯文本（首行标注 arXiv ID）
  manifest.json         每篇的元数据：id / 标题 / 分类 / 章节名 / 字符数 / sha256
  _scratch/             下载缓存，**不随提交**

版权与合规（重要）：
  这些是已发表论文的正文片段。它们只用于本地评估工具质量，
  **不进入提交材料**。提交材料里只出现聚合指标与 arXiv ID。
  因此 eval/real/ 在打包时必须排除，见本目录 README.md。

用法：
  python eval/real/fetch_arxiv.py --list                # 只搜索，看候选
  python eval/real/fetch_arxiv.py --limit 12            # 抓取并抽取
  python eval/real/fetch_arxiv.py --id 2207.13405       # 单篇（调试用）
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Dict

HERE = os.path.dirname(os.path.abspath(__file__))

# Windows 控制台默认 GBK，打印中文/符号会抛 UnicodeEncodeError。
# 评估脚本必须能在任何终端跑完，跑到一半因为打印失败而中断，
# 会让人误以为抓取本身出了问题。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
SECTIONS = os.path.join(HERE, "sections")
SCRATCH = os.path.join(HERE, "_scratch")
MANIFEST = os.path.join(HERE, "manifest.json")

API = "https://export.arxiv.org/api/query"
EPRINT = "https://arxiv.org/e-print/{}"
UA = "md-repro-checker/0.1 (research tool evaluation; contact: local)"
DELAY = 3.2                      # arXiv 要求请求间隔 >= 3 秒，务必遵守

# 覆盖面是本脚本的核心设计：case-01 是"干态聚合物熔体 + 直接 NEMD"，
# 工具在这一个场景上调对了，不代表它在别的场景上也对。
# 因此查询按**与 case-01 不同的维度**铺开：
#   体系（生物大分子 / 膜 / 水 / 离子液体 / 无机 / 低维材料）
#   软件（GROMACS / AMBER / LAMMPS / NAMD）
#   输运方法（NEMD / RNEMD / Green-Kubo / 无）
#   系综（NVE / NVT / NPT）
# 假阳性最容易出现在"字段不适用"的判断上（水模型、配体参数、质子化态），
# 所以湿体系、含配体体系、纯无机体系都必须各来一些。
QUERIES = [
    # 生物大分子 / 含配体：ligand_parameters 与 protonation_state 应当适用，
    ('abs:"molecular dynamics" AND abs:"binding free energy" AND abs:"MM/PBSA"', 8),
    ('abs:"molecular dynamics" AND abs:"protein" AND abs:"AMBER" AND abs:"explicit solvent"', 8),
    # 膜 / 界面：同时涉及水模型与脂质参数，
    ('abs:"molecular dynamics" AND abs:"lipid bilayer" AND abs:"permeation"', 6),
    # 电解质 / 离子液体：输运性质 + 非水溶剂，
    ('abs:"molecular dynamics" AND abs:"ionic conductivity" AND abs:"electrolyte"', 6),
    # 热输运其它方法：RNEMD / Green-Kubo，与 case-01 的 NEMD 形成对照，
    ('abs:"reverse nonequilibrium molecular dynamics"', 6),
    ('abs:"Green-Kubo" AND abs:"molecular dynamics" AND abs:"thermal conductivity"', 6),
    # 无机 / 低维：通常没有"水模型""质子化态"可言，最容易触发假阳性，
    ('abs:"molecular dynamics" AND abs:"thermal conductivity" AND abs:"graphene"', 6),
    ('abs:"molecular dynamics" AND abs:"silica" AND abs:"water" AND abs:"interface"', 6),
    # 反应力场 / 金属：力场体系完全不同，
    ('abs:"ReaxFF" AND abs:"molecular dynamics"', 5),
    # 泛化对照：任意 MD 方法段，用来估计"背景假阳性率"，
    ('abs:"molecular dynamics simulations" AND abs:"GROMACS"', 8),
]

# 章节标题的候选模式。真实论文里方法段的名字五花八门，
# 只认 "Methods" 会漏掉相当一部分（尤其物理类期刊）。
#
# 必须**分层**解析：真实论文的方法段常常长这样，
#   \section{Models and Methods}
#      \subsection{The EMD simulation setup}
#      \subsection{The NEMD simulation setup}
#      \subsection{Other MD simulation details}
# 若按"遇到下一个标题就结束"来切，父章节正文会被截成标题后几十个字符，
# 子章节反而胜出，抓到的方法段只是其中一小节，
# 下游会把其余十几个字段全报成"缺失"。
# 这是本评估里最贵的一类错误：看起来像工具漏报，实际是取文取错了。
SEC_PAT = re.compile(r"\\(sub)*section\*?\s*\{((?:[^{}]|\{[^{}]*\})*)\}", re.S)

# 方法段标题的候选词。
# 词表要**宁宽勿窄**：漏掉一个常见写法 = 一篇真论文被静默丢掉，
# 而多收一个候选只是排序问题（后面还有打分与 NOT_METHODS 兜着）。
# 实测教训：1007.5359v2 的方法段叫 "configuration and calculation details"，
# 早期词表里既没有 calculation、details 又要求写成 "details of"，
# 于是整段被跳过，白丢一篇。
METHODS_HINT = re.compile(
    r"(?i)(computational|simulation|simulat|method|model(?:s|ing|ling)?\b|"
    r"theoretical|theory|numerical|system\s+and|technical|force\s*field|"
    r"details|atomistic|calculation|protocol|setup|set-up|approach|"
    r"parameter|technique|implementation|scheme)")

# 明确**不是**方法段的章节名，用于排除（"Methods" 里出现这些词就跳过）
NOT_METHODS = re.compile(
    r"(?i)(result|discussion|conclusion|acknowledg|reference|"
    r"introduction|abstract|appendix|supplement)")


# -----------------------------------------------------------------------------
# 领域闸门：这条语料必须是**分子模拟**论文
#
# 光靠检索词圈不住领域。"Green-Kubo 热导率"这个查询既会命中 MD 论文，
# 也会命重离子碰撞/强子气体这类核物理论文，它们同样用 Green-Kubo 关系
# 算输运系数，同样满篇 "simulation / potential / ensemble"。
# 把这种论文混进来，检查器会在一篇与方法学无关的文章上大量报"缺失"，
# 指标里就混进了与工具质量无关的噪声。
#
# 判据分强弱两级，因为两类词的鉴别力差很多：
#   strong，只有原子级模拟才会写（力场、时间步、恒温器、MD 软件名）
#   weak，统计物理与核物理共用（potential / ensemble / simulation）
# 要求"至少 1 个强证据 + 强弱合计 3 个"，避免只凭 shared 词误判。
# -----------------------------------------------------------------------------
MD_STRONG = re.compile(
    r"(?i)(molecular[- ]dynamics|force[- ]?field|all[- ]atom|atomistic|"
    r"coarse[- ]grained|\blammps\b|\bgromacs\b|\bnamd\b|\bopenmm\b|"
    r"\bamber\b|\bcharmm\b|\bgromos\b|thermostat|barostat|"
    r"time[- ]?step|nos[eé][- ]hoover|langevin|periodic boundary|"
    r"equilibrat|kcal/mol|kj/mol|\bopls\b|\bgaff\b|reaxff|"
    r"\btip\d+p\b|\bspc/?e\b|"
    # 缩写必须单列：不少论文段落通篇只写 "MD" / "NEMD"，
    # 不写全称（实测 1812.03256v1 的 Reactive MD 段就是如此）。
    # 漏掉缩写的后果是**误挡真论文**，比多收一篇跑题论文严重得多。
    r"\bnemd\b|\brnemd\b|\bemd\b|\bmd\b)")

MD_WEAK = re.compile(
    r"(?i)(simulat|potential|ensemble|trajector|temperature|pressure|"
    r"cut[- ]?off|density|diffusion|viscosity|thermal conductivit|"
    r"nanosecond|picosecond|femtosecond)")


def md_scope_report(text: str) -> Dict[str, list]:
    """
    返回领域判定的证据明细，而不只是 True/False。

    闸门必须有可解释性：它一旦判错，是**静默删数据**，
    一篇真正的方法学论文被挡掉，指标上看不出来，只是语料少了一篇。
    所以把命中的词单独返回，好让人核对判得对不对。
    """
    low = text.lower()
    strong = sorted(set(MD_STRONG.findall(low)))
    weak = sorted(set(MD_WEAK.findall(low)))
    return {"strong": strong, "weak": weak,
            "ok": len(strong) >= 1 and (len(strong) + len(weak)) >= 3}


def is_md_paper(text: str) -> bool:
    """判断这篇论文是否属于"分子模拟"领域（用于把跑题的检索结果挡在语料外）。"""
    return md_scope_report(text)["ok"]


def title_and_abstract(tex: str) -> str:
    """取标题与摘要，判"这篇是不是分子模拟"最有效的一段文本。"""
    out = []
    m = re.search(r"\\title\s*\{", tex)
    if m:
        i, depth = m.end(), 1
        while i < len(tex) and depth:
            depth += (tex[i] == "{") - (tex[i] == "}")
            i += 1
        out.append(tex[m.end():i - 1])
    m = re.search(r"\\begin\s*\{abstract\}(.*?)\\end\s*\{abstract\}", tex, re.S)
    if m:
        out.append(m.group(1))
    return "\n".join(out)


# 合法的 arXiv 编号（新式 2401.12345 / 老式 cond-mat/0701254，均可带版本号）。
# 缓存文件名是从编号派生的，用编号形状校验可以挡住那些来源不明的缓存
# （例如下载被重定向后留下的 "ep.tar.gz"）。
ARXIV_ID = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$")


def cache_id(fn: str) -> str:
    """缓存文件名 -> arXiv 编号（文件名里 "/" 被写成了 "_"）。"""
    return fn[:-len(".tar.gz")].replace("_", "/")


# -----------------------------------------------------------------------------
# LaTeX -> 纯文本
# -----------------------------------------------------------------------------
def strip_latex(tex: str) -> str:
    """
    把 LaTeX 正文转成可读纯文本。

    目标是让正则抽取器读到与**作者原意一致的数字与术语**，排版保真只是手段：
      · 数值必须保住（"2 fs"、"300 K"、"1.2 nm"）
      · \SI{2}{\femto\second} 这类 siunitx 写法必须还原成 "2 fs"
      · 数学模式里的数值同样要保住（$T = 300$ K）
      · 引用、标签、图表环境必须删掉（否则会污染字段抽取）
    """
    s = tex

    # 0) 先处理 siunitx，否则 \SI 的数值会被后面的命令清理吃掉
    #    _SI{数值}{单位} 与 \si{单位} 共用同一个还原逻辑，
    #    所以写成 (num, unit) 两个普通参数，而不是接收 match 对象，
    #    后者会让 \si{...}（没有数值）无法复用。
    def _si(num_raw, unit):
        num = re.sub(r"\\[a-zA-Z]+|[{}]", "", num_raw)
        unit = unit.replace(r"\femto", "f").replace(r"\pico", "p")
        unit = unit.replace(r"\nano", "n").replace(r"\micro", "u")
        unit = unit.replace(r"\milli", "m").replace(r"\kilo", "k")
        unit = re.sub(r"\\per", "/", unit)
        unit = re.sub(r"\\(second|s)\b", "s", unit)
        unit = re.sub(r"\\(meter|metre|m)\b", "m", unit)
        unit = re.sub(r"\\(kelvin|K)\b", "K", unit)
        unit = re.sub(r"\\(gram|g)\b", "g", unit)
        unit = re.sub(r"\\(joule|J)\b", "J", unit)
        unit = re.sub(r"\\(mole|mol)\b", "mol", unit)
        unit = re.sub(r"\\(angstrom)\b", "A", unit)
        unit = unit.replace(r"\celsius", "C").replace(r"\degree", "deg")
        unit = re.sub(r"\\[a-zA-Z]+|[{}~]", "", unit).strip()
        return f"{num.strip()} {unit}".strip()
    s = re.sub(r"\\SI\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
               lambda m: _si(m.group(1), m.group(2)), s)
    s = re.sub(r"\\si\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}",
               lambda m: _si("", m.group(1)), s)
    s = re.sub(r"\\num\s*\{([^{}]*)\}", r"\1", s)

    # 1) 去注释：% 到行尾，但 \% 是转义的百分号
    s = re.sub(r"(?<!\\)%.*", "", s)

    # 2) 删掉图表/算法环境整体（其中的数字是图注，不是方法参数）
    for env in ("figure", "figure*", "table", "table*", "algorithm",
                "algorithmic", "tikzpicture", "thebibliography", "tabular"):
        s = re.sub(r"\\begin\s*\{%s\*?\}.*?\\end\s*\{%s\*?\}" % (env, env),
                   " ", s, flags=re.S)

    # 3) 删掉对抽取无意义、却会插入数字或噪声的命令
    s = re.sub(r"\\(?:cite|citep|citet|ref|eqref|label|autoref|footnote|"
               r"includegraphics|bibliography|input|include)\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}",
               " ", s)
    s = re.sub(r"\\(?:cite|ref|label)\b", " ", s)

    # 4) 数学模式：保留内容（数值在公式里往往是关键参数）
    s = re.sub(r"\$\$.*?\$\$", " ", s, flags=re.S)          # 行间公式整段丢弃
    s = re.sub(r"\\\[.*?\\\]", " ", s, flags=re.S)
    s = re.sub(r"\\begin\{(?:equation|eqnarray|align|displaymath)\*?\}.*?"
               r"\\end\{(?:equation|eqnarray|align|displaymath)\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\$([^$]*)\$", lambda m: " " + m.group(1) + " ", s)

    # 5) 保留文本的命令：只取其参数
    for cmd in ("text", "mathrm", "mbox", "textnormal", "emph", "textit",
                "textbf", "textrm", "SI", "unit", "si", "ensuremath", "hbox"):
        s = re.sub(r"\\%s\s*\{([^{}]*)\}" % cmd, r"\1", s)

    # 6) 常见符号命令 -> 可读字符
    repl = {
        r"\%": "%", r"\&": "&", r"\_": "_", r"\#": "#",
        r"\sim": "~", r"\pm": "+/-", r"\times": "x", r"\cdot": ".",
        r"\approx": "~=", r"\leq": "<=", r"\geq": ">=",
        r"\angstrom": "A", r"\AA": "A", r"\, ": " ", r"\;": " ",
        r"\ ": " ", r"~": " ",
    }
    for k, v in repl.items():
        s = s.replace(k, v)

    # 7) 剩下的未知命令：删命令保参数，再删残余反斜杠命令
    s = re.sub(r"\\[a-zA-Z@]+\s*\*?\s*(?:\[[^\]]*\])?", " ", s)
    s = s.replace("\\\\", " ").replace("\\", " ")

    # 8) 环境标记、花括号、排版残留
    s = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", s)
    s = s.replace("{", " ").replace("}", " ")

    # 9) 破折号与空白归一
    s = s.replace("---", " - ").replace("--", "-")
    s = s.replace("\n", " ")
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    s = re.sub(r"\s+([,.;:])", r"\1", s)
    s = re.sub(r"\.\s*\.", ".", s)
    return s.strip()


# -----------------------------------------------------------------------------
# 定位方法段
# -----------------------------------------------------------------------------
def split_sections(tex: str) -> list:
    """
    按标题层级切分，返回 [(标题, 正文, 层级), ...]。

    正文的结束位置 = 之后第一个**层级 <= 自己**的标题。
    这样父章节的正文自然包含它的全部子章节，方法段常常分散在
    若干子章节里（EMD 一套、NEMD 一套、其余细节一套），
    只取一个子章节会得到一份残缺的方法段。
    标题里的 LaTeX 命令（\\emph 等）先清掉，否则会影响下面的打分。
    """
    marks = []
    for m in SEC_PAT.finditer(tex):
        level = 1 + (m.group(1) or "").count("sub")
        raw = m.group(2)
        # 先整体删掉 \label{...}：它的花括号内容会被下面的"删命令保参数"
        # 留下，标题会变成 "{sec:methods}Models and Methods" 这种样子，
        # 既影响打分也污染输出。
        raw = re.sub(r"\\label\s*\{[^{}]*\}", " ", raw)
        title = re.sub(r"\\[a-zA-Z]+\s*", " ", raw)      # 其余命令：删命令
        title = re.sub(r"[{}]", " ", title)              # 残余花括号
        title = re.sub(r"\s+", " ", title).strip()
        marks.append((m.start(), title, level))
    if not marks:
        return []

    out = []
    for i, (pos, title, level) in enumerate(marks):
        end = len(tex)
        for j in range(i + 1, len(marks)):
            if marks[j][2] <= level:          # 同级或更高级 -> 本段结束
                end = marks[j][0]
                break
        out.append((title, tex[pos:end], level))
    return out


def pick_methods(tex: str):
    """
    挑出最像"计算方法"的那一段。

    真实论文的方法段位置极不固定：物理类常在第二节，生物类常倒数第二节，
    也有把参数塞进 SI、正文只留一句概括的。因此按"标题像不像"排序取最佳，
    而不是只看第一个候选，取错章节会直接制造假阳性。
    """
    cands = []
    for title, body, level in split_sections(tex):
        clean = re.sub(r"\s+", " ", title).strip()
        if NOT_METHODS.search(clean):
            continue
        if not METHODS_HINT.search(clean):
            continue
        cands.append((clean, body, level))

    if not cands:
        return None, None, None

    # 优先"计算/模拟"这类明确词，其次才是一般的 "Model"
    def score(t):
        tl = t.lower()
        s = 0
        if "computational" in tl or "simulation" in tl:
            s += 4
        if "method" in tl:
            s += 3
        if "model" in tl:
            s += 1
        if "system" in tl or "force field" in tl:
            s += 1
        return s

    # 排序键：先比打分，再比正文长度（方法段通常比 "Model" 这种泛标题长），
    # 最后比层级（-level 越大 = 层级数字越小 = 越靠顶层，优先取父章节）。
    # 父章节包含全部子章节，因此"宁可取大不取小"，
    # 取大了最多混进无关内容，取小了会漏掉整个方法段。
    cands.sort(key=lambda c: (score(c[0]), len(c[1]), -c[2]), reverse=True)
    return cands[0][0], cands[0][1], cands[0][2]


# -----------------------------------------------------------------------------
# 网络
# -----------------------------------------------------------------------------
def _get(url: str, timeout: int = 90) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def search(query: str, n: int) -> list:
    params = urllib.parse.urlencode({
        "search_query": query, "start": 0, "max_results": n,
        "sortBy": "relevance",
    })
    xml = _get(f"{API}?{params}").decode("utf-8", "replace")
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml)
    out = []
    for e in root.findall("a:entry", ns):
        idu = e.find("a:id", ns).text.strip()
        aid = idu.split("/abs/")[-1]
        title = " ".join(e.find("a:title", ns).text.split())
        pub = e.find("a:published", ns).text[:10]
        out.append({"id": aid, "title": title, "published": pub})
    return out


def _find_member(files: Dict[str, str], name: str):
    """在 e-print 的文件表里找 \\input 的目标，容忍路径前缀与大小写。"""
    n = name.replace("\\", "/").strip().lstrip("./").lower()
    if not n:
        return None
    for cand in (n, n + ".tex"):
        if cand in files:
            return cand
        for k in files:
            if k.endswith("/" + cand):
                return k
    return None


def _expand_inputs(tex: str, files: Dict[str, str], depth: int = 0) -> str:
    """
    把 \\input{...} / \\include{...} 就地展开。

    为什么必须做：很多期刊模板把方法段单独放在 methods.tex / SI.tex 里，
    由主文件 \\input 进来。主文件本身**不含**方法段正文，
    不展开的话，一篇完好的论文会被判成"没有方法段"而静默丢掉，
    抓取成功率因此虚低，而且丢的都是方法段写作规范的那一批。

    展开是递归的（A input B input C），带深度上限防自引用。
    """
    if depth > 6:
        return tex

    def sub(m):
        key = _find_member(files, m.group(1))
        if key is None:
            return " "                      # 找不到就删掉，不留噪声
        return "\n" + _expand_inputs(files[key], files, depth + 1) + "\n"

    return re.sub(r"\\(?:input|include)\s*\{([^{}]*)\}", sub, tex)


class NotLatexSource(Exception):
    """
    e-print 不是 LaTeX 源码。

    作者只提交 PDF 时，arXiv 的 /e-print/ 就会回一个 PDF。这件事必须**显式
    报错**，不能把二进制当文本返回：那样下游只会看到一段乱码、判成"没有
    方法段"，真实的失败原因就彻底藏起来了（实测有 11 篇栽在这里）。
    本评估刻意不解 PDF，见 README：解析器的错误率不该混进工具质量的指标。
    """


def fetch_tex(aid: str) -> str:
    """下载 e-print，返回**展开 \\input 之后**的主文件正文。

    非 LaTeX 源码（PDF / HTML 错误页 / 其它二进制）一律抛 NotLatexSource。
    """
    os.makedirs(SCRATCH, exist_ok=True)
    cache = os.path.join(SCRATCH, aid.replace("/", "_") + ".tar.gz")
    if os.path.exists(cache) and os.path.getsize(cache) > 0:
        raw = open(cache, "rb").read()
    else:
        raw = _get(EPRINT.format(aid))
        with open(cache, "wb") as fh:
            fh.write(raw)
        time.sleep(DELAY)

    # arXiv 的 e-print 可能是一个 tar.gz，也可能是单个 gzip 的 .tex
    try:
        tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    except tarfile.TarError:
        # 单文件形态没有 \\input 可展开，直接返回
        try:
            return gzip.decompress(raw).decode("utf-8", "replace")
        except OSError:
            head = raw[:400]
            if raw.startswith(b"%PDF"):
                raise NotLatexSource("e-print 是 PDF（作者未提交 LaTeX 源码）")
            if b"<html" in head.lower() or b"<!doctype" in head.lower():
                raise NotLatexSource("e-print 返回的是 HTML（多半是错误页）")
            raise NotLatexSource(f"e-print 既非 tar.gz 也非 gzip（magic={raw[:4]!r}）")

    files: Dict[str, str] = {}
    for m in tf.getmembers():
        if not m.isfile() or not m.name.lower().endswith(".tex"):
            continue
        try:
            files[m.name.replace("\\", "/").lower()] = (
                tf.extractfile(m).read().decode("utf-8", "replace"))
        except Exception:
            continue

    # 主文件 = 含 \documentclass 的**最长**那份。
    # 不能取"第一个含 \documentclass 的"：正文与补充材料往往各有一个
    # \documentclass，tar 成员顺序不保证，取错就抽到 SI 去了。
    if not files:
        return ""
    best_key = max(files, key=lambda k: ("\\documentclass" in files[k],
                                         len(files[k])))
    if "\\documentclass" not in files[best_key]:
        return ""            # 一个都没有：不是 LaTeX 稿，别硬抽
    return _expand_inputs(files[best_key], files)


# -----------------------------------------------------------------------------
def process(aid: str, rec: dict) -> dict:
    if not ARXIV_ID.match(aid):
        # 缓存文件名不像 arXiv 编号：来源不明，不能当作一篇论文计入语料
        return {**rec, "ok": False, "reason": f"not an arXiv ID: {aid!r}"}

    tex = fetch_tex(aid)
    if not tex:
        return {**rec, "ok": False, "reason": "no .tex in e-print"}

    if not split_sections(tex):
        # 少数投稿完全不用 \section（实测 1111.5611v1 通篇只有 \cite/\textbf）。
        # 章节定位对这类论文整体失效，必须**单独报出来**：
        # 混进通用的"没有方法段"里，会让人以为是选取逻辑不准，
        # 而真实原因是这份排版里根本没有可定位的章节。
        return {**rec, "ok": False,
                "reason": "no \\section commands (cannot locate methods)"}

    # 领域闸门放在选章节之前：跑题的检索结果（如核物理的 Green-Kubo 论文）
    # 不该被算作"抓取失败"，而应明确记为"不在语料范围内"，
    # 否则抓取成功率会被这类无关论文拉低，且原因看不出来。
    if not is_md_paper(strip_latex(tex)):
        return {**rec, "ok": False, "reason": "out of scope: not a molecular-simulation paper"}

    title, body, level = pick_methods(tex)
    if not body:
        return {**rec, "ok": False, "reason": "no methods-like section"}

    text = strip_latex(body)
    if len(text) < 400:
        return {**rec, "ok": False,
                "reason": f"section too short after strip ({len(text)} chars)"}

    os.makedirs(SECTIONS, exist_ok=True)
    safe = aid.replace("/", "_")
    path = os.path.join(SECTIONS, safe + ".txt")
    with io.open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# source: arXiv:{aid}\n")
        fh.write(f"# section: {title}\n")
        fh.write(f"# level: {level} "
                 f"({'完整章节' if level == 1 else '子章节，可能不完整'})\n")
        fh.write("# 仅用于本地工具评估，不随提交材料分发\n\n")
        fh.write(text + "\n")

    return {**rec, "ok": True, "section": title, "level": level, "chars": len(text),
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "file": os.path.relpath(path, HERE).replace("\\", "/")}


def _merge_manifest(man: dict, rec: dict) -> dict:
    """把一条新记录并入 manifest（按 id 覆盖），保持 id 有序。"""
    by_id = {r["id"]: r for r in man.get("records", [])}
    by_id[rec["id"]] = rec
    man["records"] = sorted(by_id.values(), key=lambda r: r["id"])
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="抓取 arXiv 方法段用于评估")
    ap.add_argument("--list", action="store_true", help="只搜索，不下载")
    ap.add_argument("--limit", type=int, default=12, help="最多成功抓取多少篇")
    ap.add_argument("--id", action="append", help="指定 arXiv ID（可多次）")
    ap.add_argument("--re-extract", action="store_true",
                    help="不重新下载，用 _scratch/ 里的缓存重新抽取"
                         "（改了抽取逻辑后用这个，避免重复请求 arXiv）")
    args = ap.parse_args(argv)

    if args.re_extract:
        # 以**下载缓存**为准，而不是以 manifest 为准。
        # manifest 是派生结果：它既可能被旧版抽取逻辑写坏（正是要重抽的原因），
        # 也可能因为一次中断而只剩半份。_scratch/ 里实际有什么，
        # 才是"能重抽什么"的唯一真相，重抽的全部意义就是不再碰网络。
        man = (json.load(open(MANIFEST, encoding="utf-8"))
               if os.path.exists(MANIFEST) else {"records": []})
        by_id = {r["id"]: r for r in man.get("records", [])}
        # 缓存文件名是 ID 把 "/" 换成 "_"；用 manifest 反查真实 ID，
        # 避免把 ID 里的下划线（老式编号如 hep_th/9901001）还原错。
        id_of_file = {r["id"].replace("/", "_") + ".tar.gz": r["id"]
                      for r in man.get("records", []) if r.get("id")}

        ids = []
        if os.path.isdir(SCRATCH):
            for fn in sorted(os.listdir(SCRATCH)):
                if not fn.endswith(".tar.gz"):
                    continue
                if os.path.getsize(os.path.join(SCRATCH, fn)) == 0:
                    continue
                aid = id_of_file.get(fn, fn[:-len(".tar.gz")].replace("_", "/"))
                if aid not in ids:
                    ids.append(aid)
        # manifest 里有、但缓存已丢的，保留记录但不重抽（并如实说明）
        lost = [i for i in by_id if i not in ids]
        if not ids:
            print("缓存里没有可重抽的 e-print。先跑一次正常抓取。")
            return 2

        n_ok = 0
        for aid in ids:
            # 只继承"篇目信息"（id/title/published），丢掉上一次抽取的派生字段，
            # 否则重抽失败时旧的字数/章节名会留在记录里，看起来像抽成功了。
            base = {k: v for k, v in
                    (by_id.get(aid) or {"id": aid, "title": "(cached)",
                                        "published": ""}).items()
                    if k not in ("ok", "reason", "section", "level", "chars",
                                 "sha256", "file")}
            try:
                r = process(aid, base)
            except Exception as e:
                r = {**base, "ok": False, "reason": f"{type(e).__name__}: {e}"}
            if r.get("ok"):
                n_ok += 1
                lv = r.get("level")
                print(f"  [ok] {aid:18s} {r['chars']:6d} 字  "
                      f"L{lv} [{r['section'][:44]}]")
            else:
                print(f"  [--] {aid:18s} {r.get('reason','')[:60]}")
            man = _merge_manifest(man, r)
        man["n_ok"] = sum(1 for r in man["records"] if r.get("ok"))
        with io.open(MANIFEST, "w", encoding="utf-8") as fh:
            json.dump(man, fh, ensure_ascii=False, indent=2)
        print(f"\n重抽完成，可用 {n_ok} / {len(ids)} 篇")
        if lost:
            # 不静默丢记录：manifest 里有、缓存里没有的，说明这一篇的
            # 正文已经不在本地了，指标里不能再算它。
            print(f"注意：{len(lost)} 篇在 manifest 里有记录但缓存已不在本地，"
                  f"本次未重抽，也未计入：")
            for i in lost:
                print(f"   - {i}")
        return 0

    if args.id:
        cands = [{"id": i, "title": "(manual)", "published": ""} for i in args.id]
    else:
        cands, seen = [], set()
        for q, n in QUERIES:
            try:
                hits = search(q, n)
            except Exception as e:
                print(f"  ! 查询失败: {q[:50]}… ({e})")
                continue
            for h in hits:
                if h["id"] in seen:
                    continue
                seen.add(h["id"])
                cands.append(h)
            time.sleep(DELAY)
        print(f"共 {len(cands)} 个候选（{len(QUERIES)} 条查询，已去重）\n")

    if args.list:
        for c in cands:
            print(f"  {c['id']:20s} {c['published']}  {c['title'][:78]}")
        return 0

    recs, ok = [], 0
    for c in cands:
        if ok >= args.limit:
            break
        try:
            r = process(c["id"], c)
        except Exception as e:
            r = {**c, "ok": False, "reason": f"{type(e).__name__}: {e}"}
        recs.append(r)
        if r["ok"]:
            ok += 1
            print(f"  [ok] {r['id']:18s} {r['chars']:6d} 字  [{r['section'][:40]}]")
        else:
            print(f"  [--] {r['id']:18s} {r.get('reason','')[:60]}")

    # 与已有 manifest 合并：允许分批抓取而不丢前一批
    old = {}
    if os.path.exists(MANIFEST):
        try:
            old = {r["id"]: r for r in json.load(
                open(MANIFEST, encoding="utf-8")).get("records", [])}
        except Exception:
            old = {}
    for r in recs:
        old[r["id"]] = r
    merged = sorted(old.values(), key=lambda r: r["id"])

    with io.open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump({
            "_note": "arXiv 方法段清单。正文在 sections/，仅本地评估用，"
                     "不随提交材料分发。",
            "_license": "各篇版权归原作者；本清单只记录 arXiv ID 与哈希，"
                        "不含正文。",
            "n_total": len(merged),
            "n_ok": sum(1 for r in merged if r.get("ok")),
            "records": merged,
        }, fh, ensure_ascii=False, indent=2)

    print(f"\n成功 {ok} 篇（本次）；manifest 现有 "
          f"{sum(1 for r in merged if r.get('ok'))} 篇可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
