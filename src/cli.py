# -*- coding: utf-8 -*-
"""
cli.py：端到端流水线

    extract.py       文献方法段  ->  结构化字段
    validate.py      结构化字段  ->  缺失/歧义/一致性/敏感性  -> 评分
    build_inputs.py  结构化字段  ->  可执行输入文件（带来源标注）
    convergence.py   模拟输出    ->  可信度等级

设计上刻意保持"三段松耦合、各自可单独调用"：
真实场景里，很多使用者只想跑其中一段（例如只想要检查单，不需要生成输入文件），
强行串成黑盒反而是负担。

用法：
    python -m src.cli examples/case-01/methods.txt
    python -m src.cli examples/case-01/methods.txt --out out/case-01 --build
    python -m src.cli paper.txt --json

    python -m src.cli --check-lammps 别人写好的.lmp
        不读方法段，直接静态检查一份现成的 LAMMPS 脚本。
           生成器与检查器共用同一套判据：能发现自己生成的文件里的问题，
           也必须能发现别人手写脚本里的同一类问题。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from . import build_inputs, convergence, validate
from .extract import extract as run_extract


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def run_pipeline(text: str, backend: str = "rule",
                 llm_client: Any = None,
                 build: bool = False,
                 outdir: Optional[str] = None) -> Dict[str, Any]:
    paths = validate.default_paths()
    schema = validate.load_yaml(paths["schema"])
    sensitivity = validate.load_yaml(paths["sensitivity"])

    extraction = run_extract(text, schema, backend=backend, llm_client=llm_client)
    validation = validate.validate(extraction, schema, sensitivity, raw_text=text)

    result: Dict[str, Any] = {
        "extraction": extraction,
        "validation": validation,
        "report": validate.render_report(extraction, validation),
    }

    if build:
        target = outdir or "out"
        # 把敏感性归因层的结论传给生成器：本场景下 severity=high 的缺失字段，
        # 意味着"即使生成了输入文件也不构成复现"。判断归知识层，执行归生成器。
        blocking = sorted({f["field"] for f in validation["findings"]
                           if f["kind"] == "impact"
                           and f.get("severity") in ("high", "critical")
                           and f.get("field")})
        b = build_inputs.build_all(extraction["fields"], target, blocking=blocking)
        result["build"] = b
        result["report"] += "\n" + render_build(b, target)

    return result


def render_build(b: Dict[str, Any], outdir: str) -> str:
    L: List[str] = []
    L.append("=" * 74)
    L.append("输入文件生成")
    L.append("=" * 74)
    L.append(f"输出目录: {outdir}")
    for p in b["files"]:
        L.append(f"  生成  {os.path.basename(p)}")
    L.append("")

    prov = b.get("provenance") or {}
    counts: Dict[str, int] = {}
    for v in prov.values():
        counts[v] = counts.get(v, 0) + 1

    L.append("【参数来源】：本节是生成器存在的理由，请勿跳过")
    L.append("  paper    = 文献明确报告，可直接复现")
    L.append("  default  = 文献未报告，工具填入推荐默认值 → 复现结果与原研究必有差异")
    L.append("  builtin  = 生成器固定值（如输出频率），不影响物理")
    L.append("")
    L.append(f"  统计 (GROMACS): paper {counts.get('paper',0)} · "
             f"default {counts.get('default',0)} · "
             f"builtin {counts.get('builtin',0)}")
    L.append("")

    defaults = sorted(k for k, v in prov.items() if v == "default")
    if defaults:
        L.append("  以下参数原研究未报告，由工具填入默认值：")
        for k in defaults:
            L.append(f"      {k}")
        L.append("      → 若要真正复现，这些必须向作者确认，或做敏感性检验。")
        L.append("")

    # LAMMPS 单独统计：两套引擎的默认值不是一回事。
    # 同一个物理量（如长程色散校正）在 .mdp 与 .lmp 里的表达与覆盖面都不同，
    # 混在一张表里会让"哪个默认值属于哪个引擎"变得不可判断，
    # 而默认值恰恰是最需要被逐一确认的东西。
    lprov = b.get("provenance_lammps") or {}
    if lprov:
        lc: Dict[str, int] = {}
        for v in lprov.values():
            lc[v] = lc.get(v, 0) + 1
        L.append(f"  统计 (LAMMPS): paper {lc.get('paper',0)} · "
                 f"default {lc.get('default',0)} · "
                 f"builtin {lc.get('builtin',0)}")
        ldef = sorted(k for k, v in lprov.items() if v == "default")
        if ldef:
            L.append("  LAMMPS 侧的默认假设（同样须逐一确认）：")
            for k in ldef:
                L.append(f"      {k}")
        lunres = sorted(k for k, v in lprov.items() if v == "unresolved")
        if lunres:
            L.append("  LAMMPS 侧必须人工填写（生成器拒绝猜测）：")
            for k in lunres:
                L.append(f"      !! {k}")
        L.append("")

    if b["unresolved"]:
        L.append("【无法默认、必须由文献提供的字段】")
        for k in b["unresolved"]:
            L.append(f"  !! {k}")
        L.append("  → 这些字段缺失时，生成的输入文件不构成对原研究的复现。")
        L.append("")

    # 警告按内容归并：同一条警告在 em/nvt/npt/prod 各出现一次，
    # 逐条列出会把真正重要的信息淹没在重复里。
    if b["warnings"]:
        merged: Dict[str, List[str]] = {}
        for w in b["warnings"]:
            stage, _, msg = w.partition("] ")
            merged.setdefault(msg, []).append(stage.lstrip("["))
        L.append("【生成时给出的警告】")
        for msg, stages in merged.items():
            L.append(f"  - [{','.join(stages)}] {msg}")
        L.append("")

    bad = {s: c for s, c in b["checks"].items() if not c["ok"]}
    if bad:
        L.append("【语法检查】")
        for s, c in bad.items():
            for e in c["errors"]:
                L.append(f"  !! [{s}] {e}")
    else:
        L.append("【语法检查】全部 .mdp 通过（参数名合法、无重复键），"
                 "全部 .lmp 通过（命令名合法、无重复 ID、无冲突恒温、"
                 "无空转 run），openmm_skeleton.py 通过 Python 语法解析")
        L.append("  说明：此项只校验语法与参数名，不等于物理设置正确，"
                 "也不等于可直接运行。")
        L.append("  LAMMPS 侧尤其注意：力场参数不在脚本里（include TODO.forcefield），")
        L.append("  生成器不会伪造 pair_coeff：伪造的力场跑得起来，"
                 "但跑的不是原来那个体系。")

    # 语法检查的警告通道：check_lammps 算出来的警告（如"有 lj/cut 却没有
    # 生效的 mix"）必须在这里打印出来。工具自己犯"静默失效"，
    # 正是本项目要抓的东西。
    # 按内容归并：同一条警告在 em/nvt/npt/prod 各出现一次。
    cwarn: Dict[str, List[str]] = {}
    for s, c in b["checks"].items():
        for w in c.get("warnings") or []:
            cwarn.setdefault(w, []).append(s)
    if cwarn:
        L.append("")
        L.append("【脚本检查的警告】：语法无误，但可能不是你想要的那个物理")
        for w, stages in cwarn.items():
            L.append(f"  !  [{','.join(stages)}] {w}")
    L.append("=" * 74)
    return "\n".join(L)


def render_lammps_check(chk: Dict[str, Any], path: str) -> str:
    """把 check_lammps() 的结果渲染成检查单。约定与主报告一致：!! 错误、! 警告。"""
    L: List[str] = []
    L.append("=" * 74)
    L.append("LAMMPS 脚本静态检查")
    L.append("=" * 74)
    L.append(f"文件    : {path}")
    L.append(f"命令数  : {chk['n_commands']}")
    if chk["thermostats"]:
        L.append(f"恒温 fix: {', '.join(chk['thermostats'])}")
    L.append("")

    if chk["errors"]:
        L.append("【错误】：脚本一定有问题，与是否运行无关")
        for e in chk["errors"]:
            L.append(f"  !! {e}")
        L.append("")

    if chk["warnings"]:
        L.append("【警告】：脚本跑得完，但可能不是你想要的那个物理")
        for w in chk["warnings"]:
            L.append(f"  !  {w}")
        L.append("")

    if chk["ok"] and not chk["warnings"]:
        L.append("未发现已知问题。")
        L.append("")

    L.append("  说明：静态检查，不运行 LAMMPS。ok 只表示无错误，")
    L.append("  不代表物理设置正确；警告项需逐条确认。")
    L.append("=" * 74)
    return "\n".join(L)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="MD 方法段可复现性检查：抽取 -> 校验 -> 生成输入文件")
    ap.add_argument("input", nargs="?",
                    help="方法段文本文件；配合 --check-lammps 时为 .lmp 脚本路径")
    ap.add_argument("--out", default="out", help="输出目录（配合 --build）")
    ap.add_argument("--build", action="store_true",
                    help="生成 GROMACS / LAMMPS / OpenMM 输入文件")
    ap.add_argument("--check-lammps", action="store_true",
                    help="把 input 当作已有的 LAMMPS 脚本做静态检查（不读方法段）")
    ap.add_argument("--backend", default="rule", choices=["rule", "llm", "hybrid"])
    ap.add_argument("--provider", default="anthropic", help="llm/hybrid 时的后端")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出全部结果")
    ap.add_argument("--quiet", action="store_true", help="只输出报告，不打印过程")
    args = ap.parse_args(argv)

    if args.input:
        text = load_text(args.input)
    else:
        ap.error("请提供方法段文本文件路径")

    # 独立入口：复用位置参数，不新增第二个"指定文件"的位置。
    if args.check_lammps:
        chk = build_inputs.check_lammps(text)
        if args.json:
            print(json.dumps(chk, ensure_ascii=False, indent=2))
        else:
            print(render_lammps_check(chk, args.input))
        # 有错误才非零：警告是给人判断的，不该卡住 CI。
        return 1 if chk["errors"] else 0

    llm_client = None
    if args.backend in ("llm", "hybrid"):
        llm_client = None
        try:                                          # pragma: no cover
            from .extract import load_llm_client
            llm_client = load_llm_client(args.provider)
        except Exception as e:                        # noqa: BLE001
            print(f"[警告] 无法加载 LLM 后端（{e}），回退到 rule 模式", file=sys.stderr)
            args.backend = "rule"

    res = run_pipeline(text, backend=args.backend, llm_client=llm_client,
                       build=args.build, outdir=args.out)

    if args.json:
        payload = dict(res)
        payload.pop("report", None)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(res["report"])

    # 退出码：有 critical 未解决即为非零，便于接进 CI
    n_crit = res["validation"]["summary"]["n_critical"]
    return 1 if n_crit else 0


if __name__ == "__main__":
    sys.exit(main())
