# -*- coding: utf-8 -*-
"""
run_demo.py：一条命令跑完全部演示

    python run_demo.py

产出（全部写入 examples/case-01/expected/）：
    01_check_v1.txt      典型写法方法段的检查单
    02_check_v2.txt      同研究补全后的检查单（对照）
    03_build_v1.txt      输入文件生成报告（含参数来源表）
    04_convergence.txt   模拟结果有效性判定（含一个"坏"案例）
    05_corpus.txt        跨文献可比性分析（合成分组，不含真实语料）
    06_eval.txt          在合成标注集上的抽取质量评估
    07_regressions.txt   289 条回归断言（夹具全合成，不依赖语料）
    generated/           实际生成的 GROMACS / LAMMPS / OpenMM 输入文件

设计意图：任何声称"可复现"的工具，自己必须能被一条命令复现。

**这是快照生成器，不是回归测试。** 它把当前输出写进 expected/，不比对
上一次的结果，没有 expected vs actual 的 diff。因此"跑通"只证明
流水线没崩，不证明数值没变。真正的回归靠两个独立的东西：

    eval/score.py                 指标是否退化（三态 F1 / 取值准确率 / 假阳性）
    eval/real/test_regressions.py 289 条断言，不依赖语料，可独立运行

把 expected/ 当基线比对是另一件事，当前没有消费者，故未实现。
"""
from __future__ import annotations

import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED = os.path.join(HERE, "examples", "case-01", "expected")
GEN = os.path.join(EXPECTED, "generated")

CASES = [
    ("01_check_v1.txt", ["-m", "src.cli",
                         os.path.join("examples", "case-01", "methods_v1_typical.txt")]),
    ("02_check_v2.txt", ["-m", "src.cli",
                         os.path.join("examples", "case-01", "methods_v2_complete.txt")]),
    # --out 传相对路径：报告里会原样印出这个路径，传绝对路径就会把
    # 跑命令那台机器的目录结构带进产出（这些产出是要给别人看的）。
    ("03_build_v1.txt", ["-m", "src.cli",
                         os.path.join("examples", "case-01", "methods_v1_typical.txt"),
                         "--build", "--out",
                         "examples/case-01/expected/generated/v1"]),
    ("04_convergence.txt", ["-m", "src.convergence", "--demo"]),
    # 跨文献比可用合成的 demo 分组，eval/real/sections/ 是受版权保护的
    # 论文正文，不属于交付物，演示脚本不能依赖它。
    ("05_corpus.txt", ["-m", "src.corpus", "--demo"]),
    ("06_eval.txt", [os.path.join("eval", "score.py")]),
    # 回归套件夹具全合成、不碰 eval/real/sections/，因此可以随包分发，
    # 评审能在自己机器上重跑，这是整个包里最"可验证"的一件东西。
    ("07_regressions.txt", [os.path.join("eval", "real", "test_regressions.py")]),
]


def main() -> int:
    os.makedirs(EXPECTED, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    for name, cmd in CASES:
        full = [sys.executable] + cmd
        proc = subprocess.run(full, cwd=HERE, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = proc.stdout.decode("utf-8", errors="replace")
        # 非零退出码是**预期行为**：CLI 在存在 critical 问题时返回 1，
        # eval 在出现假阳性或漏报时返回 1。演示脚本不应把它当失败。
        tag = "" if proc.returncode in (0, 1) else "  [异常退出码 %d]" % proc.returncode
        dest = os.path.join(EXPECTED, name)
        with io.open(dest, "w", encoding="utf-8") as fh:
            if tag:
                fh.write(tag.strip() + "\n\n")
            fh.write(out)
        print(f"  {name:24s} {len(out):6d} 字符  退出码 {proc.returncode}{tag}")

    print(f"\n全部产出位于 {EXPECTED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
