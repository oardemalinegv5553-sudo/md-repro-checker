# -*- coding: utf-8 -*-
"""
convergence.py：模拟结果的有效性判定

流水线的最后一段。前两段回答"这篇文献写的参数够不够复现"，
这一段回答"按这些参数跑出来之后，这个数能不能信"。

设计要点（也是与"只报一个数字"的分野）：
  本模块不输出一个裸的 κ 数值，而是输出
  「可信度等级 + 判据明细 + 证据」。理由很直接：
  一个未收敛的 κ 和一个收敛的 κ 在数值上无法区分，
  但在科研决策中的价值截然不同。

三个独立判据，任一不过即降级：
  J1 多窗口一致性  Green-Kubo 的 κ(t) 在平台区应稳定；
                   把统计区切成 N 个窗口，窗口均值的相对标准差即噪声水平。
  J2 平台漂移      窗口均值对时间做线性回归，斜率显著非零说明仍在漂移。
  J3 梯度线性度    NEMD 的 T(z) 必须线性，R² 低说明不在线性响应区。
  J4 能量漂移      NVE 段总能量漂移过大说明积分不稳定。

用法：
  python -m src.convergence --gk data/kappa_t.dat --tprofile data/tprofile.dat
  python -m src.convergence --demo          # 用合成数据演示（含一个"坏"例子）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# 基础工具（纯标准库实现，不引入 numpy：判据本身很简单，没必要加依赖）
# -----------------------------------------------------------------------------


def load_xy(path: str) -> Tuple[List[float], List[float]]:
    """读两列数值数据，忽略 # 注释行与空行。支持空白或逗号分隔。"""
    xs: List[float] = []
    ys: List[float] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                xs.append(float(parts[0]))
                ys.append(float(parts[1]))
            except ValueError:
                continue
    if len(xs) < 4:
        raise ValueError(f"{path}: 有效数据点不足（{len(xs)} 个）")
    return xs, ys


def mean(v: Sequence[float]) -> float:
    return sum(v) / len(v)


def stdev(v: Sequence[float]) -> float:
    if len(v) < 2:
        return 0.0
    m = mean(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def linfit(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float, float]:
    """最小二乘直线拟合，返回 (slope, intercept, r_squared)。"""
    n = len(x)
    mx, my = mean(x), mean(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    if sxx == 0:
        return 0.0, my, 0.0
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((yi - my) ** 2 for yi in y)
    ss_res = sum((yi - (slope * xi + intercept)) ** 2 for xi, yi in zip(x, y))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2


# -----------------------------------------------------------------------------
# J1 + J2：Green-Kubo κ(t) 的多窗口一致性与平台漂移
# -----------------------------------------------------------------------------
def assess_kappa(t: Sequence[float], kappa: Sequence[float],
                 transient_frac: float = 0.30,
                 n_windows: int = 5) -> Dict[str, Any]:
    """
    transient_frac: 前多少比例的关联时间视为未收敛区，剔除后再判定。
                    Green-Kubo 的 κ(t) 前期必然单调上升，把它算进平台
                    是"看起来平滑所以收敛了"这一误判的主要来源。
    """
    n = len(t)
    start = int(n * transient_frac)
    if n - start < n_windows * 2:
        return {"judge": "J1/J2", "verdict": "insufficient_data",
                "detail": f"剔除前 {transient_frac:.0%} 后仅剩 {n - start} 点，不足以分 {n_windows} 窗"}

    tp = t[start:]
    kp = kappa[start:]

    # 分窗
    size = len(kp) // n_windows
    win_means: List[float] = []
    win_centers: List[float] = []
    for i in range(n_windows):
        a = i * size
        b = a + size if i < n_windows - 1 else len(kp)
        win_means.append(mean(kp[a:b]))
        win_centers.append(mean(tp[a:b]))

    m = mean(win_means)
    sd = stdev(win_means)
    rel_std = abs(sd / m) if m else float("inf")

    slope, _, r2 = linfit(win_centers, win_means)
    span = win_centers[-1] - win_centers[0]
    # 归一化漂移：整个统计区间内的相对变化量
    drift = abs(slope * span / m) if m else float("inf")

    if rel_std < 0.05 and drift < 0.05:
        grade, note = "A", "平台稳定，窗口间离散度与漂移均在 5% 以内"
    elif rel_std < 0.10 and drift < 0.10:
        grade, note = "B", "平台基本稳定，建议延长统计时间以收紧误差棒"
    elif rel_std < 0.20 and drift < 0.20:
        grade, note = "C", "统计不足，当前误差棒可能掩盖真实的物理差异"
    else:
        grade, note = "D", "未观察到平台：κ(t) 仍在漂移或噪声主导，该数值不可用于定量比较"

    return {
        "judge": "J1/J2",
        "verdict": grade,
        "kappa_mean": m,
        "kappa_std": sd,
        "rel_std": rel_std,
        "drift_relative": drift,
        "drift_r2": r2,
        "windows": [{"t_center": c, "kappa": v} for c, v in zip(win_centers, win_means)],
        "transient_frac": transient_frac,
        "note": note,
        "evidence": f"剔除前 {transient_frac:.0%} 关联时间后，{n_windows} 个窗口的 "
                    f"κ 均值 {m:.4f}，相对标准差 {rel_std:.1%}，"
                    f"归一化漂移 {drift:.1%}",
    }


# -----------------------------------------------------------------------------
# J3：NEMD 温度梯度线性度
# -----------------------------------------------------------------------------
def assess_temperature_profile(z: Sequence[float], temp: Sequence[float]) -> Dict[str, Any]:
    slope, intercept, r2 = linfit(z, temp)
    # 单调性检查：线性拟合对"V 形"分布也会给出不低的 R²，需单独排除
    diffs = [temp[i + 1] - temp[i] for i in range(len(temp) - 1)]
    n_up = sum(1 for d in diffs if d > 0)
    monotonic = n_up in (0, len(diffs))

    if r2 > 0.99 and monotonic:
        grade, note = "A", "温度剖面高度线性，处于线性响应区"
    elif r2 > 0.98 and monotonic:
        grade, note = "B", "温度剖面基本线性"
    elif r2 > 0.95:
        grade, note = "C", "偏离线性，可能 ΔT 过大或存在界面热阻"
    else:
        grade, note = "D", "非线性显著，κ 依赖于 ΔT，失去材料常数意义"

    return {
        "judge": "J3",
        "verdict": grade,
        "slope_K_per_nm": slope,
        "r2": r2,
        "monotonic": monotonic,
        "note": note,
        "evidence": f"T(z) 线性拟合 R² = {r2:.4f}，单调性 = {monotonic}",
    }


# -----------------------------------------------------------------------------
# J4：NVE 段能量漂移
# -----------------------------------------------------------------------------
def assess_energy_drift(t: Sequence[float], energy: Sequence[float]) -> Dict[str, Any]:
    slope, _, r2 = linfit(t, energy)
    span = t[-1] - t[0]
    e0 = abs(energy[0]) if energy[0] else 1.0
    rel_drift = abs(slope * span / e0)
    per_ns = (rel_drift / span * 1000.0) if span else float("inf")

    if per_ns < 0.01:
        grade, note = "A", "能量守恒良好"
    elif per_ns < 0.05:
        grade, note = "B", "能量漂移可接受"
    elif per_ns < 0.2:
        grade, note = "C", "能量漂移偏大，建议减小时间步或收紧约束容差"
    else:
        grade, note = "D", "能量不守恒，积分不稳定，结果不可用"

    return {
        "judge": "J4",
        "verdict": grade,
        "relative_drift": rel_drift,
        "drift_percent_per_ns": per_ns,
        "note": note,
        "evidence": f"总能量相对漂移 {rel_drift:.3%}（{per_ns:.3f} %/ns）",
    }


# -----------------------------------------------------------------------------
# 汇总
# -----------------------------------------------------------------------------
def overall(judges: List[Dict[str, Any]]) -> Dict[str, Any]:
    order = {"A": 3, "B": 2, "C": 1, "D": 0, "insufficient_data": 0}
    grades = [j["verdict"] for j in judges if j.get("verdict")]
    if not grades:
        return {"grade": "D", "note": "无可用判据"}
    worst = min(grades, key=lambda g: order.get(g, 0))
    notes = {
        "A": "该结果可用于定量比较",
        "B": "该结果可用，误差棒需在引用时一并给出",
        "C": "该结果仅可用于定性趋势，不建议进入定量数据集",
        "D": "该结果不应进入任何数据集；须延长统计或修正设置后重跑",
        "insufficient_data": "数据不足以判定",
    }
    return {
        "grade": worst,
        "judges_passed": sum(1 for g in grades if g in ("A", "B")),
        "judges_total": len(grades),
        "note": notes.get(worst, ""),
    }


# -----------------------------------------------------------------------------
# 合成数据：让 Demo 在没有真实轨迹的情况下也能端到端跑通
#   生成两个案例：一个统计充分（应判 A/B），一个统计不足（应判 D）。
#   数据是合成的，仅用于演示判据行为。
# -----------------------------------------------------------------------------
def make_demo_data(outdir: str) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    # 固定种子，结果可复现。
    # 选这个种子还有一个非物理的理由：合成数据里不应出现与任何真实
    # 实验室数值字形相近的点，交付前的关键词排查会命中它，
    # 而"这只是一个随机数"需要解释，解释成本高于换一个种子。
    rnd = random.Random(20260434)
    paths: Dict[str, str] = {}

    # 两个案例要区分的是**统计是否收敛**，与"数值大小"无关：
    # good 的 κ 落在 0.086，poor 的 κ 落在 0.158，后者看起来更"大"、
    # 更"显著"，但它仍在漂移，因此不可用于比较。这正是本模块要传达的判断。
    #
    # 平台值取 0.085 而非某个"更常见的聚合物 κ 量级"数值，同样是脱敏考虑：
    # 演示数据的平台值不应与任何真实实验值接近到会被怀疑同源。
    PLATEAU = 0.085
    for tag, n, noise, drift in (("good", 400, 0.004, 0.00002),
                                 ("poor", 120, 0.022, 0.00180)):
        t, k = [], []
        for i in range(n):
            ti = 0.5 * (i + 1)                    # ps
            base = PLATEAU + drift * ti           # 再加噪声
            kv = base + rnd.gauss(0, noise)
            t.append(ti)
            k.append(kv)
        p = os.path.join(outdir, f"kappa_t_{tag}.dat")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("# synthetic demo data -- not from any real simulation\n")
            fh.write("# time_ps   kappa_W_per_mK\n")
            for a, b in zip(t, k):
                fh.write(f"{a:.3f}  {b:.6f}\n")
        paths[f"kappa_{tag}"] = p

    # 温度剖面：good 线性，poor 因 ΔT 过大而弯曲
    for tag, curve in (("good", 0.0), ("poor", 0.9)):
        z, T = [], []
        for i in range(41):
            zi = -2.0 + 0.1 * i                   # nm
            base = 300.0 + 25.0 * zi
            T.append(base + curve * 6.0 * (zi ** 2))
            z.append(zi)
        p = os.path.join(outdir, f"tprofile_{tag}.dat")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("# synthetic demo data\n# z_nm   T_K\n")
            for a, b in zip(z, T):
                fh.write(f"{a:.3f}  {b:.4f}\n")
        paths[f"tprofile_{tag}"] = p

    return paths


# -----------------------------------------------------------------------------
def run(gk: Optional[str], tprofile: Optional[str], energy: Optional[str],
        n_windows: int, transient: float) -> Dict[str, Any]:
    judges: List[Dict[str, Any]] = []
    if gk:
        t, k = load_xy(gk)
        judges.append(assess_kappa(t, k, transient_frac=transient, n_windows=n_windows))
    if tprofile:
        z, T = load_xy(tprofile)
        judges.append(assess_temperature_profile(z, T))
    if energy:
        t, E = load_xy(energy)
        judges.append(assess_energy_drift(t, E))
    return {"judges": judges, "overall": overall(judges)}


def render(result: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("=" * 74)
    L.append("模拟结果有效性判定")
    L.append("=" * 74)
    for j in result["judges"]:
        L.append(f"[{j['judge']}]  等级 {j.get('verdict')}")
        if "kappa_mean" in j:
            L.append(f"    κ = {j['kappa_mean']:.4f} ± {j['kappa_std']:.4f}"
                     f"  (相对标准差 {j['rel_std']:.1%})")
            L.append(f"    窗口均值: " +
                     ", ".join(f"{w['kappa']:.4f}" for w in j["windows"]))
        if "r2" in j:
            L.append(f"    线性拟合 R² = {j['r2']:.4f}, 单调 = {j['monotonic']}")
        L.append(f"    {j.get('note', '')}")
        L.append(f"    证据: {j.get('evidence', '')}")
        L.append("")
    o = result["overall"]
    L.append("-" * 74)
    L.append(f"综合等级: {o['grade']}   通过判据 {o['judges_passed']}/{o['judges_total']}")
    L.append(f"结论: {o['note']}")
    L.append("-" * 74)
    return "\n".join(L)


def _shown_path(p: str) -> str:
    """报告里显示路径：在当前目录之下就显示相对路径。

    绝对路径对跑命令的人没有额外信息，却会把"在哪台机器、哪个用户名下
    跑的"带进产出里，而这些产出是要给别人看的。
    """
    try:
        rel = os.path.relpath(p, os.getcwd())
    except ValueError:          # 跨盘符（Windows）时 relpath 会抛
        return p
    return p if rel.startswith(os.pardir) else rel.replace(os.sep, "/")


def main(argv: Optional[List[str]] = None) -> int:
    # 本模块的报告里有 `R²`（U+00B2），而中文版 Windows 的控制台与管道默认
    # 是 cp936，**编不出这个字符**——直接跑 `python -m src.convergence --demo`
    # 会在 print 处抛 UnicodeEncodeError，命令等于不可用。`run_demo.py` 给子进程
    # 加了 PYTHONIOENCODING=utf-8 才一直没暴露这一点。
    # 只改 stdout 的编码，不动文本内容；放在 main() 里而非模块顶层，
    # 是为了 import 本模块（回归套件就这么做）时不碰调用方的 stdout。
    # 与 eval/real/*.py 里的写法一致。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):   # stdout 被换成 StringIO，或不是文本流
        pass

    ap = argparse.ArgumentParser(description="MD 模拟结果有效性判定")
    ap.add_argument("--gk", help="Green-Kubo κ(t) 数据文件（time kappa）")
    ap.add_argument("--tprofile", help="温度剖面数据文件（z T）")
    ap.add_argument("--energy", help="总能量漂移数据文件（time E）")
    ap.add_argument("--n-windows", type=int, default=5)
    ap.add_argument("--transient", type=float, default=0.30,
                    help="剔除的未收敛区比例（默认 0.30）")
    ap.add_argument("--demo", action="store_true", help="生成合成数据并演示")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    args = ap.parse_args(argv)

    if args.demo:
        here = os.path.dirname(os.path.abspath(__file__))
        ddir = os.path.join(os.path.dirname(here), "examples", "case-01", "data")
        paths = make_demo_data(ddir)
        print(f"合成数据已写入 {_shown_path(ddir)}\n（注意：合成数据，非真实模拟结果）\n")
        rc = 0
        for tag in ("good", "poor"):
            print("#" * 74)
            print(f"### 案例: {tag}")
            print("#" * 74)
            res = run(paths[f"kappa_{tag}"], paths[f"tprofile_{tag}"], None,
                      args.n_windows, args.transient)
            print(render(res))
            print()
        return rc

    if not any((args.gk, args.tprofile, args.energy)):
        ap.print_help()
        return 2

    res = run(args.gk, args.tprofile, args.energy, args.n_windows, args.transient)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
