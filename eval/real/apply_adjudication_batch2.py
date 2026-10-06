# -*- coding: utf-8 -*-
"""apply_adjudication_batch2.py：第二批 12 篇的 87 条分歧，逐条裁定写回 adjudication.json

为什么要单独成脚本
------------------
裁定是**判断**，但判断必须可复核、可回滚。把 87 条（含 1 条已退役，见 RETIRED）
连同理由放进一个可重跑的表格里，任何人（包括未来的我）都能一眼看出每一条判了
什么、为什么、以及"换一种读法该改哪一条"。直接手改 JSON 做不到这一点。

裁定标准
--------
判据是 `_label_work/spec.md`（冻结的标注规范）与 `schema/md_method.yaml` 的
字段定义，不是"哪种读法对工具更有利"。

- **tool_wrong**：工具的状态或取值不符合规范对该字段的定义，**或取值来自错误
  的指称对象**。本批有 7 例属后者，这是本轮最值得记的发现，值都在文本里，
  但说的是别的东西：

  | 字段 | 工具报的 | 实际出处 |
  |---|---|---|
  | `2207.13405 system_size` | 54840 atoms | NEP **训练集**的原子数，不是被模拟的硅烯 |
  | `2404.14338v1 software` | LAMMPS | "三种实现…本文用 AMS"这种列举句里**被否定**的那个 |
  | `2404.14338v1 timestep` | 0.05 | Gaussian smearing 的 **width** |
  | `2404.14338v1 ligand_parameters` | resp | "respectively" 的子串 |
  | `1002.0354v1 ligand_parameters` | resp | 同上 |
  | `2604.01940v3 ligand_parameters` | CGenFF + RESP | RESP 来自积分器名 **r-RESPA**，正文从未用 RESP |
  | `2604.01940v3 simulation_length` | 100000 steps | **能量最小化**步数，不是采样时长 |
  | `2602.11098v1 initial_structure` | CHARMM-GUI | 它是**平衡协议**的来源，不是初始结构来源 |

  上表头三条是同一个缺陷：短模式 `RESP` 未加词边界，在 "respectively" /
  "corresponding" / "r-RESPA" 里命中子串。该缺陷已修（`src/extract.py` 的
  `_extract_ligand_params`），并固化进 `test_regressions.py` 第 11 组
  （子串陷阱那一组，`test_substring_traps_beyond_word_boundary`）。
  修复后 `2604.01940v3` 那条分歧消失、裁定退役；`1002.0354v1` 与 `2404.14338v1`
  改报 `missing`（标注为 `not_applicable`），仍是分歧，裁定照旧有效。

- **annotator_wrong**：只在标注员的判读**超出规范字面**时用。本批仅 2 条，是同一
  处边界：spec 把该字段定义为「可滴定**残基**的质子化态」、取值示例全是蛋白残基、
  一致性规则 R-09 也以"含蛋白体系"为前提；而标注员按"可滴定**基团**"给硅醇判了
  present。这两条是本批最可争议的裁定，改回 `tool_wrong` 只需改这一行再重跑。

- 两边都无确证的边界（未声明系综时的恒压器），按 spec 明文
  「**不要**因为"作者没写"就推定为不适用」判 tool_wrong。理由：裁定人即工具作者，
  偏向工具的方向一律不取。

悬空裁定
--------
裁定针对的是**当时那一处分歧**。工具改版后某处已经一致，旧裁定就成了"对着一个不
存在的分歧下的判决"。留着它，评分脚本会把它连同一致的判定一起算，等于凭空多出几条
错判，必须删，且删除理由要写进 `_note`（`--prune-inert` 干这件事）。

删除的前提是**那处分歧不存在了**，而不是"裁定改主意了"。本表不提供改判通道：
要改判就改 VERDICTS 里那一行再重跑，这样理由与判决始终对得上。

悬空裁定分两类存：孤例进 `RETIRED`（逐条写理由），成批出现的进 `RETIRED_BULK`
（一条共享理由 + 键列表）。后者的键**仍然留在 VERDICTS 里**，退役的是
"这次判决还算不算数"，不是"当时判得对不对"，那段判据该留着。

用法
----
    python eval/real/apply_adjudication_batch2.py --check        # 只看差异，不写
    python eval/real/apply_adjudication_batch2.py                # 写回 adjudication.json
    python eval/real/apply_adjudication_batch2.py --prune-inert  # 删掉悬空裁定
"""
from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "_label_work")
ADJ = os.path.join(WORK, "adjudication.json")

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                          # pragma: no cover
    pass

NOTE2 = (
    "\n第二批（后 12 篇）：依 spec.md 逐条裁定后写入，判据与逐条理由见 "
    "apply_adjudication_batch2.py 的 VERDICTS 表（tool_wrong 84 条、annotator_wrong 2 条，"
    "后者均为 protonation_state 的「残基 vs 基团」边界）。另有 1 条已退役（2604.01940v3|"
    "ligand_parameters），工具改版后该处分歧已不存在，见同文件的 RETIRED 表。"
)

# 已退役的裁定：所针对的分歧**因工具改版而消失**，不再写回 adjudication.json。
# 与 VERDICTS 分开存是必须的：留在 VERDICTS 里，重跑本脚本会把它重新写回去，
# 那等于对着一个已不存在的分歧继续罚分。
RETIRED = {
    "2604.01940v3|ligand_parameters": (
        "tool_wrong",
        "原裁定成立：工具把积分器名 r-RESPA 读成 RESP 电荷，与 CGenFF 拼成 "
        "\"CGenFF + RESP\"，又因 charge 与 ff 同时命中而报 present（标注为 ambiguous）。"
        "给 RESP 加词边界后该处工具改报 ambiguous，与标注一致，分歧消失，故退役。"),
    "2207.13405|system_size": (
        "tool_wrong",
        "原裁定成立：工具报的 54840 atoms 出自 \"914 structures with 54840 atoms "
        "in total\"，是 NEP **训练集**的规模，被模拟的硅烯全文未给原子数；"
        "标注按 spec「只给盒子尺寸不给原子数」判 ambiguous（它确实给了 "
        "W=13.3 nm 等尺寸）。"
        "给抽取器加上「原子数必须归属于被模拟的体系」后，该字段改报 "
        "{box_nm: 13.3} + ambiguous，与标注一致，分歧消失，故退役。"
        "改版内容见 src/extract.py 的 _SIZE_OTHER_REFERENT，"
        "并固化进 test_regressions.py 第 25 组。"),
}

# 本轮（压 missing 假阳性）批量退役的裁定：所针对的分歧**因工具改版而消失**。
#
# 与上面 RETIRED 的区别只在规模与记述方式：RETIRED 是逐条写理由的孤例，
# 这里有 40 条、退役原因是**同一个**，所以写成"共享理由 + 键列表"，
# 免得把同一句话抄 40 遍。退役**不等于**改判，VERDICTS 里那 40 条一字未动，
# 它们记的是"当时那处分歧该判谁错"，仍然有效、仍然可复核；这里记的是
# "那处分歧现在没了"。两者在 main() 里的处理也不同：退役的键不写回
# adjudication.json（写回等于对着一个不存在的分歧罚分），但也不从
# VERDICTS 删除（删了就丢了当时的判据）。
#
# 为什么共享理由够用：这 40 条全部由**同一次改版**消掉，逐条复核的方式是
# 重跑 score_real 看它是否仍报 inert（`--prune-inert --check`），
# 而不是读一句话。列表由该命令的输出生成，不是手抄的。
RETIRED_BULK_REASON = (
    "本轮（甲/乙/丙三层）改版后，工具在该处的判定已与标注一致，分歧不存在了。"
    "改版内容：力场族名与报告表打通（force_field）、部分证据层（ensemble/"
    "correlation_window/system_size/initial_structure 等）、抽取器召回扩充"
    "（temperature_bath/simulation_length/heat_flux_method）、闸门修正"
    "（constraints/dispersion_correction/protonation_state）。"
)
RETIRED_BULK = [
    "0706.3514v1|ensemble",
    "0706.3514v1|initial_structure",
    "1002.0354v1|initial_structure",
    "1002.0354v1|system_size",
    "1007.5359v2|force_field",
    "1007.5359v2|heat_flux_method",
    "1007.5359v2|initial_structure",
    "1007.5359v2|temperature_bath",
    "1208.6331v1|ensemble",
    "1208.6331v1|force_field",
    "1208.6331v1|temperature_bath",
    "1208.6331v1|thermostat",
    "1707.05453v1|protonation_state",
    "1805.00269v1|correlation_window",
    "1805.00269v1|force_field",
    "1805.00269v1|simulation_length",
    "1805.00269v1|temperature_bath",
    "1805.00269v1|thermostat",
    "1812.03256v1|constraints",
    "2108.03424v1|ensemble",
    "2110.08914v2|constraints",
    "2110.08914v2|force_field",
    "2110.08914v2|heat_flux_method",
    "2110.08914v2|system_size",
    "2207.13405|correlation_window",
    "2207.13405|ensemble",
    "2207.13405|force_field",
    "2207.13405|simulation_length",
    "2207.13405|temperature_bath",
    "2312.07027v2|dispersion_correction",
    "2312.07027v2|ensemble",
    "2312.07027v2|force_field",
    "2312.07027v2|system_size",
    "2312.07027v2|temperature_bath",
    "2404.14338v1|dispersion_correction",
    "2404.14338v1|initial_structure",
    "2506.04835v1|force_field",
    "2602.11098v1|protonation_state",
    "2602.17595v1|ligand_parameters",
    "2605.16123v1|system_size",
]

# key -> (verdict, reason)
VERDICTS = {
    # ---------------- 0706.3514v1（无定形 SiO2/水界面）----------------
    "0706.3514v1|protonation_state": ("annotator_wrong",
        "该字段规范为「可滴定残基的质子化态」，示例与 R-09 均以蛋白为前提；硅醇是表面基团不是残基"),
    "0706.3514v1|ensemble": ("tool_wrong",
        "只写了 Nose-Hoover 恒温（T=300 K）而未写体积/压力是否受控，系综不完整 → ambiguous，工具只认 NVT/NPT 字样"),
    "0706.3514v1|system_size": ("tool_wrong",
        "给了界面面积与 8500 个水分子、未给总原子数与 z 向盒长，正落 spec「只给盒子尺寸不给原子数 → ambiguity」"),
    "0706.3514v1|initial_structure": ("tool_wrong",
        "无定形 SiO2 的制备流程写得很具体：α-石英 → 2500 K → 淬火 300 K → 造两个自由表面 → 退火至缺陷密度 4.0-5.0 /nm²"),

    # ---------------- 1002.0354v1（晶体 SiO2 表面 + 水）----------------
    "1002.0354v1|protonation_state": ("annotator_wrong",
        "同上：晶体二氧化硅体系里没有残基，该字段按规范字面不适用"),
    "1002.0354v1|ligand_parameters": ("tool_wrong",
        "体系无配体、无非标准残基 → not_applicable；工具曾因 \"respectively\" 的子串报 "
        "ambiguous，加词边界后改报 missing，仍不是 not_applicable"),
    "1002.0354v1|thermostat": ("tool_wrong",
        "只写了轨迹在 T=425 K 恒温，类型与耦合常数都没给 → 提到了但不足以据以行动"),
    "1002.0354v1|system_size": ("tool_wrong",
        "盒尺寸 10.17×10.17×26 Å³ 与各体系水分子数都给了，但无总原子数 → spec 明文判 ambiguity"),
    "1002.0354v1|initial_structure": ("tool_wrong",
        "初始构象来源明确：0 K DFT/PBE 优化的晶体片层 + GCMC 定水量 + 表面类型（羟基化/重构）"),

    # ---------------- 1603.06347v2（强子气体状态方程，非 MD）----------------
    "1603.06347v2|software": ("tool_wrong", "UrQMD 具名而无版本，按 spec「只给软件名不给版本 → ambiguity」"),
    "1603.06347v2|force_field": ("tool_wrong", "强子气体不存在经典 MM 参数集，该字段不适用"),
    "1603.06347v2|electrostatics": ("tool_wrong", "无经典静电场求解对象（Ewald/PME/RF 都谈不上）"),
    "1603.06347v2|constraints": ("tool_wrong", "把强子当无内部结构的粒子，无键长约束对象"),
    "1603.06347v2|timestep": ("tool_wrong", "统计模型 + 截面输运，无经典力场的时间积分"),
    "1603.06347v2|thermostat": ("tool_wrong", "T 是热力学参量而非被热浴固定的受控量，无恒温器可言"),
    "1603.06347v2|ensemble": ("tool_wrong", "文本明写 grand canonical ensemble of non-interacting hadrons"),
    "1603.06347v2|initial_structure": ("tool_wrong", "不存在分子初始构象这一对象"),
    "1603.06347v2|dispersion_correction": ("tool_wrong", "无 LJ 截断，谈不上色散尾校正"),
    "1603.06347v2|heat_flux_method": ("tool_wrong", "不做热输运计算，按简报规则三字段判不适用"),
    "1603.06347v2|correlation_window": ("tool_wrong", "无 Green-Kubo 类计算，无关联窗口可言"),

    # ---------------- 1805.00269v1（硅/硅烯/硅纳米线，Tersoff/SW）----------------
    "1805.00269v1|software": ("tool_wrong", "GPUMD 具名而无版本，按 spec 判 ambiguity"),
    "1805.00269v1|force_field": ("tool_wrong",
        "三种体系的势函数族与参数化出处都点了名（Tersoff 原始参数化、Zhang et al. 的 SW、SW 原始参数化）"),
    "1805.00269v1|thermostat": ("tool_wrong",
        "NEMD 冷热源明写 Nosé-Hoover chain；语料把重音字母拆成 \"Nos ' e -Hoover\"，工具没认出来（语料缺陷，非工具判据错）"),
    "1805.00269v1|simulation_length": ("tool_wrong", "EMD 20 ns × 50 独立样本、NEMD 10 ns × 5 独立样本，时长与重复成对给出"),
    "1805.00269v1|initial_structure": ("tool_wrong",
        "给了胞搭建与晶向取向，但晶体结构类型、晶格常数、纳米线表面是否钝化/重构未写 → ambiguity"),
    "1805.00269v1|temperature_bath": ("tool_wrong", "热源 330 K / 冷源 270 K 明写"),
    "1805.00269v1|correlation_window": ("tool_wrong",
        "EMD 侧给了 20 ns 采样与 50 个独立样本，但 GK 积分上限与收敛判据没给 → ambiguity"),

    # ---------------- 2108.03424v1（LR-FPUT 一维晶格）----------------
    "2108.03424v1|force_field": ("tool_wrong", "解析型长程非谐弹簧势，无原子类型/键角参数集"),
    "2108.03424v1|electrostatics": ("tool_wrong", "哈密顿量里无电荷、无库仑项"),
    "2108.03424v1|constraints": ("tool_wrong", "键是连续弹簧势，无刚性键长需要约束"),
    "2108.03424v1|thermostat": ("tool_wrong", "微正则孤立环上做 RNEMD，恒温器是被刻意排除的对象"),
    "2108.03424v1|ensemble": ("tool_wrong", "明写 microcanonical system (isolated ring)，并有 usual Hamiltonian dynamics 佐证"),
    "2108.03424v1|system_size": ("tool_wrong", "明给 128 ≤ N ≤ 16384；一维定长晶格无盒长可言，粒子数即规模"),
    "2108.03424v1|initial_structure": ("tool_wrong", "只说 \"Starting from an equilibrated system\"，如何制备/平衡多久未写 → ambiguity"),
    "2108.03424v1|dispersion_correction": ("tool_wrong", "无任何 r^-6 色散相互作用"),
    "2108.03424v1|temperature_bath": ("tool_wrong", "两 slab 温度 T_1=1.1、T_2=0.9（约化单位）给出，正是冷热浴设置"),

    # ---------------- 2207.13405（硅烯，NEP，EMD/NEMD）----------------
    "2207.13405|software": ("tool_wrong", "gpumd 与 mlip 都具名而无版本，按 spec 判 ambiguity"),
    "2207.13405|force_field": ("tool_wrong", "只给框架名 NEP、超参数推给参考文献，无版本/描述符配置 → ambiguity"),
    "2207.13405|barostat": ("tool_wrong",
        "写了 \"equilibrate ... in the isothermal ensemble at 300 K and zero pressure\"，控压已声明而恒压器未给，正落 spec 的 critical 缺失"),
    "2207.13405|ensemble": ("tool_wrong", "平衡段等温系综、EMD 生产段 microcanonical，各阶段都写清了"),
    "2207.13405|simulation_length": ("tool_wrong", "EMD 5 ns × 30 次、NEMD 10 ns × 3 次，成对给出"),
    "2207.13405|system_size": ("tool_wrong",
        "工具报的 54840 atoms 是 NEP **训练集**的原子数（\"914 structures with 54840 atoms\"），被模拟的硅烯体系全文未给原子数"),
    "2207.13405|temperature_bath": ("tool_wrong", "冷热浴 T±ΔT/2 与 ΔT=20 K 明写，足以判断在线性响应区"),
    "2207.13405|correlation_window": ("tool_wrong",
        "NEMD 给了取后半段 5 ns 的稳态窗口，但 EMD/GK 一侧的关联时间与积分上限完全没给 → ambiguity"),

    # ---------------- 2404.14338v1（ReaxFF 参数化，无 MD）----------------
    "2404.14338v1|software": ("tool_wrong",
        "工具在\"stand-alone / AMS / LAMMPS 三种实现…本文用 AMS\"这类列举句里挑了**被否定**的 LAMMPS；正文给了 AMS 2023.101 与 VASP 5.4"),
    "2404.14338v1|ligand_parameters": ("tool_wrong", "体系无配体 → not_applicable；工具曾因 \"respectively\"/\"corresponding\" 的子串报 ambiguous，加词边界后改报 missing，仍不是 not_applicable"),
    "2404.14338v1|electrostatics": ("tool_wrong",
        "E_charge 与可变电荷写了（说明静电项存在），但电荷平衡方法、屏蔽参数、库仑截断都没给 → ambiguity"),
    "2404.14338v1|timestep": ("tool_wrong",
        "工具报的 0.05 是 Gaussian smearing 的 width；且全篇是静态几何优化，不存在时间步"),
    "2404.14338v1|thermostat": ("tool_wrong", "无 MD，无温度控制需求（0 K 势能面上的优化）"),
    "2404.14338v1|ensemble": ("tool_wrong", "无系综采样：DFT 几何优化 + 静态性质抽取 + CMA-ES 参数拟合"),
    "2404.14338v1|simulation_length": ("tool_wrong", "无 MD 时长；40 次重复是参数优化的独立重复，不是采样重复"),
    "2404.14338v1|initial_structure": ("tool_wrong", "训练集五组结构与晶面清单、验证集（Raybaud et al./NOMAD）及其 VASP 设置都点了名"),
    "2404.14338v1|dispersion_correction": ("tool_wrong", "明写 DFT 参考计算施加了具名的后验密度依赖色散校正 dDsC"),

    # ---------------- 2506.04835v1 / 2509.00937v1 / 2602.17595v1：恒压器边界 ----------------
    "2506.04835v1|barostat": ("tool_wrong",
        "体系含液态水、系综与控压都未声明；spec 明文禁止由\"作者没写\"推定不适用。标注员自陈本条把握不大"),
    "2509.00937v1|barostat": ("tool_wrong",
        "未给恒压器也未说明是否控压。本批最弱的一条：标注员未给正面理由，但\"不得由沉默推定 NA\"这条 spec 规则对工具同样成立"),
    "2602.17595v1|barostat": ("tool_wrong",
        "体相无定形聚合物电解质未提恒压器；若全程 NVT 应另判，但文本既未声明 NVT 也未声明 NPT，工具不能据此断言不适用"),

    # ---------------- 2602.11098v1（POPC 双层 + 乙醇 + 丙氨酸二肽 + Martini）----------------
    "2602.11098v1|protonation_state": ("tool_wrong",
        "全体组分无 His/Asp/Glu/Cys/Lys，POPC 在生理 pH 无质子化态歧义 → 该字段确不适用"),
    "2602.11098v1|ligand_parameters": ("tool_wrong", "体系含非标准小分子（乙醇），且明写用 CGenFF 参数化"),
    "2602.11098v1|barostat": ("tool_wrong", "C-rescale 具名，CG 体系给了半各向同性与 z 压缩率为 0、二肽给了 tau_P=2.0 ps"),
    "2602.11098v1|system_size": ("tool_wrong",
        "三个子体系分别给了可重建规模的定量信息：CG 给盒三边、两个全原子体系给精确组成（100 脂质+9300 水+1 乙醇；622 水+一对离子）"),
    "2602.11098v1|initial_structure": ("tool_wrong",
        "工具报的 CHARMM-GUI 修饰的是\"标准六步平衡协议\"，不是初始结构来源；初始坐标来源与二肽起始构象均未交代"),

    # ---------------- 2603.28012v1（离子电导，综述式子章节）----------------
    "2603.28012v1|electrostatics": ("tool_wrong",
        "本段是泛论（DFT/经验势/MLIP 并列），未认定任何单一力场族；工具被\"especially for ab initio MD (AIMD)\"这类泛论句误导，这是能力表引入的新误报"),
    "2603.28012v1|dispersion_correction": ("tool_wrong", "同上：段内未认定色散项存在与否，未给任何长程色散校正"),

    # ---------------- 2604.01940v3（NAMD，六烷/水，r-RESPA）----------------
    # ligand_parameters 一条已退役：RESP 加词边界后工具改报 ambiguous，与标注一致。
    # 原判据留在 RETIRED 表里，此处不重复。
    "2604.01940v3|timestep": ("tool_wrong",
        "文本给了 base time step = 2（单位被语料剥掉），按 spec 对子参数缺失的惯例应为 present；工具因缺单位改判 ambiguous"),
    "2604.01940v3|barostat": ("tool_wrong", "Langevin piston 具名，relaxation time 与 oscillation period 两个耦合参数都给了"),
    "2604.01940v3|simulation_length": ("tool_wrong",
        "工具报的 100000 steps 是**能量最小化**步数；生产/ABF 采样时长与重复次数全段未给"),

    # ---------------- cond-mat/0701254v1（有限力程粒子模型）----------------
    "cond-mat/0701254v1|electrostatics": ("tool_wrong", "全段无带电相互作用，力为有限力程 + 最小镜像，无静电梯度"),
    "cond-mat/0701254v1|constraints": ("tool_wrong", "明写牛顿方程（只为周期性作修正），无键长约束算法"),
    "cond-mat/0701254v1|thermostat": ("tool_wrong", "输运系数的动力学不含恒温器（加了会压制涨落）"),
    "cond-mat/0701254v1|ensemble": ("tool_wrong", "牛顿方程 + 固定盒子可推知 NVE，但全段无系综字样 → spec 要求照抄原文，判 ambiguity"),
    "cond-mat/0701254v1|dispersion_correction": ("tool_wrong", "未确立体系含 LJ 类长程色散，谈不上截断尾校正"),

    # ---------------- cond-mat/9910397v1（LJ 氩，非平衡热扩散）----------------
    "cond-mat/9910397v1|force_field": ("tool_wrong", "本体系\"力场\"即 LJ 势本身：给了势类型、Lorentz-Berthelot 混合规则与参数来源（Table I）"),
    "cond-mat/9910397v1|electrostatics": ("tool_wrong", "中性单原子 LJ 流体，通篇无电荷/偶极/库仑项"),
    "cond-mat/9910397v1|constraints": ("tool_wrong", "LJ 单原子粒子之间没有化学键，无约束对象"),
    "cond-mat/9910397v1|timestep": ("tool_wrong",
        "时间步 t*=4.66×10⁻³ 明写；语料把指数拆成 \". 10^ -3\"，工具只读到 4.66 并改判 ambiguous（语料缺陷）"),
    "cond-mat/9910397v1|ensemble": ("tool_wrong", "体积在模拟期间恒定 + 平均温度由 Berendsen 维持，N/V/T 三者都定了"),
    "cond-mat/9910397v1|system_size": ("tool_wrong", "1500 个 LJ 原子 + 正交盒 L*×L*×3L*（另给密度），不落 spec 的\"只给盒子尺寸\"情形"),
    "cond-mat/9910397v1|dispersion_correction": ("tool_wrong",
        "给了截断半径 r_c=2.5σ₁ 与 switching range 0.1σ_m，但未说明长程色散是否/如何校正 → ambiguity"),
    "cond-mat/9910397v1|heat_flux_method": ("tool_wrong", "温度梯度驱动的非平衡模拟，热流方法对本体系适用而本段未述（细节在 \"see below\" 的 slab 部分）"),
    "cond-mat/9910397v1|temperature_bath": ("tool_wrong", "同上：冷热两端温度与 ΔT 未给，属适用而未述"),
    "cond-mat/9910397v1|correlation_window": ("tool_wrong", "同上：本段只给输出间隔与运行步数，无统计窗口/收敛判据"),

    # ---------------- nlin/0210052v1（恒温驱动振子，Lyapunov 谱）----------------
    "nlin/0210052v1|force_field": ("tool_wrong", "非分子体系，无 AMBER/CHARMM 式经验力场"),
    "nlin/0210052v1|electrostatics": ("tool_wrong", "恒温驱动振子方程与动力学矩阵 D 都不含静电项"),
    "nlin/0210052v1|constraints": ("tool_wrong",
        "词面陷阱：文中的 constraints 指切空间偏移向量的正交归一化/定长约束（Lyapunov 指数的数学装置），不是 LINCS/SHAKE/SETTLE"),
    "nlin/0210052v1|thermostat": ("tool_wrong",
        "恒温这件事写了（相空间含摩擦系数 ζ、单位弛豫时间），但算法类型未指名、耦合常数只是无量纲单位值 → ambiguity"),
    "nlin/0210052v1|dispersion_correction": ("tool_wrong", "模型无色散/范德华项，也无截断半径"),
}


def prune_inert(check: bool) -> int:
    """删掉悬空裁定：评分脚本说"该对已一致"的那些 key。

    谁算悬空由 score_real 说了算（它才知道工具现在抽出什么），本函数不自己判，
    否则就成了"裁定脚本自己认定自己没过期"。
    """
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(HERE))
    import score_real as S

    d = json.load(io.open(ADJ, encoding="utf-8"))
    r = S.run_all(S.load_labels(), adjudication={k: v for k, v in d.items() if k != "_note"})
    inert = r.get("inert_verdicts") or []
    if not inert:
        print("没有悬空裁定。")
        return 0

    keys = [f"{it['id']}|{it['field']}" for it in inert]
    for it in inert:
        print(f"  - {it['id']:24s} {it['field']:22s} 原裁定 {it['verdict']}")

    if check:
        print(f"[--check] 共 {len(keys)} 条，未写文件。")
        return 0

    for k in keys:
        d.pop(k, None)
    n_bulk = len([k for k in keys if k in set(RETIRED_BULK)])
    bulk_note = (f"其中 {n_bulk} 条属批量退役，改版内容见 RETIRED_BULK_REASON；" if n_bulk else "")
    note = ("\n删除悬空裁定 %d 条（工具改版后该处已一致，旧裁定对着一个不存在的分歧）：%s。\n"
            "%s删除理由：这些裁定所针对的分歧已消失。\n"
            "被删的是「这次判决还算不算数」，VERDICTS 里那一条判据仍然留着。"
            % (len(keys), "、".join(sorted(keys)), bulk_note))
    d["_note"] = (d.get("_note") or "") + note
    io.open(ADJ, "w", encoding="utf-8").write(
        json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(f"已删 {len(keys)} 条，{ADJ} 余 {len(d) - 1} 条裁定")
    return 0


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    check = "--check" in argv

    if "--prune-inert" in argv:
        return prune_inert(check)

    d = json.load(io.open(ADJ, encoding="utf-8"))
    before = {k: v for k, v in d.items() if k != "_note"}

    # 批量退役的键不参与写回，也不参与冲突检查，它们即将被撤下，
    # 此刻文件里是什么值都不影响结果。
    bulk = set(RETIRED_BULK)
    effective = {k: v for k, v in VERDICTS.items() if k not in bulk}

    add, clash = {}, []
    for key, (verdict, _why) in effective.items():
        if key in before:
            if before[key] != verdict:
                clash.append((key, before[key], verdict))
        else:
            add[key] = verdict

    # 已退役的裁定若还在文件里（例如从旧备份恢复），重跑时一并撤掉，
    # 否则它会以"对着已消失的分歧罚分"的形式留在指标里。
    retire = [k for k in list(RETIRED) + list(RETIRED_BULK) if k in before]

    n_tw = sum(1 for v, _ in effective.values() if v == "tool_wrong")
    n_aw = sum(1 for v, _ in effective.values() if v == "annotator_wrong")
    print(f"有效裁定 {len(effective)} 条：tool_wrong {n_tw}，annotator_wrong {n_aw}")
    print(f"另有已退役 {len(RETIRED)} 条（逐条带理由）"
          f" + {len(RETIRED_BULK)} 条（本轮批量，理由共享）")
    print(f"其中已在文件中（不重复写）{len(effective) - len(add)} 条，新增 {len(add)} 条")
    for key in retire:
        print(f"  - 撤下已退役 {key}")
    if len(retire) > 6:
        print(f"    （共 {len(retire)} 条；批量退役的共享理由见 RETIRED_BULK_REASON）")
    for key, old, new in clash:
        print(f"  !! 冲突 {key}: 文件里是 {old}，本表要写 {new}")
    if clash:
        return 2

    if check:
        for key in sorted(add):
            print(f"  + {key:44s} {add[key]}")
        print("[--check] 未写文件。")
        return 0

    for key in retire:
        d.pop(key, None)
    d.update(add)
    if NOTE2 not in (d.get("_note") or ""):
        d["_note"] = (d.get("_note") or "") + NOTE2
    io.open(ADJ, "w", encoding="utf-8").write(
        json.dumps(d, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    print(f"已写 {ADJ}（共 {len(d) - 1} 条裁定）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
