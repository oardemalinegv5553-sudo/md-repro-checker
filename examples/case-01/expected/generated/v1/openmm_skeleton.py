# -*- coding: utf-8 -*-
"""
由 md-repro-checker 自动生成：OpenMM 最小骨架

逐项来源（与 .mdp 生成器同一套标注规则）：
  力场    : GAFF2   [paper]
  水模型  : 【不适用】   [not_applicable]
  系综    : NPT
  温度    : 300 K      [文献冷热浴平均]
  时间步  : 2.0 fs           [文献报告]
  截断    : 1.0 nm          [默认值：文献未报告截断半径]
  约束    : LINCS

凡标 TODO 或 !! 处，均不得在未确认的情况下运行。
"""
from openmm.app import *
from openmm import *
from openmm.unit import *

# TODO: 用文献报告的初始结构替换（来源: 需人工确认）
pdb = PDBFile("input.pdb")
# !! 文献未报告体系尺寸：盒子大小未定义，无法开始复现

# 力场选择：必须与文献完全一致，否则结果不可比
forcefield = ForceField(
    "TODO.xml",   # 文献报告 GAFF2，OpenMM 不自带其 XML；需先经 antechamber / openmmforcefields 生成参数文件
)

modeller = Modeller(pdb.topology, pdb.positions)
# TODO: 补齐体系尺寸
# 文献为无溶剂体系（真空 / 隐式溶剂 / 干态熔体），不添加溶剂。
# 注意：不添加溶剂时不存在水模型，createSystem 不应传入水模型 XML。

system = forcefield.createSystem(
    modeller.topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1.0*nanometer,
    constraints=HBonds,
    ewaldErrorTolerance=0.0005,
)

system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))  # 文献报告 Parrinello-Rahman，OpenMM 中以 MonteCarloBarostat 等价实现

# 文献用 Berendsen 恒温器：该恒温器压制涨落，
# 若原研究涉及扩散系数 / 热导率等输运性质，其结果存在已知系统偏差。
# OpenMM 中以 Langevin 实现，涨落行为与原研究不同。
integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 2.0*femtoseconds)
simulation = Simulation(modeller.topology, system, integrator)
simulation.context.setPositions(modeller.positions)
simulation.minimizeEnergy()

# 生产段
# 20.0 ns（文献报告）；!! 文献未报告重复次数，以下仅为单条轨迹
simulation.step(10000000)
