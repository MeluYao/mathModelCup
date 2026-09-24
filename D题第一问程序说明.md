# D 题第一问程序说明

> 本文件是程序运行与结果复现说明，不是数学建模论文正文。模型论证、结果表和图表可作为后续撰写材料，但程序不会代替最终论文定稿。

## 1. 功能范围

本程序针对 D 题第一问完成单服务区货箱组批和无人机机型选择，主要功能包括：

- 读取题目提供的无人机参数、服务区坐标、货箱需求和 DEM 数据；
- 使用 WGS84 椭球测地距离和栅格 `all_touched` 超覆盖算法计算航段参数；
- 计算水平飞行、去返程爬升、飞行时间、地面作业时间和返航 SOC；
- 分别运行 BFD+合并改进、词典序动态规划、集合划分 MILP；
- 计算固定架次数下的时间—能耗 Pareto 前沿；
- 计算返航安全余量临界点及分段稳定区间；
- 对逐箱覆盖、载质量、体积、能量和汇总指标进行独立校验；
- 生成 CSV、PNG、Markdown 分析材料和官方提交 Excel。

默认返航安全余量为 20%。当前默认口径下，词典序动态规划和 MILP 均得到 18 架次，其中 B 型 9 架次、C 型 9 架次。

## 2. 目录与模块

| 路径 | 作用 |
| --- | --- |
| `scripts/run_problem_d_q1.py` | 主入口；运行三种算法并生成完整结果 |
| `scripts/build_q1_submission.mjs` | 将推荐方案写入官方 Excel 模板并渲染预览图 |
| `src/math_model_cup/problem_d_q1.py` | 数据读取、GIS、能耗、候选架次、三种求解算法和约束校验 |
| `src/math_model_cup/problem_d_q1_analysis.py` | Pareto 前沿、安全余量临界点和稳定区间分析 |
| `src/math_model_cup/problem_d_q1_reporting.py` | CSV、图表和分析材料生成 |
| `src/math_model_cup/problem_d_q1_validation.py` | 与候选生成模块分离的独立物理校验 |
| `tests/test_problem_d_q1*.py` | 基础模型、GIS、Pareto、余量、报告、Excel 和独立校验测试 |
| `outputs/problem_d_q1/` | 默认输出目录；属于生成结果，不纳入 Git |

## 3. 输入文件

默认从仓库下的 `D题/` 目录递归查找以下原始文件：

- `调度中心与服务区.xlsx`
- `运输无人机数据.xlsx`
- `物资需求与配送时限.xlsx`
- `镇龙乡及周边30米DEM.tif`
- `山区洪涝灾害下无人机运输与通信协同优化.docx`（用于公式口径审计）
- `结果提交模板.xlsx`（用于生成正式提交表）

程序只读取这些原始资料，不修改或覆盖它们。同名文件必须唯一，否则程序会报错并停止。

## 4. Python 环境

推荐使用 Conda 和 Python 3.11：

```powershell
conda env create -f environment.yml
conda activate math-model-cup
```

主要依赖包括 NumPy、pandas、SciPy、Pillow、Matplotlib、Seaborn、openpyxl 和 pytest。

## 5. 运行方法

### 5.1 使用默认路径

在仓库根目录执行：

```powershell
python scripts/run_problem_d_q1.py
```

默认输入目录为 `D题/`，默认输出目录为 `outputs/problem_d_q1/`。

### 5.2 指定输入和输出目录

```powershell
python scripts/run_problem_d_q1.py `
  --problem-dir "D题" `
  --output-dir "outputs/problem_d_q1"
```

查看命令行帮助：

```powershell
python scripts/run_problem_d_q1.py --help
```

## 6. 生成官方提交 Excel

Python 主流程会先生成推荐方案 `dynamic_programming_trips.csv`。在已配置 `@oai/artifact-tool` 的 Codex 工作区中，执行：

```powershell
node scripts/build_q1_submission.mjs build `
  "D题/结果提交模板.xlsx" `
  "outputs/problem_d_q1/dynamic_programming_trips.csv" `
  "outputs/problem_d_q1/结果提交_Q1.xlsx" `
  "outputs/problem_d_q1/结果提交_Q1_preview.png"
```

脚本只写入模板的 `Q1_单点组批` 工作表，保留其他工作表和原模板文件。输出文件另存为 `outputs/problem_d_q1/结果提交_Q1.xlsx`。

`@oai/artifact-tool` 属于当前 Codex 工作区提供的表格运行时，不在 `environment.yml` 的 Python 依赖中。若在普通 Node.js 环境运行，需要先提供该模块。

## 7. 主要输出

### 7.1 推荐方案和算法比较

- `dynamic_programming_trips.csv`：最终推荐逐架次结果；
- `greedy_trips.csv`：BFD+合并改进结果；
- `milp_trips.csv`：集合划分 MILP 结果；
- `method_comparison.csv`：架次数、能耗、时间、利用率、最优差距和运行时间比较。

### 7.2 GIS 与安全载荷

- `gis_metadata.csv`：GeoTIFF 坐标系、分辨率、NoData、仿射变换和边界；
- `route_geometry.csv`：15 条航段的距离、高程、爬升和下降数据；
- `safe_payloads.csv`：15×3 最大安全载荷；
- `gis_solution_comparison.csv`：严格 GIS 与旧计算口径的方案比较。

### 7.3 多目标与安全余量

- `pareto_front.csv`：固定最少架次数下的时间—能耗 Pareto 前沿；
- `pareto_scenarios.csv`：最少架次、最低能耗、最短时间及增加架次情形；
- `critical_reserve_margins.csv`：候选组批临界安全余量；
- `reserve_stability_intervals.csv`：精确分段稳定区间；
- `reserve_change_events.csv`：区间切换时的载荷和组批变化。

### 7.4 独立校验

- `independent_trip_validation.csv`：逐架次独立能耗与时间复核；
- `representative_trip_validation.csv`：代表架次逐项复核；
- `box_coverage_check.csv`：80 个货箱覆盖检查；
- `constraint_check.csv`：服务区、质量、体积、能量和安全余量检查；
- `summary_consistency_check.csv`：汇总值与逐架次求和一致性检查。

### 7.5 图表

图表位于 `outputs/problem_d_q1/figures/`：

- `dem_routes.png`
- `safe_payload_heatmap.png`
- `demand_vs_capacity.png`
- `pareto_front.png`
- `trip_utilization.png`
- `safe_payload_curves.png`
- `reserve_trip_staircase.png`

## 8. 默认结果核对值

以下数值用于判断程序是否在相同输入和默认 20% 安全余量下成功复现：

| 方法 | 架次数 | B 型 | C 型 | 总能耗（kWh） | 累计作业时间（h） |
| --- | ---: | ---: | ---: | ---: | ---: |
| BFD+合并改进 | 18 | 9 | 9 | 59.530638 | 9.099564 |
| 词典序动态规划 | 18 | 9 | 9 | 59.119904 | 9.096231 |
| 集合划分 MILP | 18 | 9 | 9 | 59.119904 | 9.096231 |

最终推荐使用词典序动态规划方案。MILP 得到相同目标值，用于优化层交叉验证；独立物理校验程序负责验证候选生成之外的能耗和约束计算。

## 9. 测试

只运行 D 题第一问测试：

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests -k problem_d_q1
```

运行完整测试套件：

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

测试代码采用 `src` 目录布局。如果项目尚未安装为 Python 包，需要先在当前 PowerShell 会话中设置上述 `PYTHONPATH`；关闭终端后该设置自动失效。

测试覆盖：

- 能耗公式和量纲计算；
- WGS84 距离、DEM 超覆盖和 GIS 元数据；
- 三种算法及多目标 Pareto 前沿；
- 安全余量临界点和稳定区间；
- 80 个货箱恰好配送一次；
- 质量、体积、能量、SOC 和汇总一致性；
- 官方 Excel 模板内容、公式错误扫描和预览图。

浮点约束采用小容差判断，以吸收二进制浮点和优化求解器误差；CSV 保留高精度，展示表和提交表按字段设置统一小数位。

## 10. 注意事项

- 不要移动或重命名题目原始数据文件，除非同时通过 `--problem-dir` 指向新的完整数据目录；
- `outputs/problem_d_q1/` 会被重复运行更新，正式提交前应重新检查 Excel 和预览图；
- 不要把 `scripts/node_modules/` 加入 Git，它只是本地表格运行时链接；
- 本说明只负责程序复现。最终论文应由参赛者根据赛题要求自行组织、审核和定稿。
