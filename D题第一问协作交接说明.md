# D 题第一问协作交接说明

> 用途：供团队成员了解当前提交内容、复现实验、复核结果和继续协作。本文是工程交接说明，不是论文正文。

## 1. 当前状态

D 题第一问的程序、测试、计算结果、图表和官方提交 Excel 已推送到远端 `main`。

远端 D 题基线提交：

| 提交 | 内容 |
| --- | --- |
| `dd43eb2` | D 题第一问完整程序与测试 |
| `763f37e` | 程序运行与结果复现说明 |
| `264aee7` | CSV、Markdown、PNG 和官方提交 Excel 等结果文件 |

上述远端提交不包含 A 题文件。原始赛题资料仍受 `.gitignore` 保护，没有上传或修改。

## 2. 当前核心结论

默认采用 20% 返航安全余量。词典序动态规划和集合划分 MILP 得到相同最优结果：

- 最少架次数：18；
- B 型无人机：9 架次；
- C 型无人机：9 架次；
- 总能耗：59.119904 kWh；
- 累计作业时间：9.096231 h；
- 80 个货箱均恰好配送一次；
- 不存在跨服务区组批；
- 18 个架次的质量、体积、能量和返航 SOC 约束全部通过。

BFD+合并改进同样得到 18 架次，但总能耗为 59.530638 kWh，累计作业时间为 9.099564 h。因此当前推荐词典序动态规划方案，并使用 MILP 进行优化层交叉验证。

## 3. 文件入口

### 程序

- 主入口：[`scripts/run_problem_d_q1.py`](scripts/run_problem_d_q1.py)
- 核心模型：[`src/math_model_cup/problem_d_q1.py`](src/math_model_cup/problem_d_q1.py)
- Pareto 与安全余量分析：[`src/math_model_cup/problem_d_q1_analysis.py`](src/math_model_cup/problem_d_q1_analysis.py)
- 报告和图表生成：[`src/math_model_cup/problem_d_q1_reporting.py`](src/math_model_cup/problem_d_q1_reporting.py)
- 独立物理校验：[`src/math_model_cup/problem_d_q1_validation.py`](src/math_model_cup/problem_d_q1_validation.py)
- Excel 写入脚本：[`scripts/build_q1_submission.mjs`](scripts/build_q1_submission.mjs)

### 说明与结果

- 详细运行说明：[`D题第一问程序说明.md`](D题第一问程序说明.md)
- 成果索引：[`outputs/problem_d_q1/完整成果索引.md`](outputs/problem_d_q1/完整成果索引.md)
- 三种算法比较：[`outputs/problem_d_q1/method_comparison.csv`](outputs/problem_d_q1/method_comparison.csv)
- 推荐逐架次方案：[`outputs/problem_d_q1/dynamic_programming_trips.csv`](outputs/problem_d_q1/dynamic_programming_trips.csv)
- 官方提交文件：[`outputs/problem_d_q1/结果提交_Q1.xlsx`](outputs/problem_d_q1/结果提交_Q1.xlsx)
- 官方提交预览：[`outputs/problem_d_q1/结果提交_Q1_preview.png`](outputs/problem_d_q1/结果提交_Q1_preview.png)
- 逐箱覆盖检查：[`outputs/problem_d_q1/box_coverage_check.csv`](outputs/problem_d_q1/box_coverage_check.csv)
- 约束检查：[`outputs/problem_d_q1/constraint_check.csv`](outputs/problem_d_q1/constraint_check.csv)
- 汇总一致性检查：[`outputs/problem_d_q1/summary_consistency_check.csv`](outputs/problem_d_q1/summary_consistency_check.csv)

`outputs/problem_d_q1/论文_问题一.md` 是自动生成的建模材料草稿，不是最终论文正文。团队可引用其中的公式、表格和分析线索，但应自行审核和组织最终论文。

## 4. 本地复现

原始数据没有上传到 Git。复现前应将题目提供的完整 `D题/` 目录放在仓库根目录，至少包含：

- `调度中心与服务区.xlsx`
- `运输无人机数据.xlsx`
- `物资需求与配送时限.xlsx`
- `镇龙乡及周边30米DEM.tif`
- `山区洪涝灾害下无人机运输与通信协同优化.docx`
- `结果提交模板.xlsx`

创建环境并运行：

```powershell
conda env create -f environment.yml
conda activate math-model-cup
python scripts/run_problem_d_q1.py
```

程序默认读取 `D题/`，并更新 `outputs/problem_d_q1/`。正式更新结果前建议保留旧输出，运行后检查 Git 差异，确认变化来自预期的模型或数据调整。

## 5. 测试与验证

仓库使用 `src` 目录布局。未安装为 Python 包时，在 PowerShell 中执行：

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

远端发布前的隔离环境结果为：

- 24 项通过；
- 4 项条件性跳过；
- 2 条第三方库弃用警告；
- 0 项失败。

4 项跳过的原因是隔离环境未配置 `D_PROBLEM_DIR`，并且未配置提交工作簿路径。包含原始 D 题资料和表格运行时的完整工作环境此前已完成 28 项测试验证。

结果文件的独立检查证据：

- `box_coverage_check.csv`：80 行货箱记录；
- `constraint_check.csv`：18 个架次全部约束为 `True`；
- `summary_consistency_check.csv`：货箱数、总能耗、累计时间和逐架次约束均通过；
- `结果提交_Q1.xlsx.inspect.ndjson`：保存后工作簿内容和格式检查记录；
- `结果提交_Q1_preview.png`：提交工作表视觉预览。

## 6. 当前计算口径

当前主口径为：

- 水平能耗 = 电池可用能量 × 水平距离 ÷ 当前载荷等效航程；
- 爬升能耗 = 起飞质量 × 9.81 × 爬升高度 ÷（3.6×10⁶ × 爬升效率）；
- 起飞质量 = 空载总质量 + 当前载荷；
- 去程携带本架次货物，返程载荷为 0；
- 下降附加能耗为 0；
- 服务区作业高度为当地地面海拔 + 30 m；
- 距离采用 WGS84 椭球测地线；
- 沿线高程采用 DEM 栅格 `all_touched` 超覆盖结果。

题面没有明确给出水平能耗和爬升能耗子公式，因此能耗绝对值仍依赖上述补充解释。两种合理航程能耗解释下，18 架次以及 B 型 9 架次、C 型 9 架次的结论保持不变；具体能耗值和最低返航 SOC 会随解释变化。

## 7. 协作约定

后续修改请遵循以下原则：

1. 不修改或覆盖原始 DOCX、XLSX 和 DEM 文件。
2. 不跨服务区组批，每个货箱必须恰好出现一次。
3. 修改能耗、距离、高程或安全余量口径时，同步更新测试和全部输出结果。
4. 修改候选生成逻辑时，同时运行独立物理校验，不能只比较 DP 与 MILP。
5. 修改推荐方案后，重新生成并目视检查 `结果提交_Q1.xlsx`。
6. CSV 保留计算精度；图表和提交 Excel 只统一显示小数位，不提前舍入底层数据。
7. `scripts/node_modules/` 是本地运行时链接，不要提交。
8. 提交前检查暂存文件，避免把其他题目的代码、资料或临时文件带入 D 题提交。

## 8. 建议的协作者复核顺序

1. 阅读本交接说明和 `D题第一问程序说明.md`。
2. 查看 `method_comparison.csv` 和 `dynamic_programming_trips.csv`。
3. 检查 `box_coverage_check.csv`、`constraint_check.csv` 和 `summary_consistency_check.csv`。
4. 查看 DEM 航段图、Pareto 图和安全余量阶梯图。
5. 打开 `结果提交_Q1.xlsx`，对照预览图复核 18 个架次。
6. 若更改模型假设，重新运行完整流程并比较新旧输出，不要只手工修改提交表。

## 9. 提交前检查清单

- [ ] 当前分支基于最新远端 `main`；
- [ ] 暂存区只包含本次 D 题修改；
- [ ] D 题测试无失败；
- [ ] 80 个货箱全部且仅出现一次；
- [ ] 18 个架次的全部约束通过；
- [ ] 汇总指标与逐架次求和一致；
- [ ] Excel 无公式错误，预览图无截断或错位；
- [ ] 原始赛题资料和本地依赖目录未提交；
- [ ] 论文材料仍标记为草稿，未当作最终正文提交。
