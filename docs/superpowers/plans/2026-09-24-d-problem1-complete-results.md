# D Problem 1 Complete Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在既有 D 题第一问三种算法基础上，补齐物理口径、严格 GIS、多目标优化、精确安全余量、独立校验、论文级材料和官方提交工作簿，并用自动化测试及视觉检查证明结果可提交。

**Architecture:** 保留 `problem_d_q1.py` 的数据模型、候选生成和三种求解器，只做向后兼容的增量扩展；将严格 GIS、Pareto/灵敏度、独立校验和报告生成拆到聚焦模块。所有正式输出由统一运行脚本一次生成，官方模板由 artifact-tool 复制后定点写入，原始赛题资料只读。

**Tech Stack:** Python 3.11、NumPy、Pillow、pyproj、SciPy MILP、Matplotlib/Seaborn、pytest；Node.js 与 `@oai/artifact-tool` 用于 XLSX 编辑、检查和渲染。

**Global Constraints:** 新增文本统一 UTF-8；不覆盖 DOCX、原始 XLSX、DEM；数值计算保留双精度，仅输出时统一小数位；现有函数默认行为保持兼容；每项正式结论必须有测试或独立复核证据。

---

### Task 1: 固化题面公式审计并扩展能耗口径

**Files:**
- Modify: `src/math_model_cup/problem_d_q1.py`
- Create: `src/math_model_cup/problem_d_q1_analysis.py`
- Create: `tests/test_problem_d_q1_energy.py`
- Create: `outputs/problem_d_q1/能耗公式与物理口径.md`（由运行脚本生成）

**Steps:**
1. 写失败测试，覆盖空载质量含电池、爬升质量为“含电池空载总质量+当前载荷”、返程载荷为 0、下降附加能耗为 0、作业高度为地面海拔加 30 m。
2. 为能耗函数增加向后兼容的 `range_energy_fraction` 参数；主口径取 1.0，备选“标准航程已预留 20%”口径取 0.8。
3. 增加逐项能耗分解数据结构，明确去/返水平能耗、去/返爬升能耗、总能耗、飞行/地面时间和返航 SOC。
4. 生成题面 DOCX、公式 XML、参数表审计记录和严格量纲推导；说明两种解释的逻辑与推荐依据。
5. 运行能耗单元测试并保存两种口径的稳健性结果。

### Task 2: 用严格栅格超覆盖替换密集采样并校验 GIS

**Files:**
- Modify: `src/math_model_cup/problem_d_q1.py`
- Create: `tests/test_problem_d_q1_gis.py`
- Modify: `scripts/run_problem_d_q1.py`
- Create: `outputs/problem_d_q1/gis_metadata.csv`（生成）
- Create: `outputs/problem_d_q1/route_geometry.csv`（生成）

**Steps:**
1. 写失败测试，构造水平线、斜线、恰过像元角点和边界线，验证闭集意义下所有接触像元均被枚举。
2. 用线段与像元矩形相交判定实现 all-touched 超覆盖，删除对“每像元 32 次采样”的依赖。
3. 用 `pyproj.Geod(ellps="WGS84")` 计算椭球测地线距离，同时保留 Haversine 作为差异基线。
4. 读取并验证 GeoTIFF 坐标系、分辨率、NoData、仿射变换和边界；记录峰值像元行列号及中心经纬度。
5. 输出 15 条航段的水平距离、最高高程/坐标、巡航海拔、去返程爬升和下降高度。
6. 对 S003、S008、S014、S015 生成可追溯的人工复核明细，并比较严格 GIS 与旧计算的最大差异及组批变化。

### Task 3: 实现真正的多目标 Pareto 动态规划

**Files:**
- Modify: `src/math_model_cup/problem_d_q1_analysis.py`
- Create: `tests/test_problem_d_q1_pareto.py`
- Modify: `scripts/run_problem_d_q1.py`

**Steps:**
1. 写小型可枚举实例测试，验证非支配筛选、三锚点和架次数预算的结果。
2. 在每个服务区保留 `(架次数, 累计时间, 总能耗)` 非支配状态，再卷积得到全局 Pareto 集。
3. 分别提取架次数最少、总能耗最少、累计时间最少三个锚点；固定 `N=Nmin` 输出时间—能耗 Pareto 前沿。
4. 在 `N<=Nmin+1`、`N<=Nmin+2` 下分别提取最低能耗和最短时间方案，量化边际收益。
5. 输出完整 Pareto CSV、锚点/预算比较 CSV 和最终优先级推荐；将原求解器准确命名为“词典序动态规划”，新模块命名为“多目标 Pareto 动态规划”。

### Task 4: 计算精确临界安全余量与稳定区间

**Files:**
- Modify: `src/math_model_cup/problem_d_q1_analysis.py`
- Create: `tests/test_problem_d_q1_reserve.py`
- Modify: `scripts/run_problem_d_q1.py`

**Steps:**
1. 对每个服务区—机型—候选货箱组合计算 `rho_crit=1-E_RT(q)/E_use`，写边界与排序测试。
2. 在全部可行临界点右侧及相邻点中部重新求解，压缩为解结构不变的精确分段稳定区间。
3. 对相邻区间求差，记录首先受影响的服务区/机型、失效原组批、货箱迁移、架次数/能耗/时间变化及不可配送阈值。
4. 输出全部临界点、稳定区间、变化事件及安全载荷曲线数据。

### Task 5: 增加完全独立的物理与汇总校验

**Files:**
- Create: `src/math_model_cup/problem_d_q1_validation.py`
- Create: `tests/test_problem_d_q1_validation.py`
- Modify: `scripts/run_problem_d_q1.py`

**Steps:**
1. 不调用候选对象的能耗/时间结果，直接从原始机型、航段和货箱数据重算每个推荐架次。
2. 自动选取并详细输出 S001 近距离重载、S003/S008 远距离重载、B 型 25 kg 和最低返航 SOC 架次。
3. 验证 80 个货箱恰好一次、服务区不混装、质量/体积/能量约束、逐架次与汇总求和一致。
4. 统一设置并解释质量、体积、能量、时间和 SOC 容差，输出逐箱覆盖、约束和汇总一致性 CSV。

### Task 6: 生成论文级报告、图表和算法对比

**Files:**
- Create: `src/math_model_cup/problem_d_q1_reporting.py`
- Modify: `scripts/run_problem_d_q1.py`
- Create: `outputs/problem_d_q1/论文_问题一.md`（生成）
- Create: `outputs/problem_d_q1/完整成果索引.md`（生成）
- Create: `outputs/problem_d_q1/figures/*.png`（生成）

**Steps:**
1. 生成 DEM 航段图、15×3 最大安全载荷热力图、需求与有效容量图、Pareto 前沿图、架次质量/体积利用率图、安全余量连续曲线和最少架次数阶梯图。
2. 扩充算法对比表：目标值、最优差距、运行时间、质量/体积利用率；启发式统一命名为“BFD+合并改进”。
3. 生成可直接用于论文的假设、符号、航段/时间/能耗/安全载荷模型、组批模型、三算法流程及伪代码、多目标策略、结果、灵敏度、优缺点与适用条件。
4. 对关键图进行像素级打开检查，确认文字、图例、坐标轴和数值可读。

### Task 7: 写入并视觉检查官方提交模板

**Files:**
- Create: `scripts/build_q1_submission.mjs`
- Create: `outputs/problem_d_q1/结果提交_Q1.xlsx`（生成）
- Create: `tests/test_problem_d_q1_submission.py`

**Steps:**
1. 只读检查原模板的工作表、表头、格式和 Q1 写入区域。
2. 在首次工作簿写入命令前调用一次 artifact 操作标记脚本。
3. 用 `@oai/artifact-tool` 导入原模板，只写 `Q1_单点组批` 数据区，保留其他工作表和格式，另存目标路径。
4. 用 artifact-tool inspect 验证内容/公式错误并渲染 Q1 工作表；人工查看渲染图。
5. 用独立 Python 测试比较原/结果工作簿的工作表集合、未修改工作表内容及关键样式，验证 18 架次、80 箱唯一覆盖和显示精度。

### Task 8: 完整回归、重生成与交付核验

**Files:**
- Modify: `tests/test_problem_d_q1.py`
- Modify: `outputs/problem_d_q1/summary.md`（生成）

**Steps:**
1. 运行全部 pytest；若失败，按系统化调试定位根因后修复。
2. 运行统一生成脚本，确保所有 CSV、Markdown、PNG 和 XLSX 一次生成且结果可复现。
3. 复跑完整测试，记录通过数量、DP/MILP 一致性、独立校验和工作簿视觉检查证据。
4. 对比旧结果与最终结果，明确稳定结论、依赖能耗解释的结论和数值差异。
5. 检查 Git 变更只包含本任务代码/文档，提交实现，并列出全部最终文件的绝对可点击路径。
