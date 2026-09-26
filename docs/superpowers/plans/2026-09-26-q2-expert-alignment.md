# Q2 Expert Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 对齐第二问专家答疑边界，并新增 0.8/1.0 航程—能量标定敏感性实验。

**Architecture:** 将 `range_energy_fraction` 作为显式物理参数，从分组局部搜索贯穿到每次架次重建，并由独立校验器按相同口径复算。敏感性模块对每个水平重新求解，图表动态容纳五类因子；基准默认值保持 1.0。

**Tech Stack:** Python 3、pytest、pandas、matplotlib、现有 Q2 物理核与分组局部搜索。

## Global Constraints

- 基准模型继续使用 `range_energy_fraction=1.0` 和 20% 返航余量。
- 不实现异地换电状态模型，只明确 O01 换电/充电为保守限制。
- 不覆盖或清理工作区其他未提交修改。
- 所有文本保持 UTF-8。

---

### Task 1: 参数贯穿与独立校验

**Files:**
- Modify: `tests/test_problem_d_q2_validation.py`
- Modify: `tests/test_problem_d_q2_grouped.py`
- Modify: `src/math_model_cup/problem_d_q2_validation.py`
- Modify: `src/math_model_cup/problem_d_q2_grouped.py`

**Interfaces:**
- Produces: `validate_q2_solution(..., reserve_ratio=0.20, *, range_energy_fraction=1.0)`
- Produces: `solve_grouped_local_search(..., reserve_ratio=0.20, range_energy_fraction=1.0, ...)`

- [ ] **Step 1: 写独立校验失败测试**

构造使用 `evaluate_trip(..., range_energy_fraction=0.8)` 的计划并排程；断言默认校验产生 `energy_mismatch`，传入 `range_energy_fraction=0.8` 后通过。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_problem_d_q2_validation.py -q`

Expected: FAIL，提示 `validate_q2_solution` 不接受 `range_energy_fraction`。

- [ ] **Step 3: 最小实现独立校验参数**

在校验器签名中增加关键字参数，并传给内部 `evaluate_trip`：

```python
def validate_q2_solution(..., reserve_ratio: float = 0.20, *, range_energy_fraction: float = 1.0):
    recomputed = evaluate_trip(
        data,
        arcs,
        TripDraft(trip.plan.model_id, trip.plan.stops),
        reserve_ratio,
        range_energy_fraction=range_energy_fraction,
    )
```

- [ ] **Step 4: 写分组求解失败测试**

对小型 `schedule_case` 分别调用默认口径和 `range_energy_fraction=0.8`，断言替代口径诊断值为 0.8、独立校验通过、总能耗低于默认口径。

- [ ] **Step 5: 运行失败测试**

Run: `python -m pytest tests/test_problem_d_q2_grouped.py -q`

Expected: FAIL，提示求解器不接受该参数。

- [ ] **Step 6: 将参数传递到全部架次生成路径**

为 `_best_plan`、`_feasible_plans`、`_build_pure_ffd_plans`、`build_grouped_initial_plans`、修复、合并、重选机型和局部搜索函数增加参数，并在每个 `evaluate_trip` 调用中传入。诊断字典记录 `range_energy_fraction`。

- [ ] **Step 7: 运行相关测试**

Run: `python -m pytest tests/test_problem_d_q2_validation.py tests/test_problem_d_q2_grouped.py -q`

Expected: PASS。

### Task 2: 新增能量标定敏感性与五因子图

**Files:**
- Modify: `tests/test_problem_d_q2_experiments.py`
- Modify: `src/math_model_cup/problem_d_q2_experiments.py`

**Interfaces:**
- `_sensitivity_result(..., reserve_ratio=0.20, range_energy_fraction=1.0)`
- `run_sensitivity_analysis(...)` 新增因子 `range_energy_fraction`，水平为 0.8、1.0。

- [ ] **Step 1: 扩展敏感性失败测试**

将期望因子集合增加 `range_energy_fraction`，并断言其水平严格等于 `{0.8, 1.0}`、两行均可行且 0.8 行能耗低于 1.0 行。

- [ ] **Step 2: 运行失败测试**

Run: `python -m pytest tests/test_problem_d_q2_experiments.py -q`

Expected: FAIL，缺少新因子。

- [ ] **Step 3: 实现重新求解和对应校验**

`_sensitivity_result` 把参数同时传给 `solve_grouped_local_search` 与 `validate_q2_solution`；`run_sensitivity_analysis` 追加 0.8、1.0 两个场景。

- [ ] **Step 4: 扩展图表布局**

使用 `plt.subplots(3, 2, figsize=(7.2, 7.6))`，按因子绘图后关闭多余轴，并在基准映射中加入 `range_energy_fraction: 1.0`。

- [ ] **Step 5: 运行实验测试**

Run: `python -m pytest tests/test_problem_d_q2_experiments.py -q`

Expected: PASS，仍生成 14 个图文件。

### Task 3: 同步专家口径文档

**Files:**
- Modify: `D题第二问当前物理口径说明.md`

- [ ] **Step 1: 增加四项说明**

明确电池初始位于 O01、禁止携带备用电池、O01 换电是收缩可行域的保守假设、能耗公式不是专家确认的唯一原式。

- [ ] **Step 2: 增加双标定解释**

说明 1.0 表示标准航程消耗全部可用能量，0.8 表示标准航程已包含 20% 返航余量；基准仍为 1.0，替代值只用于稳健性分析。

- [ ] **Step 3: 文档校验**

Run: `git diff --check -- "D题第二问当前物理口径说明.md"`

Expected: 无输出且退出码为 0。

### Task 4: 全量验证与输出再生

**Files:**
- Modify: `outputs/problem_d_q2_stage3/sensitivity_analysis.csv`
- Modify: `outputs/problem_d_q2_stage3/experiment_summary.md`
- Modify: `outputs/problem_d_q2_stage3/experiment_manifest.json`
- Modify: `outputs/problem_d_q2_stage3/figures/fig05_sensitivity.png`
- Modify: `outputs/problem_d_q2_stage3/figures/fig05_sensitivity.pdf`

- [ ] **Step 1: 运行第二问测试**

Run: `python -m pytest tests/test_problem_d_q2_*.py -q`

Expected: 全部 PASS 或仅环境变量控制的真实数据测试 SKIP。

- [ ] **Step 2: 按原清单参数重新生成实验**

Run: `python scripts/run_problem_d_q2_experiments.py --problem-dir "D题" --output-dir "outputs/problem_d_q2_stage3" --seeds "20260924,20260925,20260926,20260927,20260928,20260929,20260930,20260931,20260932,20260933" --alns-iterations 20 --time-limit 5 --quick`

Expected: `sensitivity_scenarios=16`，新 CSV 含 0.8/1.0 两个能量标定场景，图文件共 14 个。

- [ ] **Step 3: 运行完整测试集**

Run: `python -m pytest -q`

Expected: 全部 PASS 或仅预期 SKIP。

- [ ] **Step 4: 最终一致性校验**

核对 1.0 敏感性行仍对应基准口径，0.8 行通过独立校验，文档明确限制最优性范围；执行 `git diff --check`。
