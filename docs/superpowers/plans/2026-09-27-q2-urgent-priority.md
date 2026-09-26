# D题第二问急送优先模型 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留旧第二问结果兼容性的前提下，实现“80箱全部按时，随后严格最小化急送及时性、能耗、架次数和完工时间”的新模型，并只比较 CP-SAT 与 Hybrid。

**Architecture:** 新增 `problem_d_q2_urgent.py` 集中定义急送分类、强制截止时间和目标向量；现有调度、校验和两个求解器通过显式 `evaluation_mode="urgent_priority"` 进入新口径。新实验脚本与输出目录独立，不覆盖 `outputs/problem_d_q2_stage3/`。

**Tech Stack:** Python 3、dataclasses、pandas、NumPy、OR-Tools CP-SAT、Matplotlib、pytest。

## Global Constraints

- 急送集合为首批保障或医疗物资，共31箱。
- 普通49箱的期望送达时间改为强制截止时间。
- 正式可行解必须80箱全部按时。
- 正式目标严格按 \((Z_{\mathrm{urgent}},E,K,M)\) 词典序比较。
- 搜索质量向量为 \((H,L,Z_{\mathrm{urgent}},E,K,M)\)。
- 只运行 CP-SAT 与 Hybrid 两个顶层算法。
- 基准安全余量为20%，能量标定系数为1.0。
- CP-SAT运行1次，Hybrid运行10个固定随机种子。
- 不运行新敏感性分析。
- 新输出写入 `outputs/problem_d_q2_urgent_priority/`。
- 旧模式、旧测试和旧输出保持可用。

---

## File Structure

- Create `src/math_model_cup/problem_d_q2_urgent.py`: 急送分类、强制截止、迟到统计和新目标向量。
- Modify `src/math_model_cup/problem_d_q2.py`: 放宽 `Q2Solution.objective` 类型以支持不同四维顺序，不改变旧字段内容。
- Modify `src/math_model_cup/problem_d_q2_schedule.py`: 调度时按模式选择截止时间与目标计算。
- Modify `src/math_model_cup/problem_d_q2_validation.py`: 新模式校验全部箱截止时间与新目标。
- Modify `src/math_model_cup/problem_d_q2_grouped.py`: 构造全箱按时种子并支持急送质量向量。
- Modify `src/math_model_cup/problem_d_q2_incumbent.py`: 按显式模式验证和比较种子。
- Modify `src/math_model_cup/problem_d_q2_alns.py`: 传播模式并使破坏/修复优先急送和迟到箱。
- Modify `src/math_model_cup/problem_d_q2_candidates.py`: 传播模式到候选组合与资源排程。
- Modify `src/math_model_cup/problem_d_q2_hybrid.py`: 以急送模式运行混合搜索。
- Modify `src/math_model_cup/problem_d_q2_milp.py`: 全箱截止约束和四阶段新目标。
- Create `src/math_model_cup/problem_d_q2_urgent_reporting.py`: 两算法输出、比较表、重复实验和图表。
- Create `scripts/run_problem_d_q2_urgent_priority.py`: 新实验入口。
- Create `tests/test_problem_d_q2_urgent.py`: 目标、截止和校验单元测试。
- Create `tests/test_problem_d_q2_urgent_solvers.py`: CP-SAT/Hybrid 集成测试。
- Modify `D题第二问论文撰写内容指南.md`: 用新实验结果替换待核定旧结果。

---

### Task 1: 急送分类、截止时间与目标函数

**Files:**
- Create: `src/math_model_cup/problem_d_q2_urgent.py`
- Modify: `src/math_model_cup/problem_d_q2.py`
- Test: `tests/test_problem_d_q2_urgent.py`

**Interfaces:**
- Produces: `URGENT_PRIORITY_MODE = "urgent_priority"`
- Produces: `is_urgent_box(box: Q2Box) -> bool`
- Produces: `required_deadline_s(box: Q2Box) -> float`
- Produces: `urgent_lateness(data: Q2Data, solution: Q2Solution) -> tuple[int, float]`
- Produces: `urgent_objective_vector(data: Q2Data, solution: Q2Solution) -> tuple[float, float, int, float]`
- Produces: `urgent_search_key(data: Q2Data, solution: Q2Solution) -> tuple[float, ...]`

- [ ] **Step 1: 写急送分类和截止时间失败测试**

```python
def test_required_deadline_uses_expected_time_for_normal_box(schedule_case):
    data, _, _ = schedule_case
    normal = next(box for box in data.boxes.values() if not is_urgent_box(box))
    assert required_deadline_s(normal) == normal.expected_time_s

def test_urgent_set_is_first_batch_or_medical(schedule_case):
    data, _, _ = schedule_case
    assert all(
        is_urgent_box(box) == (box.first_batch or box.material_type == "医疗物资")
        for box in data.boxes.values()
    )
```

- [ ] **Step 2: 运行测试并确认缺少模块失败**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent.py -q`

Expected: FAIL，提示 `problem_d_q2_urgent` 不存在。

- [ ] **Step 3: 实现基础函数**

```python
URGENT_PRIORITY_MODE = "urgent_priority"

def is_urgent_box(box: Q2Box) -> bool:
    return box.first_batch or box.material_type == "医疗物资"

def required_deadline_s(box: Q2Box) -> float:
    if is_urgent_box(box):
        assert box.hard_deadline_s is not None
        return float(box.hard_deadline_s)
    return float(box.expected_time_s)
```

`urgent_objective_vector` 返回 `(urgent_weighted_delivery, total_energy, trip_count, makespan)`；`urgent_search_key` 在前面增加迟到数量和总迟到秒数。

- [ ] **Step 4: 放宽目标类型注解并运行测试**

将 `Q2Solution.objective` 改为 `Tuple[float, ...]`，不改旧 `objective_vector`。

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent.py tests/test_problem_d_q2_validation.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2.py src/math_model_cup/problem_d_q2_urgent.py tests/test_problem_d_q2_urgent.py
git commit -m "feat: add urgent-priority Q2 objective"
```

### Task 2: 调度与独立校验支持全箱截止

**Files:**
- Modify: `src/math_model_cup/problem_d_q2_schedule.py`
- Modify: `src/math_model_cup/problem_d_q2_validation.py`
- Test: `tests/test_problem_d_q2_urgent.py`

**Interfaces:**
- Produces: `_deadline_s(box, evaluation_mode) -> float | None`
- Extends: `_make_solution(data, method, executions, runtime_s, solver_status, diagnostics=None, *, evaluation_mode: str = "published") -> Q2Solution`
- Extends: `schedule_trips_greedy(data, plans, method="greedy_schedule", *, enforce_hard_deadlines=True, evaluation_mode: str = "published") -> Q2Solution`
- Extends: `validate_q2_solution(data, arcs, solution, reserve_ratio=0.20, *, range_energy_fraction=1.0, evaluation_mode: str = "published") -> ValidationReport`

- [ ] **Step 1: 写普通物资迟到失败测试**

构造一个旧模式有效、但普通物资送达晚于 `expected_time_s` 的解：

```python
legacy = validate_q2_solution(data, arcs, solution)
urgent = validate_q2_solution(
    data, arcs, solution, evaluation_mode=URGENT_PRIORITY_MODE
)
assert legacy.is_valid
assert any(issue.code == "normal_deadline" for issue in urgent.issues)
```

- [ ] **Step 2: 运行定向测试确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent.py -q`

Expected: FAIL，提示缺少 `evaluation_mode`。

- [ ] **Step 3: 实现模式化截止与目标**

在调度器中，急送模式使用 `required_deadline_s` 排序与判断；`_make_solution` 在急送模式调用 `urgent_objective_vector`。在校验器中分别生成 `urgent_deadline` 与 `normal_deadline` 问题码，并用相同模式复算目标。

- [ ] **Step 4: 运行调度、校验和旧回归测试**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent.py tests/test_problem_d_q2_validation.py tests/test_problem_d_q2_integration.py -q`

Expected: PASS，旧模式测试结果不变。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_schedule.py src/math_model_cup/problem_d_q2_validation.py tests/test_problem_d_q2_urgent.py
git commit -m "feat: enforce all-box deadlines in urgent mode"
```

### Task 3: 构造全箱按时共同种子

**Files:**
- Modify: `src/math_model_cup/problem_d_q2_grouped.py`
- Test: `tests/test_problem_d_q2_urgent_solvers.py`

**Interfaces:**
- Extends: `solve_grouped_local_search(data, arcs, *, seed=20260924, iterations=200, method="grouped_local_search", max_stops=4, reserve_ratio=0.20, range_energy_fraction=1.0, objective_order="timeliness", evaluation_mode: str = "published") -> Q2Solution`
- Uses: `urgent_search_key(data, solution)`
- Produces: 全箱按时的 `Q2Solution`，其 `objective` 为 `(Zurgent, E, K, M)`。

- [ ] **Step 1: 写共同种子集成失败测试**

```python
solution = solve_grouped_local_search(
    data,
    arcs,
    seed=20260924,
    iterations=20,
    evaluation_mode=URGENT_PRIORITY_MODE,
)
report = validate_q2_solution(
    data, arcs, solution, evaluation_mode=URGENT_PRIORITY_MODE
)
assert report.is_valid
assert urgent_lateness(data, solution) == (0, 0.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent_solvers.py::test_grouped_seed_delivers_all_boxes_on_time -q`

Expected: FAIL，提示缺少模式参数或普通物资迟到。

- [ ] **Step 3: 传播急送模式**

所有 `_schedule_search_state`、迟到箱提取、修复、合并、换型和局部搜索调用都传递 `evaluation_mode`。急送模式下 `_grouped_quality` 直接返回 `urgent_search_key`；旧 `objective_order` 行为保持不变。

- [ ] **Step 4: 验证全箱按时种子**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_grouped.py tests/test_problem_d_q2_urgent_solvers.py::test_grouped_seed_delivers_all_boxes_on_time -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_grouped.py tests/test_problem_d_q2_urgent_solvers.py
git commit -m "feat: build all-on-time urgent Q2 seed"
```

### Task 4: Hybrid 支持急送目标

**Files:**
- Modify: `src/math_model_cup/problem_d_q2_incumbent.py`
- Modify: `src/math_model_cup/problem_d_q2_alns.py`
- Modify: `src/math_model_cup/problem_d_q2_candidates.py`
- Modify: `src/math_model_cup/problem_d_q2_hybrid.py`
- Test: `tests/test_problem_d_q2_urgent_solvers.py`

**Interfaces:**
- Extends: `solution_key(data, solution, evaluation_mode="published")`
- Extends: `ensure_valid_initial_solution(data, arcs, initial_solution, *, evaluation_mode="published") -> None`
- Extends: `finalize_seeded_solution(method, native_solution, initial_solution, *, data, started, native_source, diagnostics=None, evaluation_mode="published") -> Q2Solution`
- Extends: `solve_alns`, `solve_candidate_method`, `solve_hybrid` with `evaluation_mode`.

- [ ] **Step 1: 写 Hybrid 失败测试**

```python
solution = solve_hybrid(
    data,
    arcs,
    seed=20260924,
    iterations=20,
    rounds=1,
    time_limit_s=5.0,
    initial_solution=seed_solution,
    evaluation_mode=URGENT_PRIORITY_MODE,
)
assert validate_q2_solution(
    data, arcs, solution, evaluation_mode=URGENT_PRIORITY_MODE
).is_valid
assert solution.objective == urgent_objective_vector(data, solution)
```

- [ ] **Step 2: 运行测试确认参数缺失失败**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent_solvers.py::test_hybrid_uses_urgent_lexicographic_objective -q`

Expected: FAIL。

- [ ] **Step 3: 实现模式传播与急送邻域**

急送模式下：

- `worst_timeliness` 只按急送箱指标排序；
- 修复排序先处理违反 `required_deadline_s` 的箱，再处理急送箱；
- 候选排程调用急送模式；
- ALNS 接受判定和 incumbent 比较使用新目标元组；
- Hybrid 在 current/improved/refreshed 三者中使用新模式 `solution_key`。

- [ ] **Step 4: 运行 Hybrid 与旧求解器回归测试**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent_solvers.py tests/test_problem_d_q2_solvers.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_incumbent.py src/math_model_cup/problem_d_q2_alns.py src/math_model_cup/problem_d_q2_candidates.py src/math_model_cup/problem_d_q2_hybrid.py tests/test_problem_d_q2_urgent_solvers.py
git commit -m "feat: optimize urgent objective with hybrid solver"
```

### Task 5: CP-SAT 四阶段急送目标

**Files:**
- Modify: `src/math_model_cup/problem_d_q2_milp.py`
- Test: `tests/test_problem_d_q2_urgent_solvers.py`

**Interfaces:**
- Extends: `solve_integrated_milp(data, arcs, time_limit_s=1800.0, max_stops=3, max_candidates=1200, *, initial_solution=None, evaluation_mode: str = "published") -> Q2Solution`
- Consumes: `required_deadline_s`, `is_urgent_box`, `urgent_objective_vector`。

- [ ] **Step 1: 写 CP-SAT 新模式失败测试**

```python
solution = solve_integrated_milp(
    data,
    arcs,
    time_limit_s=5.0,
    initial_solution=seed_solution,
    evaluation_mode=URGENT_PRIORITY_MODE,
)
assert validate_q2_solution(
    data, arcs, solution, evaluation_mode=URGENT_PRIORITY_MODE
).is_valid
assert solution.objective == urgent_objective_vector(data, solution)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent_solvers.py::test_cpsat_uses_all_deadlines_and_urgent_objective -q`

Expected: FAIL。

- [ ] **Step 3: 实现新约束与阶段目标**

急送模式下：

- 每个候选架次对其所有箱添加 `start <= required_deadline - delivery_offset`；
- timing terms 只包含急送箱；
- 阶段顺序改为 `urgent_timing -> energy -> trip_count -> makespan`；
- 每阶段得到可行值后添加等值约束再进入下一阶段；
- 最终 `_make_solution` 使用急送模式并由独立校验复算。

- [ ] **Step 4: 运行 CP-SAT、Hybrid 和旧模型回归测试**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent_solvers.py tests/test_problem_d_q2_solvers.py tests/test_problem_d_q2_integration.py -q`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_milp.py tests/test_problem_d_q2_urgent_solvers.py
git commit -m "feat: add urgent-priority CP-SAT phases"
```

### Task 6: 新实验脚本、结果导出与图表

**Files:**
- Create: `src/math_model_cup/problem_d_q2_urgent_reporting.py`
- Create: `scripts/run_problem_d_q2_urgent_priority.py`
- Test: `tests/test_problem_d_q2_urgent_reporting.py`

**Interfaces:**
- Produces: `run_urgent_methods(data, arcs, output_dir: Path, *, seed: int, iterations: int, time_limit_s: float) -> dict[str, Q2Solution]`
- Produces: `run_hybrid_repetitions(data, arcs, initial_solution, *, seeds: tuple[int, ...], iterations: int, time_limit_s: float) -> pandas.DataFrame`
- Produces: `create_urgent_figures(output_dir: Path, data, solutions, repetitions) -> list[Path]`

- [ ] **Step 1: 写输出范围失败测试**

```python
def test_method_table_contains_only_cpsat_and_hybrid(tmp_path, urgent_case):
    data, arcs = urgent_case
    solutions = run_urgent_methods(
        data,
        arcs,
        tmp_path,
        seed=20260924,
        iterations=20,
        time_limit_s=5.0,
    )
    rows = pd.read_csv(tmp_path / "method_comparison.csv")
    assert set(rows["method"]) == {"integrated_milp", "hybrid"}
    assert (rows["late_box_count"] == 0).all()
```

- [ ] **Step 2: 运行测试确认模块缺失失败**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent_reporting.py -q`

Expected: FAIL。

- [ ] **Step 3: 实现报告与脚本**

`method_comparison.csv` 至少包含：

```text
method,validation,solver_status,urgent_weighted_delivery,
energy_kwh,trip_count,makespan_s,runtime_s,late_box_count,
urgent_late_count,normal_late_count,minimum_return_soc_percent
```

Hybrid 重复实验固定使用种子 `20260924`—`20260933`。脚本不调用旧敏感性分析函数，默认输出目录为 `outputs/problem_d_q2_urgent_priority/`。

- [ ] **Step 4: 运行报告测试和快速端到端实验**

Run: `$env:PYTHONPATH='src'; python -m pytest tests/test_problem_d_q2_urgent_reporting.py -q`

Run: `python scripts/run_problem_d_q2_urgent_priority.py --problem-dir "D题" --output-dir "outputs/problem_d_q2_urgent_priority" --quick --time-limit 5`

Expected: 输出两算法结果、10次 Hybrid 重复结果、图表和摘要；所有正式结果迟到箱数为0。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_urgent_reporting.py scripts/run_problem_d_q2_urgent_priority.py tests/test_problem_d_q2_urgent_reporting.py outputs/problem_d_q2_urgent_priority
git commit -m "feat: add urgent-priority Q2 experiments"
```

### Task 7: 更新论文撰写指导文档

**Files:**
- Modify: `D题第二问论文撰写内容指南.md`

**Interfaces:**
- Consumes: `outputs/problem_d_q2_urgent_priority/method_comparison.csv`
- Consumes: `outputs/problem_d_q2_urgent_priority/hybrid_repetitions.csv`

- [ ] **Step 1: 用机器结果替换待核定说明**

文档只介绍新模型、CP-SAT、Hybrid、全箱按时校验和两算法实际结果。删除旧24/22架次正式结论，除非新实验再次产生完全相同方案。

- [ ] **Step 2: 检查文档数字可追溯性**

Run: `rg -n "待核定|15架次|候选架次法.*正式|ALNS.*正式" "D题第二问论文撰写内容指南.md"`

Expected: 无匹配。

- [ ] **Step 3: 对照 CSV 自动核对关键数值**

Run:

```powershell
python -c 'import csv,pathlib; root=pathlib.Path("outputs/problem_d_q2_urgent_priority"); rows=list(csv.DictReader((root/"method_comparison.csv").open(encoding="utf-8-sig"))); text=pathlib.Path("D题第二问论文撰写内容指南.md").read_text(encoding="utf-8"); assert all(format(float(row["urgent_weighted_delivery"]), ".10f") in text and format(float(row["energy_kwh"]), ".4f") in text and str(int(float(row["trip_count"]))) in text for row in rows); print("documentation_consistency=PASS")'
```

Expected: `documentation_consistency=PASS`。

- [ ] **Step 4: 提交**

```powershell
git add "D题第二问论文撰写内容指南.md"
git commit -m "docs: explain urgent-priority Q2 model and results"
```

### Task 8: 全量验证与交付

**Files:**
- Verify only.

**Interfaces:**
- Consumes all prior tasks.

- [ ] **Step 1: 运行全量测试**

Run: `$env:PYTHONPATH='src'; python -m pytest -q`

Expected: 全部测试通过，无失败。

- [ ] **Step 2: 运行差异检查**

Run: `git diff --check`

Expected: exit code 0。

- [ ] **Step 3: 验证最终输出**

检查：

- `method_comparison.csv` 只含 CP-SAT 与 Hybrid；
- 两个方法的正式结果均为 `PASS`；
- 所有正式结果 `late_box_count=0`；
- 10次 Hybrid 运行均有明确状态；
- 文档关键数值与 CSV 一致。

- [ ] **Step 4: 提交必要的最终修正**

仅在验证产生机械性修正时提交，提交信息：

```powershell
git commit -m "test: verify urgent-priority Q2 workflow"
```
