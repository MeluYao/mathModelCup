# D 题第二问四方案最小可行版本实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在实际 D 题数据上实现四种可独立运行的第二问最小可行求解器，并生成经过统一独立校验的架次、逐箱交付、无人机和电池时间线。

**Architecture:** 先建立唯一的数据、全节点 GIS、架次物理评价、目标函数和校验接口，再分别实现受限一体化 MILP、候选架次法、ALNS 和混合算法。四个求解器只共享公共物理与校验层，均返回统一 `Q2Solution`；混合算法按定义组合候选法和 ALNS。

**Tech Stack:** Python 3.11、NumPy、pandas、Pillow、pyproj、SciPy/HiGHS、OR-Tools CP-SAT、pytest、Matplotlib、openpyxl。

## Global Constraints

- 所有 Python、Markdown 和 CSV 文本采用 UTF-8。
- 不修改或覆盖 `D题/` 下的原始 DOCX、XLSX、GeoTIFF 和提交模板。
- 默认返航安全余量为 20%，正式结果使用题面原始参数。
- 医疗物资期望时间与首批截止时间是硬约束，其他期望时间只参与及时性指标。
- 架次开始时刻是准备与装载开始时刻；交付完成时刻是该站全部交接结束时刻。
- 电池从架次开始至返航被占用，返航后充满至 100% 才能复用；不同电池可并行充电。
- 四种算法调用同一个架次评价器、目标函数和独立校验器。
- 阶段一允许方案一达到时间上限后返回最优可行解，但不得返回未通过独立校验的结果。
- 不改动现有 D1 结果和测试口径；D1 测试必须保持通过。

---

## 文件结构

新增或修改文件及职责：

- `environment.yml`：增加 OR-Tools 依赖。
- `src/math_model_cup/problem_d_q2.py`：Q2 领域对象、数据加载和公共异常。
- `src/math_model_cup/problem_d_q2_geometry.py`：16 节点有向航段矩阵。
- `src/math_model_cup/problem_d_q2_physics.py`：多点架次时间、能耗、SOC、充电与目标评价。
- `src/math_model_cup/problem_d_q2_schedule.py`：贪心和 CP-SAT 资源调度。
- `src/math_model_cup/problem_d_q2_validation.py`：独立重算与约束校验。
- `src/math_model_cup/problem_d_q2_candidates.py`：候选架次生成、剪枝和集合划分。
- `src/math_model_cup/problem_d_q2_milp.py`：受限一体化 MILP。
- `src/math_model_cup/problem_d_q2_alns.py`：ALNS 求解器。
- `src/math_model_cup/problem_d_q2_hybrid.py`：候选池、ALNS 回灌和精确调度。
- `src/math_model_cup/problem_d_q2_reporting.py`：CSV、JSON、图表和提交表源数据。
- `scripts/run_problem_d_q2.py`：统一命令行入口。
- `tests/test_problem_d_q2_data.py`：数据和领域对象。
- `tests/test_problem_d_q2_physics.py`：航段、时间、能耗和充电。
- `tests/test_problem_d_q2_schedule.py`：无人机、电池与时间窗。
- `tests/test_problem_d_q2_candidates.py`：候选与集合划分。
- `tests/test_problem_d_q2_solvers.py`：四求解器合成实例。
- `tests/test_problem_d_q2_validation.py`：独立校验器故障注入。
- `tests/test_problem_d_q2_integration.py`：真实数据短时集成测试。

---

### Task 1: Q2 领域对象和数据加载

**Files:**
- Modify: `environment.yml`
- Create: `src/math_model_cup/problem_d_q2.py`
- Create: `tests/test_problem_d_q2_data.py`

**Interfaces:**
- Produces: `load_q2_data(problem_dir: Path) -> Q2Data`
- Produces: `Node`, `Q2Box`, `AircraftModel`, `AircraftUnit`, `BatteryUnit`, `TripStop`, `TripDraft`, `TripPlan`, `TripExecution`, `DeliveryRecord`, `Q2Solution`
- Produces: `InfeasibleQ2Error`

核心结果类型固定为：

```python
@dataclass(frozen=True)
class Q2Solution:
    method: str
    trips: tuple[TripExecution, ...]
    deliveries: tuple[DeliveryRecord, ...]
    objective: tuple[float, float, float, int]
    runtime_s: float
    solver_status: str
    diagnostics: Mapping[str, object]

@dataclass(frozen=True)
class TripExecution:
    trip_id: str
    plan: TripPlan
    aircraft_id: str
    battery_id: str
    start_time_s: float
    return_time_s: float
    battery_ready_time_s: float

@dataclass(frozen=True)
class DeliveryRecord:
    box_id: str
    trip_id: str
    service_id: str
    delivery_time_s: float
```

`Q2Solution` 不保存 `Q2Data` 或航段矩阵，测试和校验始终显式传入数据，避免结果对象夹带可变上下文。

- [ ] **Step 1: 写数据加载失败测试**

```python
def test_real_loader_reads_all_transport_resources(problem_dir: Path) -> None:
    data = load_q2_data(problem_dir)
    assert len(data.nodes) == 16
    assert len(data.boxes) == 80
    assert [unit.model_id for unit in data.aircraft_units].count("A") == 4
    assert [unit.model_id for unit in data.aircraft_units].count("B") == 2
    assert [unit.model_id for unit in data.aircraft_units].count("C") == 2
    assert [battery.model_id for battery in data.batteries].count("A") == 6
    assert [battery.model_id for battery in data.batteries].count("B") == 4
    assert [battery.model_id for battery in data.batteries].count("C") == 4
    assert sum(box.mass_kg for box in data.boxes.values()) == pytest.approx(758.0)
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_data.py`

Expected: FAIL，提示 `math_model_cup.problem_d_q2` 不存在。

- [ ] **Step 3: 实现不可变领域对象和数据加载**

`Q2Box` 必须保存 `box_id`、`service_id`、`material_type`、`mass_kg`、`volume_m3`、`first_batch`、`first_deadline_s`、`expected_time_s`、`priority_weight`，并提供：

```python
@property
def hard_deadline_s(self) -> float | None:
    deadlines = []
    if self.first_batch and self.first_deadline_s is not None:
        deadlines.append(self.first_deadline_s)
    if self.material_type == "医疗物资":
        deadlines.append(self.expected_time_s)
    return min(deadlines) if deadlines else None
```

`load_q2_data` 从三份基础工作簿读取 16 个节点、80 个货箱、三类机型、8 架实体机和14组电池。电池编号固定生成 `BA01...BA06`、`BB01...BB04`、`BC01...BC04`，避免不同算法自行命名。

- [ ] **Step 4: 增加 OR-Tools 依赖并运行测试**

在 `environment.yml` 的 dependencies 中加入 `ortools`。运行：

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_data.py`

Expected: PASS，数据计数、质量和硬时限规则全部正确。

- [ ] **Step 5: 提交**

```powershell
git add environment.yml src/math_model_cup/problem_d_q2.py tests/test_problem_d_q2_data.py
git commit -m "feat: add problem D2 domain data"
```

---

### Task 2: 全节点 GIS 航段矩阵

**Files:**
- Create: `src/math_model_cup/problem_d_q2_geometry.py`
- Create: `tests/test_problem_d_q2_physics.py`

**Interfaces:**
- Consumes: `Q2Data`
- Produces: `ArcGeometry`
- Produces: `build_arc_matrix(data: Q2Data) -> Mapping[tuple[str, str], ArcGeometry]`

- [ ] **Step 1: 写方向性航段测试**

```python
def test_arc_matrix_contains_all_directed_node_pairs(real_q2_data: Q2Data) -> None:
    arcs = build_arc_matrix(real_q2_data)
    assert len(arcs) == 16 * 15
    outbound = arcs[("O01", "S015")]
    inbound = arcs[("S015", "O01")]
    assert outbound.distance_m == pytest.approx(inbound.distance_m)
    assert outbound.climb_m != inbound.climb_m
    assert outbound.cruise_altitude_m == inbound.cruise_altitude_m
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_physics.py`

Expected: FAIL，提示 `build_arc_matrix` 不存在。

- [ ] **Step 3: 实现有向航段矩阵**

复用 D1 已验证的 WGS84 距离、GeoTIFF 元数据解析和 `all_touched` 线段像元逻辑。对每个不同节点对计算：

```python
ArcGeometry(
    origin_id=origin.node_id,
    destination_id=destination.node_id,
    distance_m=distance_m,
    peak_ground_m=peak_ground,
    cruise_altitude_m=peak_ground + 50.0,
    climb_m=max(0.0, cruise_altitude - origin.operation_altitude_m),
    descent_m=max(0.0, cruise_altitude - destination.operation_altitude_m),
)
```

- [ ] **Step 4: 运行航段测试和 D1 GIS 回归测试**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_physics.py tests/test_problem_d_q1_gis.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_geometry.py tests/test_problem_d_q2_physics.py
git commit -m "feat: add all-pairs D2 route geometry"
```

---

### Task 3: 多点架次物理评价和统一目标

**Files:**
- Create: `src/math_model_cup/problem_d_q2_physics.py`
- Modify: `tests/test_problem_d_q2_physics.py`

**Interfaces:**
- Consumes: `Q2Data`, `ArcGeometry`, `TripDraft`
- Produces: `evaluate_trip(data, arcs, draft, reserve_ratio=0.20) -> TripPlan`
- Produces: `charge_time_s(end_soc: float, full_charge_time_s: float) -> float`
- Produces: `objective_vector(solution: Q2Solution) -> tuple[float, float, float, int]`

- [ ] **Step 1: 写逐段减载和交付偏移测试**

```python
def test_two_stop_trip_reduces_payload_after_first_delivery(synthetic_q2) -> None:
    draft = TripDraft(
        model_id="C",
        stops=(
            TripStop("S001", ("B1",)),
            TripStop("S002", ("B2",)),
        ),
    )
    plan = evaluate_trip(synthetic_q2.data, synthetic_q2.arcs, draft)
    assert plan.leg_payloads_kg == pytest.approx((17.0, 14.0, 0.0))
    assert plan.delivery_offsets_s["B1"] < plan.delivery_offsets_s["B2"]
    assert plan.return_soc_percent >= 20.0
```

- [ ] **Step 2: 写两阶段充电边界测试**

```python
@pytest.mark.parametrize(
    ("soc", "expected"),
    [(0.0, 3000.0), (0.9, 1050.0), (1.0, 0.0)],
)
def test_charge_time_matches_statement(soc: float, expected: float) -> None:
    assert charge_time_s(soc, 3000.0) == pytest.approx(expected)
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_physics.py`

Expected: FAIL，提示物理评价函数不存在。

- [ ] **Step 4: 实现架次评价**

按访问顺序计算初始装载、逐段剩余载荷、装载时间、每站交接、交付完成偏移、总持续时间、能耗和返航 SOC。发现重复服务区、重复货箱、货箱目的地不匹配、超质量、超体积或能量越界时抛出 `InfeasibleQ2Error`。

及时性指标实现为：

```python
weighted_delivery = sum(
    box.priority_weight * delivery.delivery_time_s / box.expected_time_s
    for delivery in solution.deliveries
) / sum(box.priority_weight for box in data.boxes.values())
```

- [ ] **Step 5: 运行物理测试**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_physics.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_physics.py tests/test_problem_d_q2_physics.py
git commit -m "feat: evaluate multi-stop D2 trips"
```

---

### Task 4: 资源调度和独立校验

**Files:**
- Create: `src/math_model_cup/problem_d_q2_schedule.py`
- Create: `src/math_model_cup/problem_d_q2_validation.py`
- Create: `tests/test_problem_d_q2_schedule.py`
- Create: `tests/test_problem_d_q2_validation.py`

**Interfaces:**
- Produces: `schedule_trips_greedy(data, plans, method) -> Q2Solution`
- Produces: `schedule_trips_cp_sat(data, plans, method, time_limit_s) -> Q2Solution`
- Produces: `validate_q2_solution(data, arcs, solution, reserve_ratio=0.20) -> ValidationReport`

- [ ] **Step 1: 写无人机和电池周转测试**

```python
def test_scheduler_waits_for_compatible_full_battery(synthetic_schedule_case) -> None:
    solution = schedule_trips_greedy(
        synthetic_schedule_case.data,
        synthetic_schedule_case.plans,
        method="test",
    )
    first, second = solution.trips
    assert first.aircraft_id == second.aircraft_id
    assert second.start_time_s >= first.return_time_s
    if first.battery_id == second.battery_id:
        assert second.start_time_s >= first.battery_ready_time_s
```

- [ ] **Step 2: 写故障注入校验测试**

```python
def test_validator_rejects_duplicate_box_delivery(valid_solution_case) -> None:
    data, arcs, valid_solution = valid_solution_case
    broken = replace(
        valid_solution,
        deliveries=valid_solution.deliveries + (valid_solution.deliveries[0],),
    )
    report = validate_q2_solution(data, arcs, broken)
    assert not report.is_valid
    assert "duplicate_box" in {issue.code for issue in report.issues}
```

- [ ] **Step 3: 运行测试并确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_schedule.py tests/test_problem_d_q2_validation.py`

Expected: FAIL。

- [ ] **Step 4: 实现贪心和 CP-SAT 调度**

贪心调度对每个架次尝试全部兼容无人机和电池，取 `max(aircraft_available, battery_available)` 最小的组合。CP-SAT 对每个架次建立开始变量和无人机、电池可选区间；电池区间长度为架次持续时间加该架次固定充电时间。硬时限约束使用 `start + delivery_offset <= hard_deadline`。

- [ ] **Step 5: 实现独立校验器**

校验器必须重新调用基础航段能耗和时间公式，不使用 `TripPlan.energy_kwh` 作为可信输入。检查覆盖、目的地、容量、能量、SOC、时限、无人机重叠、电池重叠、充电完成时间、机型兼容和汇总一致性。

- [ ] **Step 6: 运行调度和校验测试**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_schedule.py tests/test_problem_d_q2_validation.py`

Expected: PASS。

- [ ] **Step 7: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_schedule.py src/math_model_cup/problem_d_q2_validation.py tests/test_problem_d_q2_schedule.py tests/test_problem_d_q2_validation.py
git commit -m "feat: schedule and validate D2 resources"
```

---

### Task 5: 候选架次法 MVP

**Files:**
- Create: `src/math_model_cup/problem_d_q2_candidates.py`
- Create: `tests/test_problem_d_q2_candidates.py`

**Interfaces:**
- Produces: `generate_initial_candidates(data, arcs, max_stops=3) -> tuple[TripPlan, ...]`
- Produces: `prune_dominated_candidates(candidates) -> tuple[TripPlan, ...]`
- Produces: `solve_candidate_method(data, arcs, time_limit_s=300) -> Q2Solution`

- [ ] **Step 1: 写覆盖与支配剪枝测试**

```python
def test_candidate_solver_covers_each_box_once(synthetic_q2) -> None:
    solution = solve_candidate_method(
        synthetic_q2.data, synthetic_q2.arcs, time_limit_s=5
    )
    delivered = [record.box_id for record in solution.deliveries]
    assert sorted(delivered) == sorted(synthetic_q2.data.boxes)
    assert len(delivered) == len(set(delivered))
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_candidates.py`

Expected: FAIL。

- [ ] **Step 3: 实现初始候选池**

先为每个服务区复用 D1 动态规划组批，形成单点候选；再对地理距离较近且时限相容的单点候选尝试两点合并和两种访问顺序。仅保留通过 `evaluate_trip` 的候选。

- [ ] **Step 4: 实现集合划分和调度反馈**

使用 `scipy.optimize.milp` 选择候选，约束每个货箱恰好覆盖一次。目标使用及时性近似、持续时间、能耗和架次数经过锚点归一化后的词典序大权重。选中候选交给 CP-SAT 调度；若调度不可行，加入所选组合的 no-good cut 后重求，最多10轮。

- [ ] **Step 5: 运行候选法测试**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_candidates.py`

Expected: PASS。

- [ ] **Step 6: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_candidates.py tests/test_problem_d_q2_candidates.py
git commit -m "feat: add D2 candidate trip solver"
```

---

### Task 6: 受限一体化 MILP MVP

**Files:**
- Create: `src/math_model_cup/problem_d_q2_milp.py`
- Create: `tests/test_problem_d_q2_solvers.py`

**Interfaces:**
- Produces: `solve_integrated_milp(data, arcs, time_limit_s=1800, max_stops=3) -> Q2Solution`

- [ ] **Step 1: 写一体化求解器合成测试**

```python
def test_integrated_milp_returns_valid_solution(synthetic_q2) -> None:
    solution = solve_integrated_milp(
        synthetic_q2.data,
        synthetic_q2.arcs,
        time_limit_s=10,
        max_stops=3,
    )
    report = validate_q2_solution(
        synthetic_q2.data, synthetic_q2.arcs, solution
    )
    assert report.is_valid
    assert solution.method == "integrated_milp"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_solvers.py -k integrated`

Expected: FAIL。

- [ ] **Step 3: 实现受限一体化模型**

从公共候选生成器取得只含单点和两点的物理可行架次，但在一个 MILP 中同时决定候选选择、实体无人机、电池、开始时间和资源先后关系。每个候选固定持续时间、交付偏移、能耗和充电时间，使所有约束保持线性。该 MVP 的“一体化”指路线选择和资源调度在同一个模型中完成；阶段二再实现基于架次槽和载荷状态的增强模型。

必须输出求解状态、上界、下界、MIP gap 和时间限制。没有可行 incumbent 时抛出 `InfeasibleQ2Error`，不得静默回退到其他方法。

- [ ] **Step 4: 运行一体化测试**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_solvers.py -k integrated`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_milp.py tests/test_problem_d_q2_solvers.py
git commit -m "feat: add restricted integrated D2 MILP"
```

---

### Task 7: ALNS MVP

**Files:**
- Create: `src/math_model_cup/problem_d_q2_alns.py`
- Modify: `tests/test_problem_d_q2_solvers.py`

**Interfaces:**
- Produces: `solve_alns(data, arcs, seed=0, iterations=10000) -> Q2Solution`

- [ ] **Step 1: 写可复现性和可行性测试**

```python
def test_alns_is_reproducible_and_valid(synthetic_q2) -> None:
    first = solve_alns(synthetic_q2.data, synthetic_q2.arcs, seed=7, iterations=100)
    second = solve_alns(synthetic_q2.data, synthetic_q2.arcs, seed=7, iterations=100)
    assert first.objective == pytest.approx(second.objective)
    assert validate_q2_solution(synthetic_q2.data, synthetic_q2.arcs, first).is_valid
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_solvers.py -k alns`

Expected: FAIL。

- [ ] **Step 3: 实现 ALNS 最小算子集**

实现第一问单点方案初始化、随机移除、最差及时性移除、相关服务区移除、最小增量插入、Regret-2 插入、新建架次修复、模拟退火接受和自适应算子权重。每次接受新方案前必须通过公共架次评价器和资源调度器。

- [ ] **Step 4: 运行 ALNS 测试**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_solvers.py -k alns`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_alns.py tests/test_problem_d_q2_solvers.py
git commit -m "feat: add reproducible D2 ALNS solver"
```

---

### Task 8: 混合算法 MVP

**Files:**
- Create: `src/math_model_cup/problem_d_q2_hybrid.py`
- Modify: `tests/test_problem_d_q2_solvers.py`

**Interfaces:**
- Produces: `solve_hybrid(data, arcs, seed=0, iterations=10000, rounds=2) -> Q2Solution`

- [ ] **Step 1: 写回灌与可行性测试**

```python
def test_hybrid_returns_valid_solution_and_records_rounds(synthetic_q2) -> None:
    solution = solve_hybrid(
        synthetic_q2.data,
        synthetic_q2.arcs,
        seed=11,
        iterations=100,
        rounds=2,
    )
    assert validate_q2_solution(synthetic_q2.data, synthetic_q2.arcs, solution).is_valid
    assert solution.method == "hybrid"
    assert solution.diagnostics["rounds_completed"] == 2
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_solvers.py -k hybrid`

Expected: FAIL。

- [ ] **Step 3: 实现两轮混合流程**

第一轮运行候选法并精确调度；ALNS 从该解出发搜索并收集不重复的新 `TripPlan.signature`；新架次加入候选池后重求集合划分和 CP-SAT 调度。保留可行的非支配方案，并按统一 epsilon 词典序选择返回值。

- [ ] **Step 4: 运行混合算法测试**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_solvers.py -k hybrid`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_hybrid.py tests/test_problem_d_q2_solvers.py
git commit -m "feat: add D2 hybrid solver"
```

---

### Task 9: 报告、命令行入口和提交表源数据

**Files:**
- Create: `src/math_model_cup/problem_d_q2_reporting.py`
- Create: `scripts/run_problem_d_q2.py`
- Create: `tests/test_problem_d_q2_integration.py`

**Interfaces:**
- Produces: `write_q2_outputs(output_dir, data, arcs, solutions) -> None`
- Produces: `run_methods(problem_dir, output_dir, methods, seed, quick) -> Mapping[str, Q2Solution]`

- [ ] **Step 1: 写输出结构测试**

```python
def test_reporting_writes_submission_source_tables(tmp_path, valid_solution_case) -> None:
    data, arcs, valid_solution = valid_solution_case
    write_q2_outputs(
        tmp_path,
        data,
        arcs,
        {valid_solution.method: valid_solution},
    )
    assert (tmp_path / valid_solution.method / "trips.csv").exists()
    assert (tmp_path / valid_solution.method / "deliveries.csv").exists()
    assert (tmp_path / "method_comparison.csv").exists()
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_integration.py`

Expected: FAIL。

- [ ] **Step 3: 实现报告与 CLI**

CLI 支持：

```text
--problem-dir
--output-dir
--methods integrated_milp,candidate,alns,hybrid
--seed
--quick
--time-limit
```

每种方法输出 `trips.csv`、`deliveries.csv`、`aircraft_timeline.csv`、`battery_timeline.csv`、`summary.json`、`validation.csv` 和 `constraint_check.csv`。根目录输出 `method_comparison.csv` 和 `summary.md`。

- [ ] **Step 4: 运行合成集成测试**

Run: `$env:PYTHONPATH='src'; python -m pytest -q tests/test_problem_d_q2_integration.py`

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add src/math_model_cup/problem_d_q2_reporting.py scripts/run_problem_d_q2.py tests/test_problem_d_q2_integration.py
git commit -m "feat: add D2 runner and reports"
```

---

### Task 10: 阶段一真实数据验收

**Files:**
- Modify: `tests/test_problem_d_q2_integration.py`
- Create: `D题第二问阶段一完成说明.md`

**Interfaces:**
- Consumes: 全部阶段一模块
- Produces: `outputs/problem_d_q2/` 中的四方法结果和阶段一完成说明

- [ ] **Step 1: 增加真实数据短时测试**

```python
@pytest.mark.skipif(not os.environ.get("D_PROBLEM_DIR"), reason="D_PROBLEM_DIR is not configured")
def test_real_data_all_four_methods_return_valid_solutions() -> None:
    data = load_q2_data(Path(os.environ["D_PROBLEM_DIR"]))
    arcs = build_arc_matrix(data)
    solutions = run_methods(
        Path(os.environ["D_PROBLEM_DIR"]),
        Path("outputs/problem_d_q2_test"),
        methods=("integrated_milp", "candidate", "alns", "hybrid"),
        seed=20260924,
        quick=True,
    )
    assert set(solutions) == {"integrated_milp", "candidate", "alns", "hybrid"}
    assert all(validate_q2_solution(data, arcs, solution).is_valid for solution in solutions.values())
```

- [ ] **Step 2: 运行全部自动测试**

Run: `$env:PYTHONPATH='src'; $env:D_PROBLEM_DIR='D题'; python -m pytest -q`

Expected: 所有 D1 和 D2 测试通过，无失败。

- [ ] **Step 3: 运行阶段一实际求解**

Run: `python scripts/run_problem_d_q2.py --problem-dir 'D题' --output-dir 'outputs/problem_d_q2' --methods integrated_milp,candidate,alns,hybrid --seed 20260924 --quick`

Expected: 四种方法均完成，命令行打印各方法目标向量、运行时间和 `validation=PASS`。

- [ ] **Step 4: 执行独立结果复核**

检查每种方法的：

```text
货箱覆盖数 = 80
重复货箱数 = 0
缺失货箱数 = 0
硬时限违约数 = 0
无人机冲突数 = 0
电池冲突数 = 0
能量违规数 = 0
最低返航 SOC >= 20%
```

- [ ] **Step 5: 编写阶段一完成说明**

说明四种 MVP 的模型边界、运行命令、目标结果、运行时间、MIP gap、已知限制和进入阶段二前需要增强的内容。

- [ ] **Step 6: 提交阶段一成果**

```powershell
git add environment.yml src/math_model_cup/problem_d_q2*.py scripts/run_problem_d_q2.py tests/test_problem_d_q2*.py D题第二问阶段一完成说明.md
git commit -m "feat: complete D2 four-solver MVP"
```

---

## 最终验证命令

```powershell
$env:PYTHONPATH = "src"
$env:D_PROBLEM_DIR = "D题"
python -m pytest -q tests -k "problem_d_q1 or problem_d_q2"
python scripts/run_problem_d_q2.py --problem-dir "D题" --output-dir "outputs/problem_d_q2" --methods integrated_milp,candidate,alns,hybrid --seed 20260924 --quick
```

阶段一完成时必须有最新命令输出证明测试通过和四种方法结果均通过独立校验；不能以先前运行结果或仅生成文件作为完成证据。
