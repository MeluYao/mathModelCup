# D题第三问对齐新版第二问实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将新版第二问的 Hybrid 24架次方案和 ALNS代表的22架次方案分别接入第三问联合调度，输出经独立验证的对比结果与最终方案。

**Architecture:** 从最新 `main` 创建隔离工作树，只迁移旧第三问的通用通信、中继和校验模块，不回退任何第二问代码。新增第二问落盘方案适配层、单场景第三问求解层和双场景报告层；两个场景独立重建轨迹、通信段和中继排程，最终只在验证通过的方案之间按第三问五维目标比较。

**Tech Stack:** Python 3.11、pandas、NumPy、SciPy、OR-Tools CP-SAT、pytest、现有 Q2/Q3 数据类和独立校验器。

## Global Constraints

- 所有源码、Markdown 和 CSV 使用 UTF-8；提交 CSV 时保持现有 `utf-8-sig` 输出习惯。
- 基线提交为包含新版第二问的 `cbd1fd7` 之后最新 `main`。
- 不修改或覆盖 `outputs/problem_d_q2_stage3/**`。
- 不覆盖旧第三问 `outputs/problem_d_q3/results/**`，新版写入 `outputs/problem_d_q3_aligned/**`。
- 只比较通过独立验证且 `is_valid=true` 的第三问方案。
- 中继资源固定为题目给定的2架中继无人机和6个能源单元。
- 运输与中继返航余量均不低于20%。
- 不宣称全局最优；求解器未找到可行解时使用 `NO_FEASIBLE_SOLUTION_FOUND`，不写成数学不可行证明。
- 不提交或改动当前主工作区未跟踪的第四问文件。

## 文件结构

- 从旧第三问提交 `8d3c44e` 迁移：`problem_d_q3.py`、通信、轨迹、中继候选、中继排程、联合调整和独立验证模块，以及对应测试。
- 新建 `problem_d_q3_transport_adapter.py`，负责读取并复核第二问落盘解。
- 新建 `problem_d_q3_aligned.py`，负责准备通信模型并求解单个第三问场景。
- 新建 `problem_d_q3_reporting.py`，负责写出场景结果、比较表和最终选择。
- 新建 `scripts/run_problem_d_q3_aligned.py`，作为正式双场景命令行入口。
- 新建适配器、联合求解、报告和真实结果测试。
- 更新 `docs/problem_d_q3_completion_report.md`。

---

### Task 1: 在最新主分支迁移第三问基础模块

**Files:**
- Create from `8d3c44e`: `src/math_model_cup/problem_d_q3.py`
- Create from `8d3c44e`: `src/math_model_cup/problem_d_q3_communication.py`
- Create from `8d3c44e`: `src/math_model_cup/problem_d_q3_trajectory.py`
- Create from `8d3c44e`: `src/math_model_cup/problem_d_q3_relay_candidates.py`
- Create from `8d3c44e`: `src/math_model_cup/problem_d_q3_relay_schedule.py`
- Create from `8d3c44e`: `src/math_model_cup/problem_d_q3_joint_alns.py`
- Create from `8d3c44e`: `src/math_model_cup/problem_d_q3_validation.py`
- Create from `8d3c44e`: `tests/test_problem_d_q3_communication.py`
- Create from `8d3c44e`: `tests/test_problem_d_q3_data.py`
- Create from `8d3c44e`: `tests/test_problem_d_q3_trajectory.py`
- Create from `8d3c44e`: `tests/test_problem_d_q3_relay_candidates.py`
- Create from `8d3c44e`: `tests/test_problem_d_q3_relay_schedule.py`
- Create from `8d3c44e`: `tests/test_problem_d_q3_joint_alns.py`
- Modify: `tests/conftest.py`

**Interfaces:**
- Consumes: 最新主分支的 `Q2Data`、`Q2Solution`、`evaluate_trip`、`charge_time_s`、`validate_q2_solution`。
- Produces: `load_q3_data`、`CommunicationEngine`、`build_trip_trajectory`、`build_direct_segments`、`generate_relay_states`、`build_coverage_atlas`、`solve_relay_subproblem`、`validate_q3_solution`。

- [ ] **Step 1: 创建隔离工作树**

使用 `using-git-worktrees` 技能从最新 `main` 创建 `codex/d-q3-q2-alignment` 工作树，名称 `d-q3-q2-alignment-v2`。确认工作树 HEAD 包含提交 `cbd1fd7` 和设计提交 `8e21236`。

- [ ] **Step 2: 从旧第三问提交恢复限定文件**

```powershell
git restore --source 8d3c44e -- `
  src/math_model_cup/problem_d_q3.py `
  src/math_model_cup/problem_d_q3_communication.py `
  src/math_model_cup/problem_d_q3_trajectory.py `
  src/math_model_cup/problem_d_q3_relay_candidates.py `
  src/math_model_cup/problem_d_q3_relay_schedule.py `
  src/math_model_cup/problem_d_q3_joint_alns.py `
  src/math_model_cup/problem_d_q3_validation.py `
  tests/test_problem_d_q3_communication.py `
  tests/test_problem_d_q3_data.py `
  tests/test_problem_d_q3_trajectory.py `
  tests/test_problem_d_q3_relay_candidates.py `
  tests/test_problem_d_q3_relay_schedule.py `
  tests/test_problem_d_q3_joint_alns.py
```

不得恢复旧提交中的 `problem_d_q2_*.py`、`.gitignore` 或整份 `tests/conftest.py`。

- [ ] **Step 3: 合并 Q3 测试夹具**

在当前 `tests/conftest.py` 追加以下夹具，保留原有内容：

```python
@pytest.fixture(scope="session")
def q3_data(d_problem_dir: Path):
    from math_model_cup.problem_d_q3 import load_q3_data

    return load_q3_data(d_problem_dir)
```

- [ ] **Step 4: 运行迁移后的定向测试**

```powershell
$env:D_PROBLEM_DIR = 'D:\MProject\mathModelCup\D题'
python -m pytest -q tests/test_problem_d_q3_communication.py tests/test_problem_d_q3_data.py tests/test_problem_d_q3_trajectory.py tests/test_problem_d_q3_relay_candidates.py tests/test_problem_d_q3_relay_schedule.py tests/test_problem_d_q3_joint_alns.py
```

Expected: 所有 Q3 定向测试通过；若新版 Q2 签名变化导致失败，只做兼容修复，不复制旧 Q2 实现。

- [ ] **Step 5: 提交基础迁移**

```powershell
git add src/math_model_cup/problem_d_q3*.py tests/conftest.py tests/test_problem_d_q3*.py
git commit -m "feat: migrate q3 foundation onto updated q2"
```

---

### Task 2: 实现新版第二问落盘方案适配器

**Files:**
- Create: `src/math_model_cup/problem_d_q3_transport_adapter.py`
- Create: `tests/test_problem_d_q3_transport_adapter.py`

**Interfaces:**
- Consumes: `Q2Data`、`ArcGeometry`、第二问 `summary.json`、`trips.csv`、`deliveries.csv`。
- Produces: `TransportScenario` 和 `load_transport_scenario(data, arcs, result_dir, scenario_id) -> TransportScenario`。

- [ ] **Step 1: 写失败测试，冻结适配器行为**

```python
def test_load_transport_scenario_reconstructs_valid_hybrid(q2_data, q2_arcs):
    scenario = load_transport_scenario(
        q2_data,
        q2_arcs,
        Path("outputs/problem_d_q2_stage3/hybrid"),
        "hybrid",
    )
    assert scenario.scenario_id == "hybrid"
    assert scenario.q2_method == "hybrid"
    assert len(scenario.solution.trips) == 24
    assert scenario.source_objective == pytest.approx(
        (0.5158908995531455, 9534.345058907296, 69.52235221186811, 24)
    )
    assert scenario.provenance["incumbent_source"] == "hybrid_search"
    assert scenario.provenance["native_improved_seed"] is True
    assert validate_q2_solution(q2_data, q2_arcs, scenario.solution).is_valid


def test_load_transport_scenario_rejects_tampered_objective(tmp_path, q2_data, q2_arcs):
    result_dir = copy_q2_result(tmp_path, "hybrid")
    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    summary["objective"]["energy_kwh"] = 0.0
    (result_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(Q2ResultIntegrityError, match="objective"):
        load_transport_scenario(q2_data, q2_arcs, result_dir, "hybrid")
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest -q tests/test_problem_d_q3_transport_adapter.py`

Expected: FAIL，提示模块或函数不存在。

- [ ] **Step 3: 实现数据类、哈希和重建逻辑**

```python
@dataclass(frozen=True)
class TransportScenario:
    scenario_id: str
    q2_method: str
    source_dir: Path
    source_objective: tuple[float, float, float, int]
    solution: Q2Solution
    provenance: Mapping[str, object]


class Q2ResultIntegrityError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
```

`load_transport_scenario` 必须按 `trips.csv` 的 `stop_sequence` 排序服务区，再从 `deliveries.csv` 按 `trip_id + service_id` 恢复每站箱号；使用 `evaluate_trip` 重算 `TripPlan`，使用 CSV 中的开始时刻、无人机和电池构造 `TripExecution`，通过 `_make_solution` 重算交付记录和四维目标。返回前执行：

```python
report = validate_q2_solution(data, arcs, reconstructed)
if not report.is_valid:
    raise Q2ResultIntegrityError(
        "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
    )
if not np.allclose(reconstructed.objective[:3], source_objective[:3], rtol=0, atol=1e-7):
    raise Q2ResultIntegrityError("reconstructed objective does not match summary objective")
if reconstructed.objective[3] != source_objective[3]:
    raise Q2ResultIntegrityError("reconstructed trip count does not match summary objective")
```

`provenance` 至少保存 `incumbent_source`、`native_improved_seed`、`seed_retained` 以及三个输入文件的 SHA-256。

- [ ] **Step 4: 增加22架次代表方案测试**

```python
@pytest.mark.parametrize("method", ["alns", "candidate", "integrated_milp"])
def test_common_seed_methods_reconstruct_same_22_trip_solution(method, q2_data, q2_arcs):
    scenario = load_transport_scenario(
        q2_data,
        q2_arcs,
        Path("outputs/problem_d_q2_stage3") / method,
        method,
    )
    assert len(scenario.solution.trips) == 22
    assert scenario.provenance["incumbent_source"] == "provided_seed"
    assert scenario.provenance["seed_retained"] is True
```

- [ ] **Step 5: 运行测试并提交**

Run: `python -m pytest -q tests/test_problem_d_q3_transport_adapter.py`

Expected: PASS。

```powershell
git add src/math_model_cup/problem_d_q3_transport_adapter.py tests/test_problem_d_q3_transport_adapter.py
git commit -m "feat: load validated q2 scenarios for q3"
```

---

### Task 3: 封装单场景第三问联合求解

**Files:**
- Create: `src/math_model_cup/problem_d_q3_aligned.py`
- Modify: `src/math_model_cup/problem_d_q3_joint_alns.py`
- Create: `tests/test_problem_d_q3_aligned.py`

**Interfaces:**
- Consumes: `Q3Data`、`ArcGeometry`、`TransportScenario`。
- Produces: `PreparedCommunicationModel`、`AlignedScenarioResult`、`prepare_communication_model(...)`、`solve_aligned_scenario(...)`、`select_best_valid_result(...)`。

- [ ] **Step 1: 写失败测试，冻结场景隔离和选择规则**

```python
def test_select_best_valid_result_uses_q3_lexicographic_objective():
    slower_but_timely = fake_result("hybrid", (0.50, 10000.0, 80.0, 24, 8), True)
    faster = fake_result("alns", (0.51, 9000.0, 70.0, 22, 7), True)
    assert select_best_valid_result((faster, slower_but_timely)).scenario_id == "hybrid"


def test_select_best_valid_result_ignores_invalid_result():
    invalid = fake_result("hybrid", (0.40, 8000.0, 60.0, 24, 6), False)
    valid = fake_result("alns", (0.51, 9000.0, 70.0, 22, 7), True)
    assert select_best_valid_result((invalid, valid)).scenario_id == "alns"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest -q tests/test_problem_d_q3_aligned.py`

Expected: FAIL，提示新模块不存在。

- [ ] **Step 3: 实现准备模型与结果类型**

```python
@dataclass(frozen=True)
class PreparedCommunicationModel:
    communication_segments: tuple[CommunicationSegment, ...]
    dark_segments: tuple[CommunicationSegment, ...]
    relay_states: tuple[RelayState, ...]
    atlas: CoverageAtlas


@dataclass(frozen=True)
class AlignedScenarioResult:
    scenario_id: str
    q2_method: str
    q2_objective: tuple[float, float, float, int]
    solution: Q3Solution | None
    validation: Q3ValidationReport | None
    status: str
    reason: str
    runtime_s: float
    provenance: Mapping[str, object]

    @property
    def is_valid(self) -> bool:
        return self.solution is not None and self.validation is not None and self.validation.is_valid
```

`prepare_communication_model` 对每个运输架次调用 `build_trip_trajectory` 和 `build_direct_segments(max_duration_s=30.0)`，随后按固定参数 `coarse_stride_pixels=32`、高度 `(100, 200, 300)`、`max_horizontal_points=150`、`redundancy=3`、`max_states=30` 生成并剪枝中继状态。

- [ ] **Step 4: 实现单场景求解流程**

`solve_aligned_scenario` 依次执行：准备通信模型；调用段级状态网格排程；对移动后的运输解重新生成通信段；调用 `solve_relay_subproblem(max_sorties=300, stop_after_first_solution=True)`；必要时用 `_window_for_segments` 与 `assemble_relay_campaign_solution` 做确定性 campaign 修复；最后调用 `validate_q3_solution`。

段级状态网格函数签名固定为：

```python
def reschedule_transport_with_segment_state_grid(
    data: Q3Data,
    transport: Q2Solution,
    communication_segments: Sequence[CommunicationSegment],
    relay_states: Sequence[RelayState],
    atlas: CoverageAtlas,
    *,
    hard_grid_step_s: float = 60.0,
    soft_grid_step_s: float = 300.0,
    occupancy_slot_s: float = 30.0,
    soft_horizon_s: float = 30_000.0,
    time_limit_s: float = 180.0,
) -> Q2Solution:
    """Choose transport starts and one covering relay state per dark segment."""
```

不得退化为“每个运输架次固定一个中继状态”；每个暗区原子段都必须有独立状态变量。无可行解异常转换为 `AlignedScenarioResult(status="NO_FEASIBLE_SOLUTION_FOUND")`。

- [ ] **Step 5: 实现有效方案选择**

```python
def select_best_valid_result(results: Sequence[AlignedScenarioResult]) -> AlignedScenarioResult:
    valid = [result for result in results if result.is_valid]
    if not valid:
        reasons = "; ".join(f"{r.scenario_id}: {r.reason}" for r in results)
        raise RuntimeError(f"no valid aligned q3 result: {reasons}")
    return min(valid, key=lambda result: result.solution.objective.as_tuple())
```

- [ ] **Step 6: 运行定向测试并提交**

```powershell
python -m pytest -q tests/test_problem_d_q3_aligned.py tests/test_problem_d_q3_joint_alns.py
git add src/math_model_cup/problem_d_q3_aligned.py src/math_model_cup/problem_d_q3_joint_alns.py tests/test_problem_d_q3_aligned.py
git commit -m "feat: solve q3 scenarios from fixed q2 routes"
```

Expected: PASS。

---

### Task 4: 实现输出、比较表和正式命令行入口

**Files:**
- Create: `src/math_model_cup/problem_d_q3_reporting.py`
- Create: `scripts/run_problem_d_q3_aligned.py`
- Create: `tests/test_problem_d_q3_reporting.py`

**Interfaces:**
- Consumes: `AlignedScenarioResult` 序列。
- Produces: `write_scenario_result(result, output_dir)`、`write_aligned_comparison(results, selected, output_dir)`、稳定的命令行入口。

- [ ] **Step 1: 写失败测试，冻结输出结构**

```python
def test_write_aligned_results_creates_complete_files(tmp_path, valid_aligned_result):
    scenario_dir = write_scenario_result(valid_aligned_result, tmp_path / "hybrid")
    assert {path.name for path in scenario_dir.iterdir()} >= {
        "summary.json", "transport_schedule.csv", "deliveries.csv",
        "relay_schedule.csv", "communication_assignments.csv", "q3_solution.pkl",
    }


def test_comparison_records_both_inputs_and_selected_scenario(tmp_path, two_results):
    selected = select_best_valid_result(two_results)
    write_aligned_comparison(two_results, selected, tmp_path)
    comparison = pd.read_csv(tmp_path / "comparison.csv")
    assert set(comparison["scenario_id"]) == {"hybrid", "alns"}
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest -q tests/test_problem_d_q3_reporting.py`

Expected: FAIL，提示报告模块不存在。

- [ ] **Step 3: 实现场景结果写出**

沿用旧结果字段，通过 `dataclasses.asdict` 写出中继和通信明细。`summary.json` 必须新增场景 ID、第二问方法、第二问目标、输入哈希、验证问题、第三问五维目标、运输/中继架次数和通信段计数。

- [ ] **Step 4: 实现比较表和 selected 稳定引用**

`comparison.csv` 每个场景一行，包含第二问四维目标、第三问五维目标、校验状态、通信段数、直连/中继段数、中继/能源峰值、运行时间和失败原因。

`selected/selection.json` 保存：

```json
{
  "selected_scenario_id": "hybrid",
  "selected_result_directory": "../hybrid",
  "selection_rule": "lexicographic_q3_objective_among_independently_validated_results"
}
```

- [ ] **Step 5: 实现正式 CLI**

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--q2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", default="hybrid,alns")
    parser.add_argument("--joint-time-limit", type=float, default=180.0)
    parser.add_argument("--relay-time-limit", type=float, default=180.0)
    return parser.parse_args()
```

任一场景失败仍写入 comparison；至少一个场景有效时返回0，否则返回2。

- [ ] **Step 6: 运行测试并提交**

```powershell
python -m pytest -q tests/test_problem_d_q3_reporting.py tests/test_problem_d_q3_transport_adapter.py tests/test_problem_d_q3_aligned.py
git add src/math_model_cup/problem_d_q3_reporting.py scripts/run_problem_d_q3_aligned.py tests/test_problem_d_q3_reporting.py
git commit -m "feat: compare aligned q3 scenarios"
```

Expected: PASS。

---

### Task 5: 运行两套真实方案并独立复核

**Files:**
- Create: `outputs/problem_d_q3_aligned/hybrid/**`
- Create: `outputs/problem_d_q3_aligned/alns/**`
- Create: `outputs/problem_d_q3_aligned/comparison.csv`
- Create: `outputs/problem_d_q3_aligned/selected/selection.json`
- Create: `tests/test_problem_d_q3_real_outputs.py`

**Interfaces:**
- Consumes: `outputs/problem_d_q2_stage3/hybrid/**`、`outputs/problem_d_q2_stage3/alns/**`、真实 D 题数据。
- Produces: 两套可追溯第三问结果和最终选择。

- [ ] **Step 1: 运行正式双场景求解**

```powershell
python scripts/run_problem_d_q3_aligned.py `
  --problem-dir 'D:\MProject\mathModelCup\D题' `
  --q2-root 'D:\MProject\mathModelCup\outputs\problem_d_q2_stage3' `
  --output-dir 'outputs\problem_d_q3_aligned' `
  --scenarios 'hybrid,alns' `
  --joint-time-limit 180 `
  --relay-time-limit 180
```

Expected: 返回0；`comparison.csv` 有两行；至少一个方案 `is_valid=true`。

- [ ] **Step 2: 增加落盘结果复核测试**

```python
@pytest.mark.skipif(not os.environ.get("D_PROBLEM_DIR"), reason="D_PROBLEM_DIR not configured")
def test_real_aligned_outputs_reload_and_validate(d_problem_dir):
    root = Path("outputs/problem_d_q3_aligned")
    comparison = pd.read_csv(root / "comparison.csv")
    assert set(comparison["scenario_id"]) == {"hybrid", "alns"}
    assert comparison["is_valid"].any()
    selection = json.loads((root / "selected" / "selection.json").read_text("utf-8"))
    selected_dir = (root / "selected" / selection["selected_result_directory"]).resolve()
    with (selected_dir / "q3_solution.pkl").open("rb") as stream:
        solution = pickle.load(stream)
    assert len(solution.communication_assignments) > 0
```

测试必须从运输解重新构建轨迹和通信段，再调用 `validate_q3_solution`，不能只比较 pickle 与 summary。

- [ ] **Step 3: 运行真实数据集成测试**

```powershell
$env:D_PROBLEM_DIR = 'D:\MProject\mathModelCup\D题'
python -m pytest -q tests/test_problem_d_q3_real_outputs.py
```

Expected: PASS，验证问题数为0。

- [ ] **Step 4: 提交真实结果**

```powershell
git add outputs/problem_d_q3_aligned tests/test_problem_d_q3_real_outputs.py
git commit -m "results: add q3 comparison from updated q2"
```

---

### Task 6: 更新完成报告并执行全量验证

**Files:**
- Create or Modify: `docs/problem_d_q3_completion_report.md`

**Interfaces:**
- Consumes: 最终 `comparison.csv`、两个 `summary.json` 和 selected manifest。
- Produces: 与新版第二问一致的第三问完成报告。

- [ ] **Step 1: 从结果文件更新报告**

报告必须包含新版第二问算法与合作者参考关系、22/24架次输入目标、两套第三问结果、最终选择规则、独立验证、最优性边界和输出文件位置。删除把50架次旧方案描述为当前最终方案的句子；可将其保留为历史基线。

- [ ] **Step 2: 检查报告与结果一致性**

只读脚本断言报告包含两个场景 ID、选中方案 ID、两个五维目标；所有引用文件存在；无任何未完成标记；Markdown 标题无跳级。

- [ ] **Step 3: 运行完整测试**

```powershell
$env:D_PROBLEM_DIR = 'D:\MProject\mathModelCup\D题'
python -m pytest -q
```

Expected: 全部测试通过；允许明确记录的可选依赖跳过项，不允许失败。

- [ ] **Step 4: 执行最终结果复核**

```powershell
git diff --check
git status --short
```

确认只存在本任务预期文件；重新读取 `comparison.csv`、selected `summary.json` 和独立验证结果，记录最终五维目标、运输架次、中继架次、通信中断数和资源峰值。

- [ ] **Step 5: 提交报告**

```powershell
git add docs/problem_d_q3_completion_report.md
git commit -m "docs: report q3 results aligned with updated q2"
```
