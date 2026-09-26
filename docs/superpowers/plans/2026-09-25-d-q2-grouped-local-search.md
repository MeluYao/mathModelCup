# D题第二问组批局部搜索实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有统一物理核、资源调度器和独立校验器上实现单区 FFD 组批、跨区合并与可行局部搜索，并用其改进四种第二问方案及报告结果。

**Architecture:** 新增 `problem_d_q2_grouped.py`，只负责生成和改进 `TripPlan` 集合；所有容量、体积、航段、能耗和 SOC 判断继续调用 `evaluate_trip`，所有无人机、电池、充电和硬时限判断继续调用 `schedule_trips_greedy`。四个原求解器将增强构造解作为共同 incumbent，并与自身结果按现有目标向量比较。

**Tech Stack:** Python 3.11、pytest、现有 Q2 数据类、现有物理核与贪心/CP-SAT 调度器。

## Global Constraints

- 保持 UTF-8 与现有输出格式。
- 不复制或修改能耗、SOC、充电和硬时限公式。
- 不改动用户无关的未提交文件。
- 不推送远程。
- 结果必须通过现有独立校验器。

---

### Task 1: FFD 组批构造器

**Files:**
- Create: `src/math_model_cup/problem_d_q2_grouped.py`
- Create: `tests/test_problem_d_q2_grouped.py`

**Interfaces:**
- Consumes: `Q2Data`、`ArcGeometry`、`evaluate_trip`、`schedule_trips_greedy`。
- Produces: `build_grouped_initial_plans(data, arcs) -> tuple[TripPlan, ...]`。

- [ ] **Step 1: Write the failing test**

测试真实数据上每箱恰好覆盖一次、每个停靠点非空、架次数小于 80，且调度后通过独立校验。

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_problem_d_q2_grouped.py`

Expected: FAIL，因为 `problem_d_q2_grouped` 尚不存在。

- [ ] **Step 3: Write minimal implementation**

按服务区将硬时限箱优先、其余按质量和体积降序执行 first-fit decreasing；每次加入货箱时枚举机型并调用 `evaluate_trip`，选择能耗最低可行机型。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_problem_d_q2_grouped.py`

Expected: PASS。

### Task 2: 合并与局部搜索

**Files:**
- Modify: `src/math_model_cup/problem_d_q2_grouped.py`
- Modify: `tests/test_problem_d_q2_grouped.py`

**Interfaces:**
- Produces: `solve_grouped_local_search(data, arcs, seed, iterations, method) -> Q2Solution`。

- [ ] **Step 1: Write the failing test**

测试真实数据搜索结果不劣于 FFD 初始解、覆盖与资源约束通过、架次数不超过 25、无空停靠点、固定种子可复现。

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_problem_d_q2_grouped.py`

Expected: FAIL，因为局部搜索入口尚不存在。

- [ ] **Step 3: Write minimal implementation**

实现同区/跨区合并、整组移动、单箱移动、拆分、路线重排和机型重选；每个候选先用 `evaluate_trip` 校验，再用 `schedule_trips_greedy` 做完整资源仿真，只接受现有目标向量严格改进。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_problem_d_q2_grouped.py`

Expected: PASS。

### Task 3: 四方案接入增强 incumbent

**Files:**
- Modify: `src/math_model_cup/problem_d_q2_candidates.py`
- Modify: `src/math_model_cup/problem_d_q2_milp.py`
- Modify: `src/math_model_cup/problem_d_q2_alns.py`
- Modify: `src/math_model_cup/problem_d_q2_hybrid.py`
- Modify: `tests/test_problem_d_q2_solvers.py`
- Modify: `tests/test_problem_d_q2_integration.py`

**Interfaces:**
- 四个公开求解入口签名保持不变。
- 每个返回值的 `diagnostics` 记录 incumbent 来源和组批搜索指标。

- [ ] **Step 1: Write the failing test**

要求真实数据四方案都少于 80 架次、全部通过校验，且至少返回增强组批构造器或更优的自身解。

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest -q tests/test_problem_d_q2_solvers.py tests/test_problem_d_q2_integration.py`

Expected: FAIL，因为当前四方案仍退化为 80 个单箱架次。

- [ ] **Step 3: Write minimal implementation**

为四方案建立共同增强 incumbent；原算法结束后统一按目标向量选择较优可行解，并保留各自方法名和诊断信息。

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest -q tests/test_problem_d_q2_solvers.py tests/test_problem_d_q2_integration.py`

Expected: PASS。

### Task 4: 重跑实验与同步报告

**Files:**
- Modify: `outputs/problem_d_q2_stage3/**`
- Modify: `D题第二问方法与方案结果报告.md`
- Modify: `D题第二问论文正文.md`
- Modify: `D题第二问阶段三完成说明.md`

**Interfaces:**
- 输出目录结构与 CSV/JSON 字段保持兼容。

- [ ] **Step 1: Run the four methods and experiments**

Run: `python scripts/run_problem_d_q2.py --problem-dir D题 --output-dir outputs/problem_d_q2_stage3 --methods integrated_milp,candidate,alns,hybrid --seed 20260924 --quick`

- [ ] **Step 2: Run sensitivity and repeated experiments**

Run: `python scripts/run_problem_d_q2_experiments.py --problem-dir D题 --output-dir outputs/problem_d_q2_stage3`

- [ ] **Step 3: Update reports from generated metrics**

替换 80 单箱架次结论、指标表和推荐方案；明确新算法、约束验证、与旧结果对比及“未证明全局最优”。

- [ ] **Step 4: Full verification**

Run: `python -m pytest -q`

Expected: 全部测试通过；四方案输出的 validation 均为 PASS。
