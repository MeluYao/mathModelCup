# A题问题一算法实验平台 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建可运行、可复现的问题一算法实验平台，实现 C0、C1、C2、C3、C5 候选，并通过合成图和代表性正式算例逐轮筛选最终候选。

**Architecture:** 先把官方输入转换为统一的计算操作 DAG 与 tensor 超边模型，再让切图器、调度器、代理目标和官方评估器通过不可变数据对象组合。探索程序缓存每个方案的官方结果并执行连续淘汰；单算例 CLI 从同一候选生成管线输出题目规定的纯方案 JSON。

**Tech Stack:** Python 3.12 标准库、pytest、题目附件中的官方验证和问题一评估代码；不增加第三方运行依赖。

## Global Constraints

- 所有文本和 JSON 文件使用 UTF-8 编码。
- 不修改 `A题/通用神经网络处理器下的多核调度问题  附件` 中的官方代码。
- 正式排名只使用官方问题一评估器的 Makespan 和搬运统计。
- 每个候选必须通过覆盖、互斥、商图无环和同核顺序检查。
- 探索阶段允许使用代表性算例；只有冻结后的 2 至 3 个候选运行全部算例和 1 至 5 核。
- 现有 `A题/solver_minimal.py` 保留为 C0 基线，不重写其行为。
- 不提交或覆盖与本任务无关的工作区修改。

---

## File Structure

- `A题/problem1/__init__.py`：公开候选注册表和版本信息。
- `A题/problem1/graph_model.py`：计算 DAG、tensor 超边、拓扑序和图画像。
- `A题/problem1/plan_io.py`：子图方案数据结构、内部合法性检查和官方方案转换。
- `A题/problem1/objectives.py`：块特征、理论下界和可解释代理目标。
- `A题/problem1/partitioners.py`：C1、C2、C3 子图构造器。
- `A题/problem1/schedulers.py`：S1、S2 与有限前瞻调度器。
- `A题/problem1/evaluator.py`：官方评估调用、方案哈希和结果缓存。
- `A题/problem1/local_search.py`：C5 邻域生成与预算搜索。
- `A题/problem1/experiments.py`：正式图画像、代表集选择和连续淘汰。
- `A题/solver_problem1.py`：单图方案生成入口。
- `A题/experiment_problem1.py`：实验入口与汇总文件生成。
- `A题/tests/conftest.py`：测试导入路径和附件路径夹具。
- `A题/tests/graph_fixtures.py`：链、独立链、Fork-Join、高扇出和容量边界合成图。
- `A题/tests/test_graph_model.py`：图模型性质测试。
- `A题/tests/test_plan_io.py`：方案合法性和官方接口测试。
- `A题/tests/test_objectives.py`：代理目标方向性测试。
- `A题/tests/test_partitioners.py`：C1、C2、C3 行为测试。
- `A题/tests/test_schedulers.py`：S1、S2 和束搜索测试。
- `A题/tests/test_evaluator.py`：最小图官方评估和缓存测试。
- `A题/tests/test_local_search.py`：邻域合法性与改进接受规则测试。
- `A题/tests/test_experiments.py`：代表集和连续淘汰测试。
- `A题/tests/test_cli.py`：两个 CLI 的端到端测试。

---

### Task 1: 公共图模型与合成图

**Files:**
- Create: `A题/problem1/__init__.py`
- Create: `A题/problem1/graph_model.py`
- Create: `A题/tests/conftest.py`
- Create: `A题/tests/graph_fixtures.py`
- Create: `A题/tests/test_graph_model.py`

**Interfaces:**
- Produces: `GraphModel.from_json(graph: dict) -> GraphModel`
- Produces: `GraphModel.topological_order(policy: str) -> tuple[int, ...]`
- Produces: `GraphModel.profile() -> dict[str, float]`
- Produces: `make_chain_graph`, `make_parallel_chains_graph`, `make_fork_join_graph`, `make_fanout_graph`

- [ ] **Step 1: 写失败测试，定义图模型需要保留的依赖与超边语义**

```python
def test_fanout_tensor_is_one_hyperedge():
    graph = make_fanout_graph(branches=3, tensor_size=4096)
    model = GraphModel.from_json(graph)
    shared = next(t for t in model.tensors.values() if t.size == 4096)
    assert len(shared.producers) == 1
    assert len(shared.consumers) == 3
    assert set(model.topological_order("stable")) == set(model.ops)


def test_all_topological_policies_respect_dependencies():
    model = GraphModel.from_json(make_fork_join_graph())
    for policy in ("stable", "critical", "communication", "pipe"):
        order = model.topological_order(policy)
        position = {op_id: i for i, op_id in enumerate(order)}
        assert all(position[u] < position[v] for u in model.ops for v in model.succs[u])
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `python -m pytest 'A题/tests/test_graph_model.py' -q`

Expected: FAIL，提示 `ModuleNotFoundError: No module named 'problem1'`。

- [ ] **Step 3: 实现不可变图模型和四种确定性拓扑序**

```python
@dataclass(frozen=True)
class OpInfo:
    id: int
    op: str
    pipe: str
    cycles: int


@dataclass(frozen=True)
class TensorInfo:
    id: int
    pos: str
    size: int
    producers: frozenset[int]
    consumers: frozenset[int]


@dataclass(frozen=True)
class GraphModel:
    raw_graph: dict
    ops: Mapping[int, OpInfo]
    tensors: Mapping[int, TensorInfo]
    preds: Mapping[int, frozenset[int]]
    succs: Mapping[int, frozenset[int]]
    ranks_up: Mapping[int, int]
    ranks_down: Mapping[int, int]
    depth: Mapping[int, int]

    @classmethod
    def from_json(cls, graph: dict) -> "GraphModel":
        validate_graph(graph)
        op_by_id, stable, preds, succs = eligible_dag(graph)
        eligible = set(stable)
        producers, consumers = _tensor_endpoints(graph, eligible)
        tensors = {
            item["id"]: TensorInfo(
                id=item["id"], pos=item["pos"], size=int(item["size"]),
                producers=frozenset(producers[item["id"]]),
                consumers=frozenset(consumers[item["id"]]),
            )
            for item in graph["tensors"]
        }
        ops = {
            op_id: OpInfo(op_id, op_by_id[op_id]["op"], op_by_id[op_id]["pipe"],
                          max(1, int(op_by_id[op_id].get("cycles", 0))))
            for op_id in stable
        }
        ranks_up, ranks_down, depth = _dag_features(ops, preds, succs, stable)
        return cls(graph, ops, tensors, _freeze(preds), _freeze(succs),
                   ranks_up, ranks_down, depth)

    def topological_order(self, policy: str) -> tuple[int, ...]:
        priorities = _policy_priorities(self, policy)
        return _kahn_order(self.preds, self.succs, priorities)

    def profile(self) -> dict[str, float]:
        pipe_work = Counter()
        for op in self.ops.values():
            pipe_work[op.pipe] += op.cycles
        return {
            "num_ops": float(len(self.ops)),
            "num_edges": float(sum(map(len, self.succs.values()))),
            "depth": float(max(self.depth.values(), default=0)),
            "max_fanout": float(max((len(t.consumers) for t in self.tensors.values()), default=0)),
            "tensor_bytes": float(sum(t.size for t in self.tensors.values())),
            "pipe_m_work": float(pipe_work["PIPE_M"]),
            "pipe_v_work": float(pipe_work["PIPE_V"]),
        }
```

- [ ] **Step 4: 实现合成图构造器并验证每个图都通过官方 `validate_graph`**

每个构造器使用全局唯一 id，显式插入 DDR 输入、`COPY_IN`、内部 tensor、计算操作、`COPY_OUT` 和 DDR 输出。`make_parallel_chains_graph(branches, length)` 不在分支之间添加依赖；`make_fork_join_graph()` 创建一个生产者、两个并行分支和一个汇合操作；`make_fanout_graph()` 让一个内部 tensor 被多个分支消费。

- [ ] **Step 5: 运行 Task 1 测试**

Run: `python -m pytest 'A题/tests/test_graph_model.py' -q`

Expected: PASS。

- [ ] **Step 6: 提交 Task 1**

```powershell
git add -- 'A题/problem1/__init__.py' 'A题/problem1/graph_model.py' 'A题/tests/conftest.py' 'A题/tests/graph_fixtures.py' 'A题/tests/test_graph_model.py'
git commit -m "feat: add problem 1 graph model"
```

---

### Task 2: 子图方案结构和双重合法性验证

**Files:**
- Create: `A题/problem1/plan_io.py`
- Create: `A题/tests/test_plan_io.py`

**Interfaces:**
- Consumes: `GraphModel`
- Produces: `Partition.from_blocks(model, blocks, algorithm, parameters) -> Partition`
- Produces: `build_plan(model, partition, core_schedules) -> dict`
- Produces: `validate_plan(model, partition, core_schedules) -> None`
- Produces: `plan_hash(plan) -> str`

- [ ] **Step 1: 写失败测试覆盖遗漏操作、重复操作、商图环和合法方案**

```python
def test_partition_rejects_missing_or_duplicate_ops():
    model = GraphModel.from_json(make_chain_graph(length=3))
    ids = list(model.topological_order("stable"))
    with pytest.raises(ValueError, match="cover"):
        Partition.from_blocks(model, [ids[:-1]], "test", {})
    with pytest.raises(ValueError, match="duplicate"):
        Partition.from_blocks(model, [[ids[0], ids[1]], [ids[1], ids[2]]], "test", {})


def test_plan_passes_internal_and_official_validation():
    graph = make_fork_join_graph()
    model = GraphModel.from_json(graph)
    partition = Partition.from_blocks(model, [[x] for x in model.topological_order("stable")], "test", {})
    schedule = [list(range(len(partition.blocks))), []]
    plan = build_plan(model, partition, schedule)
    validate_plan(model, partition, schedule)
    derive_multicore_plan(graph, plan)
```

- [ ] **Step 2: 运行测试确认失败原因是 `Partition` 尚未实现**

Run: `python -m pytest 'A题/tests/test_plan_io.py' -q`

Expected: FAIL，提示无法导入 `Partition`。

- [ ] **Step 3: 实现分区数据结构、商图和内部检查**

```python
@dataclass(frozen=True)
class Partition:
    blocks: tuple[tuple[int, ...], ...]
    block_of: Mapping[int, int]
    predecessors: tuple[frozenset[int], ...]
    successors: tuple[frozenset[int], ...]
    boundary_bytes: Mapping[tuple[int, int], int]
    algorithm: str
    parameters: Mapping[str, object]

    @classmethod
    def from_blocks(cls, model, blocks, algorithm, parameters):
        normalized = tuple(tuple(block) for block in blocks if block)
        flat = [op_id for block in normalized for op_id in block]
        if len(flat) != len(set(flat)):
            raise ValueError("duplicate operation in partition")
        if set(flat) != set(model.ops):
            raise ValueError("partition does not cover eligible operations")
        block_of = {op_id: block_id for block_id, block in enumerate(normalized) for op_id in block}
        predecessors, successors = _contract_dag(model, block_of, len(normalized))
        _assert_acyclic(predecessors, successors)
        boundary = _boundary_tensor_bytes(model, block_of)
        return cls(normalized, MappingProxyType(block_of), predecessors, successors,
                   MappingProxyType(boundary), algorithm, MappingProxyType(dict(parameters)))
```

- [ ] **Step 4: 实现题目 JSON 转换、同核顺序检查和稳定 SHA-256 哈希**

`validate_plan` 检查核心列表合并后等于全部子图 id，并把每核顺序边加入商图后再次做拓扑检查。`build_plan` 只返回 `node_to_subgraph` 和 `core_schedules`。`plan_hash` 对排序键、无多余空格的 UTF-8 JSON 求 SHA-256。

- [ ] **Step 5: 运行 Task 2 测试**

Run: `python -m pytest 'A题/tests/test_plan_io.py' -q`

Expected: PASS。

- [ ] **Step 6: 提交 Task 2**

```powershell
git add -- 'A题/problem1/plan_io.py' 'A题/tests/test_plan_io.py'
git commit -m "feat: validate problem 1 partitions and plans"
```

---

### Task 3: 可解释代理目标与理论下界

**Files:**
- Create: `A题/problem1/objectives.py`
- Create: `A题/tests/test_objectives.py`

**Interfaces:**
- Consumes: `GraphModel`, `Partition`, `core_schedules`
- Produces: `block_statistics(model, partition) -> tuple[BlockStats, ...]`
- Produces: `estimate_schedule(model, partition, core_schedules, weights) -> Estimate`

- [ ] **Step 1: 写失败测试验证通信、等待和内存惩罚方向**

```python
def test_cutting_large_fanout_tensor_increases_ddr_bound():
    model = GraphModel.from_json(make_fanout_graph(branches=4, tensor_size=65536))
    one = Partition.from_blocks(model, [tuple(model.ops)], "one", {})
    split = Partition.from_blocks(model, [[x] for x in model.topological_order("stable")], "split", {})
    assert estimate_schedule(model, split, [list(range(len(split.blocks))), []], DEFAULT_WEIGHTS).ddr_bound > estimate_schedule(model, one, [[0], []], DEFAULT_WEIGHTS).ddr_bound


def test_over_capacity_block_has_positive_memory_penalty():
    model = GraphModel.from_json(make_capacity_graph(position="UB", live_bytes=140_000))
    partition = Partition.from_blocks(model, [tuple(model.ops)], "one", {})
    estimate = estimate_schedule(model, partition, [[0], []], DEFAULT_WEIGHTS)
    assert estimate.spill_penalty > 0
```

- [ ] **Step 2: 运行测试确认代理模块尚不存在**

Run: `python -m pytest 'A题/tests/test_objectives.py' -q`

Expected: FAIL，提示无法导入 `problem1.objectives`。

- [ ] **Step 3: 实现块特征和分量化估计结果**

```python
@dataclass(frozen=True)
class BlockStats:
    pipe_work: Mapping[str, int]
    internal_tensor_bytes: int
    incoming_bytes: int
    outgoing_bytes: int
    l1_peak: int
    ub_peak: int


@dataclass(frozen=True)
class Estimate:
    total: float
    critical_bound: float
    core_load_bound: float
    ddr_bound: float
    task_wait_penalty: float
    spill_penalty: float
    imbalance_penalty: float


DEFAULT_WEIGHTS = {
    "task_wait": 1.0,
    "spill": 1.0,
    "imbalance": 0.25,
}
```

`block_statistics` 按块内拓扑位置估计 tensor 从定义到最后使用的活跃区间，并分别扫描 L1、UB 峰值。`estimate_schedule` 用 60 B/cycle、100 cycles 和 1000 cycles 固定题面参数计算各分量。

- [ ] **Step 4: 运行方向性测试和全套现有测试**

Run: `python -m pytest 'A题/tests/test_objectives.py' 'A题/tests/test_graph_model.py' 'A题/tests/test_plan_io.py' -q`

Expected: PASS。

- [ ] **Step 5: 提交 Task 3**

```powershell
git add -- 'A题/problem1/objectives.py' 'A题/tests/test_objectives.py'
git commit -m "feat: add interpretable scheduling objective"
```

---

### Task 4: S1、S2 和有限前瞻调度器

**Files:**
- Create: `A题/problem1/schedulers.py`
- Create: `A题/tests/test_schedulers.py`

**Interfaces:**
- Produces: `schedule_heft(model, partition, num_cores, weights) -> list[list[int]]`
- Produces: `schedule_critical_fill(model, partition, num_cores, weights) -> list[list[int]]`
- Produces: `schedule_beam(model, partition, num_cores, weights, beam_width, lookahead) -> list[list[int]]`

- [ ] **Step 1: 写失败测试验证并行使用多核和确定性**

```python
@pytest.mark.parametrize("scheduler", [schedule_heft, schedule_critical_fill])
def test_parallel_chains_use_multiple_cores(scheduler):
    model = GraphModel.from_json(make_parallel_chains_graph(branches=4, length=3))
    partition = Partition.from_blocks(model, [[x] for x in model.topological_order("stable")], "singletons", {})
    schedule = scheduler(model, partition, 4, DEFAULT_WEIGHTS)
    validate_plan(model, partition, schedule)
    assert sum(bool(core) for core in schedule) >= 2
    assert schedule == scheduler(model, partition, 4, DEFAULT_WEIGHTS)


def test_beam_never_returns_worse_proxy_than_greedy():
    model, partition = fork_join_partition()
    greedy = schedule_heft(model, partition, 2, DEFAULT_WEIGHTS)
    beam = schedule_beam(model, partition, 2, DEFAULT_WEIGHTS, beam_width=4, lookahead=2)
    assert estimate_schedule(model, partition, beam, DEFAULT_WEIGHTS).total <= estimate_schedule(model, partition, greedy, DEFAULT_WEIGHTS).total
```

- [ ] **Step 2: 运行测试确认调度器尚未实现**

Run: `python -m pytest 'A题/tests/test_schedulers.py' -q`

Expected: FAIL，提示无法导入调度函数。

- [ ] **Step 3: 实现通信感知 HEFT**

维护未调度前驱数、ready 堆、每核尾部子图和预计完成时间。对每个 ready 子图枚举核心，将其临时追加到对应核心并调用增量估计；按 `(预计完成时间, 代理总分, core_id)` 选择。ready 优先级为下游关键路径、边界字节和子图 id 的确定性元组。

- [ ] **Step 4: 实现关键路径填充和有界束搜索**

S2 先固定商图最长路径，再按最早不延迟汇合点的核心放置其他子图。S3 以 S1 状态转移为基础，每层保留 `beam_width` 个状态，只展开 `lookahead` 层，然后采用最优首步并滚动执行；`beam_width=1` 时行为退化为贪心。

- [ ] **Step 5: 运行 Task 4 测试**

Run: `python -m pytest 'A题/tests/test_schedulers.py' -q`

Expected: PASS。

- [ ] **Step 6: 提交 Task 4**

```powershell
git add -- 'A题/problem1/schedulers.py' 'A题/tests/test_schedulers.py'
git commit -m "feat: add problem 1 multicore schedulers"
```

---

### Task 5: C1 多拓扑序动态规划切块

**Files:**
- Create: `A题/problem1/partitioners.py`
- Create: `A题/tests/test_partitioners.py`

**Interfaces:**
- Produces: `partition_dp(model, policy, target_work, max_ops, window, weights) -> Partition`
- Produces: `generate_c1_candidates(model, num_cores, config) -> Iterator[Partition]`

- [ ] **Step 1: 写失败测试验证无环、覆盖和弱边优先切分**

```python
def test_c1_is_valid_for_every_topological_policy():
    model = GraphModel.from_json(make_fork_join_graph())
    for policy in ("stable", "critical", "communication", "pipe"):
        partition = partition_dp(model, policy, target_work=20, max_ops=4, window=4,
                                 weights=DEFAULT_PARTITION_WEIGHTS)
        assert set(partition.block_of) == set(model.ops)
        assert partition.algorithm == "c1_dp"


def test_c1_prefers_small_boundary_when_work_is_equal():
    model = GraphModel.from_json(make_two_cut_choices_graph(small_bytes=64, large_bytes=65536))
    partition = partition_dp(model, "stable", target_work=10, max_ops=3, window=3,
                             weights=DEFAULT_PARTITION_WEIGHTS)
    assert cut_tensor_sizes(model, partition) == [64]
```

- [ ] **Step 2: 运行测试确认 C1 尚未实现**

Run: `python -m pytest 'A题/tests/test_partitioners.py::test_c1_is_valid_for_every_topological_policy' -q`

Expected: FAIL，提示 `partition_dp` 不存在。

- [ ] **Step 3: 实现有界动态规划**

```python
DEFAULT_PARTITION_WEIGHTS = {
    "work_balance": 1.0,
    "boundary_bytes": 1.0 / 60.0,
    "task_count": 100.0,
    "memory_risk": 1.0,
    "critical_cut": 1.0,
}


def partition_dp(model, policy, target_work, max_ops, window, weights):
    order = model.topological_order(policy)
    n = len(order)
    best = [math.inf] * (n + 1)
    parent = [-1] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        start_min = max(0, end - min(max_ops, window))
        for start in range(start_min, end):
            segment = order[start:end]
            cost = _segment_cost(model, order, start, end, target_work, weights)
            candidate = best[start] + cost
            if candidate < best[end]:
                best[end], parent[end] = candidate, start
    blocks = []
    cursor = n
    while cursor:
        start = parent[cursor]
        if start < 0:
            raise RuntimeError("dynamic partition has no feasible predecessor")
        blocks.append(order[start:cursor])
        cursor = start
    blocks.reverse()
    return Partition.from_blocks(model, blocks, "c1_dp", {
        "policy": policy, "target_work": target_work,
        "max_ops": max_ops, "window": window,
    })
```

`_segment_cost` 必须同时计算工作量偏差、跨段 tensor 字节、块数量、L1/UB 风险和切断关键路径的惩罚。

- [ ] **Step 4: 实现 C1 候选参数网格并去重**

对四种拓扑策略、三种目标工作量倍率和两种 `max_ops` 生成候选；用规范化 `blocks` 哈希去重。参数倍率基于总 Pipe 工作量除以核数，而不是写死只适用于某个算例的 cycles。

- [ ] **Step 5: 运行 C1 测试**

Run: `python -m pytest 'A题/tests/test_partitioners.py' -q -k c1`

Expected: PASS。

- [ ] **Step 6: 提交 Task 5**

```powershell
git add -- 'A题/problem1/partitioners.py' 'A题/tests/test_partitioners.py'
git commit -m "feat: add dynamic problem 1 partitioner"
```

---

### Task 6: C2 超图凝聚和 C3 Fork-Join 分解

**Files:**
- Modify: `A题/problem1/partitioners.py`
- Modify: `A题/tests/test_partitioners.py`

**Interfaces:**
- Produces: `partition_hypergraph(model, target_blocks, weights) -> Partition`
- Produces: `partition_fork_join(model, target_work, max_ops, weights) -> Partition`
- Produces: `generate_structural_candidates(model, num_cores, config) -> Iterator[Partition]`

- [ ] **Step 1: 写 C2 失败测试，验证高扇出 tensor 的边界减少**

```python
def test_c2_co_locates_consumers_of_large_fanout_tensor():
    model = GraphModel.from_json(make_fanout_graph(branches=4, tensor_size=65536))
    c2 = partition_hypergraph(model, target_blocks=2, weights=DEFAULT_PARTITION_WEIGHTS)
    c1 = partition_dp(model, "stable", target_work=1, max_ops=2, window=2,
                      weights=DEFAULT_PARTITION_WEIGHTS)
    assert total_boundary_bytes(c2) <= total_boundary_bytes(c1)
```

- [ ] **Step 2: 运行测试确认 `partition_hypergraph` 不存在**

Run: `python -m pytest 'A题/tests/test_partitioners.py::test_c2_co_locates_consumers_of_large_fanout_tensor' -q`

Expected: FAIL，提示无法导入或调用 C2。

- [ ] **Step 3: 实现只收缩商图直接边的超图凝聚**

初始簇为最大线性链片段；只把当前商图中存在直接依赖的两个簇作为合并候选。收缩 DAG 的直接边不会引入新环。优先队列键为 `(-gain, combined_size, left_id, right_id)`，过期候选通过簇版本号丢弃。gain 精确计算 tensor 消费簇集合变化导致的边界字节减少，并扣除工作量、并行度和内存风险。

- [ ] **Step 4: 写 C3 失败测试，验证 Fork-Join 分支保持并行**

```python
def test_c3_keeps_independent_fork_branches_separate():
    model = GraphModel.from_json(make_fork_join_graph(branch_length=4))
    partition = partition_fork_join(model, target_work=20, max_ops=8,
                                    weights=DEFAULT_PARTITION_WEIGHTS)
    left, right = fork_branch_ops(model)
    assert {partition.block_of[x] for x in left} != {partition.block_of[x] for x in right}
```

- [ ] **Step 5: 运行测试确认 C3 尚未实现**

Run: `python -m pytest 'A题/tests/test_partitioners.py::test_c3_keeps_independent_fork_branches_separate' -q`

Expected: FAIL，提示 `partition_fork_join` 不存在。

- [ ] **Step 6: 实现关键路径与 Fork-Join 分解**

先按 `ranks_down` 回溯一条确定性关键路径；把入度和出度均为 1 的连续操作压成链；在出度大于 1 的 fork 到最近公共汇合层之间禁止把不同首分支合并；未覆盖区域使用 C1 的关键路径拓扑序局部分块。超过 `max_ops` 或目标工作量的链按最小 tensor 边界拆分。

- [ ] **Step 7: 运行全部切图测试并检查确定性**

Run: `python -m pytest 'A题/tests/test_partitioners.py' -q`

Expected: PASS；同一输入重复运行得到完全相同的 `blocks`。

- [ ] **Step 8: 提交 Task 6**

```powershell
git add -- 'A题/problem1/partitioners.py' 'A题/tests/test_partitioners.py'
git commit -m "feat: add structural problem 1 partitioners"
```

---

### Task 7: 官方评估封装和内容寻址缓存

**Files:**
- Create: `A题/problem1/evaluator.py`
- Create: `A题/tests/test_evaluator.py`

**Interfaces:**
- Produces: `OfficialResult`
- Produces: `evaluate_plan(graph_path, plan, config_path, work_dir, timeout_s) -> OfficialResult`
- Produces: `EvaluationCache(path).get_or_evaluate(...) -> OfficialResult`

- [ ] **Step 1: 写失败测试，以题面最小图验证 Makespan 等于 6**

```python
def test_official_minimal_graph_makespan_is_six(tmp_path, minimal_graph_file, config_path):
    graph = json.loads(minimal_graph_file.read_text(encoding="utf-8"))
    model = GraphModel.from_json(graph)
    partition = Partition.from_blocks(model, [tuple(model.ops)], "test", {})
    plan = build_plan(model, partition, [[0], []])
    result = evaluate_plan(minimal_graph_file, plan, config_path, tmp_path, timeout_s=30)
    assert result.makespan == 6
    assert result.added_copy_bytes == 0
```

- [ ] **Step 2: 运行测试确认评估封装不存在**

Run: `python -m pytest 'A题/tests/test_evaluator.py::test_official_minimal_graph_makespan_is_six' -q`

Expected: FAIL，提示无法导入 `evaluate_plan`。

- [ ] **Step 3: 实现隔离的官方 CLI 调用**

```python
@dataclass(frozen=True)
class OfficialResult:
    makespan: int
    added_copy_bytes: int
    partition_added_copy_bytes: int
    spill_added_copy_bytes: int
    elapsed_seconds: float
    raw: Mapping[str, object]


def evaluate_plan(graph_path, plan, config_path, work_dir, timeout_s):
    key = plan_hash(plan)
    run_dir = Path(work_dir) / key
    run_dir.mkdir(parents=True, exist_ok=True)
    plan_path = run_dir / "plan.json"
    result_path = run_dir / "result.json"
    trace_path = run_dir / "trace.json"
    log_path = run_dir / "log.txt"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False) + "\n", encoding="utf-8")
    command = [sys.executable, str(PROBLEM1_EVALUATOR), str(graph_path), str(plan_path),
               "--config", str(config_path), "-o", str(result_path),
               "--trace-output", str(trace_path), "--log-output", str(log_path)]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, encoding="utf-8", capture_output=True,
                               timeout=timeout_s, check=False)
    if completed.returncode:
        raise OfficialEvaluationError(command, completed.returncode,
                                      completed.stdout, completed.stderr)
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    movement = raw["data_movement_bytes"]
    return OfficialResult(int(raw["makespan"]), int(movement["added_copy_bytes"]),
                          int(movement["partition_added_copy_bytes"]),
                          int(movement["spill_added_copy_bytes"]),
                          time.perf_counter() - started, raw)
```

- [ ] **Step 4: 实现 JSONL 缓存并测试命中不重复调用评估器**

缓存键包含输入图 SHA-256、配置 SHA-256、问题编号和方案哈希。追加记录前使用进程内锁；读取时最后一条相同键记录生效。测试通过替换 evaluator callable 为计数函数验证两次调用只执行一次。

- [ ] **Step 5: 运行 Task 7 测试**

Run: `python -m pytest 'A题/tests/test_evaluator.py' -q`

Expected: PASS。

- [ ] **Step 6: 提交 Task 7**

```powershell
git add -- 'A题/problem1/evaluator.py' 'A题/tests/test_evaluator.py'
git commit -m "feat: add cached official problem 1 evaluation"
```

---

### Task 8: 候选生成管线和单图 CLI

**Files:**
- Modify: `A题/problem1/__init__.py`
- Create: `A题/solver_problem1.py`
- Create: `A题/tests/test_cli.py`

**Interfaces:**
- Produces: `generate_candidates(model, num_cores, algorithms, budget) -> Iterator[CandidatePlan]`
- Produces CLI: `python A题/solver_problem1.py GRAPH -n CORES -o PLAN [--algorithms c1,c2,c3] [--official-budget N]`

- [ ] **Step 1: 写 CLI 失败测试**

```python
def test_solver_cli_writes_officially_valid_plan(tmp_path, fork_join_file):
    output = tmp_path / "plan.json"
    completed = subprocess.run([
        sys.executable, str(SOLVER), str(fork_join_file), "-n", "2",
        "--algorithms", "c1,c2,c3", "--official-budget", "0", "-o", str(output),
    ], text=True, encoding="utf-8", capture_output=True)
    assert completed.returncode == 0, completed.stderr
    graph = json.loads(fork_join_file.read_text(encoding="utf-8"))
    plan = json.loads(output.read_text(encoding="utf-8"))
    derive_multicore_plan(graph, plan)
    assert set(plan) == {"node_to_subgraph", "core_schedules"}
```

- [ ] **Step 2: 运行测试确认 CLI 不存在**

Run: `python -m pytest 'A题/tests/test_cli.py::test_solver_cli_writes_officially_valid_plan' -q`

Expected: FAIL，进程返回非零且提示文件不存在。

- [ ] **Step 3: 实现候选注册表和统一生成流程**

```python
@dataclass(frozen=True)
class CandidatePlan:
    partition: Partition
    core_schedules: tuple[tuple[int, ...], ...]
    proxy: Estimate
    algorithm: str
    parameters: Mapping[str, object]
    official: OfficialResult | None = None


ALGORITHMS = {
    "c0": generate_c0_candidates,
    "c1": generate_c1_candidates,
    "c2": generate_c2_candidates,
    "c3": generate_c3_candidates,
}


def candidate_key(candidate):
    if candidate.official is not None:
        return (candidate.official.makespan,
                candidate.official.added_copy_bytes,
                candidate.proxy.total)
    return (candidate.proxy.total,
            sum(candidate.partition.boundary_bytes.values()),
            candidate.algorithm)
```

每个分区至少与 S1 组合；C3 同时与 S2 组合；只有显式请求时使用 S3。代理分数排序后，`official_budget=0` 返回代理最优方案，否则对前 N 个调用官方评估并按 `candidate_key` 选择。

- [ ] **Step 4: 实现 CLI 参数校验和纯方案输出**

CLI 必须拒绝未知算法、非 1 至 5 核、负评估预算和缺失 config。输出父目录按需创建，最终文件以 UTF-8 和换行结尾写入。标准输出打印候选数量、胜出算法、子图数、代理分数和官方指标是否可用。

- [ ] **Step 5: 运行 CLI 测试和真实 `case_001` 冒烟测试**

Run: `python -m pytest 'A题/tests/test_cli.py' -q`

Run: `python 'A题/solver_problem1.py' 'A题/通用神经网络处理器下的多核调度问题  附件/data/case_001.json' -n 2 --algorithms c1,c2,c3 --official-budget 3 --config 'A题/通用神经网络处理器下的多核调度问题  附件/data/config.txt' -o 'tmp/problem1_smoke/case_001_plan.json'`

Expected: 测试 PASS；冒烟命令输出合法方案和 3 次以内的官方评估摘要。

- [ ] **Step 6: 提交 Task 8**

```powershell
git add -- 'A题/problem1/__init__.py' 'A题/solver_problem1.py' 'A题/tests/test_cli.py'
git commit -m "feat: add problem 1 solver cli"
```

---

### Task 9: 图画像、代表集选择和连续淘汰

**Files:**
- Create: `A题/problem1/experiments.py`
- Create: `A题/experiment_problem1.py`
- Create: `A题/tests/test_experiments.py`

**Interfaces:**
- Produces: `profile_cases(paths) -> list[CaseProfile]`
- Produces: `select_representatives(profiles, count) -> list[CaseProfile]`
- Produces: `successive_halving(candidates, stages, evaluator) -> ExperimentSummary`
- Produces CLI: `python A题/experiment_problem1.py profile|screen|verify ...`

- [ ] **Step 1: 写失败测试验证代表集覆盖极值且选择确定**

```python
def test_representative_selection_keeps_extremes_and_is_deterministic():
    profiles = synthetic_case_profiles()
    selected = select_representatives(profiles, count=4)
    assert selected == select_representatives(list(reversed(profiles)), count=4)
    names = {item.name for item in selected}
    assert "largest" in names
    assert "widest" in names
    assert "highest_fanout" in names
```

- [ ] **Step 2: 运行测试确认实验模块不存在**

Run: `python -m pytest 'A题/tests/test_experiments.py' -q`

Expected: FAIL，提示无法导入实验接口。

- [ ] **Step 3: 实现确定性最远点代表集选择**

```python
@dataclass(frozen=True)
class CaseProfile:
    name: str
    path: Path
    features: Mapping[str, float]
    selection_reason: str | None = None


@dataclass(frozen=True)
class ExperimentSummary:
    records: tuple[Mapping[str, object], ...]
    promoted_algorithms: tuple[str, ...]
    pareto_algorithms: tuple[str, ...]
```

先对 `log1p(num_ops)`、`log1p(num_edges)`、深度、宽度、Pipe 比例、`log1p(max_fanout)`、`log1p(tensor_bytes)`、L1/UB 压力归一化。先加入每个维度极值中贡献最多的不同算例，再使用标准化欧氏距离最远点采样补齐；距离相同时按文件名选择。

- [ ] **Step 4: 实现 Pareto 筛选和 successive halving**

每阶段记录合法率、Makespan regret、几何平均、最差退化、额外搬运和求解时间。先删除被另一候选在全部主指标支配且至少一个严格更差的候选；剩余候选按中位 regret 排名，保留 `ceil(n / eta)`，同时强制保留 C0。

- [ ] **Step 5: 实现实验 CLI 和三种模式**

`profile` 只读取正式图并输出画像 JSONL 与代表集 JSON；`screen` 按阶段配置运行代表性算例；`verify` 只接受写有 `frozen: true` 的候选配置并运行指定算例与 1 至 5 核。所有模式输出 JSONL 原始记录、CSV 汇总和 Markdown 对比表。

- [ ] **Step 6: 运行实验测试和正式图只读画像**

Run: `python -m pytest 'A题/tests/test_experiments.py' -q`

Run: `python 'A题/experiment_problem1.py' profile --data-dir 'A题/通用神经网络处理器下的多核调度问题  附件/data' --count 10 --output-dir 'outputs/problem1/profile'`

Expected: 测试 PASS；输出 100 条画像和 10 个带选择原因的代表算例，不运行官方评估器。

- [ ] **Step 7: 提交 Task 9**

```powershell
git add -- 'A题/problem1/experiments.py' 'A题/experiment_problem1.py' 'A题/tests/test_experiments.py'
git commit -m "feat: add staged problem 1 experiments"
```

---

### Task 10: C5 预算受限局部搜索

**Files:**
- Create: `A题/problem1/local_search.py`
- Create: `A题/tests/test_local_search.py`
- Modify: `A题/problem1/__init__.py`

**Interfaces:**
- Produces: `SearchBudget(max_official_evaluations, max_seconds, proxy_frontier)`
- Produces: `improve_candidate(model, initial, config, evaluator, budget) -> SearchResult`

- [ ] **Step 1: 写失败测试验证所有邻居合法且只接受词典序改进**

```python
def test_local_neighbors_are_valid():
    model, candidate = scheduled_fork_join_candidate()
    for neighbor in generate_neighbors(model, candidate):
        validate_plan(model, neighbor.partition, neighbor.core_schedules)


def test_search_respects_budget_and_keeps_best_seen():
    model, candidate = scheduled_fork_join_candidate()
    fake = MonotoneFakeEvaluator()
    result = improve_candidate(model, candidate, {}, fake,
                               SearchBudget(max_official_evaluations=3,
                                            max_seconds=10, proxy_frontier=5))
    assert fake.calls <= 3
    assert result.official_key <= result.initial_official_key
```

- [ ] **Step 2: 运行测试确认局部搜索模块不存在**

Run: `python -m pytest 'A题/tests/test_local_search.py' -q`

Expected: FAIL，提示无法导入搜索接口。

- [ ] **Step 3: 实现五类合法邻域**

```python
@dataclass(frozen=True)
class SearchBudget:
    max_official_evaluations: int
    max_seconds: float
    proxy_frontier: int


@dataclass(frozen=True)
class SearchResult:
    initial: CandidatePlan
    best: CandidatePlan
    official_evaluations: int
    elapsed_seconds: float
    stop_reason: str

    @property
    def initial_official_key(self):
        return (self.initial.official.makespan,
                self.initial.official.added_copy_bytes)

    @property
    def official_key(self):
        return (self.best.official.makespan,
                self.best.official.added_copy_bytes)
```

合并只作用于商图直接依赖块；拆分只沿块内拓扑序的候选弱边；移动与交换后用 `validate_plan` 检查；高扇出共置通过移动完整消费块实现；关键路径窗口重排只改变调度不改变切图。每个邻居先按规范哈希去重。

- [ ] **Step 4: 实现代理前沿和官方预算**

每轮按代理总分、边界字节和哈希排序，只保留 `proxy_frontier` 个邻居做官方评估。接受键为 `(makespan, added_copy_bytes)`，严格改进才更新当前解；全局最优始终保留。达到次数或时间预算立即返回，并记录停止原因。

- [ ] **Step 5: 将 C5 注册到候选管线并运行测试**

Run: `python -m pytest 'A题/tests/test_local_search.py' 'A题/tests/test_cli.py' -q`

Expected: PASS。

- [ ] **Step 6: 提交 Task 10**

```powershell
git add -- 'A题/problem1/local_search.py' 'A题/problem1/__init__.py' 'A题/tests/test_local_search.py'
git commit -m "feat: add budgeted problem 1 local search"
```

---

### Task 11: 第一轮理论检查与代表性实验

**Files:**
- Create: `A题/问题一候选算法评估.md`
- Create under ignored/generated output: `outputs/problem1/screening/*`

**Interfaces:**
- Consumes: C0、C1、C2、C3、C5 和代表集配置
- Produces: 候选机制结论、淘汰理由、保留候选和下一轮实验预算

- [ ] **Step 1: 运行全部自动化测试**

Run: `python -m pytest 'A题/tests' -q`

Expected: PASS，且无 warning 或临时文件泄漏到仓库根目录。

- [ ] **Step 2: 对合成图执行性质矩阵**

Run: `python 'A题/experiment_problem1.py' screen --synthetic --algorithms c0,c1,c2,c3 --cores 2,4 --output-dir 'outputs/problem1/screening/synthetic'`

Expected: 所有方案合法；报告分别显示链、宽图、Fork-Join、高扇出、Pipe 混合和容量边界图的方向性结果。

- [ ] **Step 3: 在 3 个结构差异最大的正式算例上首轮筛选**

Run: `python 'A题/experiment_problem1.py' screen --representatives 'outputs/problem1/profile/representatives.json' --limit 3 --algorithms c0,c1,c2,c3 --cores 2,4,5 --official-budget-per-algorithm 4 --output-dir 'outputs/problem1/screening/stage1'`

Expected: 每个算法和核数都有官方结果或明确失败记录，并生成 Pareto 表与 regret 表。

- [ ] **Step 4: 对晋级候选运行 8 至 12 个代表算例**

从 Stage 1 的机器可读晋级清单读取算法，不手工挑选。每个候选使用相同官方评估预算。

Run: `python 'A题/experiment_problem1.py' screen --representatives 'outputs/problem1/profile/representatives.json' --promoted-from 'outputs/problem1/screening/stage1/summary.json' --cores 2,4,5 --official-budget-per-algorithm 8 --output-dir 'outputs/problem1/screening/stage2'`

Expected: 输出候选间成对胜负、按图类型分组成绩、代理秩相关性和求解时间。

- [ ] **Step 5: 对前两种初解加入 C5 并进行预算消融**

分别使用 5、15、30 次官方评估预算，记录边际收益曲线。只在 C5 对代表集几何平均 Makespan 有稳定改进且最差退化不超过 2% 时将其保留为最终候选。

- [ ] **Step 6: 编写评估结论文档**

`A题/问题一候选算法评估.md` 必须逐项回答：算法假设是否成立、哪些图型受益、失败机制、代理误差、运行成本、是否晋级，以及支持结论的结果文件路径。不得只粘贴排行榜。

- [ ] **Step 7: 提交可复现结论，不提交大型 Trace**

```powershell
git add -- 'A题/问题一候选算法评估.md'
git commit -m "docs: evaluate problem 1 algorithm candidates"
```

---

### Task 12: 冻结候选与完全验证入口

**Files:**
- Create: `A题/problem1_frozen_candidates.json`
- Modify: `A题/问题一候选算法评估.md`
- Modify: `A题/tests/test_cli.py`

**Interfaces:**
- Produces: 不超过 3 个 `frozen: true` 的候选配置
- Produces: 可运行全部正式算例和 1 至 5 核的验证命令

- [ ] **Step 1: 写失败测试要求 verify 拒绝未冻结配置**

```python
def test_verify_rejects_unfrozen_candidate_file(tmp_path):
    config = tmp_path / "candidate.json"
    config.write_text(json.dumps({"frozen": False, "candidates": []}), encoding="utf-8")
    completed = run_experiment_cli("verify", "--candidates", str(config))
    assert completed.returncode != 0
    assert "frozen" in completed.stderr.lower()
```

- [ ] **Step 2: 运行测试确认当前 verify 未执行冻结检查**

Run: `python -m pytest 'A题/tests/test_cli.py::test_verify_rejects_unfrozen_candidate_file' -q`

Expected: FAIL，断言显示命令未按预期拒绝。

- [ ] **Step 3: 实现冻结配置校验**

冻结文件包含设计版本、Git commit、算法名、全部参数、随机种子、代理权重、局部搜索预算和代表集结果摘要哈希。`verify` 发现缺失字段、`frozen` 非真或当前实现版本不匹配时立即失败。

- [ ] **Step 4: 根据 Stage 2 和 C5 消融写入 2 至 3 个候选**

每个冻结候选必须机制不同或预算不同，并在评估文档中写出保留依据。若只有一个候选满足晋级标准，则冻结一个而不凑数。

- [ ] **Step 5: 运行完全验证前的干跑检查**

Run: `python 'A题/experiment_problem1.py' verify --candidates 'A题/problem1_frozen_candidates.json' --data-dir 'A题/通用神经网络处理器下的多核调度问题  附件/data' --cores 1,2,3,4,5 --dry-run --output-dir 'outputs/problem1/final'`

Expected: 列出准确的算例数、候选数、核数、预计官方评估次数、缓存命中数和输出路径，不执行正式评估。

- [ ] **Step 6: 运行全部测试和一个冻结候选单算例验证**

Run: `python -m pytest 'A题/tests' -q`

Run: `python 'A题/experiment_problem1.py' verify --candidates 'A题/problem1_frozen_candidates.json' --cases case_001 --cores 1,2,3,4,5 --output-dir 'outputs/problem1/final_smoke'`

Expected: 全部测试 PASS；单算例五种核数均产生合法官方结果。

- [ ] **Step 7: 提交冻结配置和最终入口检查**

```powershell
git add -- 'A题/problem1_frozen_candidates.json' 'A题/问题一候选算法评估.md' 'A题/tests/test_cli.py'
git commit -m "feat: freeze problem 1 finalists"
```

---

## Final Verification

- [ ] 运行 `python -m pytest 'A题/tests' -q`，要求全部通过。
- [ ] 运行仓库原有 `python -m pytest -q`，要求不引入回归。
- [ ] 对 `case_001` 和另一个代表性正式算例分别运行 2、4、5 核，确认所有冻结候选均通过官方评估器。
- [ ] 检查提交方案 JSON 顶层只有 `node_to_subgraph` 和 `core_schedules`。
- [ ] 检查实验 JSONL 中记录图哈希、配置哈希、方案哈希、算法参数、代理分量和官方结果。
- [ ] 检查 `git status --short`，确认未误提交官方附件生成的结果、Trace、日志或用户原有修改。
