from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from math_model_cup.problem_d_q3 import Q3Objective
from math_model_cup.problem_d_q3_aligned import (
    AlignedScenarioResult,
    prepare_communication_model,
    solve_relay_with_resource_repair,
    select_best_valid_result,
)
from math_model_cup.problem_d_q3_relay_schedule import RelaySchedulingError
from math_model_cup.problem_d_q3_transport_adapter import load_transport_scenario


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
Q2_RESULTS = REPOSITORY_ROOT / "outputs" / "problem_d_q2_stage3"


def _result(
    scenario_id: str,
    objective: tuple[float, float, float, int, int],
    valid: bool,
) -> AlignedScenarioResult:
    return AlignedScenarioResult(
        scenario_id=scenario_id,
        q2_method=scenario_id,
        q2_objective=(objective[0], objective[1], objective[2], objective[3]),
        solution=SimpleNamespace(objective=Q3Objective(*objective)),
        validation=SimpleNamespace(is_valid=valid),
        status="FEASIBLE",
        reason="",
        runtime_s=0.0,
        provenance={},
        prepared=None,
    )


def test_select_best_valid_result_uses_q3_lexicographic_objective() -> None:
    timely = _result("hybrid", (0.50, 10_000.0, 80.0, 24, 8), True)
    faster = _result("alns", (0.51, 9_000.0, 70.0, 22, 7), True)

    assert select_best_valid_result((faster, timely)).scenario_id == "hybrid"


def test_select_best_valid_result_ignores_invalid_result() -> None:
    invalid = _result("hybrid", (0.40, 8_000.0, 60.0, 24, 6), False)
    valid = _result("alns", (0.51, 9_000.0, 70.0, 22, 7), True)

    assert select_best_valid_result((invalid, valid)).scenario_id == "alns"


def test_prepare_communication_models_are_scenario_specific(q3_data, q3_arcs) -> None:
    hybrid = load_transport_scenario(
        q3_data.transport, q3_arcs, Q2_RESULTS / "hybrid", "hybrid"
    )
    alns = load_transport_scenario(
        q3_data.transport, q3_arcs, Q2_RESULTS / "alns", "alns"
    )

    left = prepare_communication_model(q3_data, q3_arcs, hybrid.solution)
    right = prepare_communication_model(q3_data, q3_arcs, alns.solution)

    assert left is not right
    assert left.communication_segments is not right.communication_segments
    assert {segment.transport_trip_id for segment in left.communication_segments} == {
        trip.trip_id for trip in hybrid.solution.trips
    }
    assert {segment.transport_trip_id for segment in right.communication_segments} == {
        trip.trip_id for trip in alns.solution.trips
    }


def test_relay_resource_repair_uses_relaxed_batches_then_exact_resources(
    monkeypatch,
) -> None:
    import math_model_cup.problem_d_q3_aligned as aligned

    data = object()
    arcs = object()
    initial_transport = object()
    repaired_transport = object()
    initial_model = SimpleNamespace(
        communication_segments=("initial-segments",),
        relay_states=("initial-states",),
        atlas="initial-atlas",
    )
    repaired_model = SimpleNamespace(
        communication_segments=("repaired-segments",),
        relay_states=("repaired-states",),
        atlas="repaired-atlas",
    )
    relaxed_solution = SimpleNamespace(relay_sorties=("sortie",))
    exact_solution = SimpleNamespace(solver_status="FEASIBLE")
    calls = []

    def fake_solve(_data, transport, segments, states, atlas, **kwargs):
        calls.append((transport, kwargs["enforce_resource_no_overlap"]))
        if transport is initial_transport and kwargs["enforce_resource_no_overlap"]:
            raise RelaySchedulingError("relay resource conflict")
        if not kwargs["enforce_resource_no_overlap"]:
            return relaxed_solution
        return exact_solution

    monkeypatch.setattr(aligned, "solve_relay_subproblem", fake_solve)
    monkeypatch.setattr(
        aligned,
        "relay_batches_from_solution",
        lambda solution: ("batch",) if solution is relaxed_solution else (),
    )
    monkeypatch.setattr(
        aligned,
        "reschedule_transport_with_relay_batches",
        lambda *args, **kwargs: repaired_transport,
    )
    monkeypatch.setattr(
        aligned,
        "prepare_communication_model",
        lambda _data, _arcs, transport: (
            repaired_model if transport is repaired_transport else initial_model
        ),
    )

    solution, transport, prepared = solve_relay_with_resource_repair(
        data,
        arcs,
        initial_transport,
        initial_model,
        joint_time_limit_s=10.0,
        relay_time_limit_s=20.0,
    )

    assert solution is exact_solution
    assert transport is repaired_transport
    assert prepared is repaired_model
    assert calls == [
        (initial_transport, True),
        (initial_transport, False),
        (repaired_transport, True),
    ]
