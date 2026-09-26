from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from math_model_cup.problem_d_q3 import Q3Objective
from math_model_cup.problem_d_q3_aligned import (
    AlignedScenarioResult,
    prepare_communication_model,
    select_best_valid_result,
)
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
