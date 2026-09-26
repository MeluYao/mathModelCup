from __future__ import annotations

import json

import pandas as pd

from math_model_cup.problem_d_q2_schedule import schedule_trips_greedy
from math_model_cup.problem_d_q3 import (
    CommunicationAssignment,
    CommunicationSegment,
    Position3D,
    Q3Objective,
    Q3Solution,
)
from math_model_cup.problem_d_q3_aligned import (
    AlignedScenarioResult,
    PreparedCommunicationModel,
    select_best_valid_result,
)
from math_model_cup.problem_d_q3_relay_candidates import CoverageAtlas
from math_model_cup.problem_d_q3_reporting import (
    write_aligned_comparison,
    write_scenario_result,
)
from math_model_cup.problem_d_q3_validation import Q3ValidationReport

from test_problem_d_q2_schedule import schedule_case


def _valid_result(schedule_case, scenario_id: str, weighted: float):
    data, _, plans = schedule_case
    transport = schedule_trips_greedy(data, plans, method=scenario_id)
    trip = transport.trips[0]
    node = data.nodes["O01"]
    position = Position3D(node.longitude, node.latitude, node.elevation_m + 30.0)
    segment = CommunicationSegment(
        "D1",
        trip.trip_id,
        "cruise",
        trip.start_time_s + 10.0,
        trip.start_time_s + 20.0,
        position,
        position,
        True,
        100.0,
    )
    assignment = CommunicationAssignment(
        "D1",
        trip.trip_id,
        "direct",
        None,
        segment.start_time_s,
        segment.end_time_s,
    )
    solution = Q3Solution(
        method="synthetic",
        transport=transport,
        relay_sorties=(),
        communication_assignments=(assignment,),
        objective=Q3Objective(
            weighted,
            transport.objective[1],
            transport.objective[2],
            transport.objective[3],
            0,
        ),
        runtime_s=0.1,
        solver_status="FEASIBLE",
    )
    prepared = PreparedCommunicationModel(
        communication_segments=(segment,),
        dark_segments=(),
        relay_states=(),
        atlas=CoverageAtlas((), (), {}, {}),
    )
    return AlignedScenarioResult(
        scenario_id=scenario_id,
        q2_method=scenario_id,
        q2_objective=transport.objective,
        solution=solution,
        validation=Q3ValidationReport((), 1, 0),
        status="FEASIBLE",
        reason="",
        runtime_s=0.1,
        provenance={"summary_sha256": "abc"},
        prepared=prepared,
    )


def test_write_scenario_result_creates_complete_files(tmp_path, schedule_case) -> None:
    result = _valid_result(schedule_case, "hybrid", 0.5)

    scenario_dir = write_scenario_result(result, tmp_path / "hybrid")

    assert {path.name for path in scenario_dir.iterdir()} >= {
        "summary.json",
        "transport_schedule.csv",
        "deliveries.csv",
        "relay_schedule.csv",
        "communication_assignments.csv",
        "q3_solution.pkl",
    }
    summary = json.loads((scenario_dir / "summary.json").read_text("utf-8"))
    assert summary["scenario_id"] == "hybrid"
    assert summary["is_valid"] is True
    assert summary["communication_segment_count"] == 1


def test_comparison_records_both_inputs_and_selected_scenario(
    tmp_path, schedule_case
) -> None:
    hybrid = _valid_result(schedule_case, "hybrid", 0.5)
    alns = _valid_result(schedule_case, "alns", 0.6)
    selected = select_best_valid_result((alns, hybrid))

    write_aligned_comparison((hybrid, alns), selected, tmp_path)

    comparison = pd.read_csv(tmp_path / "comparison.csv")
    assert set(comparison["scenario_id"]) == {"hybrid", "alns"}
    manifest = json.loads(
        (tmp_path / "selected" / "selection.json").read_text("utf-8")
    )
    assert manifest["selected_scenario_id"] == "hybrid"
    assert manifest["selected_result_directory"] == "../hybrid"
