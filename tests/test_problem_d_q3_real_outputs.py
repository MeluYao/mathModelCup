from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from math_model_cup.problem_d_q3_aligned import prepare_communication_model
from math_model_cup.problem_d_q3_validation import validate_q3_solution


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "outputs" / "problem_d_q3_aligned"


def test_hybrid_real_output_revalidates_from_transport_trajectory(
    q3_data,
    q3_arcs,
) -> None:
    scenario_dir = RESULT_ROOT / "hybrid"
    summary = json.loads((scenario_dir / "summary.json").read_text("utf-8"))
    with (scenario_dir / "q3_solution.pkl").open("rb") as stream:
        solution = pickle.load(stream)

    prepared = prepare_communication_model(q3_data, q3_arcs, solution.transport)
    validation = validate_q3_solution(
        q3_data,
        q3_arcs,
        prepared.communication_segments,
        prepared.relay_states,
        prepared.atlas,
        solution,
    )

    assert validation.is_valid
    assert summary["scenario_id"] == "hybrid"
    assert summary["q2_objective"] == [
        0.5158908995531455,
        9534.345058907296,
        69.52235221186811,
        24,
    ]
    assert summary["objective"] == list(solution.objective.as_tuple())
    assert summary["transport_trip_count"] == 24
    assert summary["relay_trip_count"] == 6
    assert summary["communication_segment_count"] == 204
    assert summary["direct_segment_count"] == 120
    assert summary["relay_segment_count"] == 84
    assert summary["maximum_relay_concurrency"] <= 2
    assert summary["maximum_energy_unit_concurrency"] <= 6
    assert len(pd.read_csv(scenario_dir / "transport_schedule.csv")) == 24
    assert len(pd.read_csv(scenario_dir / "deliveries.csv")) == 80
    assert len(pd.read_csv(scenario_dir / "relay_schedule.csv")) == 6
    assert len(pd.read_csv(scenario_dir / "communication_assignments.csv")) == 204


def test_selected_manifest_points_only_to_hybrid() -> None:
    manifest = json.loads(
        (RESULT_ROOT / "selected" / "selection.json").read_text("utf-8")
    )
    comparison = pd.read_csv(RESULT_ROOT / "comparison.csv")

    assert manifest["selected_scenario_id"] == "hybrid"
    assert comparison["scenario_id"].tolist() == ["hybrid"]
    assert comparison["is_valid"].tolist() == [True]
