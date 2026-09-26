from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from math_model_cup.problem_d_q2_validation import validate_q2_solution
from math_model_cup.problem_d_q3_transport_adapter import (
    Q2ResultIntegrityError,
    load_transport_scenario,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
Q2_RESULTS = REPOSITORY_ROOT / "outputs" / "problem_d_q2_stage3"


def _copy_q2_result(tmp_path: Path, method: str) -> Path:
    target = tmp_path / method
    target.mkdir()
    for name in ("summary.json", "trips.csv", "deliveries.csv"):
        shutil.copy2(Q2_RESULTS / method / name, target / name)
    return target


def test_load_transport_scenario_reconstructs_valid_hybrid(q3_data, q3_arcs) -> None:
    scenario = load_transport_scenario(
        q3_data.transport,
        q3_arcs,
        Q2_RESULTS / "hybrid",
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
    assert validate_q2_solution(
        q3_data.transport, q3_arcs, scenario.solution
    ).is_valid


def test_load_transport_scenario_reconstructs_common_seed_alns(
    q3_data, q3_arcs
) -> None:
    scenario = load_transport_scenario(
        q3_data.transport,
        q3_arcs,
        Q2_RESULTS / "alns",
        "alns",
    )

    assert len(scenario.solution.trips) == 22
    assert scenario.provenance["incumbent_source"] == "provided_seed"
    assert scenario.provenance["seed_retained"] is True
    assert validate_q2_solution(
        q3_data.transport, q3_arcs, scenario.solution
    ).is_valid


def test_load_transport_scenario_rejects_tampered_objective(
    tmp_path, q3_data, q3_arcs
) -> None:
    result_dir = _copy_q2_result(tmp_path, "hybrid")
    summary_path = result_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["objective"]["energy_kwh"] = 0.0
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(Q2ResultIntegrityError, match="objective"):
        load_transport_scenario(
            q3_data.transport,
            q3_arcs,
            result_dir,
            "hybrid",
        )
