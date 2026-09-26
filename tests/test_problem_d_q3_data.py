from __future__ import annotations

from pathlib import Path

from math_model_cup.problem_d_q3 import Q3Objective, load_q3_data


def test_load_q3_data_has_all_relay_and_link_resources(d_problem_dir: Path) -> None:
    data = load_q3_data(d_problem_dir)

    assert len(data.transport.boxes) == 80
    assert [unit.relay_id for unit in data.relay_units] == ["R01", "R02"]
    assert [unit.energy_id for unit in data.energy_units] == [
        "ER01",
        "ER02",
        "ER03",
        "ER04",
        "ER05",
        "ER06",
    ]
    assert data.relay_model.max_hover_agl_m == 300.0
    assert data.relay_model.reserve_ratio == 0.20
    assert data.link_budget.frequency_mhz == 2400.0
    assert data.link_budget.obstruction_loss_db == 10.0


def test_q3_objective_has_stable_five_component_order() -> None:
    objective = Q3Objective(
        weighted_delivery_time=0.75,
        joint_makespan_s=7200.0,
        total_energy_kwh=42.5,
        transport_trip_count=18,
        relay_trip_count=3,
    )

    assert objective.as_tuple() == (0.75, 7200.0, 42.5, 18, 3)
