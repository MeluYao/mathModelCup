from __future__ import annotations

import os
from pathlib import Path

import pytest

from math_model_cup.problem_d_q2 import load_q2_data


@pytest.fixture
def problem_dir() -> Path:
    configured = os.environ.get("D_PROBLEM_DIR")
    if not configured:
        pytest.skip("D_PROBLEM_DIR is not configured")
    return Path(configured)


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
    assert sum(box.volume_m3 for box in data.boxes.values()) == pytest.approx(2.011)


def test_loader_assigns_stable_aircraft_and_battery_ids(problem_dir: Path) -> None:
    data = load_q2_data(problem_dir)

    assert tuple(unit.aircraft_id for unit in data.aircraft_units) == tuple(
        f"U{index:02d}" for index in range(1, 9)
    )
    assert tuple(battery.battery_id for battery in data.batteries) == (
        "BA01",
        "BA02",
        "BA03",
        "BA04",
        "BA05",
        "BA06",
        "BB01",
        "BB02",
        "BB03",
        "BB04",
        "BC01",
        "BC02",
        "BC03",
        "BC04",
    )


def test_hard_deadlines_follow_first_batch_and_medical_rules(problem_dir: Path) -> None:
    data = load_q2_data(problem_dir)

    assert data.boxes["S001-MED-01"].hard_deadline_s == 3600.0
    assert data.boxes["S001-MED-02"].hard_deadline_s == 3600.0
    assert data.boxes["S012-MED-01"].hard_deadline_s == 3600.0
    assert data.boxes["S001-WAT-01"].hard_deadline_s == 3600.0
    assert data.boxes["S001-WAT-02"].hard_deadline_s is None
    assert data.boxes["S001-FOD-01"].hard_deadline_s is None
    assert sum(box.hard_deadline_s is not None for box in data.boxes.values()) == 31
