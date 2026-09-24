from __future__ import annotations

from pathlib import Path

import pytest

from math_model_cup.problem_d_q2 import (
    AircraftModel,
    AircraftUnit,
    BatteryUnit,
    Node,
    Q2Box,
    Q2Data,
    TripDraft,
    TripStop,
)
from math_model_cup.problem_d_q2_geometry import ArcGeometry
from math_model_cup.problem_d_q2_physics import evaluate_trip
from math_model_cup.problem_d_q2_schedule import (
    schedule_trips_cp_sat,
    schedule_trips_greedy,
)


@pytest.fixture
def schedule_case():
    model = AircraftModel(
        "C", "test", 70.0, 80.0, 0.25, 10.0, 100_000.0, 50_000.0,
        8.0, 0.20, 60.0, 10.0, 20.0, 5.0, 2.0, 2.0, 0.72,
    )
    data = Q2Data(
        problem_dir=Path("."),
        dem_path=Path("dem.tif"),
        nodes={
            "O01": Node("O01", "center", 0.0, 0.0, 100.0, True),
            "S001": Node("S001", "one", 0.01, 0.0, 120.0),
            "S002": Node("S002", "two", 0.02, 0.0, 120.0),
        },
        boxes={
            "B1": Q2Box("B1", "S001", "医疗物资", 3.0, 0.012, True, 3600.0, 3600.0, 24.0),
            "B2": Q2Box("B2", "S002", "饮用水", 14.0, 0.027, True, 3600.0, 3600.0, 12.0),
        },
        aircraft_models={"C": model},
        aircraft_units=(AircraftUnit("U01", "C", "O01"),),
        batteries=(BatteryUnit("BC01", "C", 3000.0),),
    )
    arcs = {}
    for service_id in ("S001", "S002"):
        arcs[("O01", service_id)] = ArcGeometry(
            "O01", service_id, 1000.0, 180.0, 230.0, 130.0, 80.0, 0, 0
        )
        arcs[(service_id, "O01")] = ArcGeometry(
            service_id, "O01", 1000.0, 180.0, 230.0, 80.0, 130.0, 0, 0
        )
    plans = tuple(
        evaluate_trip(
            data,
            arcs,
            TripDraft("C", (TripStop(data.boxes[box_id].service_id, (box_id,)),)),
        )
        for box_id in ("B1", "B2")
    )
    return data, arcs, plans


def test_greedy_scheduler_waits_for_full_battery(schedule_case) -> None:
    data, _, plans = schedule_case

    solution = schedule_trips_greedy(data, plans, method="test_greedy")

    first, second = sorted(solution.trips, key=lambda trip: trip.start_time_s)
    assert first.aircraft_id == second.aircraft_id == "U01"
    assert first.battery_id == second.battery_id == "BC01"
    assert second.start_time_s >= first.battery_ready_time_s
    assert len(solution.deliveries) == 2


def test_cp_sat_scheduler_returns_same_feasible_resource_sequence(schedule_case) -> None:
    data, _, plans = schedule_case

    solution = schedule_trips_cp_sat(data, plans, method="test_cp", time_limit_s=5)

    first, second = sorted(solution.trips, key=lambda trip: trip.start_time_s)
    assert second.start_time_s >= first.battery_ready_time_s - 1e-9
    assert max(record.delivery_time_s for record in solution.deliveries) <= 3600.0
