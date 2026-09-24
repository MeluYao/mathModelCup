from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from math_model_cup.problem_d_q2 import (
    AircraftModel,
    AircraftUnit,
    BatteryUnit,
    InfeasibleQ2Error,
    Node,
    Q2Box,
    Q2Data,
    TripDraft,
    TripStop,
    load_q2_data,
)
from math_model_cup.problem_d_q2_geometry import ArcGeometry, build_arc_matrix
from math_model_cup.problem_d_q2_physics import charge_time_s, evaluate_trip


@pytest.fixture(scope="module")
def real_q2_data() -> Q2Data:
    configured = os.environ.get("D_PROBLEM_DIR")
    if not configured:
        pytest.skip("D_PROBLEM_DIR is not configured")
    return load_q2_data(Path(configured))


@pytest.fixture
def synthetic_q2() -> SimpleNamespace:
    model = AircraftModel(
        model_id="C",
        name="test",
        empty_mass_kg=70.0,
        max_payload_kg=80.0,
        volume_capacity_m3=0.25,
        cruise_speed_mps=10.0,
        empty_range_m=100_000.0,
        full_range_m=50_000.0,
        usable_energy_kwh=8.0,
        reserve_ratio=0.20,
        preparation_time_s=60.0,
        load_time_per_box_s=10.0,
        handoff_base_time_s=20.0,
        handoff_time_per_box_s=5.0,
        climb_speed_mps=2.0,
        descent_speed_mps=2.0,
        climb_efficiency=0.72,
    )
    nodes = {
        "O01": Node("O01", "center", 0.0, 0.0, 100.0, True),
        "S001": Node("S001", "one", 0.01, 0.0, 120.0),
        "S002": Node("S002", "two", 0.02, 0.0, 130.0),
    }
    boxes = {
        "B1": Q2Box("B1", "S001", "医疗物资", 3.0, 0.012, True, 3600.0, 3600.0, 24.0),
        "B2": Q2Box("B2", "S002", "饮用水", 14.0, 0.027, True, 3600.0, 3600.0, 12.0),
    }
    data = Q2Data(
        problem_dir=Path("."),
        dem_path=Path("dem.tif"),
        nodes=nodes,
        boxes=boxes,
        aircraft_models={"C": model},
        aircraft_units=(AircraftUnit("U01", "C", "O01"),),
        batteries=(BatteryUnit("BC01", "C", 3000.0),),
    )
    arcs = {}
    for origin, destination in (("O01", "S001"), ("S001", "S002"), ("S002", "O01")):
        arcs[(origin, destination)] = ArcGeometry(
            origin,
            destination,
            distance_m=1000.0,
            peak_ground_m=180.0,
            cruise_altitude_m=230.0,
            climb_m=100.0,
            descent_m=50.0,
            peak_row=0,
            peak_column=0,
        )
    return SimpleNamespace(data=data, arcs=arcs)


def test_arc_matrix_contains_all_directed_node_pairs(real_q2_data: Q2Data) -> None:
    arcs = build_arc_matrix(real_q2_data)

    assert len(arcs) == 16 * 15
    assert all(origin != destination for origin, destination in arcs)


def test_reverse_arcs_share_terrain_but_have_directional_climb(
    real_q2_data: Q2Data,
) -> None:
    arcs = build_arc_matrix(real_q2_data)
    outbound = arcs[("O01", "S015")]
    inbound = arcs[("S015", "O01")]

    assert outbound.distance_m == pytest.approx(inbound.distance_m)
    assert outbound.cruise_altitude_m == pytest.approx(inbound.cruise_altitude_m)
    assert outbound.peak_ground_m == pytest.approx(inbound.peak_ground_m)
    assert outbound.climb_m != inbound.climb_m
    assert outbound.climb_m == pytest.approx(inbound.descent_m)
    assert outbound.descent_m == pytest.approx(inbound.climb_m)


def test_two_stop_trip_reduces_payload_after_first_delivery(synthetic_q2) -> None:
    draft = TripDraft(
        model_id="C",
        stops=(
            TripStop("S001", ("B1",)),
            TripStop("S002", ("B2",)),
        ),
    )

    plan = evaluate_trip(synthetic_q2.data, synthetic_q2.arcs, draft)

    assert plan.leg_payloads_kg == pytest.approx((17.0, 14.0, 0.0))
    assert plan.delivery_offsets_s["B1"] < plan.delivery_offsets_s["B2"]
    assert plan.return_soc_percent >= 20.0


@pytest.mark.parametrize(
    ("soc", "expected"),
    [(0.0, 3000.0), (0.9, 1050.0), (1.0, 0.0)],
)
def test_charge_time_matches_statement(soc: float, expected: float) -> None:
    assert charge_time_s(soc, 3000.0) == pytest.approx(expected)


def test_trip_rejects_payload_above_model_capacity(synthetic_q2) -> None:
    overloaded = dict(synthetic_q2.data.boxes)
    overloaded["B2"] = Q2Box(
        "B2", "S002", "饮用水", 78.0, 0.027, True, 3600.0, 3600.0, 12.0
    )
    data = Q2Data(
        problem_dir=synthetic_q2.data.problem_dir,
        dem_path=synthetic_q2.data.dem_path,
        nodes=synthetic_q2.data.nodes,
        boxes=overloaded,
        aircraft_models=synthetic_q2.data.aircraft_models,
        aircraft_units=synthetic_q2.data.aircraft_units,
        batteries=synthetic_q2.data.batteries,
    )
    draft = TripDraft(
        model_id="C",
        stops=(TripStop("S001", ("B1",)), TripStop("S002", ("B2",))),
    )

    with pytest.raises(InfeasibleQ2Error, match="payload mass"):
        evaluate_trip(data, synthetic_q2.arcs, draft)
