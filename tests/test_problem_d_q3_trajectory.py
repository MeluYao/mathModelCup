from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from math_model_cup.problem_d_q2 import (
    AircraftModel,
    Node,
    Q2Data,
    TripExecution,
    TripPlan,
    TripStop,
)
from math_model_cup.problem_d_q2_geometry import ArcGeometry
from math_model_cup.problem_d_q3_trajectory import (
    build_direct_segments,
    build_trip_trajectory,
)


def _case():
    model = AircraftModel(
        "C", "test", 70.0, 80.0, 0.25, 10.0, 100_000.0, 50_000.0,
        8.0, 0.2, 60.0, 10.0, 20.0, 5.0, 2.0, 2.0, 0.72,
    )
    data = Q2Data(
        problem_dir=Path("."),
        dem_path=Path("dem.tif"),
        nodes={
            "O01": Node("O01", "center", 0.0, 0.0, 100.0, True),
            "S001": Node("S001", "one", 0.01, 0.0, 120.0),
        },
        boxes={},
        aircraft_models={"C": model},
        aircraft_units=(),
        batteries=(),
    )
    outbound = ArcGeometry("O01", "S001", 1000.0, 180.0, 230.0, 130.0, 80.0, 0, 0)
    inbound = ArcGeometry("S001", "O01", 1000.0, 180.0, 230.0, 80.0, 130.0, 0, 0)
    duration = 60.0 + 10.0 + 65.0 + 100.0 + 40.0 + 25.0 + 40.0 + 100.0 + 65.0
    plan = TripPlan(
        model_id="C",
        stops=(TripStop("S001", ("B1",)),),
        box_ids=("B1",),
        total_mass_kg=3.0,
        total_volume_m3=0.01,
        duration_s=duration,
        energy_kwh=1.0,
        return_soc_percent=87.5,
        leg_payloads_kg=(3.0, 0.0),
        delivery_offsets_s={"B1": 300.0},
    )
    execution = TripExecution("T001", plan, "U01", "BC01", 1000.0, 1000.0 + duration, 3000.0)
    return data, {("O01", "S001"): outbound, ("S001", "O01"): inbound}, execution


def test_trajectory_covers_all_required_communication_time() -> None:
    data, arcs, execution = _case()
    segments = build_trip_trajectory(data, arcs, execution)

    assert segments[0].phase == "climb"
    assert segments[-1].phase == "descent"
    assert segments[0].start_offset_s == pytest.approx(70.0)
    assert segments[-1].end_offset_s == pytest.approx(execution.plan.duration_s)
    assert any(segment.phase == "handoff" for segment in segments)
    assert all(
        left.end_offset_s == pytest.approx(right.start_offset_s)
        for left, right in zip(segments, segments[1:])
    )


def test_direct_segments_cover_trajectory_without_gaps() -> None:
    data, arcs, execution = _case()
    trajectory = build_trip_trajectory(data, arcs, execution)

    class AlwaysDirect:
        @staticmethod
        def direct_available(position):
            return SimpleNamespace(available=True, margin_db=12.0)

    segments = build_direct_segments(AlwaysDirect(), trajectory, max_duration_s=5.0)

    assert segments[0].start_time_s == pytest.approx(
        execution.start_time_s + trajectory[0].start_offset_s
    )
    assert segments[-1].end_time_s == pytest.approx(execution.return_time_s)
    assert all(segment.direct_available for segment in segments)
    assert all(
        left.end_time_s == pytest.approx(right.start_time_s)
        for left, right in zip(segments, segments[1:])
    )
