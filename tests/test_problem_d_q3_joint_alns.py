from __future__ import annotations

from math_model_cup.problem_d_q2_validation import validate_q2_solution
from math_model_cup.problem_d_q2_schedule import schedule_trips_greedy
from math_model_cup.problem_d_q3 import (
    CommunicationAssignment,
    CommunicationSegment,
    Position3D,
    Q3Data,
    Q3Objective,
    Q3Solution,
    RelayEnergyUnit,
    RelayModel,
    RelaySortieExecution,
    RelayState,
    RelayUnit,
)
from math_model_cup.problem_d_q3_joint_alns import (
    dominates,
    relay_batches_from_solution,
    reschedule_transport_with_segment_state_grid,
    reschedule_transport_with_relay_batches,
    reschedule_transport_with_state_grid,
    schedule_hard_then_serial_soft,
    select_grouped_transport,
)
from math_model_cup.problem_d_q3_relay_candidates import CoverageAtlas

from test_problem_d_q2_schedule import schedule_case


def test_dominance_uses_all_five_objectives() -> None:
    left = Q3Objective(0.8, 100.0, 10.0, 3, 1)
    right = Q3Objective(0.9, 100.0, 11.0, 3, 2)

    assert dominates(left, right)
    assert not dominates(right, left)


def test_grouped_transport_is_reproducible_and_valid(schedule_case) -> None:
    data, arcs, _ = schedule_case

    first = select_grouped_transport(data, arcs, max_stops=2, time_limit_s=5.0)
    second = select_grouped_transport(data, arcs, max_stops=2, time_limit_s=5.0)

    assert first.objective == second.objective
    assert [trip.plan.signature for trip in first.trips] == [
        trip.plan.signature for trip in second.trips
    ]
    assert validate_q2_solution(data, arcs, first).is_valid
    assert len(first.trips) <= len(data.boxes)

    staggered = schedule_hard_then_serial_soft(
        data, [trip.plan for trip in first.trips], soft_gap_s=60.0
    )
    assert validate_q2_solution(data, arcs, staggered).is_valid


def test_relay_batch_template_reschedules_fixed_routes(schedule_case) -> None:
    transport_data, arcs, plans = schedule_case
    transport = schedule_trips_greedy(transport_data, plans, method="synthetic")
    center = transport_data.nodes["O01"]
    position = Position3D(center.longitude, center.latitude, center.elevation_m + 100.0)
    segments = tuple(
        CommunicationSegment(
            f"D{index}",
            trip.trip_id,
            "cruise",
            trip.start_time_s + 100.0,
            trip.start_time_s + 120.0,
            position,
            position,
            False,
            -1.0,
        )
        for index, trip in enumerate(transport.trips, start=1)
    )
    relay_model = RelayModel(
        "R", 20.0, 10.0, 1.0, 10.0, 0.2, 10.0, 10.0, 10.0,
        2.0, 2.0, 0.8, 0.5, 0.1, 300.0, 300.0,
    )
    q3_data = Q3Data(
        transport=transport_data,
        relay_model=relay_model,
        relay_units=(RelayUnit("R1", "R", "O01"),),
        energy_units=(RelayEnergyUnit("E1", "R", 300.0),),
        link_budget=None,
    )
    state = RelayState(
        "RS1", center.longitude, center.latitude, center.elevation_m,
        100.0, center.elevation_m + 100.0, True, 20.0, 0.1, 20.0,
    )
    sorties = tuple(
        RelaySortieExecution(
            f"RLY{index}", "RS1", "R1", "E1", 0.0, 10.0, 30.0,
            segment.end_time_s, segment.end_time_s + 10.0,
            segment.end_time_s + 100.0, center.longitude, center.latitude,
            center.elevation_m + 100.0, 0.2, 98.0,
        )
        for index, segment in enumerate(segments, start=1)
    )
    assignments = tuple(
        CommunicationAssignment(
            segment.segment_id, segment.transport_trip_id, "relay",
            sorties[index].relay_sortie_id, segment.start_time_s, segment.end_time_s,
        )
        for index, segment in enumerate(segments)
    )
    template_solution = Q3Solution(
        "synthetic", transport, sorties, assignments,
        Q3Objective(*transport.objective, len(sorties)), 0.0, "FEASIBLE",
    )

    batches = relay_batches_from_solution(template_solution)
    shifted = reschedule_transport_with_relay_batches(
        q3_data,
        transport,
        segments,
        (state,),
        batches,
        time_limit_s=5.0,
    )

    assert len(batches) == len(segments)
    assert validate_q2_solution(transport_data, arcs, shifted).is_valid
    assert shifted.diagnostics["relay_capacity"] == 1

    atlas = CoverageAtlas(
        state_ids=("RS1",),
        segment_ids=tuple(segment.segment_id for segment in segments),
        state_masks={"RS1": (1 << len(segments)) - 1},
        segment_to_states={segment.segment_id: ("RS1",) for segment in segments},
    )
    gridded = reschedule_transport_with_state_grid(
        q3_data,
        transport,
        segments,
        (state,),
        atlas,
        grid_step_s=10.0,
        soft_grid_step_s=10.0,
        occupancy_slot_s=5.0,
        time_limit_s=5.0,
    )
    assert validate_q2_solution(transport_data, arcs, gridded).is_valid
    assert gridded.diagnostics["selected_relay_states"]


def test_segment_state_grid_allows_different_states_within_one_trip(
    schedule_case,
) -> None:
    transport_data, arcs, plans = schedule_case
    transport = schedule_trips_greedy(transport_data, plans, method="synthetic")
    trip = transport.trips[0]
    center = transport_data.nodes["O01"]
    position = Position3D(
        center.longitude, center.latitude, center.elevation_m + 100.0
    )
    segments = (
        CommunicationSegment(
            "D1", trip.trip_id, "cruise", 100.0, 120.0,
            position, position, False, -1.0,
        ),
        CommunicationSegment(
            "D2", trip.trip_id, "cruise", 140.0, 160.0,
            position, position, False, -1.0,
        ),
    )
    relay_model = RelayModel(
        "R", 20.0, 10.0, 1.0, 10.0, 0.2, 0.0, 0.0, 0.0,
        2.0, 2.0, 0.8, 0.5, 0.1, 300.0, 300.0,
    )
    data = Q3Data(
        transport=transport_data,
        relay_model=relay_model,
        relay_units=(RelayUnit("R1", "R", "O01"),),
        energy_units=(RelayEnergyUnit("E1", "R", 300.0),),
        link_budget=None,
    )
    states = (
        RelayState(
            "RS1", center.longitude, center.latitude, center.elevation_m,
            100.0, center.elevation_m + 100.0, True, 0.0, 0.1, 20.0,
        ),
        RelayState(
            "RS2", center.longitude, center.latitude, center.elevation_m,
            120.0, center.elevation_m + 120.0, True, 0.0, 0.1, 20.0,
        ),
    )
    atlas = CoverageAtlas(
        state_ids=("RS1", "RS2"),
        segment_ids=("D1", "D2"),
        state_masks={"RS1": 1, "RS2": 2},
        segment_to_states={"D1": ("RS1",), "D2": ("RS2",)},
    )

    shifted = reschedule_transport_with_segment_state_grid(
        data,
        transport,
        segments,
        states,
        atlas,
        hard_grid_step_s=10.0,
        soft_grid_step_s=10.0,
        occupancy_slot_s=5.0,
        soft_horizon_s=1000.0,
        time_limit_s=5.0,
    )

    assert validate_q2_solution(transport_data, arcs, shifted).is_valid
    assert shifted.diagnostics["selected_relay_states_by_segment"] == {
        "D1": "RS1",
        "D2": "RS2",
    }
