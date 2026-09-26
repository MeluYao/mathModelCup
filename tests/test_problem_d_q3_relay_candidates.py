from __future__ import annotations

from math_model_cup.problem_d_q3 import CommunicationSegment, Position3D, RelayState
from math_model_cup.problem_d_q3_communication import CommunicationEngine
from math_model_cup.problem_d_q3_relay_candidates import (
    build_coverage_atlas,
    generate_relay_states,
    prune_dominated_states,
)


def test_relay_states_respect_height_and_backhaul(q3_data) -> None:
    node = q3_data.transport.nodes["S015"]
    position = Position3D(node.longitude, node.latitude, node.operation_altitude_m)
    dark = CommunicationSegment(
        "D001", "T001", "handoff", 1000.0, 1010.0,
        position, position, False, -1.0,
    )
    engine = CommunicationEngine(q3_data)

    states = generate_relay_states(
        q3_data,
        (dark,),
        engine=engine,
        coarse_stride_pixels=64,
        max_horizontal_points=40,
    )

    assert states
    assert all(0.0 < state.agl_m <= 300.0 for state in states)
    assert all(state.backhaul_available for state in states)
    assert len({state.state_id for state in states}) == len(states)


def test_coverage_atlas_and_dominance_use_complete_segment_coverage(q3_data) -> None:
    node = q3_data.transport.nodes["O01"]
    transport = Position3D(node.longitude, node.latitude, node.elevation_m + 30.0)
    segment = CommunicationSegment(
        "D001", "T001", "cruise", 10.0, 20.0,
        transport, transport, False, -1.0,
    )
    states = (
        RelayState("A", node.longitude, node.latitude, node.elevation_m, 100.0,
                   node.elevation_m + 100.0, True, 100.0, 0.1, 20.0),
        RelayState("B", node.longitude, node.latitude, node.elevation_m, 150.0,
                   node.elevation_m + 150.0, True, 120.0, 0.2, 15.0),
    )
    engine = CommunicationEngine(q3_data)

    atlas = build_coverage_atlas(engine, states, (segment,))
    kept = prune_dominated_states(states, atlas)

    assert atlas.segment_to_states["D001"] == ("A", "B")
    assert [state.state_id for state in kept] == ["A"]
