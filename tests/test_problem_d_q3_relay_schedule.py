from __future__ import annotations

from math_model_cup.problem_d_q2 import Q2Solution
from math_model_cup.problem_d_q3 import CommunicationSegment, Position3D, RelayState
from math_model_cup.problem_d_q3_relay_candidates import CoverageAtlas
from math_model_cup.problem_d_q3_relay_schedule import (
    minimum_simultaneous_sites,
    solve_relay_subproblem,
)


def test_one_relay_sortie_covers_two_overlapping_segments(q3_data) -> None:
    center = q3_data.transport.nodes["O01"]
    transport = Position3D(center.longitude, center.latitude, center.elevation_m + 30.0)
    segments = (
        CommunicationSegment("D1", "T1", "cruise", 1000.0, 1020.0,
                             transport, transport, False, -1.0),
        CommunicationSegment("D2", "T2", "cruise", 1005.0, 1015.0,
                             transport, transport, False, -1.0),
    )
    state = RelayState(
        "RS1", center.longitude, center.latitude, center.elevation_m,
        100.0, center.elevation_m + 100.0, True, 100.0, 0.1, 20.0,
    )
    atlas = CoverageAtlas(
        state_ids=("RS1",),
        segment_ids=("D1", "D2"),
        state_masks={"RS1": 0b11},
        segment_to_states={"D1": ("RS1",), "D2": ("RS1",)},
    )
    transport_solution = Q2Solution(
        method="synthetic", trips=(), deliveries=(),
        objective=(0.0, 0.0, 0.0, 0), runtime_s=0.0,
        solver_status="OPTIMAL",
    )

    result = solve_relay_subproblem(
        q3_data,
        transport_solution,
        segments,
        (state,),
        atlas,
        time_limit_s=5.0,
    )

    assert result.solver_status in {"OPTIMAL", "FEASIBLE"}
    assert len(result.relay_sorties) == 1
    relay_assignments = [
        assignment for assignment in result.communication_assignments
        if assignment.mode == "relay"
    ]
    assert len(relay_assignments) == 2
    assert len({assignment.relay_sortie_id for assignment in relay_assignments}) == 1
    core = minimum_simultaneous_sites(segments, atlas)
    assert core is not None
    assert core.minimum_sites == 1
