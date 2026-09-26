"""Transport trajectories and atomic communication intervals for D-Q3.

This program and code were completed with assistance from OpenAI Codex.
The authors reviewed and validated the resulting implementation.
"""

from __future__ import annotations

from dataclasses import replace
from math import ceil
from typing import Mapping, Sequence

from .problem_d_q2 import Q2Data, TripExecution
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q3 import CommunicationSegment, Position3D, TrajectorySegment
from .problem_d_q3_communication import CommunicationEngine


class TrajectoryConsistencyError(RuntimeError):
    """Raised when a reconstructed trajectory disagrees with its Q2 plan."""


def build_trip_trajectory(
    data: Q2Data,
    arcs: Mapping[tuple[str, str], ArcGeometry],
    execution: TripExecution,
) -> tuple[TrajectorySegment, ...]:
    """Rebuild all in-flight and handoff phases using the Q2 timing rules."""
    plan = execution.plan
    model = data.aircraft_models[plan.model_id]
    offset = model.preparation_time_s + len(plan.box_ids) * model.load_time_per_box_s
    result: list[TrajectorySegment] = []

    def append(
        phase: str,
        duration_s: float,
        start: Position3D,
        end: Position3D,
    ) -> None:
        nonlocal offset
        if duration_s < -1e-9:
            raise TrajectoryConsistencyError(f"negative {phase} duration")
        if duration_s <= 1e-12:
            return
        result.append(
            TrajectorySegment(
                transport_trip_id=execution.trip_id,
                phase=phase,
                start_offset_s=offset,
                end_offset_s=offset + duration_s,
                start_position=start,
                end_position=end,
                trip_start_time_s=execution.start_time_s,
            )
        )
        offset += duration_s

    def append_leg(origin_id: str, destination_id: str) -> None:
        arc = arcs[(origin_id, destination_id)]
        origin = data.nodes[origin_id]
        destination = data.nodes[destination_id]
        origin_position = Position3D(
            origin.longitude, origin.latitude, origin.operation_altitude_m
        )
        origin_cruise = Position3D(
            origin.longitude, origin.latitude, arc.cruise_altitude_m
        )
        destination_cruise = Position3D(
            destination.longitude, destination.latitude, arc.cruise_altitude_m
        )
        destination_position = Position3D(
            destination.longitude,
            destination.latitude,
            destination.operation_altitude_m,
        )
        append("climb", arc.climb_m / model.climb_speed_mps, origin_position, origin_cruise)
        append("cruise", arc.distance_m / model.cruise_speed_mps, origin_cruise, destination_cruise)
        append(
            "descent",
            arc.descent_m / model.descent_speed_mps,
            destination_cruise,
            destination_position,
        )

    origin_id = "O01"
    for stop in plan.stops:
        append_leg(origin_id, stop.service_id)
        node = data.nodes[stop.service_id]
        position = Position3D(node.longitude, node.latitude, node.operation_altitude_m)
        append(
            "handoff",
            model.handoff_base_time_s + len(stop.box_ids) * model.handoff_time_per_box_s,
            position,
            position,
        )
        origin_id = stop.service_id
    append_leg(origin_id, "O01")

    if not result:
        raise TrajectoryConsistencyError("trip has no communication trajectory")
    if abs(offset - plan.duration_s) > 1e-6:
        raise TrajectoryConsistencyError(
            f"trajectory ends at {offset:.9f}s, plan ends at {plan.duration_s:.9f}s"
        )
    for left, right in zip(result, result[1:]):
        if abs(left.end_offset_s - right.start_offset_s) > 1e-9:
            raise TrajectoryConsistencyError("trajectory contains a time gap")
    return tuple(result)


def _classify_interval(
    engine: CommunicationEngine,
    segment: TrajectorySegment,
    start_offset_s: float,
    end_offset_s: float,
    tolerance_s: float,
) -> list[tuple[float, float, bool, float]]:
    midpoint = (start_offset_s + end_offset_s) / 2.0
    results = [
        engine.direct_available(segment.position_at(value))
        for value in (start_offset_s, midpoint, end_offset_s)
    ]
    states = [result.available for result in results]
    if all(state == states[0] for state in states) or end_offset_s - start_offset_s <= tolerance_s:
        return [
            (
                start_offset_s,
                end_offset_s,
                all(states),
                min(result.margin_db for result in results),
            )
        ]
    return _classify_interval(
        engine, segment, start_offset_s, midpoint, tolerance_s
    ) + _classify_interval(engine, segment, midpoint, end_offset_s, tolerance_s)


def build_direct_segments(
    engine: CommunicationEngine,
    trajectory: Sequence[TrajectorySegment],
    *,
    max_duration_s: float = 5.0,
    boundary_tolerance_s: float = 0.05,
) -> tuple[CommunicationSegment, ...]:
    """Classify a complete trajectory into contiguous direct/dark intervals."""
    if not trajectory:
        return ()
    if max_duration_s <= 0.0 or boundary_tolerance_s <= 0.0:
        raise ValueError("subdivision durations must be positive")
    pieces: list[CommunicationSegment] = []
    for source in trajectory:
        count = max(1, int(ceil(source.duration_s / max_duration_s)))
        for index in range(count):
            start = source.start_offset_s + source.duration_s * index / count
            end = source.start_offset_s + source.duration_s * (index + 1) / count
            for local_start, local_end, available, margin in _classify_interval(
                engine, source, start, end, boundary_tolerance_s
            ):
                pieces.append(
                    CommunicationSegment(
                        segment_id="",
                        transport_trip_id=source.transport_trip_id,
                        phase=source.phase,
                        start_time_s=source.trip_start_time_s + local_start,
                        end_time_s=source.trip_start_time_s + local_end,
                        start_position=source.position_at(local_start),
                        end_position=source.position_at(local_end),
                        direct_available=available,
                        minimum_direct_margin_db=margin,
                    )
                )

    merged: list[CommunicationSegment] = []
    for piece in pieces:
        if (
            merged
            and merged[-1].phase == piece.phase
            and merged[-1].direct_available == piece.direct_available
            and abs(merged[-1].end_time_s - piece.start_time_s) <= 1e-9
        ):
            previous = merged[-1]
            merged[-1] = replace(
                previous,
                end_time_s=piece.end_time_s,
                end_position=piece.end_position,
                minimum_direct_margin_db=min(
                    previous.minimum_direct_margin_db,
                    piece.minimum_direct_margin_db,
                ),
            )
        else:
            merged.append(piece)
    identified = tuple(
        replace(segment, segment_id=f"{segment.transport_trip_id}-C{index:04d}")
        for index, segment in enumerate(merged, start=1)
    )
    for left, right in zip(identified, identified[1:]):
        if abs(left.end_time_s - right.start_time_s) > 1e-7:
            raise TrajectoryConsistencyError("communication segments contain a gap")
    expected_start = trajectory[0].trip_start_time_s + trajectory[0].start_offset_s
    expected_end = trajectory[-1].trip_start_time_s + trajectory[-1].end_offset_s
    if (
        abs(identified[0].start_time_s - expected_start) > 1e-7
        or abs(identified[-1].end_time_s - expected_end) > 1e-7
    ):
        raise TrajectoryConsistencyError("communication segments do not cover trajectory")
    return identified
