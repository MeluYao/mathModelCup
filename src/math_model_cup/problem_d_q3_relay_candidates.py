"""Relay-state generation and coverage atlas for Problem D, question 3.

This program and code were completed with assistance from OpenAI Codex.
The authors reviewed and validated the resulting implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, log10, radians, sqrt
from typing import Mapping, Sequence

import numpy as np

from .problem_d_q1 import GRAVITY_MPS2, JOULES_PER_KWH, _ellipsoidal_distance_m
from .problem_d_q3 import CommunicationSegment, Position3D, Q3Data, RelayState
from .problem_d_q3_communication import (
    CommunicationEngine,
    DemGrid,
    evaluate_link,
    raster_line_cells,
)


@dataclass(frozen=True)
class CoverageAtlas:
    state_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    state_masks: Mapping[str, int]
    segment_to_states: Mapping[str, tuple[str, ...]]


def _valid_elevation(dem: DemGrid, row: int, column: int) -> float | None:
    if row < 0 or row >= dem.rows or column < 0 or column >= dem.columns:
        return None
    value = float(dem.elevations_m[row, column])
    if not np.isfinite(value):
        return None
    if dem.nodata is not None and np.isclose(value, dem.nodata, rtol=0.0, atol=1e-12):
        return None
    return value


def _cell_for_position(dem: DemGrid, position: Position3D) -> tuple[int, int]:
    return (
        int((dem.y_origin - position.latitude) // dem.y_scale),
        int((position.longitude - dem.x_origin) // dem.x_scale),
    )


def _cell_center(dem: DemGrid, row: int, column: int) -> tuple[float, float]:
    return (
        dem.x_origin + (column + 0.5) * dem.x_scale,
        dem.y_origin - (row + 0.5) * dem.y_scale,
    )


def _relay_travel_metrics(
    data: Q3Data,
    dem: DemGrid,
    longitude: float,
    latitude: float,
    state_altitude_m: float,
) -> tuple[float, float]:
    center = data.transport.nodes["O01"]
    rows, columns = raster_line_cells(
        dem,
        center.longitude,
        center.latitude,
        longitude,
        latitude,
    )
    elevations = np.asarray(
        [
            value
            for row, column in zip(rows.tolist(), columns.tolist())
            for value in [_valid_elevation(dem, row, column)]
            if value is not None
        ],
        dtype=float,
    )
    if not elevations.size:
        return float("inf"), float("inf")
    cruise_altitude_m = max(
        center.elevation_m,
        state_altitude_m,
        float(np.max(elevations)) + 50.0,
    )
    horizontal_distance_m = _ellipsoidal_distance_m(
        center.longitude, center.latitude, longitude, latitude
    )
    model = data.relay_model
    center_climb_m = cruise_altitude_m - center.elevation_m
    state_climb_m = cruise_altitude_m - state_altitude_m
    time_s = (
        2.0 * horizontal_distance_m / model.cruise_speed_mps
        + center_climb_m / model.climb_speed_mps
        + state_climb_m / model.descent_speed_mps
        + state_climb_m / model.climb_speed_mps
        + center_climb_m / model.descent_speed_mps
    )
    horizontal_energy_kwh = (
        model.cruise_power_kw
        * (2.0 * horizontal_distance_m / model.cruise_speed_mps)
        / 3600.0
    )
    climb_energy_kwh = (
        model.takeoff_mass_kg
        * GRAVITY_MPS2
        * (center_climb_m + state_climb_m)
        / (JOULES_PER_KWH * model.climb_efficiency)
    )
    return time_s, horizontal_energy_kwh + climb_energy_kwh


def _horizontal_candidate_cells(
    dem: DemGrid,
    dark_segments: Sequence[CommunicationSegment],
    coarse_stride_pixels: int,
    max_horizontal_points: int,
) -> tuple[tuple[int, int], ...]:
    targeted: set[tuple[int, int]] = set()
    dark_cells: list[tuple[int, int]] = []
    for segment in dark_segments:
        for time_s in (
            segment.start_time_s,
            (segment.start_time_s + segment.end_time_s) / 2.0,
            segment.end_time_s,
        ):
            cell = _cell_for_position(dem, segment.position_at(time_s))
            dark_cells.append(cell)
            radius_pixels = max(1, int(ceil(240.0 / 30.0)))
            for row_delta in (-radius_pixels, 0, radius_pixels):
                for column_delta in (-radius_pixels, 0, radius_pixels):
                    targeted.add((cell[0] + row_delta, cell[1] + column_delta))
    if dark_cells:
        targeted.add(
            (
                int(round(sum(row for row, _ in dark_cells) / len(dark_cells))),
                int(round(sum(column for _, column in dark_cells) / len(dark_cells))),
            )
        )

    valid_targeted = {
        cell for cell in targeted if _valid_elevation(dem, cell[0], cell[1]) is not None
    }
    valid_dark = {
        cell for cell in dark_cells if _valid_elevation(dem, cell[0], cell[1]) is not None
    }
    coarse = [
        (row, column)
        for row in range(0, dem.rows, coarse_stride_pixels)
        for column in range(0, dem.columns, coarse_stride_pixels)
        if _valid_elevation(dem, row, column) is not None
    ]
    if dark_cells:
        coarse.sort(
            key=lambda cell: min(
                (cell[0] - dark[0]) ** 2 + (cell[1] - dark[1]) ** 2
                for dark in dark_cells
            )
        )
    def farthest_sample(
        pool: set[tuple[int, int]],
        count: int,
        initial: list[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]]:
        selected = list(initial or [])
        remaining = set(pool) - set(selected)
        if not remaining or len(selected) >= count:
            return selected[:count]
        if not selected:
            mean_row = sum(row for row, _ in remaining) / len(remaining)
            mean_column = sum(column for _, column in remaining) / len(remaining)
            first = min(
                remaining,
                key=lambda cell: (
                    (cell[0] - mean_row) ** 2 + (cell[1] - mean_column) ** 2,
                    cell,
                ),
            )
            selected.append(first)
            remaining.remove(first)
        while remaining and len(selected) < count:
            chosen = max(
                remaining,
                key=lambda cell: (
                    min(
                        (cell[0] - kept[0]) ** 2 + (cell[1] - kept[1]) ** 2
                        for kept in selected
                    ),
                    -cell[0],
                    -cell[1],
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        return selected

    dark_quota = min(max_horizontal_points, max(1, int(max_horizontal_points * 0.70)))
    cells = farthest_sample(valid_dark, dark_quota)
    cells = farthest_sample(valid_targeted, max_horizontal_points, cells)
    for cell in coarse:
        if cell not in cells:
            cells.append(cell)
        if len(cells) >= max_horizontal_points:
            break
    return tuple(cells[:max_horizontal_points])


def generate_relay_states(
    data: Q3Data,
    dark_segments: Sequence[CommunicationSegment],
    *,
    engine: CommunicationEngine | None = None,
    coarse_stride_pixels: int = 8,
    height_levels_m: Sequence[float] = (50, 100, 150, 200, 250, 300),
    max_horizontal_points: int = 500,
) -> tuple[RelayState, ...]:
    """Generate deterministic, backhaul- and reserve-feasible relay states."""
    if coarse_stride_pixels < 1 or max_horizontal_points < 1:
        raise ValueError("candidate-grid controls must be positive")
    engine = engine or CommunicationEngine(data)
    dem = engine.dem
    cells = _horizontal_candidate_cells(
        dem, dark_segments, coarse_stride_pixels, max_horizontal_points
    )
    candidates: list[RelayState] = []
    for row, column in cells:
        ground_m = _valid_elevation(dem, row, column)
        if ground_m is None:
            continue
        longitude, latitude = _cell_center(dem, row, column)
        for agl_m in height_levels_m:
            if agl_m <= 0.0 or agl_m > data.relay_model.max_hover_agl_m + 1e-9:
                continue
            altitude_m = ground_m + float(agl_m)
            position = Position3D(longitude, latitude, altitude_m)
            backhaul = evaluate_link(
                data.link_budget,
                dem,
                position,
                engine.gateway_position,
                "backhaul",
            )
            if not backhaul.available:
                continue
            round_trip_time_s, round_trip_energy_kwh = _relay_travel_metrics(
                data, dem, longitude, latitude, altitude_m
            )
            energy_limit = (1.0 - data.relay_model.reserve_ratio) * data.relay_model.usable_energy_kwh
            if round_trip_energy_kwh > energy_limit + 1e-9:
                continue
            candidates.append(
                RelayState(
                    state_id="",
                    longitude=longitude,
                    latitude=latitude,
                    ground_elevation_m=ground_m,
                    agl_m=float(agl_m),
                    altitude_m=altitude_m,
                    backhaul_available=True,
                    round_trip_time_s=round_trip_time_s,
                    round_trip_energy_kwh=round_trip_energy_kwh,
                    backhaul_margin_db=backhaul.margin_db,
                )
            )
    ordered = sorted(
        candidates,
        key=lambda state: (state.longitude, state.latitude, state.agl_m),
    )
    return tuple(
        RelayState(
            state_id=f"RS{index:05d}",
            longitude=state.longitude,
            latitude=state.latitude,
            ground_elevation_m=state.ground_elevation_m,
            agl_m=state.agl_m,
            altitude_m=state.altitude_m,
            backhaul_available=state.backhaul_available,
            round_trip_time_s=state.round_trip_time_s,
            round_trip_energy_kwh=state.round_trip_energy_kwh,
            backhaul_margin_db=state.backhaul_margin_db,
        )
        for index, state in enumerate(ordered, start=1)
    )


def build_coverage_atlas(
    engine: CommunicationEngine,
    states: Sequence[RelayState],
    segments: Sequence[CommunicationSegment],
) -> CoverageAtlas:
    """Require access and backhaul at five verification points per segment."""
    segment_ids = tuple(segment.segment_id for segment in segments)
    state_masks: dict[str, int] = {}
    segment_to_states: dict[str, list[str]] = {segment_id: [] for segment_id in segment_ids}
    access_limit_db = engine.access_limit_db
    maximum_clear_distance_km = 10.0 ** (
        (
            access_limit_db
            - 32.44
            - 20.0 * log10(engine.data.link_budget.frequency_mhz)
        )
        / 20.0
    )
    for state in states:
        mask = 0
        for index, segment in enumerate(segments):
            positions = [
                segment.position_at(
                    segment.start_time_s + fraction * segment.duration_s
                )
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
            ]
            mean_latitude = radians(
                sum(position.latitude for position in positions) / len(positions)
            )
            if any(
                sqrt(
                    ((position.latitude - state.latitude) * 111.32) ** 2
                    + (
                        (position.longitude - state.longitude)
                        * 111.32
                        * cos(mean_latitude)
                    )
                    ** 2
                    + ((position.altitude_m - state.altitude_m) / 1000.0) ** 2
                )
                > maximum_clear_distance_km * 1.01
                for position in positions
            ):
                continue
            checks = [
                evaluate_link(
                    engine.data.link_budget,
                    engine.dem,
                    position,
                    state.position,
                    "access",
                )
                for position in positions
            ]
            if all(check.available for check in checks) and state.backhaul_available:
                mask |= 1 << index
                segment_to_states[segment.segment_id].append(state.state_id)
        state_masks[state.state_id] = mask
    return CoverageAtlas(
        state_ids=tuple(state.state_id for state in states),
        segment_ids=segment_ids,
        state_masks=state_masks,
        segment_to_states={
            segment_id: tuple(sorted(state_ids))
            for segment_id, state_ids in segment_to_states.items()
        },
    )


def prune_dominated_states(
    states: Sequence[RelayState],
    atlas: CoverageAtlas,
) -> tuple[RelayState, ...]:
    """Remove states whose coverage and travel metrics are weakly dominated."""
    kept: list[RelayState] = []
    for candidate in states:
        candidate_mask = atlas.state_masks.get(candidate.state_id, 0)
        if candidate_mask == 0:
            continue
        dominated = False
        for other in states:
            if other.state_id == candidate.state_id:
                continue
            other_mask = atlas.state_masks.get(other.state_id, 0)
            covers = (other_mask | candidate_mask) == other_mask
            weakly_better = (
                other.round_trip_time_s <= candidate.round_trip_time_s + 1e-9
                and other.round_trip_energy_kwh <= candidate.round_trip_energy_kwh + 1e-12
                and other.backhaul_margin_db >= candidate.backhaul_margin_db - 1e-9
            )
            strictly_better = (
                other_mask != candidate_mask
                or other.round_trip_time_s < candidate.round_trip_time_s - 1e-9
                or other.round_trip_energy_kwh < candidate.round_trip_energy_kwh - 1e-12
                or other.backhaul_margin_db > candidate.backhaul_margin_db + 1e-9
            )
            if covers and weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            kept.append(candidate)
    return tuple(sorted(kept, key=lambda state: state.state_id))


def select_covering_state_subset(
    states: Sequence[RelayState],
    atlas: CoverageAtlas,
    *,
    redundancy: int = 3,
    max_states: int = 30,
) -> tuple[RelayState, ...]:
    """Greedily retain a small, redundant set cover for scheduling."""
    if redundancy < 1 or max_states < 1:
        raise ValueError("state-subset controls must be positive")
    state_by_id = {state.state_id: state for state in states}
    required = {
        segment_id: min(
            redundancy,
            len(
                [
                    state_id
                    for state_id in atlas.segment_to_states.get(segment_id, ())
                    if state_id in state_by_id
                ]
            ),
        )
        for segment_id in atlas.segment_ids
    }
    counts = {segment_id: 0 for segment_id in atlas.segment_ids}
    selected: list[RelayState] = []
    remaining = set(state_by_id)
    while remaining and len(selected) < max_states:
        candidate_id = max(
            remaining,
            key=lambda state_id: (
                sum(
                    1
                    for index, segment_id in enumerate(atlas.segment_ids)
                    if counts[segment_id] < required[segment_id]
                    and atlas.state_masks.get(state_id, 0) & (1 << index)
                ),
                -state_by_id[state_id].round_trip_energy_kwh,
                state_by_id[state_id].backhaul_margin_db,
                state_id,
            ),
        )
        gain = sum(
            1
            for index, segment_id in enumerate(atlas.segment_ids)
            if counts[segment_id] < required[segment_id]
            and atlas.state_masks.get(candidate_id, 0) & (1 << index)
        )
        if gain == 0:
            break
        selected.append(state_by_id[candidate_id])
        remaining.remove(candidate_id)
        mask = atlas.state_masks.get(candidate_id, 0)
        for index, segment_id in enumerate(atlas.segment_ids):
            if mask & (1 << index):
                counts[segment_id] += 1
        if all(counts[segment_id] >= required[segment_id] for segment_id in counts):
            break
    if any(counts[segment_id] == 0 for segment_id in counts):
        missing = [segment_id for segment_id, count in counts.items() if count == 0]
        raise ValueError(f"state subset loses coverage for {missing[:10]}")
    return tuple(sorted(selected, key=lambda state: state.state_id))
