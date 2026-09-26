"""CP-SAT relay scheduling for a fixed D-Q3 transport solution.

This program and code were completed with assistance from OpenAI Codex.
The authors reviewed and validated the resulting implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from time import perf_counter
from typing import Sequence

from ortools.sat.python import cp_model

from .problem_d_q2 import Q2Solution
from .problem_d_q2_physics import charge_time_s
from .problem_d_q3 import (
    CommunicationAssignment,
    CommunicationSegment,
    Q3Data,
    Q3Objective,
    Q3Solution,
    RelaySortieExecution,
    RelayState,
)
from .problem_d_q3_relay_candidates import CoverageAtlas


@dataclass(frozen=True)
class RelayWindow:
    window_id: str
    state_id: str
    segment_ids: tuple[str, ...]
    preparation_start_time_s: float
    takeoff_time_s: float
    link_ready_time_s: float
    service_end_time_s: float
    return_time_s: float
    energy_ready_time_s: float
    energy_kwh: float
    return_soc_percent: float


@dataclass(frozen=True)
class RelayInfeasibilityCore:
    start_time_s: float
    end_time_s: float
    transport_trip_ids: tuple[str, ...]
    segment_ids: tuple[str, ...]
    minimum_sites: int
    reason: str


class RelaySchedulingError(RuntimeError):
    def __init__(self, message: str, core: RelayInfeasibilityCore | None = None) -> None:
        super().__init__(message)
        self.core = core


def _window_for_segments(
    data: Q3Data,
    state: RelayState,
    segments: Sequence[CommunicationSegment],
    window_id: str,
) -> RelayWindow | None:
    service_start = min(segment.start_time_s for segment in segments)
    service_end = max(segment.end_time_s for segment in segments)
    outbound_time = state.round_trip_time_s / 2.0
    preparation_start = (
        service_start
        - data.relay_model.link_setup_time_s
        - outbound_time
        - data.relay_model.preparation_time_s
    )
    if preparation_start < -1e-9:
        return None
    takeoff = preparation_start + data.relay_model.preparation_time_s
    link_ready = takeoff + outbound_time + data.relay_model.link_setup_time_s
    return_time = service_end + outbound_time
    service_energy = (
        (data.relay_model.hover_power_kw + data.relay_model.communication_power_kw)
        * (service_end - service_start)
        / 3600.0
    )
    energy_kwh = state.round_trip_energy_kwh + service_energy
    energy_limit = (1.0 - data.relay_model.reserve_ratio) * data.relay_model.usable_energy_kwh
    if energy_kwh > energy_limit + 1e-9:
        return None
    return_soc = 100.0 * (1.0 - energy_kwh / data.relay_model.usable_energy_kwh)
    energy_ready = return_time + charge_time_s(
        return_soc / 100.0,
        data.relay_model.full_charge_time_s,
    )
    return RelayWindow(
        window_id=window_id,
        state_id=state.state_id,
        segment_ids=tuple(segment.segment_id for segment in segments),
        preparation_start_time_s=max(0.0, preparation_start),
        takeoff_time_s=takeoff,
        link_ready_time_s=link_ready,
        service_end_time_s=service_end,
        return_time_s=return_time,
        energy_ready_time_s=energy_ready,
        energy_kwh=energy_kwh,
        return_soc_percent=return_soc,
    )


def build_relay_windows(
    data: Q3Data,
    dark_segments: Sequence[CommunicationSegment],
    relay_states: Sequence[RelayState],
    atlas: CoverageAtlas,
    *,
    maximum_group_size: int = 512,
    maximum_gap_s: float = 900.0,
) -> tuple[RelayWindow, ...]:
    """Build singleton and consecutive multi-segment service windows."""
    segment_by_id = {segment.segment_id: segment for segment in dark_segments}
    windows: list[RelayWindow] = []
    signatures: set[tuple[str, tuple[str, ...]]] = set()
    for state in relay_states:
        covered = sorted(
            (
                segment_by_id[segment_id]
                for segment_id in atlas.segment_ids
                if segment_id in segment_by_id
                and atlas.state_masks.get(state.state_id, 0)
                & (1 << atlas.segment_ids.index(segment_id))
            ),
            key=lambda segment: (segment.start_time_s, segment.end_time_s, segment.segment_id),
        )
        for gap_limit in sorted({60.0, 300.0, maximum_gap_s}):
            group: list[CommunicationSegment] = []

            def flush() -> None:
                nonlocal group
                if not group:
                    return
                signature = (state.state_id, tuple(item.segment_id for item in group))
                if signature not in signatures:
                    window = _window_for_segments(
                        data,
                        state,
                        group,
                        window_id=f"W{len(windows) + 1:06d}",
                    )
                    if window is not None:
                        signatures.add(signature)
                        windows.append(window)
                group = []

            for segment in covered:
                gap = segment.start_time_s - max(
                    (item.end_time_s for item in group),
                    default=segment.start_time_s,
                )
                trial = group + [segment]
                trial_window = _window_for_segments(data, state, trial, "trial")
                if (
                    group
                    and (
                        gap > gap_limit
                        or len(trial) > maximum_group_size
                        or trial_window is None
                    )
                ):
                    flush()
                    trial = [segment]
                group = trial
            flush()

        # Sliding alternatives remove the arbitrary dependence on the first
        # segment of a partition.  Keep every feasible consecutive prefix:
        # intermediate windows are necessary when a maximal window overlaps
        # another relay task but a shorter prefix does not.
        for start_index, first in enumerate(covered):
            group = []
            previous_end = first.start_time_s
            for segment in covered[start_index : start_index + maximum_group_size]:
                if group and segment.start_time_s - previous_end > maximum_gap_s:
                    break
                trial = group + [segment]
                trial_window = _window_for_segments(data, state, trial, "trial")
                if trial_window is None:
                    break
                group = trial
                previous_end = max(previous_end, segment.end_time_s)
                signature = (state.state_id, tuple(item.segment_id for item in group))
                if signature not in signatures:
                    window = _window_for_segments(
                        data,
                        state,
                        group,
                        window_id=f"W{len(windows) + 1:06d}",
                    )
                    if window is not None:
                        signatures.add(signature)
                        windows.append(window)
    return tuple(windows)


def _infeasibility_core(
    dark_segments: Sequence[CommunicationSegment],
    atlas: CoverageAtlas,
) -> RelayInfeasibilityCore:
    missing = [
        segment for segment in dark_segments
        if not atlas.segment_to_states.get(segment.segment_id)
    ]
    if not missing:
        peak = minimum_simultaneous_sites(dark_segments, atlas)
        if peak is not None and peak.minimum_sites > 2:
            return peak
    target = missing or list(dark_segments)
    if not target:
        return RelayInfeasibilityCore(0.0, 0.0, (), (), 0, "no dark segments")
    return RelayInfeasibilityCore(
        start_time_s=min(segment.start_time_s for segment in target),
        end_time_s=max(segment.end_time_s for segment in target),
        transport_trip_ids=tuple(sorted({segment.transport_trip_id for segment in target})),
        segment_ids=tuple(sorted(segment.segment_id for segment in target)),
        minimum_sites=(
            peak.minimum_sites
            if not missing and peak is not None
            else (0 if not missing else len(missing))
        ),
        reason="missing coverage" if missing else "relay resource conflict",
    )


def minimum_simultaneous_sites(
    dark_segments: Sequence[CommunicationSegment],
    atlas: CoverageAtlas,
) -> RelayInfeasibilityCore | None:
    """Return the event window with the largest 1/2/3+ set-cover lower bound."""
    if not dark_segments:
        return None
    state_sets = {
        segment.segment_id: set(atlas.segment_to_states.get(segment.segment_id, ()))
        for segment in dark_segments
    }
    event_times = sorted(
        {
            value
            for segment in dark_segments
            for value in (segment.start_time_s, segment.end_time_s)
        }
    )
    best: RelayInfeasibilityCore | None = None
    all_states = tuple(atlas.state_ids)
    masks_by_state = atlas.state_masks
    segment_index = {segment_id: index for index, segment_id in enumerate(atlas.segment_ids)}
    for left, right in zip(event_times, event_times[1:]):
        if right <= left + 1e-9:
            continue
        midpoint = (left + right) / 2.0
        active = [
            segment for segment in dark_segments
            if segment.start_time_s < right - 1e-9
            and segment.end_time_s > left + 1e-9
            and segment.start_time_s <= midpoint + 1e-9
            and segment.end_time_s >= midpoint - 1e-9
        ]
        if not active:
            continue
        common = set(all_states)
        for segment in active:
            common &= state_sets[segment.segment_id]
        if common:
            required = 1
        else:
            active_mask = 0
            for segment in active:
                active_mask |= 1 << segment_index[segment.segment_id]
            covering_masks = [masks_by_state[state_id] & active_mask for state_id in all_states]
            required = 3
            for index, left_mask in enumerate(covering_masks):
                if left_mask == 0:
                    continue
                if any(
                    (left_mask | right_mask) == active_mask
                    for right_mask in covering_masks[index + 1 :]
                ):
                    required = 2
                    break
        core = RelayInfeasibilityCore(
            start_time_s=left,
            end_time_s=right,
            transport_trip_ids=tuple(
                sorted({segment.transport_trip_id for segment in active})
            ),
            segment_ids=tuple(sorted(segment.segment_id for segment in active)),
            minimum_sites=required,
            reason="simultaneous relay-site lower bound",
        )
        if best is None or (core.minimum_sites, len(core.segment_ids)) > (
            best.minimum_sites,
            len(best.segment_ids),
        ):
            best = core
    return best


def solve_relay_subproblem(
    data: Q3Data,
    transport: Q2Solution,
    communication_segments: Sequence[CommunicationSegment],
    relay_states: Sequence[RelayState],
    atlas: CoverageAtlas,
    *,
    time_limit_s: float = 30.0,
    max_sorties: int = 16,
    enforce_resource_no_overlap: bool = True,
    stop_after_first_solution: bool = False,
) -> Q3Solution:
    """Schedule fixed relay windows with exact aircraft and energy no-overlap."""
    started = perf_counter()
    dark_segments = tuple(
        segment for segment in communication_segments if not segment.direct_available
    )
    direct_segments = tuple(
        segment for segment in communication_segments if segment.direct_available
    )
    if not dark_segments:
        assignments = tuple(
            CommunicationAssignment(
                segment.segment_id,
                segment.transport_trip_id,
                "direct",
                None,
                segment.start_time_s,
                segment.end_time_s,
            )
            for segment in direct_segments
        )
        objective = Q3Objective(
            transport.objective[0],
            transport.objective[1],
            transport.objective[2],
            transport.objective[3],
            0,
        )
        return Q3Solution(
            "fixed_transport_cp_sat_relay",
            transport,
            (),
            assignments,
            objective,
            perf_counter() - started,
            "OPTIMAL",
            {"relay_window_count": 0},
        )

    if any(not atlas.segment_to_states.get(segment.segment_id) for segment in dark_segments):
        core = _infeasibility_core(dark_segments, atlas)
        raise RelaySchedulingError("at least one communication segment has no relay state", core)
    windows = build_relay_windows(data, dark_segments, relay_states, atlas)
    if not windows:
        raise RelaySchedulingError(
            "no energy- and time-feasible relay windows",
            _infeasibility_core(dark_segments, atlas),
        )

    model = cp_model.CpModel()
    selected = {
        window.window_id: model.new_bool_var(f"select_{window.window_id}")
        for window in windows
    }
    model.add(sum(selected.values()) <= max_sorties)
    windows_by_segment = {
        segment.segment_id: [
            window for window in windows if segment.segment_id in window.segment_ids
        ]
        for segment in dark_segments
    }
    assignment = {}
    for segment in dark_segments:
        alternatives = windows_by_segment[segment.segment_id]
        if not alternatives:
            raise RelaySchedulingError(
                f"no service window for {segment.segment_id}",
                _infeasibility_core((segment,), atlas),
            )
        variables = []
        for window in alternatives:
            variable = model.new_bool_var(
                f"assign_{segment.segment_id}_{window.window_id}"
            )
            assignment[(segment.segment_id, window.window_id)] = variable
            model.add(variable <= selected[window.window_id])
            variables.append(variable)
        model.add_exactly_one(variables)
    for window in windows:
        uses = [
            assignment[(segment_id, window.window_id)]
            for segment_id in window.segment_ids
            if (segment_id, window.window_id) in assignment
        ]
        if uses:
            model.add(sum(uses) >= selected[window.window_id])

    uncovered = {segment.segment_id for segment in dark_segments}
    greedy_windows: list[RelayWindow] = []
    while uncovered:
        best = max(
            windows,
            key=lambda window: (
                len(uncovered.intersection(window.segment_ids)),
                -window.energy_kwh,
                -(window.return_time_s - window.preparation_start_time_s),
                window.window_id,
            ),
        )
        gain = uncovered.intersection(best.segment_ids)
        if not gain:
            break
        greedy_windows.append(best)
        uncovered -= gain
    greedy_window_ids = {window.window_id for window in greedy_windows}
    if len(greedy_windows) <= max_sorties:
        for window in windows:
            model.add_hint(
                selected[window.window_id],
                int(window.window_id in greedy_window_ids),
            )
        for segment in dark_segments:
            hinted_window = next(
                (
                    window
                    for window in greedy_windows
                    if segment.segment_id in window.segment_ids
                ),
                None,
            )
            if hinted_window is None:
                continue
            for window in windows_by_segment[segment.segment_id]:
                model.add_hint(
                    assignment[(segment.segment_id, window.window_id)],
                    int(window.window_id == hinted_window.window_id),
                )

    relay_intervals = {unit.relay_id: [] for unit in data.relay_units}
    energy_intervals = {unit.energy_id: [] for unit in data.energy_units}
    relay_choice = {}
    energy_choice = {}
    for window in windows:
        relay_bools = []
        relay_start = floor(window.preparation_start_time_s)
        relay_end = ceil(window.return_time_s + data.relay_model.turnaround_time_s)
        for unit in data.relay_units:
            present = model.new_bool_var(f"{window.window_id}_{unit.relay_id}")
            relay_choice[(window.window_id, unit.relay_id)] = present
            start_var = model.new_int_var(relay_start, relay_start, f"rs_{window.window_id}_{unit.relay_id}")
            end_var = model.new_int_var(relay_end, relay_end, f"re_{window.window_id}_{unit.relay_id}")
            relay_intervals[unit.relay_id].append(
                model.new_optional_interval_var(
                    start_var,
                    relay_end - relay_start,
                    end_var,
                    present,
                    f"ri_{window.window_id}_{unit.relay_id}",
                )
            )
            relay_bools.append(present)
        model.add(sum(relay_bools) == selected[window.window_id])

        energy_bools = []
        energy_start = floor(window.preparation_start_time_s)
        energy_end = ceil(window.energy_ready_time_s)
        for unit in data.energy_units:
            present = model.new_bool_var(f"{window.window_id}_{unit.energy_id}")
            energy_choice[(window.window_id, unit.energy_id)] = present
            start_var = model.new_int_var(energy_start, energy_start, f"es_{window.window_id}_{unit.energy_id}")
            end_var = model.new_int_var(energy_end, energy_end, f"ee_{window.window_id}_{unit.energy_id}")
            energy_intervals[unit.energy_id].append(
                model.new_optional_interval_var(
                    start_var,
                    energy_end - energy_start,
                    end_var,
                    present,
                    f"ei_{window.window_id}_{unit.energy_id}",
                )
            )
            energy_bools.append(present)
        model.add(sum(energy_bools) == selected[window.window_id])
    if enforce_resource_no_overlap:
        for intervals in relay_intervals.values():
            model.add_no_overlap(intervals)
        for intervals in energy_intervals.values():
            model.add_no_overlap(intervals)

    model.minimize(
        sum(selected.values()) * 1_000_000_000
        + sum(
            int(round(window.energy_kwh * 1_000_000)) * selected[window.window_id]
            for window in windows
        ) * 100
        + sum(
            int(ceil(window.return_time_s)) * selected[window.window_id]
            for window in windows
        )
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    solver.parameters.random_seed = 20260925
    solver.parameters.stop_after_first_solution = bool(stop_after_first_solution)
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RelaySchedulingError(
            f"relay CP-SAT model failed with status {solver.status_name(status)}",
            _infeasibility_core(dark_segments, atlas),
        )

    state_by_id = {state.state_id: state for state in relay_states}
    chosen_windows = [
        window for window in windows if solver.value(selected[window.window_id])
    ]
    chosen_windows.sort(key=lambda window: (window.preparation_start_time_s, window.window_id))
    sortie_id_by_window = {}
    sorties = []
    for index, window in enumerate(chosen_windows, start=1):
        sortie_id = f"RLY{index:03d}"
        sortie_id_by_window[window.window_id] = sortie_id
        relay_id = next(
            unit.relay_id for unit in data.relay_units
            if solver.value(relay_choice[(window.window_id, unit.relay_id)])
        )
        energy_id = next(
            unit.energy_id for unit in data.energy_units
            if solver.value(energy_choice[(window.window_id, unit.energy_id)])
        )
        state = state_by_id[window.state_id]
        sorties.append(
            RelaySortieExecution(
                sortie_id,
                state.state_id,
                relay_id,
                energy_id,
                window.preparation_start_time_s,
                window.takeoff_time_s,
                window.link_ready_time_s,
                window.service_end_time_s,
                window.return_time_s,
                window.energy_ready_time_s,
                state.longitude,
                state.latitude,
                state.altitude_m,
                window.energy_kwh,
                window.return_soc_percent,
            )
        )

    assignments = [
        CommunicationAssignment(
            segment.segment_id,
            segment.transport_trip_id,
            "direct",
            None,
            segment.start_time_s,
            segment.end_time_s,
        )
        for segment in direct_segments
    ]
    for segment in dark_segments:
        window = next(
            window for window in chosen_windows
            if (segment.segment_id, window.window_id) in assignment
            and solver.value(assignment[(segment.segment_id, window.window_id)])
        )
        assignments.append(
            CommunicationAssignment(
                segment.segment_id,
                segment.transport_trip_id,
                "relay",
                sortie_id_by_window[window.window_id],
                segment.start_time_s,
                segment.end_time_s,
            )
        )
    assignments.sort(key=lambda item: (item.start_time_s, item.segment_id))
    relay_energy = sum(sortie.energy_kwh for sortie in sorties)
    joint_makespan = max(
        [transport.objective[1]] + [sortie.return_time_s for sortie in sorties]
    )
    objective = Q3Objective(
        weighted_delivery_time=transport.objective[0],
        joint_makespan_s=joint_makespan,
        total_energy_kwh=transport.objective[2] + relay_energy,
        transport_trip_count=transport.objective[3],
        relay_trip_count=len(sorties),
    )
    return Q3Solution(
        method="fixed_transport_cp_sat_relay",
        transport=transport,
        relay_sorties=tuple(sorties),
        communication_assignments=tuple(assignments),
        objective=objective,
        runtime_s=perf_counter() - started,
        solver_status="OPTIMAL" if status == cp_model.OPTIMAL else "FEASIBLE",
        diagnostics={
            "relay_window_count": len(windows),
            "dark_segment_count": len(dark_segments),
        },
    )


def assemble_relay_campaign_solution(
    data: Q3Data,
    transport: Q2Solution,
    communication_segments: Sequence[CommunicationSegment],
    relay_states: Sequence[RelayState],
    campaigns: Sequence[RelayWindow],
    *,
    method: str = "joint_segment_state_campaign",
) -> Q3Solution:
    """Color fixed relay campaigns onto official relay and energy resources."""
    started = perf_counter()
    state_by_id = {state.state_id: state for state in relay_states}
    campaign_by_segment: dict[str, RelayWindow] = {}
    for campaign in campaigns:
        if campaign.state_id not in state_by_id:
            raise RelaySchedulingError(
                f"campaign {campaign.window_id} references unknown state {campaign.state_id}"
            )
        for segment_id in campaign.segment_ids:
            if segment_id in campaign_by_segment:
                raise RelaySchedulingError(
                    f"segment {segment_id} appears in multiple relay campaigns"
                )
            campaign_by_segment[segment_id] = campaign
    dark_segment_ids = {
        segment.segment_id
        for segment in communication_segments
        if not segment.direct_available
    }
    if set(campaign_by_segment) != dark_segment_ids:
        missing = sorted(dark_segment_ids - set(campaign_by_segment))
        extra = sorted(set(campaign_by_segment) - dark_segment_ids)
        raise RelaySchedulingError(
            f"campaign partition mismatch; missing={missing[:3]}, extra={extra[:3]}"
        )

    relay_available = {unit.relay_id: 0.0 for unit in data.relay_units}
    energy_available = {unit.energy_id: 0.0 for unit in data.energy_units}
    relay_id_by_window = {}
    energy_id_by_window = {}
    ordered = sorted(
        campaigns,
        key=lambda campaign: (campaign.preparation_start_time_s, campaign.window_id),
    )
    for campaign in ordered:
        relay_id = next(
            (
                relay_id
                for relay_id, available in sorted(relay_available.items())
                if available <= campaign.preparation_start_time_s + 1e-9
            ),
            None,
        )
        if relay_id is None:
            raise RelaySchedulingError(
                f"relay capacity exceeded at campaign {campaign.window_id}"
            )
        relay_id_by_window[campaign.window_id] = relay_id
        relay_available[relay_id] = (
            campaign.return_time_s + data.relay_model.turnaround_time_s
        )

        energy_id = next(
            (
                energy_id
                for energy_id, available in sorted(energy_available.items())
                if available <= campaign.preparation_start_time_s + 1e-9
            ),
            None,
        )
        if energy_id is None:
            raise RelaySchedulingError(
                f"energy-unit capacity exceeded at campaign {campaign.window_id}"
            )
        energy_id_by_window[campaign.window_id] = energy_id
        energy_available[energy_id] = campaign.energy_ready_time_s

    sortie_id_by_window = {
        campaign.window_id: f"RLY{index:03d}"
        for index, campaign in enumerate(ordered, start=1)
    }
    sorties = []
    for campaign in ordered:
        state = state_by_id[campaign.state_id]
        sorties.append(
            RelaySortieExecution(
                relay_sortie_id=sortie_id_by_window[campaign.window_id],
                state_id=campaign.state_id,
                relay_id=relay_id_by_window[campaign.window_id],
                energy_id=energy_id_by_window[campaign.window_id],
                preparation_start_time_s=campaign.preparation_start_time_s,
                takeoff_time_s=campaign.takeoff_time_s,
                link_ready_time_s=campaign.link_ready_time_s,
                service_end_time_s=campaign.service_end_time_s,
                return_time_s=campaign.return_time_s,
                energy_ready_time_s=campaign.energy_ready_time_s,
                longitude=state.longitude,
                latitude=state.latitude,
                altitude_m=state.altitude_m,
                energy_kwh=campaign.energy_kwh,
                return_soc_percent=campaign.return_soc_percent,
            )
        )
    assignments = []
    for segment in communication_segments:
        if segment.direct_available:
            assignments.append(
                CommunicationAssignment(
                    segment.segment_id,
                    segment.transport_trip_id,
                    "direct",
                    None,
                    segment.start_time_s,
                    segment.end_time_s,
                )
            )
            continue
        campaign = campaign_by_segment[segment.segment_id]
        assignments.append(
            CommunicationAssignment(
                segment.segment_id,
                segment.transport_trip_id,
                "relay",
                sortie_id_by_window[campaign.window_id],
                segment.start_time_s,
                segment.end_time_s,
            )
        )
    assignments.sort(key=lambda assignment: (assignment.start_time_s, assignment.segment_id))
    relay_energy = sum(sortie.energy_kwh for sortie in sorties)
    joint_makespan = max(
        [transport.objective[1]] + [sortie.return_time_s for sortie in sorties]
    )
    objective = Q3Objective(
        transport.objective[0],
        joint_makespan,
        transport.objective[2] + relay_energy,
        transport.objective[3],
        len(sorties),
    )
    return Q3Solution(
        method=method,
        transport=transport,
        relay_sorties=tuple(sorties),
        communication_assignments=tuple(assignments),
        objective=objective,
        runtime_s=perf_counter() - started,
        solver_status="FEASIBLE",
        diagnostics={
            "relay_campaign_count": len(campaigns),
            "relay_energy_kwh": relay_energy,
        },
    )
