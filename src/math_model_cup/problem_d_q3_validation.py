"""Independent validation for complete Problem D question 3 solutions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_physics import charge_time_s
from .problem_d_q2_validation import ValidationIssue, validate_q2_solution
from .problem_d_q3 import CommunicationSegment, Q3Data, Q3Solution, RelayState
from .problem_d_q3_relay_candidates import CoverageAtlas


@dataclass(frozen=True)
class Q3ValidationReport:
    issues: tuple[ValidationIssue, ...]
    checked_segment_count: int
    checked_relay_sortie_count: int

    @property
    def is_valid(self) -> bool:
        return not self.issues


def _resource_overlap_issues(
    solution: Q3Solution,
    *,
    resource: str,
    turnaround_time_s: float,
) -> list[ValidationIssue]:
    grouped = {}
    for sortie in solution.relay_sorties:
        if resource == "relay":
            resource_id = sortie.relay_id
            end_time = sortie.return_time_s + turnaround_time_s
        else:
            resource_id = sortie.energy_id
            end_time = sortie.energy_ready_time_s
        grouped.setdefault(resource_id, []).append(
            (sortie.preparation_start_time_s, end_time, sortie.relay_sortie_id)
        )
    issues = []
    for resource_id, intervals in grouped.items():
        intervals.sort()
        for left, right in zip(intervals, intervals[1:]):
            if right[0] < left[1] - 1e-7:
                issues.append(
                    ValidationIssue(
                        f"{resource}_overlap",
                        f"{resource_id} overlaps {left[2]} and {right[2]}",
                        resource_id,
                    )
                )
    return issues


def validate_q3_solution(
    data: Q3Data,
    arcs: Mapping[tuple[str, str], ArcGeometry],
    communication_segments: Sequence[CommunicationSegment],
    relay_states: Sequence[RelayState],
    atlas: CoverageAtlas,
    solution: Q3Solution,
) -> Q3ValidationReport:
    issues = list(validate_q2_solution(data.transport, arcs, solution.transport).issues)
    state_by_id = {state.state_id: state for state in relay_states}
    relay_ids = {unit.relay_id for unit in data.relay_units}
    energy_ids = {unit.energy_id for unit in data.energy_units}
    sortie_by_id = {
        sortie.relay_sortie_id: sortie for sortie in solution.relay_sorties
    }
    assignment_by_segment = {}
    for assignment in solution.communication_assignments:
        if assignment.segment_id in assignment_by_segment:
            issues.append(
                ValidationIssue(
                    "duplicate_communication_assignment",
                    "communication segment is assigned more than once",
                    assignment.segment_id,
                )
            )
        assignment_by_segment[assignment.segment_id] = assignment
    expected_ids = {segment.segment_id for segment in communication_segments}
    missing = sorted(expected_ids - set(assignment_by_segment))
    extra = sorted(set(assignment_by_segment) - expected_ids)
    if missing:
        issues.append(ValidationIssue("missing_communication_segment", str(missing[:5])))
    if extra:
        issues.append(ValidationIssue("unknown_communication_segment", str(extra[:5])))

    segment_by_id = {
        segment.segment_id: segment for segment in communication_segments
    }
    for segment_id in sorted(expected_ids & set(assignment_by_segment)):
        segment = segment_by_id[segment_id]
        assignment = assignment_by_segment[segment_id]
        if assignment.transport_trip_id != segment.transport_trip_id:
            issues.append(
                ValidationIssue(
                    "communication_trip_reference",
                    "transport trip reference mismatch",
                    segment_id,
                )
            )
        if (
            abs(assignment.start_time_s - segment.start_time_s) > 1e-7
            or abs(assignment.end_time_s - segment.end_time_s) > 1e-7
        ):
            issues.append(
                ValidationIssue(
                    "communication_time",
                    "communication assignment time mismatch",
                    segment_id,
                )
            )
        if segment.direct_available:
            if assignment.mode != "direct" or assignment.relay_sortie_id is not None:
                issues.append(
                    ValidationIssue(
                        "direct_assignment",
                        "direct segment is not assigned directly",
                        segment_id,
                    )
                )
            continue
        if assignment.mode != "relay" or assignment.relay_sortie_id is None:
            issues.append(
                ValidationIssue(
                    "relay_assignment",
                    "dark segment has no relay assignment",
                    segment_id,
                )
            )
            continue
        sortie = sortie_by_id.get(assignment.relay_sortie_id)
        if sortie is None:
            issues.append(
                ValidationIssue(
                    "unknown_relay_sortie",
                    "communication assignment references an unknown relay sortie",
                    segment_id,
                )
            )
            continue
        if sortie.state_id not in atlas.segment_to_states.get(segment_id, ()):
            issues.append(
                ValidationIssue(
                    "relay_coverage",
                    "relay state does not cover the complete segment",
                    segment_id,
                )
            )
        if (
            segment.start_time_s < sortie.link_ready_time_s - 1e-7
            or segment.end_time_s > sortie.service_end_time_s + 1e-7
        ):
            issues.append(
                ValidationIssue(
                    "relay_service_window",
                    "communication segment lies outside relay service window",
                    segment_id,
                )
            )

    for sortie in solution.relay_sorties:
        state = state_by_id.get(sortie.state_id)
        if state is None:
            issues.append(
                ValidationIssue(
                    "unknown_relay_state", "unknown relay state", sortie.relay_sortie_id
                )
            )
            continue
        if sortie.relay_id not in relay_ids:
            issues.append(
                ValidationIssue(
                    "unknown_relay_unit", "unknown relay unit", sortie.relay_sortie_id
                )
            )
        if sortie.energy_id not in energy_ids:
            issues.append(
                ValidationIssue(
                    "unknown_energy_unit", "unknown energy unit", sortie.relay_sortie_id
                )
            )
        outbound = state.round_trip_time_s / 2.0
        expected_takeoff = (
            sortie.preparation_start_time_s + data.relay_model.preparation_time_s
        )
        expected_link_ready = (
            expected_takeoff + outbound + data.relay_model.link_setup_time_s
        )
        expected_return = sortie.service_end_time_s + outbound
        if abs(sortie.takeoff_time_s - expected_takeoff) > 1e-7:
            issues.append(
                ValidationIssue("relay_takeoff_time", "takeoff time mismatch", sortie.relay_sortie_id)
            )
        if abs(sortie.link_ready_time_s - expected_link_ready) > 1e-7:
            issues.append(
                ValidationIssue("relay_link_ready", "link-ready time mismatch", sortie.relay_sortie_id)
            )
        if abs(sortie.return_time_s - expected_return) > 1e-7:
            issues.append(
                ValidationIssue("relay_return_time", "return time mismatch", sortie.relay_sortie_id)
            )
        service_energy = (
            data.relay_model.hover_power_kw
            + data.relay_model.communication_power_kw
        ) * (sortie.service_end_time_s - sortie.link_ready_time_s) / 3600.0
        expected_energy = state.round_trip_energy_kwh + service_energy
        energy_limit = (
            (1.0 - data.relay_model.reserve_ratio)
            * data.relay_model.usable_energy_kwh
        )
        if expected_energy > energy_limit + 1e-7:
            issues.append(
                ValidationIssue("relay_energy_limit", "relay energy limit exceeded", sortie.relay_sortie_id)
            )
        if abs(sortie.energy_kwh - expected_energy) > 1e-7:
            issues.append(
                ValidationIssue("relay_energy", "relay energy mismatch", sortie.relay_sortie_id)
            )
        expected_soc = 100.0 * (
            1.0 - expected_energy / data.relay_model.usable_energy_kwh
        )
        if abs(sortie.return_soc_percent - expected_soc) > 1e-7:
            issues.append(
                ValidationIssue("relay_soc", "relay return SOC mismatch", sortie.relay_sortie_id)
            )
        expected_energy_ready = expected_return + charge_time_s(
            expected_soc / 100.0,
            data.relay_model.full_charge_time_s,
        )
        if abs(sortie.energy_ready_time_s - expected_energy_ready) > 1e-7:
            issues.append(
                ValidationIssue(
                    "relay_energy_ready",
                    "relay energy-ready time mismatch",
                    sortie.relay_sortie_id,
                )
            )

    issues.extend(
        _resource_overlap_issues(
            solution,
            resource="relay",
            turnaround_time_s=data.relay_model.turnaround_time_s,
        )
    )
    issues.extend(
        _resource_overlap_issues(
            solution,
            resource="energy",
            turnaround_time_s=0.0,
        )
    )
    relay_energy = sum(sortie.energy_kwh for sortie in solution.relay_sorties)
    expected_objective = (
        solution.transport.objective[0],
        max(
            [solution.transport.objective[1]]
            + [sortie.return_time_s for sortie in solution.relay_sorties]
        ),
        solution.transport.objective[2] + relay_energy,
        solution.transport.objective[3],
        len(solution.relay_sorties),
    )
    if any(
        abs(float(left) - float(right)) > 1e-7
        for left, right in zip(solution.objective.as_tuple(), expected_objective)
    ):
        issues.append(ValidationIssue("q3_objective", "Q3 objective mismatch"))
    return Q3ValidationReport(
        issues=tuple(issues),
        checked_segment_count=len(communication_segments),
        checked_relay_sortie_count=len(solution.relay_sorties),
    )
