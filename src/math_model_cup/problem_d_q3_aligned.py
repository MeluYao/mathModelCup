"""Aligned question-3 scenarios built from validated question-2 schedules."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Mapping, Sequence

from .problem_d_q2 import InfeasibleQ2Error, Q2Solution
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q3 import CommunicationSegment, Q3Data, Q3Solution, RelayState
from .problem_d_q3_communication import CommunicationEngine
from .problem_d_q3_joint_alns import (
    relay_batches_from_solution,
    reschedule_transport_with_relay_batches,
    reschedule_transport_with_segment_state_grid,
)
from .problem_d_q3_relay_candidates import (
    CoverageAtlas,
    build_coverage_atlas,
    generate_relay_states,
    prune_dominated_states,
    select_covering_state_subset,
)
from .problem_d_q3_relay_schedule import RelaySchedulingError, solve_relay_subproblem
from .problem_d_q3_trajectory import build_direct_segments, build_trip_trajectory
from .problem_d_q3_transport_adapter import TransportScenario
from .problem_d_q3_validation import Q3ValidationReport, validate_q3_solution


@dataclass(frozen=True)
class PreparedCommunicationModel:
    communication_segments: tuple[CommunicationSegment, ...]
    dark_segments: tuple[CommunicationSegment, ...]
    relay_states: tuple[RelayState, ...]
    atlas: CoverageAtlas


@dataclass(frozen=True)
class AlignedScenarioResult:
    scenario_id: str
    q2_method: str
    q2_objective: tuple[float, float, float, int]
    solution: Q3Solution | None
    validation: Q3ValidationReport | None
    status: str
    reason: str
    runtime_s: float
    provenance: Mapping[str, object]
    prepared: PreparedCommunicationModel | None

    @property
    def is_valid(self) -> bool:
        return (
            self.solution is not None
            and self.validation is not None
            and self.validation.is_valid
        )


def prepare_communication_model(
    data: Q3Data,
    arcs: Mapping[tuple[str, str], ArcGeometry],
    transport: Q2Solution,
) -> PreparedCommunicationModel:
    """Build scenario-specific trajectories, segments, relay states, and atlas."""
    engine = CommunicationEngine(data)
    communication_segments = tuple(
        segment
        for trip in transport.trips
        for segment in build_direct_segments(
            engine,
            build_trip_trajectory(data.transport, arcs, trip),
            max_duration_s=30.0,
        )
    )
    dark_segments = tuple(
        segment for segment in communication_segments if not segment.direct_available
    )
    if not dark_segments:
        return PreparedCommunicationModel(
            communication_segments,
            dark_segments,
            (),
            CoverageAtlas((), (), {}, {}),
        )
    generated = generate_relay_states(
        data,
        dark_segments,
        engine=engine,
        coarse_stride_pixels=32,
        height_levels_m=(100, 200, 300),
        max_horizontal_points=150,
    )
    full_atlas = build_coverage_atlas(engine, generated, dark_segments)
    pruned = prune_dominated_states(generated, full_atlas)
    relay_states = select_covering_state_subset(
        pruned,
        full_atlas,
        redundancy=3,
        max_states=30,
    )
    atlas = build_coverage_atlas(engine, relay_states, dark_segments)
    return PreparedCommunicationModel(
        communication_segments,
        dark_segments,
        relay_states,
        atlas,
    )


def solve_relay_with_resource_repair(
    data: Q3Data,
    arcs: Mapping[tuple[str, str], ArcGeometry],
    transport: Q2Solution,
    prepared: PreparedCommunicationModel,
    *,
    joint_time_limit_s: float,
    relay_time_limit_s: float,
) -> tuple[Q3Solution, Q2Solution, PreparedCommunicationModel]:
    """Solve exact relay resources, repairing transport timing when necessary."""
    common = {
        "time_limit_s": relay_time_limit_s,
        "max_sorties": 300,
        "stop_after_first_solution": True,
    }
    try:
        solution = solve_relay_subproblem(
            data,
            transport,
            prepared.communication_segments,
            prepared.relay_states,
            prepared.atlas,
            enforce_resource_no_overlap=True,
            **common,
        )
        return solution, transport, prepared
    except RelaySchedulingError:
        relaxed = solve_relay_subproblem(
            data,
            transport,
            prepared.communication_segments,
            prepared.relay_states,
            prepared.atlas,
            enforce_resource_no_overlap=False,
            **common,
        )
        batches = relay_batches_from_solution(relaxed)
        repaired_transport = reschedule_transport_with_relay_batches(
            data,
            transport,
            prepared.communication_segments,
            prepared.relay_states,
            batches,
            time_limit_s=joint_time_limit_s,
        )
        repaired_model = prepare_communication_model(data, arcs, repaired_transport)
        solution = solve_relay_subproblem(
            data,
            repaired_transport,
            repaired_model.communication_segments,
            repaired_model.relay_states,
            repaired_model.atlas,
            enforce_resource_no_overlap=True,
            **common,
        )
        return solution, repaired_transport, repaired_model


def solve_aligned_scenario(
    data: Q3Data,
    arcs: Mapping[tuple[str, str], ArcGeometry],
    scenario: TransportScenario,
    *,
    joint_time_limit_s: float = 180.0,
    relay_time_limit_s: float = 180.0,
) -> AlignedScenarioResult:
    """Solve and independently validate one Q2-to-Q3 aligned scenario."""
    started = perf_counter()
    prepared: PreparedCommunicationModel | None = None
    try:
        prepared = prepare_communication_model(data, arcs, scenario.solution)
        shifted = reschedule_transport_with_segment_state_grid(
            data,
            scenario.solution,
            prepared.communication_segments,
            prepared.relay_states,
            prepared.atlas,
            time_limit_s=joint_time_limit_s,
        )
        shifted_model = prepare_communication_model(data, arcs, shifted)
        solution, shifted, shifted_model = solve_relay_with_resource_repair(
            data,
            arcs,
            shifted,
            shifted_model,
            joint_time_limit_s=joint_time_limit_s,
            relay_time_limit_s=relay_time_limit_s,
        )
        diagnostics = dict(solution.diagnostics)
        diagnostics.update(
            {
                "scenario_id": scenario.scenario_id,
                "q2_method": scenario.q2_method,
                "q2_objective": scenario.source_objective,
                "q2_provenance": dict(scenario.provenance),
            }
        )
        solution = replace(solution, diagnostics=diagnostics)
        validation = validate_q3_solution(
            data,
            arcs,
            shifted_model.communication_segments,
            shifted_model.relay_states,
            shifted_model.atlas,
            solution,
        )
        return AlignedScenarioResult(
            scenario_id=scenario.scenario_id,
            q2_method=scenario.q2_method,
            q2_objective=scenario.source_objective,
            solution=solution,
            validation=validation,
            status=solution.solver_status,
            reason="" if validation.is_valid else "independent validation failed",
            runtime_s=perf_counter() - started,
            provenance=scenario.provenance,
            prepared=shifted_model,
        )
    except (InfeasibleQ2Error, RelaySchedulingError, RuntimeError, ValueError) as error:
        return AlignedScenarioResult(
            scenario_id=scenario.scenario_id,
            q2_method=scenario.q2_method,
            q2_objective=scenario.source_objective,
            solution=None,
            validation=None,
            status="NO_FEASIBLE_SOLUTION_FOUND",
            reason=str(error),
            runtime_s=perf_counter() - started,
            provenance=scenario.provenance,
            prepared=prepared,
        )


def select_best_valid_result(
    results: Sequence[AlignedScenarioResult],
) -> AlignedScenarioResult:
    """Select the lexicographically best independently valid Q3 result."""
    valid = [result for result in results if result.is_valid]
    if not valid:
        reasons = "; ".join(
            f"{result.scenario_id}: {result.reason}" for result in results
        )
        raise RuntimeError(f"no valid aligned q3 result: {reasons}")
    return min(valid, key=lambda result: result.solution.objective.as_tuple())
