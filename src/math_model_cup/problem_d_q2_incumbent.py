"""Shared incumbent handling for seeded Q2 solvers."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Mapping, Tuple

from .problem_d_q2 import InfeasibleQ2Error, Q2Data, Q2Solution
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_validation import validate_q2_solution
from .problem_d_q2_urgent import PUBLISHED_MODE, URGENT_PRIORITY_MODE, urgent_search_key


def ensure_valid_initial_solution(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    initial_solution: Q2Solution | None,
    *,
    evaluation_mode: str = PUBLISHED_MODE,
) -> None:
    """Reject an invalid seed before it can become a solver incumbent."""
    if initial_solution is None:
        return
    report = validate_q2_solution(
        data, arcs, initial_solution, evaluation_mode=evaluation_mode
    )
    if report.is_valid:
        return
    details = "; ".join(
        f"{issue.code}: {issue.message}" for issue in report.issues
    )
    raise InfeasibleQ2Error(f"invalid initial solution: {details}")


def solution_key(
    solution: Q2Solution,
    *,
    data: Q2Data | None = None,
    evaluation_mode: str = PUBLISHED_MODE,
) -> tuple[float, ...]:
    """Return the published lexicographic objective used by every native solver."""
    if evaluation_mode == URGENT_PRIORITY_MODE:
        if data is None:
            raise ValueError("urgent solution comparison requires Q2 data")
        return urgent_search_key(data, solution)
    return solution.objective


def finalize_seeded_solution(
    method: str,
    native_solution: Q2Solution,
    initial_solution: Q2Solution | None,
    *,
    started: float,
    native_source: str,
    diagnostics: Mapping[str, object] | None = None,
    data: Q2Data | None = None,
    evaluation_mode: str = PUBLISHED_MODE,
) -> Q2Solution:
    """Choose between a native result and its seed and record honest provenance."""
    improved = (
        initial_solution is not None
        and solution_key(
            native_solution, data=data, evaluation_mode=evaluation_mode
        )
        < solution_key(initial_solution, data=data, evaluation_mode=evaluation_mode)
    )
    seed_retained = initial_solution is not None and not improved
    chosen = initial_solution if seed_retained else native_solution
    merged = dict(chosen.diagnostics)
    merged.update(diagnostics or {})
    merged.update(
        {
            "incumbent_source": "provided_seed" if seed_retained else native_source,
            "seed_objective": (
                initial_solution.objective if initial_solution is not None else None
            ),
            "seed_retained": seed_retained,
            "native_improved_seed": improved,
            "native_candidate_objective": native_solution.objective,
            "has_feasible_incumbent": True,
            "evaluation_mode": evaluation_mode,
        }
    )
    merged.pop("fallback", None)
    return replace(
        chosen,
        method=method,
        runtime_s=perf_counter() - started,
        solver_status=chosen.solver_status or "FEASIBLE",
        diagnostics=merged,
    )
