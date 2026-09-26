"""Shared incumbent handling for seeded Q2 solvers."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from typing import Mapping, Tuple

from .problem_d_q2 import InfeasibleQ2Error, Q2Data, Q2Solution
from .problem_d_q2_geometry import ArcGeometry
from .problem_d_q2_validation import validate_q2_solution


def ensure_valid_initial_solution(
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    initial_solution: Q2Solution | None,
) -> None:
    """Reject an invalid seed before it can become a solver incumbent."""
    if initial_solution is None:
        return
    report = validate_q2_solution(data, arcs, initial_solution)
    if report.is_valid:
        return
    details = "; ".join(
        f"{issue.code}: {issue.message}" for issue in report.issues
    )
    raise InfeasibleQ2Error(f"invalid initial solution: {details}")


def solution_key(solution: Q2Solution) -> tuple[float, float, float, int]:
    """Return the published lexicographic objective used by every native solver."""
    return solution.objective


def finalize_seeded_solution(
    method: str,
    native_solution: Q2Solution,
    initial_solution: Q2Solution | None,
    *,
    started: float,
    native_source: str,
    diagnostics: Mapping[str, object] | None = None,
) -> Q2Solution:
    """Choose between a native result and its seed and record honest provenance."""
    improved = (
        initial_solution is not None
        and solution_key(native_solution) < solution_key(initial_solution)
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
