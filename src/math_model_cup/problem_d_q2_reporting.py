"""Comparable output tables and orchestration for problem D question 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import pandas as pd

from .problem_d_q2 import Q2Data, Q2Solution, load_q2_data
from .problem_d_q2_alns import solve_alns
from .problem_d_q2_candidates import solve_candidate_method
from .problem_d_q2_geometry import ArcGeometry, build_arc_matrix
from .problem_d_q2_hybrid import solve_hybrid
from .problem_d_q2_milp import solve_integrated_milp
from .problem_d_q2_validation import ValidationReport, validate_q2_solution


METHOD_ORDER = ("integrated_milp", "candidate", "alns", "hybrid")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _trip_rows(solution: Q2Solution) -> list[dict[str, object]]:
    rows = []
    for trip in sorted(solution.trips, key=lambda item: (item.start_time_s, item.trip_id)):
        rows.append(
            {
                "trip_id": trip.trip_id,
                "aircraft_model": trip.plan.model_id,
                "aircraft_id": trip.aircraft_id,
                "battery_id": trip.battery_id,
                "start_time_s": trip.start_time_s,
                "return_time_s": trip.return_time_s,
                "battery_ready_time_s": trip.battery_ready_time_s,
                "duration_s": trip.plan.duration_s,
                "energy_kwh": trip.plan.energy_kwh,
                "return_soc_percent": trip.plan.return_soc_percent,
                "total_mass_kg": trip.plan.total_mass_kg,
                "total_volume_m3": trip.plan.total_volume_m3,
                "stop_count": len(trip.plan.stops),
                "stop_sequence": ">".join(stop.service_id for stop in trip.plan.stops),
                "box_count": len(trip.plan.box_ids),
                "box_ids": "|".join(trip.plan.box_ids),
            }
        )
    return rows


def _delivery_rows(data: Q2Data, solution: Q2Solution) -> list[dict[str, object]]:
    rows = []
    for record in sorted(solution.deliveries, key=lambda item: item.box_id):
        box = data.boxes[record.box_id]
        hard_deadline = box.hard_deadline_s
        rows.append(
            {
                "box_id": record.box_id,
                "trip_id": record.trip_id,
                "service_id": record.service_id,
                "material_type": box.material_type,
                "mass_kg": box.mass_kg,
                "volume_m3": box.volume_m3,
                "first_batch": box.first_batch,
                "priority_weight": box.priority_weight,
                "delivery_time_s": record.delivery_time_s,
                "expected_time_s": box.expected_time_s,
                "hard_deadline_s": hard_deadline,
                "expected_lateness_s": max(0.0, record.delivery_time_s - box.expected_time_s),
                "hard_deadline_met": hard_deadline is None
                or record.delivery_time_s <= hard_deadline + 1e-7,
            }
        )
    return rows


def _validation_rows(report: ValidationReport) -> list[dict[str, object]]:
    if not report.issues:
        return [{"status": "PASS", "code": "", "reference": "", "message": ""}]
    return [
        {
            "status": "FAIL",
            "code": issue.code,
            "reference": issue.reference,
            "message": issue.message,
        }
        for issue in report.issues
    ]


def _constraint_row(
    data: Q2Data,
    solution: Q2Solution,
    report: ValidationReport,
) -> dict[str, object]:
    delivered = [record.box_id for record in solution.deliveries]
    delivered_set = set(delivered)
    issue_counts: dict[str, int] = {}
    for issue in report.issues:
        issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1
    return {
        "validation": "PASS" if report.is_valid else "FAIL",
        "required_box_count": len(data.boxes),
        "delivered_box_count": len(delivered_set),
        "missing_box_count": len(set(data.boxes) - delivered_set),
        "duplicate_box_count": len(delivered) - len(delivered_set),
        "hard_deadline_violation_count": issue_counts.get("hard_deadline", 0),
        "aircraft_overlap_count": issue_counts.get("aircraft_overlap", 0),
        "battery_overlap_count": issue_counts.get("battery_overlap", 0),
        "energy_violation_count": sum(
            trip.plan.return_soc_percent + 1e-7
            < 100.0 * data.aircraft_models[trip.plan.model_id].reserve_ratio
            for trip in solution.trips
        ),
        "minimum_return_soc_percent": min(
            (trip.plan.return_soc_percent for trip in solution.trips), default=100.0
        ),
    }


def _comparison_row(
    method: str,
    data: Q2Data,
    solution: Q2Solution,
    report: ValidationReport,
) -> dict[str, object]:
    checks = _constraint_row(data, solution, report)
    return {
        "method": method,
        "validation": checks["validation"],
        "solver_status": solution.solver_status,
        "normalized_weighted_delivery_time": solution.objective[0],
        "makespan_s": solution.objective[1],
        "energy_kwh": solution.objective[2],
        "trip_count": solution.objective[3],
        "runtime_s": solution.runtime_s,
        "delivered_box_count": checks["delivered_box_count"],
        "minimum_return_soc_percent": checks["minimum_return_soc_percent"],
        "fallback": solution.diagnostics.get("fallback", ""),
        "backend": solution.diagnostics.get("backend", ""),
    }


def _summary_markdown(comparison: pd.DataFrame, solutions: Mapping[str, Q2Solution]) -> str:
    valid = comparison[comparison["validation"] == "PASS"]
    if valid.empty:
        recommendation = "无方案通过独立校验，不能推荐。"
    else:
        objectives = {method: solutions[method].objective for method in valid["method"]}
        best_objective = min(objectives.values())
        best_methods = [method for method, value in objectives.items() if value == best_objective]
        best_rows = valid[valid["method"].isin(best_methods)]
        own_incumbents = best_rows[best_rows["fallback"].fillna("") == ""]
        if own_incumbents.empty:
            fastest = best_rows.sort_values("runtime_s").iloc[0]["method"]
            recommendation = (
                "当前最佳目标对应的入口均来自回退解，尚不能据此判定四种算法的寻优优劣。"
                f"若只考虑阶段一保底运行时间，{fastest} 最短；正式推荐需等待阶段二独立解。"
            )
        else:
            recommendation = (
                "按统一词典序目标，当前推荐："
                + "、".join(own_incumbents["method"])
                + "。目标并列时优先选择无回退、运行时间更短且可解释性更强的方案。"
            )
    display = comparison[
        [
            "method",
            "validation",
            "solver_status",
            "normalized_weighted_delivery_time",
            "makespan_s",
            "energy_kwh",
            "trip_count",
            "runtime_s",
            "fallback",
        ]
    ]
    try:
        table = display.to_markdown(index=False)
    except ImportError:
        table = display.to_csv(index=False)
    return "\n".join(
        [
            "# D题第二问四方案比较摘要",
            "",
            recommendation,
            "",
            table,
            "",
            "说明：比较仅在通过独立约束校验的方案之间进行；fallback 非空表示该方法本轮未获得自己的可行 incumbent。",
            "",
        ]
    )


def write_q2_outputs(
    output_dir: Path,
    data: Q2Data,
    arcs: Mapping[Tuple[str, str], ArcGeometry],
    solutions: Mapping[str, Q2Solution],
) -> None:
    """Write audit-ready per-method outputs and a common comparison table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    for method, solution in solutions.items():
        method_dir = output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        report = validate_q2_solution(data, arcs, solution)
        trip_rows = _trip_rows(solution)
        delivery_rows = _delivery_rows(data, solution)
        _write_csv(pd.DataFrame(trip_rows), method_dir / "trips.csv")
        _write_csv(pd.DataFrame(delivery_rows), method_dir / "deliveries.csv")
        _write_csv(
            pd.DataFrame(
                {
                    "aircraft_id": row["aircraft_id"],
                    "trip_id": row["trip_id"],
                    "start_time_s": row["start_time_s"],
                    "end_time_s": row["return_time_s"],
                    "aircraft_model": row["aircraft_model"],
                }
                for row in trip_rows
            ),
            method_dir / "aircraft_timeline.csv",
        )
        _write_csv(
            pd.DataFrame(
                {
                    "battery_id": row["battery_id"],
                    "trip_id": row["trip_id"],
                    "flight_start_time_s": row["start_time_s"],
                    "return_time_s": row["return_time_s"],
                    "ready_time_s": row["battery_ready_time_s"],
                    "charge_duration_s": row["battery_ready_time_s"] - row["return_time_s"],
                    "aircraft_model": row["aircraft_model"],
                }
                for row in trip_rows
            ),
            method_dir / "battery_timeline.csv",
        )
        _write_csv(pd.DataFrame(_validation_rows(report)), method_dir / "validation.csv")
        checks = _constraint_row(data, solution, report)
        _write_csv(pd.DataFrame([checks]), method_dir / "constraint_check.csv")
        summary = {
            "method": method,
            "solver_status": solution.solver_status,
            "validation": "PASS" if report.is_valid else "FAIL",
            "runtime_s": solution.runtime_s,
            "objective": {
                "normalized_weighted_delivery_time": solution.objective[0],
                "makespan_s": solution.objective[1],
                "energy_kwh": solution.objective[2],
                "trip_count": solution.objective[3],
            },
            "constraint_check": checks,
            "diagnostics": _json_safe(solution.diagnostics),
        }
        (method_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        comparison_rows.append(_comparison_row(method, data, solution, report))

    comparison = pd.DataFrame(comparison_rows)
    _write_csv(comparison, output_dir / "method_comparison.csv")
    (output_dir / "summary.md").write_text(
        _summary_markdown(comparison, solutions), encoding="utf-8"
    )


def run_methods(
    problem_dir: Path,
    output_dir: Path,
    methods: Sequence[str] = METHOD_ORDER,
    seed: int = 0,
    quick: bool = False,
    time_limit_s: float = 1800.0,
) -> Mapping[str, Q2Solution]:
    """Load one data set, run requested methods, validate, and write common outputs."""
    unknown = sorted(set(methods) - set(METHOD_ORDER))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; choose from {METHOD_ORDER}")
    data = load_q2_data(Path(problem_dir))
    arcs = build_arc_matrix(data)
    effective_limit = min(float(time_limit_s), 10.0) if quick else float(time_limit_s)
    solutions: dict[str, Q2Solution] = {}
    for method in methods:
        if method == "integrated_milp":
            solution = solve_integrated_milp(data, arcs, time_limit_s=effective_limit)
        elif method == "candidate":
            solution = solve_candidate_method(data, arcs, time_limit_s=effective_limit)
        elif method == "alns":
            solution = solve_alns(data, arcs, seed=seed, iterations=100 if quick else 10_000)
        else:
            solution = solve_hybrid(
                data,
                arcs,
                seed=seed,
                iterations=50 if quick else 10_000,
                rounds=1 if quick else 2,
            )
        report = validate_q2_solution(data, arcs, solution)
        if not report.is_valid:
            details = "; ".join(f"{issue.code}: {issue.message}" for issue in report.issues)
            raise RuntimeError(f"{method} produced an invalid solution: {details}")
        solutions[method] = solution
    write_q2_outputs(Path(output_dir), data, arcs, solutions)
    return solutions
