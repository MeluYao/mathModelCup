"""Run all three methods for Problem D, question 1, and write reports."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from math_model_cup.problem_d_q1 import (  # noqa: E402
    InfeasibleProblemError,
    MethodSolution,
    ProblemData,
    load_problem_data,
    model_trip_counts,
    safe_payload,
    solution_rows,
    solve_dynamic_programming,
    solve_greedy,
    solve_milp,
    validate_solution,
)
from math_model_cup.problem_d_q1_reporting import (  # noqa: E402
    generate_complete_outputs,
)


METHOD_LABELS = {
    "greedy": "BFD+合并改进",
    "dynamic_programming": "词典序动态规划",
    "milp": "集合划分 MILP",
}


def _comparison_rows(solutions: Sequence[MethodSolution]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for solution in solutions:
        counts = model_trip_counts(solution)
        rows.append(
            {
                "方法": METHOD_LABELS[solution.method],
                "架次数": solution.trip_count,
                "总能耗（kWh）": solution.total_energy_kwh,
                "累计作业时间（s）": solution.total_time_s,
                "累计作业时间（h）": solution.total_time_s / 3600.0,
                "A型架次": counts.get("A", 0),
                "B型架次": counts.get("B", 0),
                "C型架次": counts.get("C", 0),
                "算法运行时间（s）": solution.runtime_s,
            }
        )
    return rows


def _safe_payload_rows(data: ProblemData, reserve_ratio: float) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for service_id in sorted(data.geometries):
        geometry = data.geometries[service_id]
        row: Dict[str, object] = {
            "服务区编号": service_id,
            "水平距离（m）": geometry.distance_m,
            "沿线DEM峰值（m）": geometry.peak_ground_m,
            "计划巡航海拔（m）": geometry.cruise_altitude_m,
            "O01爬升高度（m）": geometry.center_climb_m,
            "服务区返程爬升高度（m）": geometry.service_climb_m,
        }
        for model_id in sorted(data.aircraft):
            row[f"{model_id}型最大安全载荷（kg）"] = safe_payload(
                data.aircraft[model_id], geometry, reserve_ratio
            )
        rows.append(row)
    return rows


def _sensitivity_rows(
    data: ProblemData, reserve_ratios: Iterable[float]
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for reserve_ratio in reserve_ratios:
        row: Dict[str, object] = {"返航安全余量": reserve_ratio}
        for model_id in sorted(data.aircraft):
            values = [
                safe_payload(data.aircraft[model_id], geometry, reserve_ratio)
                for geometry in data.geometries.values()
            ]
            row[f"{model_id}型最小安全载荷（kg）"] = min(values)
            row[f"{model_id}型最大安全载荷（kg）"] = max(values)
        try:
            solution = solve_dynamic_programming(data, reserve_ratio)
        except InfeasibleProblemError:
            row["是否可行"] = "否"
            row["最少总架次"] = None
            row["最优累计时间（h）"] = None
            row["对应总能耗（kWh）"] = None
        else:
            row["是否可行"] = "是"
            row["最少总架次"] = solution.trip_count
            row["最优累计时间（h）"] = solution.total_time_s / 3600.0
            row["对应总能耗（kWh）"] = solution.total_energy_kwh
        rows.append(row)
    return rows


def _markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    formatted = frame.copy()
    for column in formatted.select_dtypes(include="number").columns:
        formatted[column] = formatted[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}"
        )
    headers = [str(column) for column in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in formatted.columns) + " |")
    return "\n".join(lines)


def _write_summary(
    output_path: Path,
    solutions: Sequence[MethodSolution],
    comparison: pd.DataFrame,
    safe_payloads: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    lines = [
        "# D 题第一问三种方法求解结果",
        "",
        "## 计算口径",
        "",
        "题面未显式给出水平巡航能耗与爬升附加能耗的子公式。本计算采用：",
        "",
        "- 水平能耗 = 电池可用能量 × 水平距离 ÷ 当前载荷等效航程；",
        "- 爬升能耗 = 起飞质量 × 9.81 × 爬升高度 ÷ (3.6×10^6 × 爬升效率)；",
        "- 去程携带全部本架次货箱，返程空载；下降附加能耗为 0。",
        "",
        "## 方法比较",
        "",
        _markdown_table(comparison),
        "",
        "动态规划和 MILP 均按“架次数、累计作业时间、总能耗”词典序优化。",
        "",
        "## 默认 20% 安全余量下的最大安全载荷",
        "",
        _markdown_table(safe_payloads),
        "",
        "## 各方法逐架次答案",
        "",
    ]
    for solution in solutions:
        lines.extend(
            [
                f"### {METHOD_LABELS[solution.method]}",
                "",
                _markdown_table(pd.DataFrame(solution_rows(solution))),
                "",
            ]
        )
    lines.extend(
        [
            "## 返航安全余量灵敏度",
            "",
            _markdown_table(sensitivity),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_all(problem_dir: Path, output_dir: Path) -> Dict[str, MethodSolution]:
    results = generate_complete_outputs(problem_dir, output_dir)
    return dict(results.solutions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--problem-dir",
        type=Path,
        default=REPOSITORY_ROOT / "D题",
        help="directory containing the supplied D-problem files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "outputs" / "problem_d_q1",
        help="directory for generated CSV and Markdown reports",
    )
    arguments = parser.parse_args()
    solutions = run_all(arguments.problem_dir, arguments.output_dir)
    for method in ("greedy", "dynamic_programming", "milp"):
        solution = solutions[method]
        print(
            f"{METHOD_LABELS[method]}: {solution.trip_count} trips, "
            f"{solution.total_energy_kwh:.6f} kWh, "
            f"{solution.total_time_s / 3600.0:.6f} h, "
            f"runtime {solution.runtime_s:.6f} s"
        )
    print(f"Outputs: {arguments.output_dir.resolve()}")


if __name__ == "__main__":
    main()
