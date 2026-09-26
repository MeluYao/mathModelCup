"""Run Q3 for the distinct schedules produced by the updated Q2 solvers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from math_model_cup.problem_d_q2_geometry import build_arc_matrix
from math_model_cup.problem_d_q3 import load_q3_data
from math_model_cup.problem_d_q3_aligned import (
    select_best_valid_result,
    solve_aligned_scenario,
)
from math_model_cup.problem_d_q3_reporting import (
    write_aligned_comparison,
    write_scenario_result,
)
from math_model_cup.problem_d_q3_transport_adapter import load_transport_scenario


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--q2-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenarios", default="hybrid,alns")
    parser.add_argument("--joint-time-limit", type=float, default=180.0)
    parser.add_argument("--relay-time-limit", type=float, default=180.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data = load_q3_data(args.problem_dir)
    arcs = build_arc_matrix(data.transport)
    scenario_ids = tuple(
        name.strip() for name in args.scenarios.split(",") if name.strip()
    )
    results = []
    for scenario_id in scenario_ids:
        scenario = load_transport_scenario(
            data.transport,
            arcs,
            args.q2_root / scenario_id,
            scenario_id,
        )
        result = solve_aligned_scenario(
            data,
            arcs,
            scenario,
            joint_time_limit_s=args.joint_time_limit,
            relay_time_limit_s=args.relay_time_limit,
        )
        results.append(result)
        write_scenario_result(result, args.output_dir / scenario_id)
        print(
            json.dumps(
                {
                    "scenario_id": scenario_id,
                    "status": result.status,
                    "is_valid": result.is_valid,
                    "reason": result.reason,
                    "objective": None
                    if result.solution is None
                    else result.solution.objective.as_tuple(),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    try:
        selected = select_best_valid_result(results)
    except RuntimeError:
        selected = None
    write_aligned_comparison(results, selected, args.output_dir)
    if selected is None:
        return 2
    print(
        json.dumps(
            {
                "selected_scenario_id": selected.scenario_id,
                "objective": selected.solution.objective.as_tuple(),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
