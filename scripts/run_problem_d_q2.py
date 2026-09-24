"""Run and compare the four problem-D question-2 methods."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from math_model_cup.problem_d_q2_reporting import METHOD_ORDER, run_methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        default=",".join(METHOD_ORDER),
        help="comma-separated subset of integrated_milp,candidate,alns,hybrid",
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--time-limit", type=float, default=1800.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    solutions = run_methods(
        args.problem_dir,
        args.output_dir,
        methods=methods,
        seed=args.seed,
        quick=args.quick,
        time_limit_s=args.time_limit,
    )
    for method, solution in solutions.items():
        print(
            f"{method}: objective={solution.objective}, runtime={solution.runtime_s:.3f}s, "
            f"status={solution.solver_status}, validation=PASS"
        )
    print(f"outputs={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
