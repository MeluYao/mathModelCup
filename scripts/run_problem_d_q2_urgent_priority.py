from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from math_model_cup.problem_d_q2 import load_q2_data
from math_model_cup.problem_d_q2_geometry import build_arc_matrix
from math_model_cup.problem_d_q2_urgent_reporting import (
    create_urgent_figures,
    run_hybrid_repetitions,
    run_urgent_methods,
    write_urgent_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run urgent-priority Q2 experiments")
    parser.add_argument("--problem-dir", type=Path, default=ROOT / "D题")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "problem_d_q2_urgent_priority",
    )
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    iterations = min(args.iterations, 20) if args.quick else args.iterations
    time_limit_s = min(args.time_limit, 5.0) if args.quick else args.time_limit
    data = load_q2_data(args.problem_dir)
    arcs = build_arc_matrix(data)
    solutions = run_urgent_methods(
        data,
        arcs,
        args.output_dir,
        seed=args.seed,
        iterations=iterations,
        time_limit_s=time_limit_s,
    )
    repetition_solutions = {}
    repetitions = run_hybrid_repetitions(
        data,
        arcs,
        solutions["hybrid"],
        seeds=tuple(range(args.seed, args.seed + 10)),
        iterations=iterations,
        time_limit_s=time_limit_s,
        output_dir=args.output_dir,
        solutions_out=repetition_solutions,
    )
    best_repetition = min(
        repetition_solutions.values(), key=lambda solution: solution.objective
    )
    if best_repetition.objective < solutions["hybrid"].objective:
        solutions["hybrid"] = best_repetition
        write_urgent_outputs(args.output_dir, data, arcs, solutions)
    paths = create_urgent_figures(args.output_dir, data, solutions, repetitions)
    print(f"output_dir={args.output_dir.resolve()}")
    print(f"figures={len(paths)}")


if __name__ == "__main__":
    main()
