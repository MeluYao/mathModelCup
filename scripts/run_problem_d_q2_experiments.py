"""Run D2 repeated experiments, sensitivity analysis, and paper figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from math_model_cup.problem_d_q2 import load_q2_data
from math_model_cup.problem_d_q2_candidates import generate_initial_candidates
from math_model_cup.problem_d_q2_experiments import (
    create_paper_figures,
    run_alns_repetitions,
    run_sensitivity_analysis,
)
from math_model_cup.problem_d_q2_geometry import build_arc_matrix
from math_model_cup.problem_d_q2_reporting import METHOD_ORDER, run_methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds",
        default=(
            "20260924,20260925,20260926,20260927,20260928,"
            "20260929,20260930,20260931,20260932,20260933"
        ),
        help="comma-separated ALNS seeds",
    )
    parser.add_argument("--alns-iterations", type=int, default=5000)
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def _write_summary(output_dir: Path, repetitions, sensitivity, figure_count: int) -> None:
    feasible_sensitivity = sensitivity[sensitivity["feasible"]]
    no_solution_count = int(
        (sensitivity["status"] == "NO_FEASIBLE_SOLUTION_FOUND").sum()
    )
    solver_error_count = int((sensitivity["status"] == "SOLVER_ERROR").sum())
    lines = [
        "# D题第二问阶段三实验摘要",
        "",
        "## 多随机种子 ALNS",
        "",
        f"- 实验次数：{len(repetitions)}",
        f"- 全部通过校验：{bool((repetitions['validation'] == 'PASS').all())}",
        f"- 主目标均值：{repetitions['objective_primary'].mean():.10f}",
        f"- 主目标标准差：{repetitions['objective_primary'].std(ddof=0):.10f}",
        f"- 运行时间均值：{repetitions['runtime_s'].mean():.3f} s",
        "",
        "## 敏感性分析",
        "",
        f"- 场景数：{len(sensitivity)}",
        f"- 可行场景数：{len(feasible_sensitivity)}",
        f"- 未找到可行解场景数：{no_solution_count}",
        f"- 求解错误场景数：{solver_error_count}",
        "",
        "## 图表",
        "",
        f"已生成 {figure_count} 个文件，即 7 张图的 PNG 与 PDF 双格式版本。",
        "",
        "所有数字均由本次运行自动生成，详细数据见 CSV 文件。",
        "",
    ]
    (output_dir / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    iterations = args.alns_iterations
    if args.quick:
        iterations = min(iterations, 20)

    data = load_q2_data(args.problem_dir)
    arcs = build_arc_matrix(data)
    baseline_dir = output_dir / "baseline"
    solutions = run_methods(
        args.problem_dir,
        baseline_dir,
        methods=METHOD_ORDER,
        seed=seeds[0],
        quick=args.quick,
        time_limit_s=args.time_limit,
    )
    candidate_pool = generate_initial_candidates(data, arcs, max_stops=3)
    repetitions, convergence = run_alns_repetitions(
        data,
        arcs,
        seeds=seeds,
        iterations=iterations,
        candidate_pool=candidate_pool,
    )
    sensitivity = run_sensitivity_analysis(data, arcs, candidate_pool)
    repetitions.to_csv(output_dir / "alns_repetitions.csv", index=False, encoding="utf-8-sig")
    convergence.to_csv(output_dir / "alns_convergence.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(output_dir / "sensitivity_analysis.csv", index=False, encoding="utf-8-sig")
    figures = create_paper_figures(
        output_dir / "figures",
        data,
        solutions,
        repetitions,
        convergence,
        sensitivity,
    )
    _write_summary(output_dir, repetitions, sensitivity, len(figures))
    manifest = {
        "problem_dir": str(args.problem_dir.resolve()),
        "output_dir": str(output_dir),
        "methods": list(METHOD_ORDER),
        "seeds": list(seeds),
        "alns_iterations": iterations,
        "quick": args.quick,
        "time_limit_s": args.time_limit,
        "candidate_count": len(candidate_pool),
        "figure_files": [str(path.relative_to(output_dir)) for path in figures],
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"baseline_methods={len(solutions)}")
    print(f"alns_repetitions={len(repetitions)} iterations={iterations}")
    print(
        f"sensitivity_scenarios={len(sensitivity)} "
        f"feasible={int(sensitivity['feasible'].sum())}"
    )
    print(f"figure_files={len(figures)}")
    print(f"outputs={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
