from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from math_model_cup.problem_d_q1_reporting import generate_complete_outputs


@pytest.mark.skipif(
    not os.environ.get("D_PROBLEM_DIR"), reason="D_PROBLEM_DIR is not configured"
)
def test_complete_output_generation_contains_tables_report_and_figures(tmp_path: Path) -> None:
    result = generate_complete_outputs(
        Path(os.environ["D_PROBLEM_DIR"]), tmp_path
    )

    assert result.recommended_solution.trip_count == 18
    assert len(pd.read_csv(tmp_path / "route_geometry.csv")) == 15
    assert len(pd.read_csv(tmp_path / "critical_reserve_margins.csv")) == 742
    assert len(pd.read_csv(tmp_path / "independent_trip_validation.csv")) == 18
    assert len(pd.read_csv(tmp_path / "box_coverage_check.csv")) == 80
    coarse = pd.read_csv(tmp_path / "safety_margin_sensitivity.csv")
    default_row = coarse.loc[(coarse["返航安全余量"] - 0.20).abs() < 1e-12].iloc[0]
    assert default_row["最少总架次"] == 18
    assert abs(default_row["对应总能耗（kWh）"] - 59.11990368208522) < 1e-8
    reserve_events = pd.read_csv(tmp_path / "reserve_change_events.csv")
    assert reserve_events.iloc[-1]["首先受影响服务区"] == "S008"
    assert reserve_events.iloc[-1]["是否变为不可配送"] == "是"
    paper = (tmp_path / "论文_问题一.md").read_text(encoding="utf-8")
    formula_audit = (tmp_path / "能耗公式与物理口径.md").read_text(encoding="utf-8")
    assert r"3.6\times 10^6".replace("\\\\", "\\") in formula_audit
    assert "\t" not in formula_audit
    assert r"\min\left".replace("\\\\", "\\") in paper
    assert "\t" not in paper
    for heading in (
        "模型假设",
        "符号表",
        "航段、时间与能耗模型",
        "货箱组批优化模型",
        "多目标优化",
        "安全余量灵敏度",
        "模型优点、局限性和适用条件",
    ):
        assert heading in paper

    expected_figures = {
        "dem_routes.png",
        "safe_payload_heatmap.png",
        "demand_vs_capacity.png",
        "pareto_front.png",
        "trip_utilization.png",
        "safe_payload_curves.png",
        "reserve_trip_staircase.png",
    }
    assert expected_figures == {path.name for path in (tmp_path / "figures").glob("*.png")}
    for path in (tmp_path / "figures").glob("*.png"):
        with Image.open(path) as image:
            assert image.width >= 1_200
            assert image.height >= 700
