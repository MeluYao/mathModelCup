from __future__ import annotations

import os
from pathlib import Path

import pytest

from math_model_cup.problem_d_q2 import Q2Data, load_q2_data
from math_model_cup.problem_d_q2_geometry import build_arc_matrix


@pytest.fixture(scope="module")
def real_q2_data() -> Q2Data:
    configured = os.environ.get("D_PROBLEM_DIR")
    if not configured:
        pytest.skip("D_PROBLEM_DIR is not configured")
    return load_q2_data(Path(configured))


def test_arc_matrix_contains_all_directed_node_pairs(real_q2_data: Q2Data) -> None:
    arcs = build_arc_matrix(real_q2_data)

    assert len(arcs) == 16 * 15
    assert all(origin != destination for origin, destination in arcs)


def test_reverse_arcs_share_terrain_but_have_directional_climb(
    real_q2_data: Q2Data,
) -> None:
    arcs = build_arc_matrix(real_q2_data)
    outbound = arcs[("O01", "S015")]
    inbound = arcs[("S015", "O01")]

    assert outbound.distance_m == pytest.approx(inbound.distance_m)
    assert outbound.cruise_altitude_m == pytest.approx(inbound.cruise_altitude_m)
    assert outbound.peak_ground_m == pytest.approx(inbound.peak_ground_m)
    assert outbound.climb_m != inbound.climb_m
    assert outbound.climb_m == pytest.approx(inbound.descent_m)
    assert outbound.descent_m == pytest.approx(inbound.climb_m)
