from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def d_problem_dir() -> Path:
    value = os.environ.get("D_PROBLEM_DIR")
    if not value:
        pytest.skip("D_PROBLEM_DIR is not configured")
    return Path(value)


@pytest.fixture(scope="session")
def q3_data(d_problem_dir: Path):
    from math_model_cup.problem_d_q3 import load_q3_data

    return load_q3_data(d_problem_dir)


@pytest.fixture(scope="session")
def q3_arcs(q3_data):
    from math_model_cup.problem_d_q2_geometry import build_arc_matrix

    return build_arc_matrix(q3_data.transport)
