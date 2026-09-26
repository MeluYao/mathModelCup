from __future__ import annotations

import numpy as np
import pytest

from math_model_cup.problem_d_q3 import LinkBudget
from math_model_cup.problem_d_q3_communication import (
    DemGrid,
    Position3D,
    evaluate_link,
    evaluate_relay_link,
    fspl_db,
    link_loss_limit_db,
)


@pytest.fixture
def link_budget() -> LinkBudget:
    return LinkBudget(
        frequency_mhz=2400.0,
        system_loss_db=3.0,
        obstruction_loss_db=10.0,
        sensitivity_dbm=-100.0,
        fade_margin_db=10.0,
        transport_tx_dbm=20.0,
        transport_gain_dbi=3.0,
        relay_access_tx_dbm=20.0,
        relay_access_gain_dbi=6.0,
        relay_backhaul_tx_dbm=19.0,
        relay_backhaul_gain_dbi=8.0,
        gateway_tx_dbm=27.0,
        gateway_gain_dbi=12.0,
        gateway_height_agl_m=20.0,
    )


def test_fspl_at_one_kilometre_matches_2400_mhz_reference() -> None:
    assert fspl_db(2400.0, 1.0) == pytest.approx(100.044, abs=0.002)


def test_bidirectional_loss_limits_match_statement(link_budget: LinkBudget) -> None:
    assert link_loss_limit_db(link_budget, "direct") == pytest.approx(122.0)
    assert link_loss_limit_db(link_budget, "access") == pytest.approx(116.0)
    assert link_loss_limit_db(link_budget, "backhaul") == pytest.approx(126.0)


def test_obstruction_adds_fixed_loss(link_budget: LinkBudget) -> None:
    clear = DemGrid(
        elevations_m=np.zeros((3, 3), dtype=float),
        x_origin=0.0,
        y_origin=3.0,
        x_scale=1.0,
        y_scale=1.0,
    )
    blocked_values = np.zeros((3, 3), dtype=float)
    blocked_values[1, 1] = 200.0
    blocked = DemGrid(
        elevations_m=blocked_values,
        x_origin=0.0,
        y_origin=3.0,
        x_scale=1.0,
        y_scale=1.0,
    )
    origin = Position3D(0.5, 2.5, 100.0)
    destination = Position3D(2.5, 0.5, 100.0)

    clear_result = evaluate_link(link_budget, clear, origin, destination, "direct")
    blocked_result = evaluate_link(link_budget, blocked, origin, destination, "direct")

    assert not clear_result.obstructed
    assert blocked_result.obstructed
    assert blocked_result.loss_db - clear_result.loss_db == pytest.approx(10.0)


def test_relay_link_requires_both_hops(link_budget: LinkBudget) -> None:
    dem = DemGrid(
        elevations_m=np.zeros((1, 3), dtype=float),
        x_origin=0.0,
        y_origin=1.0,
        x_scale=1.0,
        y_scale=1.0,
    )
    transport = Position3D(0.5, 0.5, 100.0)
    relay = Position3D(1.5, 0.5, 100.0)
    gateway = Position3D(2.5, 0.5, 100.0)

    result = evaluate_relay_link(link_budget, dem, transport, relay, gateway)

    assert result.available == (result.access.available and result.backhaul.available)
    assert result.minimum_margin_db == pytest.approx(
        min(result.access.margin_db, result.backhaul.margin_db)
    )
