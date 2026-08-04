from __future__ import annotations
from unittest import result

import pytest

from app.services.risk_management import (
    RiskParameters,
    assess_trade_risk,
)


def test_buy_position_is_sized_from_risk_limit():
    result = assess_trade_risk(
        signal="BUY",
        confidence=0.80,
        entry_price=100.0,
        capital=10_000.0,
    )

    assert result.approved is True
    assert result.risk_amount == pytest.approx(100.0)
    assert result.position_size == pytest.approx(25.0)
    assert result.position_value == pytest.approx(2_500.0)
    assert result.stop_loss_price == pytest.approx(98.0)
    assert result.take_profit_price == pytest.approx(104.0)


def test_sell_stop_and_target_are_on_correct_side():
    result = assess_trade_risk(
        signal="SELL",
        confidence=0.80,
        entry_price=100.0,
        capital=10_000.0,
    )

    assert result.approved is True
    assert result.stop_loss_price == pytest.approx(102.0)
    assert result.take_profit_price == pytest.approx(96.0)


def test_low_confidence_trade_is_rejected():
    result = assess_trade_risk(
        signal="BUY",
        confidence=0.54,
        entry_price=100.0,
        capital=10_000.0,
    )

    assert result.approved is False
    assert "confidence" in result.reason.lower()
    assert result.position_size == 0.0


def test_hold_never_opens_position():
    result = assess_trade_risk(
        signal="HOLD",
        confidence=1.0,
        entry_price=100.0,
        capital=10_000.0,
    )

    assert result.approved is False
    assert result.position_size == 0.0
    assert result.stop_loss_price is None
    assert result.take_profit_price is None


def test_position_exposure_is_capped():
    params = RiskParameters(
        risk_per_trade_percent=10.0,
        stop_loss_percent=1.0,
        maximum_position_percent=25.0,
    )

    result = assess_trade_risk(
        signal="BUY",
        confidence=0.90,
        entry_price=100.0,
        capital=10_000.0,
        parameters=params,
    )

    assert result.approved is True
    assert result.position_value == pytest.approx(2_500.0)
    assert result.position_percent == pytest.approx(25.0)


@pytest.mark.parametrize(
    ("signal", "confidence", "entry_price", "capital"),
    [
        ("INVALID", 0.8, 100.0, 10_000.0),
        ("BUY", -0.1, 100.0, 10_000.0),
        ("BUY", 1.1, 100.0, 10_000.0),
        ("BUY", 0.8, 0.0, 10_000.0),
        ("BUY", 0.8, 100.0, 0.0),
    ],
)
def test_invalid_trade_inputs_are_rejected(
    signal: str,
    confidence: float,
    entry_price: float,
    capital: float,
):
    with pytest.raises(ValueError):
        assess_trade_risk(
            signal=signal,
            confidence=confidence,
            entry_price=entry_price,
            capital=capital,
        )


def test_custom_risk_parameters_are_respected():
    params = RiskParameters(
        risk_per_trade_percent=0.5,
        stop_loss_percent=5.0,
        take_profit_percent=10.0,
        maximum_position_percent=20.0,
        minimum_confidence=0.70,
    )

    result = assess_trade_risk(
        signal="BUY",
        confidence=0.75,
        entry_price=200.0,
        capital=20_000.0,
        parameters=params,
    )

    assert result.approved is True
    assert result.risk_amount == pytest.approx(100.0)
    assert result.position_size == pytest.approx(10.0)
    assert result.stop_loss_price == pytest.approx(190.0)
    assert result.take_profit_price == pytest.approx(220.0)
