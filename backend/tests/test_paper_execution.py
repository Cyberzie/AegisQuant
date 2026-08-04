import pytest

from app.services.paper_execution import (
    PaperExecutionEngine,
)
from app.services.portfolio import Portfolio


def make_engine(
    capital: float = 10_000.0,
) -> PaperExecutionEngine:
    return PaperExecutionEngine(
        Portfolio(
            starting_capital=capital
        ),
        transaction_cost_percent=0.10,
        slippage_percent=0.05,
    )


def test_hold_does_not_execute():
    engine = make_engine()

    result = engine.execute(
        symbol="AAPL",
        signal="HOLD",
        confidence=0.80,
        entry_price=100.0,
    )

    assert result.executed is False
    assert result.quantity == pytest.approx(0.0)
    assert result.realized_pnl == pytest.approx(0.0)


def test_buy_creates_position():
    engine = make_engine()

    result = engine.execute(
        symbol="AAPL",
        signal="BUY",
        confidence=0.80,
        entry_price=100.0,
    )

    assert result.executed is True
    assert result.quantity > 0

    position = engine.portfolio.positions["AAPL"]

    assert position.quantity == pytest.approx(
        result.quantity
    )


def test_buy_applies_positive_slippage():
    engine = make_engine()

    result = engine.execute(
        symbol="AAPL",
        signal="BUY",
        confidence=0.80,
        entry_price=100.0,
    )

    assert result.execution_price == pytest.approx(
        100.05
    )


def test_buy_reduces_cash():
    engine = make_engine()

    starting_cash = engine.portfolio.cash

    result = engine.execute(
        symbol="AAPL",
        signal="BUY",
        confidence=0.80,
        entry_price=100.0,
    )

    assert engine.portfolio.cash < starting_cash
    assert result.cash_after == pytest.approx(
        engine.portfolio.cash
    )


def test_sell_without_position_does_not_execute():
    engine = make_engine()

    result = engine.execute(
        symbol="AAPL",
        signal="SELL",
        confidence=0.80,
        entry_price=100.0,
    )

    assert result.executed is False
    assert "No open position" in result.reason


def test_sell_closes_existing_position():
    engine = make_engine()

    buy = engine.execute(
        symbol="AAPL",
        signal="BUY",
        confidence=0.80,
        entry_price=100.0,
    )

    sell = engine.execute(
        symbol="AAPL",
        signal="SELL",
        confidence=0.80,
        entry_price=110.0,
    )

    assert buy.executed is True
    assert sell.executed is True
    assert "AAPL" not in engine.portfolio.positions
    assert sell.realized_pnl > 0


def test_sell_applies_negative_slippage():
    engine = make_engine()

    engine.execute(
        symbol="AAPL",
        signal="BUY",
        confidence=0.80,
        entry_price=100.0,
    )

    result = engine.execute(
        symbol="AAPL",
        signal="SELL",
        confidence=0.80,
        entry_price=110.0,
    )

    assert result.execution_price == pytest.approx(
        109.945
    )


def test_invalid_signal_is_rejected():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="Signal must be",
    ):
        engine.execute(
            symbol="AAPL",
            signal="INVALID",
            confidence=0.80,
            entry_price=100.0,
        )


def test_invalid_price_is_rejected():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="Entry price",
    ):
        engine.execute(
            symbol="AAPL",
            signal="BUY",
            confidence=0.80,
            entry_price=0,
        )