import pytest

from app.services.ensemble_signal import EnsembleSignal
from app.services.paper_trading import PaperTradingEngine
from app.services.portfolio import Portfolio


def make_engine() -> PaperTradingEngine:
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    return PaperTradingEngine(
        portfolio,
        transaction_cost_percent=0.10,
        slippage_percent=0.05,
    )


def make_buy_ensemble() -> EnsembleSignal:
    return EnsembleSignal(
        signal="BUY",
        confidence=0.80,
        rule_signal="BUY",
        rule_confidence=0.75,
        ml_signal="BUY",
        ml_confidence=0.85,
        ml_expected_return_percent=2.5,
        rule_weight=0.4,
        ml_weight=0.6,
    )


def make_hold_ensemble() -> EnsembleSignal:
    return EnsembleSignal(
        signal="HOLD",
        confidence=0.80,
        rule_signal="HOLD",
        rule_confidence=0.80,
        ml_signal="HOLD",
        ml_confidence=0.80,
        ml_expected_return_percent=0.0,
        rule_weight=0.4,
        ml_weight=0.6,
    )


def test_buy_signal_flows_through_decision_and_execution():
    engine = make_engine()

    result = engine.process(
        symbol="AAPL",
        ensemble=make_buy_ensemble(),
        entry_price=100.0,
    )

    assert result.symbol == "AAPL"
    assert result.entry_price == 100.0
    assert result.capital_before == pytest.approx(
        10_000.0
    )

    assert result.decision.signal == "BUY"
    assert result.decision.confidence == pytest.approx(
        0.80
    )
    assert result.decision.expected_return_percent == pytest.approx(
        2.5
    )

    assert result.decision.risk.approved is True

    assert result.execution.executed is True
    assert result.execution.signal == "BUY"
    assert result.execution.quantity > 0

    assert "AAPL" in engine.portfolio.positions


def test_hold_signal_produces_no_execution():
    engine = make_engine()

    result = engine.process(
        symbol="AAPL",
        ensemble=make_hold_ensemble(),
        entry_price=100.0,
    )

    assert result.decision.signal == "HOLD"
    assert result.decision.risk.approved is False

    assert result.execution.executed is False
    assert result.execution.quantity == pytest.approx(
        0.0
    )

    assert "AAPL" not in engine.portfolio.positions


def test_low_confidence_signal_is_rejected_by_risk_layer():
    engine = make_engine()

    ensemble = EnsembleSignal(
        signal="BUY",
        confidence=0.40,
        rule_signal="BUY",
        rule_confidence=0.40,
        ml_signal="BUY",
        ml_confidence=0.40,
        ml_expected_return_percent=1.0,
        rule_weight=0.4,
        ml_weight=0.6,
    )

    result = engine.process(
        symbol="AAPL",
        ensemble=ensemble,
        entry_price=100.0,
    )

    assert result.decision.signal == "BUY"
    assert result.decision.risk.approved is False

    assert result.execution.executed is False
    assert result.execution.quantity == pytest.approx(
        0.0
    )

    assert "AAPL" not in engine.portfolio.positions


def test_sell_signal_closes_existing_position():
    engine = make_engine()

    buy = engine.process(
        symbol="AAPL",
        ensemble=make_buy_ensemble(),
        entry_price=100.0,
    )

    assert buy.execution.executed is True
    assert "AAPL" in engine.portfolio.positions

    sell_ensemble = EnsembleSignal(
        signal="SELL",
        confidence=0.80,
        rule_signal="SELL",
        rule_confidence=0.80,
        ml_signal="SELL",
        ml_confidence=0.80,
        ml_expected_return_percent=-2.0,
        rule_weight=0.4,
        ml_weight=0.6,
    )

    sell = engine.process(
        symbol="AAPL",
        ensemble=sell_ensemble,
        entry_price=110.0,
    )

    assert sell.decision.signal == "SELL"
    assert sell.decision.risk.approved is True
    assert sell.execution.executed is True

    assert "AAPL" not in engine.portfolio.positions


def test_capital_before_reflects_existing_positions():
    engine = make_engine()

    buy = engine.process(
        symbol="AAPL",
        ensemble=make_buy_ensemble(),
        entry_price=100.0,
    )

    assert buy.execution.executed is True

    result = engine.process(
        symbol="AAPL",
        ensemble=make_hold_ensemble(),
        entry_price=110.0,
    )

    assert result.capital_before > 10_000.0


def test_empty_symbol_is_rejected():
    engine = make_engine()

    with pytest.raises(ValueError, match="Symbol"):
        engine.process(
            symbol=" ",
            ensemble=make_buy_ensemble(),
            entry_price=100.0,
        )


def test_invalid_price_is_rejected():
    engine = make_engine()

    with pytest.raises(
        ValueError,
        match="Entry price",
    ):
        engine.process(
            symbol="AAPL",
            ensemble=make_buy_ensemble(),
            entry_price=0.0,
        )