import pytest

from app.services.performance_tracker import (
    PerformanceTracker,
)


def make_tracker() -> PerformanceTracker:
    return PerformanceTracker(
        starting_capital=10_000.0
    )


def test_initial_snapshot():
    tracker = make_tracker()

    snapshot = tracker.snapshot()

    assert snapshot.starting_capital == 10_000.0
    assert snapshot.current_equity == 10_000.0
    assert snapshot.realized_pnl == 0.0
    assert snapshot.total_trades == 0
    assert snapshot.winning_trades == 0
    assert snapshot.losing_trades == 0
    assert snapshot.win_rate_percent == 0.0
    assert snapshot.profit_factor == 0.0
    assert snapshot.cumulative_return_percent == 0.0
    assert snapshot.maximum_drawdown_percent == 0.0
    assert snapshot.sharpe_ratio == 0.0


def test_record_winning_trade():
    tracker = make_tracker()

    record = tracker.record_trade(
        symbol="AAPL",
        signal="SELL",
        quantity=10.0,
        execution_price=110.0,
        realized_pnl=100.0,
        current_equity=10_100.0,
    )

    assert record.symbol == "AAPL"
    assert record.realized_pnl == 100.0
    assert tracker.total_trades() == 1
    assert tracker.winning_trades() == 1
    assert tracker.losing_trades() == 0
    assert tracker.win_rate_percent() == 100.0


def test_record_losing_trade():
    tracker = make_tracker()

    tracker.record_trade(
        symbol="AAPL",
        signal="SELL",
        quantity=10.0,
        execution_price=90.0,
        realized_pnl=-100.0,
        current_equity=9_900.0,
    )

    assert tracker.total_trades() == 1
    assert tracker.winning_trades() == 0
    assert tracker.losing_trades() == 1
    assert tracker.win_rate_percent() == 0.0


def test_profit_factor():
    tracker = make_tracker()

    tracker.record_trade(
        symbol="AAPL",
        signal="SELL",
        quantity=10.0,
        execution_price=110.0,
        realized_pnl=200.0,
        current_equity=10_200.0,
    )

    tracker.record_trade(
        symbol="MSFT",
        signal="SELL",
        quantity=10.0,
        execution_price=95.0,
        realized_pnl=-100.0,
        current_equity=10_100.0,
    )

    assert tracker.profit_factor() == pytest.approx(
        2.0
    )


def test_cumulative_return():
    tracker = make_tracker()

    tracker.update_equity(10_500.0)

    assert tracker.cumulative_return_percent() == pytest.approx(
        5.0
    )


def test_maximum_drawdown():
    tracker = make_tracker()

    tracker.update_equity(11_000.0)
    tracker.update_equity(10_450.0)
    tracker.update_equity(10_800.0)

    assert tracker.maximum_drawdown_percent() == pytest.approx(
        5.0
    )


def test_sharpe_ratio_is_zero_without_variability():
    tracker = make_tracker()

    tracker.update_equity(10_000.0)
    tracker.update_equity(10_000.0)

    assert tracker.sharpe_ratio() == 0.0


def test_sharpe_ratio_is_calculated():
    tracker = make_tracker()

    tracker.update_equity(10_100.0)
    tracker.update_equity(10_200.0)
    tracker.update_equity(10_300.0)

    assert tracker.sharpe_ratio() > 0.0


def test_trade_history_is_immutable_view():
    tracker = make_tracker()

    tracker.record_trade(
        symbol="AAPL",
        signal="SELL",
        quantity=1.0,
        execution_price=110.0,
        realized_pnl=10.0,
        current_equity=10_010.0,
    )

    trades = tracker.trades

    assert isinstance(trades, tuple)
    assert len(trades) == 1


def test_invalid_starting_capital():
    with pytest.raises(
        ValueError,
        match="Starting capital",
    ):
        PerformanceTracker(0.0)


def test_invalid_trade():
    tracker = make_tracker()

    with pytest.raises(
        ValueError,
        match="Symbol",
    ):
        tracker.record_trade(
            symbol=" ",
            signal="SELL",
            quantity=1.0,
            execution_price=100.0,
            realized_pnl=10.0,
            current_equity=10_010.0,
        )


def test_invalid_signal():
    tracker = make_tracker()

    with pytest.raises(
        ValueError,
        match="BUY or SELL",
    ):
        tracker.record_trade(
            symbol="AAPL",
            signal="HOLD",
            quantity=1.0,
            execution_price=100.0,
            realized_pnl=0.0,
            current_equity=10_000.0,
        )