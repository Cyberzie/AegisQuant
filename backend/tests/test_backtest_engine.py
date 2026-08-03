from datetime import datetime, timedelta

from app.models.market_data import MarketData
from app.services.backtest_engine import backtest_market_data


def make_rows(
    closes: list[float],
) -> list[MarketData]:
    rows = []

    for index, close in enumerate(closes):
        rows.append(
            MarketData(
                id=index + 1,
                instrument_id=1,
                timestamp=datetime(2026, 1, 1)
                + timedelta(days=index),
                open=close,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=1000,
            )
        )

    return rows


def test_empty_market_data_returns_empty_result():
    result = backtest_market_data(
        [],
        symbol="AAPL",
    )

    assert result.symbol == "AAPL"
    assert result.total_rows == 0
    assert result.actionable_trades == 0
    assert result.win_rate == 0.0


def test_invalid_horizon_is_rejected():
    rows = make_rows([100.0] * 40)

    try:
        backtest_market_data(
            rows,
            symbol="AAPL",
            horizon=0,
        )
    except ValueError as exc:
        assert str(exc) == "Horizon must be greater than zero."
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_insufficient_history_produces_no_trades():
    rows = make_rows(
        [100.0 + index for index in range(20)]
    )

    result = backtest_market_data(
        rows,
        symbol="AAPL",
    )

    assert result.total_rows == 20
    assert result.actionable_trades == 0


def test_backtest_result_contains_valid_statistics():
    closes = []

    for index in range(80):
        if index % 10 < 5:
            closes.append(100.0 + index)
        else:
            closes.append(150.0 - index)

    rows = make_rows(closes)

    result = backtest_market_data(
        rows,
        symbol="AAPL",
        horizon=5,
    )

    assert result.total_rows == 80
    assert result.evaluated_rows > 0

    assert 0.0 <= result.win_rate <= 1.0

    assert result.actionable_trades == (
        result.winning_trades
        + result.losing_trades
    )

    assert len(result.trades) == (
        result.actionable_trades
    )

    for trade in result.trades:
        assert trade.signal in {"BUY", "SELL"}
        assert 0.0 <= trade.confidence <= 1.0
        assert trade.entry_price > 0
        assert trade.exit_price > 0


def test_backtest_does_not_create_overlapping_positions():
    closes = []

    for index in range(100):
        if index % 20 < 10:
            closes.append(100.0 + index)
        else:
            closes.append(200.0 - index)

    rows = make_rows(closes)

    horizon = 5

    result = backtest_market_data(
        rows,
        symbol="AAPL",
        horizon=horizon,
    )

    for previous, current in zip(
        result.trades,
        result.trades[1:],
    ):
        previous_exit = previous.timestamp + timedelta(
            days=horizon
        )

        assert current.timestamp > previous_exit


def test_trade_entry_occurs_after_signal_bar():
    closes = [
        100.0 + index
        for index in range(80)
    ]

    rows = make_rows(closes)

    result = backtest_market_data(
        rows,
        symbol="AAPL",
        horizon=5,
    )

    for trade in result.trades:
        matching_rows = [
            row
            for row in rows
            if row.timestamp == trade.timestamp
        ]

        assert matching_rows

        entry_row = matching_rows[0]

        assert trade.entry_price == entry_row.close