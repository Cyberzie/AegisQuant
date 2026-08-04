import pytest

from app.services.portfolio import Portfolio


def test_portfolio_starts_with_full_cash():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    assert portfolio.cash == pytest.approx(10_000.0)
    assert portfolio.realized_pnl == pytest.approx(0.0)
    assert portfolio.positions == {}


def test_buy_opens_position():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    position = portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=100.0,
    )

    assert position.quantity == pytest.approx(50.0)
    assert (
        position.average_entry_price
        == pytest.approx(100.0)
    )
    assert portfolio.cash == pytest.approx(5_000.0)


def test_multiple_buys_use_weighted_average_price():
    portfolio = Portfolio(
        starting_capital=20_000.0
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=100.0,
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=120.0,
    )

    position = portfolio.positions["AAPL"]

    assert position.quantity == pytest.approx(100.0)
    assert (
        position.average_entry_price
        == pytest.approx(110.0)
    )

    assert portfolio.cash == pytest.approx(9_000.0)


def test_sell_realizes_profit():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=100.0,
    )

    realized = portfolio.sell(
        symbol="AAPL",
        quantity=50.0,
        price=120.0,
    )

    assert realized == pytest.approx(1_000.0)
    assert portfolio.realized_pnl == pytest.approx(
        1_000.0
    )
    assert portfolio.cash == pytest.approx(
        11_000.0
    )
    assert portfolio.positions == {}


def test_sell_at_loss_realizes_loss():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=100.0,
    )

    realized = portfolio.sell(
        symbol="AAPL",
        quantity=50.0,
        price=90.0,
    )

    assert realized == pytest.approx(-500.0)
    assert portfolio.realized_pnl == pytest.approx(
        -500.0
    )
    assert portfolio.cash == pytest.approx(
        9_500.0
    )


def test_partial_sell_keeps_remaining_position():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=100.0,
    )

    realized = portfolio.sell(
        symbol="AAPL",
        quantity=20.0,
        price=120.0,
    )

    assert realized == pytest.approx(400.0)

    position = portfolio.positions["AAPL"]

    assert position.quantity == pytest.approx(30.0)
    assert (
        position.average_entry_price
        == pytest.approx(100.0)
    )


def test_unrealized_profit_is_calculated():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=100.0,
    )

    unrealized = portfolio.unrealized_pnl(
        {"AAPL": 110.0}
    )

    assert unrealized == pytest.approx(500.0)


def test_unrealized_loss_is_calculated():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=100.0,
    )

    unrealized = portfolio.unrealized_pnl(
        {"AAPL": 90.0}
    )

    assert unrealized == pytest.approx(-500.0)


def test_equity_includes_position_market_value():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=100.0,
    )

    equity = portfolio.equity(
        {"AAPL": 110.0}
    )

    assert equity == pytest.approx(10_500.0)


def test_snapshot_contains_portfolio_state():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=50.0,
        price=100.0,
    )

    snapshot = portfolio.snapshot(
        {"AAPL": 110.0}
    )

    assert snapshot.cash == pytest.approx(5_000.0)
    assert snapshot.equity == pytest.approx(
        10_500.0
    )
    assert snapshot.realized_pnl == pytest.approx(
        0.0
    )
    assert snapshot.unrealized_pnl == pytest.approx(
        500.0
    )
    assert snapshot.positions["AAPL"].quantity == (
        pytest.approx(50.0)
    )


def test_cannot_buy_without_enough_cash():
    portfolio = Portfolio(
        starting_capital=1_000.0
    )

    with pytest.raises(
        ValueError,
        match="Insufficient cash",
    ):
        portfolio.buy(
            symbol="AAPL",
            quantity=20.0,
            price=100.0,
        )


def test_cannot_sell_more_than_position():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    portfolio.buy(
        symbol="AAPL",
        quantity=10.0,
        price=100.0,
    )

    with pytest.raises(
        ValueError,
        match="Cannot sell more",
    ):
        portfolio.sell(
            symbol="AAPL",
            quantity=11.0,
            price=100.0,
        )


def test_cannot_sell_without_position():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    with pytest.raises(
        ValueError,
        match="No open position",
    ):
        portfolio.sell(
            symbol="AAPL",
            quantity=1.0,
            price=100.0,
        )


def test_invalid_starting_capital_is_rejected():
    with pytest.raises(
        ValueError,
        match="Starting capital",
    ):
        Portfolio(starting_capital=0)


def test_invalid_order_values_are_rejected():
    portfolio = Portfolio(
        starting_capital=10_000.0
    )

    with pytest.raises(ValueError):
        portfolio.buy(
            symbol="",
            quantity=1.0,
            price=100.0,
        )

    with pytest.raises(ValueError):
        portfolio.buy(
            symbol="AAPL",
            quantity=0,
            price=100.0,
        )

    with pytest.raises(ValueError):
        portfolio.buy(
            symbol="AAPL",
            quantity=1.0,
            price=0,
        )