import pytest

from app.services.technical_indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)


def test_sma():
    prices = [1, 2, 3, 4, 5]

    result = sma(prices, 3)

    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_ema():
    prices = [1, 2, 3, 4, 5]

    result = ema(prices, 3)

    assert result[:2] == [None, None]
    assert result[2] == pytest.approx(2.0)
    assert result[3] == pytest.approx(3.0)
    assert result[4] == pytest.approx(4.0)


def test_rsi_returns_100_for_continuous_gains():
    prices = list(range(1, 17))

    result = rsi(prices, period=14)

    assert result[13] is None
    assert result[14] == pytest.approx(100.0)
    assert result[15] == pytest.approx(100.0)


def test_macd_structure():
    prices = list(range(1, 41))

    result = macd(
        prices,
        fast_period=3,
        slow_period=5,
        signal_period=2,
    )

    assert set(result) == {
        "macd",
        "signal",
        "histogram",
    }

    assert len(result["macd"]) == len(prices)
    assert len(result["signal"]) == len(prices)
    assert len(result["histogram"]) == len(prices)


def test_bollinger_bands():
    prices = [1, 2, 3, 4, 5]

    result = bollinger_bands(
        prices,
        period=3,
        deviations=2,
    )

    assert result["middle"][:2] == [None, None]
    assert result["middle"][2] == pytest.approx(2.0)

    assert result["upper"][2] is not None
    assert result["lower"][2] is not None

    assert result["upper"][2] > result["middle"][2]
    assert result["lower"][2] < result["middle"][2]


def test_atr():
    highs = [11, 12, 13, 14, 15]
    lows = [9, 10, 11, 12, 13]
    closes = [10, 11, 12, 13, 14]

    result = atr(
        highs,
        lows,
        closes,
        period=3,
    )

    assert result[:3] == [None, None, None]
    assert result[3] == pytest.approx(2.0)
    assert result[4] == pytest.approx(2.0)


def test_invalid_period():
    with pytest.raises(ValueError):
        sma([1, 2, 3], 0)

    with pytest.raises(ValueError):
        ema([1, 2, 3], 0)

    with pytest.raises(ValueError):
        rsi([1, 2, 3], 0)


def test_atr_requires_equal_lengths():
    with pytest.raises(ValueError):
        atr(
            [10, 11, 12],
            [9, 10],
            [9.5, 10.5, 11.5],
            period=2,
        )