from app.services.signal_engine import generate_signal


def test_generate_signal_buy():
    result = generate_signal(
        rsi_14=25,
        macd=2.0,
        macd_signal=1.0,
        sma_20=100.0,
        ema_20=101.0,
        close=105.0,
    )

    assert result.signal == "BUY"
    assert result.confidence > 0


def test_generate_signal_sell():
    result = generate_signal(
        rsi_14=75,
        macd=-2.0,
        macd_signal=-1.0,
        sma_20=100.0,
        ema_20=101.0,
        close=95.0,
    )

    assert result.signal == "SELL"
    assert result.confidence > 0


def test_generate_signal_hold_when_indicators_missing():
    result = generate_signal(
        rsi_14=None,
        macd=None,
        macd_signal=None,
        sma_20=None,
        ema_20=None,
        close=100.0,
    )

    assert result.signal == "HOLD"
    assert result.confidence == 0.0


def test_generate_signal_hold_when_score_is_weak():
    result = generate_signal(
        rsi_14=50,
        macd=1.0,
        macd_signal=1.0,
        sma_20=100.0,
        ema_20=100.0,
        close=100.0,
    )

    assert result.signal == "HOLD"
    assert result.confidence == 0.0