from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalResult:
    signal: str
    confidence: float


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(value, maximum))


def generate_signal(
    *,
    rsi_14: float | None,
    macd: float | None,
    macd_signal: float | None,
    sma_20: float | None,
    ema_20: float | None,
    close: float,
) -> SignalResult:
    """
    Generate a deterministic multi-factor trading signal.

    The signal engine evaluates three broad categories of evidence:

    1. Trend
       - Price relative to SMA-20.
       - Price relative to EMA-20.

    2. Momentum
       - MACD relative to its signal line.
       - MACD histogram direction.
       - RSI regime.

    3. Confirmation
       - Agreement between trend and momentum.

    BUY and SELL require meaningful directional agreement.
    Otherwise the engine returns HOLD.

    Confidence is continuous between 0.0 and 1.0 and reflects
    the strength and agreement of the available evidence.

    No future information is used.
    """

    required = (
        rsi_14,
        macd,
        macd_signal,
        sma_20,
        ema_20,
    )

    if any(value is None for value in required):
        return SignalResult(
            signal="HOLD",
            confidence=0.0,
        )

    assert rsi_14 is not None
    assert macd is not None
    assert macd_signal is not None
    assert sma_20 is not None
    assert ema_20 is not None

    if close <= 0:
        return SignalResult(
            signal="HOLD",
            confidence=0.0,
        )

    # ---------------------------------------------------------
    # 1. TREND SCORE
    # ---------------------------------------------------------
    #
    # Price above both moving averages indicates an upward
    # trend. Price below both indicates a downward trend.
    #
    # We deliberately avoid giving a directional vote when
    # price sits between the two averages.

    trend_score = 0.0

    if close > sma_20:
        trend_score += 1.0
    elif close < sma_20:
        trend_score -= 1.0

    if close > ema_20:
        trend_score += 1.0
    elif close < ema_20:
        trend_score -= 1.0

    # Normalize from [-2, +2] to [-1, +1].
    trend_score /= 2.0

    # ---------------------------------------------------------
    # 2. MACD MOMENTUM
    # ---------------------------------------------------------
    #
    # MACD above its signal line is bullish.
    # MACD below its signal line is bearish.
    #
    # The distance between MACD and signal is also considered.
    # This prevents every tiny numerical difference from being
    # treated as equally meaningful.

    macd_difference = macd - macd_signal

    if macd_difference > 0:
        macd_score = 1.0
    elif macd_difference < 0:
        macd_score = -1.0
    else:
        macd_score = 0.0

    # ---------------------------------------------------------
    # 3. RSI REGIME
    # ---------------------------------------------------------
    #
    # RSI is interpreted differently depending on its regime.
    #
    # Very low RSI:
    #     potential bullish mean-reversion evidence.
    #
    # Very high RSI:
    #     potential bearish mean-reversion evidence.
    #
    # Moderate bullish/bearish momentum is treated more gently
    # so that a strong trend is not automatically rejected simply
    # because RSI is elevated.

    if rsi_14 < 25:
        rsi_score = 1.0
    elif rsi_14 < 35:
        rsi_score = 0.5
    elif rsi_14 > 75:
        rsi_score = -1.0
    elif rsi_14 > 65:
        rsi_score = -0.5
    else:
        rsi_score = 0.0

    # ---------------------------------------------------------
    # 4. COMBINED MOMENTUM
    # ---------------------------------------------------------

    momentum_score = (
        macd_score * 0.70
        + rsi_score * 0.30
    )

    # ---------------------------------------------------------
    # 5. FINAL DIRECTIONAL SCORE
    # ---------------------------------------------------------
    #
    # Trend receives slightly greater weight because the current
    # engine is intended to identify directional opportunities.
    #
    # Momentum provides confirmation.

    directional_score = (
        trend_score * 0.60
        + momentum_score * 0.40
    )

    # ---------------------------------------------------------
    # 6. TREND/MOMENTUM CONFIRMATION
    # ---------------------------------------------------------
    #
    # Agreement between trend and momentum increases confidence.
    # Strong disagreement suppresses the signal.

    agreement = trend_score * momentum_score

    if agreement > 0:
        confirmation_bonus = min(
            abs(agreement) * 0.20,
            0.20,
        )
    elif agreement < 0:
        confirmation_bonus = -min(
            abs(agreement) * 0.20,
            0.20,
        )
    else:
        confirmation_bonus = 0.0

    directional_score += confirmation_bonus

    directional_score = _clamp(
        directional_score,
        minimum=-1.0,
        maximum=1.0,
    )

    # ---------------------------------------------------------
    # 7. SIGNAL THRESHOLD
    # ---------------------------------------------------------
    #
    # We intentionally require stronger evidence than simply
    # having one indicator point in a direction.

    signal_threshold = 0.35

    if directional_score >= signal_threshold:
        signal = "BUY"

    elif directional_score <= -signal_threshold:
        signal = "SELL"

    else:
        signal = "HOLD"

    # ---------------------------------------------------------
    # 8. CONFIDENCE
    # ---------------------------------------------------------
    #
    # Confidence is based primarily on directional strength.
    # The closer the score is to +/-1, the stronger the evidence.
    #
    # HOLD receives reduced confidence because the evidence is
    # not sufficiently directional.

    raw_confidence = abs(directional_score)

    if signal == "HOLD":
        confidence = raw_confidence * 0.75
    else:
        confidence = raw_confidence

    confidence = _clamp(confidence)

    return SignalResult(
        signal=signal,
        confidence=confidence,
    )