from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.models.market_data import MarketData
from app.services.technical_indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)


@dataclass(frozen=True)
class MarketFeature:
    timestamp: object
    close: float
    volume: float | None

    return_1: float | None
    return_5: float | None
    volatility_10: float | None

    sma_20: float | None
    ema_20: float | None
    rsi_14: float | None

    macd: float | None
    macd_signal: float | None
    macd_histogram: float | None

    bollinger_middle: float | None
    bollinger_upper: float | None
    bollinger_lower: float | None

    atr_14: float | None


def _percentage_change(
    current: float,
    previous: float,
) -> float | None:
    if previous == 0:
        return None

    return ((current - previous) / previous) * 100


def _rolling_volatility(
    closes: Sequence[float],
    index: int,
    period: int,
) -> float | None:
    if index < period:
        return None

    window = [
        float(closes[position])
        for position in range(
            index - period + 1,
            index + 1,
        )
    ]

    returns: list[float] = []

    for position in range(1, len(window)):
        previous = window[position - 1]

        if previous == 0:
            continue

        returns.append(
            (window[position] - previous)
            / previous
        )

    if len(returns) < 2:
        return None

    mean = sum(returns) / len(returns)

    variance = sum(
        (value - mean) ** 2
        for value in returns
    ) / len(returns)

    return variance ** 0.5


def build_market_features(
    rows: Sequence[MarketData],
) -> list[MarketFeature]:
    ordered_rows = sorted(
        rows,
        key=lambda row: row.timestamp,
    )

    if not ordered_rows:
        return []

    closes = [
        float(row.close)
        for row in ordered_rows
    ]

    highs = [
        float(row.high)
        for row in ordered_rows
    ]

    lows = [
        float(row.low)
        for row in ordered_rows
    ]

    sma_values = sma(closes, 20)
    ema_values = ema(closes, 20)
    rsi_values = rsi(closes, 14)
    macd_values = macd(closes)
    bollinger_values = bollinger_bands(closes)
    atr_values = atr(
        highs,
        lows,
        closes,
        14,
    )

    features: list[MarketFeature] = []

    for index, row in enumerate(ordered_rows):
        return_1 = None

        if index >= 1:
            return_1 = _percentage_change(
                closes[index],
                closes[index - 1],
            )

        return_5 = None

        if index >= 5:
            return_5 = _percentage_change(
                closes[index],
                closes[index - 5],
            )

        features.append(
            MarketFeature(
                timestamp=row.timestamp,
                close=closes[index],
                volume=(
                    float(row.volume)
                    if row.volume is not None
                    else None
                ),
                return_1=return_1,
                return_5=return_5,
                volatility_10=_rolling_volatility(
                    closes,
                    index,
                    10,
                ),
                sma_20=sma_values[index],
                ema_20=ema_values[index],
                rsi_14=rsi_values[index],
                macd=macd_values["macd"][index],
                macd_signal=macd_values["signal"][index],
                macd_histogram=macd_values["histogram"][index],
                bollinger_middle=(
                    bollinger_values["middle"][index]
                ),
                bollinger_upper=(
                    bollinger_values["upper"][index]
                ),
                bollinger_lower=(
                    bollinger_values["lower"][index]
                ),
                atr_14=atr_values[index],
            )
        )

    return features