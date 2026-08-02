from __future__ import annotations

from collections.abc import Sequence
from math import sqrt


def _validate_prices(prices: Sequence[float]) -> list[float]:
    values = [float(price) for price in prices]

    if not values:
        raise ValueError("Price series cannot be empty.")

    return values


def sma(prices: Sequence[float], period: int) -> list[float | None]:
    values = _validate_prices(prices)

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    result: list[float | None] = [None] * len(values)

    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        result[index] = sum(window) / period

    return result


def ema(prices: Sequence[float], period: int) -> list[float | None]:
    values = _validate_prices(prices)

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    result: list[float | None] = [None] * len(values)

    if len(values) < period:
        return result

    initial_sma = sum(values[:period]) / period
    result[period - 1] = initial_sma

    multiplier = 2 / (period + 1)

    previous = initial_sma

    for index in range(period, len(values)):
        current = (
            (values[index] - previous) * multiplier
            + previous
        )
        result[index] = current
        previous = current

    return result


def rsi(prices: Sequence[float], period: int = 14) -> list[float | None]:
    values = _validate_prices(prices)

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    result: list[float | None] = [None] * len(values)

    if len(values) <= period:
        return result

    gains: list[float] = []
    losses: list[float] = []

    for index in range(1, len(values)):
        change = values[index] - values[index - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))

    average_gain = sum(gains[:period]) / period
    average_loss = sum(losses[:period]) / period

    def calculate_rsi(gain: float, loss: float) -> float:
        if loss == 0:
            return 100.0

        relative_strength = gain / loss
        return 100 - (100 / (1 + relative_strength))

    result[period] = calculate_rsi(
        average_gain,
        average_loss,
    )

    for index in range(period + 1, len(values)):
        gain = gains[index - 1]
        loss = losses[index - 1]

        average_gain = (
            (average_gain * (period - 1)) + gain
        ) / period

        average_loss = (
            (average_loss * (period - 1)) + loss
        ) / period

        result[index] = calculate_rsi(
            average_gain,
            average_loss,
        )

    return result


def macd(
    prices: Sequence[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, list[float | None]]:
    values = _validate_prices(prices)

    if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
        raise ValueError("MACD periods must be greater than zero.")

    if fast_period >= slow_period:
        raise ValueError(
            "Fast period must be less than slow period."
        )

    fast_ema = ema(values, fast_period)
    slow_ema = ema(values, slow_period)

    macd_line: list[float | None] = [None] * len(values)

    for index in range(len(values)):
        if (
            fast_ema[index] is not None
            and slow_ema[index] is not None
        ):
            macd_line[index] = (
                fast_ema[index] - slow_ema[index]
            )

    valid_macd = [
        value for value in macd_line
        if value is not None
    ]

    signal_line: list[float | None] = [None] * len(values)

    if valid_macd:
        signal_values = ema(valid_macd, signal_period)

        first_valid_index = next(
            (
                index
                for index, value in enumerate(macd_line)
                if value is not None
            ),
            None,
        )

        if first_valid_index is not None:
            for offset, value in enumerate(signal_values):
                target_index = first_valid_index + offset

                if target_index < len(signal_line):
                    signal_line[target_index] = value

    histogram: list[float | None] = [None] * len(values)

    for index in range(len(values)):
        if (
            macd_line[index] is not None
            and signal_line[index] is not None
        ):
            histogram[index] = (
                macd_line[index] - signal_line[index]
            )

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


def bollinger_bands(
    prices: Sequence[float],
    period: int = 20,
    deviations: float = 2.0,
) -> dict[str, list[float | None]]:
    values = _validate_prices(prices)

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    if deviations < 0:
        raise ValueError("Deviations cannot be negative.")

    middle = sma(values, period)

    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)

    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        mean = middle[index]

        if mean is None:
            continue

        variance = sum(
            (value - mean) ** 2
            for value in window
        ) / period

        standard_deviation = sqrt(variance)

        upper[index] = (
            mean + deviations * standard_deviation
        )
        lower[index] = (
            mean - deviations * standard_deviation
        )

    return {
        "middle": middle,
        "upper": upper,
        "lower": lower,
    }


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    high_values = _validate_prices(highs)
    low_values = _validate_prices(lows)
    close_values = _validate_prices(closes)

    if not (
        len(high_values)
        == len(low_values)
        == len(close_values)
    ):
        raise ValueError(
            "High, low, and close series must have equal lengths."
        )

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    result: list[float | None] = [None] * len(close_values)

    if len(close_values) <= period:
        return result

    true_ranges: list[float] = [
        high_values[0] - low_values[0]
    ]

    for index in range(1, len(close_values)):
        true_range = max(
            high_values[index] - low_values[index],
            abs(
                high_values[index]
                - close_values[index - 1]
            ),
            abs(
                low_values[index]
                - close_values[index - 1]
            ),
        )

        true_ranges.append(true_range)

    average_true_range = (
        sum(true_ranges[1 : period + 1]) / period
    )

    result[period] = average_true_range

    for index in range(period + 1, len(close_values)):
        average_true_range = (
            (
                average_true_range * (period - 1)
                + true_ranges[index]
            )
            / period
        )

        result[index] = average_true_range

    return result
