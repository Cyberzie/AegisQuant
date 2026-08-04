from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from app.services.backtest_engine import BacktestTrade


@dataclass(frozen=True)
class ConfidenceBucket:
    label: str
    minimum_confidence: float
    maximum_confidence: float
    trade_count: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_return_percent: float
    total_return_percent: float
    gross_profit_percent: float
    gross_loss_percent: float
    profit_factor: float


@dataclass(frozen=True)
class BacktestEvaluation:
    total_trades: int
    confidence_buckets: tuple[ConfidenceBucket, ...]
    high_confidence_win_rate: float
    low_confidence_win_rate: float
    high_confidence_average_return_percent: float
    low_confidence_average_return_percent: float


_BUCKETS = (
    ("0.00-0.49", 0.00, 0.50),
    ("0.50-0.59", 0.50, 0.60),
    ("0.60-0.69", 0.60, 0.70),
    ("0.70-0.79", 0.70, 0.80),
    ("0.80-0.89", 0.80, 0.90),
    ("0.90-1.00", 0.90, 1.01),
)


def _matches_bucket(
    confidence: float,
    minimum: float,
    maximum: float,
) -> bool:
    return (
        confidence >= minimum
        and confidence < maximum
    )


def _calculate_bucket(
    label: str,
    minimum: float,
    maximum: float,
    trades: Sequence[BacktestTrade],
) -> ConfidenceBucket:
    bucket_trades = [
        trade
        for trade in trades
        if _matches_bucket(
            trade.confidence,
            minimum,
            maximum,
        )
    ]

    trade_count = len(bucket_trades)

    winning_trades = sum(
        1
        for trade in bucket_trades
        if trade.profitable
    )

    losing_trades = (
        trade_count - winning_trades
    )

    if trade_count:
        win_rate = (
            winning_trades / trade_count
        )

        average_return_percent = (
            sum(
                trade.return_percent
                for trade in bucket_trades
            )
            / trade_count
        )

        total_return_percent = sum(
            trade.return_percent
            for trade in bucket_trades
        )
    else:
        win_rate = 0.0
        average_return_percent = 0.0
        total_return_percent = 0.0

    gross_profit_percent = sum(
        trade.return_percent
        for trade in bucket_trades
        if trade.return_percent > 0
    )

    gross_loss_percent = abs(
        sum(
            trade.return_percent
            for trade in bucket_trades
            if trade.return_percent < 0
        )
    )

    if gross_loss_percent > 0:
        profit_factor = gross_profit_percent / gross_loss_percent
    elif gross_profit_percent > 0:
        profit_factor = None
    else:
        profit_factor = 0.0

    return ConfidenceBucket(
        label=label,
        minimum_confidence=minimum,
        maximum_confidence=maximum,
        trade_count=trade_count,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        average_return_percent=(
            average_return_percent
        ),
        total_return_percent=(
            total_return_percent
        ),
        gross_profit_percent=(
            gross_profit_percent
        ),
        gross_loss_percent=(
            gross_loss_percent
        ),
        profit_factor=profit_factor,
    )


def evaluate_backtest(
    trades: Sequence[BacktestTrade],
) -> BacktestEvaluation:
    """
    Evaluate whether signal confidence contains
    useful predictive information.

    Trades are grouped into confidence buckets.

    The function does not modify the original trades
    and does not perform any optimization.
    """

    confidence_buckets = tuple(
        _calculate_bucket(
            label,
            minimum,
            maximum,
            trades,
        )
        for label, minimum, maximum in _BUCKETS
    )

    low_confidence_trades = [
        trade
        for trade in trades
        if trade.confidence < 0.70
    ]

    high_confidence_trades = [
        trade
        for trade in trades
        if trade.confidence >= 0.70
    ]

    if low_confidence_trades:
        low_confidence_win_rate = (
            sum(
                1
                for trade in low_confidence_trades
                if trade.profitable
            )
            / len(low_confidence_trades)
        )

        low_confidence_average_return_percent = (
            sum(
                trade.return_percent
                for trade in low_confidence_trades
            )
            / len(low_confidence_trades)
        )
    else:
        low_confidence_win_rate = 0.0
        low_confidence_average_return_percent = 0.0

    if high_confidence_trades:
        high_confidence_win_rate = (
            sum(
                1
                for trade in high_confidence_trades
                if trade.profitable
            )
            / len(high_confidence_trades)
        )

        high_confidence_average_return_percent = (
            sum(
                trade.return_percent
                for trade in high_confidence_trades
            )
            / len(high_confidence_trades)
        )
    else:
        high_confidence_win_rate = 0.0
        high_confidence_average_return_percent = 0.0

    return BacktestEvaluation(
        total_trades=len(trades),
        confidence_buckets=confidence_buckets,
        high_confidence_win_rate=(
            high_confidence_win_rate
        ),
        low_confidence_win_rate=(
            low_confidence_win_rate
        ),
        high_confidence_average_return_percent=(
            high_confidence_average_return_percent
        ),
        low_confidence_average_return_percent=(
            low_confidence_average_return_percent
        ),
    )