from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.models.market_data import MarketData
from app.services.baseline_evaluation import (
    BaselineComparisonResult,
    compare_walk_forward_baselines,
)


def _make_rows(count: int = 120) -> list[MarketData]:
    start = datetime(2025, 1, 1)

    rows: list[MarketData] = []
    price = 100.0

    for index in range(count):
        price += (
            0.35
            if index % 7 != 0
            else -0.20
        )

        rows.append(
            MarketData(
                timestamp=start + timedelta(
                    minutes=index
                ),
                open=price,
                high=price + 0.50,
                low=price - 0.50,
                close=price,
                volume=1000.0 + index,
            )
        )

    return rows


def test_compare_walk_forward_baselines_returns_result():
    result = compare_walk_forward_baselines(
        _make_rows(),
        symbol="TEST",
        horizon=5,
        folds=3,
    )

    assert isinstance(
        result,
        BaselineComparisonResult,
    )

    assert result.symbol == "TEST"
    assert result.horizon == 5
    assert result.dataset_rows > 0
    assert result.total_training_rows > 0
    assert result.total_validation_rows > 0
    assert 1 <= result.folds <= 3

    assert len(result.strategies) == 6


def test_all_strategies_are_present():
    result = compare_walk_forward_baselines(
        _make_rows(),
        symbol="TEST",
        horizon=5,
        folds=3,
    )

    names = {
        strategy.name
        for strategy in result.strategies
    }

    assert names == {
        "always_buy",
        "always_sell",
        "rule",
        "ml",
        "ensemble",
        "adaptive_ensemble",
    }


def test_strategy_counts_are_consistent():
    result = compare_walk_forward_baselines(
        _make_rows(),
        symbol="TEST",
        horizon=5,
        folds=3,
    )

    for strategy in result.strategies:
        assert (
            strategy.actionable_predictions
            <= strategy.prediction_count
        )

        assert (
            strategy.correct_direction_predictions
            + strategy.incorrect_direction_predictions
            == strategy.actionable_predictions
        )

        assert (
            strategy.winning_predictions
            + strategy.losing_predictions
            == strategy.actionable_predictions
        )

        assert 0.0 <= strategy.direction_accuracy <= 1.0
        assert 0.0 <= strategy.win_rate <= 1.0


def test_adaptive_ensemble_has_predictions():
    result = compare_walk_forward_baselines(
        _make_rows(),
        symbol="TEST",
        horizon=5,
        folds=3,
    )

    adaptive = next(
        strategy
        for strategy in result.strategies
        if strategy.name == "adaptive_ensemble"
    )

    assert adaptive.prediction_count > 0
    assert adaptive.actionable_predictions > 0


def test_invalid_horizon_is_rejected():
    with pytest.raises(
        ValueError,
        match="Horizon must be greater than zero",
    ):
        compare_walk_forward_baselines(
            _make_rows(),
            symbol="TEST",
            horizon=0,
        )


def test_invalid_fold_count_is_rejected():
    with pytest.raises(
        ValueError,
        match="Number of folds must be greater than zero",
    ):
        compare_walk_forward_baselines(
            _make_rows(),
            symbol="TEST",
            folds=0,
        )


def test_insufficient_dataset_is_rejected():
    with pytest.raises(
        ValueError,
        match="Dataset must contain at least 20 rows",
    ):
        compare_walk_forward_baselines(
            _make_rows(25),
            symbol="TEST",
            horizon=5,
        )