from __future__ import annotations

import pytest

from app.services.adaptive_ensemble import (
    AdaptiveEnsembleState,
)


def test_initial_weights_match_base_weights():
    state = AdaptiveEnsembleState(
        base_rule_weight=0.4,
        base_ml_weight=0.6,
        minimum_observations=5,
    )

    weights = state.weights()

    assert weights.rule_weight == pytest.approx(0.4)
    assert weights.ml_weight == pytest.approx(0.6)

    assert weights.rule_weight + weights.ml_weight == pytest.approx(
        1.0
    )

    assert weights.rule_score == pytest.approx(1.0)
    assert weights.ml_score == pytest.approx(1.0)


def test_weights_remain_within_configured_bounds():
    state = AdaptiveEnsembleState(
        minimum_weight=0.20,
        maximum_weight=0.80,
        minimum_observations=1,
    )

    for _ in range(30):
        state.observe(
            rule_signal="BUY",
            ml_signal="SELL",
            actual_direction="BUY",
            realized_return_percent=2.0,
        )

    weights = state.weights()

    assert 0.20 <= weights.rule_weight <= 0.80
    assert 0.20 <= weights.ml_weight <= 0.80

    assert weights.rule_weight + weights.ml_weight == pytest.approx(
        1.0
    )


def test_successful_rule_predictions_increase_rule_weight():
    state = AdaptiveEnsembleState(
        base_rule_weight=0.4,
        base_ml_weight=0.6,
        minimum_observations=1,
    )

    initial = state.weights()

    for _ in range(10):
        state.observe(
            rule_signal="BUY",
            ml_signal="SELL",
            actual_direction="BUY",
            realized_return_percent=2.0,
        )

    updated = state.weights()

    assert updated.rule_weight > initial.rule_weight
    assert updated.ml_weight < initial.ml_weight


def test_successful_ml_predictions_increase_ml_weight():
    state = AdaptiveEnsembleState(
        base_rule_weight=0.4,
        base_ml_weight=0.6,
        minimum_observations=1,
    )

    initial = state.weights()

    for _ in range(10):
        state.observe(
            rule_signal="SELL",
            ml_signal="BUY",
            actual_direction="BUY",
            realized_return_percent=2.0,
        )

    updated = state.weights()

    assert updated.ml_weight > initial.ml_weight
    assert updated.rule_weight < initial.rule_weight


def test_observation_count_increases_with_observations():
    state = AdaptiveEnsembleState()

    assert state.observation_count == 0

    state.observe(
        rule_signal="BUY",
        ml_signal="BUY",
        actual_direction="BUY",
        realized_return_percent=1.0,
    )

    assert state.observation_count == 1

    state.observe(
        rule_signal="SELL",
        ml_signal="SELL",
        actual_direction="SELL",
        realized_return_percent=1.0,
    )

    assert state.observation_count == 2


def test_observation_window_is_bounded():
    state = AdaptiveEnsembleState(
        window=3,
        minimum_observations=0,
    )

    for _ in range(10):
        state.observe(
            rule_signal="BUY",
            ml_signal="BUY",
            actual_direction="BUY",
            realized_return_percent=1.0,
        )

    assert state.observation_count == 3


def test_invalid_actual_direction_is_rejected():
    state = AdaptiveEnsembleState()

    with pytest.raises(
        ValueError,
        match="Actual direction must be BUY, SELL, or HOLD",
    ):
        state.observe(
            rule_signal="BUY",
            ml_signal="BUY",
            actual_direction="INVALID",
            realized_return_percent=1.0,
        )


def test_invalid_window_is_rejected():
    with pytest.raises(
        ValueError,
        match="Window must be greater than zero",
    ):
        AdaptiveEnsembleState(window=0)


def test_invalid_recency_decay_is_rejected():
    with pytest.raises(
        ValueError,
        match="Recency decay must be in the range",
    ):
        AdaptiveEnsembleState(recency_decay=0)


def test_negative_base_weight_is_rejected():
    with pytest.raises(
        ValueError,
        match="Base weights cannot be negative",
    ):
        AdaptiveEnsembleState(
            base_rule_weight=-0.1,
        )


def test_zero_base_weights_are_rejected():
    with pytest.raises(
        ValueError,
        match="At least one base weight must be positive",
    ):
        AdaptiveEnsembleState(
            base_rule_weight=0.0,
            base_ml_weight=0.0,
        )


def test_weight_bounds_are_validated():
    with pytest.raises(
        ValueError,
        match="Weight bounds must be between 0 and 1",
    ):
        AdaptiveEnsembleState(
            minimum_weight=0.9,
            maximum_weight=0.2,
        )