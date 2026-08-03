from __future__ import annotations

from dataclasses import dataclass

from app.services.adaptive_ensemble import AdaptiveWeights
from app.services.ml_model import MLPrediction
from app.services.signal_engine import SignalResult


@dataclass(frozen=True)
class EnsembleSignal:
    signal: str
    confidence: float
    rule_signal: str
    rule_confidence: float
    ml_signal: str
    ml_confidence: float
    ml_expected_return_percent: float
    rule_weight: float
    ml_weight: float


def combine_signals(
    rule_signal: SignalResult,
    ml_prediction: MLPrediction,
    *,
    adaptive_weights: AdaptiveWeights | None = None,
    rule_weight: float = 0.4,
    ml_weight: float = 0.6,
) -> EnsembleSignal:
    if adaptive_weights is not None:
        rule_weight = adaptive_weights.rule_weight
        ml_weight = adaptive_weights.ml_weight

    if rule_weight < 0:
        raise ValueError(
            "Rule weight cannot be negative."
        )

    if ml_weight < 0:
        raise ValueError(
            "ML weight cannot be negative."
        )

    total_weight = rule_weight + ml_weight

    if total_weight <= 0:
        raise ValueError(
            "At least one signal weight must be positive."
        )

    normalized_rule_weight = (
        rule_weight / total_weight
    )

    normalized_ml_weight = (
        ml_weight / total_weight
    )

    scores = {
        "BUY": 0.0,
        "SELL": 0.0,
        "HOLD": 0.0,
    }

    if rule_signal.signal not in scores:
        raise ValueError(
            f"Invalid rule signal: {rule_signal.signal}"
        )

    if ml_prediction.direction not in scores:
        raise ValueError(
            f"Invalid ML signal: {ml_prediction.direction}"
        )

    scores[rule_signal.signal] += (
        normalized_rule_weight
        * rule_signal.confidence
    )

    scores[ml_prediction.direction] += (
        normalized_ml_weight
        * ml_prediction.confidence
    )

    winning_signal = max(
        scores,
        key=scores.get,
    )

    confidence = min(
        max(scores[winning_signal], 0.0),
        1.0,
    )

    return EnsembleSignal(
        signal=winning_signal,
        confidence=confidence,
        rule_signal=rule_signal.signal,
        rule_confidence=rule_signal.confidence,
        ml_signal=ml_prediction.direction,
        ml_confidence=ml_prediction.confidence,
        ml_expected_return_percent=(
            ml_prediction.expected_return_percent
        ),
        rule_weight=normalized_rule_weight,
        ml_weight=normalized_ml_weight,
    )