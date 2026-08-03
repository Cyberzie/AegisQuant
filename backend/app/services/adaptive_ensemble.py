from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import tanh


@dataclass(frozen=True)
class PerformanceObservation:
    predicted_signal: str
    actual_direction: str
    realized_return_percent: float
    correct: bool


@dataclass(frozen=True)
class AdaptiveWeights:
    rule_weight: float
    ml_weight: float
    rule_score: float
    ml_score: float


class AdaptiveEnsembleState:
    def __init__(
        self,
        *,
        window: int = 50,
        base_rule_weight: float = 0.4,
        base_ml_weight: float = 0.6,
        minimum_weight: float = 0.20,
        maximum_weight: float = 0.80,
        minimum_observations: int = 5,
        recency_decay: float = 0.97,
    ) -> None:
        if window <= 0:
            raise ValueError("Window must be greater than zero.")

        if base_rule_weight < 0 or base_ml_weight < 0:
            raise ValueError("Base weights cannot be negative.")

        if base_rule_weight + base_ml_weight <= 0:
            raise ValueError(
                "At least one base weight must be positive."
            )

        if not 0 <= minimum_weight <= maximum_weight <= 1:
            raise ValueError(
                "Weight bounds must be between 0 and 1."
            )

        if minimum_observations < 0:
            raise ValueError(
                "Minimum observations cannot be negative."
            )

        if not 0 < recency_decay <= 1:
            raise ValueError(
                "Recency decay must be in the range (0, 1]."
            )

        self.window = window
        self.base_rule_weight = base_rule_weight
        self.base_ml_weight = base_ml_weight
        self.minimum_weight = minimum_weight
        self.maximum_weight = maximum_weight
        self.minimum_observations = minimum_observations
        self.recency_decay = recency_decay

        self._rule_observations = deque(
            maxlen=window
        )
        self._ml_observations = deque(
            maxlen=window
        )

    @staticmethod
    def _direction_correct(
        predicted: str,
        actual: str,
    ) -> bool:
        return predicted == actual

    def observe(
        self,
        *,
        rule_signal: str,
        ml_signal: str,
        actual_direction: str,
        realized_return_percent: float,
    ) -> None:
        if actual_direction not in {
            "BUY",
            "SELL",
            "HOLD",
        }:
            raise ValueError(
                "Actual direction must be BUY, SELL, or HOLD."
            )

        self._rule_observations.append(
            PerformanceObservation(
                predicted_signal=rule_signal,
                actual_direction=actual_direction,
                realized_return_percent=realized_return_percent,
                correct=self._direction_correct(
                    rule_signal,
                    actual_direction,
                ),
            )
        )

        self._ml_observations.append(
            PerformanceObservation(
                predicted_signal=ml_signal,
                actual_direction=actual_direction,
                realized_return_percent=realized_return_percent,
                correct=self._direction_correct(
                    ml_signal,
                    actual_direction,
                ),
            )
        )

    def _score(self, observations) -> float:
        if len(observations) < self.minimum_observations:
            return 1.0

        weighted_correct = 0.0
        weighted_return = 0.0
        weight_total = 0.0

        newest_index = len(observations) - 1

        for index, observation in enumerate(observations):
            age = newest_index - index
            weight = self.recency_decay ** age

            weight_total += weight

            weighted_correct += (
                weight
                * float(observation.correct)
            )

            weighted_return += (
                weight
                * tanh(
                    observation.realized_return_percent / 2.0
                )
            )

        accuracy = (
            weighted_correct / weight_total
        )

        return_component = (
            weighted_return / weight_total + 1.0
        ) / 2.0

        score = (
            accuracy * 0.65
            + return_component * 0.35
        )

        return max(
            0.05,
            min(score, 1.0),
        )

    @staticmethod
    def _normalize(
        rule_weight: float,
        ml_weight: float,
    ) -> tuple[float, float]:
        total = rule_weight + ml_weight

        if total <= 0:
            return 0.5, 0.5

        return (
            rule_weight / total,
            ml_weight / total,
        )

    def weights(self) -> AdaptiveWeights:
        rule_score = self._score(
            self._rule_observations
        )

        ml_score = self._score(
            self._ml_observations
        )

        rule_weight = (
            self.base_rule_weight
            * rule_score
        )

        ml_weight = (
            self.base_ml_weight
            * ml_score
        )

        rule_weight, ml_weight = self._normalize(
            rule_weight,
            ml_weight,
        )

        rule_weight = max(
            self.minimum_weight,
            min(
                rule_weight,
                self.maximum_weight,
            ),
        )

        ml_weight = max(
            self.minimum_weight,
            min(
                ml_weight,
                self.maximum_weight,
            ),
        )

        rule_weight, ml_weight = self._normalize(
            rule_weight,
            ml_weight,
        )

        return AdaptiveWeights(
            rule_weight=rule_weight,
            ml_weight=ml_weight,
            rule_score=rule_score,
            ml_score=ml_score,
        )

    @property
    def observation_count(self) -> int:
        return min(
            len(self._rule_observations),
            len(self._ml_observations),
        )