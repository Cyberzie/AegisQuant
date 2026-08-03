from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, tanh
from typing import Sequence

from app.services.ml_dataset import MLDataset, MLDatasetRow


@dataclass(frozen=True)
class MLPrediction:
    direction: str
    confidence: float
    expected_return_percent: float


@dataclass(frozen=True)
class MLModel:
    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    bias: float
    mean_values: tuple[float, ...]
    scale_values: tuple[float, ...]

    hidden_weights: tuple[tuple[float, ...], ...] = ()
    hidden_biases: tuple[float, ...] = ()
    output_weights: tuple[float, ...] = ()
    output_bias: float = 0.0


# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

_MIN_SCALE = 1e-12
_MAX_STANDARDIZED_VALUE = 8.0
_MAX_GRADIENT = 5.0
_MAX_PREDICTION = 1.0

# The target is expressed as a normalized return:
#
#     normalized target = return_percent / 10
#
# Therefore a model prediction of 1.0 corresponds to +10%.
_TARGET_SCALE_PERCENT = 10.0

# L2 regularization reduces excessive weights when the training sample
# is small relative to the number of features.
_DEFAULT_L2 = 1e-4

# Confidence is deliberately conservative.  Confidence is treated as
# model conviction, not as a statistically calibrated probability.
_CONFIDENCE_SCALE = 3.0

# Predictions below this magnitude are treated as effectively neutral.
_HOLD_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def _safe_scale(value: float) -> float:
    if not isfinite(value):
        return 1.0

    if abs(value) < _MIN_SCALE:
        return 1.0

    return value


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _sigmoid(value: float) -> float:
    """
    Numerically stable sigmoid.

    Returns a value in (0, 1).
    """
    if value >= 0:
        z = exp(-value)
        return 1.0 / (1.0 + z)

    z = exp(value)
    return z / (1.0 + z)


def _standardize(
    values: Sequence[float],
    means: Sequence[float],
    scales: Sequence[float],
) -> list[float]:
    standardized: list[float] = []

    for value, mean, scale in zip(
        values,
        means,
        scales,
    ):
        standardized_value = (
            float(value) - float(mean)
        ) / _safe_scale(float(scale))

        standardized.append(
            _clip(
                standardized_value,
                -_MAX_STANDARDIZED_VALUE,
                _MAX_STANDARDIZED_VALUE,
            )
        )

    return standardized


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _dataset_matrix(
    dataset: MLDataset,
) -> tuple[list[list[float]], list[float]]:
    matrix: list[list[float]] = []
    targets: list[float] = []

    for row in dataset.rows:
        values = [
            float(row.features[name])
            for name in dataset.feature_names
        ]

        if not all(isfinite(value) for value in values):
            continue

        target = float(row.target_return)

        if not isfinite(target):
            continue

        matrix.append(values)
        targets.append(target)

    return matrix, targets


def _calculate_statistics(
    matrix: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    if not matrix:
        raise ValueError(
            "Training dataset cannot be empty."
        )

    feature_count = len(matrix[0])

    if feature_count == 0:
        raise ValueError(
            "Training dataset must contain features."
        )

    means: list[float] = []
    scales: list[float] = []

    for column in range(feature_count):
        values = [
            float(row[column])
            for row in matrix
        ]

        mean = sum(values) / len(values)

        variance = (
            sum(
                (value - mean) ** 2
                for value in values
            )
            / len(values)
        )

        scale = variance ** 0.5

        means.append(mean)
        scales.append(
            _safe_scale(scale)
        )

    return means, scales


# ---------------------------------------------------------------------------
# Neural-network initialization
# ---------------------------------------------------------------------------

def _initial_hidden_weights(
    feature_count: int,
    hidden_size: int,
) -> list[list[float]]:
    """
    Deterministic Xavier-style initialization.

    The initialization is deterministic so that repeated backtests are
    reproducible and do not introduce uncontrolled randomness.
    """
    if feature_count <= 0:
        raise ValueError(
            "Feature count must be greater than zero."
        )

    if hidden_size <= 0:
        raise ValueError(
            "Hidden layer size must be greater than zero."
        )

    scale = (
        2.0
        / (feature_count + hidden_size)
    ) ** 0.5

    weights: list[list[float]] = []

    for hidden_index in range(hidden_size):
        neuron_weights: list[float] = []

        for feature_index in range(feature_count):
            pattern = (
                (
                    (hidden_index + 1) * 17
                    + (feature_index + 3) * 31
                    + hidden_index * feature_index * 7
                )
                % 101
            )

            centered = (
                pattern - 50
            ) / 50.0

            neuron_weights.append(
                scale * centered
            )

        weights.append(
            neuron_weights
        )

    return weights


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def _forward_network(
    values: Sequence[float],
    hidden_weights: Sequence[Sequence[float]],
    hidden_biases: Sequence[float],
    output_weights: Sequence[float],
    output_bias: float,
) -> tuple[list[float], float]:
    hidden: list[float] = []

    for weights, bias in zip(
        hidden_weights,
        hidden_biases,
    ):
        activation = float(bias)

        for weight, value in zip(
            weights,
            values,
        ):
            activation += (
                float(weight)
                * float(value)
            )

        hidden.append(
            tanh(activation)
        )

    output = float(output_bias)

    for weight, value in zip(
        output_weights,
        hidden,
    ):
        output += (
            float(weight)
            * float(value)
        )

    return hidden, output


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def _clip_gradient(value: float) -> float:
    return _clip(
        float(value),
        -_MAX_GRADIENT,
        _MAX_GRADIENT,
    )


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_ml_model(
    dataset: MLDataset,
    *,
    learning_rate: float = 0.01,
    epochs: int = 500,
    l2_regularization: float = _DEFAULT_L2,
) -> MLModel:
    """
    Train the AegisQuant neural ML model.

    The model uses:

    - feature standardization
    - one tanh hidden layer
    - linear output
    - gradient clipping
    - L2 regularization
    - bounded training targets

    The target remains continuous future return rather than a direct
    classification label so that the model can estimate both direction
    and expected magnitude.
    """
    if not dataset.rows:
        raise ValueError(
            "Training dataset cannot be empty."
        )

    if learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than zero."
        )

    if epochs <= 0:
        raise ValueError(
            "Epochs must be greater than zero."
        )

    if l2_regularization < 0:
        raise ValueError(
            "L2 regularization cannot be negative."
        )

    matrix, targets = _dataset_matrix(
        dataset
    )

    if not matrix:
        raise ValueError(
            "Training dataset contains no usable rows."
        )

    means, scales = _calculate_statistics(
        matrix
    )

    standardized = [
        _standardize(
            row,
            means,
            scales,
        )
        for row in matrix
    ]

    feature_count = len(
        dataset.feature_names
    )

    # Keep the network deliberately small.
    #
    # AegisQuant's current market-data samples are not large enough to
    # justify a very deep network.  A compact network reduces overfitting
    # and keeps repeated walk-forward training practical.
    hidden_size = min(
        16,
        max(6, feature_count),
    )

    hidden_weights = (
        _initial_hidden_weights(
            feature_count,
            hidden_size,
        )
    )

    hidden_biases = [
        0.0
        for _ in range(hidden_size)
    ]

    output_weights = [
        0.0
        for _ in range(hidden_size)
    ]

    output_bias = 0.0

    # Convert percentage returns to a bounded training target.
    target_values = [
        _clip(
            float(target)
            / _TARGET_SCALE_PERCENT,
            -_MAX_PREDICTION,
            _MAX_PREDICTION,
        )
        for target in targets
    ]

    sample_count = len(
        standardized
    )

    for _ in range(epochs):
        hidden_weight_gradients = [
            [0.0] * feature_count
            for _ in range(hidden_size)
        ]

        hidden_bias_gradients = [
            0.0
            for _ in range(hidden_size)
        ]

        output_weight_gradients = [
            0.0
            for _ in range(hidden_size)
        ]

        output_bias_gradient = 0.0

        for values, target in zip(
            standardized,
            target_values,
        ):
            hidden, raw_prediction = (
                _forward_network(
                    values,
                    hidden_weights,
                    hidden_biases,
                    output_weights,
                    output_bias,
                )
            )

            # Bound the training prediction before calculating the
            # residual. This prevents a single pathological observation
            # from dominating the optimization.
            prediction = _clip(
                raw_prediction,
                -_MAX_PREDICTION,
                _MAX_PREDICTION,
            )

            error = prediction - target

            output_bias_gradient += error

            for hidden_index, hidden_value in enumerate(
                hidden
            ):
                output_weight_gradients[
                    hidden_index
                ] += (
                    error
                    * hidden_value
                )

            for hidden_index in range(
                hidden_size
            ):
                hidden_value = (
                    hidden[hidden_index]
                )

                hidden_error = (
                    error
                    * output_weights[
                        hidden_index
                    ]
                    * (
                        1.0
                        - hidden_value ** 2
                    )
                )

                hidden_bias_gradients[
                    hidden_index
                ] += hidden_error

                for feature_index, value in enumerate(
                    values
                ):
                    hidden_weight_gradients[
                        hidden_index
                    ][feature_index] += (
                        hidden_error
                        * value
                    )

        step = (
            learning_rate
            / sample_count
        )

        # Hidden layer updates.
        for hidden_index in range(
            hidden_size
        ):
            for feature_index in range(
                feature_count
            ):
                gradient = (
                    hidden_weight_gradients[
                        hidden_index
                    ][feature_index]
                    / sample_count
                )

                gradient += (
                    l2_regularization
                    * hidden_weights[
                        hidden_index
                    ][feature_index]
                )

                gradient = _clip_gradient(
                    gradient
                )

                hidden_weights[
                    hidden_index
                ][feature_index] -= (
                    learning_rate
                    * gradient
                )

            hidden_bias_gradient = (
                hidden_bias_gradients[
                    hidden_index
                ]
                / sample_count
            )

            hidden_biases[
                hidden_index
            ] -= (
                learning_rate
                * _clip_gradient(
                    hidden_bias_gradient
                )
            )

            output_gradient = (
                output_weight_gradients[
                    hidden_index
                ]
                / sample_count
            )

            output_gradient += (
                l2_regularization
                * output_weights[
                    hidden_index
                ]
            )

            output_gradient = (
                _clip_gradient(
                    output_gradient
                )
            )

            output_weights[
                hidden_index
            ] -= (
                learning_rate
                * output_gradient
            )

        output_bias_gradient /= (
            sample_count
        )

        output_bias -= (
            learning_rate
            * _clip_gradient(
                output_bias_gradient
            )
        )

        # Keep the unused compatibility variables meaningful.
        #
        # `step` is retained here intentionally because it documents the
        # normalized batch-learning rate used by the training procedure.
        _ = step

    return MLModel(
        feature_names=dataset.feature_names,
        weights=tuple(output_weights),
        bias=output_bias,
        mean_values=tuple(means),
        scale_values=tuple(scales),
        hidden_weights=tuple(
            tuple(weights)
            for weights in hidden_weights
        ),
        hidden_biases=tuple(
            hidden_biases
        ),
        output_weights=tuple(
            output_weights
        ),
        output_bias=output_bias,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _calculate_confidence(
    prediction: float,
) -> float:
    """
    Convert model conviction into a conservative confidence score.

    This is intentionally NOT presented as a calibrated probability.

    A zero prediction produces 0.50 confidence.
    Increasing prediction magnitude gradually increases confidence.
    """
    magnitude = abs(
        float(prediction)
    )

    if magnitude < _HOLD_THRESHOLD:
        return 0.50

    conviction = (
        2.0
        * _sigmoid(
            magnitude
            * _CONFIDENCE_SCALE
        )
        - 1.0
    )

    confidence = (
        0.50
        + 0.50
        * conviction
    )

    return _clip(
        confidence,
        0.50,
        0.99,
    )


def predict_ml_model(
    model: MLModel,
    row: MLDatasetRow,
) -> MLPrediction:
    """
    Generate a prediction from a trained MLModel.

    Returns:

    - BUY / SELL / HOLD
    - conservative model confidence
    - expected return percentage
    """
    values = [
        float(
            row.features[name]
        )
        for name in model.feature_names
    ]

    standardized = _standardize(
        values,
        model.mean_values,
        model.scale_values,
    )

    # Preferred path: neural network.
    if (
        model.hidden_weights
        and model.hidden_biases
        and model.output_weights
    ):
        _, raw_prediction = (
            _forward_network(
                standardized,
                model.hidden_weights,
                model.hidden_biases,
                model.output_weights,
                model.output_bias,
            )
        )

    # Compatibility path for older serialized models.
    else:
        raw_prediction = model.bias

        for weight, value in zip(
            model.weights,
            standardized,
        ):
            raw_prediction += (
                float(weight)
                * float(value)
            )

    prediction = _clip(
        raw_prediction,
        -_MAX_PREDICTION,
        _MAX_PREDICTION,
    )

    expected_return_percent = (
        prediction
        * _TARGET_SCALE_PERCENT
    )

    if (
        abs(prediction)
        < _HOLD_THRESHOLD
    ):
        direction = "HOLD"
    elif prediction > 0:
        direction = "BUY"
    else:
        direction = "SELL"

    confidence = (
        _calculate_confidence(
            prediction
        )
    )

    return MLPrediction(
        direction=direction,
        confidence=confidence,
        expected_return_percent=(
            expected_return_percent
        ),
    )