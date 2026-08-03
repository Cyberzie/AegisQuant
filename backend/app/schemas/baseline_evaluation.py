from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class StrategyEvaluationResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    prediction_count: int
    actionable_predictions: int
    correct_direction_predictions: int
    incorrect_direction_predictions: int
    direction_accuracy: float
    average_return_percent: float
    total_return_percent: float
    average_net_return_percent: float
    total_net_return_percent: float
    winning_predictions: int
    losing_predictions: int
    win_rate: float
    profit_factor: float


class BaselineComparisonResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    symbol: str
    horizon: int
    dataset_rows: int
    total_training_rows: int
    total_validation_rows: int
    folds: int
    gap_rows: int
    strategies: list[StrategyEvaluationResponse]