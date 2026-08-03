from pydantic import BaseModel


class MLEvaluationResponse(BaseModel):
    symbol: str
    dataset_rows: int
    training_rows: int
    validation_rows: int
    direction_accuracy: float
    average_absolute_error_percent: float


class MLWalkForwardFoldResponse(BaseModel):
    fold_number: int
    training_rows: int
    validation_rows: int
    gap_rows: int
    correct_direction_predictions: int
    incorrect_direction_predictions: int
    direction_accuracy: float
    average_absolute_error_percent: float


class MLWalkForwardEvaluationResponse(BaseModel):
    symbol: str
    horizon: int
    dataset_rows: int
    total_training_rows: int
    total_validation_rows: int
    total_correct_predictions: int
    total_incorrect_predictions: int
    direction_accuracy: float
    average_absolute_error_percent: float
    folds: list[MLWalkForwardFoldResponse]