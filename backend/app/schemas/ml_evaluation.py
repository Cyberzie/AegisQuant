from pydantic import BaseModel


class MLEvaluationResponse(BaseModel):
    symbol: str
    dataset_rows: int
    training_rows: int
    validation_rows: int
    direction_accuracy: float
    average_absolute_error_percent: float