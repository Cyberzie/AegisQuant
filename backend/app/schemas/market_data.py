from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator


class MarketDataCreate(BaseModel):
    instrument_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    @model_validator(mode="after")
    def validate_ohlc(self):
        if self.high < self.low:
            raise ValueError("High price cannot be lower than low price.")

        if not self.low <= self.open <= self.high:
            raise ValueError(
                "Open price must be between low and high prices."
            )

        if not self.low <= self.close <= self.high:
            raise ValueError(
                "Close price must be between low and high prices."
            )

        if self.volume is not None and self.volume < 0:
            raise ValueError("Volume cannot be negative.")

        return self


class MarketDataResponse(BaseModel):
    id: int
    instrument_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None

    model_config = ConfigDict(from_attributes=True)