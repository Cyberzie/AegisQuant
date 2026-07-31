from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MarketDataCreate(BaseModel):
    instrument_id: int
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


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