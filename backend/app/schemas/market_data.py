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
    volume: float | None = None

    model_config = ConfigDict(from_attributes=True)


class MarketDataIngestionRequest(BaseModel):
    symbol: str
    provider: str | None = None
    start: datetime | None = None
    end: datetime | None = None


class MarketDataIngestionResponse(BaseModel):
    received: int
    inserted: int
    duplicates: int
    invalid: int

class MarketDataSummaryResponse(BaseModel):
    symbol: str
    data_points: int
    first_timestamp: datetime
    last_timestamp: datetime
    first_open: float
    latest_close: float
    high: float
    low: float
    change: float
    change_percent: float
