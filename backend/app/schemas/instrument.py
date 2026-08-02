from datetime import datetime

from pydantic import BaseModel, ConfigDict


class InstrumentCreate(BaseModel):
    symbol: str
    name: str
    asset_type: str
    exchange: str | None = None
    currency: str | None = None


class InstrumentUpdate(BaseModel):
    name: str | None = None
    asset_type: str | None = None
    exchange: str | None = None
    currency: str | None = None
    is_active: bool | None = None


class InstrumentResponse(BaseModel):
    id: int
    symbol: str
    name: str
    asset_type: str
    exchange: str | None
    currency: str | None
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
