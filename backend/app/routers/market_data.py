from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database.dependencies import get_db
from app.models.instrument import Instrument
from app.models.market_data import MarketData
from app.schemas.market_data import MarketDataCreate, MarketDataResponse

router = APIRouter(
    prefix="/market-data",
)


@router.post(
    "/",
    response_model=MarketDataResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_market_data(
    market_data: MarketDataCreate,
    db: Session = Depends(get_db),
):
    instrument = (
        db.query(Instrument)
        .filter(Instrument.id == market_data.instrument_id)
        .first()
    )

    if instrument is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instrument not found.",
        )

    new_market_data = MarketData(
        instrument_id=market_data.instrument_id,
        timestamp=market_data.timestamp,
        open=market_data.open,
        high=market_data.high,
        low=market_data.low,
        close=market_data.close,
        volume=market_data.volume,
    )

    db.add(new_market_data)
    db.commit()
    db.refresh(new_market_data)

    return new_market_data

@router.get(
    "/",
    response_model=list[MarketDataResponse],

)
def list_market_data(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
  
    return (
        db.query(MarketData)
        .order_by(MarketData.timestamp)
        .offset(skip)
        .limit(limit)
        .all()
    )

@router.get(
    "/instrument/{instrument_id}",
    response_model=list[MarketDataResponse],
)
def list_market_data_by_instrument(
    instrument_id: int,
    start: datetime | None = None,
    end: datetime | None = None,
    db: Session = Depends(get_db),
):
    instrument = (
        db.query(Instrument)
        .filter(Instrument.id == instrument_id)
        .first()
    )

    if instrument is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instrument not found.",
        )

    query = (
        db.query(MarketData)
        .filter(MarketData.instrument_id == instrument_id)
    )

    if start is not None:
        query = query.filter(MarketData.timestamp >= start)

    if end is not None:
        query = query.filter(MarketData.timestamp <= end)

    return (
        query
        .order_by(MarketData.timestamp)
        .all()
    )