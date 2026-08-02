from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database.dependencies import get_db
from app.models.instrument import Instrument
from app.models.market_data import MarketData
from app.providers.factory import get_market_data_provider
from app.schemas.market_data import (
    MarketDataCreate,
    MarketDataIngestionRequest,
    MarketDataIngestionResponse,
    MarketDataResponse,
    MarketDataSummaryResponse,
)
from app.services.market_data_query import (
    get_latest_market_data,
    get_market_data_by_symbol,
    get_market_data_summary,
)

from app.schemas.technical_indicators import TechnicalIndicatorsResponse
from app.services.technical_indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)

from app.services.provider_ingestion import ingest_from_provider

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
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
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
        .offset(skip)
        .limit(limit)
        .all()
    )


@router.get(
    "/symbol/{symbol}",
    response_model=list[MarketDataResponse],
)
def list_market_data_by_symbol(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    instrument = (
        db.query(Instrument)
        .filter(Instrument.symbol == symbol.upper())
        .first()
    )

    if instrument is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Instrument with symbol '{symbol}' not found.",
        )

    return get_market_data_by_symbol(
        db=db,
        symbol=symbol,
        start=start,
        end=end,
        skip=skip,
        limit=limit,
    )


@router.get(
    "/latest/{symbol}",
    response_model=list[MarketDataResponse],
)
def list_latest_market_data(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    try:
        return get_latest_market_data(
            db=db,
            symbol=symbol,
            limit=limit,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/summary/{symbol}",
    response_model=MarketDataSummaryResponse,
)
def market_data_summary(
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    db: Session = Depends(get_db),
):
    try:
        return get_market_data_summary(
            db=db,
            symbol=symbol,
            start=start,
            end=end,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

@router.get(
    "/indicators/{symbol}",
    response_model=TechnicalIndicatorsResponse,
)
def market_data_indicators(
    symbol: str,
    db: Session = Depends(get_db),
):
    rows = get_market_data_by_symbol(
        db=db,
        symbol=symbol,
        skip=0,
        limit=1000,
    )

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No market data found for instrument '{symbol}'.",
        )

    closes = [row.close for row in rows]
    highs = [row.high for row in rows]
    lows = [row.low for row in rows]

    sma_values = sma(closes, 20)
    ema_values = ema(closes, 20)
    rsi_values = rsi(closes, 14)

    macd_values = macd(closes)

    bollinger_values = bollinger_bands(closes)

    atr_values = atr(
        highs,
        lows,
        closes,
        14,
    )

    last_index = len(rows) - 1
    latest_row = rows[last_index]

    return TechnicalIndicatorsResponse(
        symbol=symbol.upper(),
        timestamp=latest_row.timestamp,
        close=latest_row.close,

        sma_20=sma_values[last_index],
        ema_20=ema_values[last_index],
        rsi_14=rsi_values[last_index],

        macd=macd_values["macd"][last_index],
        macd_signal=macd_values["signal"][last_index],
        macd_histogram=macd_values["histogram"][last_index],

        bollinger_middle=bollinger_values["middle"][last_index],
        bollinger_upper=bollinger_values["upper"][last_index],
        bollinger_lower=bollinger_values["lower"][last_index],

        atr_14=atr_values[last_index],
    )

@router.post(
    "/ingest",
    response_model=MarketDataIngestionResponse,
    status_code=status.HTTP_200_OK,
)
def ingest_market_data_from_provider(
    request: MarketDataIngestionRequest,
    db: Session = Depends(get_db),
):
    try:
        provider = get_market_data_provider(request.provider)

        result = ingest_from_provider(
            db=db,
            provider=provider,
            symbol=request.symbol,
            start=request.start,
            end=request.end,
        )

        return MarketDataIngestionResponse(
            received=result.received,
            inserted=result.inserted,
            duplicates=result.duplicates,
            invalid=result.invalid,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
