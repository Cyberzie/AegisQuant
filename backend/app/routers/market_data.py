from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database.dependencies import get_db
from app.models.instrument import Instrument
from app.models.market_data import MarketData
from app.providers.factory import get_market_data_provider
from app.schemas.backtest import BacktestResponse
from app.schemas.market_data import (
    MarketDataCreate,
    MarketDataIngestionRequest,
    MarketDataIngestionResponse,
    MarketDataResponse,
    MarketDataSummaryResponse,
)
from app.schemas.ml_evaluation import MLEvaluationResponse
from app.schemas.technical_indicators import TechnicalIndicatorsResponse
from app.services.backtest_engine import backtest_market_data
from app.services.market_data_query import (
    get_latest_market_data,
    get_market_data_by_symbol,
    get_market_data_summary,
)
from app.services.ml_evaluation import evaluate_symbol
from app.services.provider_ingestion import ingest_from_provider
from app.services.signal_engine import generate_signal
from app.services.technical_indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    rsi,
    sma,
)


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

    signal_result = generate_signal(
        rsi_14=rsi_values[last_index],
        macd=macd_values["macd"][last_index],
        macd_signal=macd_values["signal"][last_index],
        sma_20=sma_values[last_index],
        ema_20=ema_values[last_index],
        close=latest_row.close,
    )

    return TechnicalIndicatorsResponse(
        symbol=symbol.upper(),
        timestamp=latest_row.timestamp,
        close=latest_row.close,
        signal=signal_result.signal,
        confidence=signal_result.confidence,
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


@router.get(
    "/backtest/{symbol}",
    response_model=BacktestResponse,
)
def market_data_backtest(
    symbol: str,
    horizon: int = Query(5, ge=1, le=100),
    starting_capital: float = Query(
        1_000_000.0,
        gt=0,
    ),
    transaction_cost_percent: float = Query(
        0.10,
        ge=0,
    ),
    slippage_percent: float = Query(
        0.05,
        ge=0,
    ),
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
            detail=(
                f"No market data found for "
                f"instrument '{symbol}'."
            ),
        )

    try:
        result = backtest_market_data(
            rows,
            symbol=symbol,
            horizon=horizon,
            starting_capital=starting_capital,
            transaction_cost_percent=(
                transaction_cost_percent
            ),
            slippage_percent=slippage_percent,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return BacktestResponse(
        symbol=result.symbol,
        total_rows=result.total_rows,
        evaluated_rows=result.evaluated_rows,
        actionable_trades=result.actionable_trades,
        winning_trades=result.winning_trades,
        losing_trades=result.losing_trades,
        win_rate=result.win_rate,
        average_return_percent=(
            result.average_return_percent
        ),
        total_return_percent=(
            result.total_return_percent
        ),
        starting_capital=result.starting_capital,
        ending_capital=result.ending_capital,
        net_profit=result.net_profit,
        net_return_percent=result.net_return_percent,
        gross_profit=result.gross_profit,
        gross_loss=result.gross_loss,
        profit_factor=result.profit_factor,
        average_winning_trade_percent=(
            result.average_winning_trade_percent
        ),
        average_losing_trade_percent=(
            result.average_losing_trade_percent
        ),
        maximum_drawdown_percent=(
            result.maximum_drawdown_percent
        ),
        buy_and_hold_return_percent=(
            result.buy_and_hold_return_percent
        ),
        strategy_outperformance_percent=(
            result.strategy_outperformance_percent
        ),
        trades=[
            {
                "timestamp": trade.timestamp,
                "signal": trade.signal,
                "confidence": trade.confidence,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "return_percent": trade.return_percent,
                "profitable": trade.profitable,
                "position_return_percent": (
                    trade.position_return_percent
                ),
                "equity_after": trade.equity_after,
            }
            for trade in result.trades
        ],
    )


@router.get(
    "/ml-evaluation/{symbol}",
    response_model=MLEvaluationResponse,
)
def market_data_ml_evaluation(
    symbol: str,
    horizon: int = Query(5, ge=1, le=100),
    validation_fraction: float = Query(
        0.20,
        gt=0,
        lt=0.5,
    ),
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
            detail=(
                f"No market data found for "
                f"instrument '{symbol}'."
            ),
        )

    try:
        result = evaluate_symbol(
            rows,
            symbol=symbol,
            horizon=horizon,
            validation_fraction=validation_fraction,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return MLEvaluationResponse(
        symbol=result.symbol,
        dataset_rows=result.dataset_rows,
        training_rows=result.training_rows,
        validation_rows=result.validation_rows,
        direction_accuracy=result.direction_accuracy,
        average_absolute_error_percent=(
            result.average_absolute_error_percent
        ),
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