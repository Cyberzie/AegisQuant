from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from app.schemas.paper_trading import (
    PaperTradingRequest,
    PaperTradingResponse,
)
from app.services.ensemble_signal import EnsembleSignal
from app.services.paper_trading import PaperTradingEngine
from app.services.portfolio import Portfolio
from app.services.risk_management import RiskParameters

from app.database.dependencies import get_db
from app.services.market_data_query import (
    get_market_data_by_symbol,
)
from app.services.baseline_evaluation import (
    build_latest_ensemble_signal,
)

router = APIRouter(
    prefix="/paper-trading",
)

_portfolio = Portfolio(
    starting_capital=1_000_000.0
)

_engine = PaperTradingEngine(
    _portfolio,
    transaction_cost_percent=0.10,
    slippage_percent=0.05,
)


@router.post(
    "/execute",
    response_model=PaperTradingResponse,
    status_code=status.HTTP_200_OK,
)
def execute_paper_trade(
    request: PaperTradingRequest,
):
    try:
        ensemble = EnsembleSignal(
            signal=request.signal,
            confidence=request.confidence,
            rule_signal=request.rule_signal,
            rule_confidence=request.rule_confidence,
            ml_signal=request.ml_signal,
            ml_confidence=request.ml_confidence,
            ml_expected_return_percent=(
                request.ml_expected_return_percent
            ),
            rule_weight=request.rule_weight,
            ml_weight=request.ml_weight,
        )

        result = _engine.process(
            symbol=request.symbol,
            ensemble=ensemble,
            entry_price=request.entry_price,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    decision = result.decision
    risk = decision.risk
    execution = result.execution

    return PaperTradingResponse(
        symbol=result.symbol,
        timestamp=datetime.now(timezone.utc),
        entry_price=result.entry_price,
        capital_before=result.capital_before,
        equity_after=result.equity_after,
        signal=decision.signal,
        confidence=decision.confidence,
        expected_return_percent=(
            decision.expected_return_percent
        ),
        rule_weight=decision.rule_weight,
        ml_weight=decision.ml_weight,
        risk={
            "approved": risk.approved,
            "reason": risk.reason,
            "risk_amount": risk.risk_amount,
            "position_size": risk.position_size,
            "position_value": risk.position_value,
            "position_percent": risk.position_percent,
            "stop_loss_price": risk.stop_loss_price,
            "take_profit_price": risk.take_profit_price,
        },
        execution={
            "symbol": execution.symbol,
            "signal": execution.signal,
            "executed": execution.executed,
            "reason": execution.reason,
            "quantity": execution.quantity,
            "execution_price": execution.execution_price,
            "realized_pnl": execution.realized_pnl,
            "cash_after": execution.cash_after,
        },
    )


@router.get(
    "/portfolio",
)
def get_paper_portfolio():
    market_prices = {
        symbol: position.average_entry_price
        for symbol, position
        in _portfolio.positions.items()
    }

    snapshot = _portfolio.snapshot(
        market_prices
    )

    return snapshot