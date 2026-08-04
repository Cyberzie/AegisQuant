from datetime import datetime

from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.models.instrument import Instrument
from app.models.market_data import MarketData


client = TestClient(app)


def _seed_market_data(
    db,
    instrument_id: int,
    count: int = 80,
):
    start = datetime(2026, 8, 4, 8, 0, 0)

    rows = []
    price = 100.0

    for index in range(count):
        price += (
            0.30
            if index % 6 != 0
            else -0.15
        )

        rows.append(
            MarketData(
                instrument_id=instrument_id,
                timestamp=start.replace(
                    minute=index % 60,
                    hour=8 + index // 60,
                ),
                open=price,
                high=price + 0.50,
                low=price - 0.50,
                close=price,
                volume=1000.0 + index,
            )
        )

    db.add_all(rows)
    db.commit()

    return rows


def test_trading_decision_endpoint_returns_risk_aware_decision():
    db = SessionLocal()

    instrument = None
    rows = []

    try:
        instrument = (
            db.query(Instrument)
            .filter(
                Instrument.symbol == "AAPL"
            )
            .first()
        )

        assert instrument is not None

        rows = _seed_market_data(
            db,
            instrument.id,
        )

        response = client.get(
            "/market-data/trading-decision/AAPL",
            params={
                "capital": 10_000,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["symbol"] == "AAPL"
        assert data["close"] > 0

        assert data["signal"] in {
            "BUY",
            "SELL",
            "HOLD",
        }

        assert 0.0 <= data["confidence"] <= 1.0

        assert isinstance(
            data["expected_return_percent"],
            float,
        )

        assert 0.0 <= data["rule_weight"] <= 1.0
        assert 0.0 <= data["ml_weight"] <= 1.0

        assert (
            data["rule_weight"]
            + data["ml_weight"]
        ) == 1.0

        risk = data["risk"]

        assert isinstance(
            risk["approved"],
            bool,
        )

        assert risk["risk_amount"] >= 0
        assert risk["position_size"] >= 0
        assert risk["position_value"] >= 0
        assert risk["position_percent"] >= 0

        if risk["approved"]:
            assert risk["stop_loss_price"] is not None
            assert risk["take_profit_price"] is not None

    finally:
        if rows:
            db.query(MarketData).filter(
                MarketData.instrument_id
                == instrument.id,
                MarketData.timestamp
                >= rows[0].timestamp,
                MarketData.timestamp
                <= rows[-1].timestamp,
            ).delete(
                synchronize_session=False
            )

            db.commit()

        db.close()


def test_trading_decision_rejects_invalid_capital():
    response = client.get(
        "/market-data/trading-decision/AAPL",
        params={
            "capital": 0,
        },
    )

    assert response.status_code == 422