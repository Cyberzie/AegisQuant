from datetime import datetime

from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.models.instrument import Instrument
from app.models.market_data import MarketData


client = TestClient(app)


def test_ingest_market_data_from_provider():
    db = SessionLocal()

    start = datetime(2026, 8, 4, 10, 0, 0)
    end = datetime(2026, 8, 4, 12, 0, 0)

    instrument = None

    try:
        instrument = (
            db.query(Instrument)
            .filter(Instrument.symbol == "AAPL")
            .first()
        )

        assert instrument is not None

        response = client.post(
            "/market-data/ingest",
            json={
                "symbol": "AAPL",
                "provider": "mock",
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["received"] == 3
        assert data["inserted"] == 3
        assert data["duplicates"] == 0
        assert data["invalid"] == 0

    finally:
        if instrument is not None:
            db.query(MarketData).filter(
                MarketData.instrument_id == instrument.id,
                MarketData.timestamp >= start,
                MarketData.timestamp <= end,
            ).delete(
                synchronize_session=False
            )

            db.commit()

        db.close()

def test_market_data_baseline_comparison_endpoint():
    db = SessionLocal()

    instrument = None
    start = datetime(2026, 8, 4, 10, 0, 0)
    end = datetime(2026, 8, 4, 13, 0, 0)

    try:
        instrument = (
            db.query(Instrument)
            .filter(Instrument.symbol == "AAPL")
            .first()
        )

        assert instrument is not None

        response = client.post(
            "/market-data/ingest",
            json={
                "symbol": "AAPL",
                "provider": "mock",
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )

        assert response.status_code == 200

        result = client.get(
            "/market-data/baseline-comparison/AAPL",
            params={
                "horizon": 1,
                "initial_training_fraction": 0.60,
                "folds": 1,
                "gap_rows": 1,
            },
        )

        assert result.status_code == 200

        data = result.json()

        assert data["symbol"] == "AAPL"
        assert data["horizon"] == 1
        assert data["dataset_rows"] > 0
        assert data["total_training_rows"] > 0
        assert data["total_validation_rows"] > 0

        assert len(data["strategies"]) == 6

        strategy_names = {
            strategy["name"]
            for strategy in data["strategies"]
        }

        assert strategy_names == {
            "always_buy",
            "always_sell",
            "rule",
            "ml",
            "ensemble",
            "adaptive_ensemble",
        }

    finally:
        if instrument is not None:
            db.query(MarketData).filter(
                MarketData.instrument_id == instrument.id,
                MarketData.timestamp >= start,
                MarketData.timestamp <= end,
            ).delete(
                synchronize_session=False
            )

            db.commit()

        db.close()