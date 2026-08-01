import pytest
from fastapi.testclient import TestClient

from app.database.session import SessionLocal
from app.main import app
from app.models.instrument import Instrument
from app.models.market_data import MarketData


client = TestClient(app)


@pytest.mark.live
def test_live_market_data_ingestion_api():
    db = SessionLocal()

    instrument = None
    inserted_timestamps = []

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
                "provider": "twelve_data",
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert data["received"] > 0
        assert data["invalid"] == 0

        inserted_rows = (
            db.query(MarketData)
            .filter(
                MarketData.instrument_id == instrument.id,
            )
            .order_by(MarketData.timestamp.desc())
            .limit(data["inserted"])
            .all()
        )

        inserted_timestamps = [
            row.timestamp for row in inserted_rows
        ]

        print(f"\nReceived: {data['received']}")
        print(f"Inserted: {data['inserted']}")
        print(f"Duplicates: {data['duplicates']}")
        print(f"Invalid: {data['invalid']}")

    finally:
        if instrument is not None and inserted_timestamps:
            db.query(MarketData).filter(
                MarketData.instrument_id == instrument.id,
                MarketData.timestamp.in_(inserted_timestamps),
            ).delete(
                synchronize_session=False
            )

            db.commit()

        db.close()