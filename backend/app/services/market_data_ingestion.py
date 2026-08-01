from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.instrument import Instrument
from app.models.market_data import MarketData


@dataclass
class IngestionResult:
    received: int = 0
    inserted: int = 0
    duplicates: int = 0
    invalid: int = 0


def ingest_market_data(
    db: Session,
    rows: list[dict],
) -> IngestionResult:
    result = IngestionResult(received=len(rows))

    for row in rows:
        try:
            instrument_id = int(row["instrument_id"])
            timestamp = row["timestamp"]

            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)

            instrument = (
                db.query(Instrument)
                .filter(Instrument.id == instrument_id)
                .first()
            )

            if instrument is None:
                result.invalid += 1
                continue

            high = float(row["high"])
            low = float(row["low"])
            open_price = float(row["open"])
            close = float(row["close"])

            volume = row.get("volume")

            if volume is not None:
                volume = float(volume)

            if high < low:
                result.invalid += 1
                continue

            if not low <= open_price <= high:
                result.invalid += 1
                continue

            if not low <= close <= high:
                result.invalid += 1
                continue

            if volume is not None and volume < 0:
                result.invalid += 1
                continue

            existing = (
                db.query(MarketData)
                .filter(
                    MarketData.instrument_id == instrument_id,
                    MarketData.timestamp == timestamp,
                )
                .first()
            )

            if existing is not None:
                result.duplicates += 1
                continue

            market_data = MarketData(
                instrument_id=instrument_id,
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )

            db.add(market_data)
            result.inserted += 1

        except (KeyError, TypeError, ValueError):
            result.invalid += 1

    db.commit()

    return result