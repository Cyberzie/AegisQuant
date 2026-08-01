from datetime import datetime
from typing import Any

import requests

from app.core.config import settings
from app.providers.base import MarketDataProvider


class TwelveDataProvider(MarketDataProvider):
    """Market-data provider backed by Twelve Data."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        interval: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.TWELVE_DATA_API_KEY
        self.base_url = (
            base_url
            or settings.TWELVE_DATA_BASE_URL
        ).rstrip("/")
        self.interval = (
            interval
            or settings.TWELVE_DATA_INTERVAL
        )

    def get_market_data(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        params = {
            "symbol": symbol,
            "interval": self.interval,
            "apikey": self.api_key,
        }

        if start is not None:
            params["start_date"] = start.isoformat()

        if end is not None:
            params["end_date"] = end.isoformat()

        response = requests.get(
            f"{self.base_url}/time_series",
            params=params,
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") == "error":
            raise ValueError(
                payload.get(
                    "message",
                    "Twelve Data returned an API error.",
                )
            )

        values = payload.get("values", [])

        rows = []

        for value in values:
            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": datetime.fromisoformat(
                        value["datetime"]
                    ),
                    "open": float(value["open"]),
                    "high": float(value["high"]),
                    "low": float(value["low"]),
                    "close": float(value["close"]),
                    "volume": (
                        float(value["volume"])
                        if value.get("volume") is not None
                        else None
                    ),
                }
            )

        return rows