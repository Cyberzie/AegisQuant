from datetime import datetime, timedelta
from typing import Any

from app.providers.base import MarketDataProvider


class MockMarketDataProvider(MarketDataProvider):
    """Deterministic market-data provider for development and testing."""

    def get_market_data(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:

        if start is None:
            start = datetime(2026, 8, 1, 10, 0, 0)

        rows = []

        prices = [
            (230.0, 235.0, 228.0, 233.0, 1_000_000),
            (233.0, 238.0, 231.0, 236.0, 1_100_000),
            (236.0, 240.0, 234.0, 239.0, 1_200_000),
        ]

        for index, (
            open_price,
            high,
            low,
            close,
            volume,
        ) in enumerate(prices):

            timestamp = start + timedelta(hours=index)

            if end is not None and timestamp > end:
                break

            rows.append(
                {
                    "symbol": symbol,
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        return rows