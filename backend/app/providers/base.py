from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any


class MarketDataProvider(ABC):
    """Base interface for all market-data providers."""

    @abstractmethod
    def get_market_data(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return normalized OHLCV market data."""
        raise NotImplementedError