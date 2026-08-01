from app.core.config import settings
from app.providers.base import MarketDataProvider
from app.providers.mock import MockMarketDataProvider
from app.providers.twelve_data import TwelveDataProvider


def get_market_data_provider(
    provider_name: str | None = None,
) -> MarketDataProvider:
    name = provider_name or settings.MARKET_DATA_PROVIDER

    if name == "mock":
        return MockMarketDataProvider()

    if name == "twelve_data":
        return TwelveDataProvider()

    raise ValueError(
        f"Unknown market data provider: '{name}'"
    )