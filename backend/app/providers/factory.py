from app.providers.base import MarketDataProvider
from app.providers.mock import MockMarketDataProvider


def get_market_data_provider(
    provider_name: str,
) -> MarketDataProvider:
    if provider_name == "mock":
        return MockMarketDataProvider()

    raise ValueError(
        f"Unknown market data provider: '{provider_name}'"
    )