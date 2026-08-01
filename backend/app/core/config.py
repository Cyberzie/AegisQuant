from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "AegisQuant"
    ENVIRONMENT: str = "development"
    DATABASE_URL: str
    SECRET_KEY: str
    LOG_LEVEL: str = "INFO"
    MARKET_DATA_PROVIDER: str = "mock"
    TWELVE_DATA_API_KEY: str = ""
    TWELVE_DATA_BASE_URL: str = "https://api.twelvedata.com"
    TWELVE_DATA_INTERVAL: str = "1day"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()