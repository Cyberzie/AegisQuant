from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    APP_NAME: str = "AegisQuant"

    ENVIRONMENT: str = "development"

    DATABASE_URL: str

    SECRET_KEY: str

    LOG_LEVEL: str = "INFO"

    class Config:

        env_file = ".env"


@lru_cache
def get_settings():

    return Settings()


settings = get_settings()