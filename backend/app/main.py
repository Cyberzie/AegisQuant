from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.router import api_router
from app.core.exceptions import register_exception_handlers
from app.core.logger import logger
from app.middleware.request_logger import log_requests


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AegisQuant")

    yield

    logger.info("Stopping AegisQuant")


app = FastAPI(
    title="AegisQuant",
    version="0.1.0-alpha",
    description="AegisQuant AI Trading Platform",
    lifespan=lifespan,
)

app.middleware("http")(log_requests)

register_exception_handlers(app)

app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "application": "AegisQuant",
        "version": "0.1.0-alpha",
        "status": "running",
    }