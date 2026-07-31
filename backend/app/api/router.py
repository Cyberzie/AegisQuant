from fastapi import APIRouter

from app.routers.health import router as health_router
from app.routers.version import router as version_router
from app.routers.market_data import router as market_data_router
from app.routers.instruments import router as instruments_router


api_router = APIRouter()


api_router.include_router(
    health_router,
    tags=["Health"],
)

api_router.include_router(
    version_router,
    tags=["Version"],
)


api_router.include_router(
    instruments_router,
    tags=["Instruments"],
)

api_router.include_router(
    market_data_router,
    tags=["Market Data"]
)