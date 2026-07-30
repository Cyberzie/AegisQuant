from fastapi import APIRouter

from app.routers.health import router as health_router
from app.routers.version import router as version_router


api_router = APIRouter()

api_router.include_router(
    health_router,
    tags=["Health"]
)

api_router.include_router(
    version_router,
    tags=["Version"]
)