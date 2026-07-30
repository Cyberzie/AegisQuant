from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    return {
        "application": "AegisQuant",
        "status": "healthy",
    }


@router.get("/ping")
async def ping():
    return {
        "message": "pong"
    }