from fastapi import APIRouter

router = APIRouter()


@router.get("/version")
async def version():
    return {
        "version": "0.1.0-alpha"
    }