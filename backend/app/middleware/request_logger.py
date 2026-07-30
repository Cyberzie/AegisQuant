import time

from fastapi import Request
from loguru import logger


async def log_requests(request: Request, call_next):
    start = time.perf_counter()

    try:
        response = await call_next(request)

        duration = (time.perf_counter() - start) * 1000

        logger.info(
            f"{request.method} {request.url.path} "
            f"-> {response.status_code} "
            f"({duration:.2f} ms)"
        )

        return response

    except Exception:
        duration = (time.perf_counter() - start) * 1000

        logger.exception(
            f"{request.method} {request.url.path} "
            f"failed after {duration:.2f} ms"
        )

        raise