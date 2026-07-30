from pathlib import Path

from loguru import logger

import sys

Path("logs").mkdir(exist_ok=True)

logger.remove()

logger.add(
    sys.stdout,
    level="INFO"
)

logger.add(
    "logs/aegisquant.log",
    rotation="10 MB",
    retention="30 days"
)