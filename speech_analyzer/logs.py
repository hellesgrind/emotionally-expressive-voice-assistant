import sys
from loguru import logger

log_format = (
    "<green>{time:YYYY-MM-DD HH:mm}</green> | "
    "<level>{level: <8}</level>"
    " | "
    "<cyan>{name}</cyan>:"
    "<cyan>{function}</cyan>:"
    "<cyan>{line}</cyan>"
    " - "
    "{message}"
)

logger.remove(0)
logger.add(sys.stdout, format=log_format, colorize=True)
