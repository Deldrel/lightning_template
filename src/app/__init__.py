import logging

import lightning  # noqa: F401
from pedros import setup_logging

setup_logging()

for logger_name in ("lightning", "lightning.pytorch", "lightning.fabric"):
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = True

logging.getLogger("wandb").handlers.clear()
logging.getLogger("wandb").propagate = True
