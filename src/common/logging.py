from __future__ import annotations

import logging
import sys

from pythonjsonlogger import jsonlogger

from src.common.config import get_config

_CONFIGURED = False


def setup_logging() -> None:
    """Configura o root logger uma única vez por processo."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    config = get_config()

    handler = logging.StreamHandler(sys.stdout)

    if config.logging.json_format:
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={
                "asctime": "timestamp",
                "levelname": "level",
                "name": "logger",
            },
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )

    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(config.logging.level)

    # bibliotecas de terceiros tendem a ser barulhentas em DEBUG
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("azure").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger já configurado. Chame com __name__ do módulo chamador."""
    setup_logging()
    return logging.getLogger(name)
