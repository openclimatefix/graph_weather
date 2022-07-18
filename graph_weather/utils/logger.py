import logging
import sys
import time


def get_logger(name: str, debug: bool = True) -> logging.Logger:
    """
    Returns a logger with a custom level and format.
    We use ISO8601 timestamps and UTC times.

    Args:
        name : name of logger object
        debug : if True: set logging level to logging.DEBUG; else set to logging.INFO
    Returns:
        The logger object.
    """
    # create logger object
    logger = logging.getLogger(name=name)
    if not logger.hasHandlers():
        # logging level
        level = logging.DEBUG if debug else logging.INFO
        # logging format
        datefmt = "%Y-%m-%dT%H:%M:%SZ"
        msgfmt = "[%(asctime)s] [%(filename)s:%(lineno)s - %(funcName).30s] [%(levelname)s] %(message)s"
        # handler object
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(msgfmt, datefmt=datefmt)
        # record UTC time
        setattr(formatter, "converter", time.gmtime)
        handler.setFormatter(formatter)
        logger.setLevel(level)
        logger.addHandler(handler)
    return logger
