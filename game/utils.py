import logging


def log_print(*args, **kwargs):
    """Print and log output"""
    logger = logging.getLogger("cuttle")
    message = " ".join(str(arg) for arg in args)
    logger.info(message)
