import logging


def build_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Build a logger with the specified name.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler()
    logger_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(logger_formatter)
    logger.addHandler(console_handler)

    return logger
