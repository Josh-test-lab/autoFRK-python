"""
Title: Setup Logger of autoFRK-Python Project
Author: Yao-Chih Hsu
Version: 1141019
Description: Check and setup logger for autoFRK-Python Project.
Reference: References from the SSSDS4 model by Wen-Ting Wang from https://github.com/egpivo/SSSD_CP/
"""

# import modules
import logging
import colorlog

# logger config
def setup_logger() -> logging.Logger:
    """
    Set up and return a logger with colored output.
    
    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(__name__)

    # Check if logger has already been configured
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    # Create a formatter
    formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    )

    # Create a console handler
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger

# set logger level
def set_logger_level(
    logger: logging.Logger,
    level: str | int
) -> None:
    """
    Set the logging level of a given logger.

    Parameters
    ----------
    logger : logging.Logger
        The logger instance to modify.
    level : str or int
        Logging level, e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL',
        or the corresponding numeric value.
    """
    if isinstance(level, str):
        if "logging." in level:
            level = level.replace("logging.", "") 
        level = level.upper()
        if not hasattr(logging, level):
            raise ValueError(f"Invalid logging level string: {level}")
        numeric_level = getattr(logging, level)
    elif isinstance(level, int):
        numeric_level = level
    else:
        raise TypeError("Logging level must be a str or int")

    logger.setLevel(numeric_level)


# setup logger
LOGGER = setup_logger()