import logging
import colorlog

# logger config
def setup_logger() -> logging.Logger:
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

    return logger
