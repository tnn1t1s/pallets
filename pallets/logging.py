import sys
import logging


logger = logging.getLogger("pallets")


def get_log_level(level):
    if level == "DEBUG":
        return logging.DEBUG
    elif level == "INFO":
        return logging.INFO
    elif level == "WARNING":
        return logging.WARNING
    elif level == "ERROR":
        return logging.ERROR
    elif level == "CRITICAL":
        return logging.CRITICAL
    else:
        raise Exception(f"Unrecognized log level: {level}")


def init_logger(level="INFO", timestamp=False, notebook=False):
    config = { 'format': "%(levelname)s | %(message)s" }

    if notebook:
        config['stream'] = sys.stdout
    
    if timestamp:
        config['format'] = "%(asctime)s | %(levelname)s | %(message)s"
        config['datefmt'] = "%m/%d/%Y %H:%M:%S"

    logging.basicConfig(**config)
    logger_level = get_log_level(level)
    logger.setLevel(logger_level)

    logging.info("pallets v0.1")
    return logger


def log_train_config(model, criterion, epochs, learn_rate):
    """
    Logs basic elements of a training config using a consistent format.
    """
    logger.info(
        f"model: {model.__class__.__module__}.{model.__class__.__name__}"
    )
    logger.info(
        f"criterion: {criterion.__class__.__module__}.{criterion.__class__.__name__}"
    )
    logger.info(f"learn rate: {learn_rate}")
    logger.info(f"epochs: {epochs}")
