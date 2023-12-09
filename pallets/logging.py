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


def init_logger(level="INFO", timestamp=False):
    config = { 'format': "%(levelname)s | %(message)s" }

    if timestamp:
        config['format'] = "%(asctime)s | %(levelname)s | %(message)s"
        config['datefmt'] = "%m/%d/%Y %H:%M:%S"

    logging.basicConfig(**config)
    logger_level = get_log_level(level)
    logger.setLevel(logger_level)

    logging.info("pallets v0.1")
    return logger

