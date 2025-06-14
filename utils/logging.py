# trading_bot/utils/logging.py

import logging

def setup_logger(name="trading_bot", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

# Global logger (import this anywhere)
logger = setup_logger()