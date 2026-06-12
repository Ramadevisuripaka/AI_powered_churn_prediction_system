'''
In this file we are going to setup logging for all modules
'''
import logging
import os


def setup_logging(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Create logs folder if not exists
        os.makedirs("logs", exist_ok=True)

        # File handler - each module gets its own log file
        file_handler = logging.FileHandler(f"logs/{name}.log")
        file_handler.setLevel(logging.INFO)

        # Console handler - still shows in terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger