import logging
from pathlib import Path


def setup_logger(name='project', log_file='project.log', level=logging.DEBUG):
    log_dir = Path("output/log")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = log_dir / log_file

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


if __name__ == '__main__':
    logger = setup_logger()
    logger.info('This is an info message')
    logger.error('This is an error message')