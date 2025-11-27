import logging
from logging.handlers import TimedRotatingFileHandler
import os

def get_logger(name="translation_agent", log_dir="./logs"):
    """返回一个带文件与控制台输出的通用 logger"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)

    if not logger.handlers:  # 避免重复添加
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # 文件日志：每天自动新建一个文件，保留7天
        file_handler = TimedRotatingFileHandler(
            filename=os.path.join(log_dir, name + ".log"),
            when="midnight",
            backupCount=7,
            encoding="utf-8"
        )
        file_handler.setFormatter(formatter)

        # 控制台日志
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 绑定
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.INFO)

    return logger
