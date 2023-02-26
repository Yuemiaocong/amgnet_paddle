import os
import sys
import logging
def logger_setup(log_path):
    root_logger = logging.getLogger() 
    root_logger.setLevel(logging.INFO) 
    file_log_handler = logging.FileHandler(filename=log_path, mode='w', encoding='utf-8')
    file_log_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(message)s') 
    file_log_handler.setFormatter(formatter)
    root_logger.addHandler(file_log_handler)
    return root_logger

