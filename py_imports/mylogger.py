# -*- coding: utf-8 -*-
import os
import logging
import datetime
#%%
def mylogger(logging_level=logging.INFO):
    """
    Logger implementation.\n
    """
    now = datetime.datetime.now()
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        
    time_log = "{}_{}_{}_{}_{}".format(now.year, now.month, now.day, now.hour, now.minute)
    log_filename = f"DL_Demokritos_{__name__}_{time_log}.log"
    log_path = os.path.join(log_dir, log_filename)
        
    logger = logging.getLogger(__name__)
    logger.setLevel(logging_level)

    formatter_file   = logging.Formatter("%(asctime)s -- %(levelname)s -- %(message)s","%Y-%m-%d %H:%M:%S")
    formatter_stream = logging.Formatter('%(name)s:%(message)s')

    file_handler = logging.FileHandler(log_path , mode="w")
    file_handler.setFormatter(formatter_file)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter_stream)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger