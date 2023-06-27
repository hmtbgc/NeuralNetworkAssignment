import logging
import os
import datetime

def get_now():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d-%H:%M:%S-")

def init_logging(path):
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(path)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger

def get_name(args):
    args_dict = vars(args)
    name = ""
    for key, value in args_dict.items():
        if (key != "pretrained"):
            name += str(key) + "_" + str(value) + "#"
    return name

def check_and_mkdir(path):
    if (not os.path.exists(path)):
        os.mkdir(path)