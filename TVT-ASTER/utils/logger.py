# -*- coding: utf-8 -*-
import logging

def get_logger(logfile: str = "train.log") -> logging.Logger:
    logger = logging.getLogger("runner")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(); sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    fh = logging.FileHandler(logfile, mode="a", encoding="utf-8"); fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    logger.addHandler(sh); logger.addHandler(fh)
    return logger
