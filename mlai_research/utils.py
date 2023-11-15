

import sys
import yaml
import log
import functools
import time
from pyhocon import ConfigFactory
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd
import yaml

logger = log.get_logger(__name__)


def timer(func):
    """ Print the runtime of the decorated function """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        logger.info(f"Starting {func.__name__!r}.")
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logger.info(f"Finished {func.__name__!r} in {run_time:.4f} secs.")
        return value
    return wrapper_timer


@timer
def load_config(fn):
    """
    Load the configuration file from a hocon file object and returns it (https://github.com/chimpler/pyhocon).
    """
    return ConfigFactory.parse_file(f"../config/{fn}.conf")
