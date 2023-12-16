

import log
import os
import functools
import time
import glob
from pyhocon import ConfigFactory
from typing import Dict, List, Tuple
from datetime import datetime
import rasterio
import matplotlib.pyplot as plt

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


def save_plot(fig, filename):
    fig.savefig(filename)


def load_raster(path):
    # Load raster data
    img = rasterio.open(path)
    logger.info(f"Loaded image: {path}")
    logger.info(f"Image channels: {img.count}")
    logger.info(f"Image size: {img.width}x{img.height}")
    logger.info(f"Image crs: {img.crs}")
    logger.info(f"Image bounds: {img.bounds}")
    logger.info(f"Image transform: {img.transform}")
    return img


def get_filenames(path, ext, keyword=None):
    pattern = f'{path}*.{ext}'
    file_paths = glob.glob(pattern)
    
    if keyword:
        filtered_paths = [file_path for file_path in file_paths if keyword in file_path]
        return filtered_paths
    else:
        return file_paths


def plot_cropped_tifs(cropped_tifs, title):
    ncols = 3 # set number of columns (use 3 to demonstrate the change)
    nrows = len(cropped_tifs) // ncols + (len(cropped_tifs) % ncols > 0) # calculate number of rows

    plt.figure(figsize=(15, 20))
    plt.suptitle(title, fontsize=18, y=0.95)

    for n, file in enumerate(cropped_tifs):
        # add a new subplot iteratively using nrows and cols
        ax = plt.subplot(nrows, ncols, n + 1)
        # Plot raster crop
        saved_clipped_raster_io = rasterio.open(f"{file}")
        rasterio.plot.show(saved_clipped_raster_io, ax=ax)
        # chart formatting
        ax.set_title(os.path.basename(file), fontsize=8)
        ax.axis('off')
    plt.show()


def sync_crs(gdf, rasterimg) -> bool: 
    if gdf.crs != rasterimg.crs:
        gdf = gdf.set_crs(str(rasterimg.crs))
    return gdf