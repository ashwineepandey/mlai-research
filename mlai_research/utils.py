import log
import os
import functools
import time
import glob
import cv2
from pyhocon import ConfigFactory
from typing import Dict, List, Tuple
from datetime import datetime
import rasterio
import matplotlib.pyplot as plt
import numpy as np

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


# def get_filenames(path, ext, keyword=None):
#     pattern = f'{path}*.{ext}'
#     file_paths = glob.glob(pattern)
    
#     if keyword:
#         filtered_paths = [file_path for file_path in file_paths if keyword in file_path]
#         return filtered_paths
#     else:
#         return file_paths

def get_filenames(path, ext, keywords=None):
    pattern = f'{path}*.{ext}'
    file_paths = glob.glob(pattern)
    
    if keywords:
        filtered_paths = [file_path for file_path in file_paths if any(keyword in file_path for keyword in keywords)]
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


def load_rgb_images(image_paths):
    """
    Load RGB images from the provided paths.

    Parameters:
    - image_paths (List[str]): List of paths to the images.

    Returns:
    - List[np.ndarray]: List of loaded images.
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        images.append(img)
    logger.info(f'Loaded RGB img shape: {images[0].shape}')
    return images


def load_rgb_rasters(image_paths: List[str]) -> List[np.ndarray]:
    """
    Load RGBA rasters from the provided paths.

    Parameters:
    - image_paths (List[str]): List of paths to the rasters.

    Returns:
    - List[np.ndarray]: List of loaded rasters.
    """
    images = []
    for path in image_paths:
        with rasterio.open(path) as src:
            img = src.read()
            image_data = np.transpose(img, (1, 2, 0))
            rgb = image_data[:, :, :3]
        images.append(rgb)
    logger.info(f'Loaded RGB img shape: {images[0].shape}')
    return images

def load_grayscale_arrays(image_paths: List[str]) -> List[np.ndarray]:
    """
    Load grayscale images from the provided paths.

    Parameters:
    - image_paths (List[str]): List of paths to the images.

    Returns:
    - List[np.ndarray]: List of loaded images.
    """
    images = []
    for path in image_paths:
        img = np.load(path)
        image_data = np.transpose(img, (1, 2, 0))
        images.append(image_data)
    logger.info(f'Loaded GRAY img shape: {images[0].shape}')
    return images

def load_hyperspectral_rasters(image_paths: List[str]) -> List[np.ndarray]:
    """
    Load hyperspectral images from the provided paths.

    Parameters:
    - image_paths (List[str]): List of paths to the images.

    Returns:
    - List[np.ndarray]: List of loaded images.
    """
    images = []
    for path in image_paths:
        with rasterio.open(path) as src:
            img = src.read()
        # Transpose the image to have channels last
        img = img.transpose((1, 2, 0))
        images.append(img)
    logger.info(f'Loaded HYPS img shape: {images[0].shape}')
    return images


def calculate_zero_pixel_percentage(rgb_image: np.ndarray) -> float:
    """
    Calculate the percentage of zero-value pixels in an RGB image.

    Args:
        rgb_image (np.ndarray): The input RGB image as a NumPy array with shape (height, width, 3).

    Returns:
        float: The percentage of pixels in the image that are completely black (0, 0, 0).
    """
    # Check if the image is empty or not an RGB image
    if rgb_image.size == 0 or len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input is not a valid RGB image.")

    # Count the number of zero-value pixels
    zero_pixels = np.all(rgb_image == 0, axis=2)
    num_zero_pixels = np.count_nonzero(zero_pixels)

    # Calculate the total number of pixels
    total_pixels = rgb_image.shape[0] * rgb_image.shape[1]

    # Calculate the percentage of zero-value pixels
    zero_pixel_percentage = (num_zero_pixels / total_pixels) * 100

    return zero_pixel_percentage