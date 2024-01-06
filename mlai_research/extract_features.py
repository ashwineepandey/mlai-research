import os
import sys
sys.path.append('../mlai_research/')
import log
import utils
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops
import rasterio
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
import pandas as pd

logger = log.get_logger(__name__)

# def load_rgb_images(image_paths):
#     images = []
#     for path in image_paths:
#         img = cv2.imread(path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
#         images.append(img_rgb)
#     return np.array(images)


def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize the input feature array.

    Parameters:
    - features (np.ndarray): Input feature array.

    Returns:
    - np.ndarray: Normalized feature array.
    """
    return (features - features.min()) / (features.max() - features.min())

# @utils.timer
# def load_grayscale_images(image_paths):
#     images = []
#     for path in image_paths:
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         images.append(img)
#     return np.array(images)

# @utils.timer
# def load_hyperspectral_images(image_paths):
#     images = []
#     for path in image_paths:
#         with rasterio.open(path) as src:
#             img = src.read()
#             # Transpose the image to have channels last
#             img = img.transpose((1, 2, 0))
#             images.append(img)
#     return images


def plot_cropped_images(images: list[np.ndarray], titles: list[str], ncols: int = 3) -> None:
    """
    Plot a list of loaded images.

    Parameters:
    - images (list[np.ndarray]): List of loaded images.
    - titles (list[str]): List of titles for the images.
    - ncols (int): Number of columns for the subplot grid.
    """
    nrows = len(images) // ncols + (len(images) % ncols > 0) # calculate number of rows

    plt.figure(figsize=(10, 10))
    plt.suptitle('Cropped Images', fontsize=18, y=0.95)

    for n, img in enumerate(images):
        # add a new subplot iteratively using nrows and cols
        ax = plt.subplot(nrows, ncols, n + 1)
        # Plot raster crop
        ax.imshow(img)
        # chart formatting
        ax.set_title(os.path.basename(titles[n]), fontsize=8)
        ax.axis('off')
    plt.show()


def extract_color_histogram(image: np.ndarray, bins: int = 32) -> np.ndarray:
    """
    Extracts the color histogram from the input image.

    Parameters:
    - image (numpy.ndarray): The input image.
    - bins (int): The number of bins for the histogram.

    Returns:
    - numpy.ndarray: The color histogram.
    """
    # Compute the histogram of the RGB channels separately
    rhist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    ghist = cv2.calcHist([image], [1], None, [bins], [0, 256])
    bhist = cv2.calcHist([image], [2], None, [bins], [0, 256])

    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate([rhist, ghist, bhist])

    # Normalize the histogram
    # cv2.normalize(hist_features, hist_features)
    hist_features = normalize_features(hist_features)
    logger.info(f'Color histogram shape: {hist_features.shape}')
    return hist_features


def extract_texture_features(image: np.ndarray) -> np.ndarray:
    """
    Extracts texture features from the input image.

    Parameters:
    - image (numpy.ndarray): The input image.

    Returns:
    - numpy.ndarray: The texture features.
    """
    # Convert the image to uint8
    image_uint8 = (image * 255).astype(np.uint8)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)

    # Compute the GLCM of the grayscale image
    glcm = graycomatrix(gray_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])

    # Compute texture features from the GLCM
    contrast = graycoprops(glcm, 'contrast')
    dissimilarity = graycoprops(glcm, 'dissimilarity')
    homogeneity = graycoprops(glcm, 'homogeneity')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')

    # Concatenate the texture features into a single feature vector
    texture_features = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
    
    # Normalize the texture features
    # cv2.normalize(texture_features, texture_features)
    texture_features = normalize_features(texture_features)
    logger.info(f'Texture features shape: {texture_features.shape}')
    return texture_features


def extract_shape_features(image: np.ndarray, max_descriptors=10) -> np.ndarray:
    """
    Extracts shape features from the input image.

    Parameters:
    - image (numpy.ndarray): The input image.

    Returns:
    - numpy.ndarray: The shape features.
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Normalize the image to have a depth of 8 bits
    gray_image = cv2.normalize(gray_image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    # Initialize the SIFT feature extractor
    sift = cv2.xfeatures2d.SIFT_create()

    # Compute the SIFT features
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    # Ensure that the shape of descriptors is consistent
    # For example, you can take the first n descriptors if the shape is greater than n
    # Choose an appropriate number based on your needs
    if descriptors.shape[0] < max_descriptors:
        # If fewer descriptors than expected, pad with zeros
        pad_width = max_descriptors - descriptors.shape[0]
        descriptors = np.pad(descriptors, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
    elif descriptors.shape[0] > max_descriptors:
        # If more descriptors than expected, truncate
        descriptors = descriptors[:max_descriptors, :]

    # Normalize the descriptors
    # cv2.normalize(descriptors, descriptors)
    descriptors = normalize_features(descriptors)
    logger.info(f'Shape features shape: {descriptors.shape}')
    return descriptors


def extract_geometric_features(rgb_image):
    """
    Extract geometric features from an RGB image of a leaf.

    Parameters:
    rgb_image (numpy.ndarray): RGB image of a leaf.

    Returns:
    numpy.ndarray: Array containing the geometric features.
    """
    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to create a binary mask
    _, binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Get region properties from the binary mask
    props = regionprops(binary_mask)

    # Extract specific geometric features
    area = props[0].area
    perimeter = props[0].perimeter
    eccentricity = props[0].eccentricity
    extent = props[0].extent
    aspect_ratio = props[0].major_axis_length / props[0].minor_axis_length
    roundness = 4 * np.pi * area / (perimeter ** 2)
    compactness = area / props[0].convex_area

    # Combine features into a NumPy array
    geometric_features = np.array([area, perimeter, eccentricity, extent, aspect_ratio, roundness, compactness])

    # cv2.normalize(geometric_features, geometric_features)
    geometric_features = normalize_features(geometric_features)
    logger.info(f'Geometric features shape: {geometric_features.shape}')
    return geometric_features


def preprocess_input(x: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image array for ResNet.

    Parameters:
    x (np.ndarray): The input image array.

    Returns:
    np.ndarray: Preprocessed image array.
    """
    x = x.astype(np.float64)  # Convert the image to float64
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x /= 255.0
    x = (x - mean) / std
    return x


def extract_resnet_features(img_data: np.ndarray) -> np.ndarray:
    """
    Extract features using a pre-trained ResNet50 model.

    Parameters:
    img_data (np.ndarray): The image data.

    Returns:
    np.ndarray: Extracted features.
    """
    # Load the pre-trained ResNet50 model
    base_model = models.resnet50(pretrained=True)
    
    # Remove the last layer (fully connected layer)
    base_model = nn.Sequential(*list(base_model.children())[:-1])
    
    # Add an adaptive average pooling layer
    base_model.add_module('GlobalAveragePooling', nn.AdaptiveAvgPool2d((1, 1)))
    
    # Convert the image data into a torch tensor, add a batch dimension, and transpose the image
    img_data = torch.from_numpy(img_data).unsqueeze(0).permute(0, 3, 1, 2)

    # Preprocess the image data
    img_data = preprocess_input(img_data)
    
    # Convert the image data into a torch tensor and add a batch dimension
    img_data = torch.from_numpy(img_data).unsqueeze(0)
    
    # Set the model to evaluation mode and extract features
    base_model.eval()
    with torch.no_grad():
        res_feature = base_model(img_data)
    
    # Remove the batch dimension and convert the tensor to a numpy array
    res_feature = res_feature.squeeze(0).numpy()
    
    return res_feature

@utils.timer
def get_rgb_features(rgb_image: np.ndarray) -> np.ndarray:
    """
    Extract and combine various features from an RGB image.

    Parameters:
    rgb_image (np.ndarray): The RGB image.

    Returns:
    np.ndarray: Combined feature vector.
    """
    # Extract individual feature sets
    color_histogram = extract_color_histogram(rgb_image)
    texture_features = extract_texture_features(rgb_image)
    shape_features = extract_shape_features(rgb_image)
    geometric_features = extract_geometric_features(rgb_image)

    # Concatenate features into a single feature vector
    rgb_features = np.concatenate([color_histogram.flatten(), texture_features.flatten(), shape_features.flatten(), geometric_features.flatten()])
    # Normalize the feature vector
    rgb_features = normalize_features(rgb_features)
    logger.info(f'RGB features shape: {rgb_features.shape}')
    return rgb_features


def get_chm_features(chm_data: np.ndarray) -> np.ndarray:
    """
    Calculate basic statistical features from Canopy Height Model (CHM) data.

    Args:
        chm_data (np.ndarray): The CHM data as a NumPy array.

    Returns:
        np.ndarray: A NumPy array containing calculated features including mean height, 
                    maximum height, minimum height, and height range.
    """
    mean_height = np.mean(chm_data)
    max_height = np.max(chm_data)
    min_height = np.min(chm_data)
    height_range = max_height - min_height

    chm_features = np.array([mean_height, max_height, min_height, height_range])
    chm_features = normalize_features(chm_features)
    logger.info(f'CHM features shape: {chm_features.shape}')
    return chm_features


def get_hyperspectral_features(hyperspectral_data: np.ndarray) -> np.ndarray:
    """
    Calculate mean and standard deviation for each spectral band in hyperspectral data.

    Args:
        hyperspectral_data (np.ndarray): The hyperspectral data as a NumPy array.

    Returns:
        np.ndarray: A NumPy array of concatenated mean and standard deviation values 
                    for each spectral band.
    """
    mean_spectrum = np.mean(hyperspectral_data, axis=(1, 2))
    std_spectrum = np.std(hyperspectral_data, axis=(1, 2))

    hyperspectral_features = np.concatenate((mean_spectrum, std_spectrum))
    hyperspectral_features = normalize_features(hyperspectral_features)
    logger.info(f'Hyperspectral features shape: {hyperspectral_features.shape}')
    return hyperspectral_features


def combine_features(rgb_features: np.ndarray, hyperspectral_features: np.ndarray, chm_features: np.ndarray) -> np.ndarray:
    """
    Combine features from RGB, hyperspectral, and CHM data sources into a single feature vector.

    Args:
        rgb_features (np.ndarray): The features extracted from RGB data.
        hyperspectral_features (np.ndarray): The features extracted from hyperspectral data.
        chm_features (np.ndarray): The features extracted from CHM data.

    Returns:
        np.ndarray: A normalized combined feature vector.
    """
    combined_features = np.concatenate((rgb_features, hyperspectral_features, chm_features), axis=0)
    # norm_combined_features = normalize_features(combined_features)
    logger.info(f'Combined features shape: {combined_features.shape}')
    return combined_features


def extract_label_from_filename(filename: str) -> str:
    """
    Extract the label from the filename.

    Args:
        filename (str): The filename from which to extract the label.

    Returns:
        str: The extracted label.
    """
    # Extract the base filename from the path
    base_filename = os.path.basename(filename)
    # Split the base filename by underscore and extract the first part
    return base_filename.split('_')[2]


def extract_pid_from_filename(filename: str) -> str:
    """
    Extract the photo ID from the filename.

    Args:
        filename (str): The filename from which to extract the photo ID.

    Returns:
        str: The extracted photo ID.
    """
    # Extract the base filename from the path
    base_filename = os.path.basename(filename)
    # Split the base filename by underscore and extract the first part
    return base_filename.split('_')[0]


@utils.timer
def process_images(rgb_fns: list[str], chm_fns: list[str], hyps_fns: list[str]) -> pd.DataFrame:
    """
    Process a set of image files to extract and combine features for analysis.

    Args:
        rgb_fns (list[str]): List of filenames for RGB images.
        chm_fns (list[str]): List of filenames for CHM images.
        hyps_fns (list[str]): List of filenames for hyperspectral images.

    Returns:
        pd.DataFrame: A DataFrame containing combined features, labels, and plot IDs for each image.
    """
    data = []
    for rgb_fn, chm_fn, hyps_fn in zip(rgb_fns, chm_fns, hyps_fns):
        logger.info(f'Processing {rgb_fn}...')
        label = extract_label_from_filename(rgb_fn)
        pid = extract_pid_from_filename(rgb_fn)
        rgb_img = utils.load_rgb_images([rgb_fn])[0]
        if ((label == 'unk') & (utils.calculate_zero_pixel_percentage(rgb_img) < 95)) | ((label != 'unk') & (utils.calculate_zero_pixel_percentage(rgb_img) < 99)):
            chm_img = utils.load_grayscale_arrays([chm_fn])[0]
            hyps_img = utils.load_hyperspectral_rasters([hyps_fn])[0]
            rgb_features = get_rgb_features(rgb_img)
            chm_features = get_chm_features(chm_img)
            hyps_features = get_hyperspectral_features(hyps_img)
            norm_combined_features = combine_features(rgb_features, hyps_features, chm_features)
            columns = ['pid', 'label'] + [f'f_{i}_{feature_type}' for feature_type, feature_set in [('rgb', rgb_features), ('hyps', hyps_features), ('chm', chm_features)] for i in range(len(feature_set))]
            image_data = [pid, label] + norm_combined_features.tolist()
            image_df = pd.DataFrame([image_data], columns=columns)
            data.append(image_df)
        else:
            pass
    df = pd.concat(data, ignore_index=True)
    logger.info(f'Final dataframe shape: {df.shape}')
    logger.info(f'Label counts:\n{df.label.value_counts()}')
    return df


@utils.timer
def main():
    conf = utils.load_config('base')
    rgb_fns = utils.get_filenames(conf.data.path_pri, "png", 'rgb')
    chm_fns = utils.get_filenames(conf.data.path_pri, "npy", 'chm')
    hyps_fns = utils.get_filenames(conf.data.path_pri, "tif", 'hyps')
    rgb_fns = sorted(rgb_fns)
    chm_fns = sorted(chm_fns)
    hyps_fns = sorted(hyps_fns)
    df = process_images(rgb_fns, chm_fns, hyps_fns)
    df = df.fillna(0.)
    df.to_csv(f"{conf.data.path_feat}{conf.data.fn_feat}", index=False)


if __name__ == '__main__':
    main()