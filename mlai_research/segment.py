import os
import sys
sys.path.append('../mlai_research/')
import log
import utils
import cv2
import numpy as np
import imageio
import rasterio
from skimage.segmentation import felzenszwalb
from skimage.segmentation import mark_boundaries
import rasterio
import rasterio.plot
import matplotlib.pyplot as plt
from typing import Tuple, Union, List


logger = log.get_logger(__name__)

def load_rgb_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        images.append(img_rgb)
    return np.array(images)



def load_grayscale_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        images.append(img)
    return np.array(images)


def load_hyperspectral_images(image_paths):
    images = []
    for path in image_paths:
        with rasterio.open(path) as src:
            img = src.read()
            # Transpose the image to have channels last
            img = img.transpose((1, 2, 0))
            images.append(img)
    return images

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes the pixel values of the input image.

    Parameters:
    - image (numpy.ndarray): The input image.

    Returns:
    - numpy.ndarray: The normalized image.
    """
    # Normalize the image to the range [0, 255]
    normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return normalized_image


def create_center_segment_mask(segmentation: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    # Step 1: Identify the center point of the image
    center_point = (image_shape[0] // 2, image_shape[1] // 2)
    # Step 2: Identify the segment label at the center point
    center_segment_label = segmentation[center_point]
    # Step 3: Create a mask by comparing the segmentation array with the center segment label
    mask = segmentation == center_segment_label
    logger.info(f'Mask shape: {mask.shape}')
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Ensure mask is boolean
    mask = mask.astype(bool)

    # If the image has more than one channel, we need to adjust the mask to have the same number of channels
    if image.ndim > 2:
        # Expand dimensions of the mask to match the image
        mask = np.stack([mask]*image.shape[-1], axis=-1)

    # Apply the mask to the image
    masked_image = image * mask
    return masked_image


def plot_masked_images_segments(image_dict, save = False, out_dir = None, fn = None):
    num_images = len(image_dict['original'])
    
    # Set up the plot with num_images rows and 3 columns
    fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(12, 4*num_images))

    for i in range(num_images):
        # Plot the original image in the first column
        axes[i, 0].imshow(image_dict['original'][i])
        axes[i, 0].set_title('Original Image')

        # Plot the segmented image with boundaries in the second column
        axes[i, 1].imshow(mark_boundaries(image_dict['original'][i], image_dict['segmented'][i]))
        axes[i, 1].set_title('Segmented Image')

        # Plot the masked image in the third column
        axes[i, 2].imshow(image_dict['masked'][i])
        axes[i, 2].set_title('Masked Image')

    if save:
        utils.save_plot(fig, f"{out_dir}segmented_{fn}.png")
        logger.info(f"Saved plot to {out_dir}segmented_{fn}.png")
    else:
        plt.tight_layout()
        plt.show()


def plot_masked_images_gray(image_dict, save = False, out_dir = None, fn = None):
    num_images = len(image_dict['original'])
    
    # Set up the plot with num_images rows and 2 columns
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 4*num_images))

    for i in range(num_images):
        # Plot the original image in the first column
        axes[i, 0].imshow(image_dict['original'][i], cmap='gray')
        axes[i, 0].set_title('Original Image')

        # Plot the masked image in the second column
        axes[i, 1].imshow(image_dict['masked'][i], cmap='gray')
        axes[i, 1].set_title('Masked Image')

    if save:
        utils.save_plot(fig, f"{out_dir}segmented_{fn}.png")
        logger.info(f"Saved plot to {out_dir}segmented_{fn}.png")
    else:
        plt.tight_layout()
        plt.show()


def plot_masked_images_hyperspectral(image_dict, channels=[7, 4, 2], save = False, out_dir = None, fn = None):
    num_images = len(image_dict['original'])
    
    # Set up the plot with num_images rows and 2 columns
    fig, axes = plt.subplots(nrows=num_images, ncols=2, figsize=(8, 4*num_images))

    for i in range(num_images):
        # Select specified channels from the original image to represent red, green, and blue
        original_rgb = image_dict['original'][i][:, :, channels]
        # Normalize the selected channels to the range 0-1
        original_rgb = (original_rgb - original_rgb.min()) / (original_rgb.max() - original_rgb.min())
        
        # Plot the original image in the first column
        axes[i, 0].imshow(original_rgb)
        axes[i, 0].set_title('Original Image')

        # Select specified channels from the masked image to represent red, green, and blue
        masked_rgb = image_dict['masked'][i][:, :, channels]
        # Normalize the selected channels to the range 0-1
        masked_rgb = (masked_rgb - masked_rgb.min()) / (masked_rgb.max() - masked_rgb.min())
        
        # Plot the masked image in the second column
        axes[i, 1].imshow(masked_rgb)
        axes[i, 1].set_title('Masked Image')

    if save:
        utils.save_plot(fig, f"{out_dir}segmented_{fn}.png")
        logger.info(f"Saved plot to {out_dir}segmented_{fn}.png")
    else:
        plt.tight_layout()
        plt.show()


def segment_rgb(rgb_imgs):
    rgb_dct = {'original': [], 'segmented': [], 'masked': []}
    masks = []
    for rgb_img in rgb_imgs:
        # Step 1: Normalize the image
        normalized_image = normalize_image(rgb_img)

        # Step 2: Segment the normalized image
        segments = felzenszwalb(normalized_image, scale=100, sigma=0.5, min_size=50)

        # Step 3: Create a mask based on the center segment
        mask = create_center_segment_mask(segments, normalized_image.shape[:2])
        masks.append(mask)

        # Step 4: Apply the mask to the normalized image
        masked_image = apply_mask(normalized_image, mask)

        # Step 5: Plot the images
        rgb_dct['original'].append(normalized_image)
        rgb_dct['segmented'].append(segments)
        rgb_dct['masked'].append(masked_image)
    return rgb_dct, masks


def segment_grayscale(grayscale_imgs, masks):
    grayscale_dct = {'original': [], 'masked': []}
    for grayscale_img, mask in zip(grayscale_imgs, masks):
        masked_image = apply_mask(grayscale_img, mask)
        grayscale_dct['original'].append(grayscale_img)
        grayscale_dct['masked'].append(masked_image)
    return grayscale_dct




def save_segmented_images(image_dict, filenames, path, image_type):
    for i in range(len(image_dict['masked'])):
        # Get the filename without the extension
        base_filename = os.path.basename(filenames[i]).split('.')[0]
        # Create the new filename
        new_filename = os.path.join(path, base_filename + '_masked.' + image_type)
        
        if image_type in ['png', 'jpg']:
            # Save the image as a png or jpg file
            imageio.imwrite(new_filename, image_dict['masked'][i])
        elif image_type == 'tif':
            # Save the image as a tif file
            with rasterio.open(filenames[i]) as src:
                profile = src.profile
            profile.update(dtype=rasterio.float32)
            
            with rasterio.open(new_filename, 'w', **profile) as dst:
                # Transpose the dimensions of the image so that channels are the first dimension
                dst.write(image_dict['masked'][i].transpose((2, 0, 1)).astype(rasterio.float32))

@utils.timer
def main():
    conf = utils.load_config('base')
    rgb_fns = utils.get_filenames(conf.data.path_int_cr_img, "png", 'rgb')
    chm_fns = utils.get_filenames(conf.data.path_int_cr_img, "png", 'chm')
    hyps_fns = utils.get_filenames(conf.data.path_int_cr_tif, "tif", 'hyps')

    rgb_imgs = load_rgb_images(rgb_fns)
    chm_imgs = load_grayscale_images(chm_fns)
    hyps_imgs = load_hyperspectral_images(hyps_fns)

    rgb_dct, masks = segment_rgb(rgb_imgs)
    chm_dct = segment_grayscale(chm_imgs, masks)
    hyps_dct = segment_grayscale(hyps_imgs, masks)

    save_segmented_images(rgb_dct, rgb_fns, conf.data.path_pri, 'png')
    save_segmented_images(chm_dct, chm_fns, conf.data.path_pri, 'png')
    save_segmented_images(hyps_dct, hyps_fns, conf.data.path_pri, 'tif')

    plot_masked_images_segments(rgb_dct, save=True, out_dir=conf.data.path_rep, fn=conf.data.fn_rgba)
    plot_masked_images_gray(chm_dct, save=True, out_dir=conf.data.path_rep, fn=conf.data.fn_chm)
    plot_masked_images_hyperspectral(hyps_dct, save=True, out_dir=conf.data.path_rep, fn=conf.data.fn_hyps)


if __name__ == "__main__":
    main()