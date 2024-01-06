import sys
sys.path.append('../mlai_research/')
import log
import utils
import rasterio
import rasterio.plot
from rasterio.io import DatasetReader
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, mapping, Polygon
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Tuple, Union, List

logger = log.get_logger(__name__)

def load_shp_files(conf):
    gdf1 = gpd.read_file(f"{conf.data.path_base_points}{conf.data.fn_shp_combined}")
    gdf2 = gpd.read_file(f"{conf.data.path_base_points}{conf.data.fn_shp_generated}")

    # combine the two shapefiles
    gdf = gdf1.append(gdf2, ignore_index=True)
    return gdf


def load_rasters(conf):
    imgs = {}
    # 10 bands of Advanced Camera image (multispectral)
    # 3 bands of Normal Camera image
    # 1 band of Digital Surface Model
    # 1 band of Digital Terrain Model
    imgs["hyps"] = utils.load_raster(f"{conf.data.path_base_hyps}{conf.data.fn_hyps}")
    imgs["rgba"] = utils.load_raster(f"{conf.data.path_base_rgba}{conf.data.fn_rgba}")
    imgs["dsm"] = utils.load_raster(f"{conf.data.path_base_dsm}{conf.data.fn_dsm}")
    imgs["dtm"] = utils.load_raster(f"{conf.data.path_base_dtm}{conf.data.fn_dtm}")
    return imgs


def get_bbox(imgs: dict):
    box1 = box(*imgs["hyps"].bounds)
    box2 = box(*imgs["rgba"].bounds)
    box3 = box(*imgs["dsm"].bounds)
    box4 = box(*imgs["dtm"].bounds)

    intersect = box1.intersection(box2)
    intersect = intersect.intersection(box3)
    intersect = intersect.intersection(box4)
    return intersect


def clip_raster_to_bounds(name, path_int, raster, target_bounds):
    # Create a polygon from the target bounds
    target_polygon = box(*target_bounds)

    # Clip the raster using the target polygon
    clipped_data, clipped_transform = mask(raster, [target_polygon], crop=True)

    # Update the raster metadata
    clipped_meta = raster.meta.copy()
    clipped_meta.update({
        'transform': clipped_transform,
        'height': clipped_data.shape[1],
        'width': clipped_data.shape[2]
    })

    # Create a new raster dataset with the clipped data and metadata
    with rasterio.open(f"{path_int}{name}.tif", "w", **clipped_meta) as dst:
        dst.write(clipped_data)

    # Return the clipped raster dataset
    clipped_raster = rasterio.open(f"{path_int}{name}.tif")
    logger.info(f"{clipped_raster.transform}")
    return clipped_raster


def align_rasters(name, path_int, src_raster, ref_raster):
    profile = ref_raster.profile.copy()
    aligned_data = []

    for i in range(1, src_raster.count + 1):
        src_data = src_raster.read(i)
        ref_data = np.empty((ref_raster.height, ref_raster.width), dtype=src_data.dtype)
        reproject(
            src_data,
            ref_data,
            src_transform=src_raster.transform,
            src_crs=src_raster.crs,
            dst_transform=ref_raster.transform,
            dst_crs=ref_raster.crs,
            resampling=Resampling.nearest
        )
        aligned_data.append(ref_data)

    profile.update(count=len(aligned_data))

    with rasterio.open(f"{path_int}{name}.tif", "w", **profile) as dst:
        for i, data in enumerate(aligned_data, start=1):
            dst.write(data, i)

    aligned_raster = rasterio.open(f"{path_int}{name}.tif")
    logger.info(f"{aligned_raster.transform}")
    return aligned_raster


def clip_gdf(gdf, bounds):
    clipped_gdf = gdf[
        (gdf.geometry.x > bounds.left) & 
        (gdf.geometry.x < bounds.right) &
        (gdf.geometry.y > bounds.bottom) &
        (gdf.geometry.y < bounds.top)
    ]
    logger.info(f"Original gdf: {gdf.shape}")
    logger.info(f"Clipped gdf: {clipped_gdf.shape}")
    logger.info(f"Species split: {clipped_gdf.Species.value_counts()}")
    return clipped_gdf

@utils.timer
def plot_raster(gdf, rasterimg, raster_type, out_dir=None, fn=None, show=False):
    fig, ax = plt.subplots(figsize = (20,20))
    
    if raster_type == 'rgba':
        cmap = "viridis"
    else:
        cmap = "terrain"
    rasterio.plot.show(rasterimg, ax=ax, cmap=cmap)
    gdf.plot(column='Species',
                   categorical=True,
                   legend=True,
                   cmap="Set2",
                   ax=ax,
            aspect=1)
    ax.set_title("Letaba Points (Area of Interest 1)")
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.pid):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
    if show == True:
        rasterio.plot.show(rasterimg, ax=ax, cmap=cmap)
    else:
        # rasterio.plot.show(rasterimg, ax=ax, cmap=cmap)
        utils.save_plot(fig, f"{out_dir}preprocessed_{fn}.png")
        logger.info(f"Saved plot to {out_dir}preprocessed_{fn}.png")


def create_buffer(clipped_gdf, buffer=10):
    gdf_copy = clipped_gdf.copy()
    gdf_copy['buffer'] = gdf_copy.buffer(buffer, cap_style=3)
    return gdf_copy


def save_crop(path_int_cr_tif, pid, raster_type, label, out_image, out_meta):
    # Save the resized raster data
    with rasterio.open(f"{path_int_cr_tif}{pid}_{raster_type}_{label}.tif", "w", **out_meta) as dst:
        dst.write(out_image)

def save_crop_chm(path_int_cr_tifs: str, pid: str, raster_type: str, label: str, chm: np.ndarray) -> None:
    """
    Saves the canopy height model (CHM) data to a .npy file.
    
    Parameters:
    path_int_cr_tifs (str): The directory path where the output .npy file will be saved.
    pid (str): The identifier for the plot.
    raster_type (str): The type of the raster file.
    label (str): The label for the raster file.
    chm (np.ndarray): The CHM numpy array.
    """
    np.save(f"{path_int_cr_tifs}{pid}_{raster_type}_{label}.npy", chm)


def crop_buffer(raster: DatasetReader, polygon: Polygon, target_size=(87, 87)) -> Tuple[np.ndarray, dict]:
    """
    Crops a raster based on a polygon and returns the cropped image and its metadata.
    
    Parameters:
    raster (DatasetReader): The raster to be cropped.
    polygon (Polygon): The polygon used for cropping.
    target_size (tuple): The target size of the cropped image.

    Returns:
    Tuple: A tuple containing the cropped image and its metadata.
    """
    geojson_polygon = mapping(polygon)
    out_image, out_transform = mask(raster, [geojson_polygon], crop=True)
    
    # Crop the image if it's larger than the target size
    out_image = out_image[:, :target_size[0], :target_size[1]]
    
    # Update the metadata
    out_meta = raster.meta.copy()
    out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
    
    return out_image, out_meta

@utils.timer
def create_cropped_data(clipped_gdf, conf,
                     rgb, ms_aligned, dsm_aligned, chm):
    gdf_copy = create_buffer(clipped_gdf, buffer=conf.preprocess.crop_buffer)
    for _, row in gdf_copy.iterrows():

        #RGB
        rgb_img, rgb_img_meta = crop_buffer(rgb, row.buffer)
        save_crop(conf.data.path_int_cr_tif, row.pid, 'rgb', row.Species, rgb_img, rgb_img_meta)
        
        #Hyps
        hyps_img, hyps_img_meta = crop_buffer(ms_aligned, row.buffer)
        save_crop(conf.data.path_int_cr_tif, row.pid, 'hyps', row.Species, hyps_img, hyps_img_meta)

        # DSM
        dsm_img, _ = crop_buffer(dsm_aligned, row.buffer)
        save_crop_chm(conf.data.path_int_cr_tif, row.pid, 'dsm', row.Species, dsm_img)
        
        # CHM
        chm_img, _ = crop_buffer(chm, row.buffer)
        save_crop_chm(conf.data.path_int_cr_tif, row.pid, 'chm', row.Species, chm_img)


def create_canopy_height_model(name: str, path_int: str, dsm: DatasetReader, dtm: DatasetReader) -> DatasetReader:
    """
    Creates a canopy height model (CHM) by subtracting the digital terrain model (DTM) from the digital surface model (DSM).
    
    Parameters:
    name (str): The name of the output CHM file.
    path_int (str): The directory path where the output CHM file will be saved.
    dsm (DatasetReader): The DSM raster file.
    dtm (DatasetReader): The DTM raster file.

    Returns:
    DatasetReader: The CHM raster file.
    """
    dsm_data = dsm.read(1)
    dtm_data = dtm.read(1)
    chm_data = dsm_data - dtm_data
    with rasterio.open(f'{path_int}{name}.tif', 'w', **dsm.profile) as dst:
        dst.write(chm_data, 1)
    
    chm = rasterio.open(f'{path_int}{name}.tif')
    return chm

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalizes the pixel values of the input image.

    Parameters:
    - image (numpy.ndarray): The input image.

    Returns:
    - numpy.ndarray: The normalized image.
    """
    # Normalize the image to the range [0, 255]
    # normalized_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # return normalized_image
    normalized_image = ((image - np.min(image)) / (np.max(image) - np.min(image))) * 255
    return normalized_image.astype(np.uint8)


@utils.timer
def main():
    conf = utils.load_config("base")
    imgs = load_rasters(conf)
    gdf = load_shp_files(conf)

    intersecting_box = get_bbox(imgs)
    target_bounds = intersecting_box.bounds
    
    rgba_clipped = clip_raster_to_bounds('rgba', conf.data.path_int_cl, imgs['rgba'], target_bounds)
    hyps_clipped = clip_raster_to_bounds('hyps', conf.data.path_int_cl, imgs['hyps'], target_bounds)
    dsm_clipped = clip_raster_to_bounds('dsm', conf.data.path_int_cl, imgs['dsm'], target_bounds)
    dtm_clipped = clip_raster_to_bounds('dtm', conf.data.path_int_cl, imgs['dtm'], target_bounds)

    hyps_aligned = align_rasters('hyps', conf.data.path_int_al, hyps_clipped, rgba_clipped)
    dsm_aligned = align_rasters('dsm', conf.data.path_int_al, dsm_clipped, rgba_clipped)
    dtm_aligned = align_rasters('dtm', conf.data.path_int_al, dtm_clipped, rgba_clipped)

    clipped_gdf = clip_gdf(gdf, rgba_clipped.bounds)

    plot_raster(clipped_gdf, rgba_clipped, 'rgba',
                out_dir=conf.data.path_rep, fn=conf.data.fn_rgba, show=False)
    plot_raster(clipped_gdf, hyps_aligned, 'hyps',
            out_dir=conf.data.path_rep, fn=conf.data.fn_hyps, show=False)
    plot_raster(clipped_gdf, dsm_aligned, 'dsm',
                out_dir=conf.data.path_rep, fn=conf.data.fn_dsm, show=False)
    plot_raster(clipped_gdf, dtm_aligned, 'dtm',
            out_dir=conf.data.path_rep, fn=conf.data.fn_dtm, show=False)
    
    chm = create_canopy_height_model('chm', conf.data.path_int_al, dsm_aligned, dtm_aligned)

    create_cropped_data(clipped_gdf, conf,
                     rgba_clipped, hyps_aligned, dsm_aligned, chm)


if __name__ == "__main__":
    main()



    # def save_as_png(image: np.ndarray, filename: str):
#     """
#     Saves the input image as a PNG file.

#     Parameters:
#     - image (numpy.ndarray): The input image.
#     - filename (str): The output filename.
#     """
#     im = Image.fromarray((image * 255).astype(np.uint8))
#     im.save(filename)


# def save_crop_rgb(path_int_cr_tif, pid, raster_type, label, out_image, out_meta, path_int_cr_imgs):
#     resized_img = np.resize(out_image, (out_image.shape[0], 87, 87))

#     # Update the metadata with the new dimensions
#     out_meta.update({"height": 87, "width": 87})

#     logger.info(f'Resized RGB image shape: {resized_img.shape}')
    
#     with rasterio.open(f"{path_int_cr_tif}{pid}_{raster_type}_{label}.tif", "w", **out_meta) as dst:
#         dst.write(out_image)
    
    # # Normalize and convert to RGB
    # cropped_raster = rasterio.open(f"{path_int_cr_tif}{pid}_{raster_type}_{label}.tif")
    # rgb_data_hwc = convert_to_rgb(cropped_raster)
    # # normalized_image = normalize_image(rgb_data_hwc)
    # resized_img = crop_image(rgb_data_hwc, (87, 87))
    # # Save as PNG
    # logger.info(f'Resized RGB image shape: {resized_img.shape}')
    # save_as_png(resized_img, f"{path_int_cr_imgs}{pid}_{raster_type}_{label}.png")


# def equalize_histogram(image: np.ndarray) -> np.ndarray:
#     """
#     Equalizes the histogram of the input image.

#     Parameters:
#     - image (numpy.ndarray): The input image.

#     Returns:
#     - numpy.ndarray: The image with equalized histogram.
#     """
#     # Normalize the image to 0-1 range
#     # image_norm = (image - np.min(image)) / (np.max(image) - np.min(image))

#     # # Convert the normalized image to 8-bit
#     # image_8bit = np.uint8(image_norm * 255)

#     image_8bit = normalize_image(image)

#     # Flatten the image into 1D array
#     image_flattened = image_8bit.flatten()

#     # Perform histogram equalization
#     equalized_image = cv2.equalizeHist(image_flattened)

#     # Reshape the equalized image back to the original shape
#     equalized_image = equalized_image.reshape(image.shape)

#     return equalized_image



# def convert_to_rgb(rgba_aligned):
#     # Read the raster bands directly into numpy arrays.
#     rgba_data = rgba_aligned.read()
#     rgb_data = rgba_data[:3, :, :]
#     rgb_data_hwc = np.transpose(rgb_data, (1, 2, 0))
#     return rgb_data_hwc