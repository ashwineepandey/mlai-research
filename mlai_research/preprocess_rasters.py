import sys
sys.path.append('../mlai_research/')
import log
import utils
import cv2
import rasterio
import rasterio.plot
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, mapping
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from PIL import Image

logger = log.get_logger(__name__)

def load_data(conf):
    imgs = {}
    # 10 bands of Advanced Camera image (multispectral)
    # 3 bands of Normal Camera image
    # 1 band of Digital Surface Model
    # 1 band of Digital Terrain Model
    imgs["hyps"] = utils.load_raster(f"{conf.data.path_base_hyps}{conf.data.fn_hyps}")
    imgs["rgba"] = utils.load_raster(f"{conf.data.path_base_rgba}{conf.data.fn_rgba}")
    imgs["dsm"] = utils.load_raster(f"{conf.data.path_base_dsm}{conf.data.fn_dsm}")
    imgs["dtm"] = utils.load_raster(f"{conf.data.path_base_dtm}{conf.data.fn_dtm}")
    # Load shapefile
    gdf = gpd.read_file(f"{conf.data.path_base_points}{conf.data.fn_shp_combined}")
    return imgs, gdf


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
def plot_raster(gdf, rasterimg, out_dir=None, fn=None, show=False, save=False):
    fig, ax = plt.subplots(figsize = (20,20))
    rasterio.plot.show(rasterimg, ax=ax)
    gdf.plot(column='Species',
                   categorical=True,
                   legend=True,
                   cmap="Set2",
                   ax=ax,
            aspect=1)
    ax.set_title("Letaba Points Subset")
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.pid):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
    if show == True:
        rasterio.plot.show(rasterimg, ax=ax)
    if save == True:
        utils.save_plot(fig, f"{out_dir}preprocessed_{fn}.png")
        logger.info(f"Saved plot to {out_dir}preprocessed_{fn}.png")


def process_shp(clipped_gdf, buffer=10):
    gdf_copy = clipped_gdf.copy()
    gdf_copy['buffer'] = gdf_copy.buffer(buffer)
    return gdf_copy


def save_as_png(image: np.ndarray, filename: str):
    """
    Saves the input image as a PNG file.

    Parameters:
    - image (numpy.ndarray): The input image.
    - filename (str): The output filename.
    """
    im = Image.fromarray((image * 255).astype(np.uint8))
    im.save(filename)


def save_cropped_tifs(path_int_cr_tif, pid, raster_type, label, out_image, out_meta, rgb=False, path_int_cr_imgs=None):
    with rasterio.open(f"{path_int_cr_tif}{pid}_{raster_type}_{label}.tif", "w", **out_meta) as dst:
        dst.write(out_image)
    
    if rgb:
        # Normalize and convert to RGB
        cropped_raster = rasterio.open(f"{path_int_cr_tif}{pid}_{raster_type}_{label}.tif")
        rgb_data_hwc = convert_to_rgb(cropped_raster)
        normalized_image = normalize_image(rgb_data_hwc)
        # Save as PNG
        save_as_png(normalized_image, f"{path_int_cr_imgs}{pid}_{raster_type}_{label}.png")


def crop_buffer(raster, polygon, path_pri, pid, raster_type, label, rgb=False, path_int_cr_imgs=None):
    geojson_polygon = mapping(polygon)
    out_image, out_transform = mask(raster, [geojson_polygon], crop=True)
    out_meta = raster.meta.copy()
    out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
    save_cropped_tifs(path_pri, pid, raster_type, label, out_image, out_meta, rgb, path_int_cr_imgs)

@utils.timer
def create_cropped_data(clipped_gdf, conf,
                     rgb, ms_aligned, chm):
    gdf_copy = process_shp(clipped_gdf, buffer=conf.preprocess.crop_buffer)
    for _, row in gdf_copy.iterrows():
        crop_buffer(rgb, row.buffer, conf.data.path_int_cr_tif, row.pid, 'rgb', row.Species, rgb=True, path_int_cr_imgs=conf.data.path_int_cr_img)
        crop_buffer(ms_aligned, row.buffer, conf.data.path_int_cr_tif, row.pid, 'hyps', row.Species)
        crop_buffer(chm, row.buffer, conf.data.path_int_cr_tif, row.pid, 'chm', row.Species)


def create_canopy_height_model(name, path_int, dsm, dtm):
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
    normalized_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    logger.info(f'Normalized image shape: {normalized_image.shape}')
    return normalized_image


def convert_to_rgb(rgba_aligned):
    # Read the raster bands directly into numpy arrays.
    rgba_data = rgba_aligned.read()
    rgb_data = rgba_data[:3, :, :]
    rgb_data_hwc = np.transpose(rgb_data, (1, 2, 0))
    return rgb_data_hwc


@utils.timer
def main():
    conf = utils.load_config("base")
    imgs, gdf = load_data(conf)

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

    plot_raster(clipped_gdf, rgba_clipped, 
                out_dir=conf.data.path_rep, fn=conf.data.fn_rgba, show=False, save=True)
    plot_raster(clipped_gdf, hyps_aligned, 
                out_dir=conf.data.path_rep, fn=conf.data.fn_hyps, show=False, save=True)
    plot_raster(clipped_gdf, dsm_aligned, 
                out_dir=conf.data.path_rep, fn=conf.data.fn_dsm, show=False, save=True)
    plot_raster(clipped_gdf, dtm_aligned, 
                out_dir=conf.data.path_rep, fn=conf.data.fn_dtm, show=False, save=True)

    chm = create_canopy_height_model('chm', conf.data.path_int_al, dsm_aligned, dtm_aligned)

    create_cropped_data(clipped_gdf, conf,
                     rgba_clipped, hyps_aligned, chm)


if __name__ == "__main__":
    main()