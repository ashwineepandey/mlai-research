import rasterio
import rasterio.plot
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
import numpy as np
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import log

logger = log.get_logger(__name__)

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


def sync_crs(gdf, rasterimg) -> bool: 
    if gdf.crs != rasterimg.crs:
        gdf = gdf.set_crs(str(rasterimg.crs))
    return gdf


def load_data(path_raw, path_int_rs, fn_ms, fn_rgb, fn_dsm, fn_dtm, fn_shp):
    imgs = {}
    # 10 bands of Advanced Camera image (multispectral)
    # 3 bands of Normal Camera image
    # 1 band of Digital Surface Model
    # 1 band of Digital Terrain Model
    imgs["ms"] = load_raster(f"{path_int_rs}{fn_ms}")
    imgs["rgb"] = load_raster(f"{path_int_rs}{fn_rgb}")
    imgs["dsm"] = load_raster(f"{path_int_rs}{fn_dsm}")
    imgs["dtm"] = load_raster(f"{path_int_rs}{fn_dtm}")
    # Load shapefile
    gdf = gpd.read_file(f"{path_raw}{fn_shp}")
    return imgs, gdf


def get_bbox(imgs: dict):
    box1 = box(*imgs["ms"].bounds)
    box2 = box(*imgs["rgb"].bounds)
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


def save_plot(fig, filename):
    fig.savefig(filename)

def plot_raster(gdf, rasterimg, show=True):
    fig, ax = plt.subplots(figsize = (20,20))
    rasterio.plot.show(rasterimg, ax=ax)
    gdf.plot(column='Species',
                   categorical=True,
                   legend=True,
                   cmap="Set2",
                   ax=ax,
            aspect=1)
    ax.set_title("Letaba Points Subset")
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.photoID):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
    if show == True:
        plt.show()
    return fig


def main():

    conf = utils.load_config("base")

    imgs, gdf = load_data(path_raw_dml, path_int_rs, fn_ms, fn_rgb, fn_dsm, fn_dtm, fn_shp)
    intersecting_box = get_bbox(imgs)
    target_bounds = intersecting_box.bounds
    rgb_clipped = clip_raster_to_bounds('rgb', path_int_cl, imgs['rgb'], target_bounds)
    ms_clipped = clip_raster_to_bounds('ms', path_int_cl, imgs['ms'], target_bounds)
    dsm_clipped = clip_raster_to_bounds('dsm', path_int_cl, imgs['dsm'], target_bounds)
    dtm_clipped = clip_raster_to_bounds('dtm', path_int_cl, imgs['dtm'], target_bounds)
    rgb_aligned = align_rasters('rgb', path_int_al, rgb_clipped, dsm_clipped)
    ms_aligned = align_rasters('ms', path_int_al, ms_clipped, dsm_clipped)
    clipped_gdf = clip_gdf(gdf, rgb_aligned.bounds)
    clipped_gdf[['photoID', 'Species','geometry']].reset_index(drop=True).to_file(f"{path_int}letaba_points.shp")
    fig_rgb = plot_raster(clipped_gdf, rgb_aligned, show=False)
    save_plot(fig_rgb, f"{path_int}rgb.png")

    fig_ms = plot_raster(clipped_gdf, ms_aligned, show=False)
    save_plot(fig_ms, f"{path_int}ms.png")

    fig_dsm = plot_raster(clipped_gdf, dsm_clipped, show=False)
    save_plot(fig_dsm, f"{path_int}dsm.png")

    fig_dtm = plot_raster(clipped_gdf, dtm_clipped, show=False)
    save_plot(fig_dtm, f"{path_int}dtm.png")