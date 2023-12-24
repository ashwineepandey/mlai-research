import sys
sys.path.append('../mlai_research/')
import log
import utils
import rasterio
import rasterio.plot
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Tuple, Union, List


logger = log.get_logger(__name__)

def plot_raster(gdf, rasterimg, path):
    fig, ax = plt.subplots(figsize = (20,20))
    rasterio.plot.show(rasterimg, ax=ax)
    gdf.plot(column='Species',
                   categorical=True,
                   legend=True,
                   # markersize=45,
                   cmap="Set2",
                   ax=ax,
            aspect=1)
    ax.set_title("Letaba Random (Generated) Points")
    for x, y, label in zip(gdf.geometry.x, gdf.geometry.y, gdf.pid):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")
    plt.savefig(f"{path}synthetic_points.png")


def load_raster(fn):
    with rasterio.open(fn) as ds:
        bounds = ds.bounds
        transform = ds.transform
        array = ds.read(1)
        crs = ds.crs
    return bounds, transform, array, crs


def gen_random_points(num_points, bounds, seed=42):
    if seed is not None:
        np.random.seed(seed)
    minx, miny, maxx, maxy = bounds
    xs = np.random.uniform(minx, maxx, num_points)
    ys = np.random.uniform(miny, maxy, num_points)
    points = [Point(x, y) for x, y in zip(xs, ys)]
    return points


def filter_valid_points(points, transform, array):
    # Filter out points that fall within the no-data regions of the raster
    valid_points = []
    for point in points:
        # Convert the point's coordinates to row and column indices
        row, col = rasterio.transform.rowcol(transform, point.x, point.y)
        
        # Check if the point falls within the bounds of the array
        if 0 <= row < array.shape[0] and 0 <= col < array.shape[1]:
            # Check if the corresponding pixel value in the raster is not a no-data value
            if array[row, col] != 0:
                valid_points.append(point)
    return valid_points


def main():
    conf = utils.load_config("base")
    logger.info("Generating synthetic points")

    # Load the raster
    raster = utils.load_raster(f"{conf.data.path_base_rgba}{conf.data.fn_rgba}")
    
    points = gen_random_points(1000, raster.bounds)
    valid_points = filter_valid_points(points, raster.transform, raster.read(1))
    logger.info(f"Generated {len(valid_points)} valid points")

    # Convert the points to a GeoDataFrame
    gdf = gpd.GeoDataFrame({"geometry": valid_points})
    gdf.crs = raster.crs
    gdf["Species"] = "unk"
    gdf['pid'] = np.arange(1000+0, 1000+len(gdf))


    # Save the points to a shapefile
    gdf.to_file(f"{conf.data.path_base_points}{conf.data.fn_shp_generated}", driver='ESRI Shapefile')
    logger.info(f"GeoDF saved to {conf.data.path_base_points}{conf.data.fn_shp_generated}")


    # Plot the points on top of the raster
    plot_raster(gdf, raster, conf.data.path_rep)


if __name__ == "__main__":
    main()