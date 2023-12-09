import sys
sys.path.append('../mlai_research/')
import log
import utils
import geopandas as gpd
from shapely import wkb
import pandas as pd

logger = log.get_logger(__name__)

def check_crs(gdf1, gdf2) -> bool: 
    if gdf1.crs == gdf2.crs:
        return True
    else:
        print("Error: The CRS for the shape files does not match.")
        return False
    
    
def load_shp_data(dir_path, filename1, filename2):
    gdf1 = gpd.read_file(f"{dir_path}{filename1}") # import shapefile using geopandas
    logger.info(f"{filename1} Shape: {gdf1.shape}")
    gdf2 = gpd.read_file(f"{dir_path}{filename2}") # import shapefile using geopandas
    logger.info(f"{filename2} Shape: {gdf2.shape}")
    return gdf1, gdf2


def select_subset_cols(df, subset_cols):
    return df[subset_cols]


def geom_drop_z_dim(df):
    _drop_z = lambda geom: wkb.loads(wkb.dumps(geom, output_dimension=2))
    df.geometry = df.geometry.transform(_drop_z)
    return df


def process_shp_files(gdf1, gdf2):
    fil_gdf1 = select_subset_cols(gdf1, subset_cols=['Species', 'geometry'])
    fil_gdf1 = geom_drop_z_dim(fil_gdf1)
    fil_gdf2 = select_subset_cols(gdf2, subset_cols=['tag', 'geometry'])
    fil_gdf2 = fil_gdf2.rename(columns={'tag': 'Species'})
    return fil_gdf1, fil_gdf2


def filter_relevant_species(df):
    # logger.info(f"Species value counts (original label): {df['Species'].value_counts()}")
    species_map = {"Xanthium strumarium" : "Xanthium",
              "Datura stramonium": "Datura",
               "Xanthium": "Xanthium",
               "Datura": "Datura"}
    df['Species'] = df['Species'].map(species_map).fillna('Other')
    logger.info(f'Filtered Species value counts: {df.Species.value_counts()}')
    return df


@utils.timer
def main():
    conf = utils.load_config("base")
    gdf1, gdf2 = load_shp_data(f"{conf.data.path_raw}classification_points/", conf.data.fn_shp_raw1, conf.data.fn_shp_raw2)
    if check_crs(gdf1, gdf2):
        fil_gdf1, fil_gdf2 = process_shp_files(gdf1, gdf2)
        comb_gdf = pd.concat([fil_gdf1, fil_gdf2]).reset_index(drop=True)
        logger.info(f"Combined Shape: {comb_gdf.shape}")
        fil_comb_gdf = filter_relevant_species(comb_gdf)
        fil_comb_gdf['pid'] = list(range(len(fil_comb_gdf)))
        fil_comb_gdf.to_file(f"{conf.data.path_base_points}{conf.data.fn_shp_combined}", driver='ESRI Shapefile')
        logger.info(f"GeoDF saved to {conf.data.path_base_points}{conf.data.fn_shp_combined}")


if __name__ == "__main__":
    main()