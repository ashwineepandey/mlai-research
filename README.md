# Classification of Invasive Species



## Config

This configuration file (`base.conf`) is used to specify various parameters and file names required for the preprocessing and analysis of shapefiles and raster data.

### Structure

The file is divided into two main sections: `files` and `preprocess`.

#### `files` Section

This section defines the names of various input and output files used in the data processing pipeline.

- `fn_shp_raw1`: The name of the first raw shapefile to be processed.
- `fn_shp_raw2`: The name of the second raw shapefile to be processed.
- `fn_shp_combined`: The name of the output shapefile that will be created by combining `fn_shp_raw1` and `fn_shp_raw2`.
- `fn_feat`: The name of the output file where extracted features will be saved.
- `fn_train`: The name of the output file where the training dataset will be saved.
- `fn_test`: The name of the output file where the test dataset will be saved.
- `fn_val`: The name of the output file where the validation dataset will be saved.

#### `preprocess` Section

This section defines parameters used during the preprocessing stage.

- `crop_buffer`: The buffer size to use when cropping the data.



## Code

### Generate Random Points

#### Purpose


#### Usage

To execute the script:
`python gen_synthetic_points.py`


### Pre-process shape files

#### Purpose
The primary objective of this script is to preprocess shapefiles containing point data for different species and create a combined shapefile with relevant information for further analysis. The preprocessing involves checking CRS consistency, selecting and renaming columns, dropping unnecessary dimensions from geometric data, and filtering species labels.

#### Script Workflow

##### CRS Consistency Check

- **Function**: `check_crs(gdf1, gdf2) -> bool`
  - Ensures that the coordinate reference systems (CRS) of the two input shapefiles (`gdf1` and `gdf2`) match.
  - Returns `True` if CRS is consistent, otherwise logs an error message and returns `False`.

##### Load Shapefile Data

- **Function**: `load_shp_data(dir_path, filename1, filename2) -> gdf1, gdf2`
  - Loads two shapefiles (`filename1` and `filename2`) from the specified directory (`dir_path`) using GeoPandas.
  - Logs the shape of each loaded shapefile.

##### Column Subset Selection

- **Function**: `select_subset_cols(df, subset_cols) -> DataFrame`
  - Selects a subset of columns (`subset_cols`) from a given DataFrame (`df`).

##### Drop Z-Dimension from Geometry

- **Function**: `geom_drop_z_dim(df) -> DataFrame`
  - Drops the Z-dimension from the geometry of the input GeoDataFrame (`df`).
  - Utilizes Shapely library functions for geometric manipulation.

##### Process Shapefiles

- **Function**: `process_shp_files(gdf1, gdf2) -> fil_gdf1, fil_gdf2`
  - Applies column selection and geometric dimension reduction to the input GeoDataFrames (`gdf1` and `gdf2`).
  - Renames columns for consistency (tag to Species).

##### Filter Relevant Species

- **Function**: `filter_relevant_species(df) -> DataFrame`
  - Maps and filters species labels to a predefined set of relevant species.
  - Logs the filtered species value counts.

#### Main Execution

- **Function**: `main()`
  - The main function orchestrating the entire workflow.
  - Loads configuration settings, loads shapefiles, checks CRS consistency, processes shapefiles, combines GeoDataFrames, filters species, assigns unique identifiers (`pid`), and saves the resulting GeoDataFrame to a new shapefile.

#### Usage

To execute the script:
`python preprocess_shp.py`

#### Outputs
The script produces a combined shapefile (`fn_shp_combined`) containing the processed point data. The resulting shapefile is saved to the specified directory (`path_base_points`). Log messages provide information about the shape of loaded and combined GeoDataFrames, species value counts, and the saved shapefile path.

### Pre-process rasters

### Purpose
The primary objective of this script is to handle various raster data types (multispectral images, normal camera images, digital surface models, and digital terrain models) and shapefiles. The script facilitates the preprocessing of these data types to prepare them for further analysis (species identification).

#### Script Workflow

##### Load Data

- **Function**: `load_data(conf)`
  - Loads raster data and shapefiles based on configuration settings.
  - Handles multispectral images, normal camera images, digital surface models, and digital terrain models.

##### Get Bounding Box

- **Function**: `get_bbox(imgs: dict)`
  - Calculates the intersection bounding box of all input rasters.

##### Clip Raster to Bounds

- **Function**: `clip_raster_to_bounds(name, path_int, raster, target_bounds)`
  - Clips a raster to the specified target bounds and saves the clipped raster.

##### Align Rasters

- **Function**: `align_rasters(name, path_int, src_raster, ref_raster)`
  - Aligns a source raster to the reference raster's grid and saves the aligned raster.

##### Clip GeoDataFrame

- **Function**: `clip_gdf(gdf, bounds)`
  - Clips a GeoDataFrame to the specified bounds and logs the species count.

##### Plot Raster

- **Function**: `plot_raster(gdf, rasterimg, out_dir=None, fn=None, show=False, save=False)`
  - Plots the raster and GeoDataFrame overlay and optionally saves the plot.

##### Process Shapefiles

- **Function**: `process_shp(clipped_gdf, buffer=10)`
  - Processes the clipped GeoDataFrame by applying a buffer to each geometry.

##### Save as PNG

- **Function**: `save_as_png(image: np.ndarray, filename: str)`
  - Saves an image array as a PNG file.

##### Create Cropped Data

- **Function**: `create_cropped_data(clipped_gdf, conf, rgb, ms_aligned, dsm_aligned, dtm_aligned)`
  - Creates cropped data from the aligned rasters and clipped GeoDataFrame, and saves the results.

#### Main Execution

- **Function**: `main()`
  - The main function orchestrating the entire workflow.
  - Loads configuration settings, loads data, gets bounding boxes, clips rasters, aligns rasters, clips GeoDataFrame, plots rasters, processes shapefiles, and creates cropped data.


`python preprocess_rasters.py`

### Segment cropped rasters

### Purpose
This script is designed to handle various image types (RGB, grayscale, and hyperspectral). It performs image segmentation and masking, and then saves and plots the results.

#### Script Workflow

##### Load RGB Images

- **Function**: `load_rgb_images(image_paths)`
  - Loads RGB images from the specified paths and returns them as a numpy array.

##### Load Grayscale Images

- **Function**: `load_grayscale_images(image_paths)`
  - Loads grayscale images from the specified paths and returns them as a numpy array.

##### Load Hyperspectral Images

- **Function**: `load_hyperspectral_images(image_paths)`
  - Loads hyperspectral images from the specified paths and returns them as a list.

##### Normalize Image

- **Function**: `normalize_image(image: np.ndarray) -> np.ndarray`
  - Normalizes the pixel values of the input image and returns the normalized image.

##### Create Center Segment Mask

- **Function**: `create_center_segment_mask(segmentation: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray`
  - Creates a mask based on the center segment of the input image and returns the mask.

##### Apply Mask

- **Function**: `apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray`
  - Applies a mask to the input image and returns the masked image.

##### Plot Masked Images Segments

- **Function**: `plot_masked_images_segments(image_dict, save = False, out_dir = None, fn = None)`
  - Plots the original, segmented, and masked images and optionally saves the plot.

##### Plot Masked Images Gray

- **Function**: `plot_masked_images_gray(image_dict, save = False, out_dir = None, fn = None)`
  - Plots the original and masked grayscale images and optionally saves the plot.

##### Plot Masked Images Hyperspectral

- **Function**: `plot_masked_images_hyperspectral(image_dict, channels=[7, 4, 2], save = False, out_dir = None, fn = None)`
  - Plots the original and masked hyperspectral images and optionally saves the plot.

##### Segment RGB

- **Function**: `segment_rgb(rgb_imgs)`
  - Performs segmentation and masking on RGB images and returns a dictionary of the original, segmented, and masked images, as well as the masks.

##### Segment Grayscale

- **Function**: `segment_grayscale(grayscale_imgs, masks)`
  - Performs masking on grayscale images and returns a dictionary of the original and masked images.

##### Save Segmented Images

- **Function**: `save_segmented_images(image_dict, filenames, path, image_type)`
  - Saves the masked images to the specified path with the specified image type.

##### Main Function

- **Function**: `main()`
  - Loads the images, performs segmentation and masking, saves the segmented images, and plots the results.

### Usage
`python segment.py`

### Extract Features
`python extract_features.py`

Purpose: This script is used to extract features from the segmented images. It includes calculating various features for each segment and saving the features to a file.

5. Prepare Modelling Data
`python prep_model_data.py`

Purpose: This script is used to prepare the data for modeling. It includes loading the features, splitting the data into training and test sets, and saving the prepared data to disk.


Purpose: 

*** 
To explore the models - I will use a jupyter notebook instead of scripts so that I can experiment.
Refer to `modelling-{num}.ipynb` under `notebooks` dir.
*** 

6. Train model
`python train.py`

Purpose: This script trains the final model on the dataset provided and runs an evaluation on the test set.