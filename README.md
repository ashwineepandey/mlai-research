# Classification of Invasive Species

### How to run the scripts:

1. Pre-process shape files
`python preprocess_shp.py`

Purpose: Combine differing shape files provided into a combined file with only the relevant columns. 

2. Pre-process rasters
`python preprocess_rasters.py`

Purpose: This script is used to load and preprocess the raster images. It includes loading RGB, grayscale, and hyperspectral images, applying masks to the images, and saving the masked images to disk.

3. Segment cropped rasters
`python segment.py`

Purpose: This script is used to segment the cropped raster images. It includes applying segmentation algorithms to the images and saving the segmented images to disk.

4. Extract Features
`python extract_features.py`

Purpose: This script is used to extract features from the segmented images. It includes calculating various features for each segment and saving the features to a file.

5. Prepare Modelling Data
`python prep_model_data.py`

Purpose: This script is used to prepare the data for modeling. It includes loading the features, splitting the data into training and test sets, and saving the prepared data to disk.


Purpose: 

*** 
To explore the models - I will use a jupyter notebook instead of scripts so that I can experiment.
*** 

6. Train model
`python train.py`

Purpose: This script trains the final model on the dataset provided and runs an evaluation on the test set.

