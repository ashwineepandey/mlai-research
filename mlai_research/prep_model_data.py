import os
import sys
sys.path.append('../mlai_research/')
import log
import utils
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


logger = log.get_logger(__name__)

def encode_labels(df, column):
    """
    Encode labels in the specified column of the dataframe.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the labels.
    column (str): The column name of the labels.

    Returns:
    df (pandas.DataFrame): The dataframe with encoded labels.
    """
    label_encoder = preprocessing.LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    return df


def split_data(df, test_size=0.1, val_size=0.1):
    """
    Split the dataframe into training, validation, and testing sets.

    Parameters:
    df (pandas.DataFrame): The dataframe to split.
    test_size (float): The proportion of the dataset to include in the test split.
    val_size (float): The proportion of the training set to include in the validation split.

    Returns:
    train_set (pandas.DataFrame): The training set.
    val_set (pandas.DataFrame): The validation set.
    test_set (pandas.DataFrame): The testing set.
    """
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=42)
    train_set, val_set = train_test_split(train_set, test_size=val_size, random_state=42)
    return train_set, val_set, test_set


def upsample_minorities(df, label_column):
    """
    Upsample the minority classes in the dataframe to match the majority class count.

    Parameters:
    df (pandas.DataFrame): The dataframe to upsample.
    label_column (str): The column name of the labels.

    Returns:
    df_upsampled (pandas.DataFrame): The dataframe with the minority classes upsampled.
    """
    # Get the count of the most frequent class
    majority_class_count = df[label_column].value_counts().max()
    
    # Separate the majority and minority classes
    df_majority = df[df[label_column] == df[label_column].value_counts().idxmax()]
    df_minorities = df[df[label_column] != df[label_column].value_counts().idxmax()]
    
    # List to hold the upsampled dataframes
    upsampled_list = [df_majority]
    
    # Upsample each minority class and add to the list
    for class_index in df_minorities[label_column].unique():
        df_class_minority = df_minorities[df_minorities[label_column] == class_index]
        df_class_upsampled = resample(df_class_minority, 
                                      replace=True, 
                                      n_samples=majority_class_count, 
                                      random_state=42)
        upsampled_list.append(df_class_upsampled)
    
    # Concatenate all upsampled minority class DataFrames with the majority class DataFrame
    df_upsampled = pd.concat(upsampled_list)
    
    return df_upsampled

@utils.timer
def main():
    conf = utils.load_config("base")
    df = pd.read_csv(f"{conf.data.path_feat}{conf.data.fn_feat}")
    df = encode_labels(df, 'label')
    train, val, test = split_data(df)
    train = upsample_minorities(train, 'label')
    X_train = train.drop('label', axis=1).values
    y_train = train['label'].values

    X_val = val.drop('label', axis=1).values
    y_val = val['label'].values

    X_test = test.drop('label', axis=1).values
    y_test = test['label'].values

    # Save as npz files
    np.savez(f"{conf.data.path_mi}{conf.data.fn_train}", X=X_train, y=y_train)
    np.savez(f"{conf.data.path_mi}{conf.data.fn_val}", X=X_val, y=y_val)
    np.savez(f"{conf.data.path_mi}{conf.data.fn_test}", X=X_test, y=y_test)


if __name__ == "__main__":
    main()