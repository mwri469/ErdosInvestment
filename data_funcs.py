import pandas as pd
import numpy as np
import pickle as pkl
from globals import *

# Function to load the dataset and exclude unwanted columns
def load_data():
    # Load the pickle file
    with open(FILE_PATH, 'rb') as f:
        obj = pkl.load(f)

    df = pd.DataFrame(obj)
    
    # Columns to exclude from the training set
    exclude_columns = ["gvkey", "datadate", "primary", "exchcd", "ret", "exret", "rf"]
    exret = df.exret
    features = df.drop(columns=exclude_columns)
    
    return features, exret

# Function to split the data into training, validation, and testing sets
def split_data(df, predictors):
    # Filter the DataFrame based on date ranges
    train_mask = (df.index.get_level_values('date') >= TRAINING_DATES[0]) & (df.index.get_level_values('date') < TRAINING_DATES[1])
    val_mask = (df.index.get_level_values('date') >= VALIDATION_DATES[0]) & (df.index.get_level_values('date') < VALIDATION_DATES[1])
    test_mask = ~(train_mask | val_mask)

    # Split the X data
    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    # Split y data
    train_y = predictors[train_mask]
    val_y = predictors[val_mask]
    test_y = predictors[test_mask]

    return (train_df, train_y), (val_df, val_y), (test_df, test_y)

# Functions to load each split
def load_training_data():
    df, predictors = load_data()
    (X_train, y_train), _, _ = split_data(df, predictors)
    return X_train, y_train

def load_val_data():
    df, predictors = load_data()
    _, (X_val, y_val), _ = split_data(df, predictors)
    return X_val, y_val

def load_test_data():
    df, predictors = load_data()
    _, _, (X_test, y_test) = split_data(df, predictors)
    return X_test, y_test
