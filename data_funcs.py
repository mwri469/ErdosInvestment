import pandas as pd
import numpy as np

# Path to the dataset
DATASET_PATH = "../pyanomaly-master/output/merge.pickle"

# Function to load the dataset and exclude unwanted columns
def load_data():
    # Load the pickle file
    df = pd.read_pickle(DATASET_PATH)
    
    # Ensure data is sorted by 'permno' and 'date'
    df.sort_values(by=["permno", "date"], inplace=True)
    
    # Columns to exclude from the training set
    exclude_columns = ["gvkey", "datadate", "primary", "exchcd", "ret", "exret", "rf"]
    
    # Filter the columns for predictors
    predictors = [col for col in df.columns if col not in exclude_columns]
    
    return df, predictors

# Function to split the data into training, validation, and testing sets
def split_data(df, predictors):
    # Define date ranges for the splits
    train_end_date = "2010-12-31"  # Example: Adjust as needed
    val_end_date = "2019-12-31"    # Example: Adjust as needed
    
    # Create the splits
    train_data = df[df["date"] <= train_end_date]
    val_data = df[(df["date"] > train_end_date) & (df["date"] <= val_end_date)]
    test_data = df[df["date"] > val_end_date]
    
    # Extract features (X) and target (y)
    X_train = train_data[predictors]
    y_train = train_data["ret"]  # Assuming 'ret' is the target variable
    
    X_val = val_data[predictors]
    y_val = val_data["ret"]
    
    X_test = test_data[predictors]
    y_test = test_data["ret"]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

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
