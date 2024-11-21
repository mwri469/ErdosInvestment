import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from globals import *

def main():
    with open(FILE_PATH, 'rb') as f:
        obj = pkl.load(f)

    df = pd.DataFrame(obj)

def imputed_df_to_data(df):
    """
    This function takes in the imputed pd.DataFrame and converts it to a more appropriate format
    for NN training.
    NOTE: PAST, FUTURE dates are configured in globals
    ... keep at 1 if not going into LSTM

    Parameters:
    -----------
    ... df      : pd.DataFrame
    ...     -DataFrame with imputed values

    Returns:
    --------
    ... X_arr   : np.array
    ...     -numpy array of X values mapped at each index to a corresponding y-values
    ... y       : np.array
...         -numpy array of the corresponding y-values (exret)
    """
    # Get indexes
    dates = df.index.get_level_values('date')
    permno_ids = df.index.get_level_values('permno')

    # Count unique permnos
    count = 0
    permno_set = set()
    for p in permno_ids.to_list():
        if p not in permno_set:
            count += 1
            permno_set.add(p)

    exclude_columns = ["gvkey", "datadate", "primary", "exchcd", "ret", "exret", "rf", "me"]
    exret = df.exret
    features = df.drop(columns=exclude_columns)

    # Build data for lstm
    X_arr,y = np.empty(len(permno_set)), np.empty(len(permno_set))

    for j, p in tqdm(enumerate(permno_set)):
        idxs = [idx for idx, val in enumerate(permno_ids) if val == p]
        temp_df = features.iloc[idxs].to_numpy()
        temp_exret = exret.iloc[idxs].to_numpy()
        
        for i in range(len(temp_df) - PAST - FUTURE + 1):
            X_arr[j] = temp_df[i:i+PAST]
            y[j] = temp_exret[i+PAST:i+PAST+FUTURE]
        
def impute_permno(df):
    """Imputes missing values within each permno group using the median.

    Parameters:
    -----------
    ... df: A Pandas DataFrame with a MultiIndex.

    Returns:
    --------
    ... df_imputed: A DataFrame with imputed values.
    """

    # Group by permno and fill missing values with the median
    df_imputed = df.groupby('permno').transform(lambda x: x.fillna(x.median()))
    return df_imputed

# Function to load the dataset and exclude unwanted columns
def load_data():
    # Load the pickle file
    with open(FILE_PATH, 'rb') as f:
        obj = pkl.load(f)

    df = pd.DataFrame(obj)
    
    return df

# Function to split the data into training, validation, and testing sets
def split_data(df):
    # Filter the DataFrame based on date ranges
    train_mask = (df.index.get_level_values('date') >= TRAINING_DATES[0]) & (df.index.get_level_values('date') < TRAINING_DATES[1])
    val_mask = (df.index.get_level_values('date') >= VALIDATION_DATES[0]) & (df.index.get_level_values('date') < VALIDATION_DATES[1])
    test_mask = ~(train_mask | val_mask)

    # Split the X data
    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    return (train_df), (val_df), (test_df)

if __name__ == '__main__':
    main()
