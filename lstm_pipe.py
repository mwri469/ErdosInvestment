import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from globals import *
from numba import njit, prange

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

    exclude_columns = ["gvkey", "datadate", "primary", "exchcd", "ret", "exret", "me"]
    exret = df.exret
    features = df.drop(columns=exclude_columns)

    # Build data for lstm
    placeholder_X_list = np.empty(features[:PAST].shape)
    placeholder_y_list = np.empty(exret[PAST:PAST+FUTURE].shape)
    X_arr = np.array([placeholder_X_list for _ in range(len(permno_set))])
    y=np.array([placeholder_y_list for _ in range(len(permno_set))])

    for j in tqdm(range(len(permno_set))):
        p = permno_set.pop()
        idxs = [idx for idx, val in enumerate(permno_ids) if val == p]
        temp_df = features.iloc[idxs].to_numpy()
        temp_exret = exret.iloc[idxs].to_numpy()
        
        for i in range(len(temp_df) - PAST - FUTURE + 1):
            X_arr[j] = temp_df[i:i+PAST].copy()
            y[j] = temp_exret[i+PAST:i+PAST+FUTURE].copy()

    return X_arr, y

@njit(parallel=True)
def imputed_df_to_data_numba(features_np, exret_np, permno_ids, unique_permnos, PAST, FUTURE):
    """
    Numba-accelerated version of imputed_df_to_data
    
    Parameters:
    -----------
    features_np : np.ndarray
        Numpy array of features
    exret_np : np.ndarray
        Numpy array of excess returns
    permno_ids : np.ndarray
        Array of permno IDs
    unique_permnos : np.ndarray
        Array of unique permno IDs
    PAST : int
        Number of past timesteps
    FUTURE : int
        Number of future timesteps
    
    Returns:
    --------
    X_arr : np.ndarray
        Features array
    y : np.ndarray
        Target returns array
    """
    # Preallocate output arrays
    X_arr = np.zeros((len(unique_permnos), features_np.shape[0] - PAST - FUTURE + 1, PAST, features_np.shape[1]), 
                     dtype=features_np.dtype)
    y = np.zeros((len(unique_permnos), features_np.shape[0] - PAST - FUTURE + 1, FUTURE), 
                 dtype=exret_np.dtype)
    
    # Process each unique permno
    for j in prange(len(unique_permnos)):
        permno = unique_permnos[j]
        
        # Find indices for current permno
        permno_mask = (permno_ids == permno)
        temp_features = features_np[permno_mask]
        temp_exret = exret_np[permno_mask]
        
        # Sliding window creation
        for i in range(len(temp_features) - PAST - FUTURE + 1):
            X_arr[j, i, :, :] = temp_features[i:i+PAST]
            y[j, i, :] = temp_exret[i+PAST:i+PAST+FUTURE]
    
    return X_arr, y

def replace_NaNs(X: np.array):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                # Replace NaNs
                if np.isnan(X[i,j,k]):
                    if i == 0:
                        X[i,j,k] = 0
                    else:
                        X[i,j,k] = X[i-1,j,k]

    return X

def impute_permno(df):
    """Imputes missing values within each permno group using the median.

    Parameters:
    -----------
    df: A Pandas DataFrame with a MultiIndex.

    Returns:
    --------
    df_imputed: A DataFrame with imputed values.
    """
    print('USING NEW IMPUTE METHOD')

    # Remove columns with high NaN percentage
    print('Removing columns with a NaN % > 30. . .\n')
    for col in tqdm(df.columns):
        percent_NaN = df[col].isna().mean()
        if percent_NaN > 0.3:
            # print(f'\n Deleting column {col} with NaN: {percent_NaN}%. . .\n')
            df.drop(col, axis=1, inplace=True)

    print('Imputing median for each permno. . .\n')
    # Get unique permno IDs
    permno_ids = df.index.get_level_values('permno').unique()

    # Iterate over permno IDs and impute missing values
    for permno in tqdm(permno_ids):
        df.loc[df.index.get_level_values('permno') == permno] = df.loc[
            df.index.get_level_values('permno') == permno
        ].fillna(df.loc[df.index.get_level_values('permno') == permno].median())

    return df

@njit
def impute_permno_numba(data, unique_permnos):
    """
    Numba-accelerated imputation function
    
    Parameters:
    -----------
    data : numpy.ndarray
        2D numpy array of data
    unique_permnos : numpy.ndarray
        Array of unique permno IDs
    
    Returns:
    --------
    numpy.ndarray
        Imputed data array
    """
    for permno in unique_permnos:
        # Create a mask for the current permno
        mask = data[:, 0] == permno
        
        # Get data for current permno
        permno_data = data[mask, 1:]
        
        # Compute column-wise median for current permno
        col_medians = np.nanmedian(permno_data, axis=0)
        
        # Fill NaNs with computed medians
        for col in range(permno_data.shape[1]):
            nan_mask = np.isnan(permno_data[:, col])
            permno_data[nan_mask, col] = col_medians[col]
        
        # Update original data
        data[mask, 1:] = permno_data
    
    return data

def imputed_df_to_data_optimized(df, PAST=10, FUTURE=1):
    """
    Optimized conversion of imputed DataFrame to LSTM-ready format
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    PAST : int, optional
        Number of past timesteps to use
    FUTURE : int, optional
        Number of future timesteps to predict
    
    Returns:
    --------
    X_arr : np.ndarray
        Features array
    y : np.ndarray
        Target returns array
    """
    # Exclude specified columns
    exclude_columns = ["gvkey", "datadate", "primary", "exchcd", "ret", "exret", "me"]
    
    # Prepare features and excess returns
    features = df.drop(columns=exclude_columns)
    exret = df.exret
    
    # Get indexes
    permno_ids = df.index.get_level_values('permno')
    
    # Get unique permnos
    unique_permnos = permno_ids.unique()
    
    # Convert to numpy for Numba processing
    features_np = features.to_numpy()
    exret_np = exret.to_numpy()
    permno_ids_np = permno_ids.to_numpy()
    
    # Process with Numba-accelerated function
    return imputed_df_to_data_numba(features_np, exret_np, permno_ids_np, unique_permnos, PAST, FUTURE)

def impute_permno_optimized(df):
    """
    Optimized imputation function using Numba
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to impute
    
    Returns:
    --------
    pandas.DataFrame
        Imputed DataFrame
    """
    # Remove columns with high NaN percentage
    print('Removing columns with a NaN % > 30. . .\n')
    for col in tqdm(df.columns):
        percent_NaN = df[col].isna().mean()
        if percent_NaN > 0.3:
            df.drop(col, axis=1, inplace=True)
    
    # Prepare data for Numba processing
    unique_permnos = df.index.get_level_values('permno').unique().values
    
    # Convert DataFrame to numpy for processing
    data_array = df.reset_index().values
    
    # Process with Numba-accelerated function
    print('\nImputing NaN values by permno id using median')
    imputed_array = impute_permno_numba(data_array, unique_permnos)
    
    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_array[:, 1:], 
                               columns=df.columns, 
                               index=df.index)
    
    return imputed_df

# Function to load the dataset and exclude unwanted columns
def load_data():
    print('\nLoading in data. . .')
    # Load the pickle file
    with open(FILE_PATH, 'rb') as f:
        obj = pkl.load(f)

    df = pd.DataFrame(obj)

    print('\nDatasets loaded')
    dates = df.index.get_level_values('date')
    print(f'Date range : {dates.min()} -> {dates.max()}')
    print(df.info(verbose=True,show_counts=True,max_cols=10))

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
