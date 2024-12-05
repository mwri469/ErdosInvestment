
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from globals import *
from numba import njit, prange
import time

def main():
    with open(FILE_PATH, 'rb') as f:
        obj = pkl.load(f)

    df = pd.DataFrame(obj)

@njit()
def impute_permno_numba(data_flat: np.ndarray, unique_permnos: np.ndarray, n_cols: int) -> np.ndarray:
    """
    Numba-accelerated imputation function for flattened data
    
    Parameters:
    -----------
    data_flat : numpy.ndarray
        1D flattened array of data
    unique_permnos : numpy.ndarray
        Array of unique permno IDs
    n_cols : int
        Number of columns in the original 2D array
    
    Returns:
    --------
    numpy.ndarray
        Imputed data array
    """
    # Reshape the 1D array back to 2D
    data = data_flat.reshape((-1, n_cols))
    
    for idx in range(len(unique_permnos)):
        # Create a mask for the current permno
        permno = unique_permnos[idx]
        mask = data[:, 0] == permno
        
        # Get data for current permno (exclude the first column which is permno)
        permno_data = data[mask, 1:]
        
        # Compute column-wise median for current permno
        col_medians = np.zeros(permno_data.shape[1])
        for col in range(permno_data.shape[1]):
            col_values = permno_data[:, col]
            valid_values = col_values[~np.isnan(col_values)]
            if len(valid_values) > 0:
                col_medians[col] = np.median(valid_values)
        
        # Fill NaNs with computed medians
        for col in range(permno_data.shape[1]):
            nan_mask = np.isnan(permno_data[:, col])
            permno_data[nan_mask, col] = col_medians[col]
        
        # Update original data
        data[mask, 1:] = permno_data
    
    return data

def impute_permno_optimised(df):
    """
    Optimized imputation function using a memory-efficient approach
    
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
    # for col in tqdm(df.columns):
    #     percent_NaN = df[col].isna().mean()
    #     if percent_NaN > 0.3:
    #         df.drop(col, axis=1, inplace=True)
    
    print('\nColumns Removed, preparing data for optimised df imputation. . .')
    # Prepare data for Numba processing
    unique_permnos = df.index.get_level_values('permno').unique().values.to_numpy()
    
    # Convert to a memory-efficient format
    # Create a flat array with permno as the first column
    n_cols = len(df.columns) + 1  # +1 for permno
    
    # Preallocate a flat array
    data_flat = np.zeros(len(df) * n_cols, dtype=np.float64)
    
    print('\nFilling flat array')
    # Fill the flat array
    for i, (index, row) in tqdm(enumerate(df.iterrows())):
        start = i * n_cols
        data_flat[start] = index[1]  # permno
        data_flat[start+1:start+n_cols] = row.to_numpy()
    df_cols = df.columns
    df_names = df.index.names
    df_index = df.index
    del df

    print('\nBegin imputing data. . .')
    start = time.perf_counter()
    # Process with Numba-accelerated function
    imputed_flat = impute_permno_numba(data_flat, unique_permnos, n_cols)
    # Count elapsed time
    elapsed = time.perf_counter()-start
    print(f'Finished imputation in {elapsed}s')
    
    # Reconstruct DataFrame
    imputed_df = pd.DataFrame(
        imputed_flat[:, 1:], 
        columns=df_cols, 
        index=pd.MultiIndex.from_tuples(
            [(idx[0], imputed_flat[i, 0]) for i, idx in enumerate(df_index)],
            names=df_names
        )
    )
    
    return imputed_df

@njit(parallel=True)
def imputed_df_to_data_numba(features_np, exret_np, permno_ids, unique_permnos, PAST, FUTURE):
    """
    Numba-accelerated version of imputed_df_to_data with memory-efficient allocation
    
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
    # Determine the maximum possible size of the arrays
    max_size = 0
    for j in range(len(unique_permnos)):
        permno = unique_permnos[j]
        permno_mask = (permno_ids == permno)
        temp_features = features_np[permno_mask]
        temp_exret = exret_np[permno_mask]
        
        # Sliding window creation
        if len(temp_features) >= PAST + FUTURE:
            max_size += len(temp_features) - PAST - FUTURE + 1
    
    # Pre-allocate NumPy arrays with the maximum size
    X_arr = np.empty((max_size, PAST, features_np.shape[1]), dtype=np.float32)
    y = np.empty((max_size, FUTURE), dtype=np.float32)
    
    idx = 0  # Index to track where to insert new rows
    # Process each unique permno
    for j in prange(len(unique_permnos)):
        permno = unique_permnos[j]
        
        # Find indices for current permno
        permno_mask = (permno_ids == permno)
        temp_features = features_np[permno_mask]
        temp_exret = exret_np[permno_mask]
        
        # Sliding window creation
        if len(temp_features) >= PAST + FUTURE:
            for i in range(len(temp_features) - PAST - FUTURE + 1):
                X_arr[idx] = temp_features[i:i+PAST]
                y[idx] = temp_exret[i+PAST:i+PAST+FUTURE]
                idx += 1
    
    return X_arr[:idx], y[:idx]

def imputed_df_to_data_optimised(df, PAST=10, FUTURE=1):
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
    unique_permnos = permno_ids.unique().to_numpy(dtype=int)
    
    # Convert to numpy for Numba processing
    # Use float32 to reduce memory usage
    features_np = features.to_numpy(dtype=np.float32)
    exret_np = exret.to_numpy(dtype=np.float32)
    permno_ids_np = permno_ids.to_numpy(dtype=int)
    del features, exret
    
    # Process with Numba-accelerated function
    return imputed_df_to_data_numba(features_np, exret_np, permno_ids_np, unique_permnos, PAST, FUTURE)

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

def replace_NaNs(X: np.ndarray) -> np.ndarray:
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
    ...      -numpy array of the corresponding y-values (exret)
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

if __name__ == '__main__':
    main()
