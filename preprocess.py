import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle as pkl
from lstm_pipe import *
from globals import *

def preprocess_data():
    # Load the dataset
    df = load_data()
    
    # Ensure 'date' and 'permno' are in the index
    if not isinstance(df.index, pd.MultiIndex): #or df.index.names not in set(['permno', 'date']):
        raise ValueError("Expected a MultiIndex DataFrame with 'permno' and 'date' as index levels.")
    
    # Impute data, median of each permno group for NaN vals
    non_vals=["gvkey", "datadate", "primary"] #, "exchcd", "ret", "exret", "rf", "me"]
    non_vals_temp = df[non_vals]
    dropped_df = df.drop(non_vals,axis=1)
    del df
    print('Imputation of NaN values. . .\n')
    imputed_df = impute_permno_optimised(dropped_df)
    del dropped_df

    imputed_df.to_pickle('data/imputed_df.pickle')

    print('Normalisation of vals. . .\n')
    # Normalize features
    or_cols = imputed_df.columns
    scaler = StandardScaler()
    temp_transformed_df = scaler.fit_transform(imputed_df)
    transformed_df = pd.DataFrame(temp_transformed_df, index=imputed_df.index, columns=or_cols)
    del temp_transformed_df, imputed_df

    # Scaler converts to np.ndarray, so convert back to pandas.DF
    df = pd.concat([non_vals_temp, transformed_df], axis=1)
    del non_vals_temp, transformed_df
    
    # Split data into training, validation and OOS testing sets
    # Date ranges for these sets are configured in globals.py
    train_df, val_df, test_df = split_data(df)
    del df

    # Very messy code to come, had to make more memory efficient when calling functions by converting to np
    # TODO: Clean up code here, work out efficiency gains
    
    # Convert to appropriate format for converting to tensors
    print('Processing training df. . .\n')
    X_train, y_train = imputed_df_to_data_optimised(train_df)
    del train_df
    # Ensure y has no NaN values
    eps = 1e-3 # Error
    assert np.isnan(y_train).sum() < eps

    print('Processing validation df. . .\n')
    X_val, y_val = imputed_df_to_data_optimised(val_df)
    del val_df
    assert np.isnan(y_val).sum() < eps

    print('Processing OOS testing df. . .\n')
    X_oos, y_oos = imputed_df_to_data_optimised(test_df)
    del test_df
    assert np.isnan(y_oos).sum() < eps
    
    # Return tensor and scaler for potential use in inverse transformation
    return X_train, y_train, X_val, y_val, X_oos, y_oos

if __name__ == "__main__": 
    # Preprocess the data
    data_tensors, fitted_scaler = preprocess_data()
    
    # Save preprocessed tensors for future use
    # TODO: uncomment when tensors are worked out
    # torch.save(data_tensors, "preprocessed_data.pt")
    
    print("Data preprocessing complete. Tensor saved as 'preprocessed_data.pt'.")

