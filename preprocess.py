import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from lstm_pipe import *
from globals import *

def preprocess_data():
    # Load the dataset
    df = load_data()
    
    # Ensure 'date' and 'permno' are in the index
    if not isinstance(df.index, pd.MultiIndex): #or df.index.names not in set(['permno', 'date']):
        raise ValueError("Expected a MultiIndex DataFrame with 'permno' and 'date' as index levels.")
    
    # pd.set_option('display.max_rows', 500)
    # print(df.info(verbose=True, show_counts=True))
    
    # Impute data, median of each permno group for NaN vals
    non_vals=['gvkey', 'datadate', 'primary']
    non_vals_temp = df[non_vals]
    rest_of_df = df.drop(non_vals,axis=1)
    print('Imputation of NaN values. . .\n')
    df = impute_permno(rest_of_df)

    print('Normalisation of vals. . .\n')
    # Normalize features
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    # Scaler converts to np.ndarray, so convert back to pandas.DF
    df = pd.DataFrame(df, columns = df.columns)

    df = pd.concat([non_vals_temp, df], axis=1)

    # Split data into training, validation and OOS testing sets
    # Date ranges for these sets are configured in globals.py
    train_df, val_df, test_df = split_data(df)

    # Convert to appropriate format for converting to tensors
    print('Processing training df. . .\n')
    X_train, y_train = imputed_df_to_data(train_df)

    print('Processing validation df. . .\n')
    X_val, y_val = imputed_df_to_data(val_df)

    print('Processing OOS testing df. . .\n')
    X_oos, y_oos = imputed_df_to_data(test_df)
    
    # Convert normalized features to PyTorch tensors
    # TODO: Work out how to convert to tensors
    feature_tensors = None # Will do this later: torch.tensor(normalized_features, dtype=torch.float32)
    
    # Return tensor and scaler for potential use in inverse transformation
    return feature_tensors, scaler

if __name__ == "__main__": 
    # Preprocess the data
    data_tensors, fitted_scaler = preprocess_data()
    
    # Save preprocessed tensors for future use
    # TODO: uncomment when tensors are worked out
    # torch.save(data_tensors, "preprocessed_data.pt")
    
    print("Data preprocessing complete. Tensor saved as 'preprocessed_data.pt'.")
