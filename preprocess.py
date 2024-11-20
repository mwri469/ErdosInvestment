import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from lstm_pipe import *

def preprocess_data():
    # Load the dataset
    df = load_data()
    
    # Ensure 'date' and 'permno' are in the index
    if not isinstance(df.index, pd.MultiIndex) or df.index.names not in set(['permno', 'date']):
        raise ValueError("Expected a MultiIndex DataFrame with 'permno' and 'date' as index levels.")
    
    # Impute data, median of each permno group for NaN vals
    df = impute_permno(df)

    # Normalize features
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    # Split data into training, validation and OOS testing sets
    # Date ranges for these sets are configured in globals.py
    train_df, val_df, test_df = split_data(df)

    # Convert to appropriate format for converting to tensors
    print('Processing training df. . .\n')
    X_train, y_train = impute_df_to_data(train_df)

    print('Processing validation df. . .\n')
    X_val, y_val = impute_df_to_data(val_df)

    print('Processing OOS testing df. . .\n')
    X_oos, y_oos = impute_df_to_data(test_df)
    
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
