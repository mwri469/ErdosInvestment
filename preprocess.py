import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_pickle(file_path)
    
    # Ensure 'date' and 'permno' are in the index
    if not isinstance(df.index, pd.MultiIndex) or df.index.names != ['permno', 'date']:
        raise ValueError("Expected a MultiIndex DataFrame with 'permno' and 'date' as index levels.")
    
    # Drop non-feature columns
    excluded_columns = ['gvkey', 'datadate', 'primary', 'exchcd']  # Extend as needed
    features = df.drop(columns=excluded_columns)
    
    # Identify firm change boundaries
    permno_groups = features.index.get_level_values('permno')
    is_firm_boundary = permno_groups != permno_groups.shift(-1)
    
    # Remove rows at firm boundaries to avoid firm overlap
    features = features[~is_firm_boundary]
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)
    
    # Convert normalized features to PyTorch tensors
    feature_tensors = torch.tensor(normalized_features, dtype=torch.float32)
    
    # Return tensor and scaler for potential use in inverse transformation
    return feature_tensors, scaler

if __name__ == "__main__":
    # File path to the dataset
    file_path = "../pyanomaly-master/output/merge.pickle"
    
    # Preprocess the data
    data_tensors, fitted_scaler = preprocess_data(file_path)
    
    # Save preprocessed tensors for future use
    torch.save(data_tensors, "preprocessed_data.pt")
    
    print("Data preprocessing complete. Tensor saved as 'preprocessed_data.pt'.")
