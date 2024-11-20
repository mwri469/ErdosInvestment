import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from globals import *

def main():
    with open(FILE_PATH, 'rb') as f:
        obj = pkl.load(f)

    df = pd.DataFrame(obj)

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
    past = 5 # 5 months of data
    future = 1 # next month risk prem
    X_arr,y = np.empty(len(permno_set)), np.empty(len(permno_set))

    for j, p in tqdm(enumerate(permno_set)):
        idxs = [idx for idx, val in enumerate(permno_ids) if val == p]
        temp_df = features.iloc[idxs]
        temp_exret = exret.iloc[idxs]
        
        for i in range(len(temp_df) - past - future + 1):
            X_arr[j] = temp_df.iloc[i:i+past]
            y[j] = temp_exret.iloc[i+past:i+past+future]
        
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

if __name__ == '__main__':
    main()