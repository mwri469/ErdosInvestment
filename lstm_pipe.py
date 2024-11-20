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

    exclude_columns = ["gvkey", "datadate", "primary", "exchcd", "ret", "exret", "rf"]
    exret = df.exret
    features = df.drop(columns=exclude_columns)

    # Build data for lstm
    past = 5 # 5 months of data
    future = 1 # next month risk prem
    X_arr,y = [],[]

    for p in tqdm(permno_set):
        idxs = [idx for idx, val in enumerate(permno_ids) if val == p]
        temp_df = df.iloc[idxs]
        temp_exret = exret.iloc[idxs]
        
        for i in range(len(temp_df) - past - future + 1):
            X_arr.append(temp_df.iloc[i:i+past])
            y.append(temp_exret.iloc[i+past:i+past+future])
        

if __name__ == '__main__':
    main()