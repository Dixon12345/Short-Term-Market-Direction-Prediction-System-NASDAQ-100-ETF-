import pandas as pd

def merge_datasets(old_df, new_df):
    full = pd.concat([old_df, new_df])
    full = full[~full.index.duplicated(keep="last")]
    return full.sort_index()
