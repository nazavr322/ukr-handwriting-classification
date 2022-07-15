from typing import Optional
import pandas as pd


def code_labels(data: pd.DataFrame,
                col_name: Optional[str] = None) -> pd.DataFrame:
    """
    Adds a new column to a dataframe with all labels encoded into numbers
    """
    df = data.copy()
    df.label = pd.Categorical(df.label, df.label.unique())
    lbl_col_idx = df.columns.get_loc('label') # index of label column
    new_col_name = col_name if col_name else 'lbl_code'
    df.insert(lbl_col_idx + 1, new_col_name, df.label.cat.codes)
    return df

