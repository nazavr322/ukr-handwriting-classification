import pandas as pd


def code_labels(data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column to a dataframe with all labels encoded into numbers
    """
    df = data.copy()
    df.label = pd.Categorical(df.label, df.label.unique())
    lbl_col_idx = df.columns.get_loc('label') # index of label column
    df.insert(lbl_col_idx + 1, 'lbl_code', df.label.cat.codes)
    return df

