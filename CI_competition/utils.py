import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def scale_df(df):
    """
    Scale the columns of the dataframe beside T, Prior and Y
    """
    cols = df.columns
    cols_to_scale = [col for col in cols if col not in ["T", "Prior", "Y"]]
    scale = MinMaxScaler()
    df[cols_to_scale] = scale.fit_transform(df[cols_to_scale])
    return df

def preprocess_df(df):
    """
    Preprocess the dataframe
    """
    df = pd.get_dummies(df, dtype=float)
    df = scale_df(df)
    return df