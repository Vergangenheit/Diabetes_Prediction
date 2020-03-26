import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def replace_zeros(data: pd.DataFrame, cont_vars: list) -> pd.DataFrame:
    df = data.copy()
    for var in cont_vars:
        mean = df[var].mean()
        df.loc[df[var] == 0.0, var] = mean
    return df


def log_transform(data: pd.DataFrame, cont_vars: list) -> pd.DataFrame:
    X = data.copy()
    for var in cont_vars:
        if var not in ["age"]:
            X[var] = np.log(X[var])
    return X


def feature_scaling(data: pd.DataFrame, vars: list):
    df = data.copy()
    # fit scaler
    scaler = MinMaxScaler()  # create an instance
    scaler.fit(df[vars])
    df = pd.concat([df['response'].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(data[vars]),
                                 columns=vars)], axis=1)

    return df
