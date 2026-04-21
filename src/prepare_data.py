"""
Data preparation for the Muchlinski et al. (2016) civil-war-onset reproduction.

Loads the civil-war panel data, imputes missing values, and splits into
train and test sets.

Reference: Muchlinski, D., Siroky, D., He, J., & Kocher, M. (2016).
"Comparing Random Forest with Logistic Regression for Predicting
Class-Imbalanced Civil War Onset Data." Political Analysis.
"""
from __future__ import annotations

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split

TARGET = "civil_war_onset"
ID_COLS = ["country_id", "year"]
RANDOM_STATE = 42


def prepare(df: pd.DataFrame, test_size: float = 0.25):
    """Impute missing values and split into train/test.

    Returns (X_train, X_test, y_train, y_test).
    """
    df = df.drop(columns=ID_COLS).copy()

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    imputer = KNNImputer(n_neighbors=5)
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=X.columns,
        index=X_train_raw.index,
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=X.columns,
        index=X_test_raw.index,
    )
    return X_train, X_test, y_train, y_test
