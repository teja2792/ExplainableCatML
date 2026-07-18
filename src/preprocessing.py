import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

CATEGORICAL_FEATURES = ["LED_Color"]
NUMERICAL_FEATURES = ["Light_Intensity_mW_cm2", "Temperature_Rise_C"]
TARGETS = ["Activity_Rate_Constant_hr", "Selectivity_Mineralization_pct"]


def load_dataset(path="data/processed/photocatalysis_dataset.csv"):
    return pd.read_csv(path)


def build_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERICAL_FEATURES),
    ])


def split_features_target(df, target):
    feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
    X = df[feature_cols]
    y = df[target]
    return X, y


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)