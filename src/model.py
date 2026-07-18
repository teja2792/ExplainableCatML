from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from preprocessing import build_preprocessor


def build_random_forest_pipeline(n_estimators=200, random_state=42):
    preprocessor = build_preprocessor()
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_xgboost_pipeline(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42):
    preprocessor = build_preprocessor()
    model = XGBRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=random_state,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_lightgbm_pipeline(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42):
    preprocessor = build_preprocessor()
    model = LGBMRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, random_state=random_state, verbose=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


MODEL_BUILDERS = {
    "random_forest": build_random_forest_pipeline,
    "xgboost": build_xgboost_pipeline,
    "lightgbm": build_lightgbm_pipeline,
}