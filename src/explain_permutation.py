import os
import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from preprocessing import (
    load_dataset, split_features_target, train_test_split_data,
    TARGETS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES,
)
from model import MODEL_BUILDERS

df = load_dataset()
feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
rows = []

for target in TARGETS:
    target_dir = "activity" if "Activity" in target else "selectivity"
    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    for model_name in MODEL_BUILDERS:
        pipeline = joblib.load(f"models/{target_dir}/{model_name}.pkl")
        result = permutation_importance(
            pipeline, X_test, y_test, n_repeats=20, random_state=42, scoring="r2"
        )
        for feat, imp_mean, imp_std in zip(feature_cols, result.importances_mean, result.importances_std):
            rows.append({
                "Target": target_dir, "Model": model_name, "Feature": feat,
                "Importance_Mean": round(imp_mean, 4), "Importance_Std": round(imp_std, 4),
            })

results_df = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/permutation_importance.csv", index=False)
print(results_df.to_string(index=False))