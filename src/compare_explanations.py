import os
import joblib
import numpy as np
import pandas as pd
import shap

from preprocessing import load_dataset, split_features_target, TARGETS

df = load_dataset()
permutation_df = pd.read_csv("results/permutation_importance.csv")

rows = []
for target in TARGETS:
    target_dir = "activity" if "Activity" in target else "selectivity"
    pipeline = joblib.load(f"models/{target_dir}/xgboost.pkl")

    X, y = split_features_target(df, target)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Aggregate one-hot LED_Color dummies back into a single importance value
    # so SHAP is comparable to permutation importance, which scores the
    # original (pre-encoding) column as one unit.
    shap_by_feature = {}
    for name, value in zip(feature_names, mean_abs_shap):
        key = "LED_Color" if name.startswith("cat__LED_Color") else name.replace("num__", "")
        shap_by_feature[key] = shap_by_feature.get(key, 0.0) + value

    perm_subset = permutation_df[
        (permutation_df["Target"] == target_dir) & (permutation_df["Model"] == "xgboost")
    ]

    for _, row in perm_subset.iterrows():
        rows.append({
            "Target": target_dir,
            "Feature": row["Feature"],
            "SHAP_Mean_Abs": round(shap_by_feature.get(row["Feature"], 0.0), 4),
            "Permutation_Importance": row["Importance_Mean"],
        })

comparison_df = pd.DataFrame(rows)
comparison_df["SHAP_Rank"] = comparison_df.groupby("Target")["SHAP_Mean_Abs"].rank(ascending=False).astype(int)
comparison_df["Permutation_Rank"] = comparison_df.groupby("Target")["Permutation_Importance"].rank(ascending=False).astype(int)
comparison_df["Methods_Agree_On_Rank"] = comparison_df["SHAP_Rank"] == comparison_df["Permutation_Rank"]

os.makedirs("results", exist_ok=True)
comparison_df.to_csv("results/method_comparison.csv", index=False)
print(comparison_df.to_string(index=False))