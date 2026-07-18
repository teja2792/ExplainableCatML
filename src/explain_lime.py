import os
import pandas as pd
import joblib
from lime.lime_tabular import LimeTabularExplainer

from preprocessing import load_dataset, split_features_target, TARGETS, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

os.makedirs("figures", exist_ok=True)
df = load_dataset()
feature_cols = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

LED_COLORS = sorted(df["LED_Color"].unique())
color_to_code = {c: i for i, c in enumerate(LED_COLORS)}
code_to_color = {i: c for c, i in color_to_code.items()}

for target in TARGETS:
    target_dir = "activity" if "Activity" in target else "selectivity"
    pipeline = joblib.load(f"models/{target_dir}/xgboost.pkl")
    X, y = split_features_target(df, target)

    # LIME requires a fully numeric array; LED_Color is integer-coded here
    # and decoded back to its string category inside predict_fn before the
    # pipeline (which expects the original categories) is called.
    X_encoded = X.copy()
    X_encoded["LED_Color"] = X_encoded["LED_Color"].map(color_to_code)
    X_encoded = X_encoded[feature_cols].values

    def predict_fn(x_array):
        x_df = pd.DataFrame(x_array, columns=feature_cols)
        x_df["LED_Color"] = x_df["LED_Color"].round().astype(int).map(code_to_color)
        return pipeline.predict(x_df)

    explainer = LimeTabularExplainer(
        X_encoded, feature_names=feature_cols,
        categorical_features=[0], categorical_names={0: LED_COLORS},
        mode="regression", discretize_continuous=True,
    )

    instance_idx = len(X_encoded) // 2
    explanation = explainer.explain_instance(X_encoded[instance_idx], predict_fn, num_features=3)
    explanation.save_to_file(f"figures/lime_{target_dir}_instance.html")

    print(f"\n=== LIME: {target_dir}, instance {instance_idx} ===")
    print(f"Instance: {X.iloc[instance_idx].to_dict()}")
    print(explanation.as_list())