import os
import joblib
import shap
import matplotlib.pyplot as plt

from preprocessing import load_dataset, split_features_target, TARGETS

os.makedirs("figures", exist_ok=True)
df = load_dataset()

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

    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"figures/shap_{target_dir}.png", dpi=150)
    plt.close()
    print(f"SHAP summary plot saved to figures/shap_{target_dir}.png")