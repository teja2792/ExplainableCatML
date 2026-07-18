import os
import joblib
import matplotlib.pyplot as plt
from PyALE import ale

from preprocessing import load_dataset, split_features_target, TARGETS, NUMERICAL_FEATURES

os.makedirs("figures", exist_ok=True)
df = load_dataset()

for target in TARGETS:
    target_dir = "activity" if "Activity" in target else "selectivity"
    pipeline = joblib.load(f"models/{target_dir}/xgboost.pkl")
    X, y = split_features_target(df, target)

    for feature in NUMERICAL_FEATURES:
        ale_eff = ale(
            X=X, model=pipeline, feature=[feature],
            feature_type="continuous", grid_size=20, include_CI=True,
        )
        plt.title(f"ALE: {feature} on {target_dir}")
        plt.tight_layout()
        plt.savefig(f"figures/ale_{target_dir}_{feature}.png", dpi=150)
        plt.close()
        print(f"ALE plot saved: figures/ale_{target_dir}_{feature}.png")