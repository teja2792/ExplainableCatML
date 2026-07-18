import os
import numpy as np
import pandas as pd
import joblib

from preprocessing import load_dataset, split_features_target, TARGETS

os.makedirs("results", exist_ok=True)
df = load_dataset()
LED_COLORS = ["Red", "Amber", "Green"]

TARGET_CONFIG = {
    "activity": {"column": "Activity_Rate_Constant_hr", "threshold": 2.0},
    "selectivity": {"column": "Selectivity_Mineralization_pct", "threshold": 90.0},
}

rows = []
for target in TARGETS:
    target_dir = "activity" if "Activity" in target else "selectivity"
    config = TARGET_CONFIG[target_dir]
    pipeline = joblib.load(f"models/{target_dir}/xgboost.pkl")
    X, y = split_features_target(df, target)

    # Note: Temperature_Rise_C is deliberately excluded from the search --
    # it's a measured outcome of the reaction, not a variable an
    # experimentalist directly sets, so it isn't a valid "lever" to recommend
    # changing.
    predictions = pipeline.predict(X)
    worst_idx = predictions.argmin()
    original = X.iloc[[worst_idx]].copy()
    original_pred = predictions[worst_idx]

    print(f"\n=== {target_dir}: counterfactual search ===")
    print(f"Original instance: {original.to_dict(orient='records')[0]}")
    print(f"Original predicted {config['column']}: {original_pred:.3f} (threshold: {config['threshold']})")

    best_change = None
    for color in LED_COLORS:
        for intensity in np.arange(0, 8.1, 0.5):
            candidate = original.copy()
            candidate["LED_Color"] = color
            candidate["Light_Intensity_mW_cm2"] = intensity
            pred = pipeline.predict(candidate)[0]
            if pred >= config["threshold"]:
                color_changed = color != original["LED_Color"].values[0]
                intensity_delta = abs(intensity - original["Light_Intensity_mW_cm2"].values[0])
                cost = int(color_changed) + intensity_delta / 8.0
                if best_change is None or cost < best_change["cost"]:
                    best_change = {"LED_Color": color, "Light_Intensity_mW_cm2": intensity, "Predicted": pred, "cost": cost}

    if best_change:
        print(f"Minimal counterfactual: LED_Color={best_change['LED_Color']}, "
              f"Light_Intensity_mW_cm2={best_change['Light_Intensity_mW_cm2']:.1f} "
              f"-> predicted {config['column']} = {best_change['Predicted']:.3f}")
        rows.append({
            "Target": target_dir,
            "Original_LED_Color": original["LED_Color"].values[0],
            "Original_Intensity": original["Light_Intensity_mW_cm2"].values[0],
            "Original_Prediction": round(original_pred, 3),
            "Counterfactual_LED_Color": best_change["LED_Color"],
            "Counterfactual_Intensity": best_change["Light_Intensity_mW_cm2"],
            "Counterfactual_Prediction": round(best_change["Predicted"], 3),
        })
    else:
        print("No counterfactual found within the search grid that meets the threshold.")

results_df = pd.DataFrame(rows)
results_df.to_csv("results/counterfactuals.csv", index=False)
print("\nSaved to results/counterfactuals.csv")
print(results_df.to_string(index=False))