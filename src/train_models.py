import os
import joblib
import pandas as pd

from preprocessing import load_dataset, split_features_target, train_test_split_data, TARGETS
from model import MODEL_BUILDERS
from evaluation import evaluate, print_metrics

df = load_dataset()
results = []

for target in TARGETS:
    X, y = split_features_target(df, target)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    target_dir = "activity" if "Activity" in target else "selectivity"
    os.makedirs(f"models/{target_dir}", exist_ok=True)

    print(f"\n=== Target: {target} ===")
    for model_name, builder in MODEL_BUILDERS.items():
        pipeline = builder()
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = evaluate(y_test, predictions)
        print_metrics(f"{target_dir}/{model_name}", metrics)

        joblib.dump(pipeline, f"models/{target_dir}/{model_name}.pkl")
        results.append({"Target": target_dir, "Model": model_name, **metrics})

results_df = pd.DataFrame(results)
os.makedirs("results", exist_ok=True)
results_df.to_csv("results/model_comparison.csv", index=False)
print("\nFull comparison saved to results/model_comparison.csv")
print(results_df.to_string(index=False))