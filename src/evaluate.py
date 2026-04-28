"""Evaluate a saved model on a test CSV and print/save metrics.
"""
import argparse
import pandas as pd
import joblib
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def main(args):
    model = joblib.load(args.model)
    df = pd.read_csv(args.test)
    if args.target not in df.columns:
        raise ValueError("Target column not found in test CSV")
    X = df.drop(columns=[args.target]).select_dtypes(include=["number"])
    y = df[args.target].values
    y_pred = model.predict(X)
    metrics = {
        "mae": float(mean_absolute_error(y, y_pred)),
        "mse": float(mean_squared_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred))
    }
    print(metrics)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    # quick plot
    os.makedirs("models", exist_ok=True)
    plt.figure(figsize=(6,6))
    plt.scatter(y, y_pred, alpha=0.6)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig("models/pred_vs_actual.png")
    print("Saved pred_vs_actual.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--target", default="Calories")
    parser.add_argument("--output", default="models/eval_metrics.json")
    args = parser.parse_args()
    main(args)
