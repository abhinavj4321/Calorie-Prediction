"""Train script supporting: linear, rf, svr, xgboost
Saves a joblib model and a small JSON metrics file.
"""
import argparse
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def build_model(model_name):
    if model_name == "linear":
        return Pipeline([("scale", StandardScaler()), ("lr", LinearRegression())])
    elif model_name == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    elif model_name == "svr":
        return Pipeline([("scale", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    elif model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise RuntimeError("xgboost is not installed. Install it via requirements.txt") from e
        return XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=42, n_jobs=-1)
    else:
        raise ValueError("Unknown model: " + model_name)

def prepare_xy(df, target_col):
    X = df.drop(columns=[target_col], errors="ignore")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")
    y = df[target_col].values
    # keep only numeric columns for modeling by default
    X = X.select_dtypes(include=["number"])
    return X, y

def main(args):
    df = pd.read_csv(args.train)
    X, y = prepare_xy(df, args.target)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_size, random_state=42)

    model = build_model(args.model)
    print("Training model:", args.model)
    model.fit(X_train, y_train)

    # predict & evaluate
    y_pred = model.predict(X_val)
    metrics = {
        "mae": float(mean_absolute_error(y_val, y_pred)),
        "mse": float(mean_squared_error(y_val, y_pred)),
        "r2": float(r2_score(y_val, y_pred))
    }
    print("Validation metrics:", metrics)

    # save model & metrics
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    joblib.dump(model, args.model_out)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model to", args.model_out)
    print("Saved metrics to", args.metrics_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Processed train CSV (train split or full dataset)")
    parser.add_argument("--model", default="xgboost", choices=["linear","rf","svr","xgboost"])
    parser.add_argument("--model_out", default="models/model.joblib")
    parser.add_argument("--metrics_out", default="models/metrics.json")
    parser.add_argument("--target", default="Calories")
    parser.add_argument("--val_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
