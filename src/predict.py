"""Predict script: accepts a JSON string for a single sample or a CSV.
"""
import argparse
import joblib
import pandas as pd
import json

def main(args):
    model = joblib.load(args.model)
    if args.csv:
        df = pd.read_csv(args.csv)
        X = df.select_dtypes(include=["number"])
        preds = model.predict(X)
        df["pred_calories"] = preds
        out = args.output or "predictions.csv"
        df.to_csv(out, index=False)
        print("Saved predictions to", out)
    else:
        if not args.json:
            raise ValueError("Either --csv or --json must be provided")
        sample = json.loads(args.json)
        df = pd.DataFrame([sample])
        # perform same numeric selection as training
        X = df.select_dtypes(include=["number"])
        try:
            preds = model.predict(X)
            print("Predicted calories:", float(preds[0]))
        except Exception as e:
            print("Prediction failed:", e)
            print("Model expects numeric columns that match training. Consider using processed CSV flow.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--json", required=False, help="JSON string for single sample")
    parser.add_argument("--csv", required=False, help="CSV file with samples to predict")
    parser.add_argument("--output", required=False, help="Output CSV path for batch predictions")
    args = parser.parse_args()
    main(args)
