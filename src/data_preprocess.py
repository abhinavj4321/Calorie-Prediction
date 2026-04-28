"""Simple preprocessing script.
Loads a CSV, cleans, creates train/test CSVs (processed).
"""
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from .features import create_features

def load_data(path):
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    # example cleaning steps
    df = df.copy()
    # drop exact duplicates
    df = df.drop_duplicates()
    # strip whitespace in object columns
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].str.strip()
    return df

def main(args):
    df = load_data(args.input)
    df = basic_clean(df)

    # create features (returns df with engineered cols, and target)
    df = create_features(df)

    # optional: drop rows missing target
    df = df.dropna(subset=[args.target_col])

    # split
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)

    # save
    os.makedirs(args.output, exist_ok=True)
    train_out = args.output.rstrip('/') + '/train.csv'
    test_out = args.output.rstrip('/') + '/test.csv'
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"Saved processed train -> {train_out}, test -> {test_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw CSV input")
    parser.add_argument("--output", default="data/processed", help="Output folder for processed CSVs")
    parser.add_argument("--target_col", default="Calories", help="Name of target column")
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()
    main(args)
