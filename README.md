# Calorie Prediction for Various Exercises

**A clean, interview-ready repository** for predicting calories burned for different exercises using user & activity features.  
This README contains everything an interviewer needs: project overview, repo structure, how to run the code end-to-end, example commands, explanations of engineering choices, results, and suggestions for demo / talking points.

---

## Project summary
This project predicts calories burned using tabular features such as `exercise_type`, `duration_min`, user demographics (age, weight), and simple derived features (a MET-based calorie estimate). Multiple models were evaluated — **Linear Regression**, **Random Forest**, **SVR**, and **XGBoost** — with **XGBoost** performing best in the original analysis (reported MAE ≈ 1.62, MSE ≈ 5.19).

Key ideas:
- Domain-informed features (MET × weight × duration) + raw features improve predictions.
- Modular, reproducible pipeline: preprocessing → feature engineering → training → evaluation → prediction.
- Focus on clarity and a reviewer-friendly repo structure for interviews.

---

## Quick start (copy–paste)

1. **Clone repo**
```bash
git clone https://github.com/<your-username>/calorie-prediction.git
cd calorie-prediction
```

2. **Create & activate virtual environment**
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Add your dataset**
- Put your CSV file(s)

4. **Preprocess & split**
```bash
python src/data_preprocess.py --input data/raw/exercise.csv --output data/processed --target_col Calories --test_size 0.2
```

5. **Train (XGBoost by default)**
```bash
python src/train.py --train data/processed/train.csv --model_out models/xgboost_best.joblib --metrics_out models/metrics.json --model xgboost --target Calories --val_size 0.2
```

6. **Evaluate**
```bash
python src/evaluate.py --model models/xgboost_best.joblib --test data/processed/test.csv --target Calories --output models/eval_metrics.json
```

7. **Predict (single sample)**
```bash
python src/predict.py --model models/xgboost_best.joblib --json '{"exercise_type":"running","duration_min":30,"age":28,"weight_kg":70,"avg_hr":145}'
```

8. **Predict (batch CSV)**
```bash
python src/predict.py --model models/xgboost_best.joblib --csv data/processed/test.csv --output predictions.csv
```

---

## Files & purpose (for interviewers)
- `src/data_preprocess.py` — loads raw CSV, basic cleaning, calls `create_features()`, splits into train/test, and writes processed CSVs.
- `src/features.py` — feature engineering: cleans `exercise_type`, maps MET estimates, derives `cal_by_met`, and creates one-hot features for top exercise types.
- `src/train.py` — supports `linear`, `rf`, `svr`, `xgboost`; trains model, evaluates on validation split, saves model + metrics.
- `src/evaluate.py` — loads saved model, evaluates on test set, writes metrics, and saves `models/pred_vs_actual.png`.
- `src/predict.py` — single-sample JSON or CSV batch prediction; adds `pred_calories` column for batch output.
- `notebooks/` — EDA and visualizations (useful to show in interviews).
- `models/` — store model artifacts and plots (add to `.gitignore` so you don't commit large files).

---

## Evaluation & metrics
Use k-fold CV or a hold-out test set. Useful metrics:
- Mean Absolute Error (MAE) — primary metric for calories
- Mean Squared Error (MSE)
- R²

---

## Design choices & talking points
- **Why MET-based feature?** MET × weight × duration is a domain-driven baseline estimating energy expenditure; including it helps the model learn residuals and reduces bias.
- **Why XGBoost?** Robust to unscaled features, handles heterogeneous numeric features well, often strong baseline for tabular problems.
- **Feature engineering:** Show how `exercise_type` was cleaned and encoded, importance of duration and weight, and how `avg_hr` can help for intensity personalization.
- **Limitations:** Dataset labeling noise, user variability in metabolism, lack of continuous sensor time-series (accelerometer) which could improve granularity.
- **Extensions:** Per-user calibration, transfer learning for personalization, deep-learning on sensor time-series (CNN/RNN/Transformer), or deploying an inference API for a mobile demo.

---
