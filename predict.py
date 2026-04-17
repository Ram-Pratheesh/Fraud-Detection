"""
Fraud Detection — Prediction Pipeline
Usage: python predict.py <input_csv>

Loads the trained model and scores new transactions.
Outputs fraud predictions with confidence scores.
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

# ════════════════════════════════════════════════════════════════
#  LOAD MODEL & CONFIG
# ════════════════════════════════════════════════════════════════

MODEL_PATH    = "models/fraud_model.pkl"
FEATURES_PATH = "models/feature_list.pkl"
THRESH_PATH   = "models/threshold.pkl"

if not os.path.exists(MODEL_PATH):
    print("ERROR: No trained model found. Run 'python train_model.py' first.")
    sys.exit(1)

model     = joblib.load(MODEL_PATH)
FEATURES  = joblib.load(FEATURES_PATH)
THRESHOLD = joblib.load(THRESH_PATH)

# ════════════════════════════════════════════════════════════════
#  LOAD & PREPARE INPUT DATA
# ════════════════════════════════════════════════════════════════

if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = "output/features.csv"

if not os.path.exists(input_file):
    print(f"ERROR: File not found: {input_file}")
    sys.exit(1)

print(f"Loading data from: {input_file}")
df = pd.read_csv(input_file)
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")

# Handle missing features gracefully (fill with 0)
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print(f"  WARNING: Missing {len(missing)} features (filling with 0):")
    for m in missing:
        print(f"    - {m}")
        df[m] = 0

X = df[FEATURES]

# ════════════════════════════════════════════════════════════════
#  PREDICT
# ════════════════════════════════════════════════════════════════

probs = model.predict_proba(X)[:, 1]

# Hybrid scoring (if rule_score available)
if "rule_score" in df.columns:
    final_score = 0.7 * probs + 0.3 * df["rule_score"].values
else:
    final_score = probs

df["fraud_probability"]  = np.round(final_score, 4)
df["fraud_prediction"]   = (final_score > THRESHOLD).astype(int)
df["risk_level"] = pd.cut(
    final_score,
    bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
    labels=["LOW", "MEDIUM", "HIGH", "VERY HIGH", "CRITICAL"],
    include_lowest=True,
)

# ════════════════════════════════════════════════════════════════
#  OUTPUT
# ════════════════════════════════════════════════════════════════

total      = len(df)
flagged    = int(df["fraud_prediction"].sum())
clean      = total - flagged

print()
print("=" * 60)
print("  FRAUD DETECTION — PREDICTION RESULTS")
print("=" * 60)
print(f"  Total transactions:    {total}")
print(f"  Flagged as fraud:      {flagged} ({flagged/total*100:.1f}%)")
print(f"  Clean:                 {clean} ({clean/total*100:.1f}%)")
print(f"  Threshold used:        {THRESHOLD}")
print()

# Risk distribution
print("  Risk Distribution:")
risk_counts = df["risk_level"].value_counts()
for level in ["CRITICAL", "VERY HIGH", "HIGH", "MEDIUM", "LOW"]:
    count = risk_counts.get(level, 0)
    pct = count / total * 100
    bar = "#" * int(pct / 2)
    print(f"    {level:10s} {count:5d} ({pct:5.1f}%) {bar}")

print()

# If actual labels exist, show accuracy
if "label" in df.columns or "is_fraud" in df.columns:
    label_col = "label" if "label" in df.columns else "is_fraud"
    from sklearn.metrics import classification_report, confusion_matrix
    y_true = df[label_col]
    y_pred = df["fraud_prediction"]
    print("  Accuracy vs Ground Truth:")
    print(classification_report(y_true, y_pred, target_names=["Clean", "Fraud"]))

    cm = confusion_matrix(y_true, y_pred)
    print(f"    TN={cm[0][0]:5d}   FP={cm[0][1]:5d}")
    print(f"    FN={cm[1][0]:5d}   TP={cm[1][1]:5d}")
    print()

# Save results
output_file = "output/predictions.csv"
df.to_csv(output_file, index=False)
print(f"  Results saved -> {output_file}")

# Save high-risk transactions separately
high_risk = df[df["fraud_prediction"] == 1].sort_values("fraud_probability", ascending=False)
high_risk_file = "output/high_risk_transactions.csv"
high_risk.to_csv(high_risk_file, index=False)
print(f"  High-risk cases -> {high_risk_file} ({len(high_risk)} rows)")
print("=" * 60)
