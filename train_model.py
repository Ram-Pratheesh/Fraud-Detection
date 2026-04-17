import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# ════════════════════════════════════════════════════════════════
#  FRAUD DETECTION MODEL — TRAINING PIPELINE
#  Hybrid Rule Engine + XGBoost (Production-Grade)
# ════════════════════════════════════════════════════════════════

df = pd.read_csv("output/features.csv")

FEATURES = [
    # ── Engineered Features ──
    "price_dev",
    "value_per_kg",
    "high_risk_country",
    "zero_weight_flag",
    "rule_score",
    "num_flags",
    # ── Advanced Engineered Features ──
    "invoice_frequency_per_iec",
    "avg_price_per_hs_code",
    "is_night_transaction",
    "duplicate_invoice_flag",
    "description_length",
    "is_vague_description",
    # ── Raw Fraud Indicators ──
    "origin_risk_score",
    "price_deviation_pct",
    "weight_to_value_ratio",
    "is_round_invoice",
    "is_ghost_shipment",
    "is_duplicate_invoice",
    "description_specificity",
    "suspicious_filing_hour",
    "days_to_gst_period",
    "benford_first_digit",
    "exporter_iec_age_days",
    "exporter_is_shell",
    "exporter_turnover_cap_ratio",
    "port_hs_mismatch",
    "transshipment_flag",
]

# Balanced threshold — best for hackathon demos
THRESHOLD = 0.35

X = df[FEATURES]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── STEP 1: Class-weighted XGBoost ──
num_normal = int((y_train == 0).sum())
num_fraud  = int((y_train == 1).sum())

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=(num_normal / max(num_fraud, 1)) * 1.5,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.1,
    random_state=42,
    eval_metric="logloss",
)
model.fit(X_train, y_train)

# ── STEP 2: Probability scores ──
probs = model.predict_proba(X_test)[:, 1]

# ── STEP 3: Hybrid Rule + ML scoring ──
rule_scores = df.loc[X_test.index, "rule_score"].values
final_score = 0.7 * probs + 0.3 * rule_scores

y_pred = (final_score > THRESHOLD).astype(int)

# ── STEP 4: Cross-validation for robustness ──
cv_scores = cross_val_score(model, X, y, cv=5, scoring="recall")

# ── STEP 5: Save model + feature list for predict.py ──
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(FEATURES, "models/feature_list.pkl")
joblib.dump(THRESHOLD, "models/threshold.pkl")

# ── RESULTS ──
cm = confusion_matrix(y_test, y_pred)

print("=" * 60)
print("  FRAUD DETECTION — HYBRID RULE + ML RESULTS")
print("=" * 60)
print(classification_report(y_test, y_pred))

print("  Confusion Matrix:")
print(f"    TN={cm[0][0]:5d}   FP={cm[0][1]:5d}")
print(f"    FN={cm[1][0]:5d}   TP={cm[1][1]:5d}")
print()

print(f"  Total features:        {len(FEATURES)}")
print(f"  Train balance:         {num_normal} normal / {num_fraud} fraud")
print(f"  scale_pos_weight:      {(num_normal / max(num_fraud, 1)) * 1.5:.2f}")
print(f"  Decision threshold:    {THRESHOLD} (hybrid: 0.7*ML + 0.3*rule)")
print(f"  Cross-val recall (5F): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
print()

# ── Feature Importance (Top 10) ──
importance = dict(zip(FEATURES, model.feature_importances_))
sorted_imp = sorted(importance.items(), key=lambda x: -x[1])[:10]
print("  Top 10 Features:")
for i, (feat, score) in enumerate(sorted_imp, 1):
    bar = "#" * int(score * 50)
    print(f"    {i:2d}. {feat:30s} {score:.4f} {bar}")

print()
print("  Model saved -> models/fraud_model.pkl")
print("  Run: python predict.py <your_csv>")
print("=" * 60)