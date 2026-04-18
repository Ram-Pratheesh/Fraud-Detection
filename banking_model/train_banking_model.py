import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "output", "card_features.csv")
        
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run graph_features.py first.")
        return

    print(f"Loading native network features from {input_path}...")
    df = pd.read_csv(input_path)

    # 1. Prepare Columns (Drop IDs & timestamps, but One-Hot Encode categorical demographics)
    drop_cols = ['card_id', 'txn_timestamp', 'device_fingerprint', 'ip_address', 'signup_time']
    existing_drop = [col for col in drop_cols if col in df.columns]
    X_raw = df.drop(columns=existing_drop + ['is_fraud'], errors='ignore')
    
    # One-Hot Encode categories (source, browser, sex)
    categorical_cols = ['source', 'browser', 'sex']
    cat_exist = [col for col in categorical_cols if col in X_raw.columns]
    if cat_exist:
        X = pd.get_dummies(X_raw, columns=cat_exist, drop_first=True)
    else:
        X = X_raw

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = df['is_fraud']

    print(f"Features configured for XGBoost ({len(X.columns)} features total).")

    # 2. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Class distribution Before SMOTE:")
    print(y_train.value_counts())

    # 3. SMOTE
    print("Applying SMOTE Oversampling directly on genuine Fraud samples...")
    smote = SMOTE(random_state=42)
    try:
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print("Class distribution After SMOTE:")
        print(pd.Series(y_resampled).value_counts())
    except Exception as e:
        print(f"Skipping SMOTE due to class limits: {e}")
        X_resampled, y_resampled = X_train, y_train

    # 4. Train Supervised XGBoost
    print("Training Supervised XGBoost Classifier on Native Data...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_resampled, y_resampled)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n================== XGBOOST RESULTS ==================")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5. TreeExplainer (SHAP)
    print("\nFitting SHAP TreeExplainer for true Explainability...")
    explainer = shap.TreeExplainer(model)

    # 6. Save Artifacts for Streamer
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
            
    model_path = os.path.join(model_dir, "banking_model.pkl")
    explainer_path = os.path.join(model_dir, "banking_shap_explainer.pkl")
    features_path = os.path.join(model_dir, "banking_features.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)
    joblib.dump(list(X.columns), features_path) 
    
    print(f"Successfully generated clean Model and Explainer artifacts.")

if __name__ == "__main__":
    train()
