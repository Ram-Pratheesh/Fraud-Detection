# Fraud Detection Pipeline: Technical Blueprint

This document explains every single file in the codebase, the rationale behind the techniques used, and how data moves through the project. It is specifically written to help you prepare for rapid-fire technical questions from leadership. 

Read through this carefully.

## 📂 Project Architecture Overview

This project is a multi-layered, hybrid Fraud Detection System. It doesn't rely solely on one method. Instead, it uses three layers of defense:
1. **Data Gathering & Simulation:** Scraping real entities and synthetically generating transactions to mimic real-world financial data.
2. **Deterministic Rules Engine:** Hardcoded bounds (e.g., "if price > 2x average, flag it").
3. **Machine Learning:** An XGBoost model that finds hidden complex patterns the rules miss, with a hybrid final aggregation.

---

## 📄 File-by-File Breakdown

### 1. `scrape_msme.py`
*   **What it does:** A web scraper built with `requests` and `BeautifulSoup`.
*   **How it works:** It connects to the Tamil Nadu MSME portal specifically targeting their Corona Safe Units list. It extracts tables of real, verified registered businesses. 
*   **Why we need it:** ML models trained on 100% synthetic data fail in the real world. By seeding our data generator with real company names, locations, and districts, the simulated transactions closely model real Indian trade routes.
*   **Output:** Creates `real_msme_seed.csv`.

### 2. `generate_data.py`
*   **What it does:** The massive engine that generates the synthetic transactions, acting as our raw data source.
*   **How it works:** 
    *   It generates realistic Indian MSME identities (using the seed file if available).
    *   It simulates International Trade Transactions utilizing real HS Codes (e.g., 8471 for computers) and benchmarks UN Comtrade prices.
    *   **The Injection of Fraud:** Crucially, it mathematically injects specific fraud typologies into the data—like *under/over-invoicing*, *ghost shipments* (zero weight), *duplicate invoices*, and *vague descriptions*.
    *   **The Rule Engine Layer:** It contains a `flag_transaction` function that calculates a `risk_score` by checking hard limits (e.g. is the country on the FATF greylist? Are they filing between 2-4 AM?).
*   **Output:** Creates `output/transactions_flagged.csv`.

### 3. `rule_engine_demo.py`
*   **What it does:** A lightweight script used to demonstrate the baseline deterministic rule engine.
*   **Why we need it:** Showcases the foundational layer of fraud detection. Deterministic rules are fast and easily explainable (which auditors love), catching the absolute most obvious frauds before heavy Machine Learning gets involved.

### 4. `feature_engineering.py`
*   **What it does:** Transforms raw CSV columns into dense mathematical signals formatted for XGBoost. 
*   **How it works:** It engineers complex meta-features from the baseline data:
    *   *Frequency counting:* Tracking how often specific IECs (Importer/Exporter Codes) appear.
    *   *Temporal tracking:* Converting raw timestamps into boolean "is_night_transaction" flags.
    *   *Text parsing:* Measuring the word count of `goods_description` and flagging "vague" filler words (e.g., "misc", "assorted").
*   **Output:** Creates the final clean dataset `output/features.csv`, perfectly formatted for training.

### 5. `train_model.py`
*   **What it does:** The most critical file. This trains the `XGBClassifier` and optimizes it specifically for high fraud *Recall*.
*   **Key Technical Implementations (MEMORIZE THESE):**
    *   **Class Imbalance Handling:** Uses `scale_pos_weight` (automatically calculated as Normal_Count / Fraud_Count * 1.5). This artificially inflates the penalty for missing a fraudulent transaction, forcing the model to care more about the minority class.
    *   **Predict Proba vs Predict:** Instead of the default `.predict()` which cuts off at a strict 0.5 probability, we use `.predict_proba()` and manually tune the threshold down to `0.35` for the hackathon. This catches vastly more fraud (High Recall) at the slight expense of false positives.
    *   **The Hybrid Score:** We integrate the ML model and the Rule Engine into a final equation: `Final_Score = (0.7 * ML_Probability) + (0.3 * Rule_Score)`. This gives the best of both worlds.
    *   **Cross-Validation:** Uses 5-fold cross-validation to prove to the judges that the model doesn't overfit and is mathematically stable.
*   **Output:** Serializes and saves the model rules into `models/fraud_model.pkl`, `models/feature_list.pkl`, and `models/threshold.pkl`.

### 6. `predict.py`
*   **What it does:** Your Hackathon "Showcase" script. It proves the pipeline works in "production" by inferencing new, unseen CSV files.
*   **How it works:**
    *   Accepts a raw CSV provided by the judges via command line arg.
    *   **Missing Data Resilience:** If judges withhold a feature column, this script detects the mismatch and gracefully fills it with `0` rather than fatally crashing.
    *   **Human-Readable Output:** Bins mathematical output into business-friendly terms: "LOW", "MEDIUM", "HIGH", "VERY HIGH", and "CRITICAL" risk buckets.
*   **Output:** Creates `output/predictions.csv` (the full batch) and `output/high_risk_transactions.csv` (a filtered list for investigators to review).

---

## 🎯 Quick Expected Q&A

**Q: "Why did you use XGBoost instead of a Deep learning approach like Neural Networks?"**
> **A:** For tabular, financial transaction data featuring mixed categorical and continuous variables, tree-based models like XGBoost vastly outperform Deep Learning. Furthermore, XGBoost provides feature importance rankings out-of-the-box, giving us perfect *explainability* which is a legal requirement in financial sector compliance.

**Q: "How are you handling the fact that 95%+ of transactions are perfectly legal?"**
> **A:** Fraud is a classic highly-imbalanced class problem. We solve this mathematically using `scale_pos_weight` inside XGBoost to heavily penalize False Negatives, combined with a lowered threshold tuning mechanism geared toward high Recall.

**Q: "What if the ML model hallucinates or makes a mistake?"**
> **A:** That is exactly why we use a Hybrid design. The ML probabilities are averaged securely alongside our deterministic, hard-coded Rule Engine (`Final_Score = 0.7 * ML + 0.3 * Rule`). The rule engine acts as an anchor ensuring human-led logic always influences the output.
