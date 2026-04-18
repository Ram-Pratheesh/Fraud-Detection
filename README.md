paysim.csv is not included due to size limits.
Download from whatsapp and place it in: banking_model/data/
# Banking Fraud Graph Detection Pipeline

This project is a multi-layered, hybrid Fraud Detection System. It incorporates deterministic rules engines, an XGBoost Machine Learning classifier, and NetworkX Topographical Graph modeling to detect advanced financial crimes (like Money Mules, Semantic Evasion, and Smurfing).

## 🚀 How to Run the Test Suites

**1. Run Rule-Based Testing (Determinism & ML Basics)**
```bash
python test_edge_case.py
```
*This validates your rule engine layers to ensure basic thresholds (e.g., impossible geography, NLP mismatches, high-value spikes) fire correctly.*

**2. Run Advanced Graph & Network Testing (Adversarial Simulation)**
```bash
python test_adversarial_cases.py
```
*This is the core graph testing suite. It validates your real-time integration with the PaySim graph, FireHOL/Feodo blacklists, and dynamic XGBoost scoring JSON payloads against stealth attacks.*

---

## 🛠️ Setup & Installation

**1. Clone the Repository**
```bash
git clone https://github.com/Ram-Pratheesh/Fraud-Detection.git
cd Fraud-Detection
```

**2. Install Dependencies**
Ensure you have the latest packages installed.
```bash
pip install -r requirment.txt
```

---

## 📄 File-by-File Technical Breakdown

### Core Modules (The Graph & Network Layer)

#### `banking_model/realtime_engine.py`
*   **What it does:** The primary streaming engine that mimics real-time transaction ingestion.
*   **How it works:** It loads massive global datastores into memory at startup. When a transaction fires, the engine uses **NetworkX** to find real 2-hop topological connections (Money Mule tracking) and verifies the IPs against threat intelligence.
*   **Why we need it:** To simulate latency-sensitive production environments where multiple components evaluate simultaneous rules asynchronously.

#### `banking_model/explainer.py`
*   **What it does:** Generates fully transparent, human-readable JSON payloads explaining *why* a transaction was flagged.
*   **How it works:** Executes a `TreeExplainer` (SHAP) over the mathematical XGBoost output to identify the top driving log-odds variations, combined with pure logic from the `realtime_engine`.

#### `banking_model/train_banking_model.py`
*   **What it does:** Trains the core XGBoost Classifier. 
*   **How it works:** Integrates `SMOTE` (imbalanced-learn) to oversample genuine fraud classes before building an ensemble tree model optimized for extreme Recall logic.

#### `banking_model/graph_features.py`
*   **What it does:** The baseline network pre-processor used to statically extract edge counts (degrees of centrality, distinct paths) from historic CSVs.

### Testing & Validation

#### `test_adversarial_cases.py`
*   **What it does:** The absolute proving ground. Evaluates your system architecture against highly-coordinated attacks like Stealth Mules and Smurfing, explicitly reporting the dynamic JSON payload and verifying if the final score catches it.

#### `test_edge_case.py`
*   **What it does:** Unit tests for raw functional logic in trade finance scenarios.

### Datasets Utilized (`banking_model/data/`)
*   **`paysim.csv`**: Massive topological dataset simulating mobile money networks. Used to map real sender-to-receiver edges and compute hop counts.
*   **`ipblocklist.csv` (abuse.ch Feodo Tracker)**: Used for high-fidelity exact-match IP threat intelligence (Botnet C2 tracking).
*   **`firehol_level1.csv`**: Extensive CIDR blocklist mapping entire subnets notoriously associated with malware.
*   **`IpAddress_to_Country.csv`**: Provides the foundational mapping allowing dynamic geo-anomaly scoring.

*(Other legacy simulation scripts like `generate_data.py`, `feature_engineering.py`, and `predict.py` provide trade-finance foundation components)*
