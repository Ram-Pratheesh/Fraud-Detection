import json
import joblib
import os
import sys

# 1. Boot up the Trade Docs Rule Engine (Legacy Layer 1)
from rule_engine import TradeFraudRuleEngine
rule_engine = TradeFraudRuleEngine(api_key=None)

# 2. Boot up the Graph Network Engine (ML Layer 2)
sys.path.append(os.path.join(os.path.dirname(__file__), 'banking_model'))
from realtime_engine import analyze_transaction_core, load_geography_tree, load_blacklist_datasets, load_paysim_graph

# Load Geo Mappings, Blacklists, PaySim Graph, and ML Models
load_geography_tree()
load_blacklist_datasets()
load_paysim_graph()
_base = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(_base, "banking_model", "models")
try:
    xgb_explainer = joblib.load(os.path.join(model_dir, "banking_shap_explainer.pkl"))
    ml_features = joblib.load(os.path.join(model_dir, "banking_features.pkl"))
except:
    print("Warning: Run `python banking_model/train_banking_model.py` first to generate ML artifacts.")
    xgb_explainer = None
    ml_features = []

# ─────────────────────────────────────────────
# 🧪 1. PERFECTLY CAMOUFLAGED FRAUD
# (Perfect Paperwork → Rule Engine FAILS)
# (Same IP Used By 20 Cards, from Russia → Graph Engine CATCHES)
# ─────────────────────────────────────────────
stealth_fraud = {
    # --- Trade Paperwork (Perfect) ---
    "hs_code": "8471",
    "declared_value_usd": 52000,  
    "weight_kg": 1000,
    "goods_description": "computer processing units industrial grade",  
    "invoice_count_per_bol": 1,
    "abnormal_route_flag": False,
    "shared_address_flag": True,  
    
    # --- Graph Telemetry (Exposes the Fraudster) ---
    "ip_address": "185.243.103.246",  # Realistic public IP (Russian proxy)
    "device_fingerprint": "DEV-SHADOW-1",
    "card_id": "STEALTH-CARD-XX",
    "amount_inr": 52000 * 83,  # Converting USD to INR roughly
    "ip_shared_card_count": 20, # Topological flag: 20 other users share this IP!
    "degree_centrality": 0.91,
    "_distinct_ips": 5, # Multi-IP proxy usage
    "_distinct_countries": 3 # Flown through 3 countries virtually
}

# ─────────────────────────────────────────────
# 🧪 2. SPLIT TRANSACTION FRAUD (SMURFING)
# (Amounts intentionally small to bypass limits)
# ─────────────────────────────────────────────
smurfing_fraud = {
    "hs_code": "8471",
    "declared_value_usd": 15000,  # below suspicion threshold
    "current_month_txn_count": 40,  
    "export_txn_spike": True,
    
    # Graph Engine context
    "ip_address": "103.45.67.89",  # Realistic public IP (Asian proxy)
    "card_id": "SMURF-CARD-ZZ",
    "amount_inr": 15000 * 83,
    "device_shared_card_count": 5, # The exact same phone used by 5 accounts!
}

# ─────────────────────────────────────────────
# 🧪 3. SEMANTIC CHEAT (FOOL NLP)
# ─────────────────────────────────────────────
semantic_attack = {
    "hs_code": "8471",
    "declared_value_usd": 50000,
    "weight_kg": 1000,
    "goods_description": "multi-purpose integrated digital solution hardware units",  
    "suspicious_filing_hour": 13,
    
    # Graph Engine context (Caught bypassing KYC)
    "ip_address": "91.215.40.22",  # Realistic public IP (Eastern Europe proxy)
    "card_id": "NLP-CARD-WW",
    "amount_inr": 50000 * 83,
    "ip_shared_card_count": 8, # Network Topology Red Flag
}

# ─────────────────────────────────────────────
# 🧪 4. BLACKLIST VALIDATION (Known Feodo C2 IP)
# IP 50.16.16.211 is in ipblocklist.csv (QakBot C2)
# Must trigger blacklist_hits > 0
# ─────────────────────────────────────────────
blacklist_test = {
    "hs_code": "8471",
    "declared_value_usd": 48000,
    "weight_kg": 900,
    "goods_description": "server rack equipment industrial",
    "invoice_count_per_bol": 1,
    
    # Graph Engine context — uses the KNOWN Feodo IP
    "ip_address": "50.16.16.211",   # Dotted-quad: known QakBot C2 from ipblocklist.csv
    "device_fingerprint": "DEV-FEODO-TEST",
    "card_id": "BLACKLIST-CARD-TEST",
    "amount_inr": 48000 * 83,
    "ip_shared_card_count": 10,      # Trigger graph flags to reach Layer 2
    "_distinct_ips": 4,
    "_distinct_countries": 3
}

# ─────────────────────────────────────────────
# 🧪 5. FIREHOL CIDR VALIDATION
# IP 2.57.122.5 falls inside FireHOL CIDR 2.57.122.0/24
# Must trigger blacklist_hits > 0 via CIDR range matching
# ─────────────────────────────────────────────
firehol_test = {
    "hs_code": "8471",
    "declared_value_usd": 39000,
    "weight_kg": 750,
    "goods_description": "network switching equipment",
    "invoice_count_per_bol": 1,
    
    "ip_address": "2.57.122.5",     # Falls inside FireHOL CIDR 2.57.122.0/24
    "device_fingerprint": "DEV-FIREHOL-TEST",
    "card_id": "FIREHOL-CARD-TEST",
    "amount_inr": 39000 * 83,
    "ip_shared_card_count": 8,
    "_distinct_ips": 4,
    "_distinct_countries": 3
}

# ─────────────────────────────────────────────
# 🧪 6. CLEAN IP VALIDATION (Negative Test)
# IP 8.8.8.8 (Google DNS) must NOT trigger blacklist
# ─────────────────────────────────────────────
clean_ip_test = {
    "hs_code": "8471",
    "declared_value_usd": 25000,
    "weight_kg": 500,
    "goods_description": "standard computing hardware",
    "invoice_count_per_bol": 1,
    
    "ip_address": "8.8.8.8",        # Google DNS — known clean IP
    "device_fingerprint": "DEV-CLEAN-TEST",
    "card_id": "CLEAN-CARD-TEST",
    "amount_inr": 25000 * 83,
    "ip_shared_card_count": 6,
    "_distinct_ips": 4,
    "_distinct_countries": 3
}

# ─────────────────────────────────────────────
# RUN DUAL-LAYER TESTS
# ─────────────────────────────────────────────
def run_dual_test(name, txn):
    print(f"\\n{'='*80}")
    print(f" 🛡️  TEST CASE: {name}")
    print(f"{'='*80}")
    
    print("\\n[LAYER 1: Trade Document Rule Engine]")
    rule_result = rule_engine.evaluate(txn)
    rule_score = rule_result.get("risk_score", 0)
    print(f"  > Risk Score: {rule_score}/100")
    if rule_score < 70:
        print("  > ❌ STATUS: PASSED / UNDETECTED (Paperwork looked perfect)")
    else:
        print("  > ✅ STATUS: CAUGHT")

    print("\\n[LAYER 2: Banking Graph Network Engine]")
    if xgb_explainer:
        is_fraud, explanation_json = analyze_transaction_core(txn, xgb_explainer, ml_features)
        
        if is_fraud:
            print("  > 🚨 STATUS: CRITICAL FRAUD INTERCEPTED!")
            print("  > 🧠 AI EXPLANATION JSON PAYLOAD:")
            print(explanation_json)
        else:
            print("  > STATUS: CLEAN")
    else:
        print("  > Skipping ML Layer (Artifacts missing)")

if __name__ == "__main__":
    run_dual_test("STEALTH FRAUD (Camouflaged Money Mule)", stealth_fraud)
    run_dual_test("SMURFING FRAUD (Burst Velocity)", smurfing_fraud)
    run_dual_test("SEMANTIC ATTACK (Invoice NLP Evasion)", semantic_attack)
    print("\\n" + "█"*80)
    print(" 🔍  BLACKLIST VALIDATION SUITE")
    print("█"*80)
    run_dual_test("FEODO HIT (Known QakBot C2 IP)", blacklist_test)
    run_dual_test("FIREHOL HIT (CIDR Range Match)", firehol_test)
    run_dual_test("CLEAN IP (Must NOT Trigger Blacklist)", clean_ip_test)