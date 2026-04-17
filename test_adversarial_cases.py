from rule_engine import TradeFraudRuleEngine
import json

engine = TradeFraudRuleEngine(api_key=None)

# ─────────────────────────────────────────────
# 🧪 1. PERFECTLY CAMOUFLAGED FRAUD
# (Everything looks normal individually)
# ─────────────────────────────────────────────
stealth_fraud = {
    "hs_code": "8471",
    "declared_value_usd": 52000,  # very close to real → no price flag
    "weight_kg": 1000,
    "goods_description": "computer processing units industrial grade",  # good description
    "invoice_count_per_bol": 1,
    "bol_weight": 1000,
    "bol_number": "BOL-STEALTH1",
    "port_arrival_record": True,
    "letter_of_credit": True,
    "packing_list": True,
    "is_duplicate_invoice": False,

    "origin_country": "SG",  # safe country
    "destination_country": "IN",
    "transit_port": "",
    "abnormal_route_flag": False,
    "further_shipment_records": True,
    "transshipment_count": 0,

    "paid_up_capital": 300000,
    "iec_age_days": 365,
    "mca_status": "active",
    "address_hash": "HASH-STEALTH",
    "shared_address_flag": True,  # ONLY signal
    "counterparty_name": "UNKNOWN SYSTEMS PTE LTD",
    "director_id": "DIR-STEALTH",
    "related_party_flag": True,   # subtle link

    "txn_count_24hr": 2,
    "current_month_txn_count": 12,
    "avg_monthly_txn_count": 10,
    "days_to_gst_period": 7,
    "export_txn_spike": False,
    "suspicious_filing_hour": 14,
    "repeat_shipment_count_30d": 2,
}

# ─────────────────────────────────────────────
# 🧪 2. SPLIT TRANSACTION FRAUD (SMURFING)
# (Below thresholds intentionally)
# ─────────────────────────────────────────────
smurfing_fraud = {
    "hs_code": "8471",
    "declared_value_usd": 15000,  # below suspicion threshold
    "weight_kg": 300,
    "goods_description": "electronics components",
    "invoice_count_per_bol": 1,
    "bol_weight": 300,
    "bol_number": "BOL-SMURF",
    "port_arrival_record": True,
    "letter_of_credit": True,
    "packing_list": True,
    "is_duplicate_invoice": False,

    "origin_country": "MY",
    "destination_country": "IN",
    "transit_port": "",
    "abnormal_route_flag": False,
    "further_shipment_records": True,
    "transshipment_count": 0,

    "paid_up_capital": 200000,
    "iec_age_days": 200,
    "mca_status": "active",
    "address_hash": "HASH-SMURF",
    "shared_address_flag": False,
    "counterparty_name": "LEGIT TECH SDN BHD",
    "director_id": "DIR-LEGIT",
    "related_party_flag": False,

    "txn_count_24hr": 1,
    "current_month_txn_count": 40,  # suspicious volume
    "avg_monthly_txn_count": 8,     # spike
    "days_to_gst_period": 2,
    "export_txn_spike": True,
    "suspicious_filing_hour": 11,
    "repeat_shipment_count_30d": 25,
}

# ─────────────────────────────────────────────
# 🧪 3. SEMANTIC CHEAT (FOOL NLP)
# ─────────────────────────────────────────────
semantic_attack = {
    "hs_code": "8471",
    "declared_value_usd": 50000,
    "weight_kg": 1000,
    "goods_description": "multi-purpose integrated digital solution hardware units",  
    # sounds legit but vague

    "invoice_count_per_bol": 1,
    "bol_weight": 1000,
    "bol_number": "BOL-NLP",
    "port_arrival_record": True,
    "letter_of_credit": True,
    "packing_list": True,
    "is_duplicate_invoice": False,

    "origin_country": "US",
    "destination_country": "IN",
    "transit_port": "",
    "abnormal_route_flag": False,
    "further_shipment_records": True,
    "transshipment_count": 0,

    "paid_up_capital": 500000,
    "iec_age_days": 800,
    "mca_status": "active",
    "address_hash": "HASH-NLP",
    "shared_address_flag": False,
    "counterparty_name": "TECH GLOBAL INC",
    "director_id": "DIR-NLP",
    "related_party_flag": False,

    "txn_count_24hr": 1,
    "current_month_txn_count": 6,
    "avg_monthly_txn_count": 5,
    "days_to_gst_period": 15,
    "export_txn_spike": False,
    "suspicious_filing_hour": 13,
    "repeat_shipment_count_30d": 1,
}

# ─────────────────────────────────────────────
# RUN TESTS
# ─────────────────────────────────────────────
def run_test(name, txn):
    print(f"\n{'='*60}")
    print(f"  TEST CASE: {name}")
    print(f"{'='*60}")
    result = engine.evaluate(txn)
    print(json.dumps(result, indent=2))

run_test("STEALTH FRAUD", stealth_fraud)
run_test("SMURFING FRAUD", smurfing_fraud)
run_test("SEMANTIC ATTACK", semantic_attack)