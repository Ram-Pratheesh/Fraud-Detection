from rule_engine import TradeFraudRuleEngine
import json

engine = TradeFraudRuleEngine(api_key=None)

# ─────────────────────────────────────────────
# 🟡 1. MEDIUM RISK (Borderline Suspicious)
# ─────────────────────────────────────────────
medium_txn = {
    "hs_code": "8471",
    "declared_value_usd": 30000,   # slightly low vs real
    "weight_kg": 1000,
    "goods_description": "computer accessories parts",
    "invoice_count_per_bol": 1,
    "bol_weight": 950,  # small mismatch
    "bol_number": "BOL-MED123",
    "port_arrival_record": True,
    "letter_of_credit": True,
    "packing_list": True,
    "is_duplicate_invoice": False,
    "origin_country": "AE",  # moderate risk
    "destination_country": "IN",
    "transit_port": "",
    "abnormal_route_flag": False,
    "further_shipment_records": True,
    "transshipment_count": 0,
    "paid_up_capital": 200000,
    "iec_age_days": 120,
    "mca_status": "active",
    "address_hash": "HASH-MED1",
    "shared_address_flag": False,
    "counterparty_name": "GLOBAL ELECTRONICS LLC",
    "director_id": "DIR-2001",
    "related_party_flag": False,
    "txn_count_24hr": 2,
    "current_month_txn_count": 10,
    "avg_monthly_txn_count": 8,
    "days_to_gst_period": 5,
    "export_txn_spike": False,
    "suspicious_filing_hour": 11,
    "repeat_shipment_count_30d": 3,
}

# ─────────────────────────────────────────────
# 🔴 2. SMART FRAUD (Hard to detect)
# ─────────────────────────────────────────────
smart_fraud_txn = {
    "hs_code": "8471",
    "declared_value_usd": 45000,  # close to real (not obvious)
    "weight_kg": 1000,
    "goods_description": "electronic devices",
    "invoice_count_per_bol": 2,  # splitting invoices
    "bol_weight": 1000,
    "bol_number": "BOL-SMART999",
    "port_arrival_record": True,
    "letter_of_credit": True,
    "packing_list": True,
    "is_duplicate_invoice": True,  # key signal
    "origin_country": "SG",
    "destination_country": "IN",
    "transit_port": "AE",
    "abnormal_route_flag": True,
    "further_shipment_records": True,
    "transshipment_count": 2,
    "paid_up_capital": 80000,  # weak company
    "iec_age_days": 45,
    "mca_status": "active",
    "address_hash": "HASH-SMART2",
    "shared_address_flag": True,
    "counterparty_name": "UNKNOWN TRADING CO",
    "director_id": "DIR-3002",
    "related_party_flag": True,
    "txn_count_24hr": 5,
    "current_month_txn_count": 30,
    "avg_monthly_txn_count": 10,  # spike
    "days_to_gst_period": 2,
    "export_txn_spike": True,
    "suspicious_filing_hour": 3,
    "repeat_shipment_count_30d": 10,
}

# ─────────────────────────────────────────────
# 🟢 3. CLEAN BUT UNUSUAL (False Positive Test)
# ─────────────────────────────────────────────
clean_unusual_txn = {
    "hs_code": "1001",
    "declared_value_usd": 200000,
    "weight_kg": 5000,
    "goods_description": "high quality wheat export bulk shipment",
    "invoice_count_per_bol": 1,
    "bol_weight": 5000,
    "bol_number": "BOL-CLEAN777",
    "port_arrival_record": True,
    "letter_of_credit": True,
    "packing_list": True,
    "is_duplicate_invoice": False,
    "origin_country": "IN",
    "destination_country": "AE",
    "transit_port": "",
    "abnormal_route_flag": False,
    "further_shipment_records": True,
    "transshipment_count": 0,
    "paid_up_capital": 10000000,
    "iec_age_days": 2000,
    "mca_status": "active",
    "address_hash": "HASH-CLEANX",
    "shared_address_flag": False,
    "counterparty_name": "RELIANCE EXPORTS",
    "director_id": "DIR-9001",
    "related_party_flag": False,
    "txn_count_24hr": 1,
    "current_month_txn_count": 50,
    "avg_monthly_txn_count": 48,
    "days_to_gst_period": 20,
    "export_txn_spike": False,
    "suspicious_filing_hour": 13,
    "repeat_shipment_count_30d": 20,
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

run_test("MEDIUM RISK", medium_txn)
run_test("SMART FRAUD", smart_fraud_txn)
run_test("CLEAN BUT UNUSUAL", clean_unusual_txn)