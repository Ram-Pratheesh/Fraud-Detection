from rule_engine import TradeFraudRuleEngine
import json

# Initialize the Rule Engine (it will load the mock watchlists and hit the Comtrade API)
engine = TradeFraudRuleEngine(api_key=None)

# ── 1. A CLEAN, Normal Transaction ──
clean_txn = {
    "hs_code": "8471", # Computers
    "declared_value_usd": 85000, 
    "weight_kg": 1000,
    "goods_description": "computing machinery electronics processors",
    "invoice_count_per_bol": 1,
    "bol_weight": 1000,
    "bol_number": "BOL-9999XYZ",
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
    "paid_up_capital": 5000000,
    "iec_age_days": 1000,
    "mca_status": "active",
    "address_hash": "HASH-CLEAN1",
    "shared_address_flag": False,
    "counterparty_name": "APPLE INC",
    "director_id": "DIR-0001",
    "related_party_flag": False,
    "txn_count_24hr": 1,
    "current_month_txn_count": 5,
    "avg_monthly_txn_count": 5,
    "days_to_gst_period": 14,
    "export_txn_spike": False,
    "suspicious_filing_hour": 14, # 2 PM
    "repeat_shipment_count_30d": 1,
}

# ── 2. A HIGHLY SUSPICIOUS Transaction (Trade Based Money Laundering) ──
fraud_txn = {
    "hs_code": "8471", # Computers
    # FRAUD 1: Extreme under-invoicing (claiming $2,000 for 1000kg of computers)
    "declared_value_usd": 2000, 
    "weight_kg": 1000,
    # FRAUD 2: Vague description & doesn't match HS code 8471 via Cosine Similarity
    "goods_description": "assorted mixed items", 
    "invoice_count_per_bol": 1,
    # FRAUD 3: Bill of lading weight mismatch (declared 1000kg, but BOL says 500kg)
    "bol_weight": 500,
    "bol_number": "BOL-FAKE123",
    "port_arrival_record": True,
    "letter_of_credit": None, # Missing Document
    "packing_list": True,
    "is_duplicate_invoice": False,
    # FRAUD 4: FATF Sanctioned Origin
    "origin_country": "IR", # Iran (On FATF Blacklist)
    "destination_country": "IN",
    # FRAUD 5: High Risk Transshipment
    "transit_port": "AE", # UAE/Dubai
    "abnormal_route_flag": False,
    "further_shipment_records": True,
    "transshipment_count": 1,
    "paid_up_capital": 50000, # Shell company vibes
    "iec_age_days": 30, # Brand new company
    "mca_status": "active",
    "address_hash": "HASH-SHADY2",
    "shared_address_flag": False,
    # FRAUD 6: Hit against our OpenSanctions live CSV (Real sanctioned entity from your data!)
    "counterparty_name": "MICHAEL DAVID MUMMERT",
    # FRAUD 7: Hit against our PEP mock list
    "director_id": "DIR-99381",
    "related_party_flag": True, # Exporter and Importer same directors
    "txn_count_24hr": 1,
    "current_month_txn_count": 1,
    "avg_monthly_txn_count": 1,
    "days_to_gst_period": 14,
    "export_txn_spike": False,
    # FRAUD 8: Abnormal filing behavior
    "suspicious_filing_hour": 3, # 3 AM
    "repeat_shipment_count_30d": 1,
}

print("======================================================")
print("  EVALUATING CLEAN TRANSACTION")
print("======================================================")
clean_result = engine.evaluate(clean_txn)
print(json.dumps(clean_result, indent=2))
print()

print("======================================================")
print("  EVALUATING FRAUD TRANSACTION")
print("======================================================")
fraud_result = engine.evaluate(fraud_txn)
print(json.dumps(fraud_result, indent=2))
print("======================================================")
