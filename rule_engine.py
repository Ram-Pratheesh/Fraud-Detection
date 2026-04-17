import math
from data_loaders import (
    load_un_comtrade_benchmarks, 
    load_opensanctions,
    load_fatf_greylist,
    load_fatf_blacklist,
    load_pep_database,
    load_mca_struck_off
)

# Optional NLP for descriptions (lightweight fallback if scikit-learn is missing)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_NLP = True
except ImportError:
    HAS_NLP = False


# The Boss's Recommended Rule Weights
RULE_WEIGHTS = {
    # Category 1
    'under_invoicing': 20,
    'over_invoicing': 20,
    'round_invoice': 5,
    'multiple_invoices_same_shipment': 15,
    'vague_description': 10,
    'value_weight_mismatch': 25,
    
    # Category 2
    'bol_invoice_mismatch': 30,
    'ghost_shipment': 50,
    'description_hs_mismatch': 25,
    'document_reuse': 40,
    'missing_documents': 20,
    
    # Category 3
    'fatf_greylist_country': 35,
    'fatf_blacklist_country': 100,
    'high_risk_transshipment': 25,
    'abnormal_route': 20,
    'free_trade_zone_abuse': 30,
    'multi_country_routing': 25,
    
    # Category 4
    'shell_company': 40,
    'new_entity_high_value': 25,
    'struck_off_counterparty': 50,
    'shared_address_multiple_iec': 30,
    'sanctioned_entity': 100,  # Auto Critical
    'pep_connected': 30,
    'related_party_txn': 35,
    
    # Category 5
    'smurfing': 25,
    'sudden_volume_spike': 15,
    'gst_refund_timing': 20,
    'late_night_filing': 10,
    'identical_repeat_shipments': 20,
    
    # Category 7 - Indian Case Law (Kanoon / CESTAT)
    'undervaluation_rule12_pattern': 25,
    'misdeclaration_pattern': 20,
    'bol_invoice_mismatch_pattern': 25,
    'intent_inferred_multiple_discrepancies': 15,
}


class TradeFraudRuleEngine:
    def __init__(self, api_key=None):
        print("[INIT] Loading Data Sources for TradeFraudRuleEngine...")
        self.hs_benchmarks = load_un_comtrade_benchmarks(api_key)
        self.sanctions_list = load_opensanctions()
        self.fatf_countries = load_fatf_greylist()
        self.fatf_black_countries = load_fatf_blacklist()
        self.pep_db = load_pep_database()
        self.struck_off_db = load_mca_struck_off()
        
        # Hardcoded expected weight/value ratios (kg per USD) based on HS groups
        self.hs_vw_ratio_range = {
            "8471": (0.001, 0.05), # Electronics (high value, low weight)
            "2709": (1.0, 5.0),    # Oil/bulk (low value, high weight)
        }
        
        self.vectorizer = TfidfVectorizer() if HAS_NLP else None

    def evaluate(self, transaction: dict) -> dict:
        """
        Evaluate a single transaction across all 6 FATF categories.
        """
        flags = {}
        
        flags.update(self._price_rules(transaction))
        flags.update(self._document_rules(transaction))
        flags.update(self._route_rules(transaction))
        flags.update(self._entity_rules(transaction))
        flags.update(self._velocity_rules(transaction))
        # Optional: flags.update(self._card_rules(transaction))
        
        # Meta-Rules (Case Law relying on other flags)
        flags.update(self._case_law_rules(transaction, flags))
        
        # Calculate Weighted Score
        score = sum(RULE_WEIGHTS.get(f, 10) for f, is_flagged in flags.items() if is_flagged)
        
        # Cap score at 100
        score = min(score, 100)
        
        return {
            'flags': [k for k, v in flags.items() if v],
            'flag_count': sum(flags.values()),
            'risk_score': score / 100.0, # Normalize to 0-1 for XGBoost
            'risk_level': 'HIGH' if score > 70 else ('MEDIUM' if score > 35 else 'LOW')
        }

    def _price_rules(self, txn: dict) -> dict:
        hs_code = str(txn.get('hs_code', ''))
        declared_value = txn.get('declared_value_usd', 0)
        weight_kg = txn.get('weight_kg', 0)
        
        benchmark_price, _ = self.hs_benchmarks.get(hs_code, (1.0, 0.5))
        declared_unit_price = declared_value / max(weight_kg, 0.1)
        
        desc = str(txn.get('goods_description', ''))
        desc_word_count = len(desc.split())
        
        # Value-weight mismatch
        vw_ratio = weight_kg / max(declared_value, 1)
        expected_range = self.hs_vw_ratio_range.get(hs_code, (0.01, 10.0))
        vw_mismatch = vw_ratio < expected_range[0] or vw_ratio > expected_range[1]

        return {
            'under_invoicing': declared_unit_price < benchmark_price * 0.5,
            'over_invoicing':  declared_unit_price > benchmark_price * 2.0,
            'round_invoice':   declared_value % 1000 == 0,
            'multiple_invoices_same_shipment': txn.get('invoice_count_per_bol', 1) > 1,
            'vague_description': desc_word_count < 3,
            'value_weight_mismatch': vw_mismatch,
        }

    def _document_rules(self, txn: dict) -> dict:
        bol_weight = txn.get('bol_weight')
        declared_weight = txn.get('weight_kg')
        
        # Ghost shipment
        bol_num = txn.get('bol_number')
        port_arr = txn.get('port_arrival_record')
        ghost = bol_num is None or port_arr is None
        
        # Missing docs
        missing_docs = any(doc is None for doc in [
            txn.get('letter_of_credit'),
            bol_num,
            txn.get('packing_list')
        ])

        # NLP Description Mismatch (Cosine Sim)
        desc_mismatch = False
        if HAS_NLP:
            standard_hs_desc = "computing machinery electronics processors" if txn.get('hs_code') == '8471' else "general merchandise"
            try:
                tfidf = self.vectorizer.fit_transform([txn.get('goods_description', ''), standard_hs_desc])
                sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                desc_mismatch = float(sim) < 0.4
            except ValueError:
                desc_mismatch = True

        return {
            'bol_invoice_mismatch': bol_weight != declared_weight if bol_weight is not None else False,
            'ghost_shipment': ghost,
            'description_hs_mismatch': desc_mismatch,
            'document_reuse': txn.get('is_duplicate_invoice', False),
            'missing_documents': missing_docs,
        }

    def _route_rules(self, txn: dict) -> dict:
        origin = txn.get('origin_country', '')
        dest = txn.get('destination_country', '')
        transit = txn.get('transit_port', '')
        
        high_risk_transit = ["AE", "HK", "SG"] # Dubai, Hong Kong, Singapore
        
        return {
            'fatf_blacklist_country': origin in self.fatf_black_countries or dest in self.fatf_black_countries,
            'fatf_greylist_country': origin in self.fatf_countries or dest in self.fatf_countries,
            'high_risk_transshipment': transit in high_risk_transit and (origin in self.fatf_countries or origin in self.fatf_black_countries),
            'abnormal_route': txn.get('abnormal_route_flag', False),
            'free_trade_zone_abuse': dest in ["FTZ_1", "FTZ_2"] and not txn.get('further_shipment_records', True),
            'multi_country_routing': txn.get('transshipment_count', 0) > 2,
        }

    def _entity_rules(self, txn: dict) -> dict:
        val = txn.get('declared_value_usd', 0)
        
        return {
            'shell_company': txn.get('paid_up_capital', 999999) < 100000 and val > 1000000,
            'new_entity_high_value': txn.get('iec_age_days', 999) < 180 and val > 500000,
            'struck_off_counterparty': txn.get('mca_status', '') == 'struck_off' or txn.get('address_hash') in self.struck_off_db,
            'shared_address_multiple_iec': txn.get('shared_address_flag', False),
            'sanctioned_entity': txn.get('counterparty_name', '').upper() in self.sanctions_list.get('OFAC_UN_ED_LIST', []),
            'pep_connected': txn.get('director_id', '') in self.pep_db,
            'related_party_txn': txn.get('related_party_flag', False),
        }

    def _velocity_rules(self, txn: dict) -> dict:
        return {
            'smurfing': txn.get('txn_count_24hr', 0) > 5 and txn.get('declared_value_usd', 0) < 10000,
            'sudden_volume_spike': txn.get('current_month_txn_count', 1) > 3 * txn.get('avg_monthly_txn_count', 1),
            'gst_refund_timing': txn.get('days_to_gst_period', 15) < 7 and txn.get('export_txn_spike', False),
            'late_night_filing': 1 <= txn.get('suspicious_filing_hour', 12) <= 5,
            'identical_repeat_shipments': txn.get('repeat_shipment_count_30d', 0) > 3,
        }

    def _case_law_rules(self, txn: dict, current_flags: dict) -> dict:
        """
        Category 7: Derived from Indian Kanoon / CESTAT Adjudication Orders.
        Checks for legal patterns of intentional fraud established by judges.
        """
        # Case 1: Rule 12 Undervaluation Pattern
        under_inv = current_flags.get('under_invoicing', False)
        missing_docs = current_flags.get('missing_documents', False)
        rule12_pattern = under_inv and missing_docs
        
        # Case 2: Misdeclaration Pattern
        vague = current_flags.get('vague_description', False)
        desc_mismatch = current_flags.get('description_hs_mismatch', False)
        misdec_pattern = vague or desc_mismatch
        
        # Case 3: Significant BOL vs Invoice Mismatch (>20%)
        decl_weight = txn.get('weight_kg', 0)
        bol_weight = txn.get('bol_weight')
        bol_pattern = False
        if bol_weight is not None and decl_weight > 0:
            if abs(decl_weight - bol_weight) > 0.2 * decl_weight:
                bol_pattern = True
                
        # Case 4: Deliberate intent inferred from multiple discrepancies
        intent_pattern = sum(current_flags.values()) >= 3
        
        return {
            'undervaluation_rule12_pattern': rule12_pattern,
            'misdeclaration_pattern': misdec_pattern,
            'bol_invoice_mismatch_pattern': bol_pattern,
            'intent_inferred_multiple_discrepancies': intent_pattern,
        }
