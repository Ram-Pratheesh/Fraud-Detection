import math
from data_loaders import (
    load_un_comtrade_benchmarks, 
    load_opensanctions,
    load_fatf_greylist,
    load_fatf_blacklist,
    load_pep_database,
    load_mca_struck_off
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_NLP = True
except ImportError:
    HAS_NLP = False


# Maximum possible baseline score (used to normalize risk between 0 and 1 without extreme deflation)
MAX_BASELINE_SCORE = 150.0

RULE_METADATA = {
    # ── Category 1: Invoice / Price Rules ──
    'under_invoicing': {
        'weight': 25,
        'category': 'price_flags',
        'source': 'FATF & Case Law',
        'confidence': 'high',
        'explanation': 'Declared value is significantly below market benchmark, indicative of Rule 12 duty evasion.'
    },
    'over_invoicing': {
        'weight': 20,
        'category': 'price_flags',
        'source': 'FATF',
        'confidence': 'high',
        'explanation': 'Declared value is unusually high, often linked to capital flight.'
    },
    'round_invoice': {
        'weight': 5,
        'category': 'price_flags',
        'source': 'Behavioral Analysis',
        'confidence': 'low',
        'explanation': 'Invoice amount is a round number, violating Benford\'s Law expectations.'
    },
    'multiple_invoices_same_shipment': {
        'weight': 15,
        'category': 'price_flags',
        'source': 'FATF',
        'confidence': 'medium',
        'explanation': 'Multiple invoices exist for a single shipment, common in TBML.'
    },
    'value_weight_mismatch': {
        'weight': 25,
        'category': 'price_flags',
        'source': 'FATF',
        'confidence': 'high',
        'explanation': 'Physical weight and declared value ratio deviates wildly from typical HS code norms.'
    },

    # ── Category 2: Document / Shipment Rules ──
    'bol_invoice_mismatch': {
        'weight': 30,
        'category': 'document_flags',
        'source': 'Case Law',
        'confidence': 'high',
        'explanation': 'Mismatch between shipment weight and invoice suggests document fraud or deliberate misdeclaration.'
    },
    'ghost_shipment': {
        'weight': 50,
        'category': 'document_flags',
        'source': 'FATF',
        'confidence': 'high',
        'explanation': 'Missing critical arrival records or Bill of Lading, suggesting the goods do not exist.'
    },
    'description_hs_mismatch': {
        'weight': 25,
        'category': 'document_flags',
        'source': 'Case Law',
        'confidence': 'high',
        'explanation': 'NLP Cosine Similarity indicates the description diverges completely from established HS standards.'
    },
    'vague_description': {
        'weight': 15,
        'category': 'document_flags',
        'source': 'Case Law',
        'confidence': 'medium',
        'explanation': 'Goods description is extremely vague, often used to conceal actual goods.'
    },
    'document_reuse': {
        'weight': 40,
        'category': 'document_flags',
        'source': 'FATF',
        'confidence': 'high',
        'explanation': 'Identical invoice numbers mapped across unrelated shipments.'
    },
    'missing_documents': {
        'weight': 20,
        'category': 'document_flags',
        'source': 'FATF',
        'confidence': 'medium',
        'explanation': 'Crucial verification documents like Packing Lists or LCs are absent.'
    },

    # ── Category 3: Route / Geography Rules ──
    'fatf_blacklist_country': {
        'weight': 100,
        'category': 'route_flags',
        'source': 'FATF',
        'confidence': 'high',
        'explanation': 'Transaction touches a jurisdiction on the FATF Blacklist.'
    },
    'fatf_greylist_country': {
        'weight': 35,
        'category': 'route_flags',
        'source': 'FATF',
        'confidence': 'high',
        'explanation': 'Transaction touches a jurisdiction under FATF AML monitoring.'
    },
    'high_risk_transshipment': {
        'weight': 25,
        'category': 'route_flags',
        'source': 'FATF',
        'confidence': 'medium',
        'explanation': 'Routing through high-risk transshipment hubs (e.g. Dubai, Hong Kong, Singapore).'
    },
    'abnormal_route': {
        'weight': 20,
        'category': 'route_flags',
        'source': 'FATF',
        'confidence': 'low',
        'explanation': 'Route is highly illogical or non-economical for this trade pair.'
    },
    'free_trade_zone_abuse': {
        'weight': 30,
        'category': 'route_flags',
        'source': 'FATF',
        'confidence': 'medium',
        'explanation': 'Utilization of Free Trade Zones with missing onward shipment records.'
    },
    'multi_country_routing': {
        'weight': 25,
        'category': 'route_flags',
        'source': 'FATF',
        'confidence': 'medium',
        'explanation': 'Shipment involves excessive country hops, typical of sanctions evasion.'
    },

    # ── Category 4: Entity Rules ──
    'shell_company': {
        'weight': 40,
        'category': 'entity_flags',
        'source': 'MCA',
        'confidence': 'high',
        'explanation': 'Corporate entity has paid-up capital totally disproportionate to transaction value.'
    },
    'new_entity_high_value': {
        'weight': 25,
        'category': 'entity_flags',
        'source': 'MCA',
        'confidence': 'medium',
        'explanation': 'Newly incorporated entity rapidly engaging in mass value trades.'
    },
    'struck_off_counterparty': {
        'weight': 50,
        'category': 'entity_flags',
        'source': 'MCA / DGFT',
        'confidence': 'high',
        'explanation': 'Counterparty registration has been struck off or invalidated.'
    },
    'shared_address_multiple_iec': {
        'weight': 30,
        'category': 'entity_flags',
        'source': 'DGFT',
        'confidence': 'high',
        'explanation': 'Corporate address is identical to suspicious clusters of unrelated entities.'
    },
    'sanctioned_entity': {
        'weight': 100,
        'category': 'entity_flags',
        'source': 'OpenSanctions / OFAC',
        'confidence': 'high',
        'explanation': 'Entity directly matching a global sanctions watchlist has been detected.'
    },
    'pep_connected': {
        'weight': 30,
        'category': 'entity_flags',
        'source': 'PEP Database',
        'confidence': 'medium',
        'explanation': 'Directorship is deeply tied to a Politically Exposed Person.'
    },
    'related_party_txn': {
        'weight': 35,
        'category': 'entity_flags',
        'source': 'MCA Analysis',
        'confidence': 'high',
        'explanation': 'Importer and Exporter possess interconnected board structures.'
    },

    # ── Category 5: Behavioral / Velocity Rules ──
    'smurfing': {
        'weight': 25,
        'category': 'behavior_flags',
        'source': 'Behavioral Analysis',
        'confidence': 'high',
        'explanation': 'Transaction velocity spiked beneath reporting thresholds.'
    },
    'sudden_volume_spike': {
        'weight': 15,
        'category': 'behavior_flags',
        'source': 'Behavioral Analysis',
        'confidence': 'low',
        'explanation': 'Drastic deviation from historical monthly transactional volume.'
    },
    'gst_refund_timing': {
        'weight': 20,
        'category': 'behavior_flags',
        'source': 'Customs',
        'confidence': 'medium',
        'explanation': 'Suspicious shipment timing directly preceding GST refund windows.'
    },
    'late_night_filing': {
        'weight': 10,
        'category': 'behavior_flags',
        'source': 'Behavioral Analysis',
        'confidence': 'low',
        'explanation': 'Documents filed in deep off-hours, bypassing manual oversight.'
    },
    'identical_repeat_shipments': {
        'weight': 20,
        'category': 'behavior_flags',
        'source': 'Behavioral Analysis',
        'confidence': 'medium',
        'explanation': 'Machine-like replication of identical invoice parameters.'
    },
    
    # ── Category 7 Meta / Intent ──
    'intent_inferred_multiple_discrepancies': {
        'weight': 15,
        'category': 'behavior_flags',
        'source': 'Case Law',
        'confidence': 'high',
        'explanation': 'The massive volume of cumulative inconsistencies establishes deliberate intent to defraud.'
    }
}


class TradeFraudRuleEngine:
    def __init__(self, api_key=None):
        self.hs_benchmarks = load_un_comtrade_benchmarks(api_key)
        self.sanctions_list = load_opensanctions()
        self.fatf_countries = load_fatf_greylist()
        self.fatf_black_countries = load_fatf_blacklist()
        self.pep_db = load_pep_database()
        self.struck_off_db = load_mca_struck_off()
        
        self.hs_vw_ratio_range = {
            "8471": (0.001, 0.05),
            "2709": (1.0, 5.0),
        }
        self.vectorizer = TfidfVectorizer() if HAS_NLP else None


    def evaluate(self, transaction: dict) -> dict:
        """
        Evaluate a single transaction across all FATF / Case Law categories.
        Returns newly refactored JSON payload.
        """
        raw_flags = {}
        
        raw_flags.update(self._price_rules(transaction))
        raw_flags.update(self._document_rules(transaction))
        raw_flags.update(self._route_rules(transaction))
        raw_flags.update(self._entity_rules(transaction))
        raw_flags.update(self._velocity_rules(transaction))
        
        # Meta rule: Intent inferred if 3 or more flags trigger
        if sum(raw_flags.values()) >= 3:
            raw_flags['intent_inferred_multiple_discrepancies'] = True

        # Extract only the active triggered flags
        triggered_flag_names = [k for k, v in raw_flags.items() if v]
        
        total_score = 0
        detailed_flags = []
        grouped_flags = {
            "price_flags": [],
            "document_flags": [],
            "entity_flags": [],
            "route_flags": [],
            "behavior_flags": []
        }
        explanations = []

        for flag in triggered_flag_names:
            meta = RULE_METADATA.get(flag)
            if not meta:
                continue
                
            total_score += meta['weight']
            
            flag_obj = {
                "flag": flag,
                "category": meta["category"],
                "source": meta["source"],
                "confidence": meta["confidence"]
            }
            detailed_flags.append(flag_obj)
            grouped_flags[meta['category']].append(flag)
            explanations.append(meta['explanation'])

        # Normalize Risk Score (bounded realistically logic, not arbitrarily 1.0)
        score = total_score / MAX_BASELINE_SCORE
        score = min(score, 1.0)
        
        risk_level = 'HIGH' if score >= 0.70 else ('MEDIUM' if score >= 0.35 else 'LOW')
        
        # Summary Generator strictly bound to risk_level
        if risk_level == "LOW":
            summary = "This transaction shows minimal risk signals and appears compliant."
        elif risk_level == "MEDIUM":
            summary = "This transaction shows moderate risk indicators and may require further review."
        elif risk_level == "HIGH":
            # Taking the first 3 explanations, making them lower-case for sentence flow (optional)
            top_explanations = [exp.split(",")[0].lower() for exp in explanations[:3]] if explanations else ["cumulative irregularities"]
            summary = "This transaction is HIGH RISK due to multiple red flags including: " + ", and ".join(top_explanations) + "."
        else:
            summary = "Status unknown."

        return {
            'risk_score': round(score, 4),
            'risk_level': risk_level,
            'flag_count': len(triggered_flag_names),
            
            # Formatted exactly as requested
            'flags': detailed_flags,
            'grouped_flags': grouped_flags,
            'explanations': explanations,
            'summary': summary
        }

    def _price_rules(self, txn: dict) -> dict:
        hs_code = str(txn.get('hs_code', ''))
        declared_value = txn.get('declared_value_usd', 0)
        weight_kg = txn.get('weight_kg', 0)
        
        benchmark_price, _ = self.hs_benchmarks.get(hs_code, (1.0, 0.5))
        declared_unit_price = declared_value / max(weight_kg, 0.1)
        
        vw_ratio = weight_kg / max(declared_value, 1)
        expected_range = self.hs_vw_ratio_range.get(hs_code, (0.01, 10.0))
        vw_mismatch = vw_ratio < expected_range[0] or vw_ratio > expected_range[1]

        return {
            'under_invoicing': declared_unit_price < benchmark_price * 0.5,
            'over_invoicing':  declared_unit_price > benchmark_price * 2.0,
            'round_invoice':   declared_value % 1000 == 0,
            'multiple_invoices_same_shipment': txn.get('invoice_count_per_bol', 1) > 1,
            'value_weight_mismatch': vw_mismatch,
        }

    def _document_rules(self, txn: dict) -> dict:
        bol_weight = txn.get('bol_weight')
        declared_weight = txn.get('weight_kg', 0)
        
        bol_num = txn.get('bol_number')
        port_arr = txn.get('port_arrival_record')
        ghost = bol_num is None or port_arr is None
        
        missing_docs = any(doc is None for doc in [
            txn.get('letter_of_credit'),
            bol_num,
            txn.get('packing_list')
        ])

        # NLP Description Mismatch
        desc = str(txn.get('goods_description', ''))
        desc_word_count = len(desc.split())
        vague = desc_word_count < 3

        desc_mismatch = False
        if HAS_NLP:
            standard_hs_desc = "computing machinery electronics processors" if txn.get('hs_code') == '8471' else "general merchandise"
            try:
                tfidf = self.vectorizer.fit_transform([desc, standard_hs_desc])
                sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
                desc_mismatch = float(sim) < 0.4
            except ValueError:
                desc_mismatch = True

        # Incorporating the logic without duplicates
        bol_mismatch = False
        if bol_weight is not None and declared_weight > 0:
            if abs(declared_weight - bol_weight) > 0.2 * declared_weight:
                bol_mismatch = True

        return {
            'bol_invoice_mismatch': bol_mismatch,
            'ghost_shipment': ghost,
            'description_hs_mismatch': desc_mismatch,
            'document_reuse': txn.get('is_duplicate_invoice', False),
            'missing_documents': missing_docs,
            'vague_description': vague,
        }

    def _route_rules(self, txn: dict) -> dict:
        origin = txn.get('origin_country', '')
        dest = txn.get('destination_country', '')
        transit = txn.get('transit_port', '')
        high_risk_transit = ["AE", "HK", "SG"] 
        
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
