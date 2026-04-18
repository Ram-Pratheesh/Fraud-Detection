import json
import numpy as np
from datetime import datetime

def explain_alert(vector_dict, row_dict, explainer, feature_names, rules_flagged, geo_context, network_features, blacklist_hits=None, amount_context=None):
    if blacklist_hits is None:
        blacklist_hits = {"blacklisted_ips": [], "blacklisted_accounts": [], "blacklisted_devices": [], "blacklisted_ip_count": 0, "blacklisted_account_count": 0}
    if amount_context is None:
        amount_context = {"value": float(row_dict.get('amount_inr', 0)), "currency": "USD", "typical_mean_amount": float(row_dict.get('amount_inr', 0))*0.4}

    # 1. Calculate SHAP Values
    vector_list = [vector_dict.get(f, 0) for f in feature_names]
    raw_shap = explainer.shap_values(np.array([vector_list]))
    
    # Handle both XGBClassifier (returns list of arrays) and Booster (returns ndarray) outputs
    if isinstance(raw_shap, list):
        shap_values = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
    elif raw_shap.ndim == 2:
        shap_values = raw_shap[0]
    else:
        shap_values = raw_shap
    
    # Base probability
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[-1]) if len(np.atleast_1d(base_value)) > 1 else float(np.atleast_1d(base_value)[0])
    
    logit = float(base_value) + float(shap_values.sum())
    ml_probability = 1 / (1 + np.exp(-logit))  # sigmoid
    
    # 2. Extract Top 3 Driving Explanations
    reasons = []
    top_indices = np.argsort(np.abs(shap_values))[-3:][::-1]
    
    top_features = {}
    for idx in top_indices:
        feat = feature_names[idx]
        val = vector_dict.get(feat, 0)
        s_val = shap_values[idx]
        top_features[feat] = round(float(s_val), 3)
        
        human_readable = f"Feature '{feat}' value {val:,.2f} shifted risk by {s_val:+.2f} log-odds"
        if "amount" in feat:
            ratio = amount_context['value'] / max(0.1, amount_context['typical_mean_amount'])
            human_readable = f"Transaction amount is {ratio:.2f}x the account's typical average — extreme high-value anomaly"
            top_features["amount_deviation_ratio"] = round(ratio, 2)
        elif "ip_shared_card_count" in feat:
            human_readable = f"Account belongs to a dense IP mapping graph (Money Mule indicator)"
            
        reasons.append({
            "feature": feat,
            "value": round(float(val), 2),
            "shap_value": round(float(s_val), 3),
            "human_text": human_readable
        })
        
    # Append the derived features from the target JSON if they weren't caught in Top 3
    if "amount_deviation_ratio" not in top_features:
        top_features["amount_deviation_ratio"] = round(amount_context['value'] / max(0.1, amount_context['typical_mean_amount']), 2)
    
    # ═══════════════════════════════════════════════════════
    # DYNAMIC SCORING ENGINE
    # ═══════════════════════════════════════════════════════
    
    # Compute anomaly sub-scores
    ip_anomaly_score = int(min(100, network_features.get('distinct_ips', 1) * 20))
    geo_anomaly_score = int(min(100, geo_context.get('distinct_countries', 1) * 30))
    amount_anomaly_score = int(min(100, (amount_context['value'] / max(1, amount_context['typical_mean_amount'])) * 10))
    
    # Network risk score — boosted if mule path + deep hops
    is_mule_path = vector_dict.get('ip_shared_card_count', 0) > 1
    hop_count = network_features.get("hop_count", 1)
    network_risk_score = round(min(1.0, vector_dict.get('degree_centrality', 0.0) * 10), 2)
    if is_mule_path and hop_count >= 2:
        network_risk_score = min(1.0, network_risk_score + 0.3)
    
    # Blacklist check
    has_blacklist_hit = blacklist_hits.get('blacklisted_ip_count', 0) > 0
    
    # Rule score from Layer 1 (if available), or derive from number of flagged rules
    rule_score = round(float(row_dict.get('rule_score', len(rules_flagged) * 0.25)), 2)
    rule_score = min(rule_score, 1.0)
    
    # ── Weighted Base Score ──
    base_score = 0.5 * ml_probability + 0.5 * rule_score
    
    # ── Dynamic Risk Boosting ──
    boost = 0.0
    boost_reasons = []
    
    if has_blacklist_hit:
        boost += 0.25
        boost_reasons.append("IP blacklisted (+0.25)")
    
    if is_mule_path:
        boost += 0.20
        boost_reasons.append("Mule path detected (+0.20)")
    
    if geo_anomaly_score > 80:
        boost += 0.10
        boost_reasons.append("Geo anomaly critical (+0.10)")
    
    if ip_anomaly_score > 80:
        boost += 0.10
        boost_reasons.append("IP anomaly critical (+0.10)")
    
    # Network risk contribution
    boost += 0.15 * network_risk_score
    if network_risk_score > 0:
        boost_reasons.append(f"Network risk ({network_risk_score:.2f} x 0.15)")
    
    final_score = round(min(1.0, base_score + boost), 3)
    
    # ── Meaningful Risk Buckets ──
    if final_score >= 0.85:
        risk_bucket = "CRITICAL"
    elif final_score >= 0.65:
        risk_bucket = "HIGH"
    elif final_score >= 0.4:
        risk_bucket = "MEDIUM"
    else:
        risk_bucket = "LOW"
    
    # ── Recommended Action ──
    if risk_bucket == "CRITICAL":
        recommended_action = "BLOCK immediately and escalate to compliance team"
    elif risk_bucket == "HIGH":
        recommended_action = "Flag for manual review within 1 hour"
    elif risk_bucket == "MEDIUM":
        recommended_action = "Monitor — add to watchlist for next 24 hours"
    else:
        recommended_action = "Allow — no immediate action required"
    
    # Store derived scores into top_features
    top_features["geo_anomaly_score"] = round(geo_anomaly_score / 100, 2)
    top_features["network_risk_score"] = network_risk_score
    top_features["is_mule_path"] = int(is_mule_path)
    top_features["blacklisted_ip_count"] = blacklist_hits.get('blacklisted_ip_count', 0)
        
    # 3. Build Module JSON
    payload = {
        "transaction_id": str(row_dict.get('transaction_id', f"TXN_{int(datetime.now().timestamp())}")),
        "account_id": str(row_dict.get('card_id', 'UNKNOWN')),
        "receiver_account_id": str(row_dict.get('receiver_account_id', 'UNKNOWN_RECEIVER')),
        "timestamp": str(row_dict.get('txn_timestamp', datetime.now().isoformat())),
        
        "network": {
            "ip_address": str(row_dict.get('_ip_dotted', row_dict.get('ip_address', 'UNKNOWN'))),
            "device_id": str(row_dict.get('device_fingerprint', 'UNKNOWN')),
            "distinct_ip_count_in_window": network_features.get('distinct_ips', 1),
            "typical_ip_count": 1,
            "ip_bucket_span_minutes": 60
        },
        
        "geography": {
            "country": geo_context.get('country', 'UNKNOWN'),
            "distinct_country_count_in_window": geo_context.get('distinct_countries', 1),
            "typical_country_count": 1
        },
        
        "amount": amount_context,
        
        "anomaly_scores": {
            "ip_anomaly_score": ip_anomaly_score,
            "geo_anomaly_score": geo_anomaly_score,
            "amount_anomaly_score": amount_anomaly_score,
            "overall_anomaly_score": int(final_score * 100)
        },
        
        "flags": {
            "is_multi_ip_anomaly": "check_ip_anomaly" in rules_flagged,
            "is_multi_country_anomaly": "check_geo_anomaly" in rules_flagged,
            "is_high_value_anomaly": "check_high_value_spike" in rules_flagged,
            "is_blacklisted": has_blacklist_hit,
            "is_mule_path": is_mule_path
        },
        
        "graph_data": {
            "nodes": {
                "sender_account": str(row_dict.get('card_id', 'UNKNOWN')),
                "receiver_account": str(row_dict.get('receiver_account_id', 'UNKNOWN_RECEIVER')),
                "ip_address": str(row_dict.get('_ip_dotted', row_dict.get('ip_address', 'UNKNOWN'))),
                "device_id": str(row_dict.get('device_fingerprint', 'UNKNOWN')),
                "country": geo_context.get('country', 'UNKNOWN')
            },
            "edges": [
                { "from": str(row_dict.get('card_id', 'UNKNOWN')), "to": str(row_dict.get('_ip_dotted', row_dict.get('ip_address', 'UNKNOWN'))), "relation": "used_ip", "frequency": int(network_features.get('distinct_ips', 1)) },
                { "from": str(row_dict.get('card_id', 'UNKNOWN')), "to": str(row_dict.get('device_fingerprint', 'UNKNOWN')), "relation": "used_device", "frequency": int(vector_dict.get('device_shared_card_count', 1)) },
                { "from": str(row_dict.get('card_id', 'UNKNOWN')), "to": str(row_dict.get('receiver_account_id', 'UNKNOWN_RECEIVER')), "relation": "sent_to", "amount": amount_context['value'] },
                { "from": str(row_dict.get('device_fingerprint', 'UNKNOWN')), "to": str(row_dict.get('receiver_account_id', 'UNKNOWN_RECEIVER')), "relation": "also_used_by", "frequency": int(vector_dict.get('device_shared_card_count', 1)) }
            ],
            "blacklist_hits": blacklist_hits,
            "network_features": {
                "shared_device_users": int(vector_dict.get('device_shared_card_count', 0)),
                "hop_count": hop_count,
                "is_mule_path": is_mule_path,
                "accounts_reachable_in_2_hops": network_features.get("accounts_reachable_in_2_hops", []),
                "distinct_ips_linked_to_account": network_features.get('distinct_ips_linked_to_account', 1),
                "distinct_countries_linked_to_account": geo_context.get('distinct_countries', 1)
            },
            "network_risk_score": network_risk_score
        },
        
        "fraud_score": {
            "ml_probability": round(float(ml_probability), 3),
            "rule_score": rule_score,
            "base_score": round(base_score, 3),
            "risk_boost": round(boost, 3),
            "boost_breakdown": boost_reasons,
            "final_score": final_score,
            "risk_bucket": risk_bucket,
            "threshold_used": 0.35,
            "top_features": top_features
        },
        
        "explanation": {
            "summary": "",  # Filled below
            "top_reasons": reasons,
            "lime_confidence": round(min(1.0, ml_probability + 0.05), 2),
            "recommended_action": recommended_action
        }
    }
    
    # Build the explanation summary with full context
    payload["explanation"]["summary"] = (
        f"Transaction {payload['transaction_id']} is {risk_bucket} risk (score: {final_score}). "
        f"The invoice amount is {top_features.get('amount_deviation_ratio', 1.0)}x the account's typical value, "
        f"originating from {geo_context.get('country', 'UNKNOWN')}"
        f"{' — BLACKLISTED IP DETECTED' if has_blacklist_hit else ''}"
        f"{' — routing through a known mule path' if is_mule_path else ''}."
    )
    
    return json.dumps(payload, indent=2)
