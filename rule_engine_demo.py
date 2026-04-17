"""
rule_engine_demo.py
Shows the rule engine firing on real generated transactions.
This IS Layer 1 of your fraud detection pipeline.
"""

import pandas as pd
import json
from generate_data import flag_transaction, HS_BENCHMARKS

def run_demo():
    print("Loading transactions...")
    df = pd.read_csv("output/transactions_flagged.csv")

    print(f"\n{'═'*60}")
    print("  FRAUD DETECTION — RULE ENGINE DEMO (Layer 1)")
    print(f"{'═'*60}")

    # ── Sample 1: Under-invoicing ──
    sample = df[df['fraud_type'] == 'under_invoicing'].iloc[0]
    result = flag_transaction(sample)
    print(f"\n[SAMPLE 1] Under-Invoicing Case")
    print(f"  HS Code:         {sample['hs_code']}")
    print(f"  Declared Price:  ${sample['declared_unit_price']:.2f}/unit")
    print(f"  Benchmark Price: ${sample['benchmark_unit_price']:.2f}/unit")
    print(f"  Deviation:       {sample['price_deviation_pct']:.1f}%")
    print(f"  FLAGS:           {result['flags']}")
    print(f"  RISK SCORE:      {result['risk_score']} → {result['risk_level']}")

    # ── Sample 2: Ghost Shipment ──
    sample2 = df[df['fraud_type'] == 'ghost_shipment'].iloc[0]
    result2 = flag_transaction(sample2)
    print(f"\n[SAMPLE 2] Ghost Shipment")
    print(f"  Weight (kg):     {sample2['weight_kg']}")
    print(f"  Declared Value:  ${sample2['declared_value_usd']:.2f}")
    print(f"  Origin:          {sample2['origin_country']} (risk: {sample2['origin_risk_score']})")
    print(f"  FLAGS:           {result2['flags']}")
    print(f"  RISK SCORE:      {result2['risk_score']} → {result2['risk_level']}")

    # ── Sample 3: Clean transaction (should have low risk) ──
    sample3 = df[df['fraud_type'] == 'clean'].iloc[0]
    result3 = flag_transaction(sample3)
    print(f"\n[SAMPLE 3] Clean Transaction")
    print(f"  HS Code:         {sample3['hs_code']}")
    print(f"  Declared Price:  ${sample3['declared_unit_price']:.2f}/unit")
    print(f"  Origin:          {sample3['origin_country']}")
    print(f"  FLAGS:           {result3['flags'] if result3['flags'] else 'None'}")
    print(f"  RISK SCORE:      {result3['risk_score']} → {result3['risk_level']}")

    # ── Aggregate stats ──
    print(f"\n{'─'*60}")
    print("  OVERALL PERFORMANCE")
    print(f"{'─'*60}")
    print(f"  Total transactions:     {len(df):,}")
    print(f"  Actual fraud:           {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
    print(f"\n  Risk Level Distribution:")
    print(df['risk_level'].value_counts().to_string())

    print(f"\n  Per fraud-type recall (% caught as HIGH risk):")
    for ftype in df['fraud_type'].unique():
        subset = df[df['fraud_type'] == ftype]
        high = (subset['risk_level'] == 'HIGH').sum()
        pct  = high / len(subset) * 100
        print(f"    {ftype:25s}: {high:4d}/{len(subset):4d} = {pct:5.1f}%")

    print(f"\n  Top flags fired:")
    from collections import Counter
    import ast
    all_flags = []
    for flags_str in df['flags'].dropna():
        try:
            all_flags.extend(ast.literal_eval(flags_str))
        except:
            pass
    for flag, count in Counter(all_flags).most_common(10):
        print(f"    {flag:45s}: {count:,}")

    print(f"\n{'═'*60}")
    print("  NEXT: Feed risk_score + flags as features into XGBoost")
    print("  FILE: output/transactions_flagged.csv  ← XGBoost training input")
    print(f"{'═'*60}\n")

if __name__ == "__main__":
    run_demo()