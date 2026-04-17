import pandas as pd

df = pd.read_csv("output/transactions_flagged.csv")

df["price_dev"] = (df["declared_unit_price"] - df["benchmark_unit_price"]) / df["benchmark_unit_price"]
df["value_per_kg"] = df["declared_value_usd"] / (df["weight_kg"] + 1e-6)
df["high_risk_country"] = (df["origin_risk_score"] > 0.8).astype(int)
df["zero_weight_flag"] = (df["weight_kg"] == 0).astype(int)

df["rule_score"] = df["risk_score"]
df["num_flags"] = df["flags"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)

df["label"] = df["is_fraud"]

# ── ADVANCED FEATURES ──────────────────────────────────────────

# 1. Invoice frequency per exporter IEC
iec_counts = df["exporter_iec"].value_counts()
df["invoice_frequency_per_iec"] = df["exporter_iec"].map(iec_counts).fillna(0).astype(int)

# 2. Average declared unit price per HS code
avg_price_map = df.groupby("hs_code")["declared_unit_price"].mean()
df["avg_price_per_hs_code"] = df["hs_code"].map(avg_price_map).fillna(0.0)

# 3. Night transaction flag (hour between 2 and 4 inclusive)
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df["transaction_hour"] = df["timestamp"].dt.hour.fillna(-1).astype(int)
df["is_night_transaction"] = df["transaction_hour"].apply(lambda h: 1 if 2 <= h <= 4 else 0)

# 4. Duplicate invoice flag (invoice_no appears more than once)
inv_counts = df["invoice_no"].value_counts()
df["duplicate_invoice_flag"] = df["invoice_no"].map(inv_counts).fillna(1).apply(lambda x: 1 if x > 1 else 0)

# 5. Description length (word count in goods_description)
df["description_length"] = df["goods_description"].fillna("").apply(lambda x: len(str(x).split()))

# 6. Vague description flag
VAGUE_WORDS = ["misc", "goods", "items", "products", "assorted"]
df["is_vague_description"] = df["goods_description"].fillna("").apply(
    lambda x: 1 if any(w in str(x).lower() for w in VAGUE_WORDS) else 0
)

# ── END ADVANCED FEATURES ──────────────────────────────────────

df.to_csv("output/features.csv", index=False)

print("features.csv created with advanced features")