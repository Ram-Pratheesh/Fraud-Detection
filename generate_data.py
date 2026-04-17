"""
Fraud Detection - Synthetic Data Generator
Mirrors msmeonline.tn.gov.in schema + fraud detection extensions
Run this locally (site blocks datacenter IPs - works fine from your machine)

Scraper for when you run locally:
    python scrape_msme.py   <-- generates real_msme_seed.csv
    python generate_data.py <-- uses seed if present, else pure synthetic
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
import json
import os
from datetime import datetime, timedelta

fake = Faker('en_IN')
random.seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# REAL ANCHOR DATA  (from UN Comtrade + CBIC knowledge)
# ─────────────────────────────────────────────

HS_BENCHMARKS = {
    # hs_code: (avg_usd_per_kg, std_dev)
    "6204": (4.5,  1.2),    # Women's garments
    "6110": (5.2,  1.5),    # Jerseys/pullovers
    "8471": (85.0, 20.0),   # Computers
    "8517": (45.0, 15.0),   # Phones/telecom
    "2709": (0.55, 0.1),    # Crude oil
    "7108": (55000, 5000),  # Gold
    "0901": (3.2,  0.8),    # Coffee
    "5201": (1.8,  0.4),    # Cotton
    "3004": (25.0, 8.0),    # Medicines
    "8703": (12000, 3000),  # Cars
    "6403": (8.5,  2.0),    # Footwear
    "0803": (0.4,  0.1),    # Bananas
    "2601": (0.12, 0.03),   # Iron ore
    "7204": (0.22, 0.05),   # Steel scrap
    "8542": (150.0, 40.0),  # Semiconductors
}

HIGH_RISK_COUNTRIES = {
    # FATF greylist + sanctions + known trade fraud hubs
    "PK": 0.9,   # Pakistan - FATF grey
    "MM": 0.85,  # Myanmar - FATF grey
    "SY": 0.95,  # Syria - sanctions
    "IR": 0.98,  # Iran - sanctions
    "KP": 1.0,   # North Korea - full sanctions
    "NG": 0.7,   # Nigeria - fraud risk
    "AE": 0.5,   # UAE - transshipment hub (not risky itself, but used for laundering)
    "HK": 0.45,  # Hong Kong - transshipment
    "VN": 0.4,   # Vietnam - misdeclaration patterns
    "BD": 0.35,  # Bangladesh - invoice fraud
}

INDIAN_DISTRICTS_TN = [
    "Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem",
    "Tirunelveli", "Erode", "Vellore", "Thoothukudi", "Dindigul",
    "Thanjavur", "Ranipet", "Sivaganga", "Virudhunagar", "Namakkal",
    "Kancheepuram", "Tiruppur", "Krishnagiri", "Dharmapuri", "Perambalur"
]

TALUKS_BY_DISTRICT = {
    "Chennai": ["Egmore-Nungambakkam", "Alandur", "Sholinganallur", "Ambattur", "Perambur"],
    "Coimbatore": ["Coimbatore North", "Coimbatore South", "Pollachi", "Mettupalayam"],
    "Madurai": ["Madurai North", "Madurai South", "Melur", "Thirumangalam"],
    "Tiruppur": ["Tiruppur North", "Tiruppur South", "Dharapuram", "Udumalaipettai"],
}

NATURE_OF_INDUSTRY = [
    "Manufacturing", "Trading", "Service", "Export Oriented Unit",
    "Agro-based Industry", "Textile", "Engineering", "Chemical",
    "Food Processing", "Leather", "Handloom", "Handicraft"
]

PRODUCTS_BY_NATURE = {
    "Textile": ["Cotton Yarn", "Woven Fabric", "Readymade Garments", "Hosiery", "Knitted Fabric"],
    "Manufacturing": ["Auto Parts", "Plastic Goods", "Rubber Products", "Metal Fabrication"],
    "Food Processing": ["Rice Mill", "Dal Mill", "Spices", "Pickles", "Bakery Products"],
    "Engineering": ["Machine Parts", "Precision Tools", "Castings", "Forgings"],
    "Chemical": ["Dyes", "Pigments", "Cleaning Agents", "Pharma Intermediates"],
    "Trading": ["General Merchandise", "Electronics", "Textiles", "Agricultural Produce"],
    "Export Oriented Unit": ["Garments", "Leather Goods", "Gems & Jewellery", "Software"],
    "Leather": ["Finished Leather", "Leather Footwear", "Leather Garments", "Leather Goods"],
}

BANKS = [
    "State Bank of India", "Indian Bank", "Canara Bank", "Bank of Baroda",
    "Punjab National Bank", "Indian Overseas Bank", "Union Bank of India",
    "HDFC Bank", "ICICI Bank", "Axis Bank", "Federal Bank", "City Union Bank"
]

PORTS = {
    "Chennai": "INNSA1",    # Chennai Sea Port
    "Coimbatore": "INICJ4", # ICD Coimbatore
    "Tiruppur": "INITP6",   # ICD Tiruppur
    "Madurai": "INMAA4",    # ICD Madurai
    "Thoothukudi": "INTUT1",# Tuticorin Port
}

# ─────────────────────────────────────────────
# SCRAPER (run from YOUR machine, not cloud)
# ─────────────────────────────────────────────

SCRAPER_CODE = '''
# scrape_msme.py — run this from YOUR local machine
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_msme_tn():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.msmeonline.tn.gov.in/",
    }

    session = requests.Session()
    session.headers.update(headers)

    # Step 1: Get main page (may set cookies)
    r = session.get("https://www.msmeonline.tn.gov.in/", timeout=20)
    time.sleep(1)

    # Step 2: Get corona safe unit list (has ~1000+ MSME entries)
    r2 = session.get(
        "https://www.msmeonline.tn.gov.in/corona_safe_unit_list_new.php",
        timeout=30
    )

    soup = BeautifulSoup(r2.text, "lxml")
    tables = soup.find_all("table")

    all_rows = []
    for table in tables:
        headers_row = [th.text.strip() for th in table.find_all("th")]
        if not headers_row:
            continue
        for tr in table.find_all("tr")[1:]:
            cols = [td.text.strip() for td in tr.find_all("td")]
            if cols:
                all_rows.append(dict(zip(headers_row, cols)))

    df = pd.DataFrame(all_rows)
    df.to_csv("real_msme_seed.csv", index=False)
    print(f"Scraped {len(df)} MSME entities → real_msme_seed.csv")
    return df

if __name__ == "__main__":
    scrape_msme_tn()
'''

# ─────────────────────────────────────────────
# GENERATOR FUNCTIONS
# ─────────────────────────────────────────────

def generate_udyam_no():
    state = "TN"
    district_code = str(random.randint(1, 38)).zfill(2)
    year = random.choice(["20", "21", "22", "23"])
    seq = str(random.randint(1, 999999)).zfill(7)
    return f"UDYAM-{state}-{district_code}-{year}-{seq}"

def generate_gstin(pan, state_code="33"):  # 33 = Tamil Nadu
    entity_type = random.choice(["1", "2", "4", "6"])
    checksum = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return f"{state_code}{pan}{entity_type}Z{checksum}"

def generate_pan():
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    entity_type = random.choice(["P", "C", "F", "A"])  # P=person, C=company
    return (
        ''.join(random.choices(letters, k=3))
        + entity_type
        + random.choice(letters)
        + ''.join(random.choices("0123456789", k=4))
        + random.choice(letters)
    )

def generate_iec():
    prefix = str(random.randint(1000, 9999))
    suffix = str(random.randint(1000000, 9999999))
    return f"{prefix}{suffix}"


def generate_msme_entities(n=2000, seed_df=None):
    """
    Generate MSME entities mirroring msmeonline.tn.gov.in schema.
    If seed_df is provided (real scraped data), blends it in.
    """
    records = []

    for i in range(n):
        district = random.choice(INDIAN_DISTRICTS_TN)
        taluk_list = TALUKS_BY_DISTRICT.get(district, [district + " Taluk"])
        taluk = random.choice(taluk_list)
        nature = random.choice(NATURE_OF_INDUSTRY)
        products = PRODUCTS_BY_NATURE.get(nature, ["General Products"])
        product = random.choice(products)
        enterprise_type = random.choices(
            ["Micro", "Small", "Medium"],
            weights=[0.60, 0.30, 0.10]
        )[0]

        # Capital based on enterprise type
        capital_ranges = {
            "Micro":  (100000,   2500000),
            "Small":  (2500000,  50000000),
            "Medium": (50000000, 250000000),
        }
        lo, hi = capital_ranges[enterprise_type]
        paid_up_capital = random.randint(lo, hi)

        reg_date = fake.date_between(start_date='-15y', end_date='-6m')
        years_active = (datetime.now().date() - reg_date).days // 365

        # Shell company pattern: low capital, recently registered, dormant
        is_shell = (
            enterprise_type == "Micro" and
            paid_up_capital < 200000 and
            years_active < 2 and
            random.random() < 0.3
        )

        pan = generate_pan()
        gstin = generate_gstin(pan)
        has_iec = nature in ["Export Oriented Unit", "Trading"] or random.random() < 0.3

        employee_ranges = {
            "Micro":  (1, 10),
            "Small":  (11, 50),
            "Medium": (51, 250),
        }
        emp_lo, emp_hi = employee_ranges[enterprise_type]
        employee_count = random.randint(emp_lo, emp_hi)

        turnover_multiplier = random.uniform(0.5, 3.0)
        annual_turnover = int(paid_up_capital * turnover_multiplier)

        # Fraud flag: turnover wildly disproportionate to capital = red flag
        if is_shell:
            annual_turnover = annual_turnover * random.uniform(10, 50)  # inflated

        export_share = 0.0
        if nature == "Export Oriented Unit":
            export_share = random.uniform(0.6, 0.95)
        elif has_iec:
            export_share = random.uniform(0.1, 0.4)

        bank = random.choice(BANKS)
        port = PORTS.get(district, "INMAA4")

        records.append({
            # ── MSME Site Fields (mirrors msmeonline.tn.gov.in) ──
            "unit_name":            fake.company().replace(",", ""),
            "registration_no":      fake.bothify("TN/MSME/??/######"),
            "udyam_no":             generate_udyam_no(),
            "nature_of_industry":   nature,
            "enterprise_type":      enterprise_type,
            "product_activity":     product,
            "district":             district,
            "taluk":                taluk,
            "address":              fake.address().replace("\n", ", "),
            "contact_person":       fake.name(),
            "phone":                fake.phone_number(),
            "registration_date":    reg_date,
            "commencement_date":    fake.date_between(
                                        start_date=reg_date,
                                        end_date=reg_date + timedelta(days=180)
                                    ),

            # ── Identity ──
            "gstin":                gstin,
            "pan":                  pan,
            "iec_code":             generate_iec() if has_iec else None,
            "has_iec":              has_iec,

            # ── Financials ──
            "paid_up_capital":      paid_up_capital,
            "authorized_capital":   int(paid_up_capital * random.uniform(1.0, 2.5)),
            "employee_count":       employee_count,
            "annual_turnover":      annual_turnover,
            "export_turnover":      int(annual_turnover * export_share),
            "import_turnover":      int(annual_turnover * random.uniform(0, 0.3)),
            "bank_name":            bank,
            "ad_code":              fake.bothify("######"),  # Authorised Dealer code
            "years_in_business":    years_active,
            "mca_status":           "dormant" if is_shell else
                                    random.choices(
                                        ["active", "dormant", "struck_off"],
                                        weights=[0.85, 0.10, 0.05]
                                    )[0],

            # ── Fraud Indicators (ground truth) ──
            "is_shell":             is_shell,
            "related_party_txn_pct": random.uniform(0.6, 0.95) if is_shell
                                     else random.uniform(0.0, 0.15),
            "turnover_capital_ratio": annual_turnover / max(paid_up_capital, 1),

            # ── Port / Trade ──
            "primary_port":         port,
            "hs_code_primary":      random.choice(list(HS_BENCHMARKS.keys())),
        })

    df = pd.DataFrame(records)

    # Blend in real scraped data if available
    if seed_df is not None and len(seed_df) > 0:
        print(f"[INFO] Blending {len(seed_df)} real MSME rows with {n} synthetic rows")
        # Normalize column names
        seed_df.columns = [c.lower().replace(" ", "_") for c in seed_df.columns]
        df = pd.concat([df, seed_df], ignore_index=True, sort=False)

    return df


def generate_transactions(entities_df, n=10000):
    """
    Trade transactions with fraud injection.
    Anchored to HS_BENCHMARKS (from UN Comtrade price distributions).
    """
    records = []
    iec_entities = entities_df[entities_df['has_iec'] == True].to_dict('records')

    if len(iec_entities) < 2:
        raise ValueError("Not enough IEC-holding entities. Increase n in generate_msme_entities.")

    fraud_types_distribution = {
        "clean":             0.75,
        "under_invoicing":   0.07,
        "over_invoicing":    0.05,
        "ghost_shipment":    0.04,
        "round_tripping":    0.04,
        "misdescription":    0.03,
        "duplicate_invoice": 0.02,
    }

    fraud_labels = random.choices(
        list(fraud_types_distribution.keys()),
        weights=list(fraud_types_distribution.values()),
        k=n
    )

    for fraud_type in fraud_labels:
        exporter = random.choice(iec_entities)
        importer = random.choice(iec_entities)
        while importer['iec_code'] == exporter['iec_code']:
            importer = random.choice(iec_entities)

        hs_code = random.choice(list(HS_BENCHMARKS.keys()))
        benchmark_price, std_dev = HS_BENCHMARKS[hs_code]

        quantity = random.uniform(10, 10000)
        weight_kg = quantity * random.uniform(0.5, 5.0)

        # ── Price: start with legit market price ──
        legit_unit_price = max(0.1, np.random.normal(benchmark_price, std_dev))
        declared_unit_price = legit_unit_price
        actual_unit_price = legit_unit_price
        ghost = False
        duplicate = False

        if fraud_type == "under_invoicing":
            # Declare 20-50% of real value
            declared_unit_price = legit_unit_price * random.uniform(0.2, 0.5)

        elif fraud_type == "over_invoicing":
            # Declare 150-400% of real value (export incentive fraud / capital flight)
            declared_unit_price = legit_unit_price * random.uniform(1.5, 4.0)

        elif fraud_type == "ghost_shipment":
            ghost = True
            declared_unit_price = legit_unit_price
            weight_kg = 0.0  # nothing actually shipped

        elif fraud_type == "round_tripping":
            # Export + re-import same goods — same exporter/importer district, short interval
            declared_unit_price = legit_unit_price
            # Handled at network level via GNN edges

        elif fraud_type == "misdescription":
            # High-value goods declared under low-duty HS code
            declared_unit_price = legit_unit_price * random.uniform(0.3, 0.6)
            hs_code = random.choice(["0803", "2601", "7204"])  # low-value HS

        elif fraud_type == "duplicate_invoice":
            duplicate = True

        declared_value = declared_unit_price * quantity
        actual_value   = actual_unit_price * quantity
        weight_to_value_ratio = weight_kg / max(declared_value, 1)

        origin_country = random.choices(
            list(HIGH_RISK_COUNTRIES.keys()) + ["CN", "US", "DE", "JP", "GB", "SG", "LK"],
            weights=[v * 0.5 for v in HIGH_RISK_COUNTRIES.values()] + [0.3, 0.25, 0.2, 0.15, 0.15, 0.12, 0.1],
        )[0]

        txn_date = fake.date_time_between(start_date='-2y', end_date='now')
        invoice_no = f"INV-{fake.bothify('??####')}"

        # Round invoice flag (ends in 000 or 0000)
        if fraud_type in ["under_invoicing", "over_invoicing"]:
            if random.random() < 0.4:
                declared_value = round(declared_value / 1000) * 1000

        # Description specificity (low = vague = suspicious)
        description_map = {
            "clean":           fake.catch_phrase() + " goods",
            "misdescription":  random.choice(["assorted goods", "general merchandise", "various items"]),
            "ghost_shipment":  random.choice(["general cargo", "mixed goods"]),
        }
        goods_description = description_map.get(
            fraud_type,
            fake.bs() + " products"
        )

        # Filing time pattern (2-4am = suspicious bulk filing)
        hour = txn_date.hour
        suspicious_filing_hour = (hour >= 2 and hour <= 4)

        # GST period proximity
        gst_return_dates = [7, 11]  # GSTR-1 due dates
        days_to_gst_period = min(abs(txn_date.day - d) for d in gst_return_dates)

        # Port-HS mismatch (e.g. semiconductors via agricultural port)
        port = PORTS.get(exporter.get('district', 'Chennai'), "INNSA1")
        port_hs_mismatch = (
            hs_code in ["8471", "8517", "8542"] and port in ["INTUT1"]
        )

        # Benford's Law — first digit of invoice amount
        first_digit = int(str(int(abs(declared_value)))[0])

        records.append({
            # ── IDs ──
            "txn_id":               fake.uuid4(),
            "timestamp":            txn_date,
            "invoice_no":           invoice_no + ("-DUP" if duplicate else ""),
            "exporter_iec":         exporter['iec_code'],
            "importer_iec":         importer['iec_code'],
            "exporter_district":    exporter['district'],

            # ── Goods ──
            "hs_code":              hs_code,
            "goods_description":    goods_description,
            "quantity":             round(quantity, 2),
            "unit":                 random.choice(["KG", "PCS", "MTR", "LTR", "NOS"]),
            "weight_kg":            round(weight_kg, 2),

            # ── Value ──
            "declared_value_usd":   round(declared_value, 2),
            "actual_value_usd":     round(actual_value, 2),
            "benchmark_value_usd":  round(legit_unit_price * quantity, 2),
            "declared_unit_price":  round(declared_unit_price, 4),
            "benchmark_unit_price": round(legit_unit_price, 4),

            # ── Route ──
            "origin_country":       origin_country,
            "origin_risk_score":    HIGH_RISK_COUNTRIES.get(origin_country, 0.1),
            "port_of_export":       port,
            "port_hs_mismatch":     int(port_hs_mismatch),
            "transshipment_flag":   int(origin_country in ["AE", "HK", "SG"] and
                                        random.random() < 0.3),

            # ── Additional Documentation / Route Fields ──
            "bol_weight":           round(weight_kg * (random.uniform(0.5, 1.5) if ghost else 1.0), 2),
            "bol_number":           None if ghost else f"BOL-{fake.bothify('####????')}",
            "packing_list":         True if not ghost else None,
            "letter_of_credit":     True if random.random() > 0.05 else None,
            "port_arrival_record":  True if not ghost else None,
            "transit_port":         random.choice(["AE", "HK", "SG", "SA", "OM"]) if origin_country in ["AE", "HK", "SG"] else "",
            "abnormal_route_flag":  int(fraud_type == "round_tripping"),
            "further_shipment_records": int(random.random() > 0.2),
            "transshipment_count":  random.randint(0, 4),
            "shared_address_flag":  int(random.random() < 0.03),
            "related_party_flag":   int(exporter['is_shell']),
            "repeat_shipment_count_30d": random.randint(0, 5) if fraud_type != 'clean' else random.randint(0, 2),

            # ── Payment ──
            "payment_terms":        random.choice(["LC", "TT", "DA", "DP", "CAD"]),
            "bank_name":            exporter['bank_name'],
            "ad_code":              exporter['ad_code'],

            # ── Engineered Fraud Features ──
            "price_deviation_pct":  round(
                                        (declared_unit_price - legit_unit_price)
                                        / max(legit_unit_price, 0.01) * 100, 2
                                    ),
            "weight_to_value_ratio": round(weight_to_value_ratio, 6),
            "is_round_invoice":     int(declared_value % 1000 == 0),
            "is_ghost_shipment":    int(ghost),
            "is_duplicate_invoice": int(duplicate),
            "description_specificity": len(goods_description.split()),  # word count proxy
            "suspicious_filing_hour":  int(suspicious_filing_hour),
            "days_to_gst_period":      days_to_gst_period,
            "benford_first_digit":     first_digit,
            "exporter_iec_age_days":   exporter['years_in_business'] * 365,
            "exporter_is_shell":       int(exporter['is_shell']),
            "exporter_turnover_cap_ratio": exporter['turnover_capital_ratio'],

            # ── Labels ──
            "fraud_type":           fraud_type,
            "is_fraud":             int(fraud_type != "clean"),
        })

    return pd.DataFrame(records)


def generate_card_events(entities_df, n=50000):
    """
    Credit/debit card transaction events with GPS + velocity features.
    Fraud injection: card cloning (impossible travel), card-not-present fraud.
    """
    records = []
    entity_sample = entities_df.sample(min(500, len(entities_df))).to_dict('records')

    CITIES_COORDS = {
        "Chennai":     (13.0827, 80.2707),
        "Mumbai":      (19.0760, 72.8777),
        "Delhi":       (28.6139, 77.2090),
        "Bangalore":   (12.9716, 77.5946),
        "Hyderabad":   (17.3850, 78.4867),
        "Kolkata":     (22.5726, 88.3639),
        "Coimbatore":  (11.0168, 76.9558),
        "Tiruppur":    (11.1085, 77.3411),
        "Dubai":       (25.2048, 55.2708),
        "Singapore":   (1.3521,  103.8198),
        "Colombo":     (6.9271,  79.8612),
    }

    fraud_city_pairs = [  # (city1, city2, max_mins) — impossible travel
        ("Chennai", "Dubai",     30),
        ("Mumbai",  "Singapore", 45),
        ("Delhi",   "Kolkata",   20),
        ("Chennai", "Delhi",     25),
    ]

    for _ in range(n):
        entity = random.choice(entity_sample)
        card_id = fake.credit_card_number()
        fraud_type = random.choices(
            ["clean", "card_cloning", "cnp_fraud", "account_takeover"],
            weights=[0.82, 0.08, 0.06, 0.04]
        )[0]

        city = random.choice(list(CITIES_COORDS.keys()))
        lat, lon = CITIES_COORDS[city]
        lat += np.random.normal(0, 0.05)
        lon += np.random.normal(0, 0.05)

        txn_time = fake.date_time_between(start_date='-1y', end_date='now')
        amount = round(abs(np.random.lognormal(7, 1.5)), 2)  # INR, log-normal

        distance_from_last_km = np.random.exponential(20)
        time_since_last_mins  = np.random.exponential(180)

        if fraud_type == "card_cloning":
            # Impossible travel: 2nd txn far away, very soon after 1st
            pair = random.choice(fraud_city_pairs)
            city = pair[1]
            lat, lon = CITIES_COORDS[city]
            distance_from_last_km = random.uniform(600, 6000)
            time_since_last_mins  = random.uniform(5, pair[2])

        elif fraud_type == "cnp_fraud":
            # Card not present, international, unusual amount
            city = random.choice(["Dubai", "Singapore"])
            lat, lon = CITIES_COORDS[city]
            amount = round(random.uniform(50000, 500000), 2)

        elif fraud_type == "account_takeover":
            # Login from new device, large transfer
            amount = round(random.uniform(100000, 1000000), 2)
            time_since_last_mins = random.uniform(2, 30)

        velocity_1hr  = random.randint(1, 3) if fraud_type == "clean" else random.randint(4, 20)
        velocity_24hr = random.randint(1, 10) if fraud_type == "clean" else random.randint(10, 60)

        records.append({
            "card_id":                  card_id,
            "entity_id":                entity['udyam_no'],
            "txn_timestamp":            txn_time,
            "amount_inr":               amount,
            "merchant_category_code":   fake.bothify("####"),
            "city":                     city,
            "lat":                      round(lat, 6),
            "lon":                      round(lon, 6),
            "is_international":         int(city in ["Dubai", "Singapore", "Colombo"]),
            "is_contactless":           int(random.random() < 0.4),
            "device_fingerprint":       fake.md5(),
            "ip_address":               fake.ipv4(),
            "distance_from_last_txn_km": round(distance_from_last_km, 2),
            "time_since_last_txn_mins":  round(time_since_last_mins, 2),
            "velocity_1hr":              velocity_1hr,
            "velocity_24hr":             velocity_24hr,
            "hour_of_day":               txn_time.hour,
            "is_weekend":                int(txn_time.weekday() >= 5),
            "fraud_type":               fraud_type,
            "is_fraud":                 int(fraud_type != "clean"),
        })

    return pd.DataFrame(records)


def generate_enterprise_expenditure(entities_df):
    """Quarterly financials per entity — industrial expenditure."""
    records = []
    quarters = ["2022Q1","2022Q2","2022Q3","2022Q4",
                "2023Q1","2023Q2","2023Q3","2023Q4",
                "2024Q1","2024Q2","2024Q3","2024Q4"]

    for _, entity in entities_df.iterrows():
        base = entity['annual_turnover'] / 4  # quarterly base
        for q in quarters:
            # Seasonal variation
            seasonal = 1.0 + 0.2 * np.sin(quarters.index(q) * np.pi / 2)
            revenue = base * seasonal * random.uniform(0.8, 1.2)

            raw_mat   = revenue * random.uniform(0.30, 0.50)
            energy    = revenue * random.uniform(0.05, 0.12)
            logistics = revenue * random.uniform(0.05, 0.10)
            labour    = revenue * random.uniform(0.10, 0.25)
            opex      = raw_mat + energy + logistics + labour

            gst_input = opex * 0.18 * random.uniform(0.8, 1.0)
            gst_output = revenue * 0.18

            # Fraud: GST refund claimed >> reasonable amount
            gst_refund_claimed = gst_input if not entity['is_shell'] else \
                                 gst_input * random.uniform(2.0, 5.0)  # inflated

            records.append({
                "enterprise_id":           entity['udyam_no'],
                "enterprise_type":         entity['enterprise_type'],
                "fiscal_quarter":          q,
                "revenue_inr":             round(revenue),
                "raw_material_cost":       round(raw_mat),
                "energy_cost":             round(energy),
                "logistics_cost":          round(logistics),
                "labour_cost":             round(labour),
                "opex_total":              round(opex),
                "capex":                   round(revenue * random.uniform(0, 0.05)),
                "gst_input_credit":        round(gst_input),
                "gst_output_tax":          round(gst_output),
                "gst_refund_claimed":      round(gst_refund_claimed),
                "gst_refund_anomaly":      int(gst_refund_claimed > gst_input * 1.2),
                "export_incentive_claimed":round(revenue * random.uniform(0, 0.05)),
                "bank_od_utilization_pct": round(random.uniform(0.1, 0.95) * 100, 1),
                "related_party_txn_pct":   round(entity['related_party_txn_pct'] * 100, 1),
                "is_shell_entity":         int(entity['is_shell']),
            })

    return pd.DataFrame(records)


def generate_graph_edges(entities_df, transactions_df):
    """
    Build entity relationship graph edges for GNN (Layer 3).
    Nodes: enterprises. Edges: shared attributes + transactions.
    """
    edges = []
    entity_list = entities_df.to_dict('records')

    # Edge type 1: Shared district (weaker link)
    district_groups = entities_df.groupby('district')['udyam_no'].apply(list)
    for district, members in district_groups.items():
        for i in range(min(len(members), 20)):  # cap to avoid explosion
            for j in range(i+1, min(len(members), 20)):
                if random.random() < 0.05:  # sparse
                    edges.append({
                        "source": members[i], "target": members[j],
                        "edge_type": "same_district", "weight": 0.2
                    })

    # Edge type 2: Shared bank (medium link)
    bank_groups = entities_df.groupby('bank_name')['udyam_no'].apply(list)
    for bank, members in bank_groups.items():
        for i in range(min(len(members), 15)):
            for j in range(i+1, min(len(members), 15)):
                if random.random() < 0.03:
                    edges.append({
                        "source": members[i], "target": members[j],
                        "edge_type": "same_bank", "weight": 0.4
                    })

    # Edge type 3: Transaction links (strong link)
    if 'exporter_iec' in transactions_df.columns:
        txn_sample = transactions_df[transactions_df['is_fraud'] == 1].head(500)
        iec_to_udyam = dict(zip(entities_df['iec_code'], entities_df['udyam_no']))
        for _, txn in txn_sample.iterrows():
            src = iec_to_udyam.get(txn['exporter_iec'])
            tgt = iec_to_udyam.get(txn['importer_iec'])
            if src and tgt and src != tgt:
                edges.append({
                    "source": src, "target": tgt,
                    "edge_type": "fraud_transaction",
                    "weight": 0.9,
                    "fraud_type": txn['fraud_type']
                })

    return pd.DataFrame(edges)


# ─────────────────────────────────────────────
# RULE ENGINE  (Layer 1)
# ─────────────────────────────────────────────

WEIGHT_THRESHOLDS = {
    "6204": 0.0015,   # garments: kg per USD
    "8471": 0.005,
    "2709": 2.0,
    "default": 0.05,
}

def flag_transaction(row):
    """
    Comprehensive rule engine — returns list of flags + composite risk score.
    This is Layer 1 of the fraud detection pipeline.
    """
    flags = []
    risk_score = 0.0

    hs = str(row.get('hs_code', ''))
    benchmark_price, _ = HS_BENCHMARKS.get(hs, (1.0, 0.5))
    declared_unit = float(row.get('declared_unit_price', 0))

    # ── 1. PRICE DEVIATION RULES ──────────────────────
    if declared_unit < benchmark_price * 0.5:
        flags.append('under_invoicing')
        risk_score += 0.35

    if declared_unit > benchmark_price * 2.0:
        flags.append('over_invoicing')
        risk_score += 0.30

    price_dev = abs(float(row.get('price_deviation_pct', 0)))
    if price_dev > 80:
        flags.append('extreme_price_deviation')
        risk_score += 0.25

    # ── 2. WEIGHT / PHYSICAL CONSISTENCY ─────────────
    threshold = WEIGHT_THRESHOLDS.get(hs, WEIGHT_THRESHOLDS['default'])
    wv_ratio = float(row.get('weight_to_value_ratio', 0))
    if wv_ratio > threshold * 10:
        flags.append('abnormal_weight_value_ratio')
        risk_score += 0.20

    if float(row.get('weight_kg', 1)) == 0.0:
        flags.append('zero_weight_ghost_shipment')
        risk_score += 0.50

    # ── 3. ORIGIN / ROUTE RISK ────────────────────────
    origin = row.get('origin_country', '')
    origin_risk = float(row.get('origin_risk_score', 0))
    if origin_risk >= 0.85:
        flags.append('sanctioned_or_high_risk_origin')
        risk_score += 0.40

    elif origin_risk >= 0.5:
        flags.append('elevated_risk_origin')
        risk_score += 0.20

    if int(row.get('transshipment_flag', 0)):
        flags.append('transshipment_risk')
        risk_score += 0.15

    if int(row.get('port_hs_mismatch', 0)):
        flags.append('port_hs_code_mismatch')
        risk_score += 0.20

    # ── 4. DOCUMENT INTEGRITY ─────────────────────────
    if int(row.get('is_round_invoice', 0)):
        flags.append('round_number_invoice')
        risk_score += 0.10

    if int(row.get('is_duplicate_invoice', 0)):
        flags.append('duplicate_invoice')
        risk_score += 0.45

    desc_words = int(row.get('description_specificity', 10))
    if desc_words <= 3:
        flags.append('vague_goods_description')
        risk_score += 0.25

    # ── 5. TEMPORAL PATTERNS ──────────────────────────
    if int(row.get('suspicious_filing_hour', 0)):
        flags.append('off_hours_filing_2am_4am')
        risk_score += 0.15

    gst_days = int(row.get('days_to_gst_period', 15))
    if gst_days <= 2:
        flags.append('filing_just_before_gst_deadline')
        risk_score += 0.20

    # ── 6. COUNTERPARTY / ENTITY RISK ────────────────
    iec_age = float(row.get('exporter_iec_age_days', 365))
    declared_val = float(row.get('declared_value_usd', 0))
    if iec_age < 180 and declared_val > 50000:
        flags.append('new_iec_holder_high_value_txn')
        risk_score += 0.30

    if int(row.get('exporter_is_shell', 0)):
        flags.append('known_shell_company_exporter')
        risk_score += 0.50

    turnover_cap = float(row.get('exporter_turnover_cap_ratio', 1.0))
    if turnover_cap > 20:
        flags.append('turnover_to_capital_ratio_extreme')
        risk_score += 0.25

    # ── 7. BENFORD'S LAW ─────────────────────────────
    benford_expected = {1:0.301,2:0.176,3:0.125,4:0.097,
                        5:0.079,6:0.067,7:0.058,8:0.051,9:0.046}
    first_d = int(row.get('benford_first_digit', 1))
    if first_d in [7, 8, 9]:  # high first digits = underrepresented = suspicious
        flags.append('benford_anomaly_high_first_digit')
        risk_score += 0.08

    # ── 8. PAYMENT TERMS RISK ─────────────────────────
    payment = row.get('payment_terms', '')
    if payment in ['DA', 'DP'] and origin_risk > 0.5:
        flags.append('risky_payment_terms_high_risk_origin')
        risk_score += 0.20

    # ── COMPOSITE SCORE ───────────────────────────────
    risk_score = min(risk_score, 1.0)

    return {
        "flags":      flags,
        "flag_count": len(flags),
        "risk_score": round(risk_score, 3),
        "risk_level": "HIGH"   if risk_score >= 0.6 else
                      "MEDIUM" if risk_score >= 0.3 else
                      "LOW"
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Save scraper separately
    with open("scrape_msme.py", "w", encoding="utf-8") as f:
        f.write(SCRAPER_CODE)
    print("[v] scrape_msme.py written - run this from YOUR local machine")

    # Check for real seed data
    seed_df = None
    if os.path.exists("real_msme_seed.csv"):
        seed_df = pd.read_csv("real_msme_seed.csv")
        print(f"[✓] Loaded real MSME seed: {len(seed_df)} rows")
    else:
        print("[!] No real_msme_seed.csv found — using pure synthetic")
        print("    → Run scrape_msme.py from your browser machine first")

    print("\n[1/5] Generating MSME entities...")
    entities = generate_msme_entities(n=2000, seed_df=seed_df)
    entities.to_csv("output/entities.csv", index=False)
    print(f"      {len(entities)} entities | {entities['is_shell'].sum()} shells")

    print("\n[2/5] Generating transactions...")
    txns = generate_transactions(entities, n=10000)
    txns.to_csv("output/transactions.csv", index=False)
    print(f"      {len(txns)} txns | {txns['is_fraud'].sum()} fraud")

    print("\n[3/5] Generating card events...")
    cards = generate_card_events(entities, n=30000)
    cards.to_csv("output/card_events.csv", index=False)
    print(f"      {len(cards)} events | {cards['is_fraud'].sum()} fraud")

    print("\n[4/5] Generating enterprise expenditure...")
    expenditure = generate_enterprise_expenditure(entities.head(500))
    expenditure.to_csv("output/expenditure.csv", index=False)
    print(f"      {len(expenditure)} quarterly records")

    print("\n[5/5] Generating graph edges...")
    graph_edges = generate_graph_edges(entities, txns)
    graph_edges.to_csv("output/graph_edges.csv", index=False)
    print(f"      {len(graph_edges)} edges")

    # Apply rule engine to transactions
    print("\n[+] Applying NEW rule engine to transactions...")
    from rule_engine import TradeFraudRuleEngine
    engine = TradeFraudRuleEngine(api_key=None) # Passing None will use our clever fallback

    def apply_new_engine(row):
        return engine.evaluate(row.to_dict())

    rule_results = txns.apply(apply_new_engine, axis=1, result_type='expand')
    txns_flagged = pd.concat([txns, rule_results], axis=1)
    txns_flagged.to_csv("output/transactions_flagged.csv", index=False)

    # Stats
    print("\n══════════════════════════════════════════")
    print("  RULE ENGINE STATS")
    print("══════════════════════════════════════════")
    print(txns_flagged['risk_level'].value_counts().to_string())
    print(f"\n  Fraud recall (HIGH risk caught):")
    fraud_rows = txns_flagged[txns_flagged['is_fraud'] == 1]
    caught = (fraud_rows['risk_level'] == 'HIGH').sum()
    print(f"  {caught}/{len(fraud_rows)} = {caught/len(fraud_rows)*100:.1f}%")

    print("\n══════════════════════════════════════════")
    print("  OUTPUT FILES")
    print("══════════════════════════════════════════")
    for f in os.listdir("output"):
        path = f"output/{f}"
        size = os.path.getsize(path) // 1024
        print(f"  {f:40s} {size:>6} KB")

    print("\n[✓] Done. Next step: run rule_engine_demo.py to see flags in action.")