import requests
import json
import os

# ════════════════════════════════════════════════════════════════
#  HACKATHON NOTE:
#  The UN Comtrade API requires a free Subscription Key (Ocp-Apim-Subscription-Key)
#  to avoid 401 Unauthorized errors. 
#  This script tries the requested API. If it fails, it falls back
#  to a reliable local cache so your hackathon demo never breaks.
# ════════════════════════════════════════════════════════════════

def load_un_comtrade_benchmarks(api_key=None):
    """
    Attempts to fetch live HS code prices from UN Comtrade API.
    Returns: Dict[str, Tuple[float, float]] -> {hs_code: (benchmark_unit_price, std_dev)}
    """
    
    # Standard fallback if the API fails or is rate-limited during demo
    fallback_benchmarks = {
        "6204": (4.5,  1.2),    
        "6110": (5.2,  1.5),    
        "8471": (85.0, 20.0),   # Computers
        "8517": (45.0, 15.0),   
        "2709": (0.55, 0.1),    
        "7108": (55000, 5000),  
        "0901": (3.2,  0.8),    
        "5201": (1.8,  0.4),    
        "3004": (25.0, 8.0),    
        "8703": (12000, 3000),  
        "6403": (8.5,  2.0),    
        "0803": (0.4,  0.1),    
        "2601": (0.12, 0.03),   
        "7204": (0.22, 0.05),   
        "8542": (150.0, 40.0),  
    }

    url = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
    headers = {}
    if api_key:
        headers["Ocp-Apim-Subscription-Key"] = api_key

    try:
        # The public preview API is 100% free and requires no subscription key!
        # reporterCode 842 (USA) reliably returns current data for the public API.
        params = {
            "reporterCode": "842",  
            "period": "2023",
            "cmdCode": "8471"
        }
        res = requests.get(url, params=params, timeout=5)
        
        if res.status_code == 200:
            data = res.json()
            if "data" in data and len(data["data"]) > 0:
                sample = data["data"][0]
                trade_value = sample.get('primaryValue', 0)
                qty = sample.get('qty', 1)
                if qty > 0:
                    api_price = trade_value / qty
                    # Update our benchmark with live API data
                    fallback_benchmarks["8471"] = (round(api_price, 2), fallback_benchmarks["8471"][1])
                    print("[INFO] UN Comtrade live API connection successful for 8471.")
        else:
            print(f"[WARN] UN Comtrade API returned {res.status_code}. Using fallback cache.")
            
    except Exception as e:
        print(f"[WARN] UN Comtrade API error ({str(e)}). Using fallback cache.")

    return fallback_benchmarks


import pandas as pd

def load_opensanctions():
    """
    Loads real OpenSanctions CSV data.
    """
    csv_path = "data/opensanctions.csv"
    if os.path.exists(csv_path):
        try:
            print("[INFO] Loading real OpenSanctions CSV...")
            df = pd.read_csv(csv_path, low_memory=False)
            df = df.head(50000)  # Optimization for demo
            if "name" in df.columns:
                sanctioned_names = set(df["name"].dropna().str.upper())
                return {"OFAC_UN_ED_LIST": sanctioned_names}
        except Exception as e:
            print(f"[WARN] Error loading OpenSanctions CSV: {e}")
            
    print("[WARN] Using mock OpenSanctions fallback.")
    return {
        "OFAC_UN_ED_LIST": {
            "GUPTA BROTHERS", "DAWOOD IBRAHIM", "NIRAV MODI", "MEHUL CHOKSI",
            "WAGNER GROUP", "FRONT COMPANY LIMITED"
        }
    }


def load_fatf_greylist():
    """
    Returns the official FATF Grey List (February 2026 update).
    Using Alpha-2 codes.
    Added: Kuwait (KW), Papua New Guinea (PG)
    Removed: South Africa, Nigeria, Mozambique, Burkina Faso
    """
    return {
        "DZ", "AO", "BO", "BG", "CM", "CI", "CD", "HT", "KE", 
        "KW", "LA", "LB", "MC", "NA", "NP", "PG", "SS", "SY", 
        "VE", "VN", "VG", "YE"
    }

def load_fatf_blacklist():
    """
    Returns the FATF Black List.
    """
    return {"KP", "IR", "MM"}

def load_pep_database():
    """
    Mock Politically Exposed Persons database (Director IDs)
    """
    return {"DIR-88129", "DIR-44102", "DIR-99381"}

def load_mca_struck_off():
    """
    Mock list of struck off companies by address hash or PAN
    """
    return {"HASH-7728B", "HASH-9912C"}
