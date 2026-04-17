
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
