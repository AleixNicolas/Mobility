import pandas as pd
import requests
import io
import time
import re
from pathlib import Path

# --- 1. SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_ROOT = PROJECT_ROOT / "data"

SAVE_DIR = DATA_ROOT / "02_raw_downloaded" / "demographics"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = SAVE_DIR / "ine_spain_population_2024.csv"

# Mapping Provinces to their Capitals for Verification
PROVINCE_CAPITALS = {
    "01": "Vitoria", "02": "Albacete", "03": "Alicant", "04": "Almería", "05": "Ávila",
    "06": "Badajoz", "07": "Palma", "08": "Barcelona", "09": "Burgos", "10": "Cáceres",
    "11": "Cádiz", "12": "Castell", "13": "Ciudad Real", "14": "Córdoba", "15": "Coruña",
    "16": "Cuenca", "17": "Girona", "18": "Granada", "19": "Guadalajara", "20": "Donostia",
    "21": "Huelva", "22": "Huesca", "23": "Jaén", "24": "León", "25": "Lleida",
    "26": "Logroño", "27": "Lugo", "28": "Madrid", "29": "Málaga", "30": "Murcia",
    "31": "Pamplona", "32": "Ourense", "33": "Oviedo", "34": "Palencia", "35": "Palmas",
    "36": "Pontevedra", "37": "Salamanca", "38": "Santa Cruz", "39": "Santander", "40": "Segovia",
    "41": "Sevilla", "42": "Soria", "43": "Tarragona", "44": "Teruel", "45": "Toledo",
    "46": "València", "47": "Valladolid", "48": "Bilbao", "49": "Zamora", "50": "Zaragoza",
    "51": "Ceuta", "52": "Melilla"
}

def try_fetch_table(table_id, prov_code, capital_search):
    url = f"https://www.ine.es/jaxiT3/files/t/es/csv_bd/{table_id}.csv"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code != 200 or len(r.content) < 500: return None
        
        df = pd.read_csv(io.BytesIO(r.content), sep=r'[;\t]', engine='python', encoding='utf-8-sig', dtype=str)
        df.columns = [c.strip() for c in df.columns]
        
        muni_col = next((c for c in df.columns if 'Municip' in c), None)
        if not muni_col: return None
        
        # Verify this table belongs to the province by checking for the capital and the 2-digit code
        # We look for a row where the municipality name contains the capital string
        # and starts with the correct 2-digit provincial code.
        sample = df[muni_col].str.contains(f"^{prov_code}", na=False)
        has_capital = df[muni_col].str.contains(capital_search, case=False, na=False).any()
        
        if sample.any() and has_capital:
            return df, muni_col
    except:
        return None
    return None

all_data = []
# These are the typical ranges where municipal population tables live
search_ranges = [(2850, 2910), (2900, 2960)] 

print("--- STARTING NATIONAL DISCOVERY & VALIDATION ---")

for prov_code, capital in PROVINCE_CAPITALS.items():
    found = False
    print(f"Searching for Province {prov_code} ({capital})...")
    
    # We test IDs in the likely ranges
    for start, end in search_ranges:
        if found: break
        for tid in range(start, end + 1):
            result = try_fetch_table(tid, prov_code, capital)
            if result:
                df, muni_col = result
                
                # Apply your proven filtering logic
                periods = df['Periodo'].unique()
                target_year = '2024' if '2024' in periods else max(periods)
                mask = (df['Periodo'].str.strip() == target_year) & (df['Sexo'].str.strip() == 'Total')
                df_filtered = df[mask].copy()
                
                # Extract 5-digit code and name
                df_filtered[['code', 'municipality']] = df_filtered[muni_col].str.extract(r'^(\d{5})\s+(.*)')
                df_filtered = df_filtered.dropna(subset=['code'])
                
                # Filter strictly for this province's municipalities only
                df_filtered = df_filtered[df_filtered['code'].str.startswith(prov_code)]
                
                if not df_filtered.empty:
                    pop_col = 'Total' if 'Total' in df_filtered.columns else df_filtered.columns[-1]
                    df_filtered = df_filtered[['code', 'municipality', pop_col]].rename(columns={pop_col: 'population_2024'})
                    df_filtered['population_2024'] = df_filtered['population_2024'].str.replace('.', '', regex=False)
                    
                    all_data.append(df_filtered)
                    print(f" Found at Table {tid} | Count: {len(df_filtered)}")
                    found = True
                    break
        
    if not found:
        print(f"  FAILED to find valid table for {prov_code}")

# --- FINALIZATION ---
if all_data:
    df_final = pd.concat(all_data, ignore_index=True)
    df_final = df_final.sort_values("code").drop_duplicates(subset=['code']).reset_index(drop=True)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # Count how many provinces were successfully captured
    captured_provs = df_final['code'].str[:2].unique()
    print(f"\n PROCESS COMPLETE")
    print(f" Provinces Captured: {len(captured_provs)}/52")
    print(f" Total Municipalities: {len(df_final)}")