import pandas as pd
import geopandas as gpd
import json
import unicodedata
from pathlib import Path

# --- SETUP ---
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

print("--- STARTING MASTER DATA VALIDATION ---")

# 1. Securely locate files using the strict directory architecture
print("Locating files...")

JSON_PATH = DATA_ROOT / "01_raw_manual" / "dictionaries" / "affected_municipalities_dictionary.json"
INE_PATH = DATA_ROOT / "02_raw_downloaded" / "demographics" / "ine_spain_population_2024.csv"
GEOJSON_PATH = DATA_ROOT / "02_raw_downloaded" / "mobility" / "spatial" / "spatial_zones_municipalities.geojson"
RELATIONS_PATH = DATA_ROOT / "02_raw_downloaded" / "mobility" / "metadata" / "zone_relations_municipalities.csv"

# Define where the final mapping should be saved
OUTPUT_DIR = DATA_ROOT / "03_processed_shared"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAPPING_OUTPUT_PATH = OUTPUT_DIR / "final_mitma_mapping.json"

# Validate that all required files actually exist in their designated folders
required_files = [JSON_PATH, INE_PATH, GEOJSON_PATH, RELATIONS_PATH]
missing_files = [f.name for f in required_files if not f.exists()]

if missing_files:
    print(f"[ERROR] Required files are missing from their designated directories: {missing_files}")
    exit()

# --- HELPER FUNCTION ---
def normalize(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    text = text.upper().strip()
    for junk in ["L'", "LA ", "EL ", "ELS ", "LES ", "-", "'"]:
        text = text.replace(junk, " ")
    return " ".join(text.split()) # Remove extra spaces

# 2. Load Data
print("Loading datasets...")

# Affected JSON
with open(JSON_PATH, "r", encoding="utf-8") as f:
    affected_data = json.load(f)
    target_names = []
    for prov, data in affected_data.items():
        prefix = "46" if "Valencia" in prov else "16"
        for name in data["affected"]:
            target_names.append({"name": name, "norm_name": normalize(name), "prefix": prefix})

# INE Population
df_ine = pd.read_csv(INE_PATH, dtype=str)
df_ine['code'] = df_ine['code'].astype(str).str.zfill(5)
df_ine['norm_ine'] = df_ine['municipality'].apply(normalize)

# Zone Relations
df_rel = pd.read_csv(RELATIONS_PATH, dtype=str)
# The columns are usually 'municipalities' (INE) and 'municipalities_mitma' (GeoJSON/Mobility code)
ine_col = 'municipalities' if 'municipalities' in df_rel.columns else df_rel.columns[2]
mitma_col = 'municipalities_mitma' if 'municipalities_mitma' in df_rel.columns else df_rel.columns[4]

# MITMA GeoJSON (Just the properties for speed)
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    geo_data = json.load(f)
    geojson_ids = set([str(feat["properties"].get("id", "")) for feat in geo_data["features"]])

# 3. Validation Loop
results = []
for target in target_names:
    original_name = target['name']
    norm_name = target['norm_name']
    prefix = target['prefix']
    
    # Step A: Find INE Code
    ine_match = df_ine[(df_ine['norm_ine'].str.contains(norm_name, na=False)) & (df_ine['code'].str.startswith(prefix))]
    
    ine_code = "[MISSING]"
    mitma_code = "[MISSING]"
    in_geojson = "[NO]"
    
    if not ine_match.empty:
        ine_code = ine_match.iloc[0]['code']
        
        # Step B: Translate via Zone Relations
        rel_match = df_rel[df_rel[ine_col] == ine_code]
        if not rel_match.empty:
            mitma_code = rel_match.iloc[0][mitma_col]
            
            # Step C: Check GeoJSON
            if mitma_code in geojson_ids:
                in_geojson = "[YES]"

    # Manual Overrides for known tricky names (like Castelló / Catarroja mismatch)
    if "CASTELLO" in norm_name and prefix == "46":
        ine_code = "46257"
        rel_match = df_rel[df_rel[ine_col] == ine_code]
        if not rel_match.empty:
            mitma_code = rel_match.iloc[0][mitma_col]
            in_geojson = "[YES]" if mitma_code in geojson_ids else "[NO]"
            
    results.append({
        "Original Name": original_name,
        "INE Code": ine_code,
        "MITMA Code": mitma_code,
        "In GeoJSON?": in_geojson
    })

# 4. Display Results
df_results = pd.DataFrame(results)
print("\n--- VALIDATION REPORT ---")
print(df_results.to_string(index=False))

# Calculate Summary
missing_ine = len(df_results[df_results['INE Code'] == '[MISSING]'])
missing_mitma = len(df_results[df_results['MITMA Code'] == '[MISSING]'])
missing_geo = len(df_results[df_results['In GeoJSON?'] == '[NO]'])

print("\n--- SUMMARY ---")
print(f"Total Targets: {len(df_results)}")
print(f"Failed INE Match: {missing_ine}")
print(f"Failed MITMA Translation: {missing_mitma}")
print(f"Missing from GeoJSON: {missing_geo}")

if missing_geo == 0 and missing_ine == 0:
    print("\n[SUCCESS] All municipalities are perfectly mapped across all files.")
    
    # Save the perfect mapping dictionary to 03_processed_shared
    mapping_dict = df_results.set_index("Original Name")["MITMA Code"].to_dict()
    with open(MAPPING_OUTPUT_PATH, "w") as f:
        json.dump(mapping_dict, f, indent=4)
    print(f"Saved mapping successfully to {MAPPING_OUTPUT_PATH.name}")
else:
    print("\n[WARNING] Please review the [MISSING] entries in the report above.")