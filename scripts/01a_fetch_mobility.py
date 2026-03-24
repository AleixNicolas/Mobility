import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyspainmobility import Mobility, Zones

# --- Configuration ---
ZONING_LEVEL_DATA = 'municipalities' 
ZONING_LEVEL_MAP = 'municipalities' 
START_DATE = '2024-10-01'
END_DATE = '2024-11-08'

# --- Path Management ---
# Dynamically get the directory of this script
SCRIPT_DIR = Path(__file__).resolve().parent

# Define the structured data hierarchy relative to the script
DATA_ROOT = SCRIPT_DIR.parent / "data"
MOBILITY_BASE = DATA_ROOT / "02_raw_downloaded" / "mobility"

SPATIAL_DIR = MOBILITY_BASE / "spatial"
METADATA_DIR = MOBILITY_BASE / "metadata"
MOBILITY_DIR = MOBILITY_BASE / "mobility_daily"

# Safely create all necessary directories
for d in [SPATIAL_DIR, METADATA_DIR, MOBILITY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Data Root Directory: {DATA_ROOT.resolve()}\n")

# --- 1. Download Zone/Geographic Data ---
print("--- 1. Downloading Zone Geometry & Metadata ---")
zones = Zones(zones=ZONING_LEVEL_MAP, version=2)

# A. Geometry
gdf_zones = zones.get_zone_geodataframe()
map_path = SPATIAL_DIR / f"spatial_zones_{ZONING_LEVEL_MAP}.geojson"
print(f"   > Saving Map Geometry to: {map_path}")

if gdf_zones.crs is not None:
    gdf_zones.to_crs(epsg=4326).to_file(map_path, driver='GeoJSON')
else:
    gdf_zones.to_file(map_path, driver='GeoJSON')

# B. Relations 
df_zone_relations = zones.get_zone_relations()
rel_path = METADATA_DIR / f"zone_relations_{ZONING_LEVEL_MAP}.csv"
print(f"   > Saving Zone Relations to: {rel_path}")
df_zone_relations.to_csv(rel_path, index=False)

print(f"   > Geometry Records: {len(gdf_zones)}")
print(f"   > Relation Records: {len(df_zone_relations)}")

# --- 2. Download Mobility Data ---
print("\n--- 2. Downloading Mobility OD Data ---")
mobility_data = Mobility(
    version=2, 
    zones=ZONING_LEVEL_DATA, 
    start_date=START_DATE, 
    end_date=END_DATE
)

df_mobility = mobility_data.get_od_data(keep_activity=True, return_df=True)
print(f"   > Total Records Loaded: {len(df_mobility)}")

# --- 3. Save Mobility Data by Day ---
print("\n--- 3. Saving Daily Mobility Files ---")

if 'date' in df_mobility.columns:
    df_mobility['date'] = pd.to_datetime(df_mobility['date'])
    
    for date_val, group in df_mobility.groupby(df_mobility['date'].dt.date):
        date_str = str(date_val)
        filename = MOBILITY_DIR / f"mobility_{ZONING_LEVEL_DATA}_{date_str}.csv"
        
        print(f"   > Saving {filename.name} ({len(group)} rows)...")
        group.to_csv(filename, index=False)
        
    print("\nAll data has been saved.")
else:
    print("Error: 'date' column missing. Saving as single file.")
    df_mobility.to_csv(MOBILITY_DIR / "mobility_full_dump.csv", index=False)