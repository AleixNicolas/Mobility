import json
import logging
import re
import difflib
from pathlib import Path
import numpy as np
import pandas as pd

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from adjustText import adjust_text
import matplotlib.patheffects as path_effects
from matplotlib.patches import Patch
import contextily as ctx
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"
ROUTING_DIR = DATA_ROOT / "04_routing_networks"
SCENARIO_BASE = DATA_ROOT / "05_scenario_models"
MAPS_DIR = DATA_ROOT / "06_outputs" / "maps_I"

MAPS_DIR.mkdir(parents=True, exist_ok=True)

PLOT_CRS = 3857
POPULATION_COL = "population"

# ---------------------------------------------------------
# HELPER FUNCTIONS: NAME CLEANING & MATCHING
# ---------------------------------------------------------
# Hardcoded dictionary for known discrepancies and spatial overlaps
MANUAL_OVERRIDES = {
    "Castelló": "Castelló de la Ribera",
    "L'Alcúdia": "l'Alcúdia",
    "La Pobla Llarga": "La Pobla Llarga",
    "Montroi": "Montroi",
    "El Real de Gandia": "Real",          # Spatial mapping per user reference
    "Ademuz": "Castielfabib",             # Spatial mapping per user reference
    "Campillo de Altobuey": "Mira",       # Spatial mapping per user reference
    "Cofrentes": "Millares",              # Spatial mapping per user reference
    "Alcàntera de Xúquer": "Gavarda"      # Spatial mapping per user reference
}

def standardize_name(name):
    """Aggressively strips 'agregacion' and standardizes displaced articles."""
    if pd.isna(name) or not name: 
        return ""
    
    n = str(name)
    # Strip aggregation text entirely
    n = re.sub(r'(?i)\s*agregaci[óo]n de municipios.*', '', n)
    n = re.sub(r'(?i)\s*agrupaci[óo]n de municipios.*', '', n)
    
    # Handle bilingual slashes (take the first part)
    n = n.split('/')[0].strip()

    # Re-attach displaced comma articles to the front (e.g., "Alcúdia, l'" -> "L'Alcúdia")
    if n.lower().endswith(", l'"):
        n = "L'" + n[:-4].strip()
    elif n.lower().endswith(", el"):
        n = "El " + n[:-4].strip()
    elif n.lower().endswith(", la"):
        n = "La " + n[:-4].strip()
    elif n.lower().endswith(", los"):
        n = "Los " + n[:-5].strip()
    elif n.lower().endswith(", las"):
        n = "Las " + n[:-5].strip()
        
    return n

def find_json_match(gpkg_name, json_towns):
    """Finds the correct JSON node using overrides, direct match, or fuzzy matching."""
    std_name = standardize_name(gpkg_name)
    
    # 1. Check Manual Overrides First
    if std_name in MANUAL_OVERRIDES:
        override_name = MANUAL_OVERRIDES[std_name]
        if override_name in json_towns:
            return override_name
            
    # 2. Exact Match
    if std_name in json_towns:
        return std_name
        
    # 3. Fuzzy Match (80% similarity required)
    matches = difflib.get_close_matches(std_name, json_towns, n=1, cutoff=0.80)
    if matches:
        return matches[0]
        
    return None

def clean_label_name(name):
    """Creates a clean name for maps and text, pushing overrides to the display layer."""
    std = standardize_name(name)
    return MANUAL_OVERRIDES.get(std, std)

# ---------------------------------------------------------
# HELPER FUNCTIONS: RISK & PLOTTING
# ---------------------------------------------------------
def compute_individual_risk_factor(T_P, T_NP):
    if T_P is None: return 1.0
    if T_NP is None or T_P == 0: return 0.0
    return 1 - (T_NP / T_P)

def compute_municipal_risk_factor(T_P_dict, T_NP_dict, json_matched_name):
    if pd.isna(json_matched_name) or not json_matched_name:
        return np.nan

    keys = {k for k in T_P_dict.keys() & T_NP_dict.keys() if json_matched_name in k.split('__')}
    
    if not keys: 
        return np.nan

    r_sum = sum(compute_individual_risk_factor(T_P_dict[k][1], T_NP_dict[k][1]) for k in keys)
    return r_sum / len(keys)

def load_shortest_paths(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {k: (v.get("path", []), v.get("time", None)) for k, v in data.items()}

def plot_custom_map(polygons, column, cbar_label, cmap, norm, save_name, extent, highlight_critical=False):
    fig, ax = plt.subplots(figsize=(14, 14))

    # Plot polygons (Handle NaNs with hatched grey)
    polygons.plot(
        column=column, cmap=cmap, norm=norm, linewidth=0.5, edgecolor="black", ax=ax, alpha=0.85,
        missing_kwds={'color': 'lightgrey', 'hatch': '///', 'edgecolor': 'black', 'label': 'No Routing Data'}
    )
    
    legend_elements = []
    if highlight_critical:
        critical = polygons[polygons["risk"] > 0.9]
        if not critical.empty:
            critical.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2.5)
            legend_elements.append(Patch(facecolor='none', edgecolor='red', linewidth=2.5, label='Critical Isolation ($R_m > 0.9$)'))

    if polygons[column].isna().any():
        legend_elements.append(Patch(facecolor='lightgrey', hatch='///', edgecolor='black', label='No Routing Data (Match Failed)'))

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=11, attribution=False)

    texts = []
    for idx, row in polygons.iterrows():
        centroid = row.geometry.centroid
        if row["clean_label"]:
            txt = ax.text(centroid.x, centroid.y, row["clean_label"], fontsize=8, ha="center", weight='bold')
            txt.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground="white"), path_effects.Normal()])
            texts.append(txt)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5), precision=0.001)

    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_axis_off()

    scalebar_len = 20000 
    x0_sb, y0_sb = extent[0] + 5000, extent[2] + 5000
    ax.plot([x0_sb, x0_sb + scalebar_len], [y0_sb, y0_sb], color='black', lw=3)
    ax.text(x0_sb + scalebar_len/2, y0_sb - 1500, '20 km', ha='center', va='top', fontsize=10, weight='bold')

    ax.annotate('', xy=(0.95, 0.95), xytext=(0.95, 0.88), xycoords='axes fraction', arrowprops=dict(facecolor='black', edgecolor='white', width=4, headwidth=10))
    ax.text(0.95, 0.97, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=16, fontweight='bold')

    cax = ax.inset_axes([0.65, 0.83, 0.3, 0.02]) 
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label(cbar_label, fontsize=10, weight='bold', labelpad=8)
    cb.ax.tick_params(labelsize=9)

    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=11, facecolor='white')

    plt.savefig(MAPS_DIR / f"{save_name}.pdf", format="pdf", bbox_inches="tight")
    plt.close()

def extract_json_towns(T_dict):
    towns = set()
    for key in T_dict.keys():
        parts = key.split('__')
        if len(parts) == 2:
            towns.update(parts)
    return towns

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
def main():
    scenario_target = "DANA_31_10_2024"
    
    polygons = gpd.read_file(ROUTING_DIR / "affected_area.gpkg").to_crs(epsg=3857)
    
    T_NP_dict = load_shortest_paths(ROUTING_DIR / "shortest_paths_NP.json")
    T_P_dict = load_shortest_paths(SCENARIO_BASE / scenario_target / f"shortest_paths_{scenario_target}.json")

    json_towns = extract_json_towns(T_NP_dict)

    logging.info("Matching GPKG names to JSON routing nodes...")
    polygons["json_matched_name"] = polygons["name"].apply(lambda n: find_json_match(n, json_towns))
    
    # Apply clean_label_name so mapped values display correctly
    polygons["clean_label"] = polygons["name"].apply(clean_label_name)

    logging.info("Computing Isolation Risk (Rm)...")
    polygons["risk"] = polygons["json_matched_name"].apply(lambda n: compute_municipal_risk_factor(T_P_dict, T_NP_dict, n))
    
    # Debug Report for truly missing towns
    unmatched = polygons[polygons["risk"].isna()]
    if not unmatched.empty:
        with open(MAPS_DIR / f"04_unmatched_debug_{scenario_target}.txt", "w") as f:
            f.write("FAILED MATCHES (Likely missing from JSON entirely):\n")
            for _, r in unmatched.iterrows():
                f.write(f"Original: {r['name']} | Standardized: {standardize_name(r['name'])}\n")

    polygons[POPULATION_COL] = pd.to_numeric(polygons[POPULATION_COL], errors='coerce').fillna(1)
    polygons["weighted_linear"] = polygons["risk"] * polygons[POPULATION_COL]
    polygons["weighted_log"] = polygons["risk"] * np.log10(polygons[POPULATION_COL])

    # Pre-calculate Lat/Lon centroids before sorting
    polygons_4326 = polygons.to_crs(epsg=4326)
    polygons["lon"] = polygons_4326.geometry.centroid.x
    polygons["lat"] = polygons_4326.geometry.centroid.y

    xmin, ymin, xmax, ymax = polygons.total_bounds
    pad = 5000
    extent = [xmin - pad, xmax + pad, ymin - pad, ymax + pad]
    polygons = gpd.clip(polygons, box(*extent))

    plot_custom_map(polygons, "risk", "Isolation Risk Index ($R_m$)", cm.YlOrRd, Normalize(vmin=0, vmax=1), 
                    f"layer_map_risk_geographic_{scenario_target}", extent, highlight_critical=True)

    plot_custom_map(polygons, "weighted_linear", "Linear Impact (Risk x Pop)", cm.YlOrRd, 
                    Normalize(0, np.nanpercentile(polygons["weighted_linear"], 95)), 
                    f"layer_map_risk_weighted_linear_{scenario_target}", extent, highlight_critical=True)

    plot_custom_map(polygons, "weighted_log", "Log Impact ($R_m$ x $\log_{10}(Pop)$)", cm.YlOrRd, 
                    Normalize(0, polygons["weighted_log"].max()), 
                    f"layer_map_risk_weighted_log_{scenario_target}", extent, highlight_critical=True)

    with open(MAPS_DIR / f"03_municipal_risk_full_data_{scenario_target}.txt", "w") as f:
        f.write(f"=== FULL MUNICIPAL DATA REPORT ({scenario_target}) ===\n")
        f.write("Coordinates: EPSG:4326 (Longitude / Latitude)\n")
        f.write("=" * 145 + "\n")
        
        header = f"{'Municipality':<30} | {'Longitude':<12} | {'Latitude':<12} | {'Pop':<10} | {'Risk (Rm)':<10} | {'W_Linear':<15} | {'W_Log':<15}"
        f.write(header + "\n" + "-" * 145 + "\n")

        for _, row in polygons.sort_values(by="weighted_log", ascending=False, na_position='last').iterrows():
            lon_val = row["lon"]
            lat_val = row["lat"]
            pop_val = int(row[POPULATION_COL])
            risk_val = f"{row['risk']:.3f}" if pd.notna(row['risk']) else "N/A"
            w_lin_val = f"{row['weighted_linear']:.0f}" if pd.notna(row['weighted_linear']) else "N/A"
            w_log_val = f"{row['weighted_log']:.3f}" if pd.notna(row['weighted_log']) else "N/A"
            
            f.write(f"{row['clean_label']:<30} | {lon_val:<12.5f} | {lat_val:<12.5f} | {pop_val:<10,} | {risk_val:<10} | {w_lin_val:<15} | {w_log_val:<15}\n")

if __name__ == "__main__":
    main()