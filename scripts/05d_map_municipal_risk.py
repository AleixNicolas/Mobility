import json
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from adjustText import adjust_text
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import contextily as ctx

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------
# PATH CONFIGURATION
# ---------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

ROUTING_DIR = DATA_ROOT / "04_routing_networks"
SCENARIO_BASE = DATA_ROOT / "05_scenario_models"
MAPS_DIR = DATA_ROOT / "06_outputs" / "maps_I"

MAPS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def compute_individual_risk_factor(T_P, T_NP):
    """Compute risk factor for a single path."""
    if T_P is None: return 1.0
    if T_NP is None: return 0.0
    if T_P == 0: return 1.0
    return 1 - (T_NP / T_P)

def compute_municipal_risk_factor(T_P_dict, T_NP_dict, municipality):
    """Compute bidirectional average risk factor for a municipality."""
    keys = {k for k in T_P_dict.keys() & T_NP_dict.keys() if municipality in k}
    if not keys: return 0.0

    R = 0
    for k in keys:
        _, T_P = T_P_dict[k]
        _, T_NP = T_NP_dict[k]
        R += compute_individual_risk_factor(T_P, T_NP)

    return R / len(keys)

def load_shortest_paths(filepath):
    """Load JSON paths into tuple dictionaries."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    cleaned = {}
    for k, v in data.items():
        path = v.get("path", [])
        time = v.get("time", None)
        cleaned[k] = (path, time)

    return cleaned

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
def main():
    scenario_target = "DANA_31_10_2024"
    
    # Load Geometries
    study_area_path = ROUTING_DIR / "affected_area.gpkg"
    logging.info("Loading study area geometries...")
    polygons = gpd.read_file(study_area_path).to_crs(epsg=3857)

    # Load Routing Matrices
    logging.info("Loading routing matrices...")
    T_NP_dictionary = load_shortest_paths(ROUTING_DIR / "shortest_paths_NP.json")
    
    # Target the specific scenario subfolder
    scenario_dir = SCENARIO_BASE / scenario_target
    scenario_path = scenario_dir / f"shortest_paths_{scenario_target}.json"
    
    if not scenario_path.exists():
        logging.error(f"Missing routing matrix for {scenario_target} at {scenario_path}")
        return
        
    T_P_dictionary = load_shortest_paths(scenario_path)

    # Compute Risk
    logging.info(f"Computing municipal risk for {scenario_target}...")
    polygons["risk"] = polygons["name"].apply(
        lambda name: compute_municipal_risk_factor(T_P_dictionary, T_NP_dictionary, name)
    )

    # Plotting
    logging.info("Rendering map...")
    norm = Normalize(vmin=0, vmax=1) # Lock norm to 0-1 for risk interpretation
    cmap = cm.get_cmap("YlOrRd")

    fig, ax = plt.subplots(figsize=(12, 12))

    polygons.plot(column="risk", cmap=cmap, norm=norm, linewidth=0.8, edgecolor="black", ax=ax, legend=False)

    # Bounding Box
    xmin, ymin, xmax, ymax = polygons.total_bounds
    pad = 5000
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=11, attribution=False)

    # Labels
    logging.info("Calculating label placements...")
    texts = []
    for idx, row in polygons.iterrows():
        centroid = row.geometry.centroid
        txt = ax.text(centroid.x, centroid.y, row["name"], fontsize=8, ha="center", color="black", weight='medium')
        txt.set_path_effects([path_effects.Stroke(linewidth=1.2, foreground="white"), path_effects.Normal()])
        texts.append(txt)

    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5), precision=0.001)

    # Scale bar & North Arrow
    scalebar_len = 20000 
    x0, y0 = xmax - scalebar_len - 15000, ymin + 15000

    ax.plot([x0, x0 + scalebar_len], [y0, y0], color='black', lw=3)
    ax.text(x0 + scalebar_len/2, y0 - 2500, '20 km', ha='center', va='top', fontsize=10)

    ax.annotate('', xy=(x0 + scalebar_len/2, y0 + 15000), xytext=(x0 + scalebar_len/2, y0 + 5000),
                arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=8))
    ax.text(x0 + scalebar_len/2, y0 + 16000, 'N', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Colorbar
    cax = inset_axes(ax, width="25%", height="2%", loc="lower right", bbox_to_anchor=(-0.05, 0.05, 1, 1), bbox_transform=ax.transAxes)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Municipal Isolation Risk ($R_m$)", fontsize=10)
    cb.ax.tick_params(labelsize=9)

    ax.set_axis_off()
    ax.margins(0)
    fig.subplots_adjust(0, 0, 1, 1)

    save_path = MAPS_DIR / f"municipality_risk_map_{scenario_target}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    
    logging.info(f"Map successfully saved to {save_path.name}")

if __name__ == "__main__":
    main()