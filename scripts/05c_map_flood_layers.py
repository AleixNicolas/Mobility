import os
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import geopandas as gpd
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
# CONFIGURATION & PALETTES
# ---------------------------------------------------------
ZONES_ACTIVE = True
ROADS_ACTIVE = True

COLOR_PALETTE = {
    "10 yr": "#FFD700",
    "100 yr": "#FF7F00",
    "500 yr": "#B22222",
    "DANA_31_10_2024": "#8A2BE2",
    "DANA_03_11_2024": "#FF1493",
    "DANA_05_11_2024": "#00CED1",
    "DANA_06_11_2024": "#32CD32",
    "DANA_08_11_2024": "#1E90FF"
}

TARGET_SCENARIOS = [
    "DANA_31_10_2024",
    "DANA_03_11_2024",
    "DANA_05_11_2024",
    "DANA_08_11_2024"
]

def main():
    # Load base study area to enforce a static bounding box across all maps
    base_area_path = ROUTING_DIR / "affected_area.gpkg"
    if not base_area_path.exists():
        logging.error(f"Missing base geometry at {base_area_path}")
        return
        
    base_gdf = gpd.read_file(base_area_path).to_crs(epsg=3857)
    pad = 5000  # meters
    xmin, ymin, xmax, ymax = base_gdf.total_bounds
    extent = [xmin - pad, xmax + pad, ymin - pad, ymax + pad]

    for name in TARGET_SCENARIOS:
        logging.info(f"Plotting layer map for scenario: {name}")
        fig, ax = plt.subplots(figsize=(10, 10))

        scenario_folder = SCENARIO_BASE / name

        # --- Flood zones ---
        if ZONES_ACTIVE:
            zone_path = scenario_folder / f"zone_flood_{name}.gpkg"
            if zone_path.exists():
                zone_gdf = gpd.read_file(zone_path).to_crs(epsg=3857)
                zone_gdf.plot(ax=ax, color=COLOR_PALETTE.get(name, "blue"), alpha=0.4, edgecolor="none")
            else:
                logging.warning(f"Missing flood zone geometry for {name}")

        # --- Roads ---
        if ROADS_ACTIVE:
            safe_path = scenario_folder / f"safe_roads_{name}.gpkg"
            cut_path = scenario_folder / f"cut_roads_{name}.gpkg"
            
            if safe_path.exists() and cut_path.exists():
                safe_gdf = gpd.read_file(safe_path).to_crs(epsg=3857)
                cut_gdf = gpd.read_file(cut_path).to_crs(epsg=3857)

                safe_gdf.plot(ax=ax, color="black", linewidth=0.3, alpha=0.6, label="Safe roads")
                cut_gdf.plot(ax=ax, color="red", linewidth=0.6, alpha=0.8, label="Cut roads")
            else:
                logging.warning(f"Missing road geometries for {name}")

        # --- Map styling ---
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=12, attribution=False)

        # Scale bar (20km)
        scalebar_len = 20000 
        x0, y0 = extent[1] - scalebar_len - 15000, extent[2] + 5000
        ax.plot([x0, x0 + scalebar_len], [y0, y0], color='black', lw=3)
        ax.text(x0 + scalebar_len / 2, y0 - 1500, '20 km', ha='center', va='top', fontsize=10)

        # North arrow
        ax.annotate('', xy=(0.95, 0.95), xytext=(0.95, 0.88), xycoords='axes fraction',
                    arrowprops=dict(facecolor='black', edgecolor='black', width=2, headwidth=8))
        ax.text(0.95, 0.96, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=14, fontweight='bold')

        ax.axis("off")
        
        # Prevent duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="lower left")

        save_path = MAPS_DIR / f"layer_map_{name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
        plt.close()
        logging.info(f"Saved {save_path.name}")

if __name__ == "__main__":
    main()