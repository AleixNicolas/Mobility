import os
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import geopandas as gpd
from shapely.geometry import box
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
PLOT_CRS = 3857
CALC_CRS = 25830 

COLOR_PALETTE = {
    "10 yr": "#FFD700", "100 yr": "#FF7F00", "500 yr": "#B22222",
    "DANA_31_10_2024": "#8A2BE2", "DANA_03_11_2024": "#FF1493",
    "DANA_05_11_2024": "#00CED1", "DANA_06_11_2024": "#32CD32",
    "DANA_08_11_2024": "#1E90FF"
}

SCENARIO_MAPPING = {
    "10 yr": "T10 (10-Year)", "100 yr": "T100 (100-Year)", "500 yr": "T500 (500-Year)",
    "DANA_31_10_2024": "31/10/2024", "DANA_03_11_2024": "03/11/2024",
    "DANA_05_11_2024": "05/11/2024", "DANA_06_11_2024": "06/11/2024",
    "DANA_08_11_2024": "08/11/2024"
}

REFERENCE_SCENARIOS = ["10 yr", "100 yr", "500 yr"]
DANA_SCENARIOS = ["DANA_31_10_2024", "DANA_03_11_2024", "DANA_05_11_2024", "DANA_06_11_2024", "DANA_08_11_2024"]
DANA_PEAK = "DANA_31_10_2024"

ALL_TARGETS = REFERENCE_SCENARIOS + DANA_SCENARIOS

def main():
    # 1. Establish the Clip Boundary
    base_area_path = ROUTING_DIR / "affected_area.gpkg"
    if not base_area_path.exists():
        logging.error(f"Missing base geometry at {base_area_path}")
        return
        
    base_gdf = gpd.read_file(base_area_path).to_crs(epsg=PLOT_CRS)
    pad = 5000  
    xmin, ymin, xmax, ymax = base_gdf.total_bounds
    extent = [xmin - pad, xmax + pad, ymin - pad, ymax + pad]
    clip_gdf = gpd.GeoDataFrame({'geometry': [box(*extent)]}, crs=PLOT_CRS)

    # Dictionaries to store metric results and geographic data for intersection
    spatial_metrics = {}
    loaded_zones_calc = {} 
    loaded_roads_calc = {} 

    # 2. Process, Clip, Plot, and Store
    for name in ALL_TARGETS:
        logging.info(f"Processing map and extracting geometries for: {name}")
        fig, ax = plt.subplots(figsize=(12, 12))
        scenario_folder = SCENARIO_BASE / name
        
        spatial_metrics[name] = {"flood_area_km2": 0.0, "safe_roads_km": 0.0, "cut_roads_km": 0.0}
        legend_elements = []

        # --- Flood Zones ---
        zone_path = scenario_folder / f"zone_flood_{name}.gpkg"
        if zone_path.exists():
            zone_raw = gpd.read_file(zone_path).to_crs(epsg=PLOT_CRS)
            zone_clipped = gpd.clip(zone_raw, clip_gdf)
            
            if not zone_clipped.empty:
                # Store the high-accuracy geometry in our dictionary for later use
                loaded_zones_calc[name] = zone_clipped.to_crs(epsg=CALC_CRS)
                spatial_metrics[name]["flood_area_km2"] = loaded_zones_calc[name].area.sum() / 1e6
                
                # Plot
                zone_color = COLOR_PALETTE.get(name, "blue")
                zone_clipped.plot(ax=ax, color=zone_color, alpha=0.4, edgecolor="none")
                legend_elements.append(Patch(facecolor=zone_color, alpha=0.4, label=f'Inundation ({SCENARIO_MAPPING.get(name, name)})'))

        # --- Roads ---
        safe_path = scenario_folder / f"safe_roads_{name}.gpkg"
        cut_path = scenario_folder / f"cut_roads_{name}.gpkg"
        
        if safe_path.exists() and cut_path.exists():
            safe_raw = gpd.read_file(safe_path).to_crs(epsg=PLOT_CRS)
            cut_raw = gpd.read_file(cut_path).to_crs(epsg=PLOT_CRS)
            
            safe_clipped = gpd.clip(safe_raw, clip_gdf)
            cut_clipped = gpd.clip(cut_raw, clip_gdf)
            
            if not safe_clipped.empty:
                spatial_metrics[name]["safe_roads_km"] = safe_clipped.to_crs(epsg=CALC_CRS).length.sum() / 1000
                safe_clipped.plot(ax=ax, color="black", linewidth=0.3, alpha=0.5)
                
            if not cut_clipped.empty:
                # Store the high-accuracy geometry in our dictionary for later use
                loaded_roads_calc[name] = cut_clipped.to_crs(epsg=CALC_CRS)
                spatial_metrics[name]["cut_roads_km"] = loaded_roads_calc[name].length.sum() / 1000
                cut_clipped.plot(ax=ax, color="red", linewidth=0.9, alpha=0.9)
                
            legend_elements.append(Line2D([0], [0], color='black', lw=1, alpha=0.5, label='Functional Roads'))
            legend_elements.append(Line2D([0], [0], color='red', lw=2, alpha=0.9, label='Impassable / Cut Roads'))

        # --- Aesthetics ---
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, zoom=12, attribution=False)
        
        # Scale bar
        scalebar_len = 20000 
        x0, y0 = extent[1] - scalebar_len - 5000, extent[2] + 3000
        ax.plot([x0, x0 + scalebar_len], [y0, y0], color='black', lw=3, solid_capstyle='round')
        ax.text(x0 + scalebar_len / 2, y0 - 1200, '20 km', ha='center', va='top', fontsize=11, weight='bold')

        # North arrow
        ax.annotate('', xy=(0.96, 0.96), xytext=(0.96, 0.89), xycoords='axes fraction', arrowprops=dict(facecolor='black', edgecolor='white', width=4, headwidth=10))
        ax.text(0.96, 0.98, 'N', transform=ax.transAxes, ha='center', va='bottom', fontsize=16, fontweight='bold')
        ax.axis("off")
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc="lower left", framealpha=0.9, fontsize=11, facecolor='white')

        # Clean file name
        clean_fn = name.replace(" ", "_")
        plt.savefig(MAPS_DIR / f"layer_map_{clean_fn}.pdf", format='pdf', bbox_inches="tight", pad_inches=0.1)
        plt.close()

    # 3. Intersection Analysis (DANA Peak vs References)
    report_path = MAPS_DIR / "00_spatial_and_intersection_report.txt"
    with open(report_path, "w") as f:
        # --- Section A: General Spatial Metrics ---
        f.write("=== CONSOLIDATED SPATIAL IMPACT REPORT ===\n")
        f.write("Note: All geometry clipped to Study Area + 5km Buffer.\n")
        f.write("=" * 85 + "\n\n")
        
        header = f"{'Scenario':<18} | {'Flood Area (km²)':<18} | {'Safe Roads (km)':<15} | {'Cut Roads (km)':<15}"
        f.write(header + "\n" + "-" * len(header) + "\n")
        
        for scn in ALL_TARGETS:
            f.write(f"{SCENARIO_MAPPING.get(scn, scn):<18} | "
                    f"{spatial_metrics[scn]['flood_area_km2']:<18.2f} | "
                    f"{spatial_metrics[scn]['safe_roads_km']:<15.1f} | "
                    f"{spatial_metrics[scn]['cut_roads_km']:<15.1f}\n")
        
        # --- Section B: Intersection Logic ---
        f.write("\n\n" + "=" * 85 + "\n")
        f.write(f"INTERSECTION ANALYSIS: {SCENARIO_MAPPING[DANA_PEAK]} (Observed) vs. REFERENCE MODELS\n")
        f.write("=" * 85 + "\n")
        
        # Retrieve the cached peak event geometries
        dana_zone = loaded_zones_calc.get(DANA_PEAK)
        dana_roads = loaded_roads_calc.get(DANA_PEAK)
        
        if dana_zone is not None and dana_roads is not None:
            dana_total_area = spatial_metrics[DANA_PEAK]['flood_area_km2']
            dana_total_cut = spatial_metrics[DANA_PEAK]['cut_roads_km']
            
            for ref in REFERENCE_SCENARIOS:
                ref_zone = loaded_zones_calc.get(ref)
                
                if ref_zone is not None:
                    # Area Overlap
                    intersect_poly = gpd.overlay(dana_zone, ref_zone, how='intersection')
                    overlap_area = intersect_poly.area.sum() / 1e6 
                    
                    # Road Overlap (Which DANA cut roads fall inside the Reference flood zone?)
                    intersect_lines = gpd.sjoin(dana_roads, ref_zone, how='inner', predicate='within')
                    overlap_roads = intersect_lines.length.sum() / 1000 
                    
                    area_pct = (overlap_area / dana_total_area) * 100 if dana_total_area > 0 else 0
                    road_pct = (overlap_roads / dana_total_cut) * 100 if dana_total_cut > 0 else 0
                    
                    f.write(f"\n--- Comparison with {SCENARIO_MAPPING[ref]} ---\n")
                    f.write(f"Shared Flooded Area  : {overlap_area:>8.2f} km2 ({area_pct:.1f}% of DANA area)\n")
                    f.write(f"Shared Cut Roads     : {overlap_roads:>8.1f} km  ({road_pct:.1f}% of DANA cut roads)\n")

    logging.info(f"Integrated analysis complete. Report saved to {report_path.name}")

if __name__ == "__main__":
    main()