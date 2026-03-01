import logging
from pathlib import Path

import pandas as pd
import geopandas as gpd
import osmnx as ox

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ============================================================
# Configuration & Path Management
# ============================================================

ox.settings.use_cache = True
ox.settings.log_console = True

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

# Input Directories
FLOOD_INPUTS = DATA_ROOT / "01_raw_manual" / "flood_scenarios"
ROUTING_DIR = DATA_ROOT / "04_routing_networks"

# Output Directory (Base)
SCENARIO_BASE = DATA_ROOT / "05_scenario_models"
SCENARIO_BASE.mkdir(parents=True, exist_ok=True)

# ============================================================
# Utility Functions
# ============================================================

def parse_depth_range(val):
    if pd.isna(val): return None
    val = str(val).strip()
    if val.startswith('Below'): return float(val[5:].strip()) / 2
    if val.startswith('>'): return float(val[1:].strip())
    if '-' in val:
        parts = val.split('-')
        try: return (float(parts[0].strip()) + float(parts[1].strip())) / 2
        except: return None
    try: return float(val)
    except: return None


def clip_flood_zone(return_crs, name, clip_geom, input_path, output_path):
    """Clip flood polygons to study area and save to GeoPackage."""
    if output_path.exists():
        logging.info(f"Loading {name} from {output_path}")
        clipped = gpd.read_file(output_path, layer=name).to_crs(return_crs)
    else:
        logging.info(f"Clipping and saving {name} to {output_path}")
        flood = gpd.read_file(input_path).to_crs(return_crs)
        clipped = gpd.clip(flood, clip_geom)
        clipped.to_file(output_path, layer=name, driver="GPKG")
    return clipped


def flood_depth_zones(name, input_path, output_path):
    """Parse flood depth shapefiles and save standardized depth GeoPackage."""
    layer = "depth_val"
    if output_path.exists():
        logging.info(f"Loading {layer} from {output_path}")
        depth = gpd.read_file(output_path, layer=layer)
    else:
        logging.info(f"Saving {layer} to {output_path}")
        depth = gpd.read_file(input_path)
        depth["depth_val"] = depth["value"].apply(parse_depth_range)
        depth.to_file(output_path, layer=layer, driver="GPKG")
    return depth


def tag_flooded_roads(edges, nodes, flood_zones, name, scenario_dir):
    """
    Tag edges as flooded or safe using fast spatial joins, and save outputs.
    """
    tagged_path = scenario_dir / f"tagged_roads_{name}.gpkg"
    cut_gpkg_path = scenario_dir / f"cut_roads_{name}.gpkg"
    safe_gpkg_path = scenario_dir / f"safe_roads_{name}.gpkg"
    cut_graphml_path = scenario_dir / f"cut_roads_{name}.graphml"
    safe_graphml_path = scenario_dir / f"safe_roads_{name}.graphml"

    if tagged_path.exists():
        logging.info(f"Loading tagged edges for {name} from {tagged_path}")
        edges = gpd.read_file(tagged_path)
    else:
        logging.info(f"Tagging flooded edges for {name} via spatial join...")
        bounds = edges.total_bounds
        flood_subset = flood_zones.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        
        # Optimize intersection using sjoin
        edges = edges.copy()
        flooded_indices = gpd.sjoin(edges, flood_subset[['geometry']], how='inner', predicate='intersects').index.unique()
        edges["in_flood_zone"] = edges.index.isin(flooded_indices)

        edges.to_file(tagged_path, layer=name, driver="GPKG")
        logging.info(f"Saved tagged edges to {tagged_path}")

    # Ensure MultiIndex for OSMnx compatibility
    def enforce_multiindex(df):
        if not isinstance(df.index, pd.MultiIndex):
            if "u" in df.columns and "v" in df.columns:
                if "key" not in df.columns: df["key"] = 0
                df.set_index(["u", "v", "key"], inplace=True)
        return df

    edges = enforce_multiindex(edges)
    
    safe_edges = enforce_multiindex(edges[~edges["in_flood_zone"]].copy())
    cut_edges = enforce_multiindex(edges[edges["in_flood_zone"]].copy())

    # Save Edge GeoPackages
    safe_edges.to_file(safe_gpkg_path, layer=name, driver="GPKG")
    cut_edges.to_file(cut_gpkg_path, layer=name, driver="GPKG")

    # Build graphs (nodes already contain 'municipality' and 'clase' from StreetData.py)
    G_safe = ox.graph_from_gdfs(nodes, safe_edges)
    G_cut = ox.graph_from_gdfs(nodes, cut_edges)

    ox.save_graphml(G_safe, safe_graphml_path)
    ox.save_graphml(G_cut, cut_graphml_path)
    logging.info(f"Saved safe/cut GraphMLs for {name}")

    return edges, G_safe, G_cut


# ============================================================
# Core Execution Setup
# ============================================================

FILE_2ND = ROUTING_DIR / "neighbors_2_area.gpkg"
logging.info(f"Loading study area from {FILE_2ND}")
neighbors_2_area = gpd.read_file(FILE_2ND)

GRAPH_PATH = ROUTING_DIR / "G_2nd.graphml"
logging.info(f"Loading base graph from {GRAPH_PATH}")
G_2nd = ox.load_graphml(GRAPH_PATH)

nodes, edges = ox.graph_to_gdfs(G_2nd)

# Define Files relative to the new architecture
scenarios = {
    "10 yr": {"poly": "laminaspb-q10/Q10_2Ciclo_PB_20241121.shp"},
    "100 yr": {"poly": "laminaspb-q100/Q100_2Ciclo_PB_20241121_ETRS89.shp"},
    "500 yr": {"poly": "laminaspb-q500/Q500_2Ciclo_PB_20241121_ETRS89.shp"},
    "DANA_31_10_2024": {
        "poly": "EMSR773_AOI01_DEL_PRODUCT_v1/EMSR773_AOI01_DEL_PRODUCT_observedEventA_v1.shp",
        "depth": "EMSR773_AOI01_DEL_PRODUCT_v1/EMSR773_AOI01_DEL_PRODUCT_floodDepthA_v1.shp"
    },
    "DANA_03_11_2024": {
        "poly": "EMSR773_AOI01_DEL_MONIT01_v1/EMSR773_AOI01_DEL_MONIT01_observedEventA_v1.shp",
        "depth": "EMSR773_AOI01_DEL_MONIT01_v1/EMSR773_AOI01_DEL_MONIT01_floodDepthA_v1.shp"
    },
    "DANA_05_11_2024": {
        "poly": "EMSR773_AOI01_DEL_MONIT02_v1/EMSR773_AOI01_DEL_MONIT02_observedEventA_v1.shp",
        "depth": "EMSR773_AOI01_DEL_MONIT02_v1/EMSR773_AOI01_DEL_MONIT02_floodDepthA_v1.shp"
    },
    "DANA_06_11_2024": {
        "poly": "EMSR773_AOI01_DEL_MONIT03_v1/EMSR773_AOI01_DEL_MONIT03_observedEventA_v1.shp",
        "depth": "EMSR773_AOI01_DEL_MONIT03_v1/EMSR773_AOI01_DEL_MONIT03_floodDepthA_v1.shp"
    },
    "DANA_08_11_2024": {
        "poly": "EMSR773_AOI01_DEL_MONIT04_v1/EMSR773_AOI01_DEL_MONIT04_observedEventA_v1.shp",
        "depth": "EMSR773_AOI01_DEL_MONIT04_v1/EMSR773_AOI01_DEL_MONIT04_floodDepthA_v1.shp"
    }
}

# ============================================================
# Main Loop
# ============================================================

for name, files in scenarios.items():
    logging.info(f"\n=== Processing {name} ===")
    
    # Create scenario-specific subfolder
    SCENARIO_DIR = SCENARIO_BASE / name
    SCENARIO_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process Flood Polygons
    poly_input = FLOOD_INPUTS / files["poly"]
    poly_output = SCENARIO_DIR / f"zone_flood_{name}.gpkg"
    
    if poly_input.exists():
        flood_zone = clip_flood_zone(edges.crs, name, neighbors_2_area, poly_input, poly_output)
        tag_flooded_roads(edges, nodes, flood_zone, name, SCENARIO_DIR)
    else:
        logging.warning(f"Polygon source missing: {poly_input}")

    # Process Flood Depths (if available)
    if "depth" in files:
        depth_input = FLOOD_INPUTS / files["depth"]
        depth_output = SCENARIO_DIR / f"depth_{name}.gpkg"
        
        if depth_input.exists():
            flood_depth_zones(name, depth_input, depth_output)
        else:
            logging.warning(f"Depth source missing: {depth_input}")

logging.info("\nFlood zone processing complete!")