import logging
from pathlib import Path
import json

import numpy as np
import pandas as pd
import geopandas as gpd

import networkx as nx
import osmnx as ox
from scipy.spatial import cKDTree

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- 1. PATH CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

# Input Directories
MANUAL_DIR = DATA_ROOT / "01_raw_manual"
DOWNLOADED_DIR = DATA_ROOT / "02_raw_downloaded"
SHARED_DIR = DATA_ROOT / "03_processed_shared"

# Output Directory
ROUTING_DIR = DATA_ROOT / "04_routing_networks"
ROUTING_DIR.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True

# Target Output Files
FILE_STUDY = ROUTING_DIR / "affected_area.gpkg"
FILE_1ST = ROUTING_DIR / "neighbors_1_area.gpkg"
FILE_2ND = ROUTING_DIR / "neighbors_2_area.gpkg"
GRAPH_PATH_2ND = ROUTING_DIR / "G_2nd.graphml"

# Explicit Input Files
GEOJSON_PATH = DOWNLOADED_DIR / "mobility" / "spatial" / "spatial_zones_municipalities.geojson"
MAPPING_PATH = SHARED_DIR / "final_mitma_mapping.json"
DICT_PATH = MANUAL_DIR / "dictionaries" / "affected_municipalities_dictionary.json"
POI_PATH = MANUAL_DIR / "points_of_interest" / "BTN_POI_Servicios_instalaciones_gpkg" / "BTN_POI_Servicios_instalaciones_gpkg.gpkg"

# --- 2. CORE FUNCTIONS ---
def build_graph_from_layer(layer_gdf, graph_path, mitma_gdf, urban_center_dict, network_type="drive"):
    if graph_path.exists():
        logging.info(f"Loading saved graph from {graph_path}...")
        G = ox.load_graphml(graph_path)
    else:
        logging.info(f"Building graph for {graph_path}...")
        polygon = layer_gdf.unary_union

        G = ox.graph_from_polygon(
            polygon,
            network_type=network_type,
            simplify=True,
            retain_all=False,
            truncate_by_edge=True
        )

        # Estimate speed & travel time
        for u, v, k, data in G.edges(keys=True, data=True):
            if "length" in data:
                speed = None
                maxspeed = data.get("maxspeed")
                if isinstance(maxspeed, list): maxspeed = maxspeed[0]
                if maxspeed:
                    try: speed = float(str(maxspeed).split()[0])
                    except ValueError: pass
                
                if speed is None:
                    highway = data.get("highway")
                    if isinstance(highway, list): highway = highway[0]
                    speed = {
                        "motorway": 120, "motorway_link": 60, "trunk": 100,
                        "primary": 80, "secondary": 60, "tertiary": 50,
                        "residential": 30, "living_street": 10,
                        "unclassified": 40, "service": 20
                    }.get(highway, 50)

                surface = data.get("surface", "").lower()
                factor_map = {
                    "paved": 1.0, "asphalt": 1.0, "concrete": 1.0,
                    "cobblestone": 0.8, "gravel": 0.7, "dirt": 0.6,
                    "ground": 0.6, "sand": 0.5, "unpaved": 0.7,
                    "compacted": 0.85, "fine_gravel": 0.9
                }
                for key, factor in factor_map.items():
                    if key in surface:
                        speed *= factor
                        break

                speed_mps = speed * 1000 / 3600
                data["travel_time"] = data["length"] / speed_mps 

        # Assign municipalities to specific center nodes via KDTree
        nodes, edges = ox.graph_to_gdfs(G)
        if "municipality" not in nodes.columns or nodes["municipality"].isna().all():
            logging.info("Assigning municipality centers to specific graph nodes...")
            node_coords = np.array([(geom.y, geom.x) for geom in nodes.geometry])
            kdtree = cKDTree(node_coords)
            
            # Initialize all as None
            nodes["municipality"] = None
            
            for name, coords in urban_center_dict.items():
                try:
                    lat, lon = coords
                    _, idx = kdtree.query([lat, lon], k=1)
                    nearest_node = nodes.index[idx]
                    nodes.at[nearest_node, "municipality"] = name
                except Exception as e:
                    logging.warning(f"Could not assign node for {name}: {e}")
                    
            # Update graph nodes
            for node_id, row in nodes.iterrows():
                if pd.notna(row["municipality"]):
                    G.nodes[node_id]["municipality"] = row["municipality"]
                    
            ox.save_graphml(G, graph_path)
            logging.info(f"Updated graph saved to: {graph_path}")
        else:
            logging.info("Municipality names already exist in graph; skipping reassignment.")

    return G

def assign_single_attribute_to_graph(G, gdf, attr='clase', graph_path=None):
    nodes, _ = ox.graph_to_gdfs(G)
    node_coords = np.array([(geom.y, geom.x) for geom in nodes.geometry])
    kdtree = cKDTree(node_coords)

    if attr not in nodes.columns:
        nodes[attr] = None

    for idx, row in gdf.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        _, nearest_idx = kdtree.query([lat, lon], k=1)
        nearest_node = nodes.index[nearest_idx]
        nodes.at[nearest_node, attr] = row[attr]

    for node_id, row in nodes.iterrows():
        G.nodes[node_id][attr] = row[attr]

    if graph_path:
        ox.save_graphml(G, graph_path)
        logging.info(f"Graph updated with '{attr}' and saved to {graph_path}")

    return G

# --- 3. MAIN WORKFLOW ---

# A. Load Verified Geometries and Mapping
logging.info("Loading official MITMA spatial geometries...")
mitma_gdf = gpd.read_file(GEOJSON_PATH)

logging.info("Loading validated target mapping...")
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    target_mapping = json.load(f)

target_ids = set(target_mapping.values())

# B. Extract Coordinates from Dictionary
logging.info("Loading urban center coordinates...")
with open(DICT_PATH, "r", encoding="utf-8") as f:
    regions = json.load(f)

urban_center_dict = {**regions.get("Valencia", {}).get("coordinates", {}), 
                     **regions.get("Cuenca", {}).get("coordinates", {})}

# C. Generate Topologically Perfect Study Areas
if FILE_STUDY.exists() and FILE_1ST.exists() and FILE_2ND.exists():
    logging.info("Loading existing study area GPKGs...")
    affected_area = gpd.read_file(FILE_STUDY)
    neighbors_1_area = gpd.read_file(FILE_1ST)
    neighbors_2_area = gpd.read_file(FILE_2ND)
else:
    logging.info("Generating base affected area...")
    affected_area = mitma_gdf[mitma_gdf['id'].isin(target_ids)].copy()
    affected_area.to_file(FILE_STUDY, driver="GPKG")
    
    logging.info("Calculating 1st Degree Neighbors...")
    union_affected = affected_area.unary_union
    neighbors_1_area = mitma_gdf[mitma_gdf.geometry.intersects(union_affected)].copy()
    neighbors_1_area.to_file(FILE_1ST, driver="GPKG")
    
    logging.info("Calculating 2nd Degree Neighbors...")
    union_1 = neighbors_1_area.unary_union
    neighbors_2_area = mitma_gdf[mitma_gdf.geometry.intersects(union_1)].copy()
    neighbors_2_area.to_file(FILE_2ND, driver="GPKG")

# D. Build Graph
G_2nd = build_graph_from_layer(neighbors_2_area, GRAPH_PATH_2ND, mitma_gdf, urban_center_dict)

# E. Process POIs
selected_clases = [
    "Hospital",
    "Otros centros sanitarios",
    "Orden público-seguridad",
    "Emergencias"
]

if POI_PATH.exists():
    logging.info("Loading Special Interest Points...")
    special_interest_points = gpd.read_file(POI_PATH)
    filtered_points = special_interest_points[special_interest_points['clase'].isin(selected_clases)].copy()

    G_2nd = assign_single_attribute_to_graph(
        G_2nd,
        filtered_points,
        attr='clase',
        graph_path=GRAPH_PATH_2ND,
    )
else:
    logging.warning(f"Could not find POI dataset at {POI_PATH}. Skipping POI assignment.")

logging.info("Network generation complete.")