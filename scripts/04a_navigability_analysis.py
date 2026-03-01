import math
import json
import logging
from pathlib import Path
from itertools import permutations

import pandas as pd
import networkx as nx
import igraph as ig
import osmnx as ox

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===============================================================
# PATH CONFIGURATION
# ===============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

ROUTING_DIR = DATA_ROOT / "04_routing_networks"
SCENARIO_BASE = DATA_ROOT / "05_scenario_models"
GLOBAL_METRICS_DIR = SCENARIO_BASE / "global_metrics"

GLOBAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True

COMPUTE_T_P = True
COMPUTE_T_NP = True

# Target Baseline Files
GRAPH_NORMAL_PATH = ROUTING_DIR / "G_2nd.graphml"
BASELINE_PATHS = ROUTING_DIR / "shortest_paths_NP.json"

scenarios = [
    "10 yr", "100 yr", "500 yr",
    "DANA_31_10_2024", "DANA_03_11_2024",
    "DANA_05_11_2024", "DANA_06_11_2024", "DANA_08_11_2024"
]

# ===============================================================
# GRAPH CONVERSION
# ===============================================================
def convert_nx_to_igraph(G_nx, weight_attr="travel_time"):
    """Convert NetworkX graph to igraph for fast shortest paths."""
    node_list = list(G_nx.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    edge_list = []
    weights = []
    for u, v, data in G_nx.edges(data=True):
        w = data.get(weight_attr)
        if w is None: continue
        edge_list.append((node_to_index[u], node_to_index[v]))
        weights.append(w)

    G_ig = ig.Graph(directed=G_nx.is_directed())
    G_ig.add_vertices(len(node_list))
    G_ig.add_edges(edge_list)
    G_ig.es["weight"] = weights

    return G_ig, node_to_index, index_to_node

# ===============================================================
# SHORTEST PATH COMPUTATIONS
# ===============================================================
def compute_paths_with_routes(G_ig, base_special_nodes, node_to_index, index_to_node, node_to_muni):
    """
    Computes routes while ensuring a perfectly exhaustive OD matrix.
    If a node was destroyed in a flood scenario, its routes default to None.
    """
    
    # Pre-fill result dictionary with disconnected state to handle destroyed nodes
    result = {}
    for src, tgt in permutations(base_special_nodes, 2):
        src_m = node_to_muni[src]
        tgt_m = node_to_muni[tgt]
        result[f"{src_m}__{tgt_m}"] = {"path": [], "time": None}

    # Identify which baseline nodes survived the flood cut
    surviving_nodes = [n for n in base_special_nodes if n in node_to_index]
    surviving_indices = [node_to_index[n] for n in surviving_nodes]

    # Compute paths strictly between surviving nodes
    for src_idx in surviving_indices:
        distances = G_ig.distances(source=src_idx, target=surviving_indices, weights="weight")[0]
        routes = G_ig.get_shortest_paths(src_idx, to=surviving_indices, weights="weight", output="vpath")

        src_m = node_to_muni[index_to_node[src_idx]]

        for tgt_idx, dist, vpath in zip(surviving_indices, distances, routes):
            if src_idx == tgt_idx: continue

            tgt_m = node_to_muni[index_to_node[tgt_idx]]
            key = f"{src_m}__{tgt_m}"

            if not math.isinf(dist):
                result[key] = {"path": [index_to_node[i] for i in vpath], "time": dist}

    return result

def load_or_compute_paths(filepath, G_nx, base_special_nodes, node_to_muni):
    """Load or compute full paths + times."""
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)

    logging.info(f"Computing full paths for {filepath.name} ...")
    G_ig, node_to_index, index_to_node = convert_nx_to_igraph(G_nx)
    
    result = compute_paths_with_routes(G_ig, base_special_nodes, node_to_index, index_to_node, node_to_muni)

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)

    return result

# ===============================================================
# RISK FUNCTIONS
# ===============================================================
def individual_risk(TP, TNP):
    """Compute risk per pair."""
    if TP is None or math.isinf(TP): return 1
    if TP == 0: return 0
    return 1 - (TNP / TP)

def compute_risk(TP_dict, TNP_dict):
    keys = set(TP_dict.keys()) & set(TNP_dict.keys())
    if not keys: return 0

    vals = []
    for k in keys:
        TP = TP_dict[k]["time"] if isinstance(TP_dict[k], dict) else TP_dict[k]
        TNP = TNP_dict[k]["time"] if isinstance(TNP_dict[k], dict) else TNP_dict[k]
        
        # If there was no baseline path, risk change is technically undefined, skip or assume 0 change
        if TNP is None: continue 
        vals.append(individual_risk(TP, TNP))

    return sum(vals) / len(vals) if vals else 0

# ===============================================================
# MAIN
# ===============================================================
def main():
    logging.info(f"Loaded normal graph: {GRAPH_NORMAL_PATH.name}")
    G_2nd = ox.load_graphml(GRAPH_NORMAL_PATH)

    # Extract the specific OD nodes assigned in StreetData.py
    base_special_nodes = [n for n, d in G_2nd.nodes(data=True) if pd.notna(d.get("municipality"))]
    node_to_muni = {n: d.get("municipality") for n, d in G_2nd.nodes(data=True) if pd.notna(d.get("municipality"))}
    
    logging.info(f"Identified {len(base_special_nodes)} municipal center nodes for routing.")

    # ---- Compute T_NP (Baseline) ----
    if COMPUTE_T_NP:
        T_NP = load_or_compute_paths(BASELINE_PATHS, G_2nd, base_special_nodes, node_to_muni)
    else:
        T_NP = {}

    TP = {}
    R = {}

    if COMPUTE_T_P:
        for i, name in enumerate(scenarios):
            logging.info(f"\n=== Scenario: {name} ===")
            
            # FIX: Dynamically find the scenario subfolder
            SCENARIO_DIR = SCENARIO_BASE / name
            
            safe_graph_path = SCENARIO_DIR / f"safe_roads_{name}.graphml"
            out_paths = SCENARIO_DIR / f"shortest_paths_{name}.json"
            
            if not safe_graph_path.exists():
                logging.warning(f"Missing {safe_graph_path.name}. Skipping.")
                continue
                
            G_safe = ox.load_graphml(safe_graph_path)

            # Compute scenario paths using the exact baseline special nodes
            TP[name] = load_or_compute_paths(out_paths, G_safe, base_special_nodes, node_to_muni)

            # Compute scenario risk
            R[name] = compute_risk(TP[name], T_NP)

            logging.info(f"Progress: {i+1}/{len(scenarios)} → Global Risk Metric (R) = {R[name]:.4f}")

        # Save risk JSON to the shared global metrics folder
        with open(GLOBAL_METRICS_DIR / "R_G.json", "w") as f:
            json.dump(R, f, indent=2)

    logging.info("\nNavigability computations finished.")

if __name__ == "__main__":
    main()