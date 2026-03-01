import math
import json
import logging
import time
from pathlib import Path

import networkx as nx
import igraph as ig
import osmnx as ox
import pandas as pd
import geopandas as gpd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# -----------------------
# PATH CONFIGURATION
# -----------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

ROUTING_DIR = DATA_ROOT / "04_routing_networks"
SCENARIO_BASE = DATA_ROOT / "05_scenario_models"
GLOBAL_METRICS_DIR = SCENARIO_BASE / "global_metrics"

GLOBAL_METRICS_DIR.mkdir(parents=True, exist_ok=True)

ox.settings.use_cache = True
ox.settings.log_console = True

# Target Files
GRAPH_NORMAL_PATH = ROUTING_DIR / "G_2nd.graphml"
BASELINE_PATHS = ROUTING_DIR / "shortest_paths_NP.json"

# DANA Specific Paths (for edge recovery analysis)
DANA_SCENARIO = "DANA_31_10_2024"
DANA_DIR = SCENARIO_BASE / DANA_SCENARIO

DANA_PATHS = DANA_DIR / f"shortest_paths_{DANA_SCENARIO}.json"
CUT_ROADS_DANA = DANA_DIR / f"cut_roads_{DANA_SCENARIO}.gpkg"
SAFE_ROADS_DANA = DANA_DIR / f"safe_roads_{DANA_SCENARIO}.graphml"

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def convert_nx_to_igraph(G_nx, weight_attr='travel_time'):
    """Convert NetworkX graph to igraph for fast shortest paths."""
    
    node_list = list(G_nx.nodes())
    node_to_index = {node: i for i, node in enumerate(node_list)}
    index_to_node = {i: node for node, i in node_to_index.items()}

    edge_weights = {}
    for u, v, attr in G_nx.edges(data=True):
        wt = attr.get(weight_attr)
        if wt is None: continue
        u_idx, v_idx = node_to_index[u], node_to_index[v]
        
        if (u_idx, v_idx) not in edge_weights or wt < edge_weights[(u_idx, v_idx)]:
            edge_weights[(u_idx, v_idx)] = wt

    G_ig = ig.Graph(directed=G_nx.is_directed())
    G_ig.add_vertices(len(node_list))
    G_ig.add_edges(list(edge_weights.keys()))
    G_ig.es['weight'] = list(edge_weights.values())
    
    return G_ig, node_to_index, index_to_node, edge_weights

def batch_shortest_paths_no_path(G_ig, base_special_nodes, node_to_index, index_to_node, node_to_muni, weight_attr='weight'):
    """Compute shortest path times only between surviving special nodes."""
    result = {}
    surviving_indices = [node_to_index[n] for n in base_special_nodes if n in node_to_index]
    index_to_muni_map = {idx: node_to_muni[index_to_node[idx]] for idx in surviving_indices}
    
    dist_matrix = G_ig.shortest_paths_dijkstra(
        source=surviving_indices,
        target=surviving_indices,
        weights=weight_attr
    )
    
    for i, src_idx in enumerate(surviving_indices):
        for j, tgt_idx in enumerate(surviving_indices):
            if src_idx == tgt_idx: continue
            
            dist = dist_matrix[i][j]
            key = f"{index_to_muni_map[src_idx]}__{index_to_muni_map[tgt_idx]}"
            result[key] = None if math.isinf(dist) else dist
            
    return result

def compute_individual_risk_factor(T_P, T_NP):
    if T_P is None or math.isinf(T_P): return 1.0
    if T_P == 0: return 0.0
    if T_NP is None: return 1.0
    return 1 - (T_NP / T_P)

def compute_risk_factor(T_P_dict, T_NP_dict):
    keys = set(T_P_dict.keys()) & set(T_NP_dict.keys())
    if not keys: return 0.0
    
    vals = []
    for k in keys:
        T_P_time = T_P_dict[k]['time'] if isinstance(T_P_dict[k], dict) else T_P_dict[k]
        T_NP_time = T_NP_dict[k]['time'] if isinstance(T_NP_dict[k], dict) else T_NP_dict[k]
        
        if T_NP_time is None: continue 
        vals.append(compute_individual_risk_factor(T_P_time, T_NP_time))
        
    return sum(vals) / len(vals) if vals else 0.0

# -----------------------
# MAIN EXECUTION
# -----------------------
def main():
    logging.info(f"Loading main baseline graph: {GRAPH_NORMAL_PATH.name}")
    G_2nd = ox.load_graphml(GRAPH_NORMAL_PATH)
    
    base_special_nodes = [n for n, attr in G_2nd.nodes(data=True) if pd.notna(attr.get('municipality'))]
    node_to_muni = {n: attr.get('municipality') for n, attr in G_2nd.nodes(data=True) if pd.notna(attr.get('municipality'))}

    logging.info("Loading baseline routing matrix...")
    with open(BASELINE_PATHS, "r") as f:
        T_NP_dictionary = json.load(f)

    # -----------------------
    # EDGE RISKS: DANA RECOVERY
    # -----------------------
    DANA = True
    if DANA:
        if not SAFE_ROADS_DANA.exists() or not CUT_ROADS_DANA.exists():
            logging.error("DANA safe/cut road files not found. Run Phase 3 first.")
            return

        logging.info("\nStarting edge recovery analysis for DANA peak flood...")
        
        with open(DANA_PATHS, "r") as f:
            T_P_dana_dict = json.load(f)
            
        base_risk_dana = compute_risk_factor(T_P_dana_dict, T_NP_dictionary)
        logging.info(f"Baseline DANA Risk: {base_risk_dana:.4f}")

        G_dana = ox.load_graphml(SAFE_ROADS_DANA)
        flood_edges = gpd.read_file(CUT_ROADS_DANA)
        
        if not isinstance(flood_edges.index, pd.MultiIndex):
            if "u" in flood_edges.columns and "v" in flood_edges.columns:
                flood_edges.set_index(["u", "v"], inplace=True)
        
        G_ig_dana, node_to_index, index_to_node, edge_weights = convert_nx_to_igraph(G_dana)
        
        edge_risks_dana = []
        start_time = time.time()

        for i, (idx_tuple, row) in enumerate(flood_edges.iterrows()):
            u, v = idx_tuple[0], idx_tuple[1]
            weight = row.get('travel_time')
            
            if u not in node_to_index or v not in node_to_index or weight is None:
                continue
                
            u_idx, v_idx = node_to_index[u], node_to_index[v]
            
            # Simulate restoration of the edge
            G_ig_dana.add_edges([(u_idx, v_idx)])
            new_eid = G_ig_dana.get_eid(u_idx, v_idx, directed=True)
            G_ig_dana.es[new_eid]['weight'] = float(weight)
            
            T_P_temp = batch_shortest_paths_no_path(G_ig_dana, base_special_nodes, node_to_index, index_to_node, node_to_muni)
            new_risk = compute_risk_factor(T_P_temp, T_NP_dictionary)
            
            # If delta is negative, recovery reduced the risk
            delta_risk = new_risk - base_risk_dana
            edge_risks_dana.append(((u, v), delta_risk))
            
            G_ig_dana.delete_edges(new_eid)
            
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                per_edge = elapsed / i
                remaining = per_edge * (len(flood_edges) - i)
                logging.info(f"DANA Progress: {i}/{len(flood_edges)} | Est. Remaining: {remaining/60:.1f} mins")

        edge_risks_dana.sort(key=lambda x: x[1]) # Recovery usually makes delta negative, so lowest is best
        edge_risks_dana_json = [{"edge": [u, v], "delta_risk": d_risk} for (u, v), d_risk in edge_risks_dana]

        with open(GLOBAL_METRICS_DIR / "edge_risks_DANA.json", "w") as f:
            json.dump(edge_risks_dana_json, f, indent=2)

        print("\nEdge recovery computation complete for DANA.")

if __name__ == "__main__":
    main()