import math
import json
import logging
from pathlib import Path

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

ox.settings.use_cache = True
ox.settings.log_console = True

GRAPH_NORMAL_PATH = ROUTING_DIR / "G_2nd.graphml"

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
# ACCESSIBILITY ROUTING
# ===============================================================
def compute_nearest_services(G_ig, muni_nodes, service_nodes_by_class, node_to_index, index_to_node):
    """
    Computes the shortest path from each municipality to the nearest service of each class.
    """
    results = {}

    for service_class, s_nodes in service_nodes_by_class.items():
        class_results = {}
        # Only target service nodes that still exist in the current scenario graph
        valid_targets = [node_to_index[n] for n in s_nodes if n in node_to_index]
        
        if not valid_targets:
            logging.warning(f"No reachable '{service_class}' nodes in this graph.")
            continue

        for muni_name, m_node in muni_nodes.items():
            if m_node not in node_to_index:
                # The municipality node itself was destroyed/flooded
                class_results[muni_name] = {"time": None, "path": [], "destination_node": None}
                continue

            src_idx = node_to_index[m_node]
            
            # Calculate distance to ALL services of this class
            # 
            distances = G_ig.distances(source=src_idx, target=valid_targets, weights="weight")[0]
            
            # Find the minimum distance (the nearest facility)
            min_dist = min(distances)
            
            if math.isinf(min_dist):
                class_results[muni_name] = {"time": None, "path": [], "destination_node": None}
            else:
                nearest_idx = distances.index(min_dist)
                target_idx = valid_targets[nearest_idx]
                
                # Retrieve the actual route
                route = G_ig.get_shortest_paths(src_idx, to=target_idx, weights="weight", output="vpath")[0]
                
                class_results[muni_name] = {
                    "time": min_dist,
                    "path": [index_to_node[i] for i in route],
                    "destination_node": index_to_node[target_idx]
                }
                
        results[service_class] = class_results

    return results

def process_accessibility(scenario_name, G_nx, muni_nodes, service_nodes_by_class):
    """Handles igraph conversion, routing, and saving for a single scenario."""
    
    if scenario_name == "NP":
        out_path = ROUTING_DIR / "service_access_NP.json"
    else:
        # Ensure the scenario directory exists before saving
        scenario_dir = SCENARIO_BASE / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        out_path = scenario_dir / f"service_access_{scenario_name}.json"
        
    if out_path.exists():
        logging.info(f"Accessibility already computed for {scenario_name}. Skipping.")
        return

    logging.info(f"Computing service access for {scenario_name}...")
    G_ig, node_to_index, index_to_node = convert_nx_to_igraph(G_nx)
    
    results = compute_nearest_services(G_ig, muni_nodes, service_nodes_by_class, node_to_index, index_to_node)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ===============================================================
# MAIN
# ===============================================================
def main():
    logging.info(f"Loading baseline graph: {GRAPH_NORMAL_PATH.name}")
    G_2nd = ox.load_graphml(GRAPH_NORMAL_PATH)

    # 1. Extract Municipal Origins
    muni_nodes = {d.get("municipality"): n for n, d in G_2nd.nodes(data=True) if pd.notna(d.get("municipality"))}
    logging.info(f"Identified {len(muni_nodes)} municipal origins.")

    # 2. Extract Service Destinations (Grouped by Class)
    service_nodes_by_class = {}
    for n, d in G_2nd.nodes(data=True):
        clase = d.get("clase")
        if pd.notna(clase):
            if clase not in service_nodes_by_class:
                service_nodes_by_class[clase] = []
            service_nodes_by_class[clase].append(n)
            
    for c, nodes in service_nodes_by_class.items():
        logging.info(f"Identified {len(nodes)} destinations for service class: {c}")

    # 3. Compute Baseline (NP)
    process_accessibility("NP", G_2nd, muni_nodes, service_nodes_by_class)

    # 4. Compute Scenarios
    for name in scenarios:
        scenario_folder = SCENARIO_BASE / name
        safe_graph_path = scenario_folder / f"safe_roads_{name}.graphml"
        
        if not safe_graph_path.exists():
            logging.warning(f"Missing {safe_graph_path.name}. Skipping scenario {name}.")
            continue
            
        G_safe = ox.load_graphml(safe_graph_path)
        process_accessibility(name, G_safe, muni_nodes, service_nodes_by_class)

    logging.info("\nService accessibility computations finished.")

if __name__ == "__main__":
    main()