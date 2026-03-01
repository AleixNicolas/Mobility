import pandas as pd 
import geopandas as gpd 
import matplotlib.pyplot as plt 
import matplotlib.patheffects as pe
import numpy as np 
from pathlib import Path 
import json 
import re 
import warnings 
from matplotlib.colors import LogNorm, TwoSlopeNorm
from shapely.geometry import LineString 

warnings.filterwarnings("ignore") 

# --- 0. HELPER FUNCTIONS --- 
def extract_date(filename): 
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(filename)) 
    return pd.to_datetime(match.group(1)) if match else None

# --- 1. SETUP (ALIGNED TO NEW ARCHITECTURE) --- 
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

# Phase-based Inputs
MANUAL_DIR = DATA_DIR / "01_raw_manual"
DOWNLOADED_DIR = DATA_DIR / "02_raw_downloaded"
SHARED_DIR = DATA_DIR / "03_processed_shared"

# Mobility specific inputs
INPUT_FOLDER = DOWNLOADED_DIR / "mobility" / "mobility_daily"
GEOJSON_PATH = DOWNLOADED_DIR / "mobility" / "spatial" / "spatial_zones_municipalities.geojson"
RELATIONS_PATH = DOWNLOADED_DIR / "mobility" / "metadata" / "zone_relations_municipalities.csv"
POP_CSV_PATH = DOWNLOADED_DIR / "demographics" / "ine_spain_population_2024.csv"
MAPPING_PATH = SHARED_DIR / "final_mitma_mapping.json"

# Outputs
MAPS_FOLDER = DATA_DIR / "06_outputs" / "maps_M"
MAPS_FOLDER.mkdir(exist_ok=True, parents=True) 

EVENT_DATE = pd.to_datetime("2024-10-29") 

# --- 2. LOAD & AGGREGATE CORE DATA --- 
print("--- 1. LOADING OFFICIAL MITMA GEOMETRIES & POPULATION ---") 

if not MAPPING_PATH.exists():
    print(f"[ERROR] final_mitma_mapping.json not found at {MAPPING_PATH}. Run validation script first.")
    exit()

# Load mapping and create target set
with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    name_to_id_map = json.load(f)

# Reverse map for labels (handling many-to-one _AM clusters)
id_to_name_map = {}
for name, mitma_id in name_to_id_map.items():
    if mitma_id in id_to_name_map:
        id_to_name_map[mitma_id] += f" / {name}"
    else:
        id_to_name_map[mitma_id] = name

target_ids = set(name_to_id_map.values())

# Aggregate INE Population to MITMA Zones
pop_df = pd.read_csv(POP_CSV_PATH, dtype=str)
pop_df['code'] = pop_df['code'].astype(str).str.zfill(5)
clean_pop = pop_df['population_2024'].astype(str).str.replace('"', '', regex=False).str.replace('.', '', regex=False)
pop_df['pop_2024_num'] = pd.to_numeric(clean_pop, errors='coerce').fillna(0.0)

df_rel = pd.read_csv(RELATIONS_PATH, dtype=str)
ine_col = 'municipalities' if 'municipalities' in df_rel.columns else df_rel.columns[2]
mitma_col = 'municipalities_mitma' if 'municipalities_mitma' in df_rel.columns else df_rel.columns[4]

pop_merged = pd.merge(pop_df, df_rel[[ine_col, mitma_col]], left_on='code', right_on=ine_col, how='inner')
mitma_population = pop_merged.groupby(mitma_col)['pop_2024_num'].sum().reset_index()
mitma_population.rename(columns={mitma_col: 'zone_id', 'pop_2024_num': 'population'}, inplace=True)
population_map = mitma_population.set_index('zone_id')['population'].to_dict()

# Load GeoJSON and filter to target zones
gdf_all = gpd.read_file(GEOJSON_PATH)
gdf_all['zone_id'] = gdf_all['id'].astype(str).str.strip()
gdf_target = gdf_all[gdf_all['zone_id'].isin(target_ids)].copy().set_index('zone_id')

gdf_target['population'] = gdf_target.index.map(population_map).fillna(0)
gdf_target['official_name'] = gdf_target.index.map(id_to_name_map).fillna("Unknown")

print(f" > Assembled {len(gdf_target)} perfect MITMA polygons.")

# --- 3. REFERENCE POPULATION MAP --- 
print("--- 2. GENERATING REFERENCE POPULATION MAP ---") 
fig, ax = plt.subplots(figsize=(15, 15)) 
plot_pop = gdf_target['population'].apply(lambda x: x if x > 0 else 1) 
gdf_target.assign(p_plot=plot_pop).plot( 
    column='p_plot', ax=ax, cmap='viridis', legend=True, 
    norm=LogNorm(vmin=plot_pop.min(), vmax=plot_pop.max()), 
    edgecolor='black', linewidth=0.5, 
    legend_kwds={'label': "Resident Population 2024 (Log Scale)", 'shrink': 0.4} 
) 

placed_coords = [] 
for idx, row in gdf_target.sort_values('population', ascending=False).iterrows(): 
    point = row.geometry.representative_point() 
    x, y = point.x, point.y 
    too_close = any(np.sqrt((x-px)**2 + (y-py)**2) < 0.022 for px, py in placed_coords) 
    if not too_close or row['population'] > 20000: 
        short_name = str(row['official_name']).split('/')[0].strip().title()
        ax.text(x, y, short_name, fontsize=7, fontweight='bold', ha='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=0.2)) 
        placed_coords.append((x, y)) 

ax.set_axis_off() 
plt.savefig(MAPS_FOLDER / "00_reference_population.png", dpi=300, bbox_inches='tight') 
plt.close() 

# --- 4. PROCESSING MOBILITY DATA --- 
print("--- 3. PROCESSING MOBILITY DATA ---") 
all_data = [] 
files = sorted([f for f in INPUT_FOLDER.glob("*.csv") if extract_date(f.name)]) 

for f in files: 
    d = extract_date(f.name) 
    try: 
        df = pd.read_csv(f, dtype={'id_origin': str, 'id_destination': str}) 
        df['origin'] = df['id_origin'].str.strip()
        df['destination'] = df['id_destination'].str.strip()
        
        mask = df['origin'].isin(target_ids) | df['destination'].isin(target_ids) 
        if mask.any(): 
            cols_to_keep = ['origin', 'destination', 'n_trips', 'trips_total_length_km']
            chunk = df[mask][cols_to_keep].copy()
            chunk['date'] = d 
            chunk['avg_trip_distance'] = (chunk['trips_total_length_km'] / chunk['n_trips']).replace([np.inf, -np.inf], 0).fillna(0)
            all_data.append(chunk) 
    except Exception as e: 
        pass 

full_df = pd.concat(all_data, ignore_index=True) 

# --- RADIUS OF GYRATION (REACH) ---
reach_df = full_df.dropna(subset=['avg_trip_distance', 'n_trips'])

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    return (d * w).sum() / w.sum() if w.sum() > 0 else 0

reach_inc = reach_df[reach_df['destination'].isin(target_ids)].groupby(['date', 'destination']).apply(lambda x: wavg(x, 'avg_trip_distance', 'n_trips')).reset_index(name='reach_incoming')
reach_inc.rename(columns={'destination': 'zone_id'}, inplace=True)

reach_out = reach_df[reach_df['origin'].isin(target_ids)].groupby(['date', 'origin']).apply(lambda x: wavg(x, 'avg_trip_distance', 'n_trips')).reset_index(name='reach_outgoing')
reach_out.rename(columns={'origin': 'zone_id'}, inplace=True)

# --- UNDIRECTED NETWORK PREP ---
nodes_sorted = np.sort(full_df[['origin', 'destination']].values, axis=1)
full_df['u_node1'] = nodes_sorted[:, 0]
full_df['u_node2'] = nodes_sorted[:, 1]

undirected_flows = full_df.groupby(['date', 'u_node1', 'u_node2'])['n_trips'].sum().reset_index()

pre_event_edges = undirected_flows[undirected_flows['date'] < EVENT_DATE].copy()
pre_event_edges['dow'] = pre_event_edges['date'].dt.dayofweek
edge_baseline = pre_event_edges.groupby(['dow', 'u_node1', 'u_node2'])['n_trips'].mean().reset_index(name='base_trips')

# General Metrics
daily_inc = full_df[full_df['destination'].isin(target_ids)].groupby(['date', 'destination'])['n_trips'].sum().reset_index() 
daily_inc.rename(columns={'destination': 'zone_id', 'n_trips': 'abs_incoming'}, inplace=True) 
daily_out = full_df[full_df['origin'].isin(target_ids)].groupby(['date', 'origin'])['n_trips'].sum().reset_index() 
daily_out.rename(columns={'origin': 'zone_id', 'n_trips': 'abs_outgoing'}, inplace=True) 

metrics = pd.merge(daily_inc, daily_out, on=['date', 'zone_id'], how='outer').fillna(0) 
metrics = pd.merge(metrics, reach_inc, on=['date', 'zone_id'], how='left').fillna(0)
metrics = pd.merge(metrics, reach_out, on=['date', 'zone_id'], how='left').fillna(0)

# Relative and DoD Metrics 
pre_event = metrics[metrics['date'] < EVENT_DATE] 
if not pre_event.empty: 
    baseline = pre_event.groupby([pre_event['date'].dt.dayofweek, 'zone_id'])[['abs_incoming', 'abs_outgoing']].mean().reset_index() 
    baseline.columns = ['dow', 'zone_id', 'base_inc', 'base_out'] 
    metrics['dow'] = metrics['date'].dt.dayofweek 
    metrics = pd.merge(metrics, baseline, on=['dow', 'zone_id'], how='left').fillna(0) 
    metrics['rel_incoming'] = ((metrics['abs_incoming'] - metrics['base_inc']) / (metrics['base_inc'] + 1)) * 100 
    metrics['rel_outgoing'] = ((metrics['abs_outgoing'] - metrics['base_out']) / (metrics['base_out'] + 1)) * 100 
else: 
    metrics['rel_incoming'] = metrics['rel_outgoing'] = 0 

metrics = metrics.sort_values(['zone_id', 'date']) 
metrics['dod_incoming'] = metrics.groupby('zone_id')['abs_incoming'].pct_change().fillna(0) * 100 
metrics['dod_outgoing'] = metrics.groupby('zone_id')['abs_outgoing'].pct_change().fillna(0) * 100 
metrics['asymmetry'] = ((metrics['abs_incoming'] - metrics['abs_outgoing']) / (metrics['abs_incoming'] + metrics['abs_outgoing'] + 1)) * 100 

metrics['official_name'] = metrics['zone_id'].map(id_to_name_map) 
metrics['population'] = metrics['zone_id'].map(population_map) 

# ISOLATION DURATION
isolation_df = metrics[metrics['rel_incoming'] < -50].groupby('zone_id').size().reset_index(name='isolation_days')
gdf_isolation = gdf_target.join(isolation_df.set_index('zone_id'), how='left').fillna(0)

print("--- 4. GENERATING ISOLATION DURATION MAP ---")
fig, ax = plt.subplots(figsize=(10, 10))
gdf_isolation.plot(column='isolation_days', ax=ax, cmap='OrRd', legend=True,
                   edgecolor='black', linewidth=0.4,
                   legend_kwds={'shrink': 0.7, 'label': 'Days of Isolation (< -50% incoming flow)'})
ax.set_title("Total Isolation Duration", fontsize=15)
ax.set_axis_off()
plt.tight_layout()
fig.savefig(MAPS_FOLDER / "00_isolation_duration.png", dpi=150)
plt.close(fig)

# Save JSON to shared processed dir
json_out = metrics.copy() 
json_out['date'] = json_out['date'].dt.strftime('%Y-%m-%d') 
json_out.to_json(SHARED_DIR / "final_mobility_metrics.json", orient='records', indent=4) 

# --- 5. FINAL PLOTTING LOOP --- 
print("--- 5. GENERATING DAILY MAPS ---") 
plot_configs = [ 
    ('abs_incoming', 'Absolute Incoming', 'viridis', (0, None)), 
    ('abs_outgoing', 'Absolute Outgoing', 'viridis', (0, None)), 
    ('rel_incoming', 'Rel. Incoming (%)', 'RdBu', (-100, 100)), 
    ('rel_outgoing', 'Rel. Outgoing (%)', 'RdBu', (-100, 100)), 
    ('dod_incoming', 'DoD Change In (%)', 'RdBu', (-50, 50)), 
    ('dod_outgoing', 'DoD Change Out (%)', 'RdBu', (-50, 50)), 
    ('asymmetry', 'Asymmetry (%)', 'RdBu', (-50, 50)),
    ('reach_incoming', 'Avg Incoming Reach Distance (km)', 'viridis', (0, None)), 
    ('reach_outgoing', 'Avg Outgoing Reach Distance (km)', 'viridis', (0, None))  
] 

cx_deg = gdf_target.geometry.centroid.x.to_dict()
cy_deg = gdf_target.geometry.centroid.y.to_dict()

for date in sorted(metrics['date'].unique()): 
    day_str = pd.to_datetime(date).strftime('%Y-%m-%d') 
    day_data = metrics[metrics['date'] == date].set_index('zone_id') 
    merged_map = gdf_target.join(day_data, how='left', rsuffix='_metric').fillna(0) 
    
    for col, title, cmap, vlims in plot_configs: 
        fig, ax = plt.subplots(figsize=(10, 10)) 
        vmin, vmax = vlims 
        if vmin is None: vmin = merged_map[col].min() 
        if vmax is None: vmax = merged_map[col].quantile(0.95) 
        
        if cmap == 'RdBu':
            norm = TwoSlopeNorm(vmin=vmin if vmin < 0 else -1, vcenter=0, vmax=vmax if vmax > 0 else 1)
            merged_map.plot(column=col, ax=ax, cmap=cmap, norm=norm, legend=True, 
                            edgecolor='black', linewidth=0.4, missing_kwds={'color': 'lightgrey'}, 
                            legend_kwds={'shrink': 0.7, 'label': title})
        else:
            merged_map.plot(column=col, ax=ax, cmap=cmap, legend=True, 
                            vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.4, 
                            missing_kwds={'color': 'lightgrey'}, legend_kwds={'shrink': 0.7, 'label': title}) 
            
        ax.set_title(f"{title}\n{day_str}", fontsize=15) 
        ax.set_axis_off() 
        plt.tight_layout() 
        fig.savefig(MAPS_FOLDER / f"{day_str}_{col}.png", dpi=150) 
        plt.close(fig) 
        
    # --- NETWORK DISRUPTION MAP ---
    print(f"  > Saved maps for {day_str}, generating network edges...")
    
    day_edges = undirected_flows[undirected_flows['date'] == date].copy()
    day_edges['dow'] = pd.to_datetime(date).dayofweek
    day_edges = pd.merge(day_edges, edge_baseline, on=['dow', 'u_node1', 'u_node2'], how='left').fillna({'base_trips': 0})
    
    day_edges['rel_change'] = ((day_edges['n_trips'] - day_edges['base_trips']) / (day_edges['base_trips'] + 1)) * 100
    day_edges = day_edges[(day_edges['base_trips'] > 15) | (day_edges['n_trips'] > 15)]
    
    lines = []
    valid_edges = []
    for idx, row in day_edges.iterrows():
        n1, n2 = row['u_node1'], row['u_node2']
        if n1 in cx_deg and n2 in cx_deg:
            lines.append(LineString([(cx_deg[n1], cy_deg[n1]), (cx_deg[n2], cy_deg[n2])]))
            valid_edges.append(row)

    if lines:
        valid_df = pd.DataFrame(valid_edges)
        edges_gdf = gpd.GeoDataFrame(valid_df, geometry=lines, crs="EPSG:4326")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf_target.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.3)
        
        edges_gdf['linewidth'] = np.log1p(edges_gdf['base_trips']) * 0.5
        
        norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
        edges_gdf.plot(ax=ax, column='rel_change', cmap='RdBu', norm=norm, linewidth=edges_gdf['linewidth'], alpha=0.8, 
                       legend=True, legend_kwds={'shrink': 0.7, 'label': 'Relative Change vs Baseline (%)'})
        
        ax.set_title(f"Network Flow Disruption\n{day_str}", fontsize=15)
        ax.set_axis_off()
        plt.tight_layout()
        fig.savefig(MAPS_FOLDER / f"{day_str}_network_deviation.png", dpi=150)
        plt.close(fig)

print("\nWorkflow Complete.")