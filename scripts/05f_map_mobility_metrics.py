import pandas as pd 
import geopandas as gpd 
import matplotlib.pyplot as plt 
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
import numpy as np 
from pathlib import Path 
import json 
import re 
import warnings 
import contextily as ctx
from matplotlib.colors import LogNorm, TwoSlopeNorm, Normalize
from shapely.geometry import LineString 

warnings.filterwarnings("ignore") 

# --- 0. HELPER FUNCTIONS --- 
def extract_date(filename): 
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(filename)) 
    return pd.to_datetime(match.group(1)) if match else None

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    if w.sum() > 0:
        return (d * w).sum() / w.sum()
    else:
        return 0.0

# --- 1. SETUP --- 
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

MANUAL_DIR = DATA_DIR / "01_raw_manual"
DOWNLOADED_DIR = DATA_DIR / "02_raw_downloaded"
SHARED_DIR = DATA_DIR / "03_processed_shared"

INPUT_FOLDER = DOWNLOADED_DIR / "mobility" / "mobility_daily"
GEOJSON_PATH = DOWNLOADED_DIR / "mobility" / "spatial" / "spatial_zones_municipalities.geojson"
RELATIONS_PATH = DOWNLOADED_DIR / "mobility" / "metadata" / "zone_relations_municipalities.csv"
POP_CSV_PATH = DOWNLOADED_DIR / "demographics" / "ine_spain_population_2024.csv"
MAPPING_PATH = SHARED_DIR / "final_mitma_mapping.json"

MAPS_FOLDER = DATA_DIR / "06_outputs" / "maps_M"
MAPS_FOLDER.mkdir(exist_ok=True, parents=True) 

EVENT_DATE = pd.to_datetime("2024-10-29") 

# --- 2. LOAD & AGGREGATE CORE DATA --- 
print("--- 1. LOADING OFFICIAL MITMA GEOMETRIES & POPULATION ---") 

if not MAPPING_PATH.exists():
    print(f"[ERROR] final_mitma_mapping.json not found at {MAPPING_PATH}. Run validation script first.")
    exit()

with open(MAPPING_PATH, "r", encoding="utf-8") as f:
    name_to_id_map = json.load(f)

id_to_name_map = {}
for name, mitma_id in name_to_id_map.items():
    if mitma_id in id_to_name_map:
        id_to_name_map[mitma_id] += f" / {name}"
    else:
        id_to_name_map[mitma_id] = name

target_ids = set(name_to_id_map.values())

# Population Aggregation
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

# Reproject to Web Mercator for Contextily Basemaps
gdf_target = gdf_target.to_crs(epsg=3857)

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
    edgecolor='black', linewidth=0.5, alpha=0.85,
    legend_kwds={'label': "Resident Population 2024 (Log Scale)", 'shrink': 0.4} 
) 

ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, alpha=0.6)

placed_coords = [] 
for idx, row in gdf_target.sort_values('population', ascending=False).iterrows(): 
    point = row.geometry.representative_point() 
    x, y = point.x, point.y 
    too_close = any(np.sqrt((x-px)**2 + (y-py)**2) < 3000 for px, py in placed_coords) 
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
            chunk = df[mask][['origin', 'destination', 'n_trips', 'trips_total_length_km']].copy()
            chunk['date'] = d 
            chunk['avg_trip_distance'] = (chunk['trips_total_length_km'] / chunk['n_trips']).replace([np.inf, -np.inf], 0).fillna(0)
            chunk = chunk[chunk['avg_trip_distance'] <= 500]
            all_data.append(chunk) 
    except Exception as e: 
        pass 

full_df = pd.concat(all_data, ignore_index=True) 

# --- 5. CALCULATE REACH ---
reach_df = full_df.dropna(subset=['avg_trip_distance', 'n_trips'])

def compute_reach_for_subset(df_subset, suffix=""):
    r_inc = df_subset[df_subset['destination'].isin(target_ids)].groupby(['date', 'destination']).apply(lambda x: wavg(x, 'avg_trip_distance', 'n_trips')).reset_index(name=f'reach_incoming{suffix}')
    r_inc.rename(columns={'destination': 'zone_id'}, inplace=True)

    r_out = df_subset[df_subset['origin'].isin(target_ids)].groupby(['date', 'origin']).apply(lambda x: wavg(x, 'avg_trip_distance', 'n_trips')).reset_index(name=f'reach_outgoing{suffix}')
    r_out.rename(columns={'origin': 'zone_id'}, inplace=True)
    return r_inc, r_out

reach_inc, reach_out = compute_reach_for_subset(reach_df, "")
reach_inc_ext, reach_out_ext = compute_reach_for_subset(reach_df[reach_df['origin'] != reach_df['destination']], "_ext")

# --- 6. UNDIRECTED NETWORK PREP ---
nodes_sorted = np.sort(full_df[['origin', 'destination']].values, axis=1)
full_df['u_node1'] = nodes_sorted[:, 0]
full_df['u_node2'] = nodes_sorted[:, 1]

undirected_flows = full_df.groupby(['date', 'u_node1', 'u_node2'])['n_trips'].sum().reset_index()
pre_event_edges = undirected_flows[undirected_flows['date'] < EVENT_DATE].copy()
pre_event_edges['dow'] = pre_event_edges['date'].dt.dayofweek
edge_baseline = pre_event_edges.groupby(['dow', 'u_node1', 'u_node2'])['n_trips'].mean().reset_index(name='base_trips')

# --- 7. METRICS CONSOLIDATION ---
daily_inc = full_df[full_df['destination'].isin(target_ids)].groupby(['date', 'destination'])['n_trips'].sum().reset_index() 
daily_inc.rename(columns={'destination': 'zone_id', 'n_trips': 'abs_incoming'}, inplace=True) 

daily_out = full_df[full_df['origin'].isin(target_ids)].groupby(['date', 'origin'])['n_trips'].sum().reset_index() 
daily_out.rename(columns={'origin': 'zone_id', 'n_trips': 'abs_outgoing'}, inplace=True) 

metrics = pd.merge(daily_inc, daily_out, on=['date', 'zone_id'], how='outer').fillna(0) 
metrics = pd.merge(metrics, reach_inc, on=['date', 'zone_id'], how='left').fillna(0)
metrics = pd.merge(metrics, reach_out, on=['date', 'zone_id'], how='left').fillna(0)
metrics = pd.merge(metrics, reach_inc_ext, on=['date', 'zone_id'], how='left').fillna(0)
metrics = pd.merge(metrics, reach_out_ext, on=['date', 'zone_id'], how='left').fillna(0)

# Baseline Logic
metrics['dow'] = metrics['date'].dt.dayofweek
pre_event = metrics[metrics['date'] < EVENT_DATE] 

if not pre_event.empty: 
    baseline = pre_event.groupby(['dow', 'zone_id'])[['abs_incoming', 'abs_outgoing']].mean().reset_index() 
    baseline.columns = ['dow', 'zone_id', 'base_inc', 'base_out'] 
    metrics = pd.merge(metrics, baseline, on=['dow', 'zone_id'], how='left').fillna(0) 
    metrics['rel_incoming'] = ((metrics['abs_incoming'] - metrics['base_inc']) / (metrics['base_inc'] + 1)) * 100 
    metrics['rel_outgoing'] = ((metrics['abs_outgoing'] - metrics['base_out']) / (metrics['base_out'] + 1)) * 100 
else: 
    metrics['rel_incoming'] = 0.0 
    metrics['rel_outgoing'] = 0.0

metrics = metrics.sort_values(['zone_id', 'date']) 
metrics['dod_incoming'] = metrics.groupby('zone_id')['abs_incoming'].pct_change().fillna(0) * 100 
metrics['dod_outgoing'] = metrics.groupby('zone_id')['abs_outgoing'].pct_change().fillna(0) * 100 
metrics['asymmetry'] = ((metrics['abs_incoming'] - metrics['abs_outgoing']) / (metrics['abs_incoming'] + metrics['abs_outgoing'] + 1)) * 100 

metrics['official_name'] = metrics['zone_id'].map(id_to_name_map) 
metrics['population'] = metrics['zone_id'].map(population_map) 

# Save JSON
json_out = metrics.copy() 
json_out['date'] = json_out['date'].dt.strftime('%Y-%m-%d') 
json_out.to_json(SHARED_DIR / "final_mobility_metrics.json", orient='records', indent=4) 

# --- 8. CUMULATIVE IMPACT ANALYSIS MAPS ---
print("--- 4. GENERATING CUMULATIVE ISOLATION & IMPACT MAPS ---")
isolation_days = metrics[metrics['rel_incoming'] < -50].groupby('zone_id').size().reset_index(name='isolation_days')
gdf_isolation = gdf_target.join(isolation_days.set_index('zone_id'), how='left').fillna(0)

# Calculate Impact Metrics
gdf_isolation['impact_linear'] = gdf_isolation['isolation_days'] * gdf_isolation['population']
gdf_isolation['impact_log'] = gdf_isolation['isolation_days'] * np.log10(gdf_isolation['population'] + 1)

def plot_cumulative_map(gdf, column, title, cmap, norm, save_name, highlight_critical=True):
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.plot(column=column, ax=ax, cmap=cmap, norm=norm, alpha=0.85, 
             edgecolor='black', linewidth=0.4, legend=True, 
             legend_kwds={'shrink': 0.7, 'label': title})
    
    legend_elements = []
    if highlight_critical:
        # Define severe prolonged isolation as >= 5 days
        critical = gdf[gdf["isolation_days"] >= 4]
        if not critical.empty:
            critical.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2.5)
            legend_elements.append(Patch(facecolor='none', edgecolor='red', linewidth=2.5, label='Prolonged Isolation (>= 5 Days)'))
    
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, alpha=0.6)
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9, fontsize=11, facecolor='white')
        
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(MAPS_FOLDER / f"{save_name}.png", dpi=200, bbox_inches='tight')
    plt.close()

plot_cumulative_map(
    gdf_isolation, 'isolation_days', 
    "Total Isolation Duration (Days with < -50% incoming flow)", 
    "YlOrRd", Normalize(vmin=0, vmax=gdf_isolation['isolation_days'].max()), 
    "00_cumulative_isolation_days"
)

plot_cumulative_map(
    gdf_isolation, 'impact_linear', 
    "Linear Humanitarian Impact (Isolation Days x Population)", 
    "YlOrRd", Normalize(vmin=0, vmax=np.percentile(gdf_isolation['impact_linear'], 95)), 
    "00_cumulative_impact_linear"
)

plot_cumulative_map(
    gdf_isolation, 'impact_log', 
    "Logarithmic Humanitarian Impact (Isolation Days x Log10(Pop))", 
    "YlOrRd", Normalize(vmin=0, vmax=gdf_isolation['impact_log'].max()), 
    "00_cumulative_impact_log"
)

# --- 9. FINAL PLOTTING LOOP --- 
print("--- 5. GENERATING DAILY MAPS ---") 
plot_configs = [ 
    ('abs_incoming', 'Absolute Incoming', 'viridis', (0, None)), 
    ('abs_outgoing', 'Absolute Outgoing', 'viridis', (0, None)), 
    ('rel_incoming', 'Rel. Incoming (%)', 'RdBu', (-100, 100)), 
    ('rel_outgoing', 'Rel. Outgoing (%)', 'RdBu', (-100, 100)), 
    ('dod_incoming', 'DoD Change In (%)', 'RdBu', (-50, 50)), 
    ('dod_outgoing', 'DoD Change Out (%)', 'RdBu', (-50, 50)), 
    ('asymmetry', 'Asymmetry (Blue: Sink/Incoming | Red: Source/Outgoing)', 'RdBu', (-50, 50)),
    ('reach_incoming', 'Avg Incoming Reach Distance (km)', 'viridis', (0, None)), 
    ('reach_incoming_ext', 'External Incoming Reach (km)', 'plasma', (0, None))
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
        if vmin is None: 
            vmin = merged_map[col].min() 
        if vmax is None: 
            vmax = merged_map[col].quantile(0.95) 
        
        if cmap == 'RdBu':
            norm = TwoSlopeNorm(vmin=vmin if vmin < 0 else -1, vcenter=0, vmax=vmax if vmax > 0 else 1)
            merged_map.plot(column=col, ax=ax, cmap=cmap, norm=norm, legend=True, 
                            edgecolor='black', linewidth=0.4, alpha=0.8, missing_kwds={'color': 'lightgrey'}, 
                            legend_kwds={'shrink': 0.7, 'label': title})
        else:
            merged_map.plot(column=col, ax=ax, cmap=cmap, legend=True, 
                            vmin=vmin, vmax=vmax, edgecolor='black', linewidth=0.4, alpha=0.8,
                            missing_kwds={'color': 'lightgrey'}, legend_kwds={'shrink': 0.7, 'label': title}) 
            
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.PositronNoLabels, alpha=0.6)
        ax.set_title(f"{title}\n{day_str}", fontsize=15) 
        ax.set_axis_off() 
        plt.tight_layout() 
        fig.savefig(MAPS_FOLDER / f"{day_str}_{col.replace(' ', '_')}.png", dpi=150) 
        plt.close(fig) 
        
    # --- NETWORK DISRUPTION MAP ---
    print(f"  > Saved maps for {day_str}, generating network edges...")
    
    day_edges = undirected_flows[undirected_flows['date'] == date].copy()
    
    if not day_edges.empty:
        dow = pd.to_datetime(date).dayofweek
        day_edges['dow'] = dow
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
            edges_gdf = gpd.GeoDataFrame(valid_df, geometry=lines, crs=gdf_target.crs)
            
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf_target.plot(ax=ax, color='lightgrey', edgecolor='white', linewidth=0.3, alpha=0.5)
            
            edges_gdf['linewidth'] = np.log1p(edges_gdf['base_trips']) * 0.5
            
            norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
            edges_gdf.plot(ax=ax, column='rel_change', cmap='RdBu', norm=norm, linewidth=edges_gdf['linewidth'], alpha=0.9, 
                           legend=True, legend_kwds={'shrink': 0.7, 'label': 'Relative Change vs Baseline (%)'})
            
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, alpha=0.7)
            ax.set_title(f"Network Flow Disruption\n{day_str}", fontsize=15)
            ax.set_axis_off()
            plt.tight_layout()
            fig.savefig(MAPS_FOLDER / f"{day_str}_network_deviation.png", dpi=150)
            plt.close(fig)

# --- 10. TEXT REPORT EXPORT ---
print("\n--- 6. EXPORTING TEXT REPORT ---")
report_path = MAPS_FOLDER / "05f_mobility_network_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("=== REGIONAL MOBILITY NETWORK & IMPACT REPORT ===\n\n")
    
    # Extract maximum single-day drop for each municipality for extra context
    max_drop = metrics[metrics['date'] >= EVENT_DATE].groupby('zone_id')['rel_incoming'].min().reset_index(name='max_drop_pct')
    gdf_report = gdf_isolation.reset_index().merge(max_drop, on='zone_id', how='left')
    
    f.write("1. CUMULATIVE HUMANITARIAN IMPACT SUMMARY\n")
    f.write("Sorted by Logarithmic Impact Score (Descending)\n")
    f.write("-" * 115 + "\n")
    header = f"{'Municipality':<30} | {'Population':<10} | {'Iso. Days':<10} | {'Max Drop %':<12} | {'Linear Impact':<15} | {'Log Impact':<12}"
    f.write(header + "\n" + "-" * 115 + "\n")
    
    for _, row in gdf_report.sort_values('impact_log', ascending=False).iterrows():
        name = str(row['official_name']).split('/')[0].strip()
        pop = int(row['population'])
        days = int(row['isolation_days'])
        m_drop = f"{row['max_drop_pct']:.1f}%" if pd.notna(row['max_drop_pct']) else "N/A"
        lin = f"{row['impact_linear']:,.0f}"
        log_i = f"{row['impact_log']:.2f}"
        
        f.write(f"{name:<30} | {pop:<10,} | {days:<10} | {m_drop:<12} | {lin:<15} | {log_i:<12}\n")
        
    f.write("\n\n2. DAILY NETWORK DEFICIT AND DETOUR CORRIDORS\n")
    f.write("=" * 115 + "\n\n")

    for date in sorted(metrics['date'].unique()):
        if date < EVENT_DATE: continue # Only log post-event network stats
        
        day_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        day_metrics = metrics[metrics['date'] == date]
        
        avg_reach_ext = day_metrics['reach_incoming_ext'].mean()
        
        f.write(f"DATE: {day_str}\n")
        f.write(f"Regional Average Reach (External Trips): {avg_reach_ext:.2f} km\n")
        f.write("-" * 80 + "\n")
        
        day_edges = undirected_flows[undirected_flows['date'] == date].copy()
        if not day_edges.empty:
            dow = pd.to_datetime(date).dayofweek
            day_edges['dow'] = dow
            day_edges = pd.merge(day_edges, edge_baseline, on=['dow', 'u_node1', 'u_node2'], how='left').fillna({'base_trips': 0})
            
            day_edges['abs_diff'] = day_edges['n_trips'] - day_edges['base_trips']
            
            def get_edge_name(r):
                n1 = id_to_name_map.get(r['u_node1'], r['u_node1']).split('/')[0].strip()
                n2 = id_to_name_map.get(r['u_node2'], r['u_node2']).split('/')[0].strip()
                return f"{n1} <-> {n2}"
            
            day_edges['route_name'] = day_edges.apply(get_edge_name, axis=1)
            
            severed = day_edges.sort_values('abs_diff', ascending=True).head(10)
            resilient = day_edges.sort_values('abs_diff', ascending=False).head(10)
            
            f.write("Top 10 Severed Arteries (Largest Absolute Drop):\n")
            for _, r in severed.iterrows():
                f.write(f"  {r['route_name']:<40} | Drop: {r['abs_diff']:>8.0f} trips\n")
                
            f.write("\nTop 10 Resilient/Detour Arteries (Largest Absolute Increase):\n")
            for _, r in resilient.iterrows():
                f.write(f"  {r['route_name']:<40} | Surge: +{r['abs_diff']:>7.0f} trips\n")
                
        f.write("=" * 80 + "\n\n")

print(f"Text report saved to: {report_path.name}")
print("\nWorkflow Complete.")