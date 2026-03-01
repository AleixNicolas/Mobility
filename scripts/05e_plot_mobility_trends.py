import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import numpy as np
import seaborn as sns
from pathlib import Path
import json
import re
import matplotlib.dates as mdates

# --- 0. Helper Functions ---
def extract_date_from_filename(filename):
    match = re.search(r'(\d{4}-\d{2}-\d{2})', str(filename))
    if match:
        try: return pd.to_datetime(match.group(1))
        except: return None
    return None

# --- 1. Setup & Directory Management ---
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

# Input directories
MOBILITY_DIR = DATA_ROOT / "02_raw_downloaded" / "mobility" / "mobility_daily"
SHARED_DIR = DATA_ROOT / "03_processed_shared"

# Output directories
OUTPUT_ROOT = DATA_ROOT / "06_outputs"
NETWORKS_FOLDER = OUTPUT_ROOT / "networks_M" # Retained if your script generates network topology outputs
PLOTS_FOLDER = OUTPUT_ROOT / "plots_M"

for folder in [NETWORKS_FOLDER, PLOTS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# LOAD THE NEW VALIDATED MAPPING
MAPPING_JSON_PATH = SHARED_DIR / "final_mitma_mapping.json"

EVENT_DATE = pd.to_datetime("2024-10-29 00:00:00") 
RDI_START_DATE = pd.to_datetime("2024-10-28 00:00:00")

# --- 2. ID Resolution ---
print("--- 1. RESOLVING TARGET MUNICIPALITIES ---")
if not MAPPING_JSON_PATH.exists():
    print("[ERROR] final_mitma_mapping.json not found! Run validation script first.")
    exit()

with open(MAPPING_JSON_PATH, "r", encoding="utf-8") as f:
    name_to_id_map = json.load(f)

# Reverse map for labels and unique set for filtering
id_to_name_map = {v: k for k, v in name_to_id_map.items()}
target_id_set = set(name_to_id_map.values())

print(f"   > Loaded {len(target_id_set)} unique MITMA zone IDs.")

# --- 3. Loading Phase ---
print("\n--- 2. LOADING PHASE ---")
all_dataframes = []
all_files = list(MOBILITY_DIR.glob("*.csv"))
valid_files = sorted([(f, extract_date_from_filename(f.name)) for f in all_files if extract_date_from_filename(f.name)], key=lambda x: x[1])

for file_path, file_date in valid_files:
    try:
        # Load as string to preserve exact formatting including _AM
        temp_df = pd.read_csv(file_path, dtype={'id_origin': str, 'id_destination': str})
        
        # CRITICAL FIX: Do NOT slice to [:5]. Use the exact ID from the CSV.
        temp_df['origin'] = temp_df['id_origin'].str.strip()
        temp_df['destination'] = temp_df['id_destination'].str.strip()
        
        temp_df['date'] = file_date
        temp_df['timestamp'] = file_date + pd.to_timedelta(temp_df['hour'], unit='h')

        mask = temp_df['origin'].isin(target_id_set) | temp_df['destination'].isin(target_id_set)
        f_df = temp_df[mask].copy()
        
        if not f_df.empty:
            for col in ['activity_origin', 'activity_destination']:
                if col in f_df.columns:
                    f_df[col] = f_df[col].astype(str).str.strip().str.replace('_non_frequent', '_nonfrequent')
            
            len_col = next((c for c in f_df.columns if c.lower() in ['trips_total_length', 'trips_total_length_km']), None)
            f_df['avg_dist_km'] = f_df[len_col] / f_df['n_trips'].replace(0, 1) if len_col else np.nan
            all_dataframes.append(f_df)
            print(f"  Processed: {file_path.name}")
    except Exception as e: print(f"Error loading {file_path.name}: {e}")

if not all_dataframes:
    print("Error: No data loaded.")
    exit()

df = pd.concat(all_dataframes, ignore_index=True).sort_values('timestamp')

# --- 4. Data Classification ---
df['is_origin_target'] = df['origin'].isin(target_id_set)
df['is_dest_target'] = df['destination'].isin(target_id_set)
conditions = [(df['is_origin_target'] & df['is_dest_target']), (~df['is_origin_target'] & df['is_dest_target']), (df['is_origin_target'] & ~df['is_dest_target'])]
choices = ['Internal', 'External (In)', 'External (Out)']
df['flow_type'] = np.select(conditions, choices, default='Unknown')

# --- 5. Helpers ---
def add_daily_context(ax, dates):
    for i, d in enumerate(dates):
        if d.weekday() >= 5: ax.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.1, zorder=0)
    if EVENT_DATE in dates:
        ax.axvline(list(dates).index(EVENT_DATE), color='red', linestyle='--', linewidth=2, label='Event Impact', zorder=5)

def add_hourly_context(ax, timestamp_index):
    unique_days = pd.to_datetime(timestamp_index.date).unique()
    for day in unique_days:
        day_ts = pd.Timestamp(day)
        if day_ts != EVENT_DATE: ax.axvline(day_ts, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        if day_ts.weekday() >= 5: ax.axvspan(day_ts, day_ts + pd.Timedelta(days=1), color='gray', alpha=0.1, zorder=0)
    if timestamp_index.min() <= EVENT_DATE <= timestamp_index.max():
        ax.axvline(EVENT_DATE, color='red', linestyle='--', linewidth=2, label='Event Impact', zorder=5)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

# --- 6. Full Plot Generation ---

def generate_daily_plots(data):
    print("\n--- 3A. GENERATING DAILY PLOTS ---")
    inc_daily = data[data['is_dest_target']].groupby('date')['n_trips'].sum().sort_index()
    out_daily = data[data['is_origin_target']].groupby('date')['n_trips'].sum().sort_index()
    inc_act = data[data['is_dest_target']].groupby(['date', 'activity_destination'])['n_trips'].sum().unstack().fillna(0).sort_index()
    out_act = data[data['is_origin_target']].groupby(['date', 'activity_origin'])['n_trips'].sum().unstack().fillna(0).sort_index()
    comb_act = inc_act.add(out_act, fill_value=0).fillna(0).sort_index()
    labels_full = [d.strftime('%m-%d') for d in inc_act.index]
    sorted_cats = inc_act.sum().sort_values(ascending=False).index.tolist()
    colors = cm.tab20(np.linspace(0, 1, len(sorted_cats)))

    # 01 Volume
    fig1, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels_full))
    ax.bar(x - 0.17, inc_daily, 0.35, label='Incoming', color='teal', alpha=0.7)
    ax.bar(x + 0.17, out_daily, 0.35, label='Outgoing', color='coral', alpha=0.7)
    ax.set_title("DAILY 01: Mobility Volume"); ax.set_xticks(x); ax.set_xticklabels(labels_full, rotation=45)
    add_daily_context(ax, inc_act.index); ax.legend(); plt.tight_layout(); fig1.savefig(PLOTS_FOLDER / "Daily_01_Volume.png"); plt.close(fig1)

    # 02/03 Shares & Abs
    for pref, df_p, title in [('02a_Inc_Share', inc_act, 'Incoming Share %'), ('02b_Out_Share', out_act, 'Outgoing Share %')]:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.stackplot(labels_full, [(df_p.div(df_p.sum(axis=1), axis=0)*100)[c] for c in sorted_cats], labels=sorted_cats, colors=colors)
        ax.set_title(f"DAILY {title}"); add_daily_context(ax, df_p.index); fig.savefig(PLOTS_FOLDER / f"Daily_{pref}.png"); plt.close(fig)

    for pref, df_p, title in [('03a_Inc_Abs', inc_act, 'Absolute Incoming'), ('03b_Out_Abs', out_act, 'Absolute Outgoing')]:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.stackplot(labels_full, [df_p[c] for c in sorted_cats], labels=sorted_cats, colors=colors)
        ax.set_title(f"DAILY {title}"); add_daily_context(ax, df_p.index); fig.savefig(PLOTS_FOLDER / f"Daily_{pref}.png"); plt.close(fig)

    # 05 RDI Grids
    rdi_dates = [d for d in inc_act.index if d >= RDI_START_DATE]
    labels_rdi = [d.strftime('%m-%d') for d in rdi_dates]
    def plot_daily_rdi_grid(act_df, filename):
        base_avg = act_df[act_df.index < EVENT_DATE].groupby(act_df[act_df.index < EVENT_DATE].index.dayofweek).mean()
        fig, axes = plt.subplots(len(sorted_cats), 1, figsize=(10, 3*len(sorted_cats)), sharex=True)
        for i, cat in enumerate(sorted_cats):
            vals = [((act_df.loc[d, cat] - base_avg.loc[d.dayofweek, cat]) / max(base_avg.loc[d.dayofweek, cat], 0.001) * 100) for d in rdi_dates]
            axes[i].plot(labels_rdi, vals, color=colors[i], linewidth=2); axes[i].axhline(0, color='black'); axes[i].set_title(f"{cat} RDI")
            add_daily_context(axes[i], rdi_dates)
        plt.tight_layout(); fig.savefig(PLOTS_FOLDER / filename); plt.close(fig)

    plot_daily_rdi_grid(inc_act, "Daily_05a_Grid_Incoming.png")
    plot_daily_rdi_grid(out_act, "Daily_05b_Grid_Outgoing.png")
    plot_daily_rdi_grid(comb_act, "Daily_05c_Grid_Combined.png")

    # 06 Topology
    daily_topo = data.groupby(['date', 'flow_type'])['n_trips'].sum().unstack().fillna(0)
    fig6, ax = plt.subplots(figsize=(14, 6))
    daily_topo.plot(kind='bar', stacked=False, ax=ax, width=0.8, color=['#2ca02c', '#1f77b4', '#ff7f0e'])
    ax.set_title("DAILY 06: Internal vs External Flow"); ax.set_xticklabels(labels_full, rotation=45); add_daily_context(ax, daily_topo.index)
    plt.tight_layout(); fig6.savefig(PLOTS_FOLDER / "Daily_06_Flow_Topology.png"); plt.close(fig6)

    # 07 Distance
    if 'avg_dist_km' in data.columns:
        bins = np.arange(0, 102, 2); bin_centers = (bins[:-1] + bins[1:]) / 2
        for suffix, is_dest in [('a_Inc', True), ('b_Out', False)]:
            fig, ax = plt.subplots(figsize=(14, 7))
            sub = data[data['is_dest_target']] if is_dest else data[data['is_origin_target']]
            d_range = sorted([d for d in sub['date'].unique() if d >= pd.to_datetime("2024-10-28") and d.dayofweek in [0,2,4]])
            c_dist = cm.viridis(np.linspace(0, 1, len(d_range)))
            for i, d in enumerate(d_range):
                hist, _ = np.histogram(sub[sub['date']==d]['avg_dist_km'], bins=bins, weights=sub[sub['date']==d]['n_trips'])
                ax.plot(bin_centers, hist, color=c_dist[i], label=d.strftime('%Y-%m-%d'))
            ax.set_title(f"DAILY 07: Trip Length Distribution ({'Inc' if is_dest else 'Out'})"); ax.legend(); plt.tight_layout(); fig.savefig(PLOTS_FOLDER / f"Daily_07{suffix}_Dist.png"); plt.close(fig)

    # 08 Heatmaps
    for suffix, is_dest in [('a_Inc', True), ('b_Out', False)]:
        g_col = 'destination' if is_dest else 'origin'
        sub = data[data['is_dest_target']] if is_dest else data[data['is_origin_target']]
        pivot = sub.groupby(['date', g_col])['n_trips'].sum().reset_index()
        base = pivot[pivot['date'] < EVENT_DATE].groupby([g_col, pivot[pivot['date'] < EVENT_DATE]['date'].dt.dayofweek])['n_trips'].mean()
        rdi_h = pivot[pivot['date'] >= RDI_START_DATE].copy()
        rdi_h['rdi'] = rdi_h.apply(lambda r: ((r['n_trips'] - base.get((r[g_col], r['date'].dayofweek), 0)) / max(base.get((r[g_col], r['date'].dayofweek), 1), 1)) * 100, axis=1)
        
        # Apply the reverse map to get readable names instead of codes
        rdi_h['name'] = rdi_h[g_col].map(id_to_name_map).fillna(rdi_h[g_col])
        hm_data = rdi_h.pivot(index='name', columns='date', values='rdi').dropna().head(25)
        
        if not hm_data.empty:
            fig, ax = plt.subplots(figsize=(16, 12)); sns.heatmap(hm_data, cmap='RdBu', center=0, ax=ax); plt.tight_layout(); fig.savefig(PLOTS_FOLDER / f"Daily_08{suffix}_Heatmap.png"); plt.close(fig)

def generate_hourly_plots(data):
    print("\n--- 3B. GENERATING HOURLY PLOTS ---")
    h_count = data.groupby('timestamp')['n_trips'].count()
    h_backlog = h_count[::-1].cumsum()[::-1]
    print(f"Stats - Avg Feed: {h_count.mean():.2f} | Max Feed: {h_count.max()}")
    print(f"Stats - Avg Backlog: {h_backlog.mean():.2f} | Max Backlog: {h_backlog.max()}")

    inc_act = data[data['is_dest_target']].groupby(['timestamp', 'activity_destination'])['n_trips'].sum().unstack().fillna(0)
    out_act = data[data['is_origin_target']].groupby(['timestamp', 'activity_origin'])['n_trips'].sum().unstack().fillna(0)
    comb_act = inc_act.add(out_act, fill_value=0).fillna(0)
    full_range = pd.date_range(start=data['timestamp'].min(), end=data['timestamp'].max(), freq='H')
    inc_act = inc_act.reindex(full_range, fill_value=0); out_act = out_act.reindex(full_range, fill_value=0)
    sorted_cats = inc_act.sum().sort_values(ascending=False).index.tolist()
    colors = cm.tab20(np.linspace(0, 1, len(sorted_cats)))

    fig1, ax = plt.subplots(figsize=(15, 6))
    ax.plot(inc_act.index, inc_act.sum(axis=1), label='Incoming', color='teal'); ax.plot(out_act.index, out_act.sum(axis=1), label='Outgoing', color='coral', alpha=0.7)
    add_hourly_context(ax, inc_act.index); ax.legend(); fig1.savefig(PLOTS_FOLDER / "Hourly_01_Volume.png"); plt.close(fig1)

    for pref, df_p in [('02a_Inc_Share', inc_act), ('02b_Out_Share', out_act)]:
        fig, ax = plt.subplots(figsize=(15, 6)); rel = df_p.div(df_p.sum(axis=1), axis=0).fillna(0)*100
        ax.stackplot(df_p.index, [rel[c] for c in sorted_cats], labels=sorted_cats, colors=colors)
        add_hourly_context(ax, df_p.index); fig.savefig(PLOTS_FOLDER / f"Hourly_{pref}.png"); plt.close(fig)

    for pref, df_p in [('03a_Inc_Abs', inc_act), ('03b_Out_Abs', out_act)]:
        fig, ax = plt.subplots(figsize=(15, 6)); ax.stackplot(df_p.index, [df_p[c] for c in sorted_cats], labels=sorted_cats, colors=colors)
        add_hourly_context(ax, df_p.index); fig.savefig(PLOTS_FOLDER / f"Hourly_{pref}.png"); plt.close(fig)

    rdi_dates = inc_act.index[inc_act.index >= RDI_START_DATE]
    def plot_hourly_rdi_grid(act_df, filename):
        base_avg = act_df[act_df.index < EVENT_DATE].groupby([act_df[act_df.index < EVENT_DATE].index.dayofweek, act_df[act_df.index < EVENT_DATE].index.hour]).mean()
        fig, axes = plt.subplots(len(sorted_cats), 1, figsize=(12, 3*len(sorted_cats)), sharex=True)
        for i, cat in enumerate(sorted_cats):
            r_vals = [((act_df.loc[ts, cat] - base_avg.loc[(ts.dayofweek, ts.hour), cat]) / max(base_avg.loc[(ts.dayofweek, ts.hour), cat], 0.001) * 100) for ts in rdi_dates]
            axes[i].plot(rdi_dates, r_vals, color=colors[i]); axes[i].axhline(0, color='black'); add_hourly_context(axes[i], rdi_dates)
        plt.tight_layout(); fig.savefig(PLOTS_FOLDER / filename); plt.close(fig)

    plot_hourly_rdi_grid(inc_act, "Hourly_05a_Grid_Incoming.png")
    plot_hourly_rdi_grid(out_act, "Hourly_05b_Grid_Outgoing.png")
    plot_hourly_rdi_grid(comb_act.reindex(full_range, fill_value=0), "Hourly_05c_Grid_Combined.png")

    fig6, ax = plt.subplots(figsize=(16, 7))
    topo = data.groupby(['timestamp', 'flow_type'])['n_trips'].sum().unstack().fillna(0).reindex(full_range, fill_value=0)
    topo.plot(ax=ax, color=['#2ca02c', '#1f77b4', '#ff7f0e']); add_hourly_context(ax, topo.index); plt.tight_layout(); fig6.savefig(PLOTS_FOLDER / "Hourly_06_Flow_Topology.png"); plt.close(fig6)

def generate_hourly_shock_grids(data):
    print("\n--- 3D. GENERATING SHOCK GRIDS ---")
    shock_data = data[data['date'].dt.date == EVENT_DATE.date()]
    base_data = data[data['date'].dt.date == (EVENT_DATE - pd.Timedelta(days=7)).date()]
    if shock_data.empty or base_data.empty: return
    for suffix, col, is_dest in [('a_Inc', 'activity_destination', True), ('b_Out', 'activity_origin', False)]:
        s_sub = shock_data[shock_data['is_dest_target']] if is_dest else shock_data[shock_data['is_origin_target']]
        b_sub = base_data[base_data['is_dest_target']] if is_dest else base_data[base_data['is_origin_target']]
        top = s_sub.groupby(col)['n_trips'].sum().sort_values(ascending=False).index.tolist()
        fig, axes = plt.subplots(len(top), 1, figsize=(10, 3*len(top)), sharex=True)
        for i, act in enumerate(top):
            s_h = s_sub[s_sub[col]==act].groupby(s_sub['timestamp'].dt.hour)['n_trips'].sum().reindex(range(24), fill_value=0)
            b_h = b_sub[b_sub[col]==act].groupby(b_sub['timestamp'].dt.hour)['n_trips'].sum().reindex(range(24), fill_value=0)
            axes[i].plot(b_h.index, b_h.values, color='gray', linestyle='--'); axes[i].plot(s_h.index, s_h.values, color='red', marker='o'); axes[i].set_title(f"{act} ({suffix})")
        plt.tight_layout(); fig.savefig(PLOTS_FOLDER / f"Hourly_09{suffix}_Shock.png"); plt.close(fig)

# --- 7. Run ---
generate_daily_plots(df)
generate_hourly_plots(df)
generate_hourly_shock_grids(df)
print("\nWorkflow Complete.")