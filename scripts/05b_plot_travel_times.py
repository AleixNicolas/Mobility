import numpy as np
import json
import logging
from pathlib import Path
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.ticker import MaxNLocator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===============================================================
# PATH CONFIGURATION
# ===============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

ROUTING_DIR = DATA_ROOT / "04_routing_networks"
SCENARIO_BASE = DATA_ROOT / "05_scenario_models"
SHARED_DIR = DATA_ROOT / "03_processed_shared"
PLOTS_DIR = DATA_ROOT / "06_outputs" / "plots_I"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================================
# MAPPING & PALETTES
# ===============================================================
COLOR_PALETTE = {
    "DANA_31_10_2024": "#8A2BE2",
    "DANA_03_11_2024": "#FF1493",
    "DANA_05_11_2024": "#00CED1",
    "DANA_06_11_2024": "#32CD32",
    "DANA_08_11_2024": "#1E90FF",
    "Normal Conditions": "#808080"
}

SCENARIO_MAPPING = {
    "DANA_31_10_2024": "31/10/2024",
    "DANA_03_11_2024": "03/11/2024",
    "DANA_05_11_2024": "05/11/2024",
    "DANA_06_11_2024": "06/11/2024",
    "DANA_08_11_2024": "08/11/2024",
    "Normal Conditions": "Unperturbed"
}

# Comment out any day here to remove it from all plots and reports dynamically
SCENARIO_DATES = {
    "DANA_31_10_2024": "2024-10-31",
    "DANA_03_11_2024": "2024-11-03",
    "DANA_05_11_2024": "2024-11-05",
    "DANA_06_11_2024": "2024-11-06",
    "DANA_08_11_2024": "2024-11-08"
}

TARGET_SCENARIOS = list(SCENARIO_DATES.keys())

# ===============================================================
# DATA PROCESSING
# ===============================================================
def load_weight_and_volume_data():
    weight_map = {}
    trip_volumes = {scen: {"baseline": 0, "actual": 0} for scen in TARGET_SCENARIOS}
    metrics_path = SHARED_DIR / "final_mobility_metrics.json"
    
    if not metrics_path.exists():
        raise FileNotFoundError(f"Required data file missing: {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        for row in metrics:
            name = row.get("official_name")
            base_out = row.get("base_out", 0)
            if name and base_out > 0:
                clean_name = str(name).split('/')[0].strip()
                weight_map[clean_name] = max(weight_map.get(clean_name, 0), base_out)
            
            row_date = row.get("date")
            for scen, target_date in SCENARIO_DATES.items():
                if row_date == target_date:
                    trip_volumes[scen]["baseline"] += row.get("base_out", 0)
                    trip_volumes[scen]["actual"] += row.get("abs_outgoing", 0)
    return weight_map, trip_volumes

def extract_base_times(base_path):
    base_times = {}
    if base_path.exists():
        with open(base_path, 'r') as f:
            data = json.load(f)
            for k, v in data.items():
                if v["time"] is not None and v["time"] > 0:
                    base_times[k] = v["time"] / 60.0
    return base_times

def process_scenario_data(results_dict, weight_map, base_times):
    times, weights, deltas, delta_weights = [], [], [], []
    unreachable, numeric_times_count = 0, 0
    
    for key, v in results_dict.items():
        time_val = v["time"]
        src = key.split("__")[0].strip()
        w = weight_map.get(src, 1.0)
        
        if time_val is None:
            unreachable += 1
        elif isinstance(time_val, (int, float)):
            time_mins = time_val / 60.0
            if time_mins > 0:
                times.append(time_mins)
                weights.append(w)
                if key in base_times:
                    deltas.append(time_mins - base_times[key])
                    delta_weights.append(w)
            if time_mins == 0: numeric_times_count += 1 
                
    zeros = unreachable + numeric_times_count
    total = len(results_dict)
    mean_delta = np.average(deltas, weights=delta_weights) if deltas else 0
    reachability = (total - zeros) / total if total > 0 else 0
    
    return {"times": times, "weights": weights, "mean_delta": mean_delta, "reachability": reachability}, zeros, total

def calc_distribution_metrics(times, weights=None):
    if not times or len(times) <= 1:
        return {"mean": 0.0, "max": 0.0, "peak": 0.0}
    
    mean_val = np.average(times, weights=weights) if weights else np.mean(times)
    max_val = max(times)
    
    try:
        kde = gaussian_kde(times, weights=weights)
        x_vals = np.linspace(0, max_val, 1000)
        y_vals = kde(x_vals)
        peak_val = x_vals[np.argmax(y_vals)]
    except np.linalg.LinAlgError:
        peak_val = 0.0 
        
    return {"mean": mean_val, "max": max_val, "peak": peak_val}

def analyze_peak_sources(scenario_name, target_min=140, window=20):
    """
    Identifies which OD pairs are contributing to a specific peak in the distribution.
    """
    results_path = SCENARIO_BASE / scenario_name / f"shortest_paths_{scenario_name}.json"
    base_path = ROUTING_DIR / "shortest_paths_NP.json"
    
    if not results_path.exists() or not base_path.exists():
        logging.error("Source files for peak analysis missing.")
        return

    with open(results_path, 'r') as f:
        p_data = json.load(f)
    with open(base_path, 'r') as f:
        np_data = json.load(f)

    lower_bound = target_min - window
    upper_bound = target_min + window
    
    peak_contributors = []

    for key, v in p_data.items():
        time_val = v.get("time")
        if time_val is not None:
            time_mins = time_val / 60.0
            
            # Check if this route falls into our 'Second Peak' window
            if lower_bound <= time_mins <= upper_bound:
                src, dst = key.split("__")
                base_time = np_data.get(key, {}).get("time", 0) / 60.0
                delay = time_mins - base_time
                
                peak_contributors.append({
                    "od_pair": key,
                    "origin": src.strip(),
                    "destination": dst.strip(),
                    "perturbed_time": time_mins,
                    "baseline_time": base_time,
                    "delay_minutes": delay
                })

    # Sort by the largest absolute delay to see who is most impacted
    peak_contributors.sort(key=lambda x: x['delay_minutes'], reverse=True)

    # Export a text report
    output_report = PLOTS_DIR / f"peak_analysis_{scenario_name}_{target_min}min.txt"
    with open(output_report, "w", encoding="utf-8") as f:
        f.write(f"=== PEAK ANALYSIS: {scenario_name} @ {target_min} minutes ===\n")
        f.write(f"Window: {lower_bound} to {upper_bound} minutes\n")
        f.write(f"Total OD pairs in this peak: {len(peak_contributors)}\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'OD Pair':<45} | {'Base (m)':<10} | {'Scenario (m)':<12} | {'Delay (m)':<10}\n")
        f.write("-" * 90 + "\n")
        for item in peak_contributors[:50]: # Top 50 contributors
            f.write(f"{item['od_pair']:<45} | {item['baseline_time']:<10.2f} | {item['perturbed_time']:<12.2f} | {item['delay_minutes']:<10.2f}\n")

    logging.info(f"Peak analysis report saved to {output_report.name}")

# ===============================================================
# PLOTTING FUNCTIONS
# ===============================================================
def plot_kde(ax, data_dict, all_counts, metric_key, weight_key, title, xlabel, x_max=None):
    global_max_x = x_max if x_max else max([max(v[metric_key]) if v[metric_key] else 0 for v in data_dict.values()])
    x_vals = np.linspace(0, global_max_x, 500)
    plot_order = [s for s in TARGET_SCENARIOS if s in data_dict] + ["Normal Conditions"]
    max_y_observed = 0.0

    for name in plot_order:
        if name not in data_dict: continue
        vals, weights = data_dict[name][metric_key], data_dict[name][weight_key] if weight_key else None
        if len(vals) > 1:
            color, label = COLOR_PALETTE.get(name, '#000000'), SCENARIO_MAPPING.get(name, name)
            ratio = len(vals) / all_counts[name] if all_counts[name] > 0 else 0
            kde = gaussian_kde(vals, weights=weights)
            y = kde(x_vals) * ratio
            max_y_observed = max(max_y_observed, max(y))
            ax.plot(x_vals, y, label=label, color=color, linewidth=2)

    ax.set_xlim(0, global_max_x)
    ax.set_xlabel(xlabel, fontsize=12); ax.set_title(title, fontsize=13); ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10); ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    return max_y_observed

# --- PLOT OPTIONS ---

def plot_opt1_split_color(display_names, baselines, actuals, delays, reach):
    num_days = len(display_names)
    fig, (ax_vol, ax_del, ax_reach) = plt.subplots(3, 1, figsize=(max(10, 2 * num_days), 11), sharex=True, 
                                                   gridspec_kw={'height_ratios': [2, 1, 1]})
    x = np.arange(num_days)
    width = 0.35
    cmap = plt.get_cmap('RdYlGn')

    ax_vol.bar(x - width/2, baselines, width, label='Baseline', color='#E0E0E0', edgecolor='grey')
    ax_vol.bar(x + width/2, actuals, width, label='Actual', color='#4C72B0', edgecolor='black')
    ax_vol.set_ylabel("Total Trips"); ax_vol.set_title("Network Mobility Dynamics", weight='bold'); ax_vol.legend()
    
    ax_del.plot(x, delays, marker='o', color='#C44E52', linewidth=2.5); ax_del.set_ylabel("Mean Delay (m)")
    
    colors = [cmap(r) for r in reach]
    ax_reach.bar(x, reach, color=colors, edgecolor='black', alpha=0.8)
    ax_reach.set_ylabel("Reachability (%)"); ax_reach.set_ylim(0, 1.1)
    ax_reach.set_xticks(x); ax_reach.set_xticklabels(display_names)
    plt.tight_layout(); fig.savefig(PLOTS_DIR / '02_Opt1_Split.pdf', format='pdf', bbox_inches='tight'); plt.close(fig)

def plot_opt4_triple_donut_labeled(display_names, baselines, actuals, delays, reach):
    num_days = len(display_names)
    fig, axes = plt.subplots(1, num_days, figsize=(5 * num_days, 6))
    if num_days == 1: axes = [axes] # Handle single day case smoothly
    
    cmap_mob, cmap_reach, cmap_del = plt.get_cmap('Greens'), plt.get_cmap('Blues'), plt.get_cmap('Reds')
    text_outline = [pe.withStroke(linewidth=2.5, foreground='black')]
    
    for i, ax in enumerate(axes):
        ret_pct = (actuals[i] / baselines[i]) * 100
        reach_pct = reach[i] * 100
        eff_pct = max(0, 100 - (delays[i] / 0.6)) 
        
        ax.pie([ret_pct, 100-ret_pct], radius=1.0, colors=[cmap_mob(0.6), '#F0F0F0'], startangle=90, wedgeprops=dict(width=0.2, edgecolor='white'))
        ax.text(0, 0.9, "Mobility", ha='center', va='center', fontsize=9, weight='bold', color='white', path_effects=text_outline)
        
        ax.pie([reach_pct, 100-reach_pct], radius=0.78, colors=[cmap_reach(0.6), '#F0F0F0'], startangle=90, wedgeprops=dict(width=0.2, edgecolor='white'))
        ax.text(0, 0.68, "Reach", ha='center', va='center', fontsize=9, weight='bold', color='white', path_effects=text_outline)
        
        ax.pie([eff_pct, 100-eff_pct], radius=0.56, colors=[cmap_del(0.6), '#F0F0F0'], startangle=90, wedgeprops=dict(width=0.2, edgecolor='white'))
        ax.text(0, 0.46, "Speed", ha='center', va='center', fontsize=9, weight='bold', color='white', path_effects=text_outline)
        
        ax.text(0, 0, f"{display_names[i]}\n{ret_pct:.0f}% M\n{reach_pct:.0f}% R\n+{delays[i]:.1f}m", ha='center', va='center', fontsize=10, weight='bold')
        
    plt.suptitle("Daily Matrix (Labeled Rings): Mobility | Reachability | Speed", fontsize=16, weight='bold', y=1.05)
    plt.tight_layout(); fig.savefig(PLOTS_DIR / '02_Opt4_Donut_Labeled.pdf', format='pdf', bbox_inches='tight'); plt.close(fig)

def plot_opt4_variant_donut(display_names, baselines, actuals, delays, reach):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    metrics = {
        "Mobility Retention %": [(a/b)*100 for a, b in zip(actuals, baselines)],
        "Network Reachability %": [r*100 for r in reach],
        "Delay Efficiency %": [max(0, 100 - (d / 0.6)) for d in delays] 
    }
    cmaps = [plt.get_cmap('Greens'), plt.get_cmap('Blues'), plt.get_cmap('Reds')]
    text_outline = [pe.withStroke(linewidth=2.5, foreground='black')]
    
    # Dynamically scale the radii based on how many days we are plotting
    num_days = len(display_names)
    radii = np.linspace(1.0, 0.3, num_days) if num_days > 1 else [1.0]
    
    for ax, (title, values), cmap in zip(axes, metrics.items(), cmaps):
        for i, val in enumerate(values):
            color = cmap(0.3 + (i * (0.6 / max(1, num_days - 1)))) # Scale colors to avoid washing out
            ring_width = 0.6 / max(1, num_days) # Dynamic width to avoid overlapping
            ax.pie([val, 100-val], radius=radii[i], colors=[color, '#F0F0F0'], startangle=90, wedgeprops=dict(width=ring_width, edgecolor='white'))
            ax.text(0, radii[i] - (ring_width / 2), display_names[i][:5], ha='center', va='center', fontsize=8, color='white', weight='bold', path_effects=text_outline)
            
        ax.text(0, 0, title.replace(" ", "\n"), ha='center', va='center', fontsize=11, weight='bold')
    
    plt.suptitle("Timeline by Metric (Outer = First Day, Inner = Last Day)", fontsize=16, weight='bold', y=1.05)
    plt.tight_layout(); fig.savefig(PLOTS_DIR / '02_Opt4_Variant_Donut.pdf', format='pdf', bbox_inches='tight'); plt.close(fig)

def plot_opt5_triple_bar(display_names, baselines, actuals, delays, reach):
    num_days = len(display_names)
    fig, ax = plt.subplots(figsize=(max(12, 2.5 * num_days), 8))
    x = np.arange(num_days)
    width = 0.25
    
    mob_retained = [(a/b) for a, b in zip(actuals, baselines)]
    max_del = max(delays) if max(delays) > 0 else 1
    delay_impact = [d/max_del for d in delays]
    
    ax.bar(x - width, mob_retained, width, label='Mobility Retention', color='#66cc99', edgecolor='black')
    ax.bar(x, reach, width, label='Network Reachability', color='#6699ff', edgecolor='black')
    ax.bar(x + width, delay_impact, width, label='Delay Impact (Normalized)', color='#ff6666', edgecolor='black')
    
    ax.set_xticks(x); ax.set_xticklabels(display_names, fontsize=12, weight='bold')
    ax.set_ylabel("Normalized Performance Index (0.0 - 1.0)", fontsize=12)
    ax.set_title("Comparative Triple-Metric Daily Analysis", fontsize=15, weight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout(); fig.savefig(PLOTS_DIR / '02_Opt5_Triple_Bar.pdf', format='pdf', bbox_inches='tight'); plt.close(fig)

def plot_opt5_variant_bar(display_names, baselines, actuals, delays, reach):
    num_days = len(display_names)
    fig, (ax_mob, ax_reach, ax_del) = plt.subplots(3, 1, figsize=(max(10, 2 * num_days), 12), sharex=True)
    x = np.arange(num_days)
    
    mob_retained = [(a/b)*100 for a, b in zip(actuals, baselines)]
    ax_mob.bar(x, mob_retained, color='#66cc99', edgecolor='black', width=0.5)
    ax_mob.set_title("Mobility Retention (%)", weight='bold')
    ax_mob.set_ylim(0, max(mob_retained)*1.2 if mob_retained else 100)
    for i, v in enumerate(mob_retained):
        ax_mob.text(i, v + 2, f"{v:.1f}%", ha='center', weight='bold')

    reach_pct = [r*100 for r in reach]
    ax_reach.bar(x, reach_pct, color='#6699ff', edgecolor='black', width=0.5)
    ax_reach.set_title("Network Reachability (%)", weight='bold')
    ax_reach.set_ylim(0, 110)
    for i, v in enumerate(reach_pct):
        ax_reach.text(i, v + 2, f"{v:.1f}%", ha='center', weight='bold')

    ax_del.bar(x, delays, color='#ff6666', edgecolor='black', width=0.5)
    ax_del.set_title("Mean Expected Delay (minutes)", weight='bold')
    ax_del.set_ylim(0, max(delays)*1.2 if max(delays) > 0 else 10)
    for i, v in enumerate(delays):
        ax_del.text(i, v + 0.5, f"+{v:.1f}m", ha='center', weight='bold')

    for ax in (ax_mob, ax_reach, ax_del):
        ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    ax_del.set_xticks(x)
    ax_del.set_xticklabels([d[:5] for d in display_names], fontsize=12, weight='bold')

    plt.tight_layout(); fig.savefig(PLOTS_DIR / '02_Opt5_Variant_Bar_Vertical.pdf', format='pdf', bbox_inches='tight'); plt.close(fig)

# ===============================================================
# MAIN EXECUTION
# ===============================================================
def main():
    if not TARGET_SCENARIOS:
        logging.warning("No scenarios active in SCENARIO_DATES. Exiting.")
        return

    weight_map, trip_volumes = load_weight_and_volume_data()
    base_path = ROUTING_DIR / "shortest_paths_NP.json"
    base_times = extract_base_times(base_path)
    
    plot_data, zero_counts, all_counts = {}, {}, {}
    with open(base_path, 'r') as f:
        data, zeros, total = process_scenario_data(json.load(f), weight_map, base_times)
        plot_data["Normal Conditions"], zero_counts["Normal Conditions"], all_counts["Normal Conditions"] = data, zeros, total

    for name in TARGET_SCENARIOS:
        file_path = SCENARIO_BASE / name / f"shortest_paths_{name}.json"
        if not file_path.exists():
            logging.warning(f"File missing for {name}: {file_path}. Skipping.")
            continue
        with open(file_path, 'r') as f:
            data, zeros, total = process_scenario_data(json.load(f), weight_map, base_times)
            plot_data[name], zero_counts[name], all_counts[name] = data, zeros, total

    # --- FIGURE 1: ABSOLUTE NETWORK STATE (RETAINED) ---
    val_scens = [s for s in TARGET_SCENARIOS if s in plot_data]
    if not val_scens:
        logging.error("No valid scenario data loaded to plot.")
        return

    fig1, (ax1_un, ax1_wt, ax1_bar) = plt.subplots(1, 3, figsize=(22, 7), gridspec_kw={'width_ratios': [2, 2, 1]})
    global_max = max([max(v['times']) for v in plot_data.values() if v['times']])
    
    y_un = plot_kde(ax1_un, plot_data, all_counts, "times", None, "Structural Times (Unweighted)", "Minutes", global_max)
    y_wt = plot_kde(ax1_wt, plot_data, all_counts, "times", "weights", "Human Impact (Weighted)", "Minutes", global_max)
    shared_y = max(y_un, y_wt) * 1.1
    ax1_un.set_ylim(0, shared_y); ax1_wt.set_ylim(0, shared_y)
    
    r, u = [(all_counts[n]-zero_counts[n])/all_counts[n] for n in val_scens], [1-((all_counts[n]-zero_counts[n])/all_counts[n]) for n in val_scens]
    
    ax1_bar.barh(np.arange(len(val_scens)), u, color='#ff6666', label='Unreachable')
    ax1_bar.barh(np.arange(len(val_scens)), r, left=u, color='#66cc99', label='Reachable')
    ax1_bar.set_yticks(np.arange(len(val_scens))); ax1_bar.set_yticklabels([SCENARIO_MAPPING[s] for s in val_scens])
    ax1_bar.set_title("Network Reachability", fontsize=14, pad=10)
    ax1_bar.set_xlabel("Fraction of Routes", fontsize=14)
    ax1_bar.invert_yaxis(); ax1_bar.legend(loc='lower left')
    
    plt.tight_layout(); fig1.savefig(PLOTS_DIR / '01_Network_State.pdf', format='pdf', bbox_inches='tight'); plt.close(fig1)

    # --- PREPARE DATA FOR COMPARATIVE PLOTS ---
    d_names = [SCENARIO_MAPPING[s] for s in val_scens]
    bases = [trip_volumes[s]["baseline"] for s in val_scens]
    acts = [trip_volumes[s]["actual"] for s in val_scens]
    dels = [plot_data[s]["mean_delta"] for s in val_scens]
    reaches = [plot_data[s]["reachability"] for s in val_scens]
    
    # --- NUMERICAL TEXT REPORT GENERATION ---
    report_path = PLOTS_DIR / '00_numerical_plot_data.txt'
    with open(report_path, 'w') as f:
        f.write("=== COMPARATIVE NETWORK METRICS ===\n")
        f.write("Core metrics passed into the daily comparative plot functions.\n")
        f.write("=" * 85 + "\n")
        
        header = f"{'Scenario/Date':<18} | {'Baseline Vol':<13} | {'Actual Vol':<13} | {'Mean Delay (m)':<15} | {'Reachability (%)':<15}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        for i in range(len(d_names)):
            reach_pct = reaches[i] * 100
            f.write(f"{d_names[i]:<18} | {bases[i]:<13.0f} | {acts[i]:<13.0f} | {dels[i]:<15.2f} | {reach_pct:<15.1f}\n")
            
        f.write("\n\n")
        
        f.write("=== TRAVEL TIME DISTRIBUTIONS (MINUTES) ===\n")
        f.write("Descriptive statistics extracted from the network routing times (KDE curves).\n")
        f.write("=" * 95 + "\n")
        
        dist_header = f"{'Scenario/Date':<18} | {'Unweighted Peak':<16} | {'Unw. Mean':<12} | {'Weighted Peak':<16} | {'Wgt. Mean':<12} | {'Absolute Max':<12}"
        f.write(dist_header + "\n")
        f.write("-" * len(dist_header) + "\n")
        
        plot_order_with_base = ["Normal Conditions"] + val_scens
        for scn in plot_order_with_base:
            times = plot_data[scn]["times"]
            weights = plot_data[scn]["weights"]
            
            unw_metrics = calc_distribution_metrics(times, weights=None)
            wgt_metrics = calc_distribution_metrics(times, weights=weights)
            
            display_name = SCENARIO_MAPPING.get(scn, scn)
            
            f.write(f"{display_name:<18} | "
                    f"{unw_metrics['peak']:<16.2f} | "
                    f"{unw_metrics['mean']:<12.2f} | "
                    f"{wgt_metrics['peak']:<16.2f} | "
                    f"{wgt_metrics['mean']:<12.2f} | "
                    f"{unw_metrics['max']:<12.2f}\n")
    
    logging.info(f"Generated numerical data report at {report_path}")

    # Generate the remaining plots
    plot_opt1_split_color(d_names, bases, acts, dels, reaches)
    plot_opt4_triple_donut_labeled(d_names, bases, acts, dels, reaches)
    plot_opt4_variant_donut(d_names, bases, acts, dels, reaches)
    plot_opt5_triple_bar(d_names, bases, acts, dels, reaches)
    plot_opt5_variant_bar(d_names, bases, acts, dels, reaches)

    # -----------------------------------------------------------
    # PEAK INVESTIGATION (Diagnostic)
    # -----------------------------------------------------------
    target_scenario = "DANA_31_10_2024"
    if target_scenario in plot_data:
        logging.info(f"Running peak investigation for {target_scenario} at 140m...")
        # target_min is the center of the peak, window is +/- range
        analyze_peak_sources(target_scenario, target_min=140, window=15)
    
    logging.info(f"Generated vectorized 01_Network_State and all Option variations in {PLOTS_DIR}")

if __name__ == "__main__":
    main()