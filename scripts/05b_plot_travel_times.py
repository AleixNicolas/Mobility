import numpy as np
import json
import logging
from pathlib import Path
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===============================================================
# PATH CONFIGURATION
# ===============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

ROUTING_DIR = DATA_ROOT / "04_routing_networks"
SCENARIO_BASE = DATA_ROOT / "05_scenario_models"
PLOTS_DIR = DATA_ROOT / "06_outputs" / "plots_I"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ===============================================================
# MAPPING & PALETTES
# ===============================================================
COLOR_PALETTE = {
    "10 yr": "#FFD700",
    "100 yr": "#FF7F00",
    "500 yr": "#B22222",
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
    "DANA_08_11_2024": "08/11/2024",
    "Normal Conditions": "Unperturbed"
}

# Define which scenarios to plot
TARGET_SCENARIOS = [
    "DANA_31_10_2024",
    "DANA_03_11_2024",
    "DANA_05_11_2024",
    "DANA_08_11_2024"
]

# ===============================================================
# DATA LOADING & PROCESSING
# ===============================================================
def process_times(results_dict):
    """Extracts raw times, counts unreachables, and returns non-zero times in minutes."""
    raw_times = [v["time"] for v in results_dict.values()]
    unreachable = sum(t is None for t in raw_times)
    numeric_times = [t for t in raw_times if isinstance(t, (int, float))]
    
    # Convert OSMnx seconds to minutes
    non_zero_times_mins = [t / 60 for t in numeric_times if t > 0]
    
    zeros = unreachable + numeric_times.count(0)
    total = len(raw_times)
    
    return non_zero_times_mins, zeros, total

def main():
    plot_data = {}
    zero_counts = {}
    all_counts = {}

    # Load Baseline (NP)
    base_path = ROUTING_DIR / "shortest_paths_NP.json"
    if base_path.exists():
        with open(base_path, 'r') as f:
            times, zeros, total = process_times(json.load(f))
            plot_data["Normal Conditions"] = times
            zero_counts["Normal Conditions"] = zeros
            all_counts["Normal Conditions"] = total
    else:
        logging.error(f"Missing baseline routing matrix at {base_path}")
        return

    # Load Scenarios
    for name in TARGET_SCENARIOS:
        # Navigate into the specific scenario subfolder
        scen_path = SCENARIO_BASE / name / f"shortest_paths_{name}.json"
        
        if scen_path.exists():
            with open(scen_path, 'r') as f:
                times, zeros, total = process_times(json.load(f))
                plot_data[name] = times
                zero_counts[name] = zeros
                all_counts[name] = total
        else:
            logging.warning(f"Missing scenario matrix at {scen_path}")

    # ===============================================================
    # PLOTTING LOGIC
    # ===============================================================
    fig, (ax_kde, ax_bar) = plt.subplots(
        1, 2, 
        figsize=(14, 7), 
        gridspec_kw={'width_ratios': [2.5, 1]}
    )

    # 1. KDE Plot
    global_max_time = max([max(times) if times else 0 for times in plot_data.values()])
    x_vals = np.linspace(0, global_max_time, 500)

    # Plot specific order: Scenarios first, Baseline last to keep it on top
    plot_order = [s for s in TARGET_SCENARIOS if s in plot_data] + ["Normal Conditions"]

    for name in plot_order:
        times = plot_data[name]
        if len(times) > 1:
            kde = gaussian_kde(times)
            y = kde(x_vals)

            ratio = len(times) / all_counts[name] if all_counts[name] > 0 else 0
            y_rescaled = y * ratio

            color = COLOR_PALETTE.get(name, '#000000')
            label = SCENARIO_MAPPING.get(name, name)

            ax_kde.plot(x_vals, y_rescaled, label=label, color=color, linewidth=2)

    ax_kde.set_xlim(0, global_max_time)
    ax_kde.set_ylim(bottom=0)
    ax_kde.set_xlabel("Travel Time (minutes)", fontsize=20)
    ax_kde.set_ylabel("Density (Rescaled by Reachability)", fontsize=16)
    ax_kde.grid(True, alpha=0.3)
    ax_kde.legend(loc='upper right', fontsize=16)
    ax_kde.tick_params(axis='both', labelsize=16)
    ax_kde.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_kde.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # 2. Reachability Bar Chart
    valid_scenarios = [s for s in TARGET_SCENARIOS if s in plot_data]
    display_names = [SCENARIO_MAPPING.get(s, s) for s in valid_scenarios]
    
    reachable = [(all_counts[n] - zero_counts[n]) / all_counts[n] for n in valid_scenarios]
    unreachable = [1 - r for r in reachable]

    bar_positions = np.arange(len(valid_scenarios))
    ax_bar.barh(bar_positions, unreachable, color='#ff6666', label='Unreachable', edgecolor="white")
    ax_bar.barh(bar_positions, reachable, left=unreachable, color='#66cc99', label='Reachable', edgecolor="white")

    ax_bar.set_yticks(bar_positions)
    ax_bar.set_yticklabels(display_names, fontsize=16)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Fraction of cut routes", fontsize=18)
    ax_bar.tick_params(axis='x', labelsize=14)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.5)
    ax_bar.invert_yaxis()

    plt.tight_layout()

    # ===============================================================
    # SAVE OUTPUTS
    # ===============================================================
    pdf_path = PLOTS_DIR / 'Travel_times_DANA.pdf'
    png_path = PLOTS_DIR / 'Travel_times_DANA.png'
    
    plt.savefig(pdf_path, format='pdf', dpi=300)
    plt.savefig(png_path, dpi=300)
    
    logging.info(f"Saved plots to {PLOTS_DIR}")

if __name__ == "__main__":
    main()