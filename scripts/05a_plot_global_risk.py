import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ===============================================================
# PATH CONFIGURATION
# ===============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR.parent / "data"

SCENARIO_BASE = DATA_ROOT / "05_scenario_models"
GLOBAL_METRICS_DIR = SCENARIO_BASE / "global_metrics"
PLOTS_DIR = DATA_ROOT / "06_outputs" / "plots_I"

# Ensure output directory exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = GLOBAL_METRICS_DIR / "R_G.json"

# ===============================================================
# LABEL MAPPING & CATEGORIZATION
# ===============================================================
# Split into logical groups for better visualization
PREDICTIVE_MAPPING = {
    "10 yr": "High Prob. (10yr)",
    "100 yr": "Med. Prob. (100yr)",
    "500 yr": "Low Prob. (500yr)"
}

DANA_MAPPING = {
    "DANA_31_10_2024": "31/10/2024",
    "DANA_03_11_2024": "03/11/2024",
    "DANA_05_11_2024": "05/11/2024",
    "DANA_06_11_2024": "06/11/2024",
    "DANA_08_11_2024": "08/11/2024"
}

# Combined for the master plot
SCENARIO_MAPPING = {**PREDICTIVE_MAPPING, **DANA_MAPPING}

# ===============================================================
# EXECUTION
# ===============================================================
def main():
    if not INPUT_PATH.exists():
        logging.error(f"Could not find input file: {INPUT_PATH}")
        return

    logging.info("Loading Global Risk (R_G) data...")
    with open(INPUT_PATH, 'r') as f:
        R = json.load(f)

    # Extract data
    names_all, values_all, colors_all = [], [], []
    names_dana, values_dana = [], []
    names_pred, values_pred = [], []
    
    for json_key, display_label in SCENARIO_MAPPING.items():
        if json_key in R:
            val = R[json_key]
            names_all.append(display_label)
            values_all.append(val)
            
            # Segment data for specialized plots
            if json_key in PREDICTIVE_MAPPING:
                colors_all.append("slategray")
                names_pred.append(display_label)
                values_pred.append(val)
            else:
                colors_all.append("firebrick")
                names_dana.append(display_label)
                values_dana.append(val)
        else:
            logging.warning(f"Scenario '{json_key}' missing from R_G.json")

    if not values_all:
        logging.error("No valid data found to plot.")
        return

    # -----------------------------------------------------------
    # PLOT 1: Master Combined Bar Chart (Segmented Colors)
    # -----------------------------------------------------------
    logging.info("Generating Master Bar Plot...")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    bars = ax1.bar(names_all, values_all, color=colors_all, alpha=0.85, edgecolor="black")
    
    ax1.set_ylim(0, max(values_all) * 1.2) # Give breathing room for labels
    ax1.set_ylabel('Global Risk ($R_{G}$)', fontsize=12, weight='bold')
    ax1.set_title('Network Global Risk ($R_{G}$) across All Scenarios', fontsize=14, weight='bold')
    ax1.set_xticklabels(names_all, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.3f}", ha='center', va='bottom', fontsize=10)
    
    # Custom legend
    import matplotlib.patches as mpatches
    pred_patch = mpatches.Patch(color='slategray', label='Predictive Return Periods')
    dana_patch = mpatches.Patch(color='firebrick', label='DANA Empirical Timeline')
    ax1.legend(handles=[pred_patch, dana_patch], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "01_Global_Risk_Master.png", dpi=300)
    plt.close(fig1)

    # -----------------------------------------------------------
    # PLOT 2: DANA Recovery Curve (Line Chart)
    # -----------------------------------------------------------
    if values_dana:
        logging.info("Generating DANA Recovery Curve...")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(names_dana, values_dana, marker='o', color='firebrick', linewidth=2.5, markersize=8)
        
        ax2.set_ylim(0, max(values_dana) * 1.2)
        ax2.set_ylabel('Global Risk ($R_{G}$)', fontsize=12, weight='bold')
        ax2.set_title('DANA Event: Global Network Recovery Timeline', fontsize=14, weight='bold')
        ax2.grid(linestyle='--', alpha=0.7)
        
        for i, txt in enumerate(values_dana):
            ax2.annotate(f"{txt:.3f}", (names_dana[i], values_dana[i]), 
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=11, weight='bold')
            
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "02_Global_Risk_DANA_Recovery.png", dpi=300)
        plt.close(fig2)

    # -----------------------------------------------------------
    # TEXT REPORT EXPORT
    # -----------------------------------------------------------
    logging.info("Exporting Text Report...")
    report_path = PLOTS_DIR / "00_global_risk_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== GLOBAL NETWORK RISK (R_G) REPORT ===\n")
        f.write("R_G represents the aggregate topological friction/failure of the entire regional road network.\n")
        f.write("A value of 0 indicates perfect normal flow; higher values indicate network degradation.\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. PREDICTIVE THEORETICAL SCENARIOS\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Scenario':<25} | {'Global Risk (R_G)':<20}\n")
        f.write("-" * 50 + "\n")
        for name, val in zip(names_pred, values_pred):
            f.write(f"{name:<25} | {val:<20.4f}\n")
            
        f.write("\n\n2. DANA EMPIRICAL RECOVERY TIMELINE\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Date':<25} | {'Global Risk (R_G)':<20}\n")
        f.write("-" * 50 + "\n")
        for name, val in zip(names_dana, values_dana):
            f.write(f"{name:<25} | {val:<20.4f}\n")
            
    logging.info(f"Report saved to: {report_path.name}")
    logging.info("Workflow Complete.")

if __name__ == "__main__":
    main()