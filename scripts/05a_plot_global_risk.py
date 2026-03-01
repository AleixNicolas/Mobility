import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path

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
OUTPUT_PATH = PLOTS_DIR / "Risk_Factor.png"

# ===============================================================
# LABEL MAPPING
# ===============================================================
# Safely map the exact JSON keys to the desired plot labels
SCENARIO_MAPPING = {
    "10 yr": "High Prob.",
    "100 yr": "Med. Prob.",
    "500 yr": "Low Prob.",
    "DANA_31_10_2024": "31/10/2024",
    "DANA_03_11_2024": "03/11/2024",
    "DANA_05_11_2024": "05/11/2024",
    "DANA_06_11_2024": "06/11/2024",
    "DANA_08_11_2024": "08/11/2024"
}

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

    # Extract strictly mapped names and values
    names = []
    values = []
    
    for json_key, display_label in SCENARIO_MAPPING.items():
        if json_key in R:
            names.append(display_label)
            values.append(R[json_key])
        else:
            logging.warning(f"Scenario '{json_key}' missing from R_G.json")

    # Plotting
    logging.info("Generating bar plot...")
    plt.figure(figsize=(10, 6))
    plt.bar(names, values, color="#1f77b4", alpha=0.8, edgecolor="black")
    
    plt.ylim(0, 1)
    plt.ylabel('Global Risk ($R_{G}$)', fontsize=12)
    plt.title('Network Global Risk across Scenarios', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()

    # Save figure
    plt.savefig(OUTPUT_PATH, dpi=300)
    logging.info(f"Plot saved successfully to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()