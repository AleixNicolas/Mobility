# Flood Impact and Mobility Network Analysis Pipeline

This repository contains a 5-phase computational pipeline designed to assess the impact of severe flooding (such as the DANA 2024 event) on regional mobility, road network navigability, and critical service accessibility in Spain. 

The pipeline integrates official MITMA mobility data, INE demographics, EMSR flood footprints, and OSMnx-generated topological networks to compute granular isolation metrics and visualize structural network failures.

## Directory Architecture

The data directory is strictly ordered to reflect the chronological flow of data processing:

    data/
    ├── 01_raw_manual/                 # Externally sourced data (Flood polygons, POIs, Dictionaries)
    ├── 02_raw_downloaded/             # API-fetched data (Demographics, MITMA daily mobility)
    ├── 03_processed_shared/           # Validated mapping dictionaries used across all scripts
    ├── 04_routing_networks/           # Unperturbed baseline topological graphs and geometries
    ├── 05_scenario_models/            # Flood-intersected graphs, routing matrices, and global metrics
    └── 06_outputs/                    # Final visual deliverables (Maps and Plots)

---

## Script Execution & Data Flow

The codebase is divided into five chronological phases. Scripts should be executed sequentially.

### Phase 1: Data Acquisition & Alignment
*Objective: Fetch raw data and standardize mapping dictionaries.*

* **`01a_fetch_mobility.py`**
  * **Description:** Connects to the API to pull daily municipal mobility matrices.
  * **Inputs:** None (External API).
  * **Outputs:** Generates daily CSVs and spatial GeoJSONs in `02_raw_downloaded/mobility/`.
* **`01b_fetch_population.py`**
  * **Description:** Scrapes INE demographic baseline data for the target year.
  * **Inputs:** None (External Web).
  * **Outputs:** `ine_spain_population_2024.csv` in `02_raw_downloaded/demographics/`.
* **`01c_validate_mappings.py`**
  * **Description:** Master validation script. Reconciles manual municipality dictionaries with official MITMA IDs to ensure perfect spatial alignment.
  * **Inputs:** `affected_municipalities_dictionary.json` (01), `ine_spain_population_2024.csv` (02), and MITMA GeoJSON (02).
  * **Outputs:** `final_mitma_mapping.json` in `03_processed_shared/`.

### Phase 2: Baseline Environment Generation
*Objective: Build the static, unperturbed topological street network.*

* **`02a_build_street_network.py`**
  * **Description:** Downloads OSM street data, generates drivable routing graphs, assigns travel speeds, and embeds municipal metadata and POI attributes into specific nodes.
  * **Inputs:** `final_mitma_mapping.json` (03), MITMA GeoJSON (02), and POI GPKGs (01).
  * **Outputs:** `G_2nd.graphml`, `affected_area.gpkg`, and neighbor geometries in `04_routing_networks/`.

### Phase 3: Scenario Disruption
*Objective: Introduce flood geometries and slice the baseline network.*

* **`03a_generate_flood_zones.py`**
  * **Description:** Intersects the baseline graph with specific EMSR or modeled flood polygons. Severs inundated edges and isolates surviving safe graphs.
  * **Inputs:** Base `G_2nd.graphml` (04) and raw flood shapefiles (01).
  * **Outputs:** Scenario-specific `safe_roads.graphml`, `cut_roads.gpkg`, and `zone_flood.gpkg` filed inside `05_scenario_models/{scenario_name}/`.

### Phase 4: Computational Analysis
*Objective: Execute network routing algorithms to calculate isolation metrics.*

* **`04a_navigability_analysis.py`**
  * **Description:** Computes exhaustive origin-destination (OD) shortest paths between all municipal centers for both the baseline and all flood scenarios. Calculates the global risk scalar (R_G).
  * **Inputs:** `G_2nd.graphml` (04) and scenario `safe_roads.graphml` (05).
  * **Outputs:** `shortest_paths_{scenario}.json` in scenario folders, and `R_G.json` in `05_scenario_models/global_metrics/`.
* **`04b_service_access.py`**
  * **Description:** Routes every municipality to the geographically nearest service facility (e.g., Hospital, Police) under baseline and flooded conditions.
  * **Inputs:** Baseline and scenario graphs.
  * **Outputs:** `service_access_{scenario}.json` inside scenario folders.
* **`04c_edge_criticality.py`**
  * **Description:** Iteratively severs and restores individual edges to compute the marginal risk contribution (criticality) of specific road segments during recovery.
  * **Inputs:** Baseline matrix (04) and DANA matrix/cut roads (05).
  * **Outputs:** `edge_risks_NP.json` and `edge_risks_DANA.json` in `05_scenario_models/global_metrics/`.

### Phase 5: Visualization & Deliverables
*Objective: Render spatial maps and statistical plots from the analytical outputs.*

* **`05a_plot_global_risk.py`**
  * **Description:** Renders a bar chart comparing the aggregate R_G across all modeled scenarios.
  * **Inputs:** `R_G.json` (05).
  * **Outputs:** `Risk_Factor.png` in `06_outputs/plots_I/`.
* **`05b_plot_travel_times.py`**
  * **Description:** Generates dual-axis Kernel Density Estimation (KDE) distributions of surviving route travel times and total route destruction ratios.
  * **Inputs:** All `shortest_paths_{scenario}.json` files (04 & 05).
  * **Outputs:** `Travel_times_DANA.png/.pdf` in `06_outputs/plots_I/`.
* **`05c_map_flood_layers.py`**
  * **Description:** Spatial rendering of safe roads, severed roads, and flood polygons mapped over a static regional bounding box.
  * **Inputs:** `affected_area.gpkg` (04), `safe_roads.gpkg`, `cut_roads.gpkg`, and `zone_flood.gpkg` (05).
  * **Outputs:** `layer_map_{scenario}.png` in `06_outputs/maps_I/`.
* **`05d_map_municipal_risk.py`**
  * **Description:** Generates a choropleth map coloring municipalities by their bidirectional isolation risk factor.
  * **Inputs:** `affected_area.gpkg` (04) and baseline/scenario routing matrices (04 & 05).
  * **Outputs:** `municipality_risk_map_{scenario}.png` in `06_outputs/maps_I/`.
* **`05e_plot_mobility_trends.py`**
  * **Description:** Analyzes and plots daily shifts in true mobility volumes and shares compared to pre-disaster baselines.
  * **Inputs:** Raw daily mobility CSVs (02) and validated mapping (03).
  * **Outputs:** Time-series charts in `06_outputs/plots_M/`.
* **`05f_map_mobility_metrics.py`**
  * **Description:** Creates mobility heatmaps and extracts final aggregated mobility deviation metrics.
  * **Inputs:** Raw daily mobility CSVs (02) and validated mapping (03).
  * **Outputs:** Final `final_mobility_metrics.json` (03) and heatmaps in `06_outputs/maps_M/`.

---

## Installation & Environment

Due to the complex C-dependencies required by spatial routing libraries (GDAL, GEOS), we highly recommend using `conda` to build the environment via the `conda-forge` channel.

1. Clone the repository.
2. Create the environment:
   ```bash
   conda env create -f environment.yml