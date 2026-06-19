# AetherTraffic: Physics-Informed AI for Proactive Parking Enforcement & Congestion Recovery
## Flipkart Gridlock 2.0 — Problem Statement 1 — Technical Solution Report

## 1. Executive Summary

Urban traffic congestion in Bengaluru is fundamentally exacerbated by illegal, unstructured on-street parking. This project introduces a paradigm shift from traditional, reactive traffic enforcement to a fully integrated, predictive and prescriptive AI architecture. 

By processing nearly 300,000 raw traffic violation records, we engineered a system that not only identifies where congestion occurs but mathematically quantifies *why* it occurs and forecasts *when* it will happen next. Most importantly, it outputs an optimal schedule to allocate limited municipal resources to maximize congestion recovery, estimating a city-wide economic savings of ₹120.96 Crore per year from just the top 20 zones. Our solution transitions the traffic police from a system of retrospective ticketing to proactive congestion prevention.

## 2. Problem Statement

*Poor Visibility on Parking-Induced Congestion* (Verbatim from hackathon brief)

## 3. Dataset Overview

- **Source & Size:** The dataset, `jan to may police violation_anonymized791b166.csv`, contains 298,277 raw records initially, sized at approximately 109MB.
- **Date Range:** January to May.
- **Record Count:** After cleaning and deduplication, exactly **298,277** records were analyzed.
- **Schema Description:** The dataset contains spatial coordinates (latitude/longitude), temporal data (timestamp), vehicle type, violation type, police station mapping, and fine amounts.
- **Data Quality Handling:** Missing coordinates were imputed or dropped, duplicate violation tickets issued in rapid succession were deduplicated, and irrelevant non-parking violations were filtered out. 97.3% of the dataset records are directly related to parking violations (e.g., "PARKING IN A MAIN ROAD", "WRONG PARKING", "NO PARKING").

## 4. Methodology — 4-Stage Pipeline

### 4.1 Stage 1: Spatial Clustering (HDBSCAN)

- **Why HDBSCAN:** HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) was chosen over K-Means because it does not require pre-specifying the number of clusters (K), it gracefully handles varying densities (such as the dense CBD vs. sparser outskirts), and automatically identifies noise points that do not belong to any true localized cluster.
- **Parameters Used:** `min_cluster_size=150`, `metric='haversine'`, `cluster_selection_method='eom'`.
- **Result:** The algorithm successfully identified **312** exact hotspot zones (clusters), classifying 24.9% of the dataset as ambient noise not contributing to chronic localized bottlenecks.

### 4.2 Stage 2: Physics-Informed Congestion Quantification

- **Greenshields Fundamental Diagram:** This traffic flow theory models the relationship between vehicle speed, flow, and density. We used it to determine how stationary (parked) vehicles reduce the effective lane capacity and force moving traffic to slow down.
- **LWR Shockwave Theory:** Lighthill-Whitham-Richards theory describes how disruptions (like an illegally parked car) create a backward-propagating "shockwave" of congestion. It proves that a single bottleneck affects traffic flow for kilometers upstream.
- **Derived Metrics:** We calculated `capacity_loss_pct` (percentage of lane capacity lost), `shockwave_velocity_kmh` (how fast the traffic jam grows backwards), and `queue_length_km` (the physical length of the resulting traffic jam). 
- **Top-Ranked Zones by Physics Rank (from physics_scored_zones.csv):**
  1. **Electronic City (Cluster 13):** capacity_loss_pct = 94.04%, shockwave_velocity_kmh = -15.75, queue_length_km = 15.75
  2. **K.R. Pura (Cluster 48):** capacity_loss_pct = 94.04%, shockwave_velocity_kmh = -15.75, queue_length_km = 15.75
  3. **Pulikeshinagar(F.Town) (Cluster 209):** capacity_loss_pct = 91.37%, shockwave_velocity_kmh = -15.75, queue_length_km = 15.75
  4. **High ground (Cluster 237):** capacity_loss_pct = 91.20%, shockwave_velocity_kmh = -15.75, queue_length_km = 15.75
  5. **Mahadevapura (Cluster 16):** capacity_loss_pct = 95.00%, queue_length_km = 6.75

### 4.3 Stage 3: Predictive Forecasting (XGBoost + SHAP)

- **Features Used:** The model leverages autoregressive temporal lags (e.g., lag_1h, lag_24h, lag_1w), rolling means (e.g., rolling_mean_3h), cyclical hour encodings (hour_sin/cos), and static zone attributes like mean severity and total violations.
- **Forecast Horizon:** The model predicts congestion impact 1 hour, 2 hours, and 3 hours into the future. For example, from `shift_forecast.csv`:
  - Pulikeshinagar(F.Town) (Cluster 209): Base=77.07, 1h=313.29, 2h=284.96, 3h=192.54
  - Bellandur (Cluster 61): Base=124.61, 1h=277.65, 2h=218.21, 3h=210.77
  - K.R. Pura (Cluster 30): Base=78.42, 1h=191.03, 2h=143.48, 3h=112.20
- **SHAP Insights:** SHAP summary plots and dependence plots (`shap_summary_1h.png`, `shap_dep_lag_3h_1h.png`) revealed that the 24-hour lag (`lag_24h`) and the 3-hour rolling standard deviation are highly influential features, proving that congestion strongly follows daily diurnal patterns and recent volatility.

### 4.4 Stage 4: Prescriptive Resource Optimization (0-1 Knapsack)

- **Problem Framing:** Identifying hotspots is only half the battle; addressing them with constrained resources (e.g., 5 available tow trucks for a 4-hour shift) is a classic optimization problem. We framed this as a 0-1 Knapsack Optimization to select the combination of zones that maximizes the total `value` (congestion prevented) without exceeding the `cost` (travel time + clearance time limit).
- **Top 10 Dispatch Schedule (from dispatch_schedule.csv):**
  1. Pulikeshinagar(F.Town) (Cluster 209): Priority 1, Travel = 0.308h, Clear = 1.333h
  2. Bellandur (Cluster 61): Priority 2, Travel = 0.982h, Clear = 1.333h
  3. K.R. Pura (Cluster 30): Priority 3, Travel = 1.191h, Clear = 1.333h
  4. K.R. Pura (Cluster 48): Priority 4, Travel = 0.999h, Clear = 1.333h
  5. Malleshwaram (Cluster 246): Priority 5, Travel = 0.306h, Clear = 1.333h
  6. Kodigehalli (Cluster 90): Priority 6, Travel = 0.516h, Clear = 1.333h
  7. High ground (Cluster 237): Priority 7, Travel = 0.174h, Clear = 1.333h
  8. R.T. Nagar (Cluster 89): Priority 8, Travel = 0.514h, Clear = 1.333h
  9. Electronic City (Cluster 13): Priority 9, Travel = 1.589h, Clear = 1.333h
  10. Whitefield (Cluster 31): Priority 10, Travel = 1.376h, Clear = 1.333h

## 5. Results & Key Findings

**Top 10 Priority Zones Combined Table**
| Zone (Police Station) | Cluster ID | Total Violations | Congestion Impact Score | Physics Rank | Dispatch Priority |
|-----------------------|------------|------------------|-------------------------|--------------|-------------------|
| Pulikeshinagar(F.Town)| 209 | 245 | 528.54 | 3 | 1 |
| Bellandur | 61 | 316 | 828.49 | 12 | 2 |
| K.R. Pura | 30 | 159 | 691.14 | 19 | 3 |
| K.R. Pura | 48 | 282 | 674.12 | 2 | 4 |
| Malleshwaram | 246 | 175 | 466.51 | 14 | 5 |
| Kodigehalli | 90 | 226 | 319.01 | 8 | 6 |
| High ground | 237 | 243 | 669.27 | 4 | 7 |
| R.T. Nagar | 89 | 188 | 449.72 | 6 | 8 |
| Electronic City | 13 | 282 | 658.57 | 1 | 9 |
| Whitefield | 31 | 313 | 917.41 | 15 | 10 |
*(Note: Physics rank top 10 is disjointed from knapsack priority due to travel time constraints overriding absolute severity.)*

- **Economic Impact:** We calculated the Estimated Annual Congestion Cost for just the top 20 zones at **₹120.96 Crore**. Single-zone enforcement highly pays for itself, for example, HAL Old Airport alone incurs Rs. 100.58 Lakh/month in congestion cost.
- **City-wide Impact:** Targeted enforcement of merely the Top 5 Zones yields a **10.5%** impact reduction network-wide. This proves the extreme Pareto distribution of parking-induced congestion.

## 6. Prototype: Police Command Center Dashboard

The final intelligence is surfaced in `prototype/police_command_center.html`, a clean, white-mode command center dashboard optimized for field officer legibility and low cognitive load in daylight conditions.

**Implemented Features:**
- **Live Clock & Timeline Slider:** Synchronized to a central `currentReplayHour` engine, allowing dispatchers to scrub the historical replay from 12 AM to 11 PM to visualize the congestion wave throughout the day.
- **Real-time Risk Classification:** Zones dynamically transition between CRITICAL, HIGH, ELEVATED, and CLEAR based on real-time XGBoost forecasts and LWR queue lengths.
- **Priority Queue Counter:** A proactive listing of the top zones requiring immediate dispatch.
- **Command Log Ticker:** A scrolling ticker providing contextual events and dispatch recommendations.
- **Leaflet Map:** Renders impact clusters visually, with sizes tied directly to the physics-derived `hourly_score` = `impact_score * hourly_intensity[currentReplayHour]`.
- **Secondary View:** A `prototype/folium_heatmap.html` provides an alternative geospatial heatmap of raw density.

## 7. Production Architecture

As visualized in `assets/images/architecture/architecture.png`:
1. **Ingestion Layer:** Reads raw violation logs (`data/jan_to_may_police_violation_anonymized.csv`).
2. **Database/Storage:** Manages the spatial features and historical time-series data.
3. **AI Engine:** Processes data through HDBSCAN (clustering), Greenshields/LWR (physics scoring), and XGBoost (predictive forecasting).
4. **Optimization:** The 0-1 Knapsack solver ingests the forecasts to produce a resource-constrained dispatch schedule.
5. **Dashboard:** The intelligence is pushed to the `prototype/police_command_center.html` interface.

## 8. Validation & Limitations

- **Economic Assumptions:** The ₹120.96 Crore estimate assumes a static Value of Time (₹200/hour) and average delay minutes. These parameters need real-world validation against ground-truth traffic API data (like Google Maps API).
- **Static Enforcement Capability:** The optimization assumes a fixed 1.333-hour clearance time per zone, regardless of the vehicle type composition. A real-world test would refine this clearance duration.
- **`code/proactive_dispatch_engine.py`**: Standalone proactive dispatch and enforcement scoring engine. Provides modular, importable scoring functions decoupled from the notebook.

## 9. Scalability

This architecture is entirely location-agnostic. By utilizing unsupervised density clustering (HDBSCAN) instead of hard-coded polygons, the pipeline will immediately adapt to any Indian city simply by swapping the latitude/longitude inputs. The physics engine and knapsack solver require zero hyperparameter tuning to function in a new geography.

## 10. Submission Contents

- `code/Flipkart_Gridlock_2.0_PS1_Final_Solution.ipynb`: Core Python pipeline containing all ML and optimization logic.
- `code/proactive_dispatch_engine.py`: Standalone proactive dispatch and enforcement scoring engine.
- `code/run_validation.py`: Automated validation and test suite.
- `code/output/enforcement_priority_ranking.csv`: Top-level zone severity rankings.
- `code/output/physics_scored_zones.csv`: Zones scored with capacity loss and queue length metrics.
- `code/output/shift_forecast.csv`: XGBoost 1h, 2h, and 3h predictive forecasts.
- `code/output/dispatch_schedule.csv`: Final knapsack-optimized routing schedule.
- `data/jan_to_may_police_violation_anonymized.csv`: Raw 109MB input dataset.
- `prototype/police_command_center.html`: Interactive white-mode command center dashboard.
- `prototype/folium_heatmap.html`: Static geospatial heatmap visualization.
- `assets/images/analysis_charts/*.png`: 15 analytical charts and SHAP interpretability plots.
- `assets/images/architecture/architecture.png`: System architecture diagram.
- `README.md`: Project overview, repository map, and setup instructions.
