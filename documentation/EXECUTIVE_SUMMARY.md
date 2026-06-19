# AI-Driven Parking Intelligence System
## Flipkart Gridlock 2.0 | Problem Statement 1

### The Problem
Bengaluru's urban traffic flow is severely handicapped by unstructured, illegal on-street parking that acts as continuous micro-bottlenecks. The current enforcement strategy suffers from "Poor Visibility on Parking-Induced Congestion," relying on reactive ticketing rather than proactive network management.

### Our Solution
- **DETECT:** We deployed HDBSCAN geospatial clustering to automatically detect localized parking violation hotspots across the city from raw coordinate data.
- **QUANTIFY:** We mapped these coordinates to physical lane capacity loss and queue lengths using Greenshields Fundamental Diagram and LWR Shockwave Theory, then forecasted future impact using XGBoost.
- **PRESCRIBE:** We utilized a 0-1 Knapsack optimization algorithm to generate exact, travel-time-constrained dispatch schedules for limited municipal tow trucks.

### Key Results
| Metric | Result |
|--------|--------|
| **Records Processed** | 298,277 cleaned violations (109MB raw) |
| **Hotspot Zones Identified** | 312 exact enforcement clusters |
| **Top Zone (by Physics Score)** | Electronic City (94.04% capacity loss, 15.75 km queue) |
| **Economic Cost** | ₹120.96 Crore estimated annual congestion cost (Top 20 zones) |
| **Network Impact** | 10.5% city-wide impact reduction possible by enforcing just the Top 5 zones |
| **Chronic Offenders** | Extremely concentrated; single-zone enforcement (e.g., HAL Old Airport) pays for itself |

### How It Works
1. **Spatial Clustering (HDBSCAN):** Groups noisy GPS coordinates into 312 discrete enforcement zones.
2. **Physics-Informed Scoring:** Converts violation counts into actual traffic engineering metrics (capacity loss, shockwave queue lengths).
3. **Predictive Forecasting (XGBoost):** Looks 1-3 hours ahead to predict where congestion will form before it happens.
4. **Prescriptive Optimization (Knapsack):** Computes the optimal subset of zones to dispatch tow trucks to within a limited 4-hour shift constraint.
5. **Command Center Dashboard:** Surfaces the intelligence via a white-mode, daylight-readable dashboard with live risk badges and a historical replay engine.

### Top 5 Enforcement Zones
*(Ranked by 0-1 Knapsack Dispatch Priority for maximum capacity recovery)*
| Dispatch Priority | Zone (Police Station) | Impact Score | Recommended Action |
|-------------------|-----------------------|--------------|--------------------|
| 1 | Pulikeshinagar(F.Town) | 528.54 | Dispatch Tow Truck |
| 2 | Bellandur | 828.49 | Dispatch Tow Truck |
| 3 | K.R. Pura (Cluster 30) | 246.39 | Dispatch Tow Truck |
| 4 | K.R. Pura (Cluster 48) | 674.12 | Dispatch Tow Truck |
| 5 | Malleshwaram | 246.56 | Dispatch Tow Truck |

### Why This Wins
- **Causal, Not Correlative:** It doesn't just show a heatmap of tickets; it calculates the actual physical lane capacity lost using established traffic physics (LWR shockwave theory).
- **Predictive and Prescriptive:** It shifts operations from retrospective (looking at old tickets) to proactive (deploying trucks to where traffic is *about* to form).
- **Field-Ready Engineering:** The solution outputs a tangible `dispatch_schedule.csv` loaded into a clean, white-mode command center designed for actual daylight legibility by officers.

### Scalability
The architecture is entirely geometry-agnostic and relies on unsupervised density clustering rather than static city polygons. This allows the exact same pipeline to be deployed to any other city in India by simply feeding in new latitude and longitude violation data.
