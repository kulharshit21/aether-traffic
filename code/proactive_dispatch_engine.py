"""
================================================================================
PROACTIVE AI DISPATCH ENGINE
================================================================================
Upgrades the existing HDBSCAN-based parking congestion analytics system into
a Proactive AI Dispatch Engine with four advanced modules:

  Module 1: Physics-Informed Congestion Scoring (Greenshields + Shockwave)
  Module 2: Predictive Hotspot Forecasting (XGBoost time-series)
  Module 3: Explainable AI (SHAP integration)
  Module 4: Prescriptive Tow-Truck Dispatch Optimization (0-1 Knapsack)

Dependencies: pandas, numpy, xgboost, shap, matplotlib, seaborn
Compatibility: Accepts output from the existing HDBSCAN clustering pipeline.

References:
  [1] Greenshields, B.D. (1935). "A study of traffic capacity."
  [2] Daganzo, C.F. (1997). Fundamentals of Transportation and Traffic Operations.
  [3-5] Lighthill-Whitham-Richards (LWR) shockwave theory.
  [6-11] Gradient boosting for urban traffic forecasting.
  [12-17] SHAP (SHapley Additive exPlanations) for model interpretability.
  [18-20] Vehicle Routing Problem / Knapsack optimization.
================================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

# ============================================================
# Premium Plot Style (consistent with existing notebook)
# ============================================================
plt.rcParams.update({
    'figure.facecolor': '#0D1117', 'axes.facecolor': '#161B22',
    'axes.edgecolor': '#30363D', 'axes.labelcolor': '#C9D1D9',
    'text.color': '#C9D1D9', 'xtick.color': '#8B949E',
    'ytick.color': '#8B949E', 'grid.color': '#21262D',
    'grid.alpha': 0.6, 'font.family': 'sans-serif',
    'font.size': 11, 'axes.titlesize': 14, 'axes.titleweight': 'bold'
})
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
          '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
          '#F1948A', '#82E0AA', '#F8C471', '#AED6F1', '#D2B4DE']

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# MODULE 1: PHYSICS-INFORMED CONGESTION SCORING
# ============================================================================
# Uses the Greenshields Fundamental Diagram of Traffic Flow to replace the
# heuristic Composite Record Impact Score with a physically-grounded
# Capacity Loss Score for each HDBSCAN cluster.
#
# Greenshields linear speed-density model:
#   v(k) = v_f * (1 - k / k_j)
#
# where:
#   v_f  = free-flow speed (km/h)
#   k_j  = jam density (vehicles per km per lane)
#   k    = current density
#
# Traffic flow q = k * v(k) = k * v_f * (1 - k/k_j)
# Maximum capacity: q_max = v_f * k_j / 4  (at k = k_j / 2)
#
# Shockwave velocity (Lighthill-Whitham-Richards):
#   w = (q_B - q_A) / (k_B - k_A)
#
# where A = upstream free-flow state, B = congested state downstream
# of the parking bottleneck.
# ============================================================================

def compute_greenshields_flow(density, v_free, k_jam):
    """
    Compute traffic flow using Greenshields' linear speed-density model.
    
    Parameters
    ----------
    density : float or np.array
        Current traffic density (vehicles per km per lane).
    v_free : float
        Free-flow speed (km/h). Typical urban arterial: 40-60 km/h.
    k_jam : float
        Jam density (vehicles per km per lane). Typical: 120-180 veh/km/lane.
    
    Returns
    -------
    tuple : (speed, flow)
        speed in km/h, flow in vehicles/hour/lane
    
    References
    ----------
    Greenshields, B.D. (1935). "A study of traffic capacity."
    """
    # Clamp density to [0, k_jam] to avoid negative speeds
    density = np.clip(density, 0, k_jam)
    speed = v_free * (1.0 - density / k_jam)
    flow = density * speed
    return speed, flow


def compute_shockwave_velocity(q_upstream, k_upstream, q_downstream, k_downstream):
    """
    Compute shockwave propagation velocity using the LWR model.
    
    The shockwave velocity describes how fast a queue propagates upstream
    from a bottleneck (e.g., illegally parked vehicles reducing capacity).
    
    Parameters
    ----------
    q_upstream : float
        Flow rate in the upstream free-flow region (veh/hr/lane).
    k_upstream : float
        Density in the upstream free-flow region (veh/km/lane).
    q_downstream : float
        Flow rate in the congested region downstream of bottleneck.
    k_downstream : float
        Density in the congested region downstream of bottleneck.
    
    Returns
    -------
    float : Shockwave velocity (km/h). Negative = propagates upstream.
    
    References
    ----------
    Lighthill & Whitham (1955); Richards (1956).
    """
    dk = k_downstream - k_upstream
    if abs(dk) < 1e-9:
        return 0.0
    return (q_downstream - q_upstream) / dk


def estimate_effective_lane_reduction(violation_count, road_width_lanes=2):
    """
    Estimate the fraction of road capacity lost due to illegally parked
    vehicles acting as spatial bottlenecks.
    
    Heuristic: Each parked vehicle effectively blocks a fraction of a lane.
    Multiple violations in the same zone compound the blockage.
    
    Parameters
    ----------
    violation_count : int
        Number of violations (parked vehicles) in the cluster.
    road_width_lanes : int
        Assumed number of lanes on the road segment (default 2).
    
    Returns
    -------
    float : Fraction of capacity blocked, clipped to [0, 0.95].
    """
    # Each parked vehicle blocks approximately 0.3-0.5 of a lane equivalent
    # (accounting for the "moving bottleneck" effect and driver reaction).
    # We use a diminishing-returns (logistic) model so that more violations
    # increase blockage but saturate near total blockage.
    #
    # lane_equivalents_blocked = violation_count * 0.4 / road_width_lanes
    # fraction = 1 - exp(-0.05 * lane_equivalents_blocked)
    lane_eq_blocked = violation_count * 0.4 / road_width_lanes
    fraction = 1.0 - np.exp(-0.05 * lane_eq_blocked)
    return np.clip(fraction, 0.0, 0.95)


def physics_informed_congestion_score(zone_df,
                                       v_free=45.0,
                                       k_jam=150.0,
                                       road_lanes=2,
                                       observation_hours=1.0):
    """
    MODULE 1 MAIN FUNCTION
    
    Recalculates congestion impact for each HDBSCAN cluster using the
    Greenshields fundamental diagram and shockwave theory.
    
    Parameters
    ----------
    zone_df : pd.DataFrame
        The enforcement_priority_ranking DataFrame with columns:
        cluster, total_violations, congestion_impact_score, mean_severity,
        center_lat, center_lon, etc.
    v_free : float
        Free-flow speed in km/h (default 45 for Bengaluru urban arterials).
    k_jam : float
        Jam density in vehicles/km/lane (default 150).
    road_lanes : int
        Assumed average lane count (default 2).
    observation_hours : float
        Time window for the capacity loss calculation (default 1 hour).
    
    Returns
    -------
    pd.DataFrame : zone_df augmented with:
        - capacity_free : Maximum capacity under free-flow (veh/hr/lane)
        - capacity_bottleneck : Reduced capacity due to parking bottleneck
        - capacity_loss_score : Absolute capacity lost (veh/hr/lane)
        - capacity_loss_pct : Percentage of capacity lost
        - shockwave_velocity : Upstream queue propagation speed (km/h)
        - queue_length_km : Estimated queue length over observation window
    """
    print("\n" + "="*70)
    print("MODULE 1: PHYSICS-INFORMED CONGESTION SCORING")
    print("="*70)
    print(f"  Greenshields Parameters: v_free={v_free} km/h, k_jam={k_jam} veh/km/lane")
    print(f"  Road Lanes (assumed): {road_lanes}")
    print(f"  Observation Window: {observation_hours} hour(s)")
    
    # Maximum capacity under Greenshields: q_max = v_f * k_j / 4
    # This occurs at the critical density k_c = k_j / 2
    k_critical = k_jam / 2.0
    _, q_max = compute_greenshields_flow(k_critical, v_free, k_jam)
    
    print(f"\n  Free-flow Capacity (q_max): {q_max:.1f} veh/hr/lane")
    print(f"  Critical Density (k_c): {k_critical:.1f} veh/km/lane")
    
    results = []
    
    for _, zone in zone_df.iterrows():
        violations = zone['total_violations']
        
        # Step 1: Estimate the effective capacity reduction fraction
        blockage_fraction = estimate_effective_lane_reduction(violations, road_lanes)
        
        # Step 2: Compute bottleneck capacity
        # The bottleneck reduces the effective number of lanes, so the
        # maximum throughput of the bottleneck section drops.
        effective_lanes_remaining = road_lanes * (1.0 - blockage_fraction)
        capacity_bottleneck = q_max * effective_lanes_remaining
        capacity_free = q_max * road_lanes
        
        # Step 3: Capacity loss (absolute and percentage)
        capacity_loss = capacity_free - capacity_bottleneck
        capacity_loss_pct = (capacity_loss / capacity_free) * 100.0 if capacity_free > 0 else 0.0
        
        # Step 4: Shockwave analysis
        # Upstream state A: free-flow at some typical density < k_critical
        # We assume upstream demand is at 70% of capacity (typical peak hour).
        k_upstream = k_critical * 0.7  # vehicles arriving at 70% of critical density
        _, q_upstream = compute_greenshields_flow(k_upstream, v_free, k_jam)
        q_upstream *= road_lanes  # total flow across all lanes
        
        # Downstream state B: congested. The bottleneck creates a queue.
        # Vehicles queue up at a density higher than critical.
        # The downstream flow equals the bottleneck capacity.
        q_downstream = capacity_bottleneck
        # Solve for k_downstream from q = k * v_f * (1 - k/k_j)
        # This is a quadratic: v_f * k - (v_f/k_j)*k^2 = q_downstream/road_lanes
        # => (v_f/k_j)*k^2 - v_f*k + q_downstream/road_lanes = 0
        a_coeff = v_free / k_jam
        b_coeff = -v_free
        c_coeff = q_downstream / max(effective_lanes_remaining, 0.01)
        
        discriminant = b_coeff**2 - 4 * a_coeff * c_coeff
        if discriminant >= 0:
            # Take the larger root (congested branch of fundamental diagram)
            k_downstream = (-b_coeff + np.sqrt(discriminant)) / (2 * a_coeff)
        else:
            # If discriminant < 0, demand exceeds capacity → full jam
            k_downstream = k_jam
        
        # Compute shockwave velocity (per-lane basis)
        _, q_up_per_lane = compute_greenshields_flow(k_upstream, v_free, k_jam)
        _, q_dn_per_lane = compute_greenshields_flow(k_downstream, v_free, k_jam)
        
        w = compute_shockwave_velocity(q_up_per_lane, k_upstream,
                                       q_dn_per_lane, k_downstream)
        
        # Queue length = |w| * observation_hours (how far the queue propagates)
        queue_length = abs(w) * observation_hours
        
        # Step 5: Composite physics-informed score
        # Combines capacity loss with shockwave propagation severity
        # and zone-specific severity weighting from the original heuristic.
        severity_multiplier = zone.get('mean_severity', 1.0)
        physics_score = capacity_loss * (1.0 + queue_length / 5.0) * severity_multiplier
        
        results.append({
            'cluster': zone['cluster'],
            'capacity_free': round(capacity_free, 2),
            'capacity_bottleneck': round(capacity_bottleneck, 2),
            'capacity_loss_score': round(physics_score, 2),
            'capacity_loss_pct': round(capacity_loss_pct, 2),
            'shockwave_velocity_kmh': round(w, 3),
            'queue_length_km': round(queue_length, 3),
        })
    
    results_df = pd.DataFrame(results)
    
    # Merge back into the original zone_df
    output = zone_df.merge(results_df, on='cluster', how='left')
    
    # Re-rank zones by the new physics-informed score
    output = output.sort_values('capacity_loss_score', ascending=False).reset_index(drop=True)
    output['physics_rank'] = range(1, len(output) + 1)
    
    print(f"\n  ✅ Physics-informed scores computed for {len(output)} zones.")
    print(f"  Top 5 zones by Capacity Loss Score:")
    top5 = output.head(5)
    for _, z in top5.iterrows():
        print(f"    Rank {z['physics_rank']}: Cluster {int(z['cluster'])} "
              f"({z.get('police_station','N/A')}) — "
              f"C_loss={z['capacity_loss_score']:.1f}, "
              f"C_loss%={z['capacity_loss_pct']:.1f}%, "
              f"Queue={z['queue_length_km']:.2f} km")
    
    return output


# ============================================================================
# MODULE 2: PREDICTIVE HOTSPOT FORECASTING (TIME-SERIES)
# ============================================================================
# Builds a multivariate time-series per cluster, engineers lag features and
# cyclical temporal features, then trains an XGBoost regressor to forecast
# the Capacity_Loss_Score 1–3 hours into the future.
# ============================================================================

def build_cluster_timeseries(df, zone_physics_df, time_col='violation_date_time',
                              cluster_col='cluster_label'):
    """
    Convert per-violation records into an aggregated hourly time-series
    per HDBSCAN cluster. Each row = (cluster, date, hour) with aggregated
    features.
    
    Parameters
    ----------
    df : pd.DataFrame
        The cleaned violation-level DataFrame with columns including:
        violation_date_time, hour, cluster_label, record_impact,
        severity_weight, poi_weight, persistence_score, validation_weight.
    zone_physics_df : pd.DataFrame
        Output of Module 1 with capacity_loss_score per cluster.
    time_col : str
        Name of the datetime column.
    cluster_col : str
        Name of the cluster assignment column.
    
    Returns
    -------
    pd.DataFrame : Hourly time-series indexed by (cluster, date, hour).
    """
    print("\n  Building hourly time-series per cluster...")
    
    # Filter out noise points (cluster == -1)
    df_valid = df[df[cluster_col] >= 0].copy()
    
    # Ensure datetime
    df_valid[time_col] = pd.to_datetime(df_valid[time_col])
    df_valid['ts_date'] = df_valid[time_col].dt.date
    df_valid['ts_hour'] = df_valid[time_col].dt.hour
    df_valid['ts_dayofweek'] = df_valid[time_col].dt.dayofweek  # 0=Mon, 6=Sun
    
    # Aggregate per (cluster, date, hour)
    agg = df_valid.groupby([cluster_col, 'ts_date', 'ts_hour']).agg(
        violation_count=('record_impact', 'count'),
        total_impact=('record_impact', 'sum'),
        mean_severity=('severity_weight', 'mean'),
        mean_poi=('poi_weight', 'mean'),
        mean_persistence=('persistence_score', 'mean'),
        mean_validation=('validation_weight', 'mean'),
        dayofweek=('ts_dayofweek', 'first'),
    ).reset_index()
    
    # Merge the physics-informed capacity_loss_score as a static zone attribute
    # (scaled by hourly violation density for a time-varying signal)
    zone_map = zone_physics_df.set_index('cluster')['capacity_loss_score'].to_dict()
    agg['zone_capacity_loss'] = agg[cluster_col].map(zone_map).fillna(0)
    
    # The target: hourly capacity loss = zone_capacity_loss * (violation_count / zone_total_violations)
    zone_total = zone_physics_df.set_index('cluster')['total_violations'].to_dict()
    agg['zone_total_violations'] = agg[cluster_col].map(zone_total).fillna(1)
    agg['hourly_capacity_loss'] = (
        agg['zone_capacity_loss'] * agg['violation_count'] / agg['zone_total_violations']
    )
    
    print(f"  ✅ Time-series built: {len(agg)} rows across "
          f"{agg[cluster_col].nunique()} clusters.")
    
    return agg


def engineer_temporal_features(ts_df, cluster_col='cluster_label'):
    """
    Engineer lag features and cyclical temporal encodings for the
    time-series DataFrame.
    
    Features added:
        - hour_sin, hour_cos : Cyclical encoding of hour-of-day
        - dow_sin, dow_cos   : Cyclical encoding of day-of-week
        - lag_1h, lag_2h, lag_3h : Target value at t-1, t-2, t-3
        - lag_24h            : Target value at t-24 (same hour yesterday)
        - rolling_mean_3h    : 3-hour rolling mean of target
        - rolling_std_3h     : 3-hour rolling standard deviation of target
    
    Parameters
    ----------
    ts_df : pd.DataFrame
        Output of build_cluster_timeseries().
    cluster_col : str
        Name of the cluster column.
    
    Returns
    -------
    pd.DataFrame : Augmented with temporal features, NaN rows dropped.
    """
    print("\n  Engineering temporal features (lags, cyclical, rolling)...")
    
    df = ts_df.copy()
    
    # Cyclical encoding of hour (period = 24)
    df['hour_sin'] = np.sin(2 * np.pi * df['ts_hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['ts_hour'] / 24.0)
    
    # Cyclical encoding of day-of-week (period = 7)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7.0)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7.0)
    
    # Sort by cluster and time for correct lag computation
    df = df.sort_values([cluster_col, 'ts_date', 'ts_hour']).reset_index(drop=True)
    
    target = 'hourly_capacity_loss'
    
    # Lag features per cluster
    for lag in [1, 2, 3, 24]:
        col_name = f'lag_{lag}h'
        df[col_name] = df.groupby(cluster_col)[target].shift(lag)
    
    # Rolling statistics per cluster
    df['rolling_mean_3h'] = (
        df.groupby(cluster_col)[target]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df['rolling_std_3h'] = (
        df.groupby(cluster_col)[target]
        .transform(lambda x: x.rolling(3, min_periods=1).std().fillna(0))
    )
    
    # Drop rows where critical lag features are NaN
    before = len(df)
    df = df.dropna(subset=['lag_1h']).reset_index(drop=True)
    # Fill remaining NaNs (e.g., lag_24h for first day) with 0
    df = df.fillna(0)
    after = len(df)
    
    print(f"  ✅ Features engineered. Rows: {before} → {after} (after dropping NaN lags).")
    
    return df


def train_xgboost_forecaster(ts_features_df, forecast_horizons=[1, 2, 3],
                              cluster_col='cluster_label'):
    """
    Train an XGBoost Gradient Boosting Regressor to forecast
    hourly_capacity_loss at horizons of 1, 2, and 3 hours.
    
    Uses a time-based train/test split (last 20% of data for testing).
    
    Parameters
    ----------
    ts_features_df : pd.DataFrame
        Output of engineer_temporal_features().
    forecast_horizons : list of int
        Forecast horizons in hours (default [1, 2, 3]).
    cluster_col : str
        Name of the cluster column.
    
    Returns
    -------
    dict : {
        'models': {horizon: trained_xgb_model},
        'feature_names': list of feature column names,
        'test_data': pd.DataFrame of test set with predictions,
        'metrics': {horizon: {'rmse': float, 'mae': float, 'r2': float}}
    }
    """
    try:
        import xgboost as xgb
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'xgboost', '-q'])
        import xgboost as xgb
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    print("\n" + "="*70)
    print("MODULE 2: PREDICTIVE HOTSPOT FORECASTING")
    print("="*70)
    
    feature_cols = [
        'violation_count', 'total_impact', 'mean_severity', 'mean_poi',
        'mean_persistence', 'mean_validation', 'zone_capacity_loss',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'lag_1h', 'lag_2h', 'lag_3h', 'lag_24h',
        'rolling_mean_3h', 'rolling_std_3h'
    ]
    
    target = 'hourly_capacity_loss'
    
    # Time-based split: 80% train, 20% test
    df = ts_features_df.sort_values(['ts_date', 'ts_hour']).reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()
    
    print(f"\n  Training samples: {len(train_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {target}")
    
    models = {}
    metrics = {}
    
    for h in forecast_horizons:
        print(f"\n  --- Training XGBoost for {h}-hour forecast ---")
        
        # Create shifted target for this horizon
        train_y = train_df.groupby(cluster_col)[target].shift(-h)
        valid_mask = train_y.notna()
        
        X_train = train_df.loc[valid_mask, feature_cols]
        y_train = train_y[valid_mask]
        
        test_y = test_df.groupby(cluster_col)[target].shift(-h)
        valid_test_mask = test_y.notna()
        
        X_test = test_df.loc[valid_test_mask, feature_cols]
        y_test = test_y[valid_test_mask]
        
        if len(X_train) == 0 or len(X_test) == 0:
            print(f"    ⚠ Insufficient data for {h}-hour horizon. Skipping.")
            continue
        
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  verbose=False)
        
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        models[h] = model
        metrics[h] = {'rmse': round(rmse, 4), 'mae': round(mae, 4), 'r2': round(r2, 4)}
        
        test_df.loc[valid_test_mask, f'pred_{h}h'] = y_pred
        
        print(f"    RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
    
    print(f"\n  ✅ XGBoost models trained for horizons: {list(models.keys())}")
    
    return {
        'models': models,
        'feature_names': feature_cols,
        'test_data': test_df,
        'metrics': metrics
    }


def generate_shift_forecast(models_dict, ts_features_df,
                             zone_physics_df,
                             cluster_col='cluster_label',
                             top_n=20):
    """
    Generate a forecast of Predicted_Impact_Scores for the upcoming
    enforcement shift (next 1-3 hours) for the top-N priority zones.
    
    Parameters
    ----------
    models_dict : dict
        Output of train_xgboost_forecaster().
    ts_features_df : pd.DataFrame
        The full time-series features DataFrame.
    zone_physics_df : pd.DataFrame
        The physics-scored zone DataFrame.
    cluster_col : str
        Cluster column name.
    top_n : int
        Number of top clusters to forecast (default 20).
    
    Returns
    -------
    pd.DataFrame : Forecast table with predicted impact scores per horizon.
    """
    print("\n  Generating shift forecast for top zones...")
    
    feature_cols = models_dict['feature_names']
    models = models_dict['models']
    
    # Get the latest observation per cluster (most recent hour)
    latest = (ts_features_df
              .sort_values(['ts_date', 'ts_hour'])
              .groupby(cluster_col)
              .tail(1)
              .reset_index(drop=True))
    
    # Focus on top-N zones by physics rank
    top_clusters = zone_physics_df.head(top_n)['cluster'].values
    latest_top = latest[latest[cluster_col].isin(top_clusters)].copy()
    
    if len(latest_top) == 0:
        print("    ⚠ No matching data for top zones.")
        return pd.DataFrame()
    
    forecasts = []
    for _, row in latest_top.iterrows():
        cluster_id = int(row[cluster_col])
        zone_info = zone_physics_df[zone_physics_df['cluster'] == cluster_id]
        station = zone_info['police_station'].values[0] if len(zone_info) > 0 else 'Unknown'
        
        entry = {
            'cluster': cluster_id,
            'police_station': station,
            'current_capacity_loss': round(row.get('hourly_capacity_loss', 0), 2),
        }
        
        X = row[feature_cols].values.reshape(1, -1)
        X = pd.DataFrame(X, columns=feature_cols).astype(float)
        
        for h, model in models.items():
            pred = float(model.predict(X)[0])
            entry[f'predicted_{h}h'] = round(max(pred, 0), 2)
        
        forecasts.append(entry)
    
    forecast_df = pd.DataFrame(forecasts)
    
    # Sort by worst predicted impact (use 1h forecast as primary sort)
    if 'predicted_1h' in forecast_df.columns:
        forecast_df = forecast_df.sort_values('predicted_1h', ascending=False).reset_index(drop=True)
    
    print(f"  ✅ Forecast generated for {len(forecast_df)} zones.")
    
    return forecast_df


# ============================================================================
# MODULE 3: EXPLAINABLE AI (XAI) — SHAP INTEGRATION
# ============================================================================
# Integrates SHAP (SHapley Additive exPlanations) to interpret the XGBoost
# model. Generates local explanations for the top predicted hotspots and
# global summary/dependency plots for the dashboard.
# ============================================================================

def explain_predictions_shap(models_dict, ts_features_df,
                              forecast_df,
                              cluster_col='cluster_label',
                              horizon=1,
                              top_n_explain=10):
    """
    MODULE 3 MAIN FUNCTION
    
    Use SHAP TreeExplainer to generate:
      1. Local explanations for the top-N predicted hotspots
      2. Global SHAP summary plot
      3. SHAP dependence plots for top features
    
    Parameters
    ----------
    models_dict : dict
        Output of train_xgboost_forecaster().
    ts_features_df : pd.DataFrame
        Full time-series features DataFrame.
    forecast_df : pd.DataFrame
        Output of generate_shift_forecast().
    cluster_col : str
        Cluster column name.
    horizon : int
        Which forecast horizon's model to explain (default 1).
    top_n_explain : int
        How many top hotspots to generate local explanations for.
    
    Returns
    -------
    dict : {
        'shap_values': np.array,
        'expected_value': float,
        'feature_names': list,
        'local_explanations': list of dicts,
        'X_explain': pd.DataFrame
    }
    """
    try:
        import shap  # type: ignore
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap', '-q'])
        import shap  # type: ignore
    
    print("\n" + "="*70)
    print("MODULE 3: EXPLAINABLE AI (SHAP)")
    print("="*70)
    
    if horizon not in models_dict['models']:
        print(f"  ⚠ No model found for horizon={horizon}h. Available: "
              f"{list(models_dict['models'].keys())}")
        return None
    
    model = models_dict['models'][horizon]
    feature_cols = models_dict['feature_names']
    
    # Build the explanation dataset: latest observation per cluster,
    # filtered to the top clusters from the forecast
    latest = (ts_features_df
              .sort_values(['ts_date', 'ts_hour'])
              .groupby(cluster_col)
              .tail(1)
              .reset_index(drop=True))
    
    top_clusters = forecast_df.head(top_n_explain)['cluster'].values
    X_explain = latest[latest[cluster_col].isin(top_clusters)][feature_cols].copy()
    
    if len(X_explain) == 0:
        print("  ⚠ No data to explain.")
        return None
    
    X_explain = X_explain.reset_index(drop=True)
    
    print(f"\n  Using SHAP TreeExplainer on {horizon}-hour XGBoost model...")
    print(f"  Explaining {len(X_explain)} hotspot predictions.\n")
    
    # SHAP TreeExplainer (exact, fast for tree-based models)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_explain)
    expected_value = explainer.expected_value
    
    # --- Local Explanations ---
    local_explanations = []
    for i in range(min(top_n_explain, len(X_explain))):
        cluster_id = int(top_clusters[i]) if i < len(top_clusters) else -1
        sv = shap_values[i]
        total_abs = np.sum(np.abs(sv))
        
        contributions = {}
        for j, fname in enumerate(feature_cols):
            pct = (abs(sv[j]) / total_abs * 100) if total_abs > 0 else 0
            contributions[fname] = {
                'shap_value': round(float(sv[j]), 4),
                'contribution_pct': round(pct, 1)
            }
        
        # Sort by absolute contribution
        sorted_contribs = sorted(contributions.items(),
                                  key=lambda x: abs(x[1]['shap_value']),
                                  reverse=True)
        
        explanation = {
            'cluster': cluster_id,
            'predicted_value': round(float(expected_value + np.sum(sv)), 4),
            'base_value': round(float(expected_value), 4),
            'top_drivers': sorted_contribs[:5]
        }
        local_explanations.append(explanation)
        
        # Print human-readable explanation
        print(f"  🔍 Cluster {cluster_id}:")
        print(f"     Predicted {horizon}h Impact = {explanation['predicted_value']:.2f}")
        print(f"     Base value (avg) = {explanation['base_value']:.2f}")
        print(f"     Top contributing features:")
        for fname, info in sorted_contribs[:5]:
            direction = "↑" if info['shap_value'] > 0 else "↓"
            print(f"       {direction} {fname}: {info['contribution_pct']:.1f}% "
                  f"(SHAP={info['shap_value']:+.4f})")
        print()
    
    # --- Generate SHAP Plots ---
    try:
        # Summary plot (beeswarm)
        fig_summary, ax_summary = plt.subplots(figsize=(12, 8))
        fig_summary.patch.set_facecolor('#0D1117')
        shap.summary_plot(shap_values, X_explain, feature_names=feature_cols,
                          show=False, plot_size=None)
        plt.title(f'SHAP Summary — {horizon}h Forecast Model',
                  fontsize=16, fontweight='bold', color='#C9D1D9', pad=15)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/shap_summary_{horizon}h.png', dpi=150,
                    bbox_inches='tight', facecolor='#0D1117')
        plt.close()
        print(f"  📊 SHAP summary plot saved: {OUTPUT_DIR}/shap_summary_{horizon}h.png")
    except Exception as e:
        print(f"  ⚠ Could not generate SHAP summary plot: {e}")
    
    try:
        # Bar plot (mean absolute SHAP)
        fig_bar, ax_bar = plt.subplots(figsize=(12, 8))
        fig_bar.patch.set_facecolor('#0D1117')
        shap.summary_plot(shap_values, X_explain, feature_names=feature_cols,
                          plot_type='bar', show=False, plot_size=None)
        plt.title(f'SHAP Feature Importance — {horizon}h Forecast',
                  fontsize=16, fontweight='bold', color='#C9D1D9', pad=15)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/shap_importance_{horizon}h.png', dpi=150,
                    bbox_inches='tight', facecolor='#0D1117')
        plt.close()
        print(f"  📊 SHAP importance plot saved: {OUTPUT_DIR}/shap_importance_{horizon}h.png")
    except Exception as e:
        print(f"  ⚠ Could not generate SHAP importance plot: {e}")
    
    # Dependence plots for top 3 features by mean |SHAP|
    try:
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        top_3_idx = np.argsort(mean_abs_shap)[-3:][::-1]
        
        for rank, idx in enumerate(top_3_idx):
            fname = feature_cols[idx]
            fig_dep, ax_dep = plt.subplots(figsize=(10, 6))
            fig_dep.patch.set_facecolor('#0D1117')
            shap.dependence_plot(idx, shap_values, X_explain,
                                 feature_names=feature_cols,
                                 show=False, ax=ax_dep)
            ax_dep.set_title(f'SHAP Dependence: {fname} ({horizon}h)',
                             fontsize=14, fontweight='bold', color='#C9D1D9')
            plt.tight_layout()
            plt.savefig(f'{OUTPUT_DIR}/shap_dep_{fname}_{horizon}h.png', dpi=150,
                        bbox_inches='tight', facecolor='#0D1117')
            plt.close()
            print(f"  📊 SHAP dependence plot saved: {OUTPUT_DIR}/shap_dep_{fname}_{horizon}h.png")
    except Exception as e:
        print(f"  ⚠ Could not generate SHAP dependence plots: {e}")
    
    print(f"\n  ✅ SHAP explanations generated for {len(local_explanations)} hotspots.")
    
    return {
        'shap_values': shap_values,
        'expected_value': expected_value,
        'feature_names': feature_cols,
        'local_explanations': local_explanations,
        'X_explain': X_explain
    }


# ============================================================================
# MODULE 4: PRESCRIPTIVE AI — TOW-TRUCK DISPATCH OPTIMIZATION
# ============================================================================
# Formulates the tow-truck deployment as a 0-1 Knapsack problem.
#
# Objective: Maximize total "Congestion Impact Recovered"
#   (the sum of predicted capacity loss scores of cleared clusters).
#
# Constraints:
#   - Finite tow-truck resource hours (W)
#   - Maximum vehicle carrying capacity (Q) per truck
#   - Estimated travel + clearance time per zone
# ============================================================================

def estimate_travel_time(lat1, lon1, lat2, lon2, avg_speed_kmh=25.0):
    """
    Estimate travel time between two points using the Haversine distance
    and an assumed average travel speed for urban Bengaluru.
    
    Parameters
    ----------
    lat1, lon1 : float
        Origin coordinates.
    lat2, lon2 : float
        Destination coordinates.
    avg_speed_kmh : float
        Average urban travel speed (default 25 km/h for Bengaluru).
    
    Returns
    -------
    float : Estimated travel time in hours.
    """
    R = 6371.0  # Earth radius in km
    
    lat1_r, lat2_r = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_km = R * c
    
    return distance_km / avg_speed_kmh


def estimate_clearance_time(violation_count, vehicles_per_hour=6):
    """
    Estimate time to clear a zone based on its violation count.
    
    Parameters
    ----------
    violation_count : int
        Number of violations to clear.
    vehicles_per_hour : int
        Average clearance rate (vehicles towed/cleared per hour per truck).
    
    Returns
    -------
    float : Estimated clearance time in hours.
    """
    return violation_count / vehicles_per_hour


def knapsack_dispatch_optimization(forecast_df, zone_physics_df,
                                    total_truck_hours=24.0,
                                    num_trucks=5,
                                    truck_capacity=8,
                                    clearance_rate=6,
                                    depot_lat=12.97,
                                    depot_lon=77.59,
                                    horizon_key='predicted_1h'):
    """
    MODULE 4 MAIN FUNCTION
    
    Solve the tow-truck dispatch problem as a 0-1 Knapsack:
    
    Maximize: sum(predicted_capacity_loss[i] * x[i]) for selected zones i
    Subject to:
      sum(cost[i] * x[i]) <= total_truck_hours
      x[i] ∈ {0, 1}
    
    where cost[i] = travel_time_to_zone[i] + clearance_time[i] + return_time[i]
    
    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of generate_shift_forecast().
    zone_physics_df : pd.DataFrame
        Physics-scored zone DataFrame.
    total_truck_hours : float
        Total available resource hours (W).
    num_trucks : int
        Number of tow trucks available (affects total capacity).
    truck_capacity : int
        Max vehicles each truck can carry per trip.
    clearance_rate : int
        Vehicles cleared per hour per truck.
    depot_lat, depot_lon : float
        Coordinates of the tow-truck depot/dispatch center.
    horizon_key : str
        Which forecast column to use as the "value" (default 'predicted_1h').
    
    Returns
    -------
    dict : {
        'schedule': pd.DataFrame (prioritized dispatch plan),
        'total_impact_recovered': float,
        'total_hours_used': float,
        'total_hours_available': float,
        'zones_cleared': int,
        'zones_skipped': int
    }
    """
    print("\n" + "="*70)
    print("MODULE 4: PRESCRIPTIVE AI — TOW-TRUCK DISPATCH OPTIMIZATION")
    print("="*70)
    print(f"  Total truck hours available (W): {total_truck_hours}")
    print(f"  Number of trucks: {num_trucks}")
    print(f"  Truck capacity: {truck_capacity} vehicles/trip")
    print(f"  Clearance rate: {clearance_rate} vehicles/hr/truck")
    print(f"  Depot location: ({depot_lat}, {depot_lon})")
    print(f"  Optimization target: '{horizon_key}'")
    
    if horizon_key not in forecast_df.columns:
        print(f"  ⚠ Column '{horizon_key}' not found in forecast data. "
              f"Using 'current_capacity_loss' instead.")
        horizon_key = 'current_capacity_loss'
    
    # Build the items for the knapsack
    items = []
    for _, row in forecast_df.iterrows():
        cluster_id = int(row['cluster'])
        zone_info = zone_physics_df[zone_physics_df['cluster'] == cluster_id]
        
        if len(zone_info) == 0:
            continue
        
        zone = zone_info.iloc[0]
        zone_lat = zone['center_lat']
        zone_lon = zone['center_lon']
        violations = int(zone['total_violations'])
        station = zone.get('police_station', 'Unknown')
        
        # Value = predicted capacity loss to be recovered
        value = max(float(row.get(horizon_key, 0)), 0.0)
        
        # Cost = travel time (round trip) + clearance time
        travel_one_way = estimate_travel_time(depot_lat, depot_lon, zone_lat, zone_lon)
        
        # Clearance time: we can deploy multiple trucks to a zone
        # but each truck handles 'truck_capacity' vehicles at 'clearance_rate'
        vehicles_to_clear = min(violations, num_trucks * truck_capacity)
        clearance_time = vehicles_to_clear / (clearance_rate * num_trucks)
        
        total_time = 2 * travel_one_way + clearance_time  # round trip + clearance
        
        items.append({
            'cluster': cluster_id,
            'police_station': station,
            'value': round(value, 2),
            'cost_hours': round(total_time, 3),
            'travel_time_hours': round(2 * travel_one_way, 3),
            'clearance_time_hours': round(clearance_time, 3),
            'violations_to_clear': vehicles_to_clear,
            'center_lat': zone_lat,
            'center_lon': zone_lon,
        })
    
    if len(items) == 0:
        print("  ⚠ No valid zones to optimize.")
        return {'schedule': pd.DataFrame(), 'total_impact_recovered': 0,
                'total_hours_used': 0, 'total_hours_available': total_truck_hours,
                'zones_cleared': 0, 'zones_skipped': 0}
    
    items_df = pd.DataFrame(items)
    
    # --- 0-1 Knapsack via Dynamic Programming ---
    # Discretize costs to integer weights (in minutes) for DP
    n = len(items_df)
    W_minutes = int(total_truck_hours * 60)  # Total budget in minutes
    
    costs_minutes = (items_df['cost_hours'] * 60).astype(int).values
    values = (items_df['value'] * 100).astype(int).values  # Scale for integer DP
    
    # DP table
    print(f"\n  Solving 0-1 Knapsack: {n} zones, budget = {W_minutes} minutes...")
    
    dp = np.zeros((n + 1, W_minutes + 1), dtype=np.int64)
    
    for i in range(1, n + 1):
        w_i = costs_minutes[i - 1]
        v_i = values[i - 1]
        for w in range(W_minutes + 1):
            dp[i][w] = dp[i - 1][w]
            if w >= w_i and dp[i - 1][w - w_i] + v_i > dp[i][w]:
                dp[i][w] = dp[i - 1][w - w_i] + v_i
    
    # Backtrack to find selected items
    selected = []
    w = W_minutes
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            selected.append(i - 1)
            w -= costs_minutes[i - 1]
    
    selected.reverse()
    
    # Build the dispatch schedule
    schedule = items_df.iloc[selected].copy()
    schedule = schedule.sort_values('value', ascending=False).reset_index(drop=True)
    schedule['dispatch_priority'] = range(1, len(schedule) + 1)
    
    total_impact = schedule['value'].sum()
    total_hours = schedule['cost_hours'].sum()
    
    print(f"\n  ✅ OPTIMIZATION RESULTS:")
    print(f"  {'─'*50}")
    print(f"  Zones to dispatch:     {len(schedule)} / {n}")
    print(f"  Total impact recovered: {total_impact:.2f}")
    print(f"  Total hours used:       {total_hours:.2f} / {total_truck_hours}")
    print(f"  Efficiency:             {total_impact/total_hours:.2f} impact/hour"
          if total_hours > 0 else "  Efficiency: N/A")
    print(f"  {'─'*50}")
    
    print(f"\n  📋 DISPATCH SCHEDULE:")
    print(f"  {'─'*75}")
    print(f"  {'Pri':>3} {'Cluster':>7} {'Station':<20} {'Impact':>8} {'Cost(h)':>8} "
          f"{'Clear':>6} {'Travel(h)':>9}")
    print(f"  {'─'*75}")
    for _, s in schedule.iterrows():
        print(f"  {s['dispatch_priority']:>3} {int(s['cluster']):>7} "
              f"{s['police_station']:<20} {s['value']:>8.2f} {s['cost_hours']:>8.3f} "
              f"{int(s['violations_to_clear']):>6} {s['travel_time_hours']:>9.3f}")
    print(f"  {'─'*75}")
    
    return {
        'schedule': schedule,
        'total_impact_recovered': round(total_impact, 2),
        'total_hours_used': round(total_hours, 2),
        'total_hours_available': total_truck_hours,
        'zones_cleared': len(schedule),
        'zones_skipped': n - len(schedule)
    }


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================
# Ties all four modules together into a single end-to-end pipeline.
# ============================================================================

def run_proactive_dispatch_engine(df, zone_ranking_df,
                                   cluster_col='cluster_label',
                                   # Module 1 params
                                   v_free=45.0, k_jam=150.0, road_lanes=2,
                                   # Module 2 params
                                   forecast_horizons=[1, 2, 3],
                                   # Module 3 params
                                   shap_horizon=1, top_n_explain=10,
                                   # Module 4 params
                                   total_truck_hours=24.0,
                                   num_trucks=5,
                                   truck_capacity=8):
    """
    End-to-end pipeline that runs all four modules sequentially.
    
    Parameters
    ----------
    df : pd.DataFrame
        The cleaned, feature-engineered violation-level DataFrame from
        the existing HDBSCAN pipeline. Must contain:
        violation_date_time, hour, cluster_label, record_impact,
        severity_weight, poi_weight, persistence_score, validation_weight
    zone_ranking_df : pd.DataFrame
        The enforcement_priority_ranking DataFrame (or equivalent) with:
        cluster, total_violations, congestion_impact_score, mean_severity,
        center_lat, center_lon, police_station, etc.
    cluster_col : str
        Name of the cluster label column in df.
    
    Returns
    -------
    dict : {
        'zone_physics': pd.DataFrame,   # Module 1 output
        'models_dict': dict,             # Module 2 output
        'forecast': pd.DataFrame,        # Module 2 forecast
        'shap_results': dict,            # Module 3 output
        'dispatch': dict,                # Module 4 output
    }
    """
    print("╔" + "═"*68 + "╗")
    print("║  PROACTIVE AI DISPATCH ENGINE — FULL PIPELINE                      ║")
    print("╚" + "═"*68 + "╝")
    
    # ── MODULE 1 ──
    zone_physics = physics_informed_congestion_score(
        zone_ranking_df, v_free=v_free, k_jam=k_jam, road_lanes=road_lanes
    )
    
    # ── MODULE 2 ──
    ts_df = build_cluster_timeseries(df, zone_physics, cluster_col=cluster_col)
    ts_features = engineer_temporal_features(ts_df, cluster_col=cluster_col)
    models_dict = train_xgboost_forecaster(
        ts_features, forecast_horizons=forecast_horizons, cluster_col=cluster_col
    )
    forecast = generate_shift_forecast(
        models_dict, ts_features, zone_physics, cluster_col=cluster_col, top_n=20
    )
    
    # ── MODULE 3 ──
    shap_results = None
    if len(forecast) > 0 and shap_horizon in models_dict['models']:
        shap_results = explain_predictions_shap(
            models_dict, ts_features, forecast,
            cluster_col=cluster_col, horizon=shap_horizon,
            top_n_explain=top_n_explain
        )
    else:
        print("\n  ⚠ Skipping SHAP (insufficient forecast data or missing model).")
    
    # ── MODULE 4 ──
    dispatch = {'schedule': pd.DataFrame()}
    if len(forecast) > 0:
        dispatch = knapsack_dispatch_optimization(
            forecast, zone_physics,
            total_truck_hours=total_truck_hours,
            num_trucks=num_trucks,
            truck_capacity=truck_capacity
        )
    else:
        print("\n  ⚠ Skipping dispatch optimization (no forecast data).")
    
    # ── SAVE OUTPUTS ──
    zone_physics.to_csv(f'{OUTPUT_DIR}/physics_scored_zones.csv', index=False)
    forecast.to_csv(f'{OUTPUT_DIR}/shift_forecast.csv', index=False)
    if len(dispatch.get('schedule', pd.DataFrame())) > 0:
        dispatch['schedule'].to_csv(f'{OUTPUT_DIR}/dispatch_schedule.csv', index=False)
    
    print("\n" + "╔" + "═"*68 + "╗")
    print("║  PIPELINE COMPLETE                                                 ║")
    print("╚" + "═"*68 + "╝")
    print(f"\n  Output files saved to: {OUTPUT_DIR}/")
    print(f"    • physics_scored_zones.csv")
    print(f"    • shift_forecast.csv")
    print(f"    • dispatch_schedule.csv")
    if shap_results:
        print(f"    • shap_summary_{shap_horizon}h.png")
        print(f"    • shap_importance_{shap_horizon}h.png")
        print(f"    • shap_dep_*_{shap_horizon}h.png")
    
    return {
        'zone_physics': zone_physics,
        'models_dict': models_dict,
        'forecast': forecast,
        'shap_results': shap_results,
        'dispatch': dispatch,
    }


# ============================================================================
# USAGE EXAMPLE (run from existing notebook after HDBSCAN step)
# ============================================================================
if __name__ == '__main__':
    print("="*70)
    print("  This module is designed to be imported into the existing notebook.")
    print("  After your HDBSCAN clustering step, call:")
    print()
    print("    from proactive_dispatch_engine import run_proactive_dispatch_engine")
    print()
    print("    results = run_proactive_dispatch_engine(")
    print("        df=df,                          # cleaned violation DataFrame")
    print("        zone_ranking_df=zone_ranking,   # enforcement_priority_ranking")
    print("        cluster_col='cluster_label',    # column with HDBSCAN labels")
    print("    )")
    print("="*70)
