import argparse
import numpy as np
import pandas as pd

def get_polarity_col(df):
    """
    Checks for the existence of a polarity column under common names.
    Returns the column name if found, otherwise None.
    """
    for col in ['polarity', 'pol', 'p']:
        if col in df.columns:
            return col
    return None

def calc_limit_velocity(df, delta_t, t_grid, cluster_id):
    print(f"  -> [Limit] Cluster {cluster_id}: Processing {len(df)} events across {len(t_grid)} intervals.")
    
    # Extract raw numpy arrays for massive speedup
    times = df['t'].values
    xs = df['x'].values
    ys = df['y'].values
    velocities = []
    
    for T in t_grid:
        mask_minus = (times >= T - delta_t) & (times < T)
        mask_plus = (times >= T) & (times < T + delta_t)

        if not np.any(mask_minus) or not np.any(mask_plus):
            velocities.append([np.nan, np.nan])
            continue

        x_minus = np.array([xs[mask_minus].mean(), ys[mask_minus].mean()])
        t_minus = times[mask_minus].mean()

        x_plus = np.array([xs[mask_plus].mean(), ys[mask_plus].mean()])
        t_plus = times[mask_plus].mean()

        dt = t_plus - t_minus
        if dt == 0:
            velocities.append([np.nan, np.nan])
        else:
            v = (x_plus - x_minus) / dt
            velocities.append(v.tolist())
            
    return velocities

def calc_momentum_velocity(df, delta_t, t_grid, cluster_id):
    """
    Method 2: Momentum
    Calculates velocity iteratively using a decaying mass and center of mass.
    """
    print(f"  -> [Momentum] Cluster {cluster_id}: Iterating over {len(df)} events with dt={delta_t}.")
    df = df.sort_values('t')
    times = df['t'].values
    xs = df['x'].values
    ys = df['y'].values

    v_history, t_history = [], []
    m = 0.0
    x_com = np.array([0.0, 0.0])
    v = np.array([0.0, 0.0])
    last_t = times[0]

    for i in range(len(times)):
        t_curr = times[i]
        x_p = np.array([xs[i], ys[i]])
        dt = t_curr - last_t

        # Exponential decay: e^(-dt / delta_t) handles scaling for large dt
        decay = np.exp(-dt / delta_t)
        m1 = m * decay + 1

        if i == 0:
            x_com1 = x_p
            v1 = np.array([0.0, 0.0])
        else:
            x_com1 = ((m1 - 1) * x_com + x_p) / m1
            # Instantaneous velocity is displacement of COM over dt
            if dt > 0:
                inst_v = (x_p - x_com) / (m1 * dt)
            else:
                inst_v = np.array([0.0, 0.0])
                
            v1 = ((m1 - 1) * v + inst_v) / m1

        m, x_com, v, last_t = m1, x_com1, v1, t_curr
        v_history.append(v.copy())
        t_history.append(t_curr)

    # Sample at t_grid intervals
    velocities = []
    t_history = np.array(t_history)
    v_history = np.array(v_history)

    for T in t_grid:
        valid_idx = np.where(t_history <= T)[0]
        if len(valid_idx) == 0:
            velocities.append([np.nan, np.nan])
        else:
            velocities.append(v_history[valid_idx[-1]].tolist())
            
    return velocities

def calc_directional_votes(df, delta_t, t_grid, cluster_id):
    pol_col = get_polarity_col(df)
    if not pol_col:
        print(f"  -> [Votes] Cluster {cluster_id}: WARNING - No polarity column found. Skipping.")
        return [[np.nan, np.nan]] * len(t_grid)

    print(f"  -> [Votes] Cluster {cluster_id}: Tallying directional votes over {len(t_grid)} intervals.")
    
    # Extract raw numpy arrays
    times = df['t'].values
    pols = df[pol_col].values
    velocities = []
    
    for T in t_grid:
        mask = (times >= T) & (times < T + delta_t)
        pols_window = pols[mask]
        N = len(pols_window)
        
        if N == 0:
            velocities.append([np.nan, np.nan])
            continue

        # Use numpy to count values efficiently
        pol_counts = np.bincount(pols_window, minlength=4)
        vx = (pol_counts[1] - pol_counts[3]) / N
        vy = (pol_counts[0] - pol_counts[2]) / N
        velocities.append([vx, vy])
        
    return velocities

def calc_decay_votes(df, delta_t, t_grid, cluster_id):
    """
    Method 4: Directional Votes with Decay
    Iteratively updates directional votes with an exponential decay factor.
    """
    pol_col = get_polarity_col(df)
    if not pol_col:
        print(f"  -> [Decay Votes] Cluster {cluster_id}: WARNING - No polarity column found. Skipping.")
        return [[np.nan, np.nan]] * len(t_grid)

    print(f"  -> [Decay Votes] Cluster {cluster_id}: Computing exponentially decayed votes.")
    df = df.sort_values('t')
    times = df['t'].values
    pols = df[pol_col].values

    v_history, t_history = [], []
    m = 0.0
    V = np.array([0.0, 0.0]) # Accumulated vote vector
    last_t = times[0]

    for i in range(len(times)):
        t_curr = times[i]
        pol = pols[i]
        dt = t_curr - last_t

        decay = np.exp(-dt / delta_t)
        m1 = m * decay + 1

        dx = 1 if pol == 1 else (-1 if pol == 3 else 0)
        dy = 1 if pol == 0 else (-1 if pol == 2 else 0)
        dv = np.array([dx, dy])

        V1 = V * decay + dv
        m, V, last_t = m1, V1, t_curr

        t_history.append(t_curr)
        v_history.append((V1 / m1).tolist())

    # Sample at t_grid intervals
    velocities = []
    t_history = np.array(t_history)
    v_history = np.array(v_history)

    for T in t_grid:
        valid_idx = np.where(t_history <= T)[0]
        if len(valid_idx) == 0:
            velocities.append([np.nan, np.nan])
        else:
            velocities.append(v_history[valid_idx[-1]].tolist())
            
    return velocities

def estimate_velocities(input_file, output_file, method='all', delta_t=1000):
    print(f"Loading {input_file}...")
    df = pd.read_parquet(input_file)
    
    cluster_col = 'cluster_id' if 'cluster_id' in df.columns else 'cluster'
    if cluster_col not in df.columns:
        print("Error: No cluster column found.")
        return

    initial_len = len(df)
    df = df[df[cluster_col] != -1]
    filtered_len = len(df)
    if initial_len != filtered_len:
        print(f"Ignored {initial_len - filtered_len} noise events (cluster -1).")
        
    if len(df) == 0:
        print("Error: No events remaining after filtering cluster -1.")
        return

    all_results = []
    print(f"Starting velocity estimation for {df[cluster_col].nunique()} clusters...")
    
    for cluster_id, cluster_df in df.groupby(cluster_col):
        # CREATE T_GRID SPECIFIC TO THIS CLUSTER
        t_min = cluster_df['t'].min()
        t_max = cluster_df['t'].max()
        # Align grid to delta_t boundaries if desired, or just start at t_min
        t_grid = np.arange(t_min, t_max + delta_t, delta_t) 
        
        res_df = pd.DataFrame({'t': t_grid, 'cluster_id': cluster_id})
        
        print(f"\n--- Processing Cluster: {cluster_id} ---")
        
        if method in ['limit', 'all']:
            res_df['limit'] = calc_limit_velocity(cluster_df, delta_t, t_grid, cluster_id)
        if method in ['momentum', 'all']:
            res_df['momentum'] = calc_momentum_velocity(cluster_df, delta_t, t_grid, cluster_id)
        if method in ['votes', 'all']:
            res_df['votes'] = calc_directional_votes(cluster_df, delta_t, t_grid, cluster_id)
        if method in ['decay', 'all']:
            res_df['decay_votes'] = calc_decay_votes(cluster_df, delta_t, t_grid, cluster_id)
            
        all_results.append(res_df)

    final_df = pd.concat(all_results, ignore_index=True)

    print(f"\nSaving results to {output_file}...")
    final_df.to_parquet(output_file)
    print("Done!")

def main():
    parser = argparse.ArgumentParser(description="Estimate cluster velocities from event data.")
    parser.add_argument("input_file", type=str, help="Path to input .parquet file.")
    parser.add_argument("output_file", type=str, help="Path to save the output .parquet file.")
    parser.add_argument(
        "-m", "--method",
        type=str,
        default='all',
        choices=['limit', 'momentum', 'votes', 'decay', 'all'],
        help="Velocity estimation method to run."
    )
    parser.add_argument(
        "-dt", "--delta_t",
        type=int,
        default=1000,
        help="Time interval delta_t for velocity calculation (default: 1000)."
    )

    args = parser.parse_args()
    estimate_velocities(args.input_file, args.output_file, args.method, args.delta_t)

if __name__ == "__main__":
    # ---------------------------------------------------------
    # OPTION 1: Run via command line (CLI)
    # e.g., python velocity_eval.py input.parquet output.parquet -m all -dt 1000
    # ---------------------------------------------------------
    # main()

    # ---------------------------------------------------------
    # OPTION 2: Hardcoded execution
    # ---------------------------------------------------------
    # Uncomment the line below to run directly from IDE without CLI arguments
    estimate_velocities("data/E_patch_dstream.parquet", "data/E_patch_velocity.parquet", method="all", delta_t=100_000)