import argparse
import numpy as np
import pandas as pd

def evaluate_results(df):
    """
    Compares momentum, votes, and decay_votes against 'limit' as ground truth.
    Measures Angular Error (accuracy) and Angular Jitter (smoothness).
    """
    print("\n" + "="*50)
    print(" VELOCITY ESTIMATION METRICS (Averaged across clusters)")
    print("="*50)
    
    methods = ['limit', 'momentum', 'votes', 'decay_votes']
    metrics = {m: {'error': [], 'jitter': []} for m in methods}
    
    for cluster_id, group in df.groupby('cluster_id'):
        
        # 1. Clean the data (Drop rows where limit or the method is NaN)
        arrays = {}
        for m in methods:
            if m in group.columns:
                # Convert list [vx, vy] to a 2D numpy array, dropping NaNs
                arr = np.stack(group[m].values)
                arrays[m] = arr
                
        # We need a shared mask where at least 'limit' is valid to use it as ground truth
        limit_valid = ~np.isnan(arrays['limit'][:, 0])
        
        for m in methods:
            if m not in arrays: continue
            
            arr = arrays[m]
            method_valid = ~np.isnan(arr[:, 0])
            shared_valid = limit_valid & method_valid
            
            v_pred = arr[shared_valid]
            v_true = arrays['limit'][shared_valid]
            
            if len(v_pred) < 2:
                continue
                
            # --- ANGULAR ERROR (vs Limit) ---
            if m != 'limit':
                norm_pred = np.linalg.norm(v_pred, axis=1)
                norm_true = np.linalg.norm(v_true, axis=1)
                
                # Avoid division by zero
                nz = (norm_pred > 1e-6) & (norm_true > 1e-6)
                if np.any(nz):
                    cos_theta = np.sum(v_pred[nz] * v_true[nz], axis=1) / (norm_pred[nz] * norm_true[nz])
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    angles = np.degrees(np.arccos(cos_theta))
                    metrics[m]['error'].append(angles.mean())

            # --- ANGULAR JITTER (Frame-to-Frame Smoothness) ---
            v_current = v_pred[1:]
            v_prev = v_pred[:-1]
            
            norm_curr = np.linalg.norm(v_current, axis=1)
            norm_prev = np.linalg.norm(v_prev, axis=1)
            
            nz_j = (norm_curr > 1e-6) & (norm_prev > 1e-6)
            if np.any(nz_j):
                cos_j = np.sum(v_current[nz_j] * v_prev[nz_j], axis=1) / (norm_curr[nz_j] * norm_prev[nz_j])
                cos_j = np.clip(cos_j, -1.0, 1.0)
                jitter_angles = np.degrees(np.arccos(cos_j))
                metrics[m]['jitter'].append(jitter_angles.mean())

    # Print summary table
    print(f"{'Method':<15} | {'Angular Error (vs Limit)':<25} | {'Angular Jitter (Smoothness)':<25}")
    print("-" * 70)
    for m in methods:
        if m in metrics and metrics[m]['jitter']:
            err_str = f"{np.mean(metrics[m]['error']):.2f}°" if m != 'limit' else "0.00° (Ground Truth)"
            jit_str = f"{np.mean(metrics[m]['jitter']):.2f}° / frame"
            print(f"{m:<15} | {err_str:<25} | {jit_str:<25}")
    print("="*50 + "\n")
if __name__ == "__main__":
    print("Loading saved velocities for evaluation...")
    saved_df = pd.read_parquet("data/E_patch_dstream_2x2_velocity.parquet")
    evaluate_results(saved_df)