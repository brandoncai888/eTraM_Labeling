import numpy as np
import pandas as pd
from scipy.ndimage import label

class DStreamEventClusterer:
    def __init__(self, sensor_width=346, sensor_height=260, grid_size=5, 
                 tau_us=50000, dense_threshold=10, update_interval_us=5000, medium_density_threshold=5, verify_time_us=20000):
        """
        Parameters:
        - sensor_width/height: Dimensions of your event camera.
        - grid_size: Pixel size of each grid cell (e.g., 5x5 pixels).
        - tau_us: Decay time constant in microseconds (e.g., 50,000 us = 50 ms).
        - dense_threshold: Density required for a cell to be considered part of an object.
        - update_interval_us: How often to run the global clustering (e.g., every 5 ms).
        """
        self.grid_size = grid_size
        self.tau = tau_us
        self.dense_threshold = dense_threshold
        self.medium_density_threshold = medium_density_threshold
        self.update_interval = update_interval_us
        self.verify_time_us = verify_time_us # minimum age for a cluster before assigning IDs (e.g., 20 ms)

        self.next_cluster_id = 1
        self.id_last_seen = {}          # persistent_id -> last_seen_t
        self.id_first_seen = {}         # persistent_id -> first_seen_t 
        self.id_active = set()          # currently active persistent IDs (optional convenience)

        # Optional tuning knobs (also put in __init__ if you want)
        self.min_cluster_cells = 3     # minimum number of cells for a valid cluster (can help filter noise)
        self.min_overlap_cells = 3     # absolute overlap needed to reuse an ID
        self.min_overlap_frac = 0.2     # OR overlap fraction of new component area
        self.max_inactive_time_us = 200000  # prune IDs not seen for this long

        # Calculate grid dimensions
        self.grid_w = (sensor_width // grid_size) + 1
        self.grid_h = (sensor_height // grid_size) + 1
        
        # Initialize state matrices
        self.density_grid = np.zeros((self.grid_h, self.grid_w))
        self.last_time_grid = np.zeros((self.grid_h, self.grid_w))
        self.cluster_grid = np.zeros((self.grid_h, self.grid_w), dtype=int)
        
        self.last_clustering_time = 0

    def process_dataframe(self, df):
        """Processes the DataFrame and returns an array of cluster IDs."""
        
        # Extract to numpy arrays for much faster iteration than Pandas looping
        xs = df['x'].values
        ys = df['y'].values
        ts = df['t'].values
        
        # Array to store the resulting cluster ID for each event
        assigned_ids = np.full(len(df), -1, dtype=int) 
        
        for i in range(len(df)):
            if i % 1000000 == 0:
                print(f"Processing event {i}/{len(df)}...")
            x, y, t = xs[i], ys[i], ts[i]
            
            # 1. Map event to its grid cell
            gx, gy = x // self.grid_size, y // self.grid_size
            
            # 2. Online Phase: Update Density for this specific cell
            dt = t - self.last_time_grid[gy, gx]
            
            # Apply continuous exponential decay
            decay_factor = np.exp(-dt / self.tau)
            self.density_grid[gy, gx] = (self.density_grid[gy, gx] * decay_factor) + 1
            self.last_time_grid[gy, gx] = t
            
            # 3. Offline Phase: Periodically re-evaluate clusters globally
            if (t - self.last_clustering_time) >= self.update_interval:
                self._update_global_clusters(t)
                self.last_clustering_time = t
                
            # 4. Assign ID in real-time
            # If the cell is sufficiently dense, give the event the cell's current cluster ID
            if self.density_grid[gy, gx] >= self.medium_density_threshold:
                assigned_ids[i] = self.cluster_grid[gy, gx]
                if assigned_ids[i] == 0:
                    assigned_ids[i] = -1
                elif t - self.id_first_seen[assigned_ids[i]] < self.verify_time_us:
                    assigned_ids[i] = -1
            # (If not dense enough, it remains -1, which acts as 'noise')

        return assigned_ids

    def _update_global_clusters(self, current_t):
        prev_persistent_grid = self.cluster_grid

        # --- 1) Decay snapshot ---
        time_diffs = current_t - self.last_time_grid
        decay_factors = np.exp(-time_diffs / self.tau)
        current_density_snapshot = self.density_grid * decay_factors

        structure = np.ones((3, 3), dtype=int)  # 8-connectivity

        # --- 2) Define seeds (dense) and allowed region (>= medium) ---
        dense_mask = current_density_snapshot >= self.dense_threshold
        allowed_mask = current_density_snapshot >= self.medium_density_threshold  # dense is included

        # If nothing dense, nothing to flood from
        if not np.any(dense_mask):
            self.cluster_grid = np.zeros_like(self.cluster_grid, dtype=int)
            if self.max_inactive_time_us is not None:
                to_del = [cid for cid, ts in self.id_last_seen.items()
                        if (current_t - ts) > self.max_inactive_time_us]
                for cid in to_del:
                    self.id_last_seen.pop(cid, None)
                    self.id_active.discard(cid)
            return
        
        # --- 3) Flood fill = connected components on allowed region, keep only those that contain dense ---
        # Label all connected components in the allowed region
        allowed_labels, num_allowed = label(allowed_mask, structure=structure)

        # Find which allowed-components have at least one dense cell (seed)
        dense_allowed_ids = np.unique(allowed_labels[dense_mask])
        dense_allowed_ids = dense_allowed_ids[dense_allowed_ids > 0]

        # Build final local labels: keep only those components, relabel to 1..K
        keep = np.isin(allowed_labels, dense_allowed_ids)
        local_labels, num_local = label(keep, structure=structure)

        # --- 3b) Filter out small components (minimum number of grid cells) ---
        # Requires: self.min_cluster_cells set in __init__ (e.g., 5). Set to None/0 to disable.
        if getattr(self, "min_cluster_cells", None) not in (None, 0, 1):
            flat = local_labels.ravel()
            counts = np.bincount(flat)  # counts[label] = number of cells in that component
            if counts.size > 1:
                counts[0] = 0  # ignore background
                small_labs = np.where(counts < self.min_cluster_cells)[0]
                if small_labs.size > 0:
                    local_labels[np.isin(local_labels, small_labs)] = 0
                    # relabel to compact labels back to 1..K
                    local_labels, num_local = label(local_labels > 0, structure=structure)

        # ---- from here on, keep your overlap-vote code unchanged, using local_labels ----
        if local_labels.max() == 0:
            self.cluster_grid = np.zeros_like(self.cluster_grid, dtype=int)
            if self.max_inactive_time_us is not None:
                to_del = [cid for cid, ts in self.id_last_seen.items()
                        if (current_t - ts) > self.max_inactive_time_us]
                for cid in to_del:
                    self.id_last_seen.pop(cid, None)
                    self.id_active.discard(cid)
            return
        
        # --- 4) Overlap-vote mapping local label -> persistent ID ---
        new_persistent_grid = np.zeros_like(prev_persistent_grid, dtype=int)

        # Track which persistent IDs got claimed this frame and by whom
        persistent_claims = {}  # pid -> list of tuples (local_id, overlap_count, local_area)

        # Precompute pixel lists per local component
        max_local = int(local_labels.max())
        local_to_coords = [None] * (max_local + 1)
        local_area = np.zeros(max_local + 1, dtype=int)

        ys, xs = np.nonzero(local_labels)
        labs = local_labels[ys, xs]
        # group coords by label (simple approach; good enough for modest grids)
        for y, x, lab in zip(ys, xs, labs):
            if local_to_coords[lab] is None:
                local_to_coords[lab] = []
            local_to_coords[lab].append((y, x))
            local_area[lab] += 1

        local_to_pid = {}  # local_id -> chosen persistent id (or None for new)

        for lab in range(1, max_local + 1):
            coords = local_to_coords[lab]
            if coords is None:
                continue

            cy = np.fromiter((p[0] for p in coords), dtype=int, count=len(coords))
            cx = np.fromiter((p[1] for p in coords), dtype=int, count=len(coords))

            prev_ids = prev_persistent_grid[cy, cx]
            prev_ids = prev_ids[prev_ids > 0]

            if prev_ids.size == 0:
                local_to_pid[lab] = None
                continue

            vals, counts = np.unique(prev_ids, return_counts=True)
            best_idx = np.argmax(counts)
            best_pid = int(vals[best_idx])
            best_overlap = int(counts[best_idx])

            # gating: require enough overlap to reuse an existing ID
            if best_overlap < self.min_overlap_cells and best_overlap < (self.min_overlap_frac * local_area[lab]):
                local_to_pid[lab] = None
                continue

            local_to_pid[lab] = best_pid

            # record claims for split-resolution
            persistent_claims.setdefault(best_pid, []).append((lab, best_overlap, local_area[lab]))

        # --- 5) Resolve splits: one old pid claimed by multiple locals ---
        # Keep the best-overlap local as the old pid; others get new pids.
        used_pids = set()
        for pid, claims in persistent_claims.items():
            if len(claims) == 1:
                used_pids.add(pid)
                continue

            # choose winner: max overlap, tie-break by larger area
            claims_sorted = sorted(claims, key=lambda t: (t[1], t[2]), reverse=True)
            winner_lab = claims_sorted[0][0]
            used_pids.add(pid)

            for (lab, _, _) in claims_sorted[1:]:
                # force new ID for other split parts
                local_to_pid[lab] = None

        # --- 6) Assign new IDs for unmatched locals, handle merges implicitly ---
        # Merge case: multiple old IDs under one new local -> overlap-vote already picks survivor.
        for lab in range(1, max_local + 1):
            if local_to_coords[lab] is None:
                continue
            pid = local_to_pid.get(lab, None)
            if pid is None:
                pid = self.next_cluster_id
                self.next_cluster_id += 1
                local_to_pid[lab] = pid

            # paint persistent grid
            coords = local_to_coords[lab]
            cy = np.fromiter((p[0] for p in coords), dtype=int, count=len(coords))
            cx = np.fromiter((p[1] for p in coords), dtype=int, count=len(coords))
            new_persistent_grid[cy, cx] = pid

            # update bookkeeping
            self.id_last_seen[pid] = current_t
            if pid not in self.id_active:
                self.id_first_seen[pid] = current_t
            self.id_active.add(pid)

        # --- 7) Retire IDs that disappeared (optional immediate retire) ---
        # If you want immediate retire of IDs not present in this update:
        # present_now = set(np.unique(new_persistent_grid)) - {0}
        # for pid in list(self.id_active):
        #     if pid not in present_now:
        #         self.id_active.discard(pid)

        # --- 8) Prune stale IDs by timeout ---
        if self.max_inactive_time_us is not None:
            to_del = [cid for cid, ts in self.id_last_seen.items()
                    if (current_t - ts) > self.max_inactive_time_us]
            for cid in to_del:
                self.id_last_seen.pop(cid, None)
                self.id_active.discard(cid)

        # commit
        self.cluster_grid = new_persistent_grid


# Assuming dataframe is loaded as 'df' with columns ['x', 'y', 't']
df = pd.read_parquet('data/E_patch.parquet')

# Initialize the clusterer (adjust sensor dimensions to match your camera, e.g., Prophesee, Davis)
clusterer = DStreamEventClusterer(sensor_width=320, sensor_height=180, grid_size=4, tau_us=50000, dense_threshold=20, update_interval_us=3000, medium_density_threshold=10, verify_time_us=10000)
print("Starting clustering process...")
# Run the clustering
df['cluster'] = clusterer.process_dataframe(df)
print("Clustering completed.")
df.to_parquet('data/E_patch_dstream.parquet')  # Save the results for later analysis
# View the events that were actually assigned to a cluster (filtering out noise)
clustered_events = df[df['cluster'] != -1]
print(clustered_events.head())