import numpy as np
import pandas as pd
import time

try:
    from sklearn.cluster import MeanShift
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# ---------------------------------------------------------------------------
# Bandwidth selection
# ---------------------------------------------------------------------------

def silverman_bandwidth(X: np.ndarray) -> float:
    """
    Silverman's rule of thumb for selecting a scalar mean shift bandwidth.

    Multivariate generalisation for d dimensions:
        h = (4 / (d + 2))^(1/(d+4))  *  n^(-1/(d+4))  *  sigma_avg

    where sigma_avg is the mean per-dimension standard deviation of X.
    This reduces to the classic 1-D formula (1.06 * sigma * n^(-1/5)) when d=1.

    Args:
        X: Feature matrix of shape (n_samples, n_features).

    Returns:
        Scalar bandwidth estimate in the units of X.
    """
    n, d = X.shape
    sigma_avg = float(X.std(axis=0).mean())
    h = (4.0 / (d + 2)) ** (1.0 / (d + 4)) * n ** (-1.0 / (d + 4)) * sigma_avg
    return float(h)


# ---------------------------------------------------------------------------
# CPU mean shift
# ---------------------------------------------------------------------------

def run_meanshift(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    spatial_threshold: float = 10_000.0,
    bandwidth: float = None,
    bin_seeding: bool = True,
    min_bin_freq: int = 1,
    label_col: str = 'cluster',
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Run mean shift clustering on event camera data in (x, y, t) space.

    The temporal axis is scaled by 1/spatial_threshold before clustering so
    that `spatial_threshold` timestamp units ≈ 1 pixel of spatial distance.
    Silverman's rule selects the kernel bandwidth automatically from the scaled
    features when no explicit bandwidth is supplied.

    Args:
        df:                DataFrame with event data.
        x_col:             Column for x coordinate (pixels).
        y_col:             Column for y coordinate (pixels).
        t_col:             Column for timestamp.
        spatial_threshold: Temporal-to-spatial scale factor applied as
                           t_scaled = t / spatial_threshold.
                           Larger values compress time relative to space
                           (less sensitive to temporal separation). Smaller
                           values expand time (more temporally sensitive).
                           Default: 10000, suitable for µs timestamps with
                           pixel-space coordinates.
        bandwidth:         Gaussian kernel bandwidth in scaled feature space.
                           None → estimated automatically via Silverman's rule.
        bin_seeding:       Use sklearn's binned seed-point optimisation.
                           Greatly accelerates convergence on large datasets.
        min_bin_freq:      Minimum number of events per bin when bin_seeding is
                           True.  Increase (e.g. 5–50) to thin out seed points
                           and speed up computation on dense data.
        label_col:         Name of the output cluster ID column.
        n_jobs:            Number of parallel workers for sklearn (-1 = all CPUs).

    Returns:
        DataFrame with columns [x_col, y_col, t_col, label_col].
        Row order and index match the input.  Cluster IDs are contiguous
        non-negative integers (0, 1, 2, …).  Mean shift assigns every point
        to a cluster — there is no noise label.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    x = df[x_col].to_numpy(np.float32)
    y = df[y_col].to_numpy(np.float32)
    t = (df[t_col].to_numpy(np.float64) / spatial_threshold).astype(np.float32)
    features = np.column_stack([x, y, t])

    if bandwidth is None:
        bandwidth = silverman_bandwidth(features)
        print(f"Silverman's rule bandwidth: {bandwidth:.6f} "
              f"(scaled feature space, spatial_threshold={spatial_threshold})")

    print(f"Mean shift on {len(df):,} events "
          f"(spatial_threshold={spatial_threshold}, bandwidth={bandwidth:.6f}, "
          f"bin_seeding={bin_seeding}, min_bin_freq={min_bin_freq})...")

    ms = MeanShift(
        bandwidth=bandwidth,
        bin_seeding=bin_seeding,
        min_bin_freq=min_bin_freq,
        n_jobs=n_jobs,
    )
    labels = ms.fit_predict(features).astype(np.int32)

    n_clusters = int(labels.max()) + 1 if len(labels) > 0 else 0
    print(f"Found {n_clusters:,} clusters")

    out = df[[x_col, y_col, t_col]].copy()
    out[label_col] = labels
    return out


# ---------------------------------------------------------------------------
# GPU mean shift helpers (cupy, no cuML MeanShift needed)
# ---------------------------------------------------------------------------

def _bin_seeds(X: np.ndarray, bandwidth: float, min_bin_freq: int = 1) -> np.ndarray:
    """
    Create bin-seeded starting points by snapping events to a bandwidth-sized
    grid, keeping only bins that contain at least min_bin_freq events.

    Args:
        X:            Feature matrix (n, d), CPU numpy array.
        bandwidth:    Grid cell size (same units as X).
        min_bin_freq: Minimum events per bin to keep as a seed.

    Returns:
        (n_seeds, d) float32 array of seed centroids.
    """
    grid = np.floor(X / bandwidth).astype(np.int64)
    unique_bins, counts = np.unique(grid, axis=0, return_counts=True)
    valid = unique_bins[counts >= min_bin_freq]
    return ((valid + 0.5) * bandwidth).astype(np.float32)


def _ms_iterate_gpu(
    features_gpu: 'cp.ndarray',
    seeds_gpu: 'cp.ndarray',
    bandwidth: float,
    max_iter: int,
    convergence_tol: float,
    seed_batch_size: int,
) -> 'cp.ndarray':
    """
    Run Gaussian mean shift iterations on seed_gpu using cupy.

    Processes seeds in batches to keep GPU memory bounded.  Each seed is
    shifted toward the weighted mean of ALL events using a Gaussian kernel
    of width `bandwidth`.  Iterates until the maximum per-seed shift drops
    below convergence_tol or max_iter is reached.

    Memory per batch: ~3 × seed_batch_size × n_events × 4 bytes
    (dot product + sq_dists + weights matrices).

    Args:
        features_gpu:    (n, d) cupy float32 — all events on the GPU.
        seeds_gpu:       (n_seeds, d) cupy float32 — starting modes.
        bandwidth:       Gaussian kernel width.
        max_iter:        Maximum iterations.
        convergence_tol: Stop when max seed shift < this value.
        seed_batch_size: Seeds processed per GPU kernel call.

    Returns:
        (n_seeds, d) cupy float32 — converged mode positions.
    """
    n_seeds = seeds_gpu.shape[0]
    inv_2bw2 = 1.0 / (2.0 * bandwidth ** 2)
    feat_sq = cp.sum(features_gpu ** 2, axis=1)   # (n,)  reused every iter
    modes = seeds_gpu.copy()

    for it in range(max_iter):
        t0 = time.time()
        new_modes = cp.empty_like(modes)

        for s in range(0, n_seeds, seed_batch_size):
            batch = modes[s:s + seed_batch_size]          # (B, d)
            batch_sq = cp.sum(batch ** 2, axis=1)         # (B,)

            # Squared distances via ||a-b||^2 = ||a||^2 + ||b||^2 - 2a·b
            dot = cp.dot(batch, features_gpu.T)           # (B, n)
            sq_dists = batch_sq[:, None] + feat_sq[None, :] - 2.0 * dot  # (B, n)

            weights = cp.exp(-sq_dists * inv_2bw2)        # (B, n)
            w_sum = weights.sum(axis=1, keepdims=True)    # (B, 1)
            new_modes[s:s + seed_batch_size] = cp.dot(weights, features_gpu) / w_sum

        # Ensure kernels finished before measuring elapsed time
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass
        iter_time = time.time() - t0

        max_shift = float(cp.max(cp.sqrt(cp.sum((new_modes - modes) ** 2, axis=1))))
        modes = new_modes

        print(f"  Iter {it+1}: max_shift={max_shift:.6f}, time={iter_time:.3f}s")

        if max_shift < convergence_tol:
            print(f"  Converged after {it + 1} iterations (max shift {max_shift:.6f}, last_iter_time={iter_time:.3f}s)")
            break
    else:
        print(f"  Reached max_iter={max_iter} (max shift {max_shift:.6f})")

    return modes


def _merge_modes(modes: np.ndarray, bandwidth: float) -> np.ndarray:
    """
    Greedy merge of converged modes that are within bandwidth / 2 of each
    other.  Runs on CPU — the number of modes is much smaller than n_events.

    Args:
        modes:     (n_seeds, d) float32 CPU array of converged mode positions.
        bandwidth: Modes closer than bandwidth/2 are merged.

    Returns:
        (n_clusters, d) float32 array of unique cluster centres.
    """
    threshold = bandwidth / 2.0
    used = np.zeros(len(modes), dtype=bool)
    centres = []

    for i in range(len(modes)):
        if used[i]:
            continue
        dists = np.sqrt(np.sum((modes - modes[i]) ** 2, axis=1))
        close = dists < threshold
        used[close] = True
        centres.append(modes[close].mean(axis=0))

    return np.array(centres, dtype=np.float32)


def _assign_labels_gpu(
    features_gpu: 'cp.ndarray',
    centres_gpu: 'cp.ndarray',
    point_batch_size: int,
) -> np.ndarray:
    """
    Assign each event to its nearest cluster centre using cupy.

    Processes events in batches to keep GPU memory bounded.
    Memory per batch: ~3 × point_batch_size × n_clusters × 4 bytes.

    Args:
        features_gpu:     (n, d) cupy float32 — all events.
        centres_gpu:      (n_clusters, d) cupy float32 — cluster centres.
        point_batch_size: Events processed per GPU kernel call.

    Returns:
        (n,) int32 CPU numpy array of cluster IDs.
    """
    n = features_gpu.shape[0]
    c_sq = cp.sum(centres_gpu ** 2, axis=1)   # (n_clusters,)
    labels = cp.empty(n, dtype=cp.int32)

    for i in range(0, n, point_batch_size):
        batch = features_gpu[i:i + point_batch_size]       # (B, d)
        b_sq = cp.sum(batch ** 2, axis=1)                  # (B,)
        dot = cp.dot(batch, centres_gpu.T)                 # (B, n_clusters)
        sq_dists = b_sq[:, None] + c_sq[None, :] - 2.0 * dot
        labels[i:i + point_batch_size] = cp.argmin(sq_dists, axis=1)

    return cp.asnumpy(labels).astype(np.int32)


# ---------------------------------------------------------------------------
# GPU mean shift — public function
# ---------------------------------------------------------------------------

def run_meanshift_gpu(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    spatial_threshold: float = 10_000.0,
    bandwidth: float = None,
    max_iter: int = 300,
    convergence_tol: float = 1e-3,
    min_bin_freq: int = 1,
    seed_batch_size: int = 50,
    point_batch_size: int = 100_000,
    label_col: str = 'cluster',
) -> pd.DataFrame:
    """
    GPU-accelerated mean shift clustering using cupy (no cuML MeanShift needed).

    cuML does not expose a MeanShift estimator; this implementation runs the
    full algorithm directly on the GPU using cupy matrix operations:

      1. Bin seeding (CPU) — snap events to a bandwidth-sized grid and keep
         bins with ≥ min_bin_freq events as starting seeds.  Seeds << events.
      2. Mean shift iterations (GPU) — each seed is shifted toward the
         Gaussian-weighted mean of all events.  Seeds are processed in
         seed_batch_size batches to bound GPU memory usage.
      3. Mode merging (CPU) — converged seeds within bandwidth/2 of each
         other are averaged into a single cluster centre.
      4. Label assignment (GPU) — each event is assigned to its nearest
         cluster centre, processed in point_batch_size batches.

    GPU memory estimate per batch:
      Seed iteration:    ~3 × seed_batch_size × n_events × 4 B
      Label assignment:  ~3 × point_batch_size × n_clusters × 4 B

    Requires: cupy (ships with any RAPIDS/cuML installation).

    Args:
        df:                DataFrame with event data.
        x_col:             Column for x coordinate (pixels).
        y_col:             Column for y coordinate (pixels).
        t_col:             Column for timestamp.
        spatial_threshold: Temporal-to-spatial scale (t_scaled = t / spatial_threshold).
                           Default: 10000.
        bandwidth:         Gaussian kernel bandwidth in scaled feature space.
                           None → Silverman's rule (computed on CPU).
        max_iter:          Maximum mean shift iterations.
        convergence_tol:   Stop iterating when max seed shift < this value
                           (in scaled feature space units).
        min_bin_freq:      Minimum events per grid bin to create a seed.
                           Increase (e.g. 5–50) to reduce the seed count and
                           speed up iteration on dense data.
        seed_batch_size:   Seeds processed per GPU kernel call.
                           Reduce if you see GPU OOM errors during iteration.
        point_batch_size:  Events processed per GPU kernel call during label
                           assignment.  Reduce if you see GPU OOM errors.
        label_col:         Name of the output cluster ID column.

    Returns:
        DataFrame with columns [x_col, y_col, t_col, label_col].
        Row order and index match the input.  Cluster IDs are contiguous
        non-negative integers; no noise label.

    Raises:
        ImportError: If cupy is not installed.
    """
    if not HAS_CUPY:
        raise ImportError(
            "cupy is not installed. Install RAPIDS: https://rapids.ai/start.html"
        )

    x = df[x_col].to_numpy(np.float32)
    y = df[y_col].to_numpy(np.float32)
    t = (df[t_col].to_numpy(np.float64) / spatial_threshold).astype(np.float32)
    features = np.column_stack([x, y, t])   # CPU

    if bandwidth is None:
        bandwidth = silverman_bandwidth(features)
        print(f"Silverman's rule bandwidth: {bandwidth:.6f} "
              f"(scaled feature space, spatial_threshold={spatial_threshold})")

    # Step 1: bin seeds (CPU — fast)
    seeds = _bin_seeds(features, bandwidth, min_bin_freq)
    print(f"[GPU] Mean shift: {len(df):,} events, {len(seeds):,} seeds, "
          f"bandwidth={bandwidth:.6f}, max_iter={max_iter}, "
          f"seed_batch={seed_batch_size}...")

    # Step 2: iterate on GPU
    features_gpu = cp.asarray(features)
    seeds_gpu = cp.asarray(seeds)
    modes_gpu = _ms_iterate_gpu(
        features_gpu, seeds_gpu, bandwidth, max_iter, convergence_tol, seed_batch_size
    )
    modes_cpu = cp.asnumpy(modes_gpu)

    # Step 3: merge nearby modes (CPU — modes << n)
    centres = _merge_modes(modes_cpu, bandwidth)
    print(f"  {len(seeds):,} seeds → {len(centres):,} clusters after merging")

    # Step 4: assign labels (GPU, chunked over points)
    centres_gpu = cp.asarray(centres)
    labels = _assign_labels_gpu(features_gpu, centres_gpu, point_batch_size)
    del features_gpu, centres_gpu

    out = df[[x_col, y_col, t_col]].copy()
    out[label_col] = labels
    return out


# ---------------------------------------------------------------------------
# Dispatcher / public API
# ---------------------------------------------------------------------------

def cluster_events_meanshift(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    spatial_threshold: float = 10_000.0,
    bandwidth: float = None,
    bin_seeding: bool = True,
    min_bin_freq: int = 1,
    max_iter: int = 300,
    convergence_tol: float = 1e-3,
    seed_batch_size: int = 50,
    point_batch_size: int = 100_000,
    label_col: str = 'cluster',
    use_gpu: bool = False,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Cluster event camera data with mean shift.

    Dispatches to GPU (cupy) or CPU (sklearn) implementation.

    Args:
        df:                DataFrame with x_col, y_col, t_col columns.
        x_col:             Column for x coordinate.
        y_col:             Column for y coordinate.
        t_col:             Column for timestamp.
        spatial_threshold: Temporal-to-spatial scale (t_scaled = t / spatial_threshold).
                           10000 works well for µs timestamps with pixel coords.
        bandwidth:         Kernel bandwidth in scaled feature space.
                           None → Silverman's rule.
        bin_seeding:       (CPU only) Use sklearn's binned seeding.
        min_bin_freq:      Minimum events per grid bin to create a seed.
                           Shared by CPU (sklearn) and GPU paths.
        max_iter:          (GPU only) Maximum mean shift iterations.
        convergence_tol:   (GPU only) Stop when max seed shift < this value.
        seed_batch_size:   (GPU only) Seeds per GPU kernel call — reduce on OOM.
        point_batch_size:  (GPU only) Events per GPU kernel call during label
                           assignment — reduce on OOM.
        label_col:         Output cluster column name.
        use_gpu:           Route to GPU (cupy) backend when True; falls back to
                           CPU if cupy is unavailable.
        n_jobs:            (CPU only) Parallel workers (-1 = all CPUs).

    Returns:
        DataFrame with [x_col, y_col, t_col, label_col] columns.
        Cluster IDs are contiguous non-negative integers; no noise label.

    Example:
        >>> from extract import extract, save_df
        >>> df = extract('events.parquet')
        >>> clustered = cluster_events_meanshift(df, spatial_threshold=10_000, use_gpu=True)
        >>> print(clustered['cluster'].nunique(), 'clusters')
    """
    if use_gpu:
        if not HAS_CUPY:
            print("Warning: cupy not available, falling back to CPU mean shift.")
        else:
            return run_meanshift_gpu(
                df=df,
                x_col=x_col,
                y_col=y_col,
                t_col=t_col,
                spatial_threshold=spatial_threshold,
                bandwidth=bandwidth,
                max_iter=max_iter,
                convergence_tol=convergence_tol,
                min_bin_freq=min_bin_freq,
                seed_batch_size=seed_batch_size,
                point_batch_size=point_batch_size,
                label_col=label_col,
            )

    return run_meanshift(
        df=df,
        x_col=x_col,
        y_col=y_col,
        t_col=t_col,
        spatial_threshold=spatial_threshold,
        bandwidth=bandwidth,
        bin_seeding=bin_seeding,
        min_bin_freq=min_bin_freq,
        label_col=label_col,
        n_jobs=n_jobs,
    )


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from extract import extract, save_df

    df = extract('data/val_day_014_td.parquet')
    print(f"Loaded {len(df):,} events, columns: {list(df.columns)}")

    clustered = cluster_events_meanshift(
        df,
        spatial_threshold=2_000.0,   # µs per pixel equivalent
        use_gpu=True,                 # set False to use CPU sklearn
        min_bin_freq=10,               # skip sparse bins → fewer seeds → faster
        max_iter=10,
        seed_batch_size=22,           # reduce if GPU OOM during iteration
        point_batch_size=12_000,     # reduce if GPU OOM during label assignment
        bandwidth=300.0,
    )

    save_df(clustered, 'data/val_day_014_td_meanshift.parquet')
    print(clustered.head())
