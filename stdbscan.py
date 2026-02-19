import time
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN as skDBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    import cupy as cp
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


# ---------------------------------------------------------------------------
# Shared chunking + overlap-stitching helper
# ---------------------------------------------------------------------------

def _run_chunked(
    df: pd.DataFrame,
    chunk_fn,           # callable(df_slice) -> np.ndarray of int32 labels
    chunk_size: int,
    overlap: int,
    label_col: str,
    tag: str = 'ST-DBSCAN',
) -> pd.DataFrame:
    """
    Process df in sliding windows of chunk_size events, overlapping adjacent
    windows by `overlap` events.  A union-find structure merges cluster IDs
    that co-occur in the overlap region, stitching clusters across boundaries.

    Args:
        df:         Input DataFrame (sorted by time).
        chunk_fn:   Function that clusters one window; returns int32 label array.
        chunk_size: Total window size (core + overlap tail).
        overlap:    Events shared between consecutive windows.
        label_col:  Column name to write final labels into.
        tag:        String shown in progress messages.

    Returns:
        Copy of df with label_col appended.  Noise = -1; cluster IDs are
        contiguous non-negative integers remapped after stitching.
    """
    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    n = len(df)
    n_chunks = (n + stride - 1) // stride
    print(f"Chunked {tag}: {n:,} events, {n_chunks} window(s), "
          f"chunk_size={chunk_size:,}, overlap={overlap:,}...")

    all_labels = np.full(n, -1, dtype=np.int32)
    next_cluster_id = 0

    # Path-compressed union-find
    uf: dict = {}

    def uf_find(x: int) -> int:
        root = x
        while uf.get(root, root) != root:
            root = uf.get(root, root)
        while uf.get(x, x) != root:
            uf[x], x = root, uf.get(x, x)
        return root

    def uf_union(a: int, b: int) -> None:
        a, b = uf_find(a), uf_find(b)
        if a != b:
            uf[max(a, b)] = min(a, b)

    prev_tail: np.ndarray = None   # offset labels for prev window's tail
    t_total_start = time.time()

    for i in range(n_chunks):
        t_window_start = time.time()
        start    = i * stride
        end      = min(start + chunk_size, n)
        core_end = min(start + stride, n)
        core_len = core_end - start

        raw = chunk_fn(df.iloc[start:end]).astype(np.int32)

        # Offset non-noise labels to be globally unique
        labels = raw.copy()
        mask = labels >= 0
        if mask.any():
            labels[mask] += next_cluster_id
            next_cluster_id = int(labels[mask].max()) + 1

        # Stitch with previous window via overlap region
        if prev_tail is not None and overlap > 0:
            n_ov = min(overlap, len(prev_tail), len(labels))
            for pl, cl in zip(prev_tail[:n_ov], labels[:n_ov]):
                if pl >= 0 and cl >= 0:
                    uf_union(int(pl), int(cl))

        # Write core region (overlap tail gets re-clustered next iteration)
        all_labels[start:core_end] = labels[:core_len]

        # Last window: also write any leftover tail
        if i == n_chunks - 1 and end > core_end:
            all_labels[core_end:end] = labels[core_len:]

        prev_tail = labels[core_len:] if end > core_end else None

        n_clust = int((labels[:core_len] >= 0).sum())
        elapsed = time.time() - t_window_start
        total_elapsed = time.time() - t_total_start
        eta = (total_elapsed / (i + 1)) * (n_chunks - i - 1)
        print(f"  Window {i+1}/{n_chunks}: {core_len:,} core events, "
              f"{n_clust:,} clustered, {next_cluster_id:,} raw IDs | "
              f"{elapsed:.1f}s | elapsed {total_elapsed:.0f}s | ETA {eta:.0f}s")

    # Apply union-find and remap to contiguous IDs
    if next_cluster_id > 0:
        lookup = np.array([uf_find(k) for k in range(next_cluster_id)], dtype=np.int32)
        _, inverse = np.unique(lookup, return_inverse=True)
        lookup = inverse.astype(np.int32)
        valid = all_labels >= 0
        all_labels[valid] = lookup[all_labels[valid]]
        n_final = int(lookup.max()) + 1 if valid.any() else 0
        print(f"After stitching: {n_final:,} final clusters "
              f"(merged from {next_cluster_id:,} raw IDs)")

    out = df.copy()
    out[label_col] = all_labels
    return out


# ---------------------------------------------------------------------------
# CPU ST-DBSCAN  (sklearn DBSCAN, Chebyshev metric)
# ---------------------------------------------------------------------------

def run_stdbscan(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    eps_spatial: float = 2.0,
    eps_temporal: float = 10_000.0,
    min_pts: int = 10,
    label_col: str = 'cluster',
) -> pd.DataFrame:
    """
    CPU ST-DBSCAN using sklearn DBSCAN with Chebyshev (L-inf) distance.

    The timestamp axis is scaled so eps_temporal maps to eps_spatial, giving a
    box-shaped spatio-temporal neighbourhood:

        |dx| <= eps_spatial  AND  |dy| <= eps_spatial  AND  |dt| <= eps_temporal

    This is an axis-aligned box rather than a circular spatial disc, but is
    much faster and equivalent for event camera data.

    Args:
        df:           DataFrame with event data, sorted by time.
        x_col:        Column for x coordinate (pixels).
        y_col:        Column for y coordinate (pixels).
        t_col:        Column for timestamp (any unit; eps_temporal uses same unit).
        eps_spatial:  Maximum spatial distance between neighbours (pixels).
        eps_temporal: Maximum temporal distance between neighbours (timestamp units).
        min_pts:      Minimum neighbourhood size for a core point.
        label_col:    Output column name.

    Returns:
        Copy of df with integer label_col. Noise = -1.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required: pip install scikit-learn")

    print(f"CPU ST-DBSCAN on {len(df):,} events "
          f"(eps_s={eps_spatial}, eps_t={eps_temporal}, min_pts={min_pts})...")

    t_scale = float(eps_spatial) / float(eps_temporal)
    x = df[x_col].to_numpy(np.float32)
    y = df[y_col].to_numpy(np.float32)
    t = (df[t_col].to_numpy(np.float64) * t_scale).astype(np.float32)
    features = np.column_stack([x, y, t])

    labels = skDBSCAN(
        eps=eps_spatial,
        min_samples=min_pts,
        metric='chebyshev',
        algorithm='ball_tree',
        n_jobs=-1,
    ).fit_predict(features).astype(np.int32)

    out = df.copy()
    out[label_col] = labels
    return out


# ---------------------------------------------------------------------------
# GPU ST-DBSCAN  (cuML DBSCAN, Euclidean metric on scaled features)
# ---------------------------------------------------------------------------

def run_stdbscan_gpu(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    eps_spatial: float = 2.0,
    eps_temporal: float = 10_000.0,
    min_pts: int = 10,
    label_col: str = 'cluster',
    chunk_size: int = None,
    overlap: int = None,
) -> pd.DataFrame:
    """
    GPU-accelerated ST-DBSCAN using cuML DBSCAN with Euclidean distance.

    Same temporal scaling as run_stdbscan, but cuML uses Euclidean (L2)
    distance, giving an ellipsoidal neighbourhood:

        sqrt(dx^2 + dy^2 + (dt * scale)^2) <= eps_spatial

    This is the closest approximation of ST-DBSCAN available through cuML and
    runs entirely on the GPU.

    Args:
        df:           DataFrame with event data, sorted by time.
        x_col:        Column for x (pixels).
        y_col:        Column for y (pixels).
        t_col:        Column for timestamp.
        eps_spatial:  Neighbourhood radius in scaled 3-D space.
        eps_temporal: Temporal radius (same unit as t_col).
        min_pts:      Minimum neighbourhood size for a core point.
        label_col:    Output column name.
        chunk_size:   Process in windows of this many events (None = full dataset).
                      Use when the full dataset exceeds GPU memory or cuML's int32 limit.
        overlap:      Events shared between consecutive windows (default: chunk_size // 5).

    Returns:
        Copy of df with integer label_col. Noise = -1.
    """
    if not HAS_CUML:
        raise ImportError(
            "cuML is not installed. Install RAPIDS: https://rapids.ai/start.html"
        )

    print(f"[GPU ST-DBSCAN] Running on GPU with cuML...")
    t_scale = float(eps_spatial) / float(eps_temporal)

    def _gpu_chunk(df_slice: pd.DataFrame) -> np.ndarray:
        x = df_slice[x_col].to_numpy(np.float32)
        y = df_slice[y_col].to_numpy(np.float32)
        t = (df_slice[t_col].to_numpy(np.float64) * t_scale).astype(np.float32)
        features_gpu = cp.asarray(np.column_stack([x, y, t]))
        labels = cp.asnumpy(
            cuDBSCAN(
                eps=eps_spatial,
                min_samples=min_pts,
                metric='euclidean',
            ).fit_predict(features_gpu)
        ).astype(np.int32)
        del features_gpu
        return labels

    if chunk_size is not None:
        ov = overlap if overlap is not None else chunk_size // 5
        return _run_chunked(df, _gpu_chunk, chunk_size, ov, label_col, 'GPU ST-DBSCAN')

    print(f"GPU ST-DBSCAN on {len(df):,} events "
          f"(eps_s={eps_spatial}, eps_t={eps_temporal}, min_pts={min_pts})...")
    labels = _gpu_chunk(df)
    out = df.copy()
    out[label_col] = labels
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def cluster_events_stdbscan(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    eps_spatial: float = 2.0,
    eps_temporal: float = 10_000.0,
    min_pts: int = 10,
    label_col: str = 'cluster',
    use_gpu: bool = False,
    chunk_size: int = None,
    overlap: int = None,
) -> pd.DataFrame:
    """
    Cluster event camera data with ST-DBSCAN on the entire dataset.

    Dispatches to GPU (cuML Euclidean) or CPU (sklearn Chebyshev).

    Args:
        df:           DataFrame with x_col, y_col, t_col columns, sorted by time.
        x_col:        Column for x coordinate (pixels).
        y_col:        Column for y coordinate (pixels).
        t_col:        Column for timestamp.
        eps_spatial:  Spatial neighbourhood radius (pixels).
        eps_temporal: Temporal neighbourhood radius (same unit as t_col).
        min_pts:      Minimum neighbourhood size for a core point.
        label_col:    Output cluster column name.
        use_gpu:      Use cuML GPU backend when True.
        chunk_size:   Process in sliding windows of this many events.
                      Required for very large datasets that exceed GPU memory or
                      cuML's internal int32 indexing limit (~500k–2M events).
                      If None and GPU fails with a CUDA error, automatically
                      retries with chunk_size=500_000.
        overlap:      Events shared between consecutive windows
                      (default: chunk_size // 5).

    Returns:
        Copy of df with integer label_col. Noise = -1.

    Examples:
        # GPU processing, large dataset
        clustered = cluster_events_stdbscan(
            df, eps_spatial=3, eps_temporal=5000, min_pts=50,
            use_gpu=True, chunk_size=500_000,
        )

        # CPU processing
        clustered = cluster_events_stdbscan(
            df, eps_spatial=2, eps_temporal=10_000, min_pts=10,
        )
    """
    gpu_kwargs = dict(
        df=df,
        x_col=x_col, y_col=y_col, t_col=t_col,
        eps_spatial=eps_spatial, eps_temporal=eps_temporal,
        min_pts=min_pts, label_col=label_col,
        chunk_size=chunk_size, overlap=overlap,
    )
    cpu_kwargs = dict(
        df=df,
        x_col=x_col, y_col=y_col, t_col=t_col,
        eps_spatial=eps_spatial, eps_temporal=eps_temporal,
        min_pts=min_pts, label_col=label_col,
    )
    if use_gpu:
        if not HAS_CUML:
            print("Warning: cuML not available, falling back to CPU ST-DBSCAN.")
        else:
            try:
                return run_stdbscan_gpu(**gpu_kwargs)
            except RuntimeError as e:
                if chunk_size is None and 'CUDA' in str(e):
                    auto_chunk = 500_000
                    print(f"Warning: GPU ST-DBSCAN failed with CUDA error, "
                          f"retrying with chunk_size={auto_chunk:,}...")
                    return run_stdbscan_gpu(**{**gpu_kwargs, 'chunk_size': auto_chunk})
                print(f"Warning: GPU ST-DBSCAN failed ({type(e).__name__}: {e}), "
                      f"falling back to CPU.")
            except Exception as e:
                print(f"Warning: GPU ST-DBSCAN failed ({type(e).__name__}: {e}), "
                      f"falling back to CPU.")
    return run_stdbscan(**cpu_kwargs)


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from extract import extract, save_df
    from clustering import cluster_summary

    df = extract('data/val_day_014_td.parquet')
    print(f"Loaded {len(df):,} events, columns: {list(df.columns)}")

    clustered = cluster_events_stdbscan(
        df,
        eps_spatial=4.5,        # pixels
        eps_temporal=10000.0,   # µs  (adjust to match your timestamp unit)
        min_pts=100,
        use_gpu=True,
        chunk_size=350_000,     # prevent cuML int32 overflow on large datasets
        overlap=25_000,    # some events shared between windows to stitch clusters
    )

    save_df(clustered, 'data/val_day_014_td_stdbscan.parquet')

    summary = cluster_summary(clustered)
    print(summary.head(20))
