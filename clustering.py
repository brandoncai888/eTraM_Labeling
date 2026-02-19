import numpy as np
import pandas as pd

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    import cupy as cp
    HAS_CUML = True
except ImportError:
    HAS_CUML = False


def build_feature_matrix(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    time_scale: float = 1e-3,
) -> np.ndarray:
    """
    Construct the (N, 3) feature matrix for clustering.

    The temporal axis is multiplied by `time_scale` to bring it into the
    same unit space as the spatial axes.  For example, if x/y are in pixels
    and t is in microseconds, a time_scale of 0.001 means 1 µs ≈ 0.001 px,
    so events must be within ~1000 µs *and* ~1 px to be considered neighbours.

    Args:
        df:         DataFrame containing event data.
        x_col:      Column name for x coordinate.
        y_col:      Column name for y coordinate.
        t_col:      Column name for timestamp.
        time_scale: Multiplicative scale applied to t before clustering.
                    Larger values make the algorithm more sensitive to time
                    separation; smaller values collapse events in time.

    Returns:
        np.ndarray of shape (N, 3) with columns [x, y, t_scaled].
    """
    x = df[x_col].to_numpy(dtype=np.float32)
    y = df[y_col].to_numpy(dtype=np.float32)
    t = df[t_col].to_numpy(dtype=np.float64) * time_scale

    return np.column_stack([x, y, t]).astype(np.float32)


def run_hdbscan(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    time_scale: float = 1e3,
    min_cluster_size: int = 10,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    metric: str = 'euclidean',
    label_col: str = 'cluster',
) -> pd.DataFrame:
    """
    Run HDBSCAN on event camera data in (x, y, t) space.

    Noise points are assigned label -1.

    Args:
        df:                         DataFrame with event data.
        x_col:                      Column name for x coordinate.
        y_col:                      Column name for y coordinate.
        t_col:                      Column name for timestamp.
        time_scale:                 Scale factor for t (see build_feature_matrix).
        min_cluster_size:           Minimum number of events to form a cluster.
        min_samples:                Number of samples in a neighbourhood for a
                                    point to be considered a core point.
                                    Defaults to min_cluster_size when None.
        cluster_selection_epsilon:  Distance threshold; sub-clusters within this
                                    radius are merged. 0 disables merging.
        cluster_selection_method:   'eom' (excess of mass, default) or 'leaf'.
        metric:                     Distance metric passed to HDBSCAN.
        label_col:                  Name of the output column added to df.

    Returns:
        Copy of df with an integer `label_col` column appended.
    """
    if not HAS_HDBSCAN:
        raise ImportError(
            "hdbscan is not installed. Install it with: pip install hdbscan"
        )

    print(f"Starting HDBSCAN clustering on {len(df):,} events...")
    features = build_feature_matrix(df, x_col, y_col, t_col, time_scale)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
    )
    labels = clusterer.fit_predict(features)

    out = df.copy()
    out[label_col] = labels
    return out


def run_hdbscan_gpu(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    time_scale: float = 1e-3,
    min_cluster_size: int = 10,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    metric: str = 'euclidean',
    label_col: str = 'cluster',
) -> pd.DataFrame:
    """
    GPU-accelerated HDBSCAN using cuML (requires RAPIDS).

    Same interface as run_hdbscan; falls back with an ImportError if cuML
    is not available.

    Args:
        df:                         DataFrame with event data.
        x_col:                      Column name for x coordinate.
        y_col:                      Column name for y coordinate.
        t_col:                      Column name for timestamp.
        time_scale:                 Scale factor for t.
        min_cluster_size:           Minimum cluster size.
        min_samples:                Core point neighbourhood size.
        cluster_selection_epsilon:  Sub-cluster merge radius.
        cluster_selection_method:   'eom' or 'leaf'.
        metric:                     Distance metric.
        label_col:                  Output column name.

    Returns:
        Copy of df with an integer `label_col` column appended.
    """
    if not HAS_CUML:
        raise ImportError(
            "cuML is not installed. Install RAPIDS: https://rapids.ai/start.html"
        )

    print(f"Starting GPU HDBSCAN clustering (cuML) on {len(df):,} events...")
    features = build_feature_matrix(df, x_col, y_col, t_col, time_scale)
    features_gpu = cp.asarray(features)

    clusterer = cuHDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples is not None else min_cluster_size,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
    )
    labels = clusterer.fit_predict(features_gpu)

    out = df.copy()
    out[label_col] = cp.asnumpy(labels).astype(np.int32)
    return out


def run_hdbscan_gpu_chunked(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    time_scale: float = 1e-3,
    min_cluster_size: int = 10,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    metric: str = 'euclidean',
    label_col: str = 'cluster',
    chunk_size: int = 2_000_000,
    overlap: int = 0,
) -> pd.DataFrame:
    """
    GPU-accelerated HDBSCAN processed in temporal chunks to fit GPU memory.

    Events are assumed to be sorted by time. Each chunk overlaps with the next
    by `overlap` events. Events in the overlap region are clustered by both
    adjacent chunks; a union-find structure merges cluster IDs that co-occur
    in the overlap, stitching clusters across chunk boundaries. Noise (-1) is
    never merged.

    Args:
        df:         DataFrame with event data.
        chunk_size: Maximum number of events per GPU batch (includes overlap).
                    Reduce if you see OOM errors.
        overlap:    Number of events shared between adjacent chunks.
                    Set to roughly the expected maximum cluster size in events
                    so boundary-spanning clusters are properly linked.
                    0 disables stitching (clusters may be split at boundaries).

    Returns:
        Copy of df with an integer `label_col` column appended.
        Cluster IDs are contiguous non-negative integers; noise is -1.
    """
    if not HAS_CUML:
        raise ImportError(
            "cuML is not installed. Install RAPIDS: https://rapids.ai/start.html"
        )

    stride = chunk_size - overlap
    if stride <= 0:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    n = len(df)
    n_chunks = (n + stride - 1) // stride
    print(f"Chunked GPU HDBSCAN: {n:,} events, {n_chunks} chunk(s), "
          f"chunk_size={chunk_size:,}, overlap={overlap:,}...")

    all_labels = np.full(n, -1, dtype=np.int32)
    next_cluster_id = 0

    # Union-find (path-compressed, dict-based)
    uf_parent: dict = {}

    def uf_find(x: int) -> int:
        root = x
        while uf_parent.get(root, root) != root:
            root = uf_parent.get(root, root)
        while uf_parent.get(x, x) != root:
            uf_parent[x], x = root, uf_parent.get(x, x)
        return root

    def uf_union(a: int, b: int) -> None:
        a, b = uf_find(a), uf_find(b)
        if a != b:
            uf_parent[max(a, b)] = min(a, b)

    prev_overlap_labels = None  # offset labels for the tail of the previous chunk

    for i in range(n_chunks):
        start = i * stride
        end = min(start + chunk_size, n)
        core_end = min(start + stride, n)  # events [start, core_end) are "owned" by this chunk
        core_len = core_end - start

        chunk = df.iloc[start:end]
        features = build_feature_matrix(chunk, x_col, y_col, t_col, time_scale)
        features_gpu = cp.asarray(features)

        clusterer = cuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples if min_samples is not None else min_cluster_size,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
            metric=metric,
        )
        raw_labels = cp.asnumpy(clusterer.fit_predict(features_gpu)).astype(np.int32)
        del features_gpu

        # Offset to make IDs globally unique before stitching
        chunk_labels = raw_labels.copy()
        mask = chunk_labels >= 0
        if mask.any():
            chunk_labels[mask] += next_cluster_id
            next_cluster_id = int(chunk_labels[mask].max()) + 1

        # Stitch: events [start, start+overlap) appear in prev chunk's tail too
        if prev_overlap_labels is not None and overlap > 0:
            actual_overlap = min(overlap, len(prev_overlap_labels), len(chunk_labels))
            for prev_lbl, curr_lbl in zip(
                prev_overlap_labels[:actual_overlap],
                chunk_labels[:actual_overlap],
            ):
                if prev_lbl >= 0 and curr_lbl >= 0:
                    uf_union(int(prev_lbl), int(curr_lbl))

        # Write only the core region; overlap tail is re-clustered in next chunk
        all_labels[start:core_end] = chunk_labels[:core_len]

        # Last chunk: also write the tail (nothing will re-process it)
        if i == n_chunks - 1 and end > core_end:
            all_labels[core_end:end] = chunk_labels[core_len:]

        prev_overlap_labels = chunk_labels[core_len:] if end > core_end else None

        n_clustered = int((chunk_labels[:core_len] >= 0).sum())
        print(f"  Chunk {i+1}/{n_chunks}: {core_len:,} core events, "
              f"{n_clustered:,} clustered, {next_cluster_id:,} raw cluster IDs so far")

    # Apply union-find: build a compact O(next_cluster_id) lookup, then apply in O(n)
    if next_cluster_id > 0:
        lookup = np.array([uf_find(lbl) for lbl in range(next_cluster_id)], dtype=np.int32)
        _, inverse = np.unique(lookup, return_inverse=True)
        lookup = inverse.astype(np.int32)  # root → contiguous ID

        valid = all_labels >= 0
        all_labels[valid] = lookup[all_labels[valid]]
        n_final = int(lookup.max()) + 1
        print(f"After stitching: {n_final:,} final clusters "
              f"(merged from {next_cluster_id:,} raw IDs)")

    out = df.copy()
    out[label_col] = all_labels
    return out


def cluster_events(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    time_scale: float = 1e-3,
    min_cluster_size: int = 10,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    metric: str = 'euclidean',
    label_col: str = 'cluster',
    use_gpu: bool = False,
    chunk_size: int = None,
    overlap: int = 0,
) -> pd.DataFrame:
    """
    Cluster event camera data with HDBSCAN.

    Dispatches to the GPU implementation when use_gpu=True, otherwise uses
    the CPU implementation.

    Args:
        df:                         DataFrame with at least x, y, t columns.
        x_col:                      Column name for x coordinate.
        y_col:                      Column name for y coordinate.
        t_col:                      Column name for timestamp.
        time_scale:                 Multiplicative factor applied to t before
                                    clustering. Tunes how much temporal
                                    separation contributes to distance.
                                    Example: 1e-3 for µs timestamps gives
                                    1 µs = 0.001 px equivalent distance.
        min_cluster_size:           Minimum number of events per cluster.
                                    Increase to suppress small clusters.
        min_samples:                Minimum neighbourhood size to be a core
                                    point. Defaults to min_cluster_size.
                                    Lower values create more clusters.
        cluster_selection_epsilon:  Merge sub-clusters within this distance.
                                    0 disables (default).
        cluster_selection_method:   'eom' (default) keeps the excess-of-mass
                                    clusters; 'leaf' returns finer clusters.
        metric:                     Distance metric ('euclidean', 'l1', ...).
        label_col:                  Name added to the returned DataFrame.
        use_gpu:                    Use cuML GPU backend if True.
        chunk_size:                 If set, process the dataset in chunks of
                                    this many events on the GPU. Useful when
                                    the full dataset exceeds GPU memory.
                                    Ignored when use_gpu=False.

    Returns:
        Copy of df with an integer `label_col` column.
        Noise points have label -1.

    Example:
        >>> df = extract('events.h5')
        >>> result = cluster_events(df, time_scale=5e-4, min_cluster_size=20)
        >>> n_clusters = result['cluster'].nunique() - (1 if -1 in result['cluster'].values else 0)
    """
    kwargs = dict(
        df=df,
        x_col=x_col,
        y_col=y_col,
        t_col=t_col,
        time_scale=time_scale,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
        label_col=label_col,
    )
    if use_gpu:
        if not HAS_CUML:
            print("Warning: GPU/cuML not available, falling back to CPU HDBSCAN.")
        elif chunk_size is not None:
            try:
                return run_hdbscan_gpu_chunked(**kwargs, chunk_size=chunk_size, overlap=overlap)
            except Exception as e:
                raise RuntimeError(
                    f"Chunked GPU clustering failed on chunk (chunk_size={chunk_size}): "
                    f"{type(e).__name__}: {e}\n"
                    f"Try a smaller chunk_size."
                ) from e
        else:
            try:
                return run_hdbscan_gpu(**kwargs)
            except MemoryError as e:
                raise RuntimeError(
                    f"GPU out of memory for {len(df):,} events. "
                    f"Re-run with chunk_size=2_000_000 to process in batches."
                ) from e
            except Exception as e:
                print(f"Warning: GPU clustering failed ({type(e).__name__}: {e}), falling back to CPU HDBSCAN.")
    return run_hdbscan(**kwargs)


def cluster_summary(df: pd.DataFrame, label_col: str = 'cluster') -> pd.DataFrame:
    """
    Return per-cluster statistics from a clustered event DataFrame.

    Args:
        df:        DataFrame returned by cluster_events (must have label_col).
        label_col: Name of the cluster label column.

    Returns:
        DataFrame indexed by cluster label with columns:
            n_events, x_min, x_max, y_min, y_max, t_min, t_max, t_span.
        The noise cluster (-1) is included if present.
    """
    groups = df.groupby(label_col)
    summary = pd.DataFrame({
        'n_events': groups.size(),
        'x_min':    groups['x'].min(),
        'x_max':    groups['x'].max(),
        'y_min':    groups['y'].min(),
        'y_max':    groups['y'].max(),
        't_min':    groups['t'].min(),
        't_max':    groups['t'].max(),
    })
    summary['t_span'] = summary['t_max'] - summary['t_min']
    return summary

if __name__ == "__main__":
    # Example usage
    from extract import extract,save_df

    df = extract('data/val_day_014_td.h5')
    clustered = cluster_events(df, time_scale=2000, min_cluster_size=100, use_gpu=True, chunk_size=50_000,overlap=5_000)

    save_df(clustered, 'data/val_day_014_td_clustered.parquet')

    summary = cluster_summary(clustered)
    print(summary.head())