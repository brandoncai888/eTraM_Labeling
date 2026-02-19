import numpy as np
import pandas as pd

try:
    import hdbscan as hdbscan_lib
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    import cupy as cp
    HAS_CUML = True
except ImportError:
    HAS_CUML = False

# Shared chunked-window + union-find stitching helper
from stdbscan import _run_chunked


# ---------------------------------------------------------------------------
# CPU HDBSCAN
# ---------------------------------------------------------------------------

def run_hdbscan(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    spatial_threshold: float = 10_000.0,
    min_cluster_size: int = 10,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    metric: str = 'euclidean',
    label_col: str = 'cluster',
) -> pd.DataFrame:
    """
    CPU HDBSCAN on event camera data in (x, y, t) space.

    The temporal axis is scaled as t_scaled = t / spatial_threshold so that
    `spatial_threshold` timestamp units ≈ 1 pixel of spatial distance.

    HDBSCAN does not require an explicit radius parameter.  It finds clusters
    of varying density and labels sparse points as noise (-1).

    Args:
        df:                         DataFrame with event data, sorted by time.
        x_col:                      Column for x coordinate (pixels).
        y_col:                      Column for y coordinate (pixels).
        t_col:                      Column for timestamp.
        spatial_threshold:          Temporal-to-spatial scale factor.
                                    t_scaled = t / spatial_threshold.
                                    10 000 means 10 000 µs ≡ 1 pixel.
        min_cluster_size:           Minimum events to form a cluster.
                                    Primary tuning knob — increase to suppress
                                    small / noisy clusters.
        min_samples:                Neighbourhood size for core-point test.
                                    Defaults to min_cluster_size. Lower values
                                    create more (noisier) clusters.
        cluster_selection_epsilon:  Merge sub-clusters within this distance
                                    (in scaled feature space).  0 disables.
                                    Useful for merging fragments of the same
                                    object that split due to occlusion.
        cluster_selection_method:   'eom' (excess-of-mass, default) or 'leaf'
                                    for finer, more uniform clusters.
        metric:                     Distance metric for HDBSCAN.
        label_col:                  Output column name.

    Returns:
        Copy of df with integer label_col.  Noise = -1.
    """
    if not HAS_HDBSCAN:
        raise ImportError(
            "hdbscan is not installed. Install it with: pip install hdbscan"
        )

    x = df[x_col].to_numpy(np.float32)
    y = df[y_col].to_numpy(np.float32)
    t = (df[t_col].to_numpy(np.float64) / spatial_threshold).astype(np.float32)
    features = np.column_stack([x, y, t])

    print(f"CPU HDBSCAN on {len(df):,} events "
          f"(spatial_threshold={spatial_threshold}, "
          f"min_cluster_size={min_cluster_size}, "
          f"cluster_selection_epsilon={cluster_selection_epsilon})...")

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
    )
    labels = clusterer.fit_predict(features).astype(np.int32)

    n_clusters = int((labels >= 0).any() and labels.max() + 1)
    n_noise = int((labels == -1).sum())
    print(f"Found {n_clusters:,} clusters, {n_noise:,} noise points "
          f"({100 * n_noise / len(df):.1f}%)")

    out = df.copy()
    out[label_col] = labels
    return out


# ---------------------------------------------------------------------------
# GPU HDBSCAN  (cuML, Euclidean metric)
# ---------------------------------------------------------------------------

def run_hdbscan_gpu(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    spatial_threshold: float = 10_000.0,
    min_cluster_size: int = 10,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    label_col: str = 'cluster',
    chunk_size: int = None,
    overlap: int = None,
    min_overlap_frac: float = 0.0,
    max_centroid_dist: float = float('inf'),
) -> pd.DataFrame:
    """
    GPU-accelerated HDBSCAN using cuML (requires RAPIDS).

    Same temporal scaling as run_hdbscan.  cuML uses Euclidean distance only.
    Supports optional chunked processing for datasets that exceed GPU memory
    or cuML's internal int32 index limit (~500k–2M events depending on GPU).

    Args:
        df:                         DataFrame with event data, sorted by time.
        x_col:                      Column for x coordinate (pixels).
        y_col:                      Column for y coordinate (pixels).
        t_col:                      Column for timestamp.
        spatial_threshold:          Temporal-to-spatial scale (t = t / spatial_threshold).
        min_cluster_size:           Minimum events per cluster.
        min_samples:                Core-point neighbourhood size (defaults to
                                    min_cluster_size).
        cluster_selection_epsilon:  Sub-cluster merge radius (scaled space). 0 = off.
        cluster_selection_method:   'eom' or 'leaf'.
        label_col:                  Output column name.
        chunk_size:                 Process in sliding windows of this many events.
                                    Use when the dataset exceeds GPU memory.
                                    None = full dataset in one pass.
        overlap:                    Events shared between adjacent windows for
                                    cluster stitching (default: chunk_size // 5).

    Returns:
        Copy of df with integer label_col.  Noise = -1.

    Raises:
        ImportError: If cuML is not installed.
    """
    if not HAS_CUML:
        raise ImportError(
            "cuML is not installed. Install RAPIDS: https://rapids.ai/start.html"
        )

    _min_samples = min_samples if min_samples is not None else min_cluster_size

    def _gpu_chunk(df_slice: pd.DataFrame) -> np.ndarray:
        x = df_slice[x_col].to_numpy(np.float32)
        y = df_slice[y_col].to_numpy(np.float32)
        t = (df_slice[t_col].to_numpy(np.float64) / spatial_threshold).astype(np.float32)
        features_gpu = cp.asarray(np.column_stack([x, y, t]))
        clusterer = cuHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=_min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            cluster_selection_method=cluster_selection_method,
        )
        labels = cp.asnumpy(clusterer.fit_predict(features_gpu)).astype(np.int32)
        del features_gpu
        return labels

    if chunk_size is not None:
        ov = overlap if overlap is not None else chunk_size // 5
        return _run_chunked(df, _gpu_chunk, chunk_size, ov, label_col, 'GPU HDBSCAN',
                            x_col=x_col, y_col=y_col,
                            min_overlap_frac=min_overlap_frac,
                            max_centroid_dist=max_centroid_dist)

    print(f"[GPU] HDBSCAN on {len(df):,} events "
          f"(spatial_threshold={spatial_threshold}, "
          f"min_cluster_size={min_cluster_size}, "
          f"cluster_selection_epsilon={cluster_selection_epsilon})...")

    labels = _gpu_chunk(df)

    n_clusters = int((labels >= 0).any() and labels.max() + 1)
    n_noise = int((labels == -1).sum())
    print(f"Found {n_clusters:,} clusters, {n_noise:,} noise points "
          f"({100 * n_noise / len(df):.1f}%)")

    out = df.copy()
    out[label_col] = labels
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def cluster_events_hdbscan(
    df: pd.DataFrame,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    spatial_threshold: float = 10_000.0,
    min_cluster_size: int = 10,
    min_samples: int = None,
    cluster_selection_epsilon: float = 0.0,
    cluster_selection_method: str = 'eom',
    metric: str = 'euclidean',
    label_col: str = 'cluster',
    use_gpu: bool = False,
    chunk_size: int = None,
    overlap: int = None,
    min_overlap_frac: float = 0.0,
    max_centroid_dist: float = float('inf'),
) -> pd.DataFrame:
    """
    Cluster event camera data with HDBSCAN.

    Dispatches to GPU (cuML Euclidean) or CPU (hdbscan library).
    Supports chunked GPU processing for large datasets.

    Args:
        df:                         DataFrame with x_col, y_col, t_col, sorted
                                    by time.
        x_col:                      Column for x coordinate (pixels).
        y_col:                      Column for y coordinate (pixels).
        t_col:                      Column for timestamp.
        spatial_threshold:          Temporal-to-spatial scale factor applied as
                                    t_scaled = t / spatial_threshold.
                                    10 000 means 10 000 µs ≡ 1 pixel.
                                    Decrease to make clustering more temporally
                                    sensitive; increase to span longer windows.
        min_cluster_size:           Minimum number of events per cluster.
                                    Primary tuning parameter.  Increase to
                                    reject small clusters / background noise.
        min_samples:                Core-point neighbourhood size.  Defaults to
                                    min_cluster_size.  Lower = more clusters,
                                    higher = stricter noise rejection.
        cluster_selection_epsilon:  After HDBSCAN, merge cluster pairs whose
                                    centres are within this distance (scaled
                                    feature space).  Useful for re-joining
                                    object fragments split by occlusion or
                                    density changes.  0 = disabled (default).
        cluster_selection_method:   'eom' (default) for larger, robust clusters;
                                    'leaf' for finer, more uniform clusters.
        metric:                     (CPU only) distance metric.  cuML always
                                    uses Euclidean.
        label_col:                  Output cluster column name.
        use_gpu:                    Route to GPU (cuML) backend when True.
                                    Falls back to CPU if cuML is unavailable.
        chunk_size:                 (GPU only) Sliding window size in events.
                                    Use when the full dataset exceeds GPU
                                    memory or cuML's int32 limit (~500k–2M).
                                    None = full dataset in one GPU pass.
        overlap:                    (GPU only) Events shared between adjacent
                                    windows for cluster stitching.
                                    Default: chunk_size // 5.
        min_overlap_frac:           (GPU chunked only) Only merge two clusters
                                    across a chunk boundary if their co-occurring
                                    overlap events are at least this fraction of
                                    the smaller cluster's overlap presence.
                                    Prevents a single bridge point from merging
                                    two large separate clusters.  0.0 = off.
        max_centroid_dist:          (GPU chunked only) Only merge two clusters
                                    if their centroids in the overlap region are
                                    within this many pixels of each other.
                                    inf = off.

    Returns:
        Copy of df with integer label_col.  Noise = -1.

    Examples:
        # GPU, large dataset, with merge guards
        clustered = cluster_events_hdbscan(
            df,
            spatial_threshold=5_000,
            min_cluster_size=50,
            cluster_selection_epsilon=2.0,
            use_gpu=True,
            chunk_size=300_000,
            overlap=25_000,
            min_overlap_frac=0.3,
            max_centroid_dist=15.0,
        )

        # CPU
        clustered = cluster_events_hdbscan(
            df,
            spatial_threshold=10_000,
            min_cluster_size=20,
        )
    """
    gpu_kwargs = dict(
        df=df,
        x_col=x_col, y_col=y_col, t_col=t_col,
        spatial_threshold=spatial_threshold,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        label_col=label_col,
        chunk_size=chunk_size,
        overlap=overlap,
        min_overlap_frac=min_overlap_frac,
        max_centroid_dist=max_centroid_dist,
    )
    cpu_kwargs = dict(
        df=df,
        x_col=x_col, y_col=y_col, t_col=t_col,
        spatial_threshold=spatial_threshold,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method=cluster_selection_method,
        metric=metric,
        label_col=label_col,
    )

    if use_gpu:
        if not HAS_CUML:
            print("Warning: cuML not available, falling back to CPU HDBSCAN.")
        else:
            try:
                return run_hdbscan_gpu(**gpu_kwargs)
            except RuntimeError as e:
                if chunk_size is None and 'CUDA' in str(e):
                    auto_chunk = 500_000
                    print(f"Warning: GPU HDBSCAN failed (CUDA error), "
                          f"retrying with chunk_size={auto_chunk:,}...")
                    return run_hdbscan_gpu(**{**gpu_kwargs, 'chunk_size': auto_chunk})
                print(f"Warning: GPU HDBSCAN failed ({type(e).__name__}: {e}), "
                      f"falling back to CPU.")
            except Exception as e:
                print(f"Warning: GPU HDBSCAN failed ({type(e).__name__}: {e}), "
                      f"falling back to CPU.")

    return run_hdbscan(**cpu_kwargs)


# ---------------------------------------------------------------------------
# Summary utility
# ---------------------------------------------------------------------------

def cluster_summary(df: pd.DataFrame, label_col: str = 'cluster') -> pd.DataFrame:
    """
    Per-cluster statistics from a clustered event DataFrame.

    Args:
        df:        DataFrame with cluster labels (must have x, y, t, label_col).
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


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from extract import extract, save_df

    df = extract('data/val_day_014_td.parquet')
    print(f"Loaded {len(df):,} events, columns: {list(df.columns)}")


    clustered = cluster_events_hdbscan(
        df,
        spatial_threshold=2_000.0,          # 5000 µs ≡ 1 pixel
        min_cluster_size=50,                 # reject clusters < 200 events
        min_samples=20,                      # stricter noise rejection
        cluster_selection_epsilon=2.0,       # merge fragments within 2 px
        cluster_selection_method='eom',
        use_gpu=True,
        chunk_size=100_000,                  # fit in GPU memory
        overlap=10_000,                      # stitch clusters across windows
        min_overlap_frac=0.6,                # require substantial overlap for stitching
        max_centroid_dist=15.0,              # only stitch if centroids are close
    )

    save_df(clustered, 'data/val_day_014_td_hdbscan_small_min_samples.parquet')


    summary = cluster_summary(clustered)
    print(summary.head(20))
