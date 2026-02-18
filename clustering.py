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
        return run_hdbscan_gpu(**kwargs)
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
    clustered = cluster_events(df, time_scale=10000, min_cluster_size=50,use_gpu=True)

    save_df(clustered, 'data/val_day_014_td_clustered.parquet')

    summary = cluster_summary(clustered)
    print(summary.head())