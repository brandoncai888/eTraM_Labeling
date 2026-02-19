import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

from extract import extract

import open3d as o3d
import numpy as np
import pandas as pd

import open3d as o3d
import numpy as np
import pandas as pd

import open3d as o3d
import numpy as np
import pandas as pd

def plot_3d_open3d(
    source,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    p_col: str = 'p',
    id_col: str = 'id',
    h5_key: str = None,
    t_start: float = None,
    t_end: float = None,
    max_points: int = 1_000_000,
    color_by_polarity: bool = False,
    color_by_id: bool = False,
    point_size: float = 0.1,
    show_axes: bool = True
):
    """
    Visualize event camera data in 3D with support for Polarity or ID-based coloring.
    """
    # 1. Load data
    df = extract(source, h5_key) if isinstance(source, str) else source

    # 2. Filter by Time Range
    if t_start is not None:
        df = df[df[t_col] >= t_start]
    if t_end is not None:
        df = df[df[t_col] <= t_end]

    # 3. Slice the first max_points
    if max_points is not None and len(df) > max_points:
        df = df.iloc[:max_points]

    if df.empty:
        print("No events found.")
        return

    # 4. Extract coordinates
    points = np.zeros((len(df), 3))
    points[:, 0] = df[x_col].values
    points[:, 1] = df[t_col].values/10000.0  # Scale time for better visualization

    # Use display height to invert image Y such that displayed Z = (height - y).
    # Prefer a `height` column in the dataframe; otherwise default to 720.
    if 'height' in df.columns:
        display_height = int(df['height'].iloc[0])
    else:
        display_height = 720

    points[:, 2] = display_height - df[y_col].values

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 5. Handle Coloring
    colors = np.zeros((len(df), 3))

    if color_by_id and id_col in df.columns:
        ids = df[id_col].values
        unique_ids = np.unique(ids)
        
        # Generate a deterministic color palette for unique IDs
        rng = np.random.default_rng(seed=42)
        palette = rng.random((len(unique_ids), 3))

        # Noise events (id == -1) are always black
        if -1 in unique_ids:
            palette[np.searchsorted(unique_ids, -1)] = [0.0, 0.0, 0.0]

        # Map IDs to their corresponding color in the palette
        # id_map creates an index for each unique ID
        id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
        indices = np.array([id_to_idx[uid] for uid in ids])
        colors = palette[indices]
        
    elif color_by_polarity and p_col in df.columns:
        p = df[p_col].values
        colors[p == 1] = [0.25, 0.41, 0.88] # RoyalBlue
        colors[p == 0] = [1.0, 0.38, 0.28]  # Tomato
    else:
        colors[:] = [0.1, 0.1, 0.1] # Dark Grey default

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 6. No additional coordinate flip needed; Z already set to (height - y)

    # 7. Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Events: {len(df):,} pts", width=1280, height=720)
    vis.add_geometry(pcd)
    
    if show_axes:
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=df[x_col].max() * 0.2, origin=[0, 0, 0]
        )
        vis.add_geometry(axes)

    vis.get_render_option().point_size = point_size
    vis.get_render_option().background_color = np.asarray([1, 1, 1]) 
    
    vis.run()
    vis.destroy_window()


def animate_frames(
    source,
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    p_col: str = 'p',
    id_col: str = 'id',
    color_by_id: bool = False,
    h5_key: str = None,
    dt: int = 10_000,
    width: int = None,
    height: int = None,
    color_by_polarity: bool = False,
    interval: int = 50,
    figsize: tuple = (10, 6),
    save_path: str = None,
    fps: int = 20,
    max_frames: int = None,
):
    """
    Animate event camera data as a sequence of frames.

    Each frame accumulates all events within a `dt`-wide time window into a
    numpy image array and renders it with imshow. Render time is proportional
    to the sensor resolution, not the number of events.

    Args:
        source:             Path to an event file, or a DataFrame already loaded.
        x_col:              Column name for x.
        y_col:              Column name for y.
        t_col:              Column name for timestamp.
        p_col:              Column name for polarity.
        h5_key:             HDF5 group key for .h5 files.
        dt:                 Time window per frame in the same units as t.
        width:              Sensor width in pixels. Auto-detected from data if None.
        height:             Sensor height in pixels. Auto-detected from data if None.
        color_by_polarity:  If True, ON events are blue and OFF events are red.
                            If False, all events are black dots on white.
        interval:           Delay between frames in milliseconds (for display).
        figsize:            Figure size in inches.
        save_path:          Save animation to this path (.gif or .mp4).
                            Requires pillow (gif) or ffmpeg (mp4).
        fps:                Frames per second when saving.
        max_frames:         Cap the total number of frames rendered. None = all.

    Returns:
        matplotlib.animation.FuncAnimation object.
    """
    df = extract(source, h5_key) if isinstance(source, str) else source
    df = df.sort_values(t_col).reset_index(drop=True)

    t_vals = df[t_col].to_numpy()
    x_vals = df[x_col].to_numpy(dtype=np.intp)
    y_vals = df[y_col].to_numpy(dtype=np.intp)
    p_vals = df[p_col].to_numpy() if p_col in df.columns else None
    id_vals = df[id_col].to_numpy() if id_col in df.columns else None

    # Pre-compute deterministic palette mapping for IDs if requested
    if color_by_id and id_vals is not None:
        unique_ids = np.unique(id_vals)
        rng = np.random.default_rng(seed=42)
        palette = rng.random((len(unique_ids), 3))

        # Noise events (id == -1) are always black
        if -1 in unique_ids:
            palette[np.searchsorted(unique_ids, -1)] = [0.0, 0.0, 0.0]

        id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
        indices_all = np.array([id_to_idx[uid] for uid in id_vals])

    if width is None:
        width = int(df['width'].iloc[0]) if 'width' in df.columns else int(x_vals.max()) + 1
    if height is None:
        height = int(df['height'].iloc[0]) if 'height' in df.columns else int(y_vals.max()) + 1

    boundaries = np.arange(int(t_vals[0]), int(t_vals[-1]) + dt, dt)
    n_frames = len(boundaries) - 1
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    # Pre-compute frame boundaries once
    left_indices  = np.searchsorted(t_vals, boundaries[:n_frames],      side='left')
    right_indices = np.searchsorted(t_vals, boundaries[1:n_frames + 1], side='left')

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.1)
    ax.set_axis_off()

    state = {'frame': 0, 'paused': False}

    def frame_gen():
        while True:
            yield state['frame']
            if not state['paused']:
                state['frame'] = (state['frame'] + 1) % n_frames

    pause_ax  = fig.add_axes([0.10, 0.02, 0.13, 0.05])
    back5_ax  = fig.add_axes([0.25, 0.02, 0.13, 0.05])
    ff15_ax   = fig.add_axes([0.40, 0.02, 0.13, 0.05])
    back15_ax = fig.add_axes([0.55, 0.02, 0.13, 0.05])
    ff30_ax   = fig.add_axes([0.70, 0.02, 0.13, 0.05])

    pause_btn  = Button(pause_ax,  'Pause')
    back5_btn  = Button(back5_ax,  '◀◀ -5')
    ff15_btn   = Button(ff15_ax,   '+15 ▶▶')
    back15_btn = Button(back15_ax, '◀◀ -15')
    ff30_btn   = Button(ff30_ax,   '+30 ▶▶')

    def toggle(_event):
        state['paused'] = not state['paused']
        if state['paused']:
            ani.pause()
            pause_btn.label.set_text('Resume')
        else:
            ani.resume()
            pause_btn.label.set_text('Pause')
        fig.canvas.draw_idle()

    def make_jump(delta):
        def _jump(_event):
            state['frame'] = max(0, min(state['frame'] + delta, n_frames - 1))
            update(state['frame'])
            fig.canvas.draw_idle()
        return _jump

    pause_btn.on_clicked(toggle)
    back5_btn.on_clicked(make_jump(-5))
    ff15_btn.on_clicked(make_jump(15))
    back15_btn.on_clicked(make_jump(-15))
    ff30_btn.on_clicked(make_jump(30))

    # Initialise with a white canvas; origin='upper' puts row 0 (y=0) at top
    # If coloring by polarity or by id we need an RGB canvas, otherwise single channel
    if (color_by_polarity and p_vals is not None) or (color_by_id and id_vals is not None):
        init_img = np.ones((height, width, 3), dtype=np.float32)
    else:
        init_img = np.ones((height, width), dtype=np.float32)

    im    = ax.imshow(init_img, cmap='gray_r', vmin=0, vmax=1, origin='upper',
                      interpolation='nearest')
    title = ax.set_title('')

    def update(frame_idx):
        lo = left_indices[frame_idx]
        hi = right_indices[frame_idx]

        xs = x_vals[lo:hi]
        ys = y_vals[lo:hi]

        # Color by ID (deterministic palette) takes precedence when requested
        if 'id_vals' in locals() and color_by_id and id_vals is not None:
            img = np.ones((height, width, 3), dtype=np.float32)   # white
            ids_idx = indices_all[lo:hi]
            if ids_idx.size:
                img[ys, xs] = palette[ids_idx]

        elif color_by_polarity and p_vals is not None:
            img = np.ones((height, width, 3), dtype=np.float32)   # white
            ps  = p_vals[lo:hi]
            on  = ps == 1
            img[ys[on],  xs[on]]  = [0.22, 0.49, 0.72]   # blue  — ON
            img[ys[~on], xs[~on]] = [0.84, 0.15, 0.16]   # red   — OFF
        else:
            img = np.zeros((height, width), dtype=np.uint8)
            np.add.at(img, (ys, xs), 1)
            img = np.clip(img, 0, 1)                       # binary: any event → black

        im.set_data(img)
        title.set_text(
            f'Frame {frame_idx + 1}/{n_frames}   '
            f't = [{boundaries[frame_idx]:,.0f} – {boundaries[frame_idx + 1]:,.0f}] µs   '
            f'events: {hi - lo:,}'
        )
        return im, title

    ani = animation.FuncAnimation(
        fig, update, frames=frame_gen(), interval=interval, blit=True,
        cache_frame_data=False,
    )

    if save_path is not None:
        ext = save_path.rsplit('.', 1)[-1].lower()
        writer = 'pillow' if ext == 'gif' else 'ffmpeg'
        ani.save(save_path, writer=writer, fps=fps)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()

    return ani


if __name__ == '__main__':
    FILE = 'data/val_day_014_td_stdbscan_small_eps.parquet'

    # 3-D scatter of a random subset
    plot_3d_open3d(FILE, max_points=42240827, t_start=0, t_end=63037503, color_by_id=True, id_col="cluster")

    # Animated frames
    animate_frames(FILE, dt=100_000, max_frames=1892, color_by_id=True, id_col="cluster")
