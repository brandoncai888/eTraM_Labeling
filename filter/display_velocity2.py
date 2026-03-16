import time
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.patches import FancyArrowPatch
import open3d as o3d

from extract import extract

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
    points[:, 1] = df[t_col].values / 1000.0  # Scale time for better visualization

    # Use display height to invert image Y such that displayed Z = (height - y).
    if 'height' in df.columns:
        display_height = int(df['height'].iloc[0])
    else:
        display_height = 720

    points[:, 2] = df[y_col].values

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

        # Explicit color mapping for specific IDs
        id_color_map = {
            0: [1.0, 0.0, 0.0],    # red
            1: [0.0, 0.0, 1.0],    # blue
            2: [0.0, 1.0, 0.0],    # green
            3: [0.8, 0.8, 0.0],    # yellow
        }

        # Noise events (id == -1) are always black
        if -1 in unique_ids:
            palette[np.searchsorted(unique_ids, -1)] = [0.0, 0.0, 0.0]

        id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
        
        for uid, col in id_color_map.items():
            if uid in id_to_idx:
                palette[id_to_idx[uid]] = col

        indices = np.array([id_to_idx[uid] for uid in ids])
        colors = palette[indices]
        
    elif color_by_polarity and p_col in df.columns:
        p = df[p_col].values
        colors[p == 1] = [0.25, 0.41, 0.88] # RoyalBlue
        colors[p == 0] = [1.0, 0.38, 0.28]  # Tomato
    else:
        colors[:] = [0.1, 0.1, 0.1] # Dark Grey default

    pcd.colors = o3d.utility.Vector3dVector(colors)

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

    vis.poll_events()
    vis.update_renderer()
    ctr = vis.get_view_control()
    ctr.set_front([0.026849934981634876, 0.99938368720401627, -0.022612535063256663])
    ctr.set_lookat([354.3481600111744, 1986.9705234695316, 303.55364828999643])
    ctr.set_up([0.066193756836045309, -0.024348526697291598, -0.99750966702263177])
    ctr.set_zoom(0.64000000000000012)

    vis.run()
    vis.destroy_window()


def animate_frames(
    source,
    velocities_source: str = None,
    velocity_method: str = 'decay_votes',
    velocity_scale: float = 20.0,
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
    color_by_polarity: bool = True,
    interval: int = 50,
    figsize: tuple = (10, 6),
    save_path: str = None,
    fps: int = 20,
    max_frames: int = None,
    exclude_ids: list = None,
    show_velocity_arrows: bool = True,
):
    """
    Animate event camera data as a sequence of frames, with pre-computed velocities.
    """
    df = extract(source, h5_key) if isinstance(source, str) else source
    df = df.sort_values(t_col).reset_index(drop=True)

    t_vals = df[t_col].to_numpy()
    x_vals = df[x_col].to_numpy(dtype=np.intp)
    y_vals = df[y_col].to_numpy(dtype=np.intp)
    p_vals = df[p_col].to_numpy() if p_col in df.columns else None
    id_vals = df[id_col].to_numpy() if id_col in df.columns else None

    # Load and pre-process velocities into a fast dictionary format
    vel_dict = None
    if show_velocity_arrows and velocities_source is not None:
        print(f"Loading velocities from {velocities_source}...")
        df_vel = pd.read_parquet(velocities_source)
        vel_dict = {}
        # Group by cluster ID for fast lookup later
        cluster_col = 'cluster_id' if 'cluster_id' in df_vel.columns else 'cluster'
        for cid, group in df_vel.groupby(cluster_col):
            t_arr = group['t'].values
            # Extract the [vx, vy] lists into a 2D numpy array
            v_arr = np.stack(group[velocity_method].values)
            vel_dict[cid] = (t_arr, v_arr)

    # Filter out excluded IDs
    if exclude_ids is not None and id_vals is not None:
        mask = ~np.isin(id_vals, exclude_ids)
        t_vals = t_vals[mask]
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
        p_vals = p_vals[mask] if p_vals is not None else None
        id_vals = id_vals[mask]

    # Pre-compute deterministic palette mapping
    if color_by_id and id_vals is not None:
        unique_ids = np.unique(id_vals)
        rng = np.random.default_rng(seed=42)
        palette = rng.random((len(unique_ids), 3))

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

    left_indices  = np.searchsorted(t_vals, boundaries[:n_frames],      side='left')
    right_indices = np.searchsorted(t_vals, boundaries[1:n_frames + 1], side='left')

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.1)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip y-axis to match image origin at top
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, alpha=0.3)

    state = {'frame': 0, 'paused': False}
    fps_state = {'last': None, 'history': deque(maxlen=30), 'count': 0}

    def frame_gen():
        while True:
            yield state['frame']
            if not state['paused']:
                state['frame'] = (state['frame'] + 1) % n_frames

    btn_width = 0.12
    btn_height = 0.05
    btn_bottom = 0.03 
    
    back15_ax = fig.add_axes([0.30, btn_bottom, btn_width, btn_height])
    pause_ax  = fig.add_axes([0.44, btn_bottom, btn_width, btn_height])
    ff15_ax   = fig.add_axes([0.58, btn_bottom, btn_width, btn_height])

    pause_btn  = Button(pause_ax,  'Pause')
    ff15_btn   = Button(ff15_ax,   '▶▶ +20')
    back15_btn = Button(back15_ax, '◀◀ -20')

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
    ff15_btn.on_clicked(make_jump(20))
    back15_btn.on_clicked(make_jump(-20))
    
    if (color_by_polarity and p_vals is not None) or (color_by_id and id_vals is not None):
        init_img = np.ones((height, width, 3), dtype=np.float32)
    else:
        init_img = np.ones((height, width), dtype=np.float32)

    im = ax.imshow(init_img, cmap='gray_r', vmin=0, vmax=1, origin='upper', interpolation='nearest')
    
    MAX_ARROWS = 100
    arrow_pool = []
    for _ in range(MAX_ARROWS):
        a = FancyArrowPatch((0,0),(0,0), arrowstyle='->', mutation_scale=15, linewidth=1.5, color='black')
        a.set_visible(False)
        ax.add_patch(a)
        arrow_pool.append(a)
    title = ax.set_title('')

    def update(frame_idx):
        _t = time.perf_counter()
        if fps_state['last'] is not None:
            fps_state['history'].append(_t - fps_state['last'])
        fps_state['last'] = _t
        fps_state['count'] += 1

        live_fps = (1.0 / (sum(fps_state['history']) / len(fps_state['history']))
                    if fps_state['history'] else 0.0)
        if fps_state['count'] % 30 == 0:
            print(f"  Frame {frame_idx + 1}/{n_frames} | FPS: {live_fps:.1f}")

        lo = left_indices[frame_idx]
        hi = right_indices[frame_idx]
        current_t = boundaries[frame_idx]

        xs = x_vals[lo:hi]
        ys = y_vals[lo:hi]
        ids = id_vals[lo:hi] if id_vals is not None else None
        ps = p_vals[lo:hi] if p_vals is not None else None

        for a in arrow_pool:
            a.set_visible(False)
            
        # Draw Arrows using pre-computed velocities
        if show_velocity_arrows and ids is not None and vel_dict is not None:
            ids_unique = np.unique(ids)
            for i, cid in enumerate(ids_unique):
                if cid == -1 or i >= MAX_ARROWS: # skip noise and overflow
                    continue
                    
                xs_i = xs[ids == cid]
                ys_i = ys[ids == cid]
                
                # Center of Mass for arrow start
                xs_i_mean = xs_i.mean() 
                ys_i_mean = ys_i.mean()
                
                u, v = 0.0, 0.0
                
                # Nearest neighbor lookup for velocity at current_t
                if cid in vel_dict:
                    t_arr, v_arr = vel_dict[cid]
                    idx = np.abs(t_arr - current_t).argmin()
                    v_vec = v_arr[idx]
                    
                    if len(v_vec) == 2 and not np.isnan(v_vec[0]) and not np.isnan(v_vec[1]):
                        u = v_vec[0] * velocity_scale
                        v = v_vec[1] * velocity_scale

                arrow_pool[i].set_positions((xs_i_mean, ys_i_mean), (xs_i_mean + u, ys_i_mean + v))
                arrow_color = 'black'#palette[id_to_idx[cid]] * 0.7 if (color_by_id and id_vals is not None) else 'black'
                arrow_pool[i].set_color(arrow_color)
                arrow_pool[i].set_visible(True)

        # Drawing the image logic
        if 'id_vals' in locals() and color_by_id and id_vals is not None:
            img = np.ones((height, width, 3), dtype=np.float32)   # white
            ids_idx = indices_all[lo:hi]
            if ids_idx.size:
                img[ys, xs] = palette[ids_idx]

        elif color_by_polarity and p_vals is not None:
             # Explicit color mapping
            id_color_map = {
                0: [1.0, 0.0, 0.0],    # red
                1: [0.0, 0.3, 1.0],    # blue
                2: [0.0, 1.0, 0.3],    # green
                3: [0.9, 0.9, 0.0],    # yellow
            }
            img = np.ones((height, width, 3), dtype=np.float32)   # white
            ps  = p_vals[lo:hi]
            for uid, col in id_color_map.items():
                on = ps == uid
                img[ys[on],  xs[on]]  = col 
        else:
            img = np.zeros((height, width), dtype=np.uint8)
            np.add.at(img, (ys, xs), 1)
            img = np.clip(img, 0, 1)  # binary: any event -> black

        im.set_data(img)
        title.set_text(
            f'Frame {frame_idx + 1}/{n_frames}   '
            f't = [{current_t:,.0f} - {boundaries[frame_idx + 1]:,.0f}] µs   '
            f'events: {hi - lo:,}   FPS: {live_fps:.1f} = {live_fps * dt/1_000_000:.2f} simulated seconds/second'
        )
        return (im, title, *arrow_pool)

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
    EVENT_FILE = 'data/E_patch_stdbscan.parquet'
    VELOCITY_FILE = 'data/E_patch_stdbscan_velocity.parquet' 
    # Animated frames
    animate_frames(
        source=EVENT_FILE, 
        velocities_source=VELOCITY_FILE,    
        velocity_method='decay_votes',    # choose 'limit', 'momentum', 'votes', or 'decay_votes'
        velocity_scale=100.0,                 
        dt=250_000, 
        max_frames=1892, 
        color_by_id=True, 
        id_col="cluster",
        exclude_ids=[],
        p_col="pol",
        color_by_polarity=False, 
        show_velocity_arrows=True            # Renamed from id_pol_arrows
    )