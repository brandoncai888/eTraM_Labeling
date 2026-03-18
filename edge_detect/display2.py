import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button

import pandas as pd
import open3d as o3d
from extract import extract

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
    exclude_ids: list = None,
    alpha: float = 0.3,  # NEW: Controls opacity per event. Lower = more transparent.
):
    """
    Animate event camera data as a sequence of frames.

    Each frame accumulates all events within a `dt`-wide time window into a
    numpy image array and renders it with imshow. Overlapping events increase
    pixel opacity, resulting in darker colors for areas with dense activity.
    """
    df = extract(source, h5_key) if isinstance(source, str) else source
    df = df.sort_values(t_col).reset_index(drop=True)
    print(f"Loaded {len(df):,} events from source.")
    
    t_vals = df[t_col].to_numpy()
    x_vals = df[x_col].to_numpy(dtype=np.intp)
    y_vals = df[y_col].to_numpy(dtype=np.intp)
    p_vals = df[p_col].to_numpy() if p_col in df.columns else None
    id_vals = df[id_col].to_numpy() if id_col in df.columns else None

    # Filter out excluded IDs
    if exclude_ids is not None and id_vals is not None:
        mask = ~np.isin(id_vals, exclude_ids)
        t_vals = t_vals[mask]
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
        p_vals = p_vals[mask] if p_vals is not None else None
        id_vals = id_vals[mask]

    # Pre-compute deterministic palette mapping for IDs if requested
    if color_by_id and id_vals is not None:
        unique_ids = np.unique(id_vals)
        rng = np.random.default_rng(seed=42)
        palette = rng.random((len(unique_ids), 3))

        # Explicit color mapping for specific IDs
        id_color_map = {
            0: [0.66, 0.0, 0.34],    # red
            1: [0.0, 0.0, 1.0],    # blue
            2: [0.0, 0.66, 0.34],    # green
            3: [0.5, 0.5, 0.0],    # yellow
        }

        # Noise events (id == -1) are always black
        if -1 in unique_ids:
            palette[np.searchsorted(unique_ids, -1)] = [0.0, 0.0, 0.0]

        id_to_idx = {uid: i for i, uid in enumerate(unique_ids)}
        for uid, col in id_color_map.items():
            if uid in id_to_idx:
                palette[id_to_idx[uid]] = col

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

    # --- RECENTERED BUTTONS ---
    btn_width = 0.12
    btn_height = 0.05
    btn_bottom = 0.03
    
    back15_ax = fig.add_axes([0.30, btn_bottom, btn_width, btn_height])
    pause_ax  = fig.add_axes([0.44, btn_bottom, btn_width, btn_height])
    ff15_ax   = fig.add_axes([0.58, btn_bottom, btn_width, btn_height])

    pause_btn  = Button(pause_ax,  'Pause')
    ff15_btn   = Button(ff15_ax,   '▶▶ +5')
    back15_btn = Button(back15_ax, '◀◀ -5')

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
    ff15_btn.on_clicked(make_jump(5))
    back15_btn.on_clicked(make_jump(-5))
    
    if (color_by_polarity and p_vals is not None) or (color_by_id and id_vals is not None):
        init_img = np.ones((height, width, 3), dtype=np.float32)
    else:
        init_img = np.ones((height, width), dtype=np.float32)

    im = ax.imshow(init_img, cmap='gray_r', vmin=0, vmax=1, origin='upper',
                   interpolation='nearest',extent=[0, width, height, 0])
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

        xs = x_vals[lo:hi]
        ys = y_vals[lo:hi]

        if 'id_vals' in locals() and color_by_id and id_vals is not None:
            img = np.ones((height, width, 3), dtype=np.float32)   # white background
            ids_idx = indices_all[lo:hi]
            
            # Vectorized alpha blending per ID
            for idx in np.unique(ids_idx):
                mask = (ids_idx == idx)
                counts = np.zeros((height, width), dtype=np.float32)
                np.add.at(counts, (ys[mask], xs[mask]), 1)
                
                # Blend factor increases with event count
                beta = (1.0 - alpha) ** counts
                img = img * beta[..., None] + palette[idx] * (1.0 - beta[..., None])

        elif color_by_polarity and p_vals is not None:
            img = np.ones((height, width, 3), dtype=np.float32)   # white background
            ps  = p_vals[lo:hi]
            on  = ps == 1
            off = ~on
            
            color_on = np.array([0.22, 0.49, 0.72], dtype=np.float32)
            color_off = np.array([0.84, 0.15, 0.16], dtype=np.float32)

            # Accumulate ON events
            counts_on = np.zeros((height, width), dtype=np.float32)
            np.add.at(counts_on, (ys[on], xs[on]), 1)
            beta_on = (1.0 - alpha) ** counts_on
            img = img * beta_on[..., None] + color_on * (1.0 - beta_on[..., None])

            # Accumulate OFF events (layered on top)
            counts_off = np.zeros((height, width), dtype=np.float32)
            np.add.at(counts_off, (ys[off], xs[off]), 1)
            beta_off = (1.0 - alpha) ** counts_off
            img = img * beta_off[..., None] + color_off * (1.0 - beta_off[..., None])

        else:
            # Grayscale single-channel accumulation
            counts = np.zeros((height, width), dtype=np.float32)
            np.add.at(counts, (ys, xs), 1)
            
            # Maps 0 events -> 0 (white in gray_r), infinite events -> 1 (black)
            img = 1.0 - (1.0 - alpha) ** counts

        im.set_data(img)
        title.set_text(
            f'Frame {frame_idx + 1}/{n_frames}   '
            f't = [{boundaries[frame_idx]:,.0f} – {boundaries[frame_idx + 1]:,.0f}] µs   '
            f'events: {hi - lo:,}   FPS: {live_fps:.1f}'
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
    FILE = 'data/E_patch5_states.parquet'
    # Animated frames
    animate_frames(FILE, t_col='index', dt=25, color_by_id=True, id_col="state", exclude_ids=[], alpha=1)