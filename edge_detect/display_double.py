import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import pandas as pd
from extract import extract

def animate_dual_frames(
    source_hi,
    source_lo,
    window_length: int = 10_000, # Time window to look back from the current low-res event
    scale_factor: int = 5,       # Box size for low-res outlines (e.g., 5 for 5x5)
    offset_x: int = 0,           # NEW: X alignment offset
    offset_y: int = 0,           # NEW: Y alignment offset
    x_col: str = 'x',
    y_col: str = 'y',
    t_col: str = 't',
    p_col: str = 'p',            # Used for high-res polarities (0, 1, 2, 3)
    state_col: str = 'state',    # Used for low-res states
    width: int = None,
    height: int = None,
    interval: int = 100,
    figsize: tuple = (10, 6),
    save_path: str = None,
    fps: int = 10,
    max_frames: int = None,
    alpha: float = 0.3,
):
    """
    Animate high-res and low-res event camera data together.
    Steps through the low-res events, using their timestamps to define a trailing
    time window for the high-res events.
    """
    # 1. Load both datasets
    df_hi = extract(source_hi) if isinstance(source_hi, str) else source_hi
    df_lo = extract(source_lo) if isinstance(source_lo, str) else source_lo

    df_hi = df_hi.sort_values(t_col).reset_index(drop=True)
    df_lo = df_lo.sort_values(t_col).reset_index(drop=True)
    
    print(f"Loaded {len(df_hi):,} high-res and {len(df_lo):,} low-res events.")

    # High-res arrays
    t_hi = df_hi[t_col].to_numpy()
    x_hi = df_hi[x_col].to_numpy(dtype=np.intp)
    y_hi = df_hi[y_col].to_numpy(dtype=np.intp)
    p_hi = df_hi[p_col].to_numpy(dtype=np.intp) # Assuming 0, 1, 2, 3

    # Low-res arrays
    t_lo = df_lo[t_col].to_numpy()
    x_lo = df_lo[x_col].to_numpy(dtype=np.intp)
    y_lo = df_lo[y_col].to_numpy(dtype=np.intp)
    state_lo = df_lo[state_col].to_numpy(dtype=np.intp)

    # 2. Define Colors
    # 4 Polarities for High-Res (e.g., Red, Blue, Green, Orange)
    hi_colors = np.array([
        [1.0, 0.0, 0.0],  # Red        (Hue: 0°)
        [0.5, 1.0, 0.0],  # Chartreuse (Hue: 90°)
        [0.0, 1.0, 1.0],  # Cyan       (Hue: 180°)
        [0.5, 0.0, 1.0]   # Purple     (Hue: 270°)
    ], dtype=np.float32)

    # State colors for Low-Res Outlines (Distinct from high-res)
    unique_states = np.unique(state_lo)
    rng = np.random.default_rng(seed=42)
    lo_palette = rng.random((3, 3))
    # Hardcode a few states if you prefer:
    lo_palette[0] = [0.7, 0.7, 0.7] # State 0: On (Gray)
    lo_palette[1] = [0.2, 0.2, 0.2] # State 1: Edge (Black)
    lo_palette[2] = [1.0, 1.0, 1.0] # State -1: Off (White)

    # 3. Setup Plot Boundaries
    if width is None:
        width = int(x_hi.max()) + 1
    if height is None:
        height = int(y_hi.max()) + 1

    # Find unique timestamps/indices so we process them as discrete frames
    unique_t_lo = np.unique(t_lo)
    n_frames = len(unique_t_lo)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(bottom=0.1)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
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

    # --- Buttons ---
    btn_width, btn_height, btn_bottom = 0.12, 0.05, 0.03
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

    init_img = np.ones((height, width, 3), dtype=np.float32)
    im = ax.imshow(init_img, vmin=0, vmax=1, origin='upper',
                   interpolation='nearest', extent=[0, width, height, 0])
    title = ax.set_title('')

    # --- Helper: Draw Blended Outlines & Shading ---
    def draw_outlines(img_array, xs, ys, colors, size, W, H, off_x, off_y, 
                      border_alpha=0.6, fill_alpha=0.15):
        """
        Draws semi-transparent bounding boxes so high-res events show through.
        - border_alpha: Opacity of the 1-pixel edge (0.0 to 1.0)
        - fill_alpha: Opacity of the interior shading (0.0 to 1.0)
        """
        for x, y, c in zip(xs, ys, colors):
            x_start = (x * size) + off_x
            y_start = (y * size) + off_y
            
            x_end = x_start + size
            y_end = y_start + size
            
            # Skip if completely out of bounds
            if x_start >= W or y_start >= H or x_end <= 0 or y_end <= 0: 
                continue
            
            # Clip bounds for safe slicing
            x_s = max(x_start, 0)
            x_e = min(x_end, W)
            y_s = max(y_start, 0)
            y_e = min(y_end, H)
            
            # 1. SHADE THE INTERIOR
            # Blend the existing pixels with the state color
            region = img_array[y_s:y_e, x_s:x_e]
            img_array[y_s:y_e, x_s:x_e] = region * (1.0 - fill_alpha) + c * fill_alpha
            
            # # 2. DRAW THE BORDERS
            # # Top edge
            # if 0 <= y_start < H: 
            #     img_array[y_start, x_s:x_e] = img_array[y_start, x_s:x_e] * (1.0 - border_alpha) + c * border_alpha
            # # Bottom edge
            # if 0 <= y_end - 1 < H: 
            #     img_array[y_end - 1, x_s:x_e] = img_array[y_end - 1, x_s:x_e] * (1.0 - border_alpha) + c * border_alpha
            # # Left edge
            # if 0 <= x_start < W: 
            #     img_array[y_s:y_e, x_start] = img_array[y_s:y_e, x_start] * (1.0 - border_alpha) + c * border_alpha
            # # Right edge
            # if 0 <= x_end - 1 < W: 
            #     img_array[y_s:y_e, x_end - 1] = img_array[y_s:y_e, x_end - 1] * (1.0 - border_alpha) + c * border_alpha

    # --- Animation Update Loop ---
    def update(frame_idx):
        _t = time.perf_counter()
        if fps_state['last'] is not None:
            fps_state['history'].append(_t - fps_state['last'])
        fps_state['last'] = _t
        fps_state['count'] += 1

        live_fps = (1.0 / (sum(fps_state['history']) / len(fps_state['history']))
                    if fps_state['history'] else 0.0)

        # 1. Determine Time Window based on the Unique Low-Res Index
        t_current = unique_t_lo[frame_idx]
        t_start = t_current - window_length

        # 2. Slice High-Res Data
        hi_lo_idx = np.searchsorted(t_hi, t_start, side='right') 
        hi_hi_idx = np.searchsorted(t_hi, t_current, side='right')
        xs = x_hi[hi_lo_idx:hi_hi_idx]
        ys = y_hi[hi_lo_idx:hi_hi_idx]
        ps = p_hi[hi_lo_idx:hi_hi_idx]

        # 3. Accumulate High-Res Events into Image
        img = np.ones((height, width, 3), dtype=np.float32) # White background
        
        for pol in range(4): # 4 Polarities
            mask = (ps == pol)
            counts = np.zeros((height, width), dtype=np.float32)
            np.add.at(counts, (ys[mask], xs[mask]), 1)
            
            beta = (1.0 - alpha) ** counts
            img = img * beta[..., None] + hi_colors[pol] * (1.0 - beta[..., None])

        # 4. Find ALL Low-Res events that share this exact t_current
        lo_start_idx = np.searchsorted(t_lo, t_current, side='left')
        lo_end_idx = np.searchsorted(t_lo, t_current, side='right')
        
        xs_lo_slice = x_lo[lo_start_idx:lo_end_idx]
        ys_lo_slice = y_lo[lo_start_idx:lo_end_idx]
        states_slice = state_lo[lo_start_idx:lo_end_idx]
        outline_colors = lo_palette[states_slice]
        
        # 5. Burn Outlines onto Image
        draw_outlines(
            img, xs_lo_slice, ys_lo_slice, outline_colors, 
            scale_factor, width, height, offset_x, offset_y,
            border_alpha=0.4,  # Adjust border visibility here
            fill_alpha=0.3     # Adjust interior shading here
        )

        # Update Display
        im.set_data(img)
        title.set_text(
            f'Low-Res Idx {frame_idx}/{n_frames} | '
            f't = [{t_start:,.0f} – {t_current:,.0f}] | '
            f'Hi-Res Events: {hi_hi_idx - hi_lo_idx:,} | FPS: {live_fps:.1f}'
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
    k = 5
    voxel = 4
    FILE_HI = f'data/E_patch{k}-{voxel}_cropped.parquet'
    FILE_LO = f'data/E_patch{k}-{voxel}_states.parquet'
    animate_dual_frames(FILE_HI, FILE_LO, t_col='index',p_col='pol', offset_x=0, offset_y=0, window_length=20_000, scale_factor=voxel, alpha=0.05)