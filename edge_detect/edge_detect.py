import pandas as pd
import numpy as np
from math import sqrt
from extract import extract

import numpy as np

def counts_kxk(df, x_min, x_max, y_min, y_max, width, height):
    counts = np.zeros((width,height), dtype=int)
    
    # Calculate the width and height of each bin
    dx = (x_max - x_min) / width
    dy = (y_max - y_min) / height
    
    for _, row in df.iterrows():
        x, y = row['x'], row['y']
        
        # Calculate grid index directly
        i = int((x - x_min) / dx)
        j = int((y - y_min) / dy)
        
        # Clamp to bounds to handle edge cases
        i = max(0, min(i, width - 1))
        j = max(0, min(j, height - 1))
        
        counts[i, j] += 1
        
    return counts

def edge_detect(df_name, width, height, voxel, x_min, x_max, k, window_length):
    window_increment = window_length//2
    pix = width * height
    avg = window_length/pix
    std_dev = sqrt(window_length*(pix-1)/pix/pix)
    dense_threshold = avg + 5 * std_dev
    undense_threshold = avg - 2 * std_dev

    x_max = x_min + width*voxel
    y_max = y_min + height*voxel
    states = np.zeros((width+2,height+2),dtype = int) # -1: off, 0: on, 1: edge
    states = states - 1
    
    # set borders to on as padding
    for i in range(height+2):
        states[0,i] = 0
        states[width+1,i] = 0
    for i in range(width+2):
        states[i,0] = 0
        states[i,height+1] = 0
    df_states = []
    
    direction = np.zeros((width+2,height+2),dtype = int) - 1


    df = extract(df_name)

    df_cropped = df[(df['x'] >= x_min) & (df['x'] < x_max) & (df['y'] >= y_min) & (df['y'] < y_max)].copy().reset_index(drop=True)
    df_cropped['index'] = df_cropped.index
    print(df_cropped)
    df_cropped.to_parquet(f'data/E_patch{k}-{voxel}_cropped.parquet', index=False)

    for i in range(window_length,df_cropped.shape[0],window_increment):
        print(f"Processing window {(i-window_length)//window_increment}, {i-window_length} to {i}")
        df_window = df_cropped.iloc[i-window_length:i]
        density_grid = counts_kxk(df_window, x_min, x_max, y_min, y_max, width, height)
        
        #print(density_grid)
        states_new = states.copy()
        for x in range(1,width+1):
            for y in range(1,height+1):
                if states[x,y] == -1 and density_grid[x-1,y-1] >= dense_threshold:
                    states_new[x,y] = 1
                elif density_grid[x-1,y-1] <= undense_threshold:
                    states_new[x,y] = -1
        for x in range(1,width+1):
            for y in range(1,height+1):
                if states[x,y] == 1:
                    states_new[x,y] = 0
        states = states_new
        for x in range(1,width+1):
            for y in range(1,height+1):
                df_states.append({'x': x-1, 'y':y-1, 'state': states[x, y], 'index': i})
        #print(states)
    df_states = pd.DataFrame(df_states)
    df_states.to_parquet(f'data/E_patch{k}-{voxel}_states.parquet', index=False)
    print("Saved!!!")
    
    

if __name__ == "__main__":
    k = 5

    df_name = f"data/E_patch{k}.parquet"
    width = 1280//k
    height = 720//k

    voxel = 2
    x_min = 0
    y_min = 0
    window_length = 10_000
    edge_detect(df_name,width//voxel,height//voxel,voxel,x_min,y_min,k,window_length)
    