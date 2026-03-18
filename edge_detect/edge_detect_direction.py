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

def counts_kxk_direction(df, x_min, x_max, y_min, y_max, width, height):
    counts = np.zeros((width,height), dtype=int)
    counts_direction = np.zeros((2,width,height),dtype=int)
    
    # Calculate the width and height of each bin
    dx = (x_max - x_min) / width
    dy = (y_max - y_min) / height
    
    for direction in range(4):
        df_2 = df.loc[df["pol"]==direction].copy()
        increment = 1
        if direction >= 2:
            increment = -increment
        for _, row in df_2.iterrows():
            x, y = row['x'], row['y']
            
            # Calculate grid index directly
            i = int((x - x_min) / dx)
            j = int((y - y_min) / dy)
            
            # Clamp to bounds to handle edge cases
            i = max(0, min(i, width - 1))
            j = max(0, min(j, height - 1))
            
            counts[i, j] += 1
            counts_direction[direction%2,i,j] += increment
        
    return counts, counts_direction

def get_direction(y,x):
    if x == 0:
        if y > 0:
            return 2
        else:
            return 6
    
    slope = y/x
    if x > 0:
        if slope < -2.5:
            return 2
        if slope < -0.4:
            return 3
        if slope < 0.4:
            return 4
        if slope < 2.5:
            return 5
        else:
            return 6
    if slope < -2.5:
        return 6
    if slope < -0.4:
        return 7
    if slope < 0.4:
        return 8
    if slope < 2.5:
        return 9
    else:
        return 2
    
def check_direction(density_grid,direction_grid,x,y):
    d = get_direction(direction_grid[0,x-1,y-1],direction_grid[1,x-1,y-1])
    if d == 2 and density_grid[x,y-1] >= 0:
        return d
    if d == 3 and density_grid[x-1,y-1] >= 0:
        return d
    if d == 4 and density_grid[x-1,y] >= 0:
        return d
    if d == 5 and density_grid[x-1,y+1] >= 0:
        return d
    if d == 6 and density_grid[x,y+1] >= 0:
        return d
    if d == 7 and density_grid[x+1,y+1] >= 0:
        return d
    if d == 8 and density_grid[x+1,y] >= 0:
        return d
    if d == 9 and density_grid[x+1,y-1] >= 0:
        return d
    return 1

def edge_detect(df_name, width, height, voxel, x_min, x_max, k, window_length):
    window_increment = window_length//2
    pix = width * height
    avg = window_length/pix
    std_dev = sqrt(window_length*(pix-1)/pix/pix)
    dense_threshold = avg + 5 * std_dev
    undense_threshold = avg - 2 * std_dev

    x_max = x_min + width*voxel
    y_max = y_min + height*voxel
    states = np.zeros((width+2,height+2),dtype = int) # -1: off, 0: on, 1: edge, 2,3,4,5,6,7,8,9 - directional edge
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
        density_grid, direction_grid = counts_kxk_direction(df_window, x_min, x_max, y_min, y_max, width, height)
        
        #print(density_grid)
        states_new = states.copy()
        for x in range(1,width+1):
            for y in range(1,height+1):
                if states[x,y] == -1 and density_grid[x-1,y-1] >= dense_threshold:
                    states_new[x,y] = check_direction(states,direction_grid,x,y)
                elif density_grid[x-1,y-1] <= undense_threshold:
                    states_new[x,y] = -1
        for x in range(1,width+1):
            for y in range(1,height+1):
                if states[x,y] >= 1:
                    states_new[x,y] = 0
        states = states_new
        for x in range(1,width+1):
            for y in range(1,height+1):
                df_states.append({'x': x-1, 'y':y-1, 'state': states[x, y], 'index': i})
        #print(states)
    df_states = pd.DataFrame(df_states)
    df_states.to_parquet(f'data/E_patch{k}-{voxel}_dstates.parquet', index=False)
    print("Saved!!!")
    
    

if __name__ == "__main__":
    k = 5

    df_name = f"data/E_patch{k}.parquet"
    width = 1280//k
    height = 720//k

    voxel = 4
    x_min = 0
    y_min = 0
    window_length = 50_000
    edge_detect(df_name,width//voxel,height//voxel,voxel,x_min,y_min,k,window_length)
    