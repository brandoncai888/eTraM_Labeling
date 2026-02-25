import numpy as np # import numpy as np
import pickle
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#from matplotlib.colors import ListedColormap
from math import floor, ceil, sqrt
from collections import defaultdict

from util import *

def subdivide(W, H, k, Et, Epix, Epol):
    w, h = ceil(W/k), ceil(H/k)
    Ex, Ey = XY(W, H, Epix)
    
    Et_sub = [[None for j in range(w)] for i in range(h)]
    Epix_sub = [[None for j in range(w)] for i in range(h)]
    Epol_sub = [[None for j in range(w)] for i in range(h)]
    for i in range(h):
        idx1 = (Ey >= i*k) & (Ey < (i+1)*k)
        Et_i = Et[idx1]
        Epol_i = Epol[idx1]
        Ey_i = Ey[idx1]
        Ex_i = Ex[idx1]
        print(i)
        for j in range(w):
            idx = ( (Ex_i >= j*k) & (Ex_i < (j+1)*k) ) #= np.logical_and.reduce((Ey >= i*k, Ey < (i+1)*k, Ex >= j*k, Ex < (j+1)*k)) #
            Et_sub[i][j] = Et_i[idx]
            Epix_sub[i][j] = ((Ey_i[idx]-i*k)*k + (Ex_i[idx]-j*k))
            Epol_sub[i][j] = Epol_i[idx]
    return Et_sub, Epix_sub, Epol_sub

def combine(W, H, k, Et_sub, Epix_sub, Epol_sub):
    Et = np.concatenate([np.concatenate(Et_sub[i]) for i in range(len(Et_sub))])
    Epix = list()
    for i in range(len(Et_sub)):
        for j in range(len(Et_sub[i])):
            Ex, Ey = XY(k, k, Epix_sub[i][j])
            Epix.append((i*k+Ey)*W + (j*k+Ex))
    Epix = np.concatenate(Epix); Epix = Epix.astype(int)
    Epol = np.concatenate([np.concatenate(Epol_sub[i]) for i in range(len(Epol_sub))]); Epol = Epol.astype(int)

    asort = np.argsort(Et)
    return Et[asort], Epix[asort], Epol[asort]


def reduce(Et_sub, Epix_sub, Epol_sub):
    w, h = len(Et_sub[0]), len(Et_sub)
    Et = list()
    Epix = list()
    Epol = list()

    for i in range(h):
        for j in range(w):
            Et.extend(Et_sub[i][j])
            Epix.extend([i*w+j for _ in Epix_sub[i][j]])
            Epol.extend(Epol_sub[i][j])

    Et = np.array(Et); asort = np.argsort(Et)
    Epix = np.array(Epix)
    Epol = np.array(Epol)
    return Et[asort], Epix[asort], Epol[asort]


def patch_direction(W, H, k, Et, Epix):
    Et_sub, Epix_sub, Epol_sub = subdivide(W, H, k, Et, Epix, np.ones_like(Epix))
    w, h = len(Et_sub[0]), len(Et_sub)

    Et_out = [[list() for j in range(w)] for i in range(h)]
    Epix_out = [[list() for j in range(w)] for i in range(h)]
    Epol_out = [[list() for j in range(w)] for i in range(h)]

    
    # states = np.array([np.zeros((1,)) for _ in range(4)])
    states = np.array([np.zeros((k,)) for _ in range(4)])
    for i in range(h):
        print(i)
        for j in range(w):
            Ex, Ey = XY(k, k, Epix_sub[i][j])
            for tdx in range(len(Et_sub[i][j])):
                t = Et_sub[i][j][tdx]
                x, y = Ex[tdx], Ey[tdx]

                # x1 = 0; y1 = 0
                x1 = x; y1 = y

                if y == states[0,x1]:
                    states[0,x1] += 1
                elif y > states[0,x1]:
                    states[0,x1] = 0

                if x == states[1,y1]:
                    states[1,y1] += 1
                elif x > states[1,y1]:
                    states[1,y1] = 0

                if k-y-1 == states[2,x1]:
                    states[2,x1] += 1
                elif k-y-1 > states[2,x1]:
                    states[2,x1] = 0

                if k-x-1 == states[3,y1]:
                    states[3,y1] += 1
                elif k-x-1 > states[3,y1]:
                    states[3,y1] = 0

                for sdir in range(len(states)):
                    for spos in range(states.shape[1]):
                        if states[sdir,floor(spos)] == k:
                            states[sdir,floor(spos)] = 0
                            Et_out[i][j].append(t)
                            Epix_out[i][j].append(y*k+x)
                            Epol_out[i][j].append(sdir)

    # return combine(W, H, k, Et_out, Epix_out, Epol_out)
    return reduce(Et_out, Epix_out, Epol_out)


def multipatch_direction(W, H, k, Et, Epix, Epol):
    Ex, Ey = XY(W, H, Epix)
    Et_out = list()
    Epix_out = list()
    Epol_out = list()

    states = np.zeros((H+2, W+2, 4), dtype=int)
    for tdx in range(len(Et)):
        i, j = Ey[tdx]+1, Ex[tdx]+1
        pol = Epol[tdx]; pol_rev = (pol+2) % 4

        if pol == 0:
            states[i,j,pol] = states[i-1,j,pol] + 1
            state_rev = states[i+1,j,pol_rev]
            states[i:i+1+state_rev,j,pol_rev] = 0
        elif pol == 1:
            states[i,j,pol] = states[i,j-1,pol] + 1
            state_rev = states[i,j+1,pol_rev]
            states[i,j:j+1+state_rev,pol_rev] = 0
        elif pol == 2:
            states[i,j,pol] = states[i+1,j,pol] + 1
            state_rev = states[i-1,j,pol_rev]
            states[i:i-1-state_rev:-1,j,pol_rev] = 0
        elif pol == 3:
            states[i,j,pol] = states[i,j+1,pol] + 1
            state_rev = states[i,j-1,pol_rev]
            states[i,j:j-1-state_rev:-1,pol_rev] = 0


        if states[i,j,pol] == k:
            Et_out.append(Et[tdx])
            Epix_out.append(Epix[tdx])
            Epol_out.append(pol)
            if pol == 0:
                states[i:i-k:-1,j,pol] = 0
                # states[i:i-k:-1,j,pol_rev] = 0
            elif pol == 1:
                states[i,j:j-k:-1,pol] = 0
                # states[i,j:j-k:-1,pol_rev] = 0
            elif pol == 2:
                states[i:i+k,j,pol] = 0
                # states[i,j:j-k:-1,pol_rev] = 0
            elif pol == 3:
                states[i,j:j+k,pol] = 0
                # states[i,j:j-k:-1,pol_rev] = 0

    return np.array(Et_out), np.array(Epix_out), np.array(Epol_out)


def roi_mask(W, H, k, Et, Epix, Epol):
    Ex, Ey = XY(W, H, Epix)
    mask = np.zeros((len(Et), (H+2)*k, (W+2)*k), dtype=int)
    for tdx in range(len(Et)):
        x, y = Ex[tdx]+1, Ey[tdx]+1
        mask[tdx,:,:] = mask[tdx-1,:,:]
        # mask[tdx,y*k:(y+1)*k,x*k:(x+1)*k] = 0
        if Epol[tdx] == 0:
            mask[tdx,(y+1)*k:(y+2)*k,x*k:(x+1)*k] = 1
            mask[tdx,(y-1)*k:y*k,x*k:(x+1)*k] = 0
        elif Epol[tdx] == 1:
            mask[tdx,y*k:(y+1)*k,(x+1)*k:(x+2)*k] = 1
            mask[tdx,y*k:(y+1)*k,(x-1)*k:x*k] = 0
        elif Epol[tdx] == 2:
            mask[tdx,(y-1)*k:y*k,x*k:(x+1)*k] = 1
            mask[tdx,y*k:(y+1)*k,x*k:(x+1)*k] = 0
        elif Epol[tdx] == 3:
            mask[tdx,y*k:(y+1)*k,(x-1)*k:x*k] = 1
            mask[tdx,y*k:(y+1)*k,x*k:(x+1)*k] = 0
    return mask[:,k:-k,k:-k]


def apply_mask(W, H, Et, Epix, Epol, Et_roi, RM):
    Ex, Ey = XY(W, H, Epix)

    Et_out = list()
    Epix_out = list()
    Epol_out = list()

    tdx = np.where(Et > Et_roi[0])[0][0]
    for tdx_roi in range(len(Et_roi)-1):
        while tdx < len(Et) and Et[tdx] <= Et_roi[tdx_roi+1]:
            if RM[tdx_roi,Ey[tdx],Ex[tdx]] == 1:
                Et_out.append(Et[tdx])
                Epix_out.append(Epix[tdx])
                Epol_out.append(Epol[tdx])
            tdx += 1
    return np.array(Et_out), np.array(Epix_out), np.array(Epol_out)


def apply_roi(w, h, k, Et_multi, Epix_multi, Epol_multi, W, H, Et, Epix, Epol):
    """
    Combined roi_mask and apply_mask without allocating arrays or storing mask states.
    Builds 2D mask incrementally and applies filtering simultaneously.
    
    Args:
        w, h, k: ROI mask dimensions (from multipatch_direction output)
        Et_multi, Epix_multi, Epol_multi: Events from multipatch_direction
        W, H: Original image dimensions
        Et, Epix, Epol: Original events to filter
    
    Returns:
        Et_out, Epix_out, Epol_out: Filtered events within active ROI regions
    """
    Ex_multi, Ey_multi = XY(w, h, Epix_multi)
    Ex, Ey = XY(W, H, Epix)
    
    Et_out = list()
    Epix_out = list()
    Epol_out = list()
    
    # Only current 2D mask needed (with padding for boundary safety)
    current_mask = np.zeros(((h + 2) * k, (w + 2) * k), dtype=int)
    tdx = np.where(Et > Et_multi[0])[0][0]
    
    for tdx_roi in range(len(Et_multi) - 1):
        # Update mask based on current Et_multi event
        x, y = Ex_multi[tdx_roi] + 1, Ey_multi[tdx_roi] + 1
        pol = Epol_multi[tdx_roi]
        
        if pol == 0:
            current_mask[(y+1)*k:(y+2)*k, x*k:(x+1)*k] = 1
            current_mask[(y-1)*k:y*k, x*k:(x+1)*k] = 0
        elif pol == 1:
            current_mask[y*k:(y+1)*k, (x+1)*k:(x+2)*k] = 1
            current_mask[y*k:(y+1)*k, (x-1)*k:x*k] = 0
        elif pol == 2:
            current_mask[(y-1)*k:y*k, x*k:(x+1)*k] = 1
            current_mask[y*k:(y+1)*k, x*k:(x+1)*k] = 0
        elif pol == 3:
            current_mask[y*k:(y+1)*k, (x-1)*k:x*k] = 1
            current_mask[y*k:(y+1)*k, x*k:(x+1)*k] = 0
        
        # Filter original events in time window using current mask
        while tdx < len(Et) and Et[tdx] <= Et_multi[tdx_roi + 1]:
            # Access mask with padding offset
            if current_mask[Ey[tdx] + k, Ex[tdx] + k] == 1:
                Et_out.append(Et[tdx])
                Epix_out.append(Epix[tdx])
                Epol_out.append(Epol[tdx])
            tdx += 1
    
    return np.array(Et_out), np.array(Epix_out), np.array(Epol_out)