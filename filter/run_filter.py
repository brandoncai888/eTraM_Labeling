import numpy as np # import numpy as np
import pickle
import pandas as pd
import time
from util import *
from filter2 import *

if __name__ == "__main__":

    filename = "data/val_day_014_td.parquet"
    print(f"file read...")
    t1 = time.time()
    df = pd.read_parquet(filename)
    df["y"] = df["y"].astype(np.uint32)
    df["x"] = df["x"].astype(np.uint32)
    W = int(df["width"].iloc[0])
    H = int(df["height"].iloc[0])
    k = 4
    w = W//k
    h = H//k
    Et = np.array(df["t"])
    Epix = np.array(df["x"]+df["y"]*W)
    Epol = np.array(df["p"])
    print(f"{round(time.time()-t1,0)}s")
    t1 = time.time()

    print("patch_direction...")
    Et_patch, Epix_patch, Epol_patch = patch_direction(W,H,k,Et,Epix)
    print(f"{round(time.time()-t1,0)}s")
    t1 = time.time()
    print("multipatch_direction...")
    Et_multi, Epix_multi, Epol_multi = multipatch_direction(w,h,k,Et_patch,Epix_patch,Epol_patch)
    print(f"{round(time.time()-t1,0)}s")
    t1 = time.time()
    print("apply_roi...")
    Et_out, Epix_out, Epol_out = apply_roi(w,h,k,Et_multi,Epix_multi,Epol_multi,W,H,Et,Epix,Epol)
    print(f"{round(time.time()-t1,0)}s")
    t1 = time.time()
    print("saving...")


    Ex, Ey = XY(W,H,Epix_out)
    df = pd.DataFrame({"t": Et_out, "x": Ex, "y": Ey, "pol": Epol_out})
    df.to_parquet("E_out.parquet", index=False)
    df = pd.read_parquet("E_out.parquet")
    print(df.head())
    np.savez("E_patch.npz", Et_patch=Et_patch, Epix_patch=Epix_patch, Epol_patch=Epol_patch)
    np.savez("E_multi.npz", Et_multi=Et_multi, Epix_multi=Epix_multi, Epol_multi=Epol_multi)
    np.savez("E_out.npz", Et_out=Et_out, Epix_out=Epix_out, Epol_out=Epol_out)

    print(f"{round(time.time()-t1,0)}s")
    t1 = time.time()