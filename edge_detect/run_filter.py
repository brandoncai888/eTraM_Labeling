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
    k = 5
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
    
    if type=="out":
        Ex, Ey = XY(W,H,Epix_patch)
    else:
        Ex, Ey = XY(W//k,H//k,Epix_patch)
    df = pd.DataFrame({"t": Et_patch, "x": Ex , "y": Ey, "pol": Epol_patch})
    df.to_parquet(f"data/E_patch{k}.parquet", index=False)