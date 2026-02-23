import cupy as np
import pandas as pd
import time
from filter.filter2 import *

if __name__ == "__main__":
    filename = "data/val_day_014_td_cut.csv"

    t1 = time.time()
    df = pd.read_csv(filename)
    W = int(df["width"].iloc[0])
    H = int(df["height"].iloc[0])
    k = 4
    Et = np.array(df["t"])
    Epix = np.array(df["x"]+df["y"]*W)
    Epol = np.array(df["p"])
    print(W,H,k,Et,Epix,Epol)
    print(f"Read File Time: {time.time() - t1}")

    t1 = time.time()
    Et_sub, Epix_sub, Epol_sub = subdivide(W,H,k,Et,Epix,Epol)
    pd.DataFrame({"Et_sub": Et_sub, "Epix_sub": Epix_sub, "Epol_sub": Epol_sub}).to_csv("data/val_day_014_td_cut_subdivide.csv", index=False)
    print(f"subdivide time: {time.time()-t1}")