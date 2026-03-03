import numpy as np

import pandas as pd

from util import *

type="patch"
W = 1280
H = 720
k = 4

E_patch = np.load("E_"+type+".npz")
Et_patch=E_patch["Et_"+type] 
Epix_patch=E_patch["Epix_"+type].astype(np.uint64)
Epol_patch=E_patch["Epol_"+type]
if type=="out":
    Ex, Ey = XY(W,H,Epix_patch)
else:
    Ex, Ey = XY(W//k,H//k,Epix_patch)
df = pd.DataFrame({"t": Et_patch, "x": Ex , "y": Ey, "pol": Epol_patch})
df.to_parquet("E_"+type+".parquet", index=False)