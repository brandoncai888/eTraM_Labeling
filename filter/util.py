import cupy as np
import pandas as pd

def XY(W,H,Epix):
    Ex = Epix / W
    Ey = Epix % W
    return Ex,Ey