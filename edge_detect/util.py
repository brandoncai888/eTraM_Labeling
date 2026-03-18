import numpy as np # import numpy as np 
import pandas as pd

def XY(W,H,Epix):
    Ey = Epix / W
    Ex = Epix % W
    return Ex.astype(int),Ey.astype(int)