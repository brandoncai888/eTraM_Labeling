import pandas as pd

filename = "data/val_day_014_td.csv"

df = pd.read_csv(filename)

tmax = df["t"].max()

df = df.loc[df['t']<tmax/5]

df.to_csv(filename[:-4]+"_cut.csv")
