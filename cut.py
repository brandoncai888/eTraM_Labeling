import pandas as pd

filename = "data/val_day_014_td.parquet"

df = pd.read_parquet(filename)

tmax = df["t"].max()

df = df.loc[df['t']<tmax*2/5]
df = df.loc[df['t']>tmax/10]

df.to_parquet(filename[:-8]+"_cut.parquet")
