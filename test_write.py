import pandas as pd
import pathlib

ORIG_DIR = pathlib.Path("/Users/sumeetdas/Antigravity_NSE_Data/nse_master_bhav_with_delivery_2014_onwards.parquet")
y = 2026

in_file = ORIG_DIR / f"year={y}" / f"part-{y}.parquet"
year_df = pd.read_parquet(in_file)
print(year_df.shape)
