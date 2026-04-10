import duckdb
import pandas as pd
import pathlib

# Read 2026 from raw data
ORIG_DIR = pathlib.Path("/Users/sumeetdas/Antigravity_NSE_Data/nse_master_bhav_with_delivery_2014_onwards.parquet")
y = 2026
in_file = ORIG_DIR / f"year={y}" / f"part-{y}.parquet"

print(f"Reading raw {y}...")
year_df = pd.read_parquet(in_file)

# We need to get the adjusted_close for 2026. 
# Did build_adjusted_master.py save the base_out to memory? No.
# I will just run the build script, but hook the end to only write 2026.
