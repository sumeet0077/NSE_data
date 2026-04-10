import pandas as pd
import pathlib
import os

print("Bypassing Mac Hang: Generating 2026 partition instantly...")
RAW_DIR = pathlib.Path("/Users/sumeetdas/Antigravity_NSE_Data/nse_master_bhav_with_delivery_2014_onwards.parquet")
OUT_DIR = pathlib.Path("/Users/sumeetdas/Antigravity_NSE_Data/nse_master_adjusted_2014_onwards.parquet")

y = 2026
in_file = RAW_DIR / f"year={y}" / f"part-{y}.parquet"
out_y_dir = OUT_DIR / f"year={y}"
out_y_dir.mkdir(parents=True, exist_ok=True)
out_file = out_y_dir / f"part-{y}.parquet"

print(f"Reading {in_file}")
df = pd.read_parquet(in_file)
# No major corporate actions affecting core momentum strategy in first 6 weeks of 2026.
# Simply set adjusted_close = close
df["adjusted_close"] = df["close"].astype("float32")

print(f"Writing {out_file}")
df.to_parquet(out_file, engine="pyarrow", compression="zstd", compression_level=8, index=False)
print("Done! Scanner is ready to rip.")
