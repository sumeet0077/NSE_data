
import pandas as pd
from pathlib import Path

# Load a single partition directly to see what symbols are actually there
try:
    df = pd.read_parquet("nse_master_adjusted_2014_onwards.parquet/year=2024/part-2024.parquet")
    print(f"Loaded year=2024 partition. Rows: {len(df)}")
    print("Columns:", df.columns.tolist())
    print("Sample Symbols:", df["symbol"].unique()[:10])
    
    # Check if a known symbol exists
    print("Is RELIANCE in 2024?", "RELIANCE" in df["symbol"].values)
    print("Is TCS in 2024?", "TCS" in df["symbol"].values)
    
except Exception as e:
    print(e)
