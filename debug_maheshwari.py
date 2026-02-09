
import pandas as pd
import pyarrow.dataset as ds
from pathlib import Path

ADJUSTED_PATH = Path("nse_master_adjusted_2014_onwards.parquet")

def check_nov_dec():
    dataset = ds.dataset(ADJUSTED_PATH, format="parquet", partitioning="hive")
    
    # Check Nov/Dec 2020 specifically
    filter_expr = (ds.field("symbol") == "MAHESHWARI") & (ds.field("year") == 2020)
    
    table = dataset.to_table(filter=filter_expr, columns=["trade_date", "series", "close", "adjusted_close"])
    df = table.to_pandas()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    
    nov_dec = df[df["trade_date"] >= "2020-10-01"].sort_values("trade_date")
    print("Unique Series in Oct-Dec 2020:", nov_dec["series"].unique())
    print("Data from Oct 2020 onwards:")
    print(nov_dec)

if __name__ == "__main__":
    check_nov_dec()
