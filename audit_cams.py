import duckdb

query = """
SELECT 
    symbol, 
    trade_date, 
    close, 
    adjusted_close,
    adjusted_close / close as split_ratio
FROM read_parquet('/Users/sumeetdas/Antigravity_NSE_Data/nse_master_adjusted_2014_onwards.parquet/*/*.parquet', union_by_name=True)
WHERE symbol IN ('NUVAMA', 'CAMS') 
  AND trade_date >= '2025-10-01'
ORDER BY symbol, trade_date DESC
LIMIT 40
"""

df = duckdb.query(query).df()
print(df.to_string())
