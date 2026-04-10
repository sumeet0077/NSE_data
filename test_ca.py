import duckdb
con = duckdb.connect()
query = "SELECT * FROM read_parquet('/Users/sumeetdas/Antigravity_NSE_Data/nse_corporate_actions_equities_2014_onwards.parquet') WHERE exDate >= '2026-01-01'"
df = con.query(query).df()
print(df)
