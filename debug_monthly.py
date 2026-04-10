import duckdb
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

con = duckdb.connect()

query = """
WITH DailyFiltered AS (
    SELECT symbol, trade_date, adj_open as open, adj_high as high, adj_low as low, adj_close as close, volume, series 
    FROM read_parquet('/Users/sumeetdas/Antigravity_NSE_Data/nse_master_adjusted_2014_onwards.parquet/*/*.parquet', union_by_name=True)
    WHERE trade_date >= '2025-10-01' 
      AND series = 'EQ'
),
MonthlyOHLC AS (
    SELECT 
        symbol,
        DATE_TRUNC('month', trade_date)::DATE AS month_start,
        MAX(high) AS m_high,
        MIN(low) AS m_low,
        AVG(volume * close) as avg_monthly_turnover
    FROM DailyFiltered
    GROUP BY symbol, month_start
),
MonthlyPivots AS (
    SELECT 
        symbol,
        month_start,
        avg_monthly_turnover,
        m_close + ((m_high - m_low) * 1.1 / 2) AS h4,
        m_close - ((m_high - m_low) * 1.1 / 2) AS l4
    FROM (
         SELECT 
            symbol,
            DATE_TRUNC('month', trade_date)::DATE AS month_start,
            MAX(high) AS m_high,
            MIN(low) AS m_low,
            ARG_MAX(close, trade_date) AS m_close,
            AVG(volume * close) as avg_monthly_turnover
        FROM DailyFiltered
        GROUP BY symbol, month_start
        HAVING COUNT(trade_date) >= 15
    )
),
PivotHistory AS (
    SELECT 
        symbol,
        month_start,
        avg_monthly_turnover,
        h4 as target_h4,
        l4 as target_l4,
        LAG(h4, 1) OVER (PARTITION BY symbol ORDER BY month_start) as m1_h4,
        LAG(l4, 1) OVER (PARTITION BY symbol ORDER BY month_start) as m1_l4,
        LAG(month_start, 1) OVER (PARTITION BY symbol ORDER BY month_start) as m1_date,
        
        LAG(h4, 2) OVER (PARTITION BY symbol ORDER BY month_start) as m2_h4,
        LAG(l4, 2) OVER (PARTITION BY symbol ORDER BY month_start) as m2_l4,
        LAG(month_start, 2) OVER (PARTITION BY symbol ORDER BY month_start) as m2_date,
        
        LAG(h4, 3) OVER (PARTITION BY symbol ORDER BY month_start) as m3_h4,
        LAG(l4, 3) OVER (PARTITION BY symbol ORDER BY month_start) as m3_l4,
        LAG(month_start, 3) OVER (PARTITION BY symbol ORDER BY month_start) as m3_date
    FROM MonthlyPivots
),
LatestMonth AS (
    SELECT * 
    FROM PivotHistory 
    WHERE month_start = (
        SELECT MAX(month_start) FROM PivotHistory WHERE month_start < (SELECT MAX(month_start) FROM PivotHistory)
    )
)
SELECT 
    lm.symbol,
    lm.month_start,
    lm.m1_date,
    lm.target_h4,
    lm.target_l4,
    lm.m1_h4,
    lm.m1_l4,
    CASE WHEN lm.target_h4 < lm.m1_h4 AND lm.target_l4 > lm.m1_l4 
         AND DATE_DIFF('month', lm.m1_date, lm.month_start) = 1 THEN 1 ELSE 0 END as is_target_inside
FROM LatestMonth lm
WHERE CASE WHEN lm.target_h4 < lm.m1_h4 AND lm.target_l4 > lm.m1_l4 AND DATE_DIFF('month', lm.m1_date, lm.month_start) = 1 THEN 1 ELSE 0 END = 1
LIMIT 5
"""

df = con.query(query).df()
print(df)
