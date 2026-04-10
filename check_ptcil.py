import duckdb
import pandas as pd

con = duckdb.connect()

query = """
    WITH DailyFiltered AS (
        SELECT symbol, trade_date, open, high, low, close, deliv_qty as volume, series 
        FROM read_parquet('/Users/sumeetdas/Antigravity_NSE_Data/nse_master_bhav_with_delivery_2014_onwards.parquet/*/*.parquet', union_by_name=True)
        WHERE trade_date >= '2024-01-01' AND series = 'EQ' AND symbol LIKE 'PTC%'
    ),
    MonthlyOHLC AS (
        SELECT 
            symbol,
            DATE_TRUNC('month', trade_date)::DATE AS month_start,
            MAX(high) AS m_high,
            MIN(low) AS m_low,
            ARG_MAX(close, trade_date) AS m_close,
            AVG(volume) as avg_monthly_vol,
            ARG_MAX(trade_date, trade_date) AS last_trade_date,
            COUNT(trade_date) as trading_days
        FROM DailyFiltered
        GROUP BY symbol, month_start
    ),
    CamarillaPivots AS (
        SELECT 
            symbol,
            month_start,
            last_trade_date,
            m_close as current_close,
            avg_monthly_vol,
            trading_days,
            LAG(m_high) OVER (PARTITION BY symbol ORDER BY month_start) AS prev_high,
            LAG(m_low) OVER (PARTITION BY symbol ORDER BY month_start) AS prev_low,
            LAG(m_close) OVER (PARTITION BY symbol ORDER BY month_start) AS prev_close
        FROM MonthlyOHLC
    ),
    CamarillaCalculated AS (
        SELECT 
            *,
            (prev_high - prev_low) AS c_range,
            prev_close + ((prev_high - prev_low) * 1.1 / 2) AS h4,
            prev_close + ((prev_high - prev_low) * 1.1 / 4) AS h3,
            prev_close - ((prev_high - prev_low) * 1.1 / 4) AS l3,
            prev_close - ((prev_high - prev_low) * 1.1 / 2) AS l4
        FROM CamarillaPivots
        WHERE prev_high IS NOT NULL
    ),
    InsideCamarilla AS (
        SELECT 
            *,
            LAG(h3) OVER (PARTITION BY symbol ORDER BY month_start) as lag1_h3,
            LAG(l3) OVER (PARTITION BY symbol ORDER BY month_start) as lag1_l3,
            LAG(h4) OVER (PARTITION BY symbol ORDER BY month_start) as lag1_h4,
            LAG(l4) OVER (PARTITION BY symbol ORDER BY month_start) as lag1_l4
        FROM CamarillaCalculated
    ),
    CompressionFlags AS (
        SELECT 
            *,
            CASE 
                WHEN h3 < lag1_h3 AND l3 > lag1_l3 THEN 1 
                ELSE 0 
            END AS is_inside_h3_l3,
            (h3 - l3) / current_close * 100 AS h3_l3_width_pct,
            (current_close - l3) / NULLIF((h3 - l3), 0) * 100 AS position_in_zone_pct
        FROM InsideCamarilla
    )
    SELECT 
        symbol, month_start,
        current_close as "Close",
        avg_monthly_vol,
        trading_days,
        is_inside_h3_l3 AS "Inside Now",
        h3, l3, lag1_h3, lag1_l3, h4, l4
    FROM CompressionFlags
    WHERE month_start >= '2025-11-01'
    ORDER BY symbol, month_start
"""

df = con.query(query).df()
print(df.to_string())
