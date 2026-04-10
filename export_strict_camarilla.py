#!/usr/bin/env python3
"""
Monthly Camarilla Compression Scanner
======================================
Scans NSE equities for stocks in multi-month Camarilla H4/L4 compression.
Outputs a dated Excel file to the Desktop with full metadata.

Run daily after the market close:
    python3 /Users/sumeetdas/Antigravity_NSE_Data/export_strict_camarilla.py
"""
import sys
import logging
import subprocess
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ── Dependencies ──────────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    yf = None

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("camarilla_scanner")

# ── Config ────────────────────────────────────────────────────────────────────
PARQUET_PATH = "/Users/sumeetdas/Antigravity_NSE_Data/nse_master_adjusted_2014_onwards.parquet/*/*.parquet"
OUTPUT_DIR   = Path("/Users/sumeetdas/Desktop")
MIN_PRICE    = 50.0          # ₹ minimum close
MIN_TURNOVER = 10_000_000    # ₹ daily avg turnover (1 Crore)
MIN_TRADING_DAYS = 15        # Min days in a CLOSED month to qualify as pivot source
MAX_DATA_STALENESS_DAYS = 4  # Warn if parquet data is older than this many calendar days
ETF_KEYWORDS = ['ETF','LIQUID','BEES','GOLD','SILV','GILT','SML250','MID150',
                 'NIFTY','BANK','INFRA','COMP','SENSEX','CRAMC']

def run_camarilla_scan(con, latest_trade_date, nifty_20d_return, scan_dir, mode='current'):
    log.info(f"--- Running Camarilla compression scan for mode: {mode.upper()} ---")
    
    # ── Step 3 ─ Run DuckDB Camarilla Query ───────────────────────────────────
    target_month_filter = "WHERE month_start < (SELECT MAX(month_start) FROM PivotHistory)" if mode == 'current' else ""
    
    QUERY = f"""
    WITH DailyFiltered AS (
        SELECT symbol, trade_date, adj_open as open, adj_high as high, adj_low as low,
               adj_close as close, volume, series 
        FROM read_parquet('{PARQUET_PATH}', union_by_name=True)
        WHERE trade_date >= '2025-01-01' 
          AND series = 'EQ'
    ),
    DailyWithMA AS (
        SELECT symbol, trade_date, close, high, low, volume,
               AVG(close) OVER(PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma20,
               AVG(close) OVER(PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 49 PRECEDING AND CURRENT ROW) as ma50,
               AVG(close) OVER(PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 199 PRECEDING AND CURRENT ROW) as ma200,
               LAG(close, 1) OVER(PARTITION BY symbol ORDER BY trade_date) as prev_close,
               AVG(volume) OVER(PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING) as prev_vol20
        FROM DailyFiltered
    ),
    DailyWithTR AS (
        SELECT *,
               GREATEST(high - low, ABS(high - prev_close), ABS(low - prev_close)) as tr,
               CASE WHEN prev_vol20 > 0 AND volume > 3 * prev_vol20 THEN 1 ELSE 0 END as is_hvn
        FROM DailyWithMA
    ),
    DailyWithATR AS (
        SELECT *,
               AVG(tr) OVER(PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) as atr14,
               SUM(is_hvn) OVER(PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as hvn30,
               LAG(close, 20) OVER(PARTITION BY symbol ORDER BY trade_date) as close_20d_ago
        FROM DailyWithTR
    ),
    LatestMA AS (
        SELECT symbol, 
               ARG_MAX(close, trade_date) as current_close,
               MAX(trade_date) as current_trade_date,
               ARG_MAX(prev_close, trade_date) as prev_day_close,
               ARG_MAX(close_20d_ago, trade_date) as close_20d_ago,
               ARG_MAX(ma20, trade_date)  as last_ma20, 
               ARG_MAX(ma50, trade_date)  as last_ma50, 
               ARG_MAX(ma200, trade_date) as last_ma200,
               ARG_MAX(atr14, trade_date) as last_atr14,
               ARG_MAX(hvn30, trade_date) as last_hvn30
        FROM DailyWithATR 
        GROUP BY symbol
    ),
    GlobalMaxMonth AS (
        SELECT DATE_TRUNC('month', MAX(trade_date))::DATE as max_month
        FROM DailyFiltered
    ),
    MonthlyPivots AS (
        SELECT symbol, month_start, avg_monthly_turnover,
               m_close + ((m_high - m_low) * 1.1 / 2) AS h4,
               m_close - ((m_high - m_low) * 1.1 / 2) AS l4
        FROM (
             SELECT symbol,
                    DATE_TRUNC('month', trade_date)::DATE AS month_start,
                    MAX(high) AS m_high,
                    MIN(low)  AS m_low,
                    ARG_MAX(close, trade_date) AS m_close,
                    AVG(volume * close) as avg_monthly_turnover,
                    COUNT(trade_date) as trading_days
             FROM DailyFiltered
             GROUP BY symbol, month_start
             HAVING COUNT(trade_date) >= {MIN_TRADING_DAYS}
                 OR DATE_TRUNC('month', MAX(trade_date))::DATE = (SELECT max_month FROM GlobalMaxMonth)
        )
    ),
    PivotHistory AS (
        SELECT symbol, month_start, avg_monthly_turnover,
               h4 as target_h4,  l4 as target_l4,
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
        SELECT * FROM PivotHistory 
        WHERE month_start = (
            SELECT MAX(month_start) FROM PivotHistory
            {target_month_filter}
        )
    )
    SELECT 
        lm.symbol, lm.month_start, ma.current_trade_date,
        ma.current_close, lm.avg_monthly_turnover,
        lm.target_h4, lm.target_l4,
        CASE WHEN lm.target_h4 < lm.m1_h4 AND lm.target_l4 > lm.m1_l4
             AND DATE_DIFF('month', lm.m1_date, lm.month_start) = 1 THEN 1 ELSE 0 END as is_target_inside,
        CASE WHEN lm.m1_h4 < lm.m2_h4 AND lm.m1_l4 > lm.m2_l4
             AND DATE_DIFF('month', lm.m2_date, lm.m1_date) = 1 THEN 1 ELSE 0 END as is_m1_inside,
        CASE WHEN lm.m2_h4 < lm.m3_h4 AND lm.m2_l4 > lm.m3_l4
             AND DATE_DIFF('month', lm.m3_date, lm.m2_date) = 1 THEN 1 ELSE 0 END as is_m2_inside,
        ma.prev_day_close, ma.close_20d_ago,
        ma.last_ma20, ma.last_ma50, ma.last_ma200, ma.last_atr14, ma.last_hvn30
    FROM LatestMonth lm
    LEFT JOIN LatestMA ma ON lm.symbol = ma.symbol
    WHERE ma.current_close >= {MIN_PRICE} AND lm.avg_monthly_turnover >= {MIN_TURNOVER}
    """

    try:
        df = con.query(QUERY).df()
    except Exception as e:
        log.error(f"FATAL: DuckDB query failed: {e}")
        return None, None, None

    log.info(f"Raw query returned {len(df)} symbol rows for '{mode}'.")

    # ── Step 4 ─ Score and filter ─────────────────────────────────────────────
    df['consecutive_score'] = (
        df['is_target_inside']
        + (df['is_target_inside'] * df['is_m1_inside'])
        + (df['is_target_inside'] * df['is_m1_inside'] * df['is_m2_inside'])
    )
    results = df[df['consecutive_score'] > 0].copy()

    if results.empty:
        log.warning(f"No stocks met the Inside Camarilla criteria for '{mode}'.")
        return None, None, None

    log.info(f"Found {len(results)} Inside Camarilla setups.")

    # ── Step 5 ─ Compute derived columns ─────────────────────────────────────
    target_month_dt   = pd.to_datetime(results['month_start'].iloc[0]) + pd.DateOffset(months=1)
    target_month_name = target_month_dt.strftime('%B_%Y')

    active_symbols = tuple(results['symbol'].tolist())
    if active_symbols:
        symbols_str = f"('{active_symbols[0]}')" if len(active_symbols) == 1 else str(active_symbols)
        query_ema = f"""
            SELECT symbol, trade_date, adj_close
            FROM read_parquet('{PARQUET_PATH}', union_by_name=True)
            WHERE trade_date >= '2025-01-01'
              AND symbol IN {symbols_str}
            ORDER BY symbol, trade_date
        """
        try:
            ema_df = con.query(query_ema).df()
            ema_df['ema20'] = ema_df.groupby('symbol')['adj_close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
            last_ema = ema_df.groupby('symbol').last().reset_index()
            results = results.merge(last_ema[['symbol', 'ema20']], on='symbol', how='left')
            results['EMA 20 Dist %'] = ((results['current_close'] - results['ema20']) / results['ema20'] * 100).round(2)
        except Exception as e:
            log.warning(f"Failed to calculate EMA 20: {e}")
            results['EMA 20 Dist %'] = np.nan
    else:
        results['EMA 20 Dist %'] = np.nan

    results['current_trade_date'] = pd.to_datetime(results['current_trade_date']).dt.strftime('%Y-%m-%d')
    results['Compression %'] = ((results['target_h4'] - results['target_l4']) / results['current_close'] * 100).round(2)
    results['1D Return %']   = ((results['current_close'] - results['prev_day_close']) / results['prev_day_close'] * 100).round(2)
    results['ATRP %']        = (results['last_atr14'] / results['current_close'] * 100).round(2)
    results['Traded Value (Cr)'] = (results['avg_monthly_turnover'] / 10_000_000).round(2)

    def map_label(s):
        return {1: 'Single', 2: 'Double', 3: 'Triple'}.get(s, str(s))

    results['Compression Level'] = results['consecutive_score'].apply(map_label)

    results['Above 200MA'] = np.where(results['current_close'] > results['last_ma200'], 'Yes', 'No')
    results['Above 50MA']  = np.where(results['current_close'] > results['last_ma50'],  'Yes', 'No')
    results['Above 20MA']  = np.where(results['current_close'] > results['last_ma20'],  'Yes', 'No')

    # RS vs Nifty50
    if nifty_20d_return and nifty_20d_return > 0:
        results['RS (vs Nifty50)'] = (results['current_close'] / results['close_20d_ago'] / nifty_20d_return).round(2)
    else:
        results['RS (vs Nifty50)'] = np.nan

    # ETF filter
    def is_fund(sym):
        sym = str(sym).upper()
        if any(c.isdigit() for c in sym) and not sym.endswith('LTD'):
            return True
        return any(k in sym for k in ETF_KEYWORDS)

    results['is_fund'] = results['symbol'].apply(is_fund)

    # TradingView hyperlink
    def tv_link(sym):
        s = sym.replace('&', '_').replace('-', '_')
        return f'=HYPERLINK("https://in.tradingview.com/chart/?symbol=NSE:{s}", "TradingView")'

    results.rename(columns={
        'symbol': 'Symbol', 'current_close': 'Current Close',
        'last_hvn30': 'HVN', 'current_trade_date': 'Latest Extract Date'
    }, inplace=True)
    results['TradingView'] = results['Symbol'].apply(tv_link)
    results['TV Helper']   = results['Symbol'].apply(lambda x: f"NSE:{str(x).replace('&', '_').replace('-', '_')},")

    COLS = ['Symbol', 'Latest Extract Date', 'Current Close', '1D Return %',
            'Traded Value (Cr)', 'Compression Level', 'Compression %',
            'TradingView', 'TV Helper', 'Above 200MA', 'Above 50MA', 'Above 20MA', 'EMA 20 Dist %', 'HVN', 'RS (vs Nifty50)', 'ATRP %']
    DECIMAL_COLS = {'Compression %', 'ATRP %', 'Traded Value (Cr)', 'Current Close', '1D Return %', 'RS (vs Nifty50)', 'EMA 20 Dist %'}

    equities = results[~results['is_fund']][COLS].sort_values(['Compression Level', 'Current Close'], ascending=[False, False])
    funds    = results[ results['is_fund']][COLS].sort_values(['Compression Level', 'Current Close'], ascending=[False, False])

    # ── Step 7 ─ Summary ─────────────────────────────────────────────────────
    triple = len(equities[equities['Compression Level'] == 'Triple'])
    double = len(equities[equities['Compression Level'] == 'Double'])
    single = len(equities[equities['Compression Level'] == 'Single'])

    log.info("=" * 55)
    log.info(f"  Scan Mode      : {mode.upper()}")
    log.info(f"  Target Month   : {target_month_name.replace('_', ' ')}")
    log.info(f"  Equities Found : {len(equities)}  (Triple={triple}, Double={double}, Single={single})")
    log.info(f"  ETFs/Funds     : {len(funds)}")
    log.info("=" * 55)
    
    return target_month_name, equities, DECIMAL_COLS


if __name__ == "__main__":
    log.info("Validating data source...")
    parquet_dir = Path("/Users/sumeetdas/Antigravity_NSE_Data/nse_master_adjusted_2014_onwards.parquet")
    if not parquet_dir.exists():
        log.error(f"FATAL: Parquet directory not found: {parquet_dir}")
        sys.exit(1)

    con = duckdb.connect()

    latest_date_row = con.execute(
        f"SELECT MAX(trade_date) as latest FROM read_parquet('{PARQUET_PATH}', union_by_name=True) WHERE series='EQ'"
    ).fetchone()
    
    if not latest_date_row or not latest_date_row[0]:
        log.error("FATAL: Could not read latest trade_date from parquet. Data may be corrupt.")
        sys.exit(1)

    latest_trade_date = pd.to_datetime(latest_date_row[0]).date()
    staleness = (datetime.now().date() - latest_trade_date).days
    log.info(f"Latest trade date in parquet: {latest_trade_date} ({staleness} days ago)")

    if staleness > MAX_DATA_STALENESS_DAYS:
        log.warning(
            f"DATA IS STALE — latest trade date is {latest_trade_date} ({staleness} days ago). "
            f"Run the daily updater first! Continuing anyway..."
        )

    log.info("Fetching Nifty50 data for Relative Strength calculation...")
    nifty_20d_return = None
    if yf is None:
        log.warning("yfinance not installed — RS column will be empty.")
    else:
        for attempt in range(3):
            try:
                nifty_df = yf.download('^NSEI', period='60d', interval='1d', progress=False, auto_adjust=True)
                nifty_close = nifty_df['Close'].dropna().sort_index()
                if len(nifty_close) >= 21:
                    n_curr = float(nifty_close.iloc[-1].iloc[0])
                    n_20d  = float(nifty_close.iloc[-21].iloc[0])
                    nifty_20d_return = n_curr / n_20d
                    log.info(f"Nifty50: current={n_curr:.1f}, 20d ago={n_20d:.1f}, 20d return={nifty_20d_return:.4f}")
                else:
                    log.warning(f"Nifty data too short ({len(nifty_close)} rows). RS will be empty.")
                break
            except Exception as e:
                log.warning(f"Nifty fetch attempt {attempt+1}/3 failed: {e}")
        else:
            log.warning("All Nifty fetch attempts failed. RS column will be empty.")

    scan_dir = OUTPUT_DIR / "Camarilla Scans"
    scan_dir.mkdir(parents=True, exist_ok=True)
    
    # Run both scans
    c_month, c_eq, c_cols = run_camarilla_scan(con, latest_trade_date, nifty_20d_return, scan_dir, mode='current')
    print("")
    n_month, n_eq, n_cols = run_camarilla_scan(con, latest_trade_date, nifty_20d_return, scan_dir, mode='next')
    
    # ── Write Consolidated Excel ──────────────────────────────────────────────
    today_str = datetime.now().strftime('%Y%m%d')
    out_path = scan_dir / f"Camarilla Scans.xlsx"
    tmp_path = out_path.with_suffix('.tmp.xlsx')
    
    log.info(f"Writing final consolidated Excel → {out_path}")
    try:
        with pd.ExcelWriter(str(tmp_path), engine='xlsxwriter') as writer:
            wb = writer.book
            fmt2d = wb.add_format({'num_format': '0.00'})
            fmt3d = wb.add_format({'num_format': '0.000'})
            bold  = wb.add_format({'bold': True, 'bg_color': '#E6F0FF', 'border': 1})

            def write_sheet(df, cols, sheet_name):
                if df is None or df.empty:
                    return
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                ws = writer.sheets[sheet_name]
                for idx, col in enumerate(df.columns):
                    series = df[col]
                    max_len = 15 if col == 'TradingView' else max(series.astype(str).map(len).max() if not series.empty else 10, len(str(col))) + 2
                    if col == 'RS (vs Nifty50)':
                        ws.set_column(idx, idx, max_len, fmt3d)
                    elif col in cols:
                        ws.set_column(idx, idx, max_len, fmt2d)
                    else:
                        ws.set_column(idx, idx, max_len)

            if c_eq is not None:
                write_sheet(c_eq, c_cols, 'Current Month Camarilla')
            if n_eq is not None:
                write_sheet(n_eq, n_cols, 'Next Month Prediction')
                
            def create_tv_watchlist_df(symbols_series, chunk_size=30):
                symbols = symbols_series.tolist()
                tv_symbols = [f"NSE:{str(s).replace('&', '_').replace('-', '_')}" for s in symbols]
                chunks = [",".join(tv_symbols[i:i+chunk_size]) for i in range(0, len(tv_symbols), chunk_size)]
                return pd.DataFrame({'TV Watchlist (Copy 30 at a time)': chunks})

            if c_eq is not None and not c_eq.empty:
                tv_df = create_tv_watchlist_df(c_eq['Symbol'])
                write_sheet(tv_df, set(), 'TV Watchlist (Current)')
                
            if n_eq is not None and not n_eq.empty:
                tv_df_next = create_tv_watchlist_df(n_eq['Symbol'])
                write_sheet(tv_df_next, set(), 'TV Watchlist (Next)')

        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)
        
        # Copy to Desktop if directory exists and path is different
        desktop_dir = Path.home() / "Desktop" / "Camarilla Scans"
        desktop_file = desktop_dir / "Camarilla Scans.xlsx"
        if desktop_dir.exists() and out_path.resolve() != desktop_file.resolve():
            import shutil
            shutil.copy2(str(out_path), str(desktop_file))
        
        try:
            subprocess.Popen(['open', str(out_path)])
            log.info("Excel file opened automatically.")
        except Exception:
            pass
            
    except Exception as e:
        log.error(f"FATAL: Failed to write Excel: {e}")
        if tmp_path.exists():
            tmp_path.unlink()
        sys.exit(1)
