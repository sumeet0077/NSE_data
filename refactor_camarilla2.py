import sys
from pathlib import Path
import re

def refactor_cam():
    path = Path("/Users/sumeetdas/Antigravity_NSE_Data/export_strict_camarilla.py")
    content = path.read_text()
    
    # 1. replace Step 6 write excel logic with `return target_month_name, equities, funds`
    # Find start of Step 6:
    step6_start = content.find("    # ── Step 6 ─ Write Excel ──────────────────────────────────────────────────")
    if step6_start == -1:
        print("Could not find step 6")
        return
        
    step7_start = content.find("    # ── Step 7 ─ Summary ─────────────────────────────────────────────────────")
    
    pre_step6 = content[:step6_start]
    post_step7 = content[step7_start:]
    
    # We want to replace the whole writing block with just returning the tuples
    replacement = "    return target_month_name, equities, funds\n\n"
    
    # Also we want to remove Step 7 from the function and move it out? Wait, Step 7 does logging. We can keep it or do it before return.
    # Let's just return early.
    
    # Actually, the easiest way to manipulate this strictly is with regex or exact replace.
    # Let's write a targeted replace.
    
    pass

def custom_refactor():
    file_path = "/Users/sumeetdas/Antigravity_NSE_Data/export_strict_camarilla.py"
    with open(file_path, "r") as f:
        content = f.read()

    # Replace the return in DuckDB fail to return None
    content = content.replace("log.error(f\"FATAL: DuckDB query failed: {e}\")\n        return", "log.error(f\"FATAL: DuckDB query failed: {e}\")\n        return None, None, None")
    content = content.replace("log.warning(f\"No stocks met the Inside Camarilla criteria for '{mode}'.\")\n        return", "log.warning(f\"No stocks met the Inside Camarilla criteria for '{mode}'.\")\n        return None, None, None")

    import re
    # We will slice out everything from "Step 6" to the end of the function.
    step6_regex = re.compile(r"    # ── Step 6 ─ Write Excel.*?if __name__ == \"__main__\":", re.DOTALL)
    
    new_tail = """    # ── Step 7 ─ Summary ─────────────────────────────────────────────────────
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


if __name__ == "__main__":"""
    
    content = step6_regex.sub(new_tail, content)
    
    # Now replace the main block
    main_replacement = """if __name__ == "__main__":
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

        if out_path.exists():
            out_path.unlink()
        tmp_path.rename(out_path)
        
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
"""
    # Replace from "if __name__" to the end
    import re
    main_regex = re.compile(r"if __name__ == \"__main__\":.*", re.DOTALL)
    content = main_regex.sub(main_replacement, content)
    
    with open(file_path, "w") as f:
        f.write(content)

if __name__ == "__main__":
    custom_refactor()
