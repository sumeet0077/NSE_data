import sys
with open("/Users/sumeetdas/Antigravity_NSE_Data/linearity_scanner.py", "r") as f:
    text = f.read()

# Replace the main() function
import re

new_main = """def main():
    parser = argparse.ArgumentParser(description="NSE Stock Linearity Scanner")
    parser.add_argument("--min_r2", type=float, default=0.85, help="Minimum R-squared")
    parser.add_argument("--min_er", type=float, default=0.70, help="Minimum Efficiency Ratio")
    parser.add_argument("--min_net_change", type=float, default=8.0, help="Minimum net price change %%")
    parser.add_argument("--min_volume", type=int, default=500000, help="Minimum average volume")
    parser.add_argument("--min_turnover", type=float, default=1000000.0, help="Minimum average daily turnover (Price * Volume)")
    parser.add_argument("--max_drop", type=float, default=-5.0, help="Max single day drop allowed")
    parser.add_argument("--charts", action="store_true", help="Generate charts for top 10")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    
    args = parser.parse_args()
    
    # Handle Desktop path
    output_path = Path(args.output_dir).expanduser()
    if "Desktop" in args.output_dir:
        output_path = Path("~/Desktop/linearity scan").expanduser()
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    xlsx_file = output_path / "Linearity Scans.xlsx"
    tmp_file = output_path / "Linearity Scans.tmp.xlsx"
    
    try:
        with pd.ExcelWriter(tmp_file, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            for lookback in [60, 120]:
                config = {
                    'lookback': lookback,
                    'min_r2': args.min_r2,
                    'min_er': args.min_er,
                    'min_net_change': args.min_net_change,
                    'min_volume': args.min_volume,
                    'min_turnover': args.min_turnover,
                    'max_single_day_drop_limit': args.max_drop,
                    'parquet_path': 'nse_master_adjusted_2014_onwards.parquet',
                    'output_dir': output_path
                }
                
                logging.info(f"--- Running Linearity Scan (Lookback: {lookback}) ---")
                scanner = LinearityScanner(config)
                raw_df = scanner.load_data()
                
                if raw_df.empty:
                    logging.error(f"No data found for {lookback}D. Skipping.")
                    continue

                results_df = scanner.run_scan()
                filtered_df = scanner.filter_and_rank(results_df)
                
                sheet_name = f'Lookback {lookback}'
                filtered_df.to_excel(writer, index=False, sheet_name=sheet_name)
                worksheet = writer.sheets[sheet_name]
                
                if not filtered_df.empty:
                    (max_row, max_col) = filtered_df.shape
                    column_settings = [{'header': str(column)} for column in filtered_df.columns]
                    
                    header_fmt = workbook.add_format({
                        'bold': True,
                        'align': 'center',
                        'valign': 'vcenter',
                        'border': 1,
                        'bg_color': '#F2F2F2'
                    })

                    for col_num, value in enumerate(filtered_df.columns.values):
                        worksheet.write(0, col_num, str(value), header_fmt)
                    
                    worksheet.add_table(0, 0, max_row, max_col - 1, {
                        'columns': column_settings,
                        'style': 'Table Style Light 1',
                        'header_row': True
                    })
                    
                    worksheet.freeze_panes(1, 0)
                    
                    for i, col in enumerate(filtered_df.columns):
                        column_len = max(filtered_df[col].astype(str).map(len).max(), len(str(col))) + 2
                        worksheet.set_column(i, i, min(column_len, 50))
                
                print(f"\\n--- TOP LINEAR TRENDS ({lookback}D) ---")
                if not filtered_df.empty:
                    print(filtered_df[['symbol', 'r2', 'one_day_chg_pct', 'ema_20_dist_pct', 'ma_status']].head(15).to_string(index=False))
                else:
                    print("No matches found.")
            
        if xlsx_file.exists():
            xlsx_file.unlink()
        tmp_file.rename(xlsx_file)
        logging.info(f"Consolidated Linearity scans saved to {xlsx_file}")
        
        # open automatically 
        import subprocess
        subprocess.Popen(['open', str(xlsx_file)])

    except Exception as e:
        logging.error(f"Failed to save Excel file: {e}")
        if tmp_file.exists():
            tmp_file.unlink()

if __name__ == "__main__":
    main()
"""

# Replace `def main():` and everything below it
text, _ = re.subn(r"def main\(\):.*", new_main, text, flags=re.DOTALL)

with open("/Users/sumeetdas/Antigravity_NSE_Data/linearity_scanner.py", "w") as f:
    f.write(text)
