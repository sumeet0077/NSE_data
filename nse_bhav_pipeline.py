#!/usr/bin/env python3
"""
Build NSE equity bhavcopy master dataset (2014 onward) with delivery fields where available.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import logging
import random
import time
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


START_DATE = dt.date(2014, 1, 1)
END_DATE = dt.date.today()

RAW_ROOT = Path("raw_downloads")
PROCESSED_DAILY_ROOT = Path("processed") / "daily"
FINAL_DATASET_DIR = Path("nse_master_bhav_with_delivery_2014_onwards.parquet")
METADATA_PATH = Path("metadata.json")
LOG_PATH = Path("pipeline.log")

MAX_RETRIES = 6
BACKOFF_START = 30
BACKOFF_MAX = 600
MIN_CONTENT_BYTES = 1024

# Kept low enough to complete full backfill in one run; backoff handles throttling spikes.
DELAY_LOW = 0.2
DELAY_HIGH = 0.8

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.160 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.160 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.160 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edg/131.0.2903.112",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chromium/132.0.6834.110 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.265 Safari/537.36",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )


def daterange(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    cur = start
    while cur <= end:
        yield cur
        cur += dt.timedelta(days=1)


def random_headers() -> dict[str, str]:
    referer = "https://www.nseindia.com/" if random.random() < 0.5 else "https://archives.nseindia.com/"
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Referer": referer,
        "Accept": "*/*",
        "Connection": "keep-alive",
    }


def is_probably_html(content: bytes) -> bool:
    head = content[:512].decode("utf-8", errors="ignore").lower()
    return "<!doctype html" in head or "<html" in head


def expected_raw_path(trade_date: dt.date, url: str) -> Path:
    ym_dir = RAW_ROOT / f"{trade_date.year:04d}" / f"{trade_date.month:02d}"
    ym_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".zip" if url.lower().endswith(".zip") else ".csv"
    return ym_dir / f"{trade_date.isoformat()}{suffix}"


def download_with_retries(session: requests.Session, url: str) -> tuple[bytes | None, str | None]:
    backoff = BACKOFF_START
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=random_headers(), timeout=45)
        except requests.RequestException as exc:
            logging.warning("Request error for %s (attempt %d/%d): %s", url, attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                return None, f"request_exception: {exc}"
            time.sleep(min(backoff, BACKOFF_MAX))
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue

        status = resp.status_code
        content = resp.content
        if status == 200:
            if len(content) < MIN_CONTENT_BYTES:
                return None, f"too_small_{len(content)}"
            if is_probably_html(content):
                return None, "html_payload"
            return content, None

        if status in (403, 429) or 500 <= status < 600:
            logging.warning(
                "Retryable status %d for %s (attempt %d/%d). sleeping %ss",
                status,
                url,
                attempt,
                MAX_RETRIES,
                backoff,
            )
            if attempt == MAX_RETRIES:
                return None, f"http_{status}"
            time.sleep(min(backoff, BACKOFF_MAX))
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue

        return None, f"http_{status}"

    return None, "unknown_error"


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    for kwargs in (
        {"engine": "c", "on_bad_lines": "skip"},
        {"engine": "python", "on_bad_lines": "skip"},
    ):
        try:
            df = pd.read_csv(io.BytesIO(content), skipinitialspace=True, **kwargs)
            if not df.empty:
                return df
        except Exception:
            continue

    text = content.decode("utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("=")]
    if not lines:
        return pd.DataFrame()
    cleaned = "\n".join(lines)
    for kwargs in (
        {"engine": "c", "on_bad_lines": "skip"},
        {"engine": "python", "on_bad_lines": "skip"},
    ):
        try:
            df = pd.read_csv(io.StringIO(cleaned), skipinitialspace=True, **kwargs)
            return df
        except Exception:
            continue
    return pd.DataFrame()


def extract_csv_from_zip(content: bytes) -> bytes | None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_members:
                return None
            return zf.read(csv_members[0])
    except Exception:
        return None


def normalize_columns(df: pd.DataFrame, trade_date_hint: dt.date, source_url: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_upper = {c.upper(): c for c in df.columns}

    def get_col(*names: str) -> str | None:
        for name in names:
            if name in col_upper:
                return col_upper[name]
        return None

    symbol_col = get_col("SYMBOL", "TCKRSYMB")
    series_col = get_col("SERIES", "SCTYSRS")
    date_col = get_col("DATE1", "TRADE_DATE", "TIMESTAMP", "TRADDT")
    open_col = get_col("OPEN_PRICE", "OPEN", "OPNPRIC")
    high_col = get_col("HIGH_PRICE", "HIGH", "HGHPRIC")
    low_col = get_col("LOW_PRICE", "LOW", "LWPRIC")
    close_col = get_col("CLOSE_PRICE", "CLOSE", "CLSPRIC")
    volume_col = get_col("TTL_TRD_QNTY", "TOTTRDQTY", "TTLTRADGVOL")
    turnover_lacs_col = get_col("TURNOVER_LACS")
    turnover_raw_col = get_col("TOTTRDVAL", "TTLTRFVAL")
    trades_col = get_col("NO_OF_TRADES", "TOTALTRADES", "TTLNBOFTXSEXCTD")
    deliv_qty_col = get_col("DELIV_QTY", "DELIVERABLE_QTY", "DELIVQTY")
    deliv_pct_col = get_col("DELIV_PER", "%DELIV", "DELIVPER")

    out = pd.DataFrame()
    out["symbol"] = df[symbol_col].astype(str).str.strip() if symbol_col else ""
    out["series"] = df[series_col].astype(str).str.strip() if series_col else ""

    if date_col:
        parsed_date = pd.to_datetime(df[date_col].astype(str).str.strip(), errors="coerce")
        parsed_date = parsed_date.fillna(pd.Timestamp(trade_date_hint))
    else:
        parsed_date = pd.Series(pd.Timestamp(trade_date_hint), index=df.index)
    out["trade_date"] = pd.to_datetime(parsed_date.dt.date)

    def to_float(col: str | None) -> pd.Series:
        if not col:
            return pd.Series([pd.NA] * len(df), dtype="Float32")
        return pd.to_numeric(df[col], errors="coerce").astype("Float32")

    def to_uint(col: str | None) -> pd.Series:
        if not col:
            return pd.Series([pd.NA] * len(df), dtype="UInt32")
        vals = pd.to_numeric(df[col], errors="coerce")
        max_val = vals.max(skipna=True)
        if pd.notna(max_val) and max_val > 4294967295:
            return vals.round().astype("UInt64")
        return vals.round().astype("UInt32")

    out["open"] = to_float(open_col)
    out["high"] = to_float(high_col)
    out["low"] = to_float(low_col)
    out["close"] = to_float(close_col)
    out["volume"] = to_uint(volume_col)

    if turnover_lacs_col:
        out["turnover_lacs"] = pd.to_numeric(df[turnover_lacs_col], errors="coerce").astype("Float32")
    elif turnover_raw_col:
        out["turnover_lacs"] = (pd.to_numeric(df[turnover_raw_col], errors="coerce") / 100000.0).astype("Float32")
    else:
        out["turnover_lacs"] = pd.Series([pd.NA] * len(df), dtype="Float32")

    out["trades"] = to_uint(trades_col)

    if deliv_qty_col:
        out["deliv_qty"] = to_uint(deliv_qty_col)
    else:
        out["deliv_qty"] = pd.Series([pd.NA] * len(df), dtype="UInt32")

    out["deliv_pct"] = to_float(deliv_pct_col)

    out["source_url"] = source_url
    out["year"] = out["trade_date"].dt.year.astype("Int16")

    out = out[(out["symbol"] != "") & out["trade_date"].notna()]
    out["symbol"] = out["symbol"].astype("category")
    out["series"] = out["series"].astype("category")
    return out


def url_candidates(trade_date: dt.date) -> list[str]:
    ddmmyyyy = trade_date.strftime("%d%m%Y")
    dd = trade_date.strftime("%d")
    mmm = trade_date.strftime("%b").upper()
    yyyy = trade_date.strftime("%Y")
    yyyymmdd = trade_date.strftime("%Y%m%d")

    candidates = [
        f"https://archives.nseindia.com/content/equities/sec_bhavdata_full_{ddmmyyyy}.csv",
        f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
        f"https://archives.nseindia.com/content/historical/EQUITIES/{yyyy}/{mmm}/cm{dd}{mmm}{yyyy}bhav.csv.zip",
    ]
    if trade_date >= dt.date(2024, 7, 1):
        candidates.extend(
            [
                f"https://archives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip",
                f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip",
            ]
        )
    return candidates


def process_one_date(
    session: requests.Session, trade_date: dt.date
) -> tuple[str, str | None, int, bool]:
    daily_dir = PROCESSED_DAILY_ROOT / str(trade_date.year)
    daily_dir.mkdir(parents=True, exist_ok=True)
    daily_file = daily_dir / f"{trade_date.isoformat()}.parquet"
    if daily_file.exists():
        return "already_processed", None, 0, False

    candidates = url_candidates(trade_date)
    for url in candidates:
        content, reason = download_with_retries(session, url)
        if content is None:
            continue

        raw_path = expected_raw_path(trade_date, url)
        raw_path.write_bytes(content)

        if url.lower().endswith(".zip"):
            csv_bytes = extract_csv_from_zip(content)
            if not csv_bytes:
                continue
        else:
            csv_bytes = content

        df = read_csv_bytes(csv_bytes)
        norm = normalize_columns(df, trade_date, url)
        if norm.empty:
            continue

        # Trim rows where there is no meaningful price and no volume.
        keep_mask = (
            norm["open"].notna()
            | norm["high"].notna()
            | norm["low"].notna()
            | norm["close"].notna()
            | norm["volume"].notna()
        )
        norm = norm[keep_mask]
        if norm.empty:
            continue

        norm.to_parquet(
            daily_file,
            engine="pyarrow",
            compression="zstd",
            compression_level=8,
            index=False,
        )

        delivery_missing = bool(norm["deliv_pct"].isna().all())
        return "ok", url, len(norm), delivery_missing

    return "not_trading_or_unavailable", None, 0, False


def aggregate_year(year: int) -> tuple[int, int, int, int, dt.date | None, dt.date | None, bool]:
    year_dir = PROCESSED_DAILY_ROOT / str(year)
    files = sorted(year_dir.glob("*.parquet"))
    if not files:
        return 0, 0, 0, 0, None, None, False

    dfs = [pd.read_parquet(path) for path in files]
    df = pd.concat(dfs, ignore_index=True)

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df[df["trade_date"].notna()]
    df["year"] = df["trade_date"].dt.year.astype("Int16")

    df = df.sort_values(["trade_date", "symbol"], kind="mergesort")
    # Prefer the first occurrence if duplicates exist (though they shouldn't in a clean run)
    df = df.drop_duplicates(subset=["trade_date", "symbol", "series"], keep="first")

    df["symbol"] = df["symbol"].astype("category")
    df["series"] = df["series"].astype("category")
    df["open"] = pd.to_numeric(df["open"], errors="coerce").astype("Float32")
    df["high"] = pd.to_numeric(df["high"], errors="coerce").astype("Float32")
    df["low"] = pd.to_numeric(df["low"], errors="coerce").astype("Float32")
    df["close"] = pd.to_numeric(df["close"], errors="coerce").astype("Float32")
    df["turnover_lacs"] = pd.to_numeric(df["turnover_lacs"], errors="coerce").astype("Float32")
    df["deliv_pct"] = pd.to_numeric(df["deliv_pct"], errors="coerce").astype("Float32")

    for c in ("volume", "trades", "deliv_qty"):
        vals = pd.to_numeric(df[c], errors="coerce")
        max_val = vals.max(skipna=True)
        if pd.notna(max_val) and max_val > 4294967295:
            df[c] = vals.round().astype("UInt64")
        else:
            df[c] = vals.round().astype("UInt32")

    out_dir = FINAL_DATASET_DIR / f"year={year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"part-{year}.parquet"
    write_df = df.drop(columns=["year"], errors="ignore")
    write_df.to_parquet(
        out_file,
        engine="pyarrow",
        compression="zstd",
        compression_level=8,
        index=False,
    )

    rows = len(df)
    unique_symbols = int(df["symbol"].nunique(dropna=True))
    rows_2015_plus = int((df["trade_date"].dt.year >= 2015).sum())
    deliv_2015_plus = int(((df["trade_date"].dt.year >= 2015) & df["deliv_pct"].notna()).sum())
    min_date = df["trade_date"].min().date() if not df.empty else None
    max_date = df["trade_date"].max().date() if not df.empty else None
    feb1_exists = bool((df["trade_date"].dt.date == dt.date(2026, 2, 1)).any())
    return rows, unique_symbols, rows_2015_plus, deliv_2015_plus, min_date, max_date, feb1_exists


def main() -> None:
    setup_logging()
    logging.info("Pipeline start. range=%s to %s", START_DATE, END_DATE)

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    PROCESSED_DAILY_ROOT.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"Accept-Language": "en-US,en;q=0.9"})

    # Warm-up to build NSE cookie context.
    try:
        session.get("https://www.nseindia.com/", headers=random_headers(), timeout=30)
    except Exception:
        pass

    total_dates = (END_DATE - START_DATE).days + 1
    done = 0
    ok = 0
    skipped = 0
    already = 0
    delivery_missing_days = 0
    last_success: dt.date | None = None

    for d in daterange(START_DATE, END_DATE):
        done += 1
        status, used_url, nrows, delivery_missing = process_one_date(session, d)
        if status == "ok":
            ok += 1
            last_success = d
            if delivery_missing:
                delivery_missing_days += 1
        elif status == "already_processed":
            already += 1
            last_success = d
        else:
            skipped += 1

        if done % 50 == 0:
            logging.info(
                "Progress %d/%d | ok=%d already=%d skipped=%d | last_success=%s",
                done,
                total_dates,
                ok,
                already,
                skipped,
                last_success,
            )
            logging.info("Delivery-missing trading days so far: %d", delivery_missing_days)

        if status != "already_processed":
            time.sleep(random.uniform(DELAY_LOW, DELAY_HIGH))

    logging.info(
        "Download+normalize done. total=%d ok=%d already=%d skipped=%d delivery_missing_days=%d last_success=%s",
        total_dates,
        ok,
        already,
        skipped,
        delivery_missing_days,
        last_success,
    )

    if FINAL_DATASET_DIR.exists():
        for p in sorted(FINAL_DATASET_DIR.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        FINAL_DATASET_DIR.rmdir()
    FINAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    all_years = sorted({d.year for d in daterange(START_DATE, END_DATE)})
    total_rows = 0
    all_symbols: set[str] = set()
    rows_2015_plus = 0
    deliv_2015_plus = 0
    min_trade_date: dt.date | None = None
    max_trade_date: dt.date | None = None
    feb1_2026_exists = False

    for year in all_years:
        (
            y_rows,
            y_unique_symbols,
            y_rows_2015_plus,
            y_deliv_2015_plus,
            y_min,
            y_max,
            y_feb1,
        ) = aggregate_year(year)
        if y_rows == 0:
            continue

        sample_df = pd.read_parquet(FINAL_DATASET_DIR / f"year={year}" / f"part-{year}.parquet", columns=["symbol"])
        all_symbols.update(sample_df["symbol"].astype(str).unique().tolist())

        total_rows += y_rows
        rows_2015_plus += y_rows_2015_plus
        deliv_2015_plus += y_deliv_2015_plus
        feb1_2026_exists = feb1_2026_exists or y_feb1

        if y_min and (min_trade_date is None or y_min < min_trade_date):
            min_trade_date = y_min
        if y_max and (max_trade_date is None or y_max > max_trade_date):
            max_trade_date = y_max

        logging.info("Year %d aggregated rows=%d unique_symbols=%d", year, y_rows, y_unique_symbols)

    deliv_pct_coverage_after_2015 = (deliv_2015_plus / rows_2015_plus * 100.0) if rows_2015_plus else 0.0

    metadata = {
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "min_trade_date": min_trade_date.isoformat() if min_trade_date else None,
        "max_trade_date": max_trade_date.isoformat() if max_trade_date else None,
        "row_count": total_rows,
        "unique_symbols": len(all_symbols),
        "rows_2015_onwards": rows_2015_plus,
        "rows_with_deliv_pct_2015_onwards": deliv_2015_plus,
        "deliv_pct_coverage_2015_onwards_percent": round(deliv_pct_coverage_after_2015, 4),
        "feb_1_2026_exists": feb1_2026_exists,
        "generated_at": dt.datetime.now().isoformat(),
        "output_dataset_dir": str(FINAL_DATASET_DIR.resolve()),
        "raw_downloads_dir": str(RAW_ROOT.resolve()),
        "processed_daily_dir": str(PROCESSED_DAILY_ROOT.resolve()),
        "log_path": str(LOG_PATH.resolve()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logging.info("Metadata written to %s", METADATA_PATH.resolve())
    logging.info("DONE: %s", json.dumps(metadata))


if __name__ == "__main__":
    main()
