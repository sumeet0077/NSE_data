#!/usr/bin/env python3
"""
Daily NSE bhavcopy updater service.

Features:
- Attempts update at 18:00 IST, retries at 19:00 and 20:00 IST if needed.
- Multi-source fallback fetch logic with retries/backoff and anti-block headers.
- Enforces delivery data presence by default (can be overridden via CLI).
- Idempotent parquet merge/update by date into year partition.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import random
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests


WORKDIR = Path(__file__).resolve().parent
RAW_ROOT = WORKDIR / "raw_downloads"
PROCESSED_DAILY_ROOT = WORKDIR / "processed" / "daily"
MASTER_DIR = WORKDIR / "nse_master_bhav_with_delivery_2014_onwards.parquet"
METADATA_PATH = WORKDIR / "metadata.json"
STATE_PATH = WORKDIR / "daily_update_state.json"
LOG_PATH = WORKDIR / "daily_updater.log"

START_DATE = dt.date(2014, 1, 1)

MAX_RETRIES = 6
BACKOFF_START = 30
BACKOFF_MAX = 600
MIN_CONTENT_BYTES = 1024
MIN_SLEEP_BETWEEN_ATTEMPTS = 0.8
MAX_SLEEP_BETWEEN_ATTEMPTS = 2.5

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


import subprocess
import sys

@dataclass(frozen=True)
class FetchCandidate:
    method: str
    url: str
    is_zip: bool


def run_post_process() -> bool:
    """
    Runs the adjusted master builder script after a successful daily update.
    Returns True if successful, False otherwise.
    """
    logging.info("Starting post-process: Building Adjusted Master Dataset...")
    builder_script = WORKDIR / "build_adjusted_master.py"
    
    try:
        # Run the builder script using the same python interpreter
        result = subprocess.run(
            [sys.executable, str(builder_script)],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info("Adjusted Master Build SUCCESS.")
        # Optional: Log the last few lines of output
        if result.stdout:
            logging.info("Builder Output (tail): %s", result.stdout[-500:])
        return True
    except subprocess.CalledProcessError as e:
        logging.error("Adjusted Master Build FAILED (Exit Code %d)", e.returncode)
        logging.error("Stdout: %s", e.stdout[-1000:] if e.stdout else "None")
        logging.error("Stderr: %s", e.stderr[-1000:] if e.stderr else "None")
        return False
    except Exception as e:
        logging.error("Adjusted Master Build Exception: %s", e)
        return False


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )


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


def build_candidates(trade_date: dt.date) -> list[FetchCandidate]:
    ddmmyyyy = trade_date.strftime("%d%m%Y")
    dd = trade_date.strftime("%d")
    mmm = trade_date.strftime("%b").upper()
    yyyy = trade_date.strftime("%Y")
    yyyymmdd = trade_date.strftime("%Y%m%d")

    # Keep first source as delivery-enriched products feed for current-era dates.
    candidates = [
        FetchCandidate(
            method="products_sec_bhavdata_full",
            url=f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
            is_zip=False,
        ),
        FetchCandidate(
            method="archives_cm_udiff_zip",
            url=f"https://archives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip",
            is_zip=True,
        ),
        FetchCandidate(
            method="nsearchives_cm_udiff_zip",
            url=f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip",
            is_zip=True,
        ),
        FetchCandidate(
            method="historical_cm_zip",
            url=f"https://archives.nseindia.com/content/historical/EQUITIES/{yyyy}/{mmm}/cm{dd}{mmm}{yyyy}bhav.csv.zip",
            is_zip=True,
        ),
        FetchCandidate(
            method="archives_legacy_equities_csv",
            url=f"https://archives.nseindia.com/content/equities/sec_bhavdata_full_{ddmmyyyy}.csv",
            is_zip=False,
        ),
    ]
    return candidates


def download_with_retries(session: requests.Session, candidate: FetchCandidate) -> tuple[bytes | None, str]:
    backoff = BACKOFF_START
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(candidate.url, headers=random_headers(), timeout=45)
        except requests.RequestException as exc:
            logging.warning(
                "Request error (%s) %s attempt %d/%d: %s",
                candidate.method,
                candidate.url,
                attempt,
                MAX_RETRIES,
                exc,
            )
            if attempt == MAX_RETRIES:
                return None, f"request_exception:{exc}"
            time.sleep(min(backoff, BACKOFF_MAX))
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue

        code = resp.status_code
        content = resp.content
        if code == 200:
            if len(content) < MIN_CONTENT_BYTES:
                return None, f"small_payload:{len(content)}"
            if is_probably_html(content):
                return None, "html_payload"
            return content, "ok"

        if code in (403, 429) or 500 <= code < 600:
            logging.warning(
                "Retryable http_%d (%s) %s attempt %d/%d. backoff=%ss",
                code,
                candidate.method,
                candidate.url,
                attempt,
                MAX_RETRIES,
                backoff,
            )
            if attempt == MAX_RETRIES:
                return None, f"http_{code}"
            time.sleep(min(backoff, BACKOFF_MAX))
            backoff = min(backoff * 2, BACKOFF_MAX)
            continue

        return None, f"http_{code}"

    return None, "unknown_error"


def extract_csv_from_zip(content: bytes) -> bytes | None:
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                return None
            return zf.read(csv_names[0])
    except Exception:
        return None


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
            return pd.read_csv(io.StringIO(cleaned), skipinitialspace=True, **kwargs)
        except Exception:
            continue
    return pd.DataFrame()


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
    out["deliv_qty"] = to_uint(deliv_qty_col) if deliv_qty_col else pd.Series([pd.NA] * len(df), dtype="UInt32")
    out["deliv_pct"] = to_float(deliv_pct_col)
    out["source_url"] = source_url
    out["year"] = out["trade_date"].dt.year.astype("Int16")

    out = out[(out["symbol"] != "") & out["trade_date"].notna()]
    keep_mask = (
        out["open"].notna()
        | out["high"].notna()
        | out["low"].notna()
        | out["close"].notna()
        | out["volume"].notna()
    )
    out = out[keep_mask]
    if out.empty:
        return out

    out["symbol"] = out["symbol"].astype("category")
    out["series"] = out["series"].astype("category")
    return out


def raw_path_for_date(trade_date: dt.date, is_zip: bool) -> Path:
    out_dir = RAW_ROOT / f"{trade_date.year:04d}" / f"{trade_date.month:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = ".zip" if is_zip else ".csv"
    return out_dir / f"{trade_date.isoformat()}{ext}"


def fetch_daily_dataframe(
    session: requests.Session,
    trade_date: dt.date,
    require_delivery: bool = True,
) -> tuple[pd.DataFrame | None, str | None, list[str]]:
    errors: list[str] = []

    for candidate in build_candidates(trade_date):
        content, reason = download_with_retries(session, candidate)
        if content is None:
            errors.append(f"{candidate.method}:{reason}")
            continue

        raw_path = raw_path_for_date(trade_date, candidate.is_zip)
        raw_path.write_bytes(content)

        csv_bytes = extract_csv_from_zip(content) if candidate.is_zip else content
        if csv_bytes is None:
            errors.append(f"{candidate.method}:zip_extract_failed")
            continue

        parsed = read_csv_bytes(csv_bytes)
        norm = normalize_columns(parsed, trade_date, candidate.url)
        if norm.empty:
            errors.append(f"{candidate.method}:no_valid_rows")
            continue

        # Ensure date alignment if source contains noisy rows.
        norm = norm[norm["trade_date"].dt.date == trade_date]
        if norm.empty:
            errors.append(f"{candidate.method}:no_rows_for_trade_date")
            continue

        if require_delivery and int(norm["deliv_pct"].notna().sum()) == 0:
            errors.append(f"{candidate.method}:delivery_missing")
            time.sleep(random.uniform(MIN_SLEEP_BETWEEN_ATTEMPTS, MAX_SLEEP_BETWEEN_ATTEMPTS))
            continue

        return norm, candidate.url, errors

        # NOTE: return exits the loop, so spacing is only needed for failure/continue paths.

    return None, None, errors


def ensure_master_layout() -> None:
    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DAILY_ROOT.mkdir(parents=True, exist_ok=True)


def write_daily_processed(df: pd.DataFrame, trade_date: dt.date) -> Path:
    out_dir = PROCESSED_DAILY_ROOT / str(trade_date.year)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{trade_date.isoformat()}.parquet"
    df.to_parquet(out_file, engine="pyarrow", compression="zstd", compression_level=8, index=False)
    return out_file


def read_year_partition(year: int) -> pd.DataFrame:
    year_file = MASTER_DIR / f"year={year}" / f"part-{year}.parquet"
    if not year_file.exists():
        columns = [
            "symbol",
            "series",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover_lacs",
            "trades",
            "deliv_qty",
            "deliv_pct",
            "source_url",
        ]
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(year_file)


def merge_into_master(df: pd.DataFrame, trade_date: dt.date) -> tuple[int, int]:
    year = trade_date.year
    year_dir = MASTER_DIR / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)
    year_file = year_dir / f"part-{year}.parquet"

    existing = read_year_partition(year)
    before_rows = len(existing)

    incoming = df.drop(columns=["year"], errors="ignore").copy()
    incoming["trade_date"] = pd.to_datetime(incoming["trade_date"])

    merged = pd.concat([existing, incoming], ignore_index=True)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"], errors="coerce")
    merged = merged[merged["trade_date"].notna()]
    merged = merged.sort_values(["trade_date", "symbol", "series"], kind="mergesort")
    merged = merged.drop_duplicates(subset=["trade_date", "symbol", "series"], keep="last")

    merged["symbol"] = merged["symbol"].astype("category")
    merged["series"] = merged["series"].astype("category")
    merged["open"] = pd.to_numeric(merged["open"], errors="coerce").astype("Float32")
    merged["high"] = pd.to_numeric(merged["high"], errors="coerce").astype("Float32")
    merged["low"] = pd.to_numeric(merged["low"], errors="coerce").astype("Float32")
    merged["close"] = pd.to_numeric(merged["close"], errors="coerce").astype("Float32")
    merged["turnover_lacs"] = pd.to_numeric(merged["turnover_lacs"], errors="coerce").astype("Float32")
    merged["deliv_pct"] = pd.to_numeric(merged["deliv_pct"], errors="coerce").astype("Float32")

    for c in ("volume", "trades", "deliv_qty"):
        vals = pd.to_numeric(merged[c], errors="coerce")
        max_val = vals.max(skipna=True)
        if pd.notna(max_val) and max_val > 4294967295:
            merged[c] = vals.round().astype("UInt64")
        else:
            merged[c] = vals.round().astype("UInt32")

    tmp_file = year_file.with_suffix(".tmp.parquet")
    merged.to_parquet(tmp_file, engine="pyarrow", compression="zstd", compression_level=8, index=False)
    tmp_file.replace(year_file)
    return before_rows, len(merged)


def update_metadata() -> dict[str, object]:
    df = pd.read_parquet(MASTER_DIR, columns=["trade_date", "symbol", "deliv_pct", "year"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df = df[df["trade_date"].notna()]

    rows_2015_plus_mask = df["trade_date"].dt.year >= 2015
    rows_2015_plus = int(rows_2015_plus_mask.sum())
    rows_with_deliv_2015_plus = int(df.loc[rows_2015_plus_mask, "deliv_pct"].notna().sum())
    coverage = (rows_with_deliv_2015_plus / rows_2015_plus * 100.0) if rows_2015_plus else 0.0

    metadata = {
        "start_date": START_DATE.isoformat(),
        "end_date": dt.date.today().isoformat(),
        "min_trade_date": df["trade_date"].min().date().isoformat(),
        "max_trade_date": df["trade_date"].max().date().isoformat(),
        "row_count": int(len(df)),
        "unique_symbols": int(df["symbol"].nunique()),
        "rows_2015_onwards": rows_2015_plus,
        "rows_with_deliv_pct_2015_onwards": rows_with_deliv_2015_plus,
        "deliv_pct_coverage_2015_onwards_percent": round(coverage, 4),
        "feb_1_2026_exists": bool((df["trade_date"].dt.date == dt.date(2026, 2, 1)).any()),
        "generated_at": dt.datetime.now().isoformat(),
        "output_dataset_dir": str(MASTER_DIR.resolve()),
        "raw_downloads_dir": str(RAW_ROOT.resolve()),
        "processed_daily_dir": str(PROCESSED_DAILY_ROOT.resolve()),
        "log_path": str(LOG_PATH.resolve()),
        "years_present": sorted(df["year"].dropna().astype(int).unique().tolist()),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def update_for_date(
    session: requests.Session,
    trade_date: dt.date,
    require_delivery: bool,
) -> tuple[bool, str]:
    ensure_master_layout()

    # Fast idempotency check from processed daily cache.
    daily_file = PROCESSED_DAILY_ROOT / str(trade_date.year) / f"{trade_date.isoformat()}.parquet"
    if daily_file.exists():
        try:
            existing_daily = pd.read_parquet(daily_file)
            if not existing_daily.empty and int(existing_daily["deliv_pct"].notna().sum()) > 0:
                merge_into_master(existing_daily, trade_date)
                update_metadata()
                return True, "already_present_daily_cache_remerged"
        except Exception:
            pass

    df, source_url, errors = fetch_daily_dataframe(session, trade_date, require_delivery=require_delivery)
    if df is None:
        err_msg = "; ".join(errors[-8:]) if errors else "unknown_fetch_failure"
        return False, f"all_methods_failed:{err_msg}"

    write_daily_processed(df, trade_date)
    before_rows, after_rows = merge_into_master(df, trade_date)
    metadata = update_metadata()
    return (
        True,
        f"source={source_url} rows={len(df)} year_rows_before={before_rows} year_rows_after={after_rows} max_trade_date={metadata['max_trade_date']}",
    )


def load_state() -> dict[str, object]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state: dict[str, object]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def next_slot_datetime(now_ist: dt.datetime, slots: list[int]) -> dt.datetime:
    today = now_ist.date()
    candidates: list[dt.datetime] = []
    for day_delta in (0, 1):
        day = today + dt.timedelta(days=day_delta)
        for hour in slots:
            candidates.append(
                dt.datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=now_ist.tzinfo)
            )
    future = [x for x in candidates if x > now_ist]
    return min(future)


def should_attempt_slot(state: dict[str, object], trade_date: dt.date, hour: int) -> bool:
    date_key = trade_date.isoformat()
    if state.get("date") != date_key:
        return True
    if state.get("success") is True:
        return False
    attempted = set(state.get("attempted_hours", []))
    return hour not in attempted


def record_slot_attempt(
    state: dict[str, object],
    trade_date: dt.date,
    hour: int,
    success: bool,
    message: str,
) -> dict[str, object]:
    date_key = trade_date.isoformat()
    if state.get("date") != date_key:
        state = {
            "date": date_key,
            "attempted_hours": [],
            "success": False,
            "last_message": "",
            "updated_at": "",
        }
    attempted = list(state.get("attempted_hours", []))
    if hour not in attempted:
        attempted.append(hour)
    state["attempted_hours"] = sorted(attempted)
    state["success"] = bool(success)
    state["last_message"] = message
    state["updated_at"] = dt.datetime.now().isoformat()
    if success:
        state["success_hour"] = hour
    save_state(state)
    return state


def run_missed_slots_catchup(
    session: requests.Session,
    require_delivery: bool,
    slots: list[int],
    timezone: str,
) -> None:
    tz = ZoneInfo(timezone)
    now = dt.datetime.now(tz)
    today = now.date()
    current_hour = now.hour
    state = load_state()

    for slot_hour in slots:
        if slot_hour > current_hour:
            continue
        if not should_attempt_slot(state, today, slot_hour):
            continue

        logging.info("Catch-up attempt for missed/past slot=%02d:00 date=%s", slot_hour, today)
        success, msg = update_for_date(session, today, require_delivery=require_delivery)
        state = record_slot_attempt(state, today, slot_hour, success, msg)
        if success:
            logging.info("Catch-up SUCCESS date=%s slot=%02d:00 %s", today, slot_hour, msg)
            run_post_process()
            return
        logging.warning("Catch-up FAILED date=%s slot=%02d:00 %s", today, slot_hour, msg)



def load_metadata_safe() -> dict[str, Any]:
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def run_multi_day_catchup(
    session: requests.Session,
    require_delivery: bool,
    timezone: str,
) -> bool:
    """
    Checks if multiple days have been missed (e.g. laptop closed for days).
    Runs updates for all missing weekdays between last_success and today.
    Returns True if ANY update was performed (so we can trigger post-process once).
    """
    meta = load_metadata_safe()
    max_date_str = meta.get("max_trade_date")
    if not max_date_str:
        return False

    try:
        last_date = dt.date.fromisoformat(max_date_str)
    except ValueError:
        return False

    tz = ZoneInfo(timezone)
    today = dt.datetime.now(tz).date()
    
    # Safety: Don't look back too far to avoid infinite loops if something is wrong
    if (today - last_date).days > 30:
        logging.warning("Last update was over 30 days ago (%s). Catch-up limit capped at 30 days.", last_date)
        last_date = today - dt.timedelta(days=30)

    # Start checking from the day AFTER the last successful update
    current = last_date + dt.timedelta(days=1)
    updates_performed = False

    while current < today:
        # Try EVERY day. Markets can be open on weekends (Budget Sat/Sun, Mahurat Trading).
        # The fetcher handles "no data" gracefully (returns False).
        logging.info("Multi-day catch-up: Checking date %s", current)
        success, msg = update_for_date(session, current, require_delivery=require_delivery)
        if success:
            logging.info("Multi-day catch-up SUCCESS for %s: %s", current, msg)
            updates_performed = True
        else:
            # likely a holiday or closed weekend
            logging.info("Multi-day catch-up NO DATA for %s: %s", current, msg)
             
        current += dt.timedelta(days=1)

    return updates_performed


def run_service(require_delivery: bool, slots: list[int], timezone: str) -> None:
    tz = ZoneInfo(timezone)
    session = requests.Session()
    session.headers.update({"Accept-Language": "en-US,en;q=0.9"})

    try:
        session.get("https://www.nseindia.com/", headers=random_headers(), timeout=30)
    except Exception:
        pass

    logging.info("Service started. timezone=%s slots=%s require_delivery=%s", timezone, slots, require_delivery)
    state = load_state()

    while True:
        # 1. Check for multi-day gaps (long downtime)
        if run_multi_day_catchup(session, require_delivery, timezone):
            logging.info("Multi-day catch-up performed updates. Triggering post-process...")
            run_post_process()

        # 2. Check for today's missed slots
        run_missed_slots_catchup(
            session=session,
            require_delivery=require_delivery,
            slots=slots,
            timezone=timezone,
        )

        now = dt.datetime.now(tz)
        slot_dt = next_slot_datetime(now, slots)
        wait_seconds = max((slot_dt - now).total_seconds(), 0.0)
        logging.info("Next slot at %s (in %.1f minutes)", slot_dt.isoformat(), wait_seconds / 60.0)

        while True:
            now_inner = dt.datetime.now(tz)
            remaining = (slot_dt - now_inner).total_seconds()
            if remaining <= 0:
                break
            time.sleep(min(remaining, 60.0))

        trade_date = slot_dt.date()
        hour = slot_dt.hour
        state = load_state()

        if not should_attempt_slot(state, trade_date, hour):
            logging.info("Skipping slot %s: already attempted or success achieved for date %s", hour, trade_date)
            continue

        logging.info("Running update attempt for trade_date=%s at slot=%02d:00", trade_date, hour)
        success, msg = update_for_date(session, trade_date, require_delivery=require_delivery)
        state = record_slot_attempt(state, trade_date, hour, success, msg)
        if success:
            logging.info("SUCCESS date=%s slot=%02d:00 %s", trade_date, hour, msg)
            # Trigger adjustment build
            run_post_process()
        else:
            logging.warning("FAILED date=%s slot=%02d:00 %s", trade_date, hour, msg)


def run_once(trade_date: dt.date, require_delivery: bool) -> int:
    session = requests.Session()
    session.headers.update({"Accept-Language": "en-US,en;q=0.9"})
    try:
        session.get("https://www.nseindia.com/", headers=random_headers(), timeout=30)
    except Exception:
        pass

    success, msg = update_for_date(session, trade_date, require_delivery=require_delivery)
    if success:
        logging.info("ONE-SHOT SUCCESS for %s: %s", trade_date, msg)
        run_post_process()
        return 0
    logging.error("ONE-SHOT FAILED for %s: %s", trade_date, msg)
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NSE daily updater (6PM/7PM/8PM IST scheduler + one-shot mode)")
    parser.add_argument("--service", action="store_true", help="Run long-lived scheduler service")
    parser.add_argument("--run-once", action="store_true", help="Run one immediate update")
    parser.add_argument("--date", type=str, default="", help="Trade date YYYY-MM-DD (for --run-once)")
    parser.add_argument("--timezone", type=str, default="Asia/Kolkata", help="Scheduler timezone")
    parser.add_argument("--slots", type=str, default="18,19,20", help="Comma-separated retry slot hours")
    parser.add_argument(
        "--allow-missing-delivery",
        action="store_true",
        help="Allow update even if delivery fields are missing",
    )
    return parser.parse_args()


def parse_date_or_today(date_str: str, timezone: str) -> dt.date:
    if date_str:
        return dt.date.fromisoformat(date_str)
    return dt.datetime.now(ZoneInfo(timezone)).date()


def parse_slots(slot_str: str) -> list[int]:
    out: list[int] = []
    for item in slot_str.split(","):
        v = int(item.strip())
        if v < 0 or v > 23:
            raise ValueError(f"Invalid hour: {v}")
        out.append(v)
    if not out:
        raise ValueError("No slots provided")
    return sorted(set(out))


def main() -> int:
    setup_logging()
    args = parse_args()
    require_delivery = not args.allow_missing_delivery
    slots = parse_slots(args.slots)

    if args.service:
        run_service(require_delivery=require_delivery, slots=slots, timezone=args.timezone)
        return 0

    if args.run_once:
        trade_date = parse_date_or_today(args.date, args.timezone)
        return run_once(trade_date=trade_date, require_delivery=require_delivery)

    logging.error("Specify either --service or --run-once")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
