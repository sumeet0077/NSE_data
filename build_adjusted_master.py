#!/usr/bin/env python3
"""
Create a new NSE master Parquet dataset with an added `adjusted_close` column.

CRITICAL SAFETY:
- Reads from:  ./nse_master_bhav_with_delivery_2014_onwards.parquet
- Writes ONLY: ./nse_master_adjusted_2014_onwards.parquet
- Never modifies the original dataset.

Adjustment goal:
- Backward-adjusted close suitable for long-horizon analysis (splits/bonuses/rights/dividends).
- Primary method per symbol: yfinance (no auth), else NSE corporate actions, else raw close.

This script is designed to be run end-to-end, once.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import random
import re
import resource
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import yfinance as yf


# Default Paths (can be overridden via CLI)
ORIG_DIR = Path("nse_master_bhav_with_delivery_2014_onwards.parquet")
OUT_DIR = Path("nse_master_adjusted_2014_onwards.parquet")
METADATA_PATH = Path("metadata.json")
REPORT_PATH = Path("adjusted_close_report.json")
LOG_PATH = Path("adjusted_close_build.log")
CA_CACHE_PATH = Path("nse_corporate_actions_equities_2014_onwards.parquet")

# yfinance settings
YF_START = "2013-12-01"
YF_TIMEOUT = 30
YF_BATCH_SIZE = 200
YF_MAX_RETRIES = 3
YF_RATE_LIMITED = False

# NSE corporate actions endpoint (public, no auth, but needs cookies)
NSE_CA_URL = "https://www.nseindia.com/api/corporates-corporateActions"
NSE_CA_MAX_RETRIES = 4

ALIGN_TOLERANCE_DAYS = 5

# Other public fallbacks (often key-gated; used only if NSE manual fails)
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"
FMP_URL = "https://financialmodelingprep.com/api/v3"
EODHD_URL = "https://eodhistoricaldata.com/api"


USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.160 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.160 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.6834.160 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:134.0) Gecko/20100101 Firefox/134.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.7; rv:134.0) Gecko/20100101 Firefox/134.0",
]

UNSUPPORTED_YF_RE = [
    re.compile(r"^\d+(?:GS|GR)\d+[A-Z]?$", re.I),  # Govt securities, e.g. 679GS2034A
    re.compile(r"^\d+D\d+$", re.I),  # T-bill like 91D031122
    re.compile(r"^\d+(?:NABAR|IIFCL)\d+$", re.I),  # bond-like symbols in CM file
]
UNSUPPORTED_YF_SUFFIX_RE = re.compile(r"-RE\d*$", re.I)  # rights entitlement suffixes


def should_skip_yfinance(symbol: str) -> bool:
    s = (symbol or "").strip()
    if not s:
        return True
    if UNSUPPORTED_YF_SUFFIX_RE.search(s):
        return True
    for rx in UNSUPPORTED_YF_RE:
        if rx.match(s):
            return True
    return False


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
    )
    # yfinance can be very noisy for missing/delisted symbols; keep our logs readable.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("yfinance").propagate = False


def load_metadata() -> dict[str, Any]:
    if METADATA_PATH.exists():
        try:
            return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def parse_nse_date(s: str) -> dt.date:
    # exDate is like "09-Feb-2026"
    return dt.datetime.strptime(s.strip(), "%d-%b-%Y").date()


def random_headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    }


def fetch_nse_corporate_actions(from_date: str, to_date: str) -> list[dict[str, Any]]:
    """
    Fetch NSE corporate actions (both equities and SME) for a date window.
    from_date/to_date are DD-MM-YYYY.
    """
    import requests

    s = requests.Session()

    # Warm-up to obtain cookies.
    try:
        s.get("https://www.nseindia.com/", headers=random_headers(), timeout=20)
    except Exception:
        pass

    results: list[dict[str, Any]] = []
    for idx_name in ("equities", "sme"):
        params = {"index": idx_name, "from_date": from_date, "to_date": to_date}
        backoff = 3
        for attempt in range(1, NSE_CA_MAX_RETRIES + 1):
            try:
                r = s.get(NSE_CA_URL, params=params, headers=random_headers(), timeout=60)
            except requests.RequestException as exc:
                logging.warning("NSE CA (%s) request error attempt %d/%d: %s", idx_name, attempt, NSE_CA_MAX_RETRIES, exc)
                if attempt == NSE_CA_MAX_RETRIES:
                    break
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue

            if r.status_code == 200:
                try:
                    data = r.json()
                    if isinstance(data, list):
                        results.extend(data)
                        break
                except Exception:
                    pass
            else:
                logging.warning(
                    "NSE CA (%s) http_%d attempt %d/%d (from=%s to=%s)",
                    idx_name,
                    r.status_code,
                    attempt,
                    NSE_CA_MAX_RETRIES,
                    from_date,
                    to_date,
                )
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    return results


@dataclass(frozen=True)
class ParsedEvent:
    kind: str  # bonus|split|rights|dividend|distribution|reorg
    a: float = 0.0  # numerator (e.g., bonus/right shares)
    b: float = 0.0  # denominator (e.g., existing shares)
    premium: float = 0.0
    amount: float = 0.0
    # For rights issues, NSE subjects sometimes state the *issue price* ("@ Rs 50")
    # and sometimes state only the *premium* ("premium Rs 40") which must be added
    # to the face value to obtain the issue price. This flag disambiguates.
    premium_is_issue_price: bool = False


# IMPORTANT: In raw strings, use single backslashes for regex escapes (\s, \d, \.) etc.
# Double-escaping (e.g. r"\\s") matches literal "\s" and breaks parsing.
BONUS_RE = re.compile(r"bonus(?:\s+issue)?(?:\s+of)?\s*(\d+)\s*:\s*(\d+)", re.I)

# Broadened Rights Issue regex suite to handle "Premium Of Rs", "@ Premium X", "@ Prm", "At Par", and bare numbers
RIGHTS_RATIO_RE = re.compile(r"rights[^\d]*(\d+)\s*:\s*(\d+)", re.I)
RIGHTS_PREM_RE = re.compile(r"(?:prem|premium|prm)(?:\s+of)?\s*(?:rs\.?|re\.?)?\s*(\d+(?:\.\d+)?)", re.I)
RIGHTS_AT_RE = re.compile(r"(?:@|at)\s*(?:rs\.?|re\.?)?\s*(\d+(?:\.\d+)?)", re.I)
RIGHTS_AT_PAR_RE = re.compile(r"at\s+par\b", re.I)

SPLIT_RE = re.compile(
    r"(?:split|sub-?division|subdivision).*?from\s*(?:rs\.?|re\.?)\s*(\d+(?:\.\d+)?)\s*.*?to\s*(?:rs\.?|re\.?)\s*(\d+(?:\.\d+)?)",
    re.I,
)
SPLIT_SIMPLE_RE = re.compile(
    r"(?:split|sub-?division|subdivision).*?(?:rs\.?|re\.?)\s*(\d+(?:\.\d+)?)\s*(?:/-)?\s*to\s*(?:rs\.?|re\.?)\s*(\d+(?:\.\d+)?)",
    re.I,
)
DIV_RE = re.compile(r"dividend[^\d]*(?:rs\.?|re\.?)\s*(\d+(?:\.\d+)?)", re.I)
DIV_PCT_RE = re.compile(r"dividend[^\d]*(\d+(?:\.\d+)?)\s*%", re.I)
DIST_RE = re.compile(r"distribution\s*-\s*(?:rs\.?|re\.?)\s*(\d+(?:\.\d+)?)", re.I)
REORG_RE = re.compile(
    r"\b(demerger|de-?merger|spin\s*-?off|spinoff|scheme of arrangement|restructur|amalgamation|merger|hiving\s*off|hive\s*off)\b",
    re.I,
)


def parse_subject(subject: str, face_val: float) -> list[ParsedEvent]:
    s = (subject or "").strip()
    s = re.sub(r"\s+", " ", s)
    out: list[ParsedEvent] = []
    if not s:
        return out

    for m in BONUS_RE.finditer(s):
        out.append(ParsedEvent(kind="bonus", a=float(m.group(1)), b=float(m.group(2))))

    # Split (face value subdivision)
    m = SPLIT_RE.search(s) or SPLIT_SIMPLE_RE.search(s)
    if m:
        old = float(m.group(1))
        new = float(m.group(2))
        if old > 0 and new > 0:
            out.append(ParsedEvent(kind="split", a=old, b=new))

    # Rights (generalized parser capturing 100% of equity rights offerings)
    rm = RIGHTS_RATIO_RE.search(s)
    if rm:
        a = float(rm.group(1))
        b = float(rm.group(2))
        prem_m = RIGHTS_PREM_RE.search(s)
        at_m = RIGHTS_AT_RE.search(s)
        at_par_m = RIGHTS_AT_PAR_RE.search(s)

        if prem_m:
            prem = float(prem_m.group(1))
            out.append(ParsedEvent(kind="rights", a=a, b=b, premium=prem, premium_is_issue_price=False))
        elif at_m and not at_par_m:
            issue_price = float(at_m.group(1))
            out.append(ParsedEvent(kind="rights", a=a, b=b, premium=issue_price, premium_is_issue_price=True))
        else:
            # At par or unspecified subscription price
            out.append(ParsedEvent(kind="rights", a=a, b=b, premium=0.0, premium_is_issue_price=False))

    # Distribution (InvIT/REIT style) and Dividend (tracked, but decoupled from primary price series)
    m = DIST_RE.search(s)
    if m:
        amt = float(m.group(1))
        if amt > 0:
            out.append(ParsedEvent(kind="distribution", amount=amt))
    else:
        m = DIV_RE.search(s)
        if m:
            amt = float(m.group(1))
            if amt > 0:
                out.append(ParsedEvent(kind="dividend", amount=amt))
        else:
            m = DIV_PCT_RE.search(s)
            if m and face_val and face_val > 0:
                pct = float(m.group(1))
                amt = face_val * pct / 100.0
                if amt > 0:
                    out.append(ParsedEvent(kind="dividend", amount=amt))

    # Corporate re-orgs (demergers/spin-offs/schemes).
    # Filter out preference shares, NCRPS, debentures, and bonds which are debt instruments, not equity spin-offs.
    is_debt_or_pref = bool(re.search(r"\b(ncrps|preferences?|preferential|debentures?|bonds?)\b", s, re.I))
    if not out and REORG_RE.search(s) and not is_debt_or_pref:
        is_explicit = bool(re.search(r"\b(demerger|de-?merger|spin\s*-?off|spinoff|hiving\s*off|hive\s*off)\b", s, re.I))
        out.append(ParsedEvent(kind="reorg", premium=1.0 if is_explicit else 0.0))

    return out


def is_significant_ca(actions: list[dict[str, Any]], reference_date: dt.date, window_days: int = 15) -> bool:
    """
    Check if any split, bonus, or rights issue happened within the window relative to reference_date.
    """
    cutoff = reference_date - dt.timedelta(days=window_days)
    for rec in actions:
        subj = str(rec.get("subject", "") or "").lower()
        if any(k in subj for k in ("split", "bonus", "rights", "reorg")):
            try:
                ex_d = parse_nse_date(str(rec.get("exDate", "")).strip())
                if ex_d >= cutoff:
                    return True
            except Exception:
                continue
    return False


def has_recent_ca(actions: list[dict[str, Any]], reference_date: dt.date, window_days: int = 30) -> bool:
    """
    Check if ANY corporate action (including dividends) happened within the window relative to reference_date.
    """
    cutoff = reference_date - dt.timedelta(days=window_days)
    for rec in actions:
        try:
            ex_d = parse_nse_date(str(rec.get("exDate", "")).strip())
            if ex_d >= cutoff:
                return True
        except Exception:
            continue
    return False


def align_effective_date(ex_date: dt.date, dates: pd.DatetimeIndex) -> pd.Timestamp | None:
    """
    Align an ex-date to the nearest available trading date >= ex_date within tolerance.
    """
    if not len(dates):
        return None
    ex_ts = pd.Timestamp(ex_date)
    # if exact present
    if ex_ts in dates:
        return ex_ts
    pos = dates.searchsorted(ex_ts)
    if pos >= len(dates):
        return None
    cand = dates[pos]
    if (cand - ex_ts).days <= ALIGN_TOLERANCE_DAYS:
        return cand
    return None


def max_abs_dev_from_one(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype="float64")
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - 1.0)))


def build_manual_factor_series(
    symbol: str,
    dates: pd.DatetimeIndex,
    ref_close: pd.Series,
    ref_open: pd.Series,
    actions: list[dict[str, Any]],
) -> np.ndarray:
    """
    Returns factor per date (float32), where adjusted_close = close * factor.
    """
    capital_factor_by_date: dict[pd.Timestamp, float] = {}
    reorg_factor_by_date: dict[pd.Timestamp, float] = {}

    prev_close = ref_close.shift(1)

    for rec in actions:
        try:
            ex_d = parse_nse_date(str(rec.get("exDate", "")).strip())
        except Exception:
            continue

        eff = align_effective_date(ex_d, dates)
        if eff is None:
            continue
        if eff > dates[-1]:
            continue

        subj = str(rec.get("subject", "") or "")
        face_val = 0.0
        try:
            face_val = float(str(rec.get("faceVal", "") or "0").strip() or 0.0)
        except Exception:
            face_val = 0.0

        for ev in parse_subject(subj, face_val):
            f: float | None = None
            if ev.kind == "split":
                old = ev.a
                new = ev.b
                if old > 0 and new > 0:
                    f = new / old
            elif ev.kind == "bonus":
                a = ev.a
                b = ev.b
                if a > 0 and b > 0:
                    f = b / (a + b)
            elif ev.kind == "rights":
                a = ev.a
                b = ev.b
                if a > 0 and b > 0:
                    r = a / b
                    sub_price = ev.premium if ev.premium_is_issue_price else (face_val + ev.premium)
                    p0 = float(prev_close.loc[eff]) if pd.notna(prev_close.loc[eff]) else float("nan")
                    if pd.notna(p0) and p0 > 0 and sub_price > 0:
                        terp = (p0 + r * sub_price) / (1.0 + r)
                        k = terp / p0
                        # Out-of-the-Money Invariant: rights issues NEVER inflate historical prices
                        f = min(float(k), 1.0)
            elif ev.kind in ("dividend", "distribution"):
                # Decoupled from primary series (Capital-Adjusted OHLC standard: splits, bonuses, rights, demergers)
                f = 1.0
            elif ev.kind == "reorg":
                p_prev_close = float(prev_close.loc[eff]) if pd.notna(prev_close.loc[eff]) else float("nan")
                p_ex_open = float(ref_open.loc[eff]) if pd.notna(ref_open.loc[eff]) else float("nan")
                p_ex_close = float(ref_close.loc[eff]) if (eff in ref_close.index and pd.notna(ref_close.loc[eff])) else float("nan")

                if pd.notna(p_prev_close) and pd.notna(p_ex_open) and p_prev_close > 0 and p_ex_open > 0:
                    # In historical pre-SPOS demergers (e.g. ADANIENT 2015-06-03), opening quotes were stale cum-prices
                    # and the post-demerger equilibrium was established during the day.
                    p_ex_price = p_ex_close if (pd.notna(p_ex_close) and p_ex_close > 0 and (p_ex_close / p_ex_open) < 0.60) else p_ex_open
                    obs = p_ex_price / p_prev_close
                    is_explicit = (ev.premium > 0.5)
                    if is_explicit:
                        # Explicit demerger / spin-off (e.g. RELIANCE, PEL, ARVIND)
                        f = obs if obs < 0.995 else 1.0
                    else:
                        # Scheme of arrangement / restructuring: require at least 5% carve-out (k <= 0.95)
                        f = obs if obs <= 0.95 else 1.0

            if f is None or not (0.0 < f < 5.0):
                continue

            if ev.kind == "reorg":
                # Deduplicate reorg factors per date: only one observed gap ratio per ex-date
                reorg_factor_by_date[eff] = min(reorg_factor_by_date.get(eff, 1.0), float(f))
            else:
                capital_factor_by_date[eff] = capital_factor_by_date.get(eff, 1.0) * float(f)

    # Combine capital actions and deduplicated reorg factor per date
    all_eff_dates = set(capital_factor_by_date.keys()).union(reorg_factor_by_date.keys())
    events_factor: dict[pd.Timestamp, float] = {}
    for d in all_eff_dates:
        rf = reorg_factor_by_date.get(d, 1.0)
        cf = capital_factor_by_date.get(d, 1.0)
        if rf < 0.999:
            # Reorg observed gap P_ex / P_prev already embeds any co-occurring bonus/split on the same day (e.g. AHLEAST)
            events_factor[d] = rf
        else:
            events_factor[d] = cf

    # Backward cumulative product (anchor at most recent date: factor=1.0)
    out = np.empty(len(dates), dtype="float64")
    cf = 1.0
    for i in range(len(dates) - 1, -1, -1):
        out[i] = cf
        d = dates[i]
        if d in events_factor:
            cf *= events_factor[d]

    return out.astype("float32")


def try_alpha_vantage_factor(symbol: str, dates: pd.DatetimeIndex) -> pd.Series | None:
    """
    Alpha Vantage demo endpoint is generally US-only; this is a best-effort no-auth try.
    Returns a factor series indexed by date if successful.
    """
    import requests

    # "demo" key is public but limited; likely won't support NSE tickers.
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": f"{symbol}.NS",
        "outputsize": "full",
        "apikey": "demo",
    }
    try:
        r = requests.get(ALPHAVANTAGE_URL, params=params, timeout=30)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        payload = r.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or "Time Series (Daily)" not in payload:
        return None
    ts = payload.get("Time Series (Daily)", {})
    if not isinstance(ts, dict) or not ts:
        return None
    rows = []
    for d, v in ts.items():
        try:
            rows.append(
                (
                    pd.Timestamp(d),
                    float(v.get("4. close", "nan")),
                    float(v.get("5. adjusted close", "nan")),
                )
            )
        except Exception:
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "close", "adj"]).dropna()
    if df.empty:
        return None
    df = df.sort_values("date")
    f = (df["adj"] / df["close"]).replace([np.inf, -np.inf], np.nan)
    out = pd.Series(f.to_numpy(dtype="float32"), index=pd.DatetimeIndex(df["date"]))
    out = out.reindex(dates, method="ffill", tolerance=pd.Timedelta(days=ALIGN_TOLERANCE_DAYS))
    if float(out.notna().mean()) < 0.5:
        return None
    return out


def try_fmp_factor(symbol: str, dates: pd.DatetimeIndex) -> pd.Series | None:
    """
    FMP is typically API-key gated; try without a key and skip if blocked.
    """
    import requests

    url = f"{FMP_URL}/historical-price-full/{symbol}.NS"
    try:
        r = requests.get(url, timeout=30)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    try:
        payload = r.json()
    except Exception:
        return None
    hist = payload.get("historical")
    if not isinstance(hist, list) or not hist:
        return None
    df = pd.DataFrame(hist)
    if df.empty or "date" not in df or "close" not in df:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.floor("D")
    df = df.dropna(subset=["date", "close"]).sort_values("date")
    # No adjusted close without key in most cases; can't build factor reliably.
    return None


def try_eodhd_factor(symbol: str, dates: pd.DatetimeIndex) -> pd.Series | None:
    """
    EODHD is token-gated; skip immediately.
    """
    return None


class SymbolAliasGraph:
    """
    Maintains a Disjoint-Set Union (Union-Find) graph of official symbol renames
    from NSE's symbolchange.csv and data/symbol_changes.json.
    Seamlessly links all historical and current tickers for a company.
    """

    def __init__(self, csv_path: Path = Path("symbolchange.csv")):
        self.csv_path = csv_path
        self.parent: dict[str, str] = {}
        self._load()

    def find(self, i: str) -> str:
        if i not in self.parent:
            self.parent[i] = i
            return i
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: str, j: str) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j

    def _load(self) -> None:
        import csv
        if self.csv_path.exists():
            try:
                with open(self.csv_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    count = 0
                    for row in reader:
                        if len(row) >= 3:
                            old_sym = row[1].strip()
                            new_sym = row[2].strip()
                            if old_sym and new_sym:
                                self.union(old_sym, new_sym)
                                count += 1
                logging.info("SymbolAliasGraph loaded %d mappings from %s", count, self.csv_path.name)
            except Exception as e:
                logging.error("Failed to parse symbolchange.csv: %s", e)

        # Merge unified symbol_changes.json curated overrides
        json_candidates = [
            Path(__file__).resolve().parent.parent / "Projects" / "nse-industry-insights" / "data" / "symbol_changes.json",
            Path("/Users/sumeetdas/Projects/nse-industry-insights/data/symbol_changes.json"),
        ]
        for jp in json_candidates:
            if jp.exists():
                try:
                    with open(jp, "r", encoding="utf-8") as f:
                        j_data = json.load(f)
                        for item in j_data:
                            old_s = item.get("old_symbol", "").strip()
                            new_s = item.get("new_symbol", "").strip()
                            if old_s and new_s:
                                self.union(old_s, new_s)
                    logging.info("SymbolAliasGraph augmented with %s", jp.name)
                    break
                except Exception as je:
                    logging.warning("Could not merge symbol_changes.json: %s", je)

    def get_aliases(self, symbol: str) -> set[str]:
        sym = symbol.strip()
        root = self.find(sym)
        aliases = {sym}
        for k in self.parent:
            if self.find(k) == root:
                aliases.add(k)
        return aliases


def extract_yf_series(df: pd.DataFrame, ticker: str, price_field: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        key = (price_field, ticker)
        if key not in df.columns:
            return None
        s = df[key]
    else:
        if price_field not in df.columns:
            return None
        s = df[price_field]
    s = s.dropna()
    if s.empty:
        return None
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s


def yf_download_batch(tickers: list[str], start: str, end: str) -> pd.DataFrame | None:
    global YF_RATE_LIMITED
    backoff = 3
    for attempt in range(1, YF_MAX_RETRIES + 1):
        try:
            df = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                threads=True,
                timeout=YF_TIMEOUT,
                progress=False,
            )
            if df is None:
                raise RuntimeError("yfinance returned None")
            # yfinance can silently return empty frames under throttling or symbol issues.
            if hasattr(df, "empty") and bool(df.empty):
                return df
            return df
        except Exception as exc:
            # If Yahoo is throttling, continuing to retry just burns time; fall back to NSE method.
            msg = str(exc)
            if type(exc).__name__ in ("YFRateLimitError", "YFRateLimitException") or "Rate limited" in msg or "Too Many Requests" in msg:
                logging.error("yfinance rate limited; disabling yfinance for this run: %s", exc)
                YF_RATE_LIMITED = True
                return None
            logging.warning("yfinance batch failed attempt %d/%d: %s", attempt, YF_MAX_RETRIES, exc)
            if attempt == YF_MAX_RETRIES:
                return None
            time.sleep(backoff + random.uniform(1.0, 4.0))
            backoff = min(backoff * 2, 120)
    return None


def main(target_symbol: str | None = None, force: bool = False) -> None:
    # Try to bump open file limit to avoid OSError: Too many open files
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        desired = min(4096, hard)
        if soft < desired:
            resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    except Exception as exc:
        pass

    setup_logging()

    if not ORIG_DIR.exists():
        raise SystemExit(f"Missing original dataset dir: {ORIG_DIR}")
    if OUT_DIR.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing output: {OUT_DIR}. Pass --force to overwrite.")

    meta = load_metadata()
    max_trade_date = meta.get("max_trade_date") or meta.get("end_date")
    if not max_trade_date:
        max_trade_date = dt.date.today().isoformat()
    max_trade_date_dt = dt.date.fromisoformat(str(max_trade_date))

    logging.info("Loading original Parquet metadata from %s", ORIG_DIR)
    dataset = ds.dataset(ORIG_DIR, format="parquet", partitioning="hive")
    logging.info("Original schema: %s", dataset.schema)

    # Read only the columns needed to compute factors (avoid loading source_url, etc.)
    # Read only the columns needed to compute factors (avoid loading source_url, etc.)
    logging.info("Reading base columns (symbol, series, trade_date, open, high, low, close, year)...")
    # Some partitions use different dictionary index widths for `series`; force string to avoid invalid casts.
    base_schema = pa.schema(
        [
            ("symbol", pa.string()),
            ("series", pa.string()),
            ("trade_date", pa.timestamp("ms")),
            ("open", pa.float32()),
            ("high", pa.float32()),
            ("low", pa.float32()),
            ("close", pa.float32()),
            ("year", pa.int32()),
        ]
    )
    partitioning = ds.partitioning(pa.schema([("year", pa.int32())]), flavor="hive")
    base_ds = ds.dataset(ORIG_DIR, format="parquet", partitioning=partitioning, schema=base_schema)
    base = base_ds.to_table(columns=["symbol", "series", "trade_date", "open", "high", "low", "close", "year"]).to_pandas()
    base["trade_date"] = pd.to_datetime(base["trade_date"]).dt.floor("D")
    for col in ["open", "high", "low", "close"]:
        base[col] = pd.to_numeric(base[col], errors="coerce").astype("float32")
    base["year"] = pd.to_numeric(base["year"], errors="coerce").fillna(0).astype("int32")
    base["open"] = pd.to_numeric(base["open"], errors="coerce").astype("float32")

    # Validate key uniqueness per partition-year (dataset can include previous-year dates in next-year partition)
    dup = int(base.duplicated(subset=["year", "symbol", "series", "trade_date"]).sum())
    if dup:
        logging.warning("Found %d duplicate (year,symbol,series,trade_date) rows in base data.", dup)

    # Ensure categoricals (for stable merging later)
    base["symbol"] = base["symbol"].astype("category")
    base["series"] = base["series"].astype("category")

    logging.info(
        "Loaded base: rows=%d unique_symbols=%d date_range=%s..%s",
        len(base),
        int(base["symbol"].nunique()),
        str(base["trade_date"].min().date()),
        str(base["trade_date"].max().date()),
    )

    # --- Optional: Filter for specific symbols (e.g. for testing) ---
    if target_symbol:
        target_symbol = target_symbol.upper()
        logging.info("Filtering dataset for symbol: %s", target_symbol)
        base = base[base["symbol"] == target_symbol]
        if base.empty:
            logging.error("Symbol %s not found in dataset", target_symbol)
            return
        base["symbol"] = base["symbol"].cat.remove_unused_categories()
        base["series"] = base["series"].cat.remove_unused_categories()
        logging.info("Filtered base: rows=%d unique_symbols=%d", len(base), int(base["symbol"].nunique()))

    logging.info("Sorting base by symbol/trade_date...")
    base = base.sort_values(["symbol", "trade_date", "series"], kind="mergesort").reset_index(drop=True)

    sym_cats = base["symbol"].cat.categories
    series_cats = base["series"].cat.categories

    # Build slice boundaries per symbol for fast access
    sym_codes = base["symbol"].cat.codes.to_numpy()
    change = np.r_[True, sym_codes[1:] != sym_codes[:-1]]
    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], len(sym_codes)]
    symbols = [sym_cats[c] for c in sym_codes[starts]]
    # starts/ends align with symbols in category order because we sorted by category order
    # But categories might include symbols not present after sorting? They all are present.

    ca_df = pd.DataFrame()
    if CA_CACHE_PATH.exists():
        try:
            ca_df = pd.read_parquet(CA_CACHE_PATH)
            logging.info("Loaded NSE corporate actions cache %s rows=%d", CA_CACHE_PATH, len(ca_df))
        except Exception as exc:
            logging.warning("Failed to read corporate actions cache %s: %s (will refetch)", CA_CACHE_PATH, exc)
            ca_df = pd.DataFrame()

    # Determine fetch ranges
    fetch_windows = []
    if ca_df.empty:
        logging.info("Cache empty/missing. Fetching complete NSE corporate actions history (2014..%s)", max_trade_date_dt)
        for year in range(2014, max_trade_date_dt.year + 1):
            fd = dt.date(year, 1, 1)
            td = dt.date(year, 12, 31) if year < max_trade_date_dt.year else max_trade_date_dt
            fetch_windows.append((fd, td))
    else:
        # Update cache with the last 45 days up to 15 days in the future to catch newly announced actions
        fd = dt.date.today() - dt.timedelta(days=45)
        td = dt.date.today() + dt.timedelta(days=15)
        logging.info("Cache found. Fetching latest NSE corporate actions update window (%s -> %s)", fd, td)
        fetch_windows.append((fd, td))

    ca_records: list[dict[str, Any]] = []
    for fd, td in fetch_windows:
        from_s = fd.strftime("%d-%m-%Y")
        to_s = td.strftime("%d-%m-%Y")
        logging.info("NSE CA fetch window %s -> %s", from_s, to_s)
        try:
            chunk = fetch_nse_corporate_actions(from_s, to_s)
            ca_records.extend(chunk)
            logging.info("  got %d actions", len(chunk))
        except Exception as exc:
            logging.error("Failed to fetch NSE corporate actions for %s..%s: %s", from_s, to_s, exc)
        if len(fetch_windows) > 1:
            time.sleep(random.uniform(1.0, 2.5))

    if ca_records:
        new_ca_df = pd.DataFrame(ca_records)
        if not new_ca_df.empty:
            if not ca_df.empty:
                ca_df = pd.concat([ca_df, new_ca_df], ignore_index=True)
            else:
                ca_df = new_ca_df
            
            try:
                dedupe_cols = [c for c in ["symbol", "exDate", "subject", "purpose"] if c in ca_df.columns]
                # Filter out any exact duplicate announcements
                if dedupe_cols:
                    ca_df = ca_df.drop_duplicates(subset=dedupe_cols, keep="last").reset_index(drop=True)
                
                # Convert object columns to string to prevent PyArrow type crashes
                for c in ca_df.columns:
                    if ca_df[c].dtype == 'object':
                        ca_df[c] = ca_df[c].astype(str)
                
                ca_df.to_parquet(
                    CA_CACHE_PATH,
                    engine="pyarrow",
                    compression="zstd",
                    compression_level=8,
                    index=False,
                )
                logging.info("Saved/Updated NSE corporate actions cache to %s (total rows=%d)", CA_CACHE_PATH, len(ca_df))
            except Exception as exc:
                logging.warning("Failed to save corporate actions cache %s: %s", CA_CACHE_PATH, exc)

    # Ingest curated corporate actions if present (e.g. data/corporate_actions.json)
    curated_paths = [
        Path(__file__).resolve().parent.parent / "Projects" / "nse-industry-insights" / "data" / "corporate_actions.json",
        Path("/Users/sumeetdas/Projects/nse-industry-insights/data/corporate_actions.json"),
    ]
    curated_records = []
    for cp in curated_paths:
        if cp.exists():
            try:
                with open(cp, "r", encoding="utf-8") as f:
                    curated_list = json.load(f)
                for item in curated_list:
                    c_sym = item.get("symbol", "").strip().upper()
                    ex_d = item.get("ex_date", "").strip()
                    if not c_sym or not ex_d:
                        continue
                    dt_obj = dt.datetime.strptime(ex_d, "%Y-%m-%d")
                    ex_str = dt_obj.strftime("%d-%b-%Y")
                    subj = item.get("description", "") or f"{item.get('action', '')} {item.get('ratio', '')}"
                    curated_records.append({
                        "symbol": c_sym,
                        "series": "SM" if item.get("action") in ("SPLIT", "BONUS") else "EQ",
                        "exDate": ex_str,
                        "subject": subj,
                        "faceVal": "10",
                        "comp": c_sym,
                    })
                logging.info("Loaded %d curated actions from %s", len(curated_records), cp.name)
            except Exception as ce:
                logging.warning("Could not merge curated corporate actions: %s", ce)
            break

    if curated_records:
        curated_df = pd.DataFrame(curated_records)
        if not ca_df.empty:
            ca_df = pd.concat([ca_df, curated_df], ignore_index=True)
        else:
            ca_df = curated_df

    alias_graph = SymbolAliasGraph()

    if not ca_df.empty:
        ca_df["symbol"] = ca_df["symbol"].astype(str).str.strip()
        ca_df["ex_date"] = pd.to_datetime(ca_df["exDate"], format="%d-%b-%Y", errors="coerce").dt.floor("D")
        ca_df = ca_df[ca_df["ex_date"].notna()]
        ca_df = ca_df[ca_df["ex_date"].dt.date <= max_trade_date_dt]

    ca_by_symbol: dict[str, list[dict[str, Any]]] = {}
    if not ca_df.empty:
        for sym, grp in ca_df.groupby("symbol", sort=False):
            ca_by_symbol[str(sym)] = grp.to_dict(orient="records")
    logging.info("Corporate actions loaded: total=%d symbols_with_actions=%d", len(ca_df), len(ca_by_symbol))

    # Results arrays
    num_rows = len(base)
    adjusted_open = np.full(num_rows, np.nan, dtype="float32")
    adjusted_high = np.full(num_rows, np.nan, dtype="float32")
    adjusted_low = np.full(num_rows, np.nan, dtype="float32")
    adjusted_close = np.full(num_rows, np.nan, dtype="float32")

    sym_stats: dict[str, dict[str, Any]] = {}

    def symbol_slice(sym_idx: int) -> tuple[int, int]:
        return int(starts[sym_idx]), int(ends[sym_idx])

    symbol_to_idx = {str(sym).strip(): j for j, sym in enumerate(symbols)}
    # Group symbols by connected alias root in Union-Find graph
    alias_groups: dict[str, list[str]] = {}
    for s in symbols:
        s_str = str(s).strip()
        root = alias_graph.find(s_str)
        alias_groups.setdefault(root, []).append(s_str)

    logging.info("Computing capital-adjusted price factors for %d symbols across %d alias groups...", len(symbols), len(alias_groups))
    for root, group_syms in alias_groups.items():
        # Combine rows for all symbols in the group to build continuous lifecycle reference prices
        if len(group_syms) == 1:
            sym_str = group_syms[0]
            j = symbol_to_idx[sym_str]
            start_i, end_i = symbol_slice(j)
            sym_rows = base.iloc[start_i:end_i]
            
            if not sym_rows["trade_date"].duplicated().any():
                ref_dates = pd.DatetimeIndex(sym_rows["trade_date"].to_numpy(), name="trade_date")
                ref_close = pd.Series(sym_rows["close"].to_numpy(dtype="float32"), index=ref_dates)
                ref_open = pd.Series(sym_rows["open"].to_numpy(dtype="float32"), index=ref_dates)
            else:
                prio = (sym_rows["series"] != "EQ").astype("int8")
                ref_rows = sym_rows.assign(_prio=prio).sort_values(["trade_date", "_prio"], kind="mergesort")
                ref_rows = ref_rows.drop_duplicates(subset=["trade_date"], keep="first")
                ref_dates = pd.DatetimeIndex(ref_rows["trade_date"].to_numpy(), name="trade_date")
                ref_close = pd.Series(ref_rows["close"].to_numpy(dtype="float32"), index=ref_dates)
                ref_open = pd.Series(ref_rows["open"].to_numpy(dtype="float32"), index=ref_dates)
        else:
            slices = []
            for s in group_syms:
                s_i, e_i = symbol_slice(symbol_to_idx[s])
                slices.append(base.iloc[s_i:e_i])
            combined_rows = pd.concat(slices, ignore_index=True)
            prio = (combined_rows["series"] != "EQ").astype("int8")
            ref_rows = combined_rows.assign(_prio=prio).sort_values(["trade_date", "_prio"], kind="mergesort")
            ref_rows = ref_rows.drop_duplicates(subset=["trade_date"], keep="first")
            ref_dates = pd.DatetimeIndex(ref_rows["trade_date"].to_numpy(), name="trade_date")
            ref_close = pd.Series(ref_rows["close"].to_numpy(dtype="float32"), index=ref_dates)
            ref_open = pd.Series(ref_rows["open"].to_numpy(dtype="float32"), index=ref_dates)

        # Collect actions across all aliases in the Disjoint-Set Union graph
        aliases = alias_graph.get_aliases(root)
        group_actions = []
        seen_actions = set()
        for a_sym in aliases:
            for act in ca_by_symbol.get(a_sym, []):
                act_key = (str(act.get("exDate", "")), str(act.get("subject", "")))
                if act_key not in seen_actions:
                    seen_actions.add(act_key)
                    group_actions.append(act)

        try:
            factor_series = build_manual_factor_series(
                symbol=root,
                dates=ref_dates,
                ref_close=ref_close,
                ref_open=ref_open,
                actions=group_actions,
            )
            m_dev = max_abs_dev_from_one(factor_series)
            used = "nse_corporate_actions" if m_dev >= 1e-6 else "raw_no_adjustment"
        except Exception as exc:
            logging.error("Failed calculating manual factor for alias group %s: %s", root, exc)
            factor_series = np.ones(len(ref_dates), dtype="float32")
            used = "raw_no_adjustment"

        for sym_str in group_syms:
            j = symbol_to_idx[sym_str]
            start_i, end_i = symbol_slice(j)
            sym_rows = base.iloc[start_i:end_i]
            
            pos = ref_dates.get_indexer(sym_rows["trade_date"].to_numpy())
            row_factor = factor_series[pos].astype("float32", copy=False)

            adjusted_open[start_i:end_i] = (sym_rows["open"].to_numpy(dtype="float32") * row_factor).astype("float32")
            adjusted_high[start_i:end_i] = (sym_rows["high"].to_numpy(dtype="float32") * row_factor).astype("float32")
            adjusted_low[start_i:end_i] = (sym_rows["low"].to_numpy(dtype="float32") * row_factor).astype("float32")
            adjusted_close[start_i:end_i] = (sym_rows["close"].to_numpy(dtype="float32") * row_factor).astype("float32")

            sym_stats[sym_str] = {
                "method": used,
                "yfinance_coverage": 1.0,
                "filled_from_manual_count": 0,
                "rows": int(end_i - start_i),
                "unique_dates": int(len(sym_rows["trade_date"].unique())),
                "reason": "",
            }

    base_out = base[["symbol", "series", "trade_date", "year"]].copy()
    base_out["adj_open"] = adjusted_open.astype("float32", copy=False)
    base_out["adj_high"] = adjusted_high.astype("float32", copy=False)
    base_out["adj_low"] = adjusted_low.astype("float32", copy=False)
    base_out["adj_close"] = adjusted_close.astype("float32", copy=False)

    # Sanity: no missing adjusted prices
    missing = int(pd.isna(base_out["adj_close"]).sum())
    if missing:
        logging.warning("adjusted_close has %d missing values. Filling with raw OHLC.", missing)
        for col, raw in [("adj_open", "open"), ("adj_high", "high"), ("adj_low", "low"), ("adj_close", "close")]:
            base_out.loc[pd.isna(base_out[col]), col] = base.loc[pd.isna(base_out[col]), raw].to_numpy(dtype="float32")

    # Release memory mapped file handles from PyArrow and sockets from yfinance
    import gc
    try:
        del dataset
        del base_ds
    except Exception:
        pass
    gc.collect()

    # Write output dataset, year-partitioned, to staging dir first for atomic safety
    staging_dir = OUT_DIR.parent / f"{OUT_DIR.name}.staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Writing NEW dataset to staging dir %s (partitioned by year)...", staging_dir)

    years = sorted(base_out["year"].dropna().astype(int).unique().tolist())
    for y in years:
        in_file = ORIG_DIR / f"year={y}" / f"part-{y}.parquet"
        if not in_file.exists():
            # fallback to scanning by filter if file naming differs
            logging.warning("Missing expected input year file %s; skipping year %s", in_file, y)
            continue

        logging.info("  year=%s reading original partition...", y)
        year_df = pd.read_parquet(in_file)
        # Filter by target_symbol if set (for single-symbol testing)
        if target_symbol:
            year_df = year_df[year_df["symbol"] == target_symbol]
            
        # Deduplicate the partition just in case (e.g. daily updater appended duplicate or unmapped NaN symbols)
        year_df.drop_duplicates(["symbol", "series", "trade_date"], keep="last", inplace=True)
            
        # Align categoricals to global categories so merge preserves category dtype
        year_df["symbol"] = year_df["symbol"].astype("category").cat.set_categories(sym_cats)
        year_df["series"] = year_df["series"].astype("category").cat.set_categories(series_cats)
        year_df["trade_date"] = pd.to_datetime(year_df["trade_date"]).dt.floor("D")

        map_year = base_out[base_out["year"] == y][["symbol", "series", "trade_date", "adj_open", "adj_high", "adj_low", "adj_close"]].copy()
        map_year["symbol"] = map_year["symbol"].cat.set_categories(sym_cats)
        map_year["series"] = map_year["series"].cat.set_categories(series_cats)

        # Deduplicate the right side again just in case there are multiple NaNs mapping to the same date
        map_year.drop_duplicates(["symbol", "series", "trade_date"], keep="last", inplace=True)

        merged = year_df.merge(
            map_year,
            on=["symbol", "series", "trade_date"],
            how="left",
            validate="one_to_one",
        )
        if merged["adj_close"].isna().any():
            # Shouldn't happen; fill with raw OHLC just in case
            n = int(merged["adj_close"].isna().sum())
            logging.warning("  year=%s missing adjusted prices rows=%d; filling with raw.", y, n)
            for c_adj, c_raw in [("adj_open", "open"), ("adj_high", "high"), ("adj_low", "low"), ("adj_close", "close")]:
                merged[c_adj] = merged[c_adj].fillna(merged[c_raw]).astype("float32")
        else:
            for col in ["adj_open", "adj_high", "adj_low", "adj_close"]:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("float32")

        out_year_dir = staging_dir / f"year={y}"
        out_year_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_year_dir / f"part-{y}.parquet"
        # Drop 'year' column before writing so it doesn't conflict with partition metadata
        merged.drop(columns=["year"], errors="ignore").to_parquet(
            out_file, engine="pyarrow", compression="zstd", compression_level=8, index=False
        )

    # Atomic swap into OUT_DIR
    if OUT_DIR.exists():
        logging.info("Replacing existing %s with newly built dataset...", OUT_DIR)
        backup_dir = OUT_DIR.parent / f"{OUT_DIR.name}.old"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        OUT_DIR.rename(backup_dir)
        staging_dir.rename(OUT_DIR)
        shutil.rmtree(backup_dir)
    else:
        staging_dir.rename(OUT_DIR)
    logging.info("Successfully installed clean dataset at %s", OUT_DIR)

    # ----------------------------
    # Validation + report
    # ----------------------------
    methods = pd.Series({k: v.get("method", "raw") for k, v in sym_stats.items()}, name="method")
    method_counts = methods.value_counts().to_dict()

    # Row-weighted coverage by method
    rows_by_method: dict[str, int] = {}
    for sym, st in sym_stats.items():
        m = st.get("method", "raw")
        rows_by_method[m] = rows_by_method.get(m, 0) + int(st.get("rows", 0))
    total_rows = int(sum(rows_by_method.values())) or int(len(base_out))
    row_pct = {k: round(v / total_rows * 100.0, 4) for k, v in rows_by_method.items()}

    # Per-year coverage breakdown by method (row-weighted)
    method_by_symbol: list[str] = [sym_stats.get(str(s), {}).get("method", "raw") for s in symbols]
    # Map symbol codes to method strings without doing a slow per-row map
    method_for_row = np.array(method_by_symbol, dtype=object)[base["symbol"].cat.codes.to_numpy()]
    per_year_method: dict[int, dict[str, float]] = {}
    for y in years:
        mask = base["year"].to_numpy() == y
        if not mask.any():
            continue
        vals, counts = np.unique(method_for_row[mask], return_counts=True)
        denom = int(counts.sum())
        per_year_method[int(y)] = {str(v): round(int(c) / denom * 100.0, 4) for v, c in zip(vals, counts)}

    # Overall value-change metric (adj_close != close)
    ratio_all = (base_out["adj_close"].to_numpy(dtype="float32") / base["close"].to_numpy(dtype="float32")).astype(
        "float64", copy=False
    )
    ratio_all = ratio_all[np.isfinite(ratio_all)]
    value_changed_pct = float(np.mean(np.abs(ratio_all - 1.0) > 1e-6) * 100.0) if ratio_all.size else 0.0

    # Recent match check (last ~2 years)
    recent_start = pd.Timestamp(max_trade_date_dt - dt.timedelta(days=730))
    recent_mask = base["trade_date"] >= recent_start
    recent_ratio = (base_out.loc[recent_mask, "adj_close"].to_numpy() / base.loc[recent_mask, "close"].to_numpy())
    recent_ratio = recent_ratio[np.isfinite(recent_ratio)]
    recent_abs_dev = float(np.nanmean(np.abs(recent_ratio - 1.0))) if recent_ratio.size else float("nan")

    # Statistical sanity per year (mean/median abs deviation)
    abs_dev = np.abs(
        base_out["adj_close"].to_numpy(dtype="float32") / base["close"].to_numpy(dtype="float32") - 1.0
    )
    abs_dev = np.where(np.isfinite(abs_dev), abs_dev, np.nan)
    dev_df = pd.DataFrame({"year": base["year"].to_numpy(dtype="int32"), "abs_dev": abs_dev})
    per_year_stats = (
        dev_df.groupby("year", sort=True)["abs_dev"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .assign(mean=lambda d: d["mean"].round(6), median=lambda d: d["median"].round(6))
    )
    per_year_dev = {int(r["year"]): {"mean_abs_dev": float(r["mean"]), "median_abs_dev": float(r["median"]), "rows": int(r["count"])} for _, r in per_year_stats.iterrows()}

    # Outliers in recent period (>10% deviation)
    recent_abs_dev_arr = np.abs(
        base_out.loc[recent_mask, "adj_close"].to_numpy(dtype="float32")
        / base.loc[recent_mask, "close"].to_numpy(dtype="float32")
        - 1.0
    )
    recent_outlier_pct = float(np.nanmean(recent_abs_dev_arr > 0.10) * 100.0) if recent_abs_dev_arr.size else 0.0

    # Known corporate action tests (8-10 examples). We will attempt to validate one event per symbol if present.
    def get_ref_factor_series(sym: str) -> pd.Series:
        try:
            sym_idx = symbols.index(sym)
        except ValueError:
            return pd.Series(dtype="float32")
        s0, s1 = symbol_slice(sym_idx)
        rows = pd.DataFrame(
            {
                "trade_date": base.loc[s0:s1 - 1, "trade_date"].to_numpy(),
                "series": base.loc[s0:s1 - 1, "series"].astype(str).to_numpy(),
                "close": base.loc[s0:s1 - 1, "close"].to_numpy(dtype="float32"),
                "adj": base_out.loc[s0:s1 - 1, "adj_close"].to_numpy(dtype="float32"),
            }
        )
        # Prefer EQ where possible
        rows["_prio"] = (rows["series"] != "EQ").astype(int)
        rows = rows.sort_values(["trade_date", "_prio"], kind="mergesort").drop_duplicates("trade_date", keep="first")
        f = (rows["adj"] / rows["close"]).replace([np.inf, -np.inf], np.nan)
        return pd.Series(f.to_numpy(dtype="float32"), index=pd.DatetimeIndex(rows["trade_date"]))

    test_symbols = [
        "RELIANCE",
        "INFY",
        "TCS",
        "BAJFINANCE",
        "VBL",
        "HDFCBANK",
        "WIPRO",
        "SBIN",
        "ICICIBANK",
        "MARUTI",
    ]
    action_test_results: list[dict[str, Any]] = []
    for sym in test_symbols:
        actions = ca_by_symbol.get(sym, [])
        # Only split/bonus/rights/demerger actions (capital actions)
        picked: dict[str, Any] | None = None
        for rec in actions:
            subj = str(rec.get("subject", "") or "").lower()
            if any(k in subj for k in ("split", "bonus", "rights", "demerger", "spin-off")):
                picked = rec
                break
        if picked is None:
            action_test_results.append({"symbol": sym, "status": "skipped", "reason": "no_capital_actions_found"})
            continue

        subj = str(picked.get("subject", "") or "")
        try:
            ex_d = parse_nse_date(str(picked.get("exDate", "")).strip())
        except Exception:
            action_test_results.append({"symbol": sym, "status": "skipped", "reason": "bad_exDate"})
            continue
        face_val = 0.0
        try:
            face_val = float(str(picked.get("faceVal", "") or "0").strip() or 0.0)
        except Exception:
            face_val = 0.0

        fs = get_ref_factor_series(sym)
        if fs.empty:
            action_test_results.append({"symbol": sym, "status": "skipped", "reason": "symbol_not_in_dataset"})
            continue

        eff = align_effective_date(ex_d, fs.index)
        if eff is None:
            action_test_results.append({"symbol": sym, "status": "skipped", "reason": "exDate_not_in_prices"})
            continue

        # prev trading date
        pos = fs.index.searchsorted(eff)
        if pos <= 0:
            action_test_results.append({"symbol": sym, "status": "skipped", "reason": "no_prev_trading_day"})
            continue
        prev_d = fs.index[pos - 1]
        actual = float(fs.loc[prev_d] / fs.loc[eff]) if pd.notna(fs.loc[prev_d]) and pd.notna(fs.loc[eff]) else float("nan")

        # expected factor product from parsed events
        expected = 1.0
        ref_close_series = None
        try:
            sym_idx = symbols.index(sym)
            s0, s1 = symbol_slice(sym_idx)
            # Build reference close per date for prev close lookup
            tmp = pd.DataFrame(
                {
                    "trade_date": base.loc[s0:s1 - 1, "trade_date"].to_numpy(),
                    "series": base.loc[s0:s1 - 1, "series"].astype(str).to_numpy(),
                    "close": base.loc[s0:s1 - 1, "close"].to_numpy(dtype="float32"),
                }
            )
            tmp["_prio"] = (tmp["series"] != "EQ").astype(int)
            tmp = tmp.sort_values(["trade_date", "_prio"], kind="mergesort").drop_duplicates("trade_date", keep="first")
            ref_close_series = pd.Series(tmp["close"].to_numpy(dtype="float32"), index=pd.DatetimeIndex(tmp["trade_date"]))
        except Exception:
            ref_close_series = None

        events = parse_subject(subj, face_val)
        for ev in events:
            if ev.kind == "split":
                old, new = ev.a, ev.b
                expected *= (new / old) if old and new else 1.0
            elif ev.kind == "bonus":
                a, b = ev.a, ev.b
                expected *= (b / (a + b)) if a and b else 1.0
            elif ev.kind == "rights" and ref_close_series is not None:
                a, b = ev.a, ev.b
                if a and b:
                    r = a / b
                    sub_price = ev.premium if ev.premium_is_issue_price else (face_val + ev.premium)
                    p0 = float(ref_close_series.shift(1).get(eff, np.nan))
                    if np.isfinite(p0) and p0 > 0 and sub_price > 0:
                        terp = (p0 + r * sub_price) / (1.0 + r)
                        expected *= min(float(terp / p0), 1.0)
            elif ev.kind in ("dividend", "distribution"):
                # Decoupled: cash dividends do not affect primary capital-adjusted series
                pass

        # Evaluate
        if not np.isfinite(actual) or not np.isfinite(expected) or expected <= 0:
            action_test_results.append(
                {"symbol": sym, "status": "failed", "exDate": str(ex_d), "subject": subj, "reason": "nan_factor"}
            )
            continue
        rel_err = abs(actual - expected) / expected if expected else abs(actual - expected)
        status = "passed" if rel_err <= 0.03 else "failed"
        action_test_results.append(
            {
                "symbol": sym,
                "status": status,
                "exDate": str(ex_d),
                "subject": subj,
                "expected_factor": round(expected, 6),
                "actual_factor": round(actual, 6),
                "rel_error": round(rel_err, 6),
            }
        )

    # SMA discontinuity check for a small sample of symbols with split/bonus/rights actions
    sma_checks: list[dict[str, Any]] = []
    sample_syms = [s for s in symbols if s in ca_by_symbol]
    random.shuffle(sample_syms)
    sample_syms = sample_syms[:10]
    for sym in sample_syms:
        # find a split/bonus/rights action
        acts = ca_by_symbol.get(sym, [])
        act = None
        for rec in acts:
            subj = str(rec.get("subject", "") or "").lower()
            if any(k in subj for k in ("split", "bonus", "rights")):
                act = rec
                break
        if act is None:
            continue
        fs = get_ref_factor_series(sym)
        if fs.empty:
            continue
        # Build raw/adj close series
        sym_idx = symbols.index(sym)
        s0, s1 = symbol_slice(sym_idx)
        tmp = pd.DataFrame(
            {
                "trade_date": base.loc[s0:s1 - 1, "trade_date"].to_numpy(),
                "series": base.loc[s0:s1 - 1, "series"].astype(str).to_numpy(),
                "close": base.loc[s0:s1 - 1, "close"].to_numpy(dtype="float32"),
                "adj": base_out.loc[s0:s1 - 1, "adj_close"].to_numpy(dtype="float32"),
            }
        )
        tmp["_prio"] = (tmp["series"] != "EQ").astype(int)
        tmp = tmp.sort_values(["trade_date", "_prio"], kind="mergesort").drop_duplicates("trade_date", keep="first")
        tmp = tmp.set_index(pd.DatetimeIndex(tmp["trade_date"]))
        raw = tmp["close"].astype("float64")
        adj = tmp["adj"].astype("float64")
        raw_sma = raw.rolling(200).mean()
        adj_sma = adj.rolling(200).mean()
        try:
            ex_d = parse_nse_date(str(act.get("exDate", "")).strip())
        except Exception:
            continue
        eff = align_effective_date(ex_d, raw_sma.index)
        if eff is None or eff not in raw_sma.index:
            continue
        pos = raw_sma.index.searchsorted(eff)
        if pos <= 0:
            continue
        # require SMA values present
        if not np.isfinite(raw_sma.iloc[pos]) or not np.isfinite(raw_sma.iloc[pos - 1]):
            continue
        if not np.isfinite(adj_sma.iloc[pos]) or not np.isfinite(adj_sma.iloc[pos - 1]):
            continue
        raw_jump = float(raw_sma.iloc[pos] / raw_sma.iloc[pos - 1])
        adj_jump = float(adj_sma.iloc[pos] / adj_sma.iloc[pos - 1])
        sma_checks.append(
            {
                "symbol": sym,
                "exDate": str(ex_d),
                "subject": str(act.get("subject", "")),
                "raw_sma_jump": round(raw_jump, 6),
                "adj_sma_jump": round(adj_jump, 6),
                "adjusted_smoother": bool(abs(adj_jump - 1.0) < abs(raw_jump - 1.0)),
            }
        )

    report = {
        "original_dataset_dir": str(ORIG_DIR.resolve()),
        "output_dataset_dir": str(OUT_DIR.resolve()),
        "rows": int(len(base_out)),
        "unique_symbols": int(base_out["symbol"].nunique()),
        "min_trade_date": str(base_out["trade_date"].min().date()),
        "max_trade_date": str(base_out["trade_date"].max().date()),
        "symbol_method_counts": method_counts,
        "row_coverage_percent_by_method": row_pct,
        "per_year_row_coverage_percent_by_method": per_year_method,
        "value_changed_rows_percent": round(value_changed_pct, 4),
        "per_year_abs_dev_stats": per_year_dev,
        "recent_2y_outlier_rows_pct_gt_10pct": round(recent_outlier_pct, 6),
        "recent_2y_mean_abs_adjustment_deviation": round(recent_abs_dev, 6) if np.isfinite(recent_abs_dev) else None,
        "known_action_tests": action_test_results,
        "sma_discontinuity_checks": sma_checks,
        "generated_at": dt.datetime.now().isoformat(),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logging.info("Build complete. Report written to %s", REPORT_PATH)
    logging.info("Row coverage by method (%%): %s", row_pct)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build Adjusted NSE Master Parquet")
    parser.add_argument("--orig-dir", type=str, help="Path to input Parquet dataset")
    parser.add_argument("--out-dir", type=str, help="Path to output Parquet dataset")
    parser.add_argument("--report-path", type=str, help="Path to JSON report file")
    parser.add_argument("--symbol", type=str, help="Filter for single symbol (debugging)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    args = parser.parse_args()

    if args.orig_dir: ORIG_DIR = Path(args.orig_dir)
    if args.out_dir: OUT_DIR = Path(args.out_dir)
    if args.report_path: REPORT_PATH = Path(args.report_path)

    # Avoid inheriting pandas thread settings that can explode CPU.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main(target_symbol=args.symbol, force=args.force)
