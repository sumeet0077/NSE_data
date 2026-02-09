#!/usr/bin/env python3
"""
Timer-friendly wrapper for nse_daily_update_service.py.

Why this exists:
- systemd timers should run a short-lived process that exits.
- We still want 18:00/19:00/20:00 IST retry behavior AND "skip if already succeeded".

This wrapper:
- checks daily_update_state.json to see if today's update already succeeded,
- runs one update attempt for the given slot hour if needed,
- records the attempt outcome back into the state file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from zoneinfo import ZoneInfo

import requests

import nse_daily_update_service as svc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NSE update slot runner (for systemd timers)")
    p.add_argument("--hour", type=int, required=True, help="Slot hour to record/attempt (0-23)")
    p.add_argument("--timezone", type=str, default="Asia/Kolkata", help="Timezone for 'today'")
    p.add_argument(
        "--allow-missing-delivery",
        action="store_true",
        help="Allow update even if delivery fields are missing",
    )
    p.add_argument("--date", type=str, default="", help="Override trade date YYYY-MM-DD (optional)")
    return p.parse_args()


def main() -> int:
    svc.setup_logging()
    args = parse_args()

    if args.hour < 0 or args.hour > 23:
        logging.error("Invalid --hour=%s", args.hour)
        return 2

    tz = ZoneInfo(args.timezone)
    if args.date:
        trade_date = dt.date.fromisoformat(args.date)
    else:
        trade_date = dt.datetime.now(tz).date()

    require_delivery = not args.allow_missing_delivery
    state = svc.load_state()
    if not svc.should_attempt_slot(state, trade_date, args.hour):
        logging.info("Skipping slot=%02d:00 date=%s: already succeeded/attempted", args.hour, trade_date)
        return 0

    session = requests.Session()
    session.headers.update({"Accept-Language": "en-US,en;q=0.9"})
    try:
        session.get("https://www.nseindia.com/", headers=svc.random_headers(), timeout=30)
    except Exception:
        pass

    logging.info("Running timer slot attempt date=%s slot=%02d:00 require_delivery=%s", trade_date, args.hour, require_delivery)
    ok, msg = svc.update_for_date(session, trade_date, require_delivery=require_delivery)
    svc.record_slot_attempt(state, trade_date, args.hour, ok, msg)
    if ok:
        logging.info("SLOT SUCCESS date=%s slot=%02d:00 %s", trade_date, args.hour, msg)
        return 0
    logging.warning("SLOT FAILED date=%s slot=%02d:00 %s", trade_date, args.hour, msg)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

