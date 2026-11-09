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
    p.add_argument("--minute", type=int, default=0, help="Slot minute to record/attempt (0-59, default 0)")
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

    if args.minute < 0 or args.minute > 59:
        logging.error("Invalid --minute=%s", args.minute)
        return 2

    tz = ZoneInfo(args.timezone)
    if args.date:
        trade_date = dt.date.fromisoformat(args.date)
    else:
        trade_date = dt.datetime.now(tz).date()

    slot = (args.hour, args.minute)
    slot_id = svc.get_slot_id(args.hour, args.minute)
    require_delivery = not args.allow_missing_delivery
    state = svc.load_state()

    session = requests.Session()
    session.headers.update({"Accept-Language": "en-US,en;q=0.9"})
    try:
        session.get("https://www.nseindia.com/", headers=svc.random_headers(), timeout=30)
    except Exception:
        pass

    # Multi-day catch-up: automatically detect and backfill any missing weekend/holiday sessions (e.g. Muhurat trading)
    catchup_performed = svc.run_multi_day_catchup(
        session,
        require_delivery=require_delivery,
        timezone=args.timezone,
        up_to_date=trade_date,
    )
    if catchup_performed:
        logging.info("Multi-day catch-up performed updates for previous missing sessions.")

    if not svc.should_attempt_slot(state, trade_date, slot):
        logging.info("Skipping slot=%s date=%s: already succeeded/attempted", slot_id, trade_date)
        if catchup_performed:
            logging.info("Triggering post-process due to multi-day catch-up...")
            svc.run_post_process()
        return 0

    # Strategy: Relax delivery requirement for early slots (before 18:00 IST), matching nse_daily_update_service
    effective_require_delivery = require_delivery if args.hour >= 18 else False
    if not effective_require_delivery and require_delivery:
        logging.info("Relaxing delivery requirement for early slot %s", slot_id)

    logging.info("Running timer slot attempt date=%s slot=%s require_delivery=%s", trade_date, slot_id, effective_require_delivery)
    ok, msg = svc.update_for_date(session, trade_date, require_delivery=effective_require_delivery)
    svc.record_slot_attempt(state, trade_date, slot, ok, msg)
    if ok:
        logging.info("SLOT SUCCESS date=%s slot=%s %s", trade_date, slot_id, msg)
        svc.run_post_process()
        return 0

    logging.warning("SLOT FAILED date=%s slot=%s %s", trade_date, slot_id, msg)
    if catchup_performed:
        logging.info("Triggering post-process due to multi-day catch-up despite current slot failure...")
        svc.run_post_process()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

