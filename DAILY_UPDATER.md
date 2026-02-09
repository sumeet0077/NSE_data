# NSE Daily Updater

## What is running

- Service label: `com.nsedata.daily-updater`
- Scheduler: daily at `18:00`, retry `19:00`, retry `20:00` (Asia/Kolkata)
- Script: `~/Antigravity_NSE_Data/run_daily_updater.sh`
- Workspace: `/Users/sumeetdas/Antigravity_NSE_Data`

## Commands

### Status

```bash
launchctl print gui/$(id -u)/com.nsedata.daily-updater | sed -n '1,80p'
```

### View logs

```bash
tail -f "/Users/sumeetdas/Antigravity_NSE_Data/daily_updater_stdout.log"
tail -f "/Users/sumeetdas/Antigravity_NSE_Data/daily_updater_stderr.log"
```

### Force one immediate update (today)

```bash
cd "/Users/sumeetdas/Antigravity_NSE_Data"
.venv/bin/python nse_daily_update_service.py --run-once
```

### Force one immediate update (specific date)

```bash
cd "/Users/sumeetdas/Antigravity_NSE_Data"
.venv/bin/python nse_daily_update_service.py --run-once --date 2026-02-06
```

### Restart service

```bash
launchctl kickstart -k gui/$(id -u)/com.nsedata.daily-updater
```

### Stop service
Note: The plist is now located in `~/Library/LaunchAgents/` and points to `~/Antigravity_NSE_Data/run_daily_updater.sh`.

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.nsedata.daily-updater.plist
```

### Start service

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.nsedata.daily-updater.plist
launchctl enable gui/$(id -u)/com.nsedata.daily-updater
launchctl kickstart -k gui/$(id -u)/com.nsedata.daily-updater
```
