# Fetching TLEs for D2C satellite visibility

The **🛰️ D2C visibility** tab needs Starlink Direct-to-Cell orbital data (TLEs)
in the `TLE/` folder. This guide covers the two sources you need and how to
automate them.

- **CelesTrak** → *today's* elements. Automatic, one click or one command. Use
  it for **new/recent** measurements (e.g. the PRISM scans, Sep 1–2).
- **Space-Track** → *historical* elements. One command, needs a free login. Use
  it for the **campaign back-data** (the QP logs, Jul 22 – Aug 30), because SGP4
  is only accurate near a TLE's epoch and CelesTrak does not keep old elements.

Everything below assumes you start from the project root:

```bash
cd /Users/yuxichen/Documents/GitHub/Starlink_D2C
```

and that `./.venv/bin/python` is your virtual-env Python (adjust if different).

> **zsh tip (default macOS shell):** paste the **commands only**, not the `#` explanations around them. Interactive zsh doesn't treat `#` as a comment and will try to expand `[DTC]` as a filename pattern, printing `zsh: no matches found: [DTC]`. That message is harmless — the command above it still ran.

---

## 0. One-time setup

Install/update dependencies (this also installs `sgp4` and `certifi`, which the
fetch needs):

```bash
./.venv/bin/pip install -r requirements.txt
```

**macOS SSL note.** If a fetch fails with
`CERTIFICATE_VERIFY_FAILED … unable to get local issuer certificate`, your
python.org build has no CA bundle. The `certifi` install above fixes it. If you
still hit it, run Python's certificate installer once:

```bash
open "/Applications/Python 3.12/Install Certificates.command"
```

Restart the Streamlit app after installing dependencies (a new package is only
picked up on restart).

---

## 1. Current TLEs (CelesTrak) — automatic

**From the app:** open the sidebar → *🛰️ Satellite visibility (D2C)* → click
**“Fetch today's Starlink TLEs”**. It saves `TLE/celestrak_YYYY-MM-DD.tle`.

**From the terminal:**

```bash
./.venv/bin/python -m qpviz.satellites fetch TLE
./.venv/bin/python -m qpviz.satellites summary TLE
```

It fetches at most once per day (skips if today's file already exists; add
`--force` to override). This is all you need for measurements taken in the last
day or two.

---

## 2. Historical TLEs (Space-Track) — for the campaign dates

CelesTrak cannot give you July/August elements. Space-Track can. Create a free
account at <https://www.space-track.org>, then run **one command** — it learns
which Starlink are D2C from CelesTrak's `[DTC]` tags, pulls just those
satellites' history for your date range, and writes them back tagged `[DTC]`:

```bash
export SPACETRACK_USER="you@example.com"
export SPACETRACK_PASS="your-password"
./.venv/bin/python -m qpviz.satellites archive \
    --start 2026-07-22 --end 2026-08-30 --tle-dir TLE
```

- Dates are your **QP data window** (`YYYY-MM-DD`). The range is padded one day
  each side by default (`--pad-days N` to change).
- Output: `TLE/spacetrack_2026-07-21_2026-08-31.tle`.
- Queries run in polite, rate-limited chunks — a large range may take a minute.

You can keep the password out of your shell history by leaving `SPACETRACK_PASS`
unset; the command prompts for it without echoing.

---

## 3. Which source for which data

| Your data | Dates | Source | Command |
|---|---|---|---|
| PRISM scans | 2026-09-01 → 02 | CelesTrak (today) | `fetch TLE` |
| QP logs | 2026-07-22 → 08-30 | Space-Track history | `archive --start … --end …` |
| Anything from now on | recent | CelesTrak (daily) | `fetch TLE` |

Drop as many dated files in `TLE/` as you like — the app buckets each sample by
date and propagates it from the **nearest-epoch** TLE, so overlapping snapshots
only help.

---

## 4. Automate the daily fetch

**cron (simple):**

```cron
# 06:15 every day: refresh current Starlink TLEs for the explorer
15 6 * * *  cd /Users/yuxichen/Documents/GitHub/Starlink_D2C && ./.venv/bin/python -m qpviz.satellites fetch TLE >> TLE/fetch.log 2>&1
```

Edit with `crontab -e`.

**launchd (macOS-native, survives reboots):** save as
`~/Library/LaunchAgents/com.starlink.tlefetch.plist`, then
`launchctl load ~/Library/LaunchAgents/com.starlink.tlefetch.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.starlink.tlefetch</string>
  <key>WorkingDirectory</key><string>/Users/yuxichen/Documents/GitHub/Starlink_D2C</string>
  <key>ProgramArguments</key>
  <array>
    <string>/Users/yuxichen/Documents/GitHub/Starlink_D2C/.venv/bin/python</string>
    <string>-m</string><string>qpviz.satellites</string>
    <string>fetch</string><string>TLE</string>
  </array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>15</integer></dict>
  <key>StandardOutPath</key><string>/Users/yuxichen/Documents/GitHub/Starlink_D2C/TLE/fetch.log</string>
  <key>StandardErrorPath</key><string>/Users/yuxichen/Documents/GitHub/Starlink_D2C/TLE/fetch.log</string>
</dict></plist>
```

Automating the **archive** fetch is rarely needed — it's a one-off backfill for
fixed historical dates.

---

## 5. Verify it worked

```bash
./.venv/bin/python -m qpviz.satellites summary TLE
```

In the app sidebar, the *Satellite visibility* panel shows the D2C count and the
epoch span. The visibility tab prints the **TLE epoch range vs your data window**
— the closer they overlap, the more accurate the counts.

Don't forget the **“Measurement clock → UTC offset”** field: set it to your logs'
offset (0 if the timestamps are already UTC; −4 for US-Eastern EDT). A wrong
offset rotates the whole sky in longitude and makes counts meaningless.

---

## 6. Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `CERTIFICATE_VERIFY_FAILED` | macOS Python has no CA bundle. `pip install -r requirements.txt` (certifi), or run *Install Certificates.command* (§0). |
| `Space-Track login failed` | Wrong `SPACETRACK_USER`/`SPACETRACK_PASS`, or the account isn't activated. Log in once on the website first. |
| `No D2C NORAD ids to query` | CelesTrak couldn't be reached to learn the D2C set — check network / SSL, then retry. |
| Visibility tab still says *No TLEs loaded* | Files aren't in the folder the sidebar's **TLE folder** points at, or they carry no `[DTC]` name tag. Run `summary` to confirm the count. |
| Counts look wrong / zero everywhere | TLE epochs far from the data dates (pull the matching Space-Track range), or the **UTC offset** is unset. |
| Space-Track fetch is slow or throttled | Normal — it chunks and pauses to respect rate limits. Narrow the date range if you only need part of it. |

---

## How the app uses these files (for reference)

1. Reads every `*.tle` / `*.txt` in the TLE folder; keeps satellites whose name
   contains `[DTC]`.
2. For each **QP GPS** sample, picks that satellite's TLE with the **nearest
   epoch** (samples bucketed by date), propagates with **SGP4**, converts to a
   topocentric **elevation**, and counts satellites at/above the mask (25°
   default).
3. Caches the result, so it recomputes only when the data, TLEs, or UTC offset
   change.
