# TLE folder — Starlink Direct-to-Cell orbital data

Drop **two-line element (TLE)** files here. The Measurement Explorer reads every
`*.tle` / `*.txt` file in this folder, keeps the Direct-to-Cell satellites
(identified by a **`[DTC]`** tag in the TLE name line, the CelesTrak
convention), and propagates them with SGP4 to estimate how many were overhead at
each QP measurement instant.

Data files are **not** committed (see `.gitignore`) — only this README is.

## Two ways to populate it

**1. Daily latest (going forward).** Fetch the current Starlink set from
CelesTrak — at most once a day (it skips if today's file already exists):

```bash
python -m qpviz.satellites fetch "TLE"
python -m qpviz.satellites summary "TLE"
```

You can also click **“Fetch today's Starlink TLEs”** in the app sidebar.

Automate it daily with cron (Linux) …

```cron
# every day at 06:15, refresh TLEs for the explorer
15 6 * * *  cd /path/to/Starlink_D2C && ./.venv/bin/python -m qpviz.satellites fetch TLE
```

… or launchd (macOS): a `StartCalendarInterval` agent running the same command.

**2. Historical snapshots (for the existing campaign dates).** SGP4 is only
accurate near a TLE's epoch, and CelesTrak only serves *today's* elements. For
the July–August 2026 measurements you need TLEs from **around those dates**, from
the Space-Track GP-history archive (free account at
<https://www.space-track.org>). One command does the whole thing — it learns
which Starlink are D2C from CelesTrak's `[DTC]` tags, pulls just those
satellites' history for your date range, and writes them tagged `[DTC]` here:

```bash
# your QP data spans 2026-07-22 → 2026-08-30
export SPACETRACK_USER="you@example.com"
export SPACETRACK_PASS="…"
python -m qpviz.satellites archive --start 2026-07-22 --end 2026-08-30 --tle-dir TLE
```

It queries in polite chunks (Space-Track rate-limits), pads the range by a day
each side, and saves `TLE/spacetrack_<start>_<end>.tle`. The app buckets samples
by date and propagates each from the **nearest-epoch** TLE, so the more of the
range you cover, the more accurate the counts.

> Note: the D2C set is taken from CelesTrak *today*, so a satellite that flew
> during the campaign but has since decayed may be missed. For most purposes the
> current set is a close approximation.

> D2C satellites must carry the `[DTC]` tag in the name line. CelesTrak's
> `GROUP=starlink` feed includes it; if a source omits names, add a `0 NAME`
> line above each element set.
