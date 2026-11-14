# Starlink D2C — Measurement Explorer

An interactive Streamlit app for exploring Starlink Direct-to-Cell measurement
data from three instruments side by side — **Qualipoc (QP)** connected-mode
drive/walk logs, **PRISM** SDR sweeps, and **SigCap** (an Android logger that
captures the serving cell and ping RTT, and runs the iperf / HTTP tests). Pick a technology and metric, filter down to the
cells, region, and minutes you care about, and compare them on a map, as CDF/PDF
curves, by which cells were serving and for how long, and against how many D2C
satellites were overhead.

![The CDF tab](docs/screenshot-cdf.png)

> This README is a **setup guide**. For what each tab does in depth, see the
> [Features](#features) section; for satellite TLEs, see
> [`docs/FETCHING_TLE.md`](docs/FETCHING_TLE.md).

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick start](#2-quick-start)
3. [The data (already in the repo)](#3-the-data-already-in-the-repo)
4. [Run it](#4-run-it)
5. [Configuration](#5-configuration)
6. [Share it with others (deploy)](#6-share-it-with-others-deploy)
7. [Features](#features)
8. [Project layout](#project-layout)
9. [Tests](#tests)
10. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

- **Python 3.10+** (developed on 3.12).
- **git**, and ~500 MB free disk for the repo, its data, and the parquet cache.
- Access to this **private** GitHub repo. The measurement data lives inside it
  as plain CSV, so a clone gives you both the code and the data in one step —
  no separate download.

## 2. Quick start

```bash
git clone https://github.com/YuxiChen145/Starlink_D2C.git
cd Starlink_D2C

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

That's it — the clone already contains the data, so the app has something to
show on first run. Streamlit opens `http://localhost:8501`.

## 3. The data (already in the repo)

The measurement CSVs **ship in the repo as plain `.csv`** (~150 MB), so a fresh
`git clone` already has them. The app reads three folders next to `app.py`:

```
Starlink_D2C/
├── QP CSV/       ← Qualipoc exports (the "… - Limited.csv" files)
├── PRISM CSV/    ← PRISM SDR scan CSVs (one file per location/carrier)
├── Sigcap CSV/   ← SigCap sessions (one CSV per iperf / HTTP capture; subfolders OK)
└── TLE/          ← (optional) Starlink TLEs for satellite visibility
```

- **QP CSV/** — the Qualipoc exports (`LTE Radio [connected] - Limited.csv`,
  `NR PDSCH [per-carrier] - Limited.csv`, …). The app looks these up **by exact
  file name** and auto-detects which are present; a re-export must replace the
  file of the same name (a `… - Limited 2.csv` copy is ignored). Either
  measurement source alone is enough to run.
- **PRISM CSV/** — the SDR scan files (the collection name comes from the file
  name). PRISM carries signal metrics only and has no GPS in the current
  captures, so it is excluded from the map and satellite tabs.
- **Sigcap CSV/** — SigCap sessions, one CSV per capture. Sessions may sit
  directly in the folder or be grouped into subfolders (any depth; hidden and
  `__MACOSX` folders are skipped). Use SigCap's **`.debug_general`** export:
  the per-cell `.debug_cellular` export has no serving-cell or iperf/HTTP/ping
  columns and is skipped. The collection name
  comes from the file name; both the `…_910_iperf` and the
  `[D2C_Static_OpenField…]` naming conventions are recognised, and macOS
  "copy 2" duplicates are ignored). SigCap adds the serving cell (PCI / Cell Id),
  RSRP / RSRQ / RSSI and **ping RTT**, and has GPS, so it appears on the map.
  It also ran the iperf / HTTP / ping tests, so its columns say which test was
  active on each row. Its own iperf / HTTP throughput figure is **not** shown:
  it is not the throughput through the chip and transmitter, which the QP
  PDSCH / PUSCH exports measure.
- **TLE/** — optional; only needed for the **D2C visibility** tab. TLEs are
  *not* committed; fetch them locally — see
  [`docs/FETCHING_TLE.md`](docs/FETCHING_TLE.md).

**Refreshing or adding data** — drop the new or updated CSV into its folder and
commit it like any other file:

```bash
git add "PRISM CSV/2026_10_02_LTE_8665_NewPlace_static.csv"
git commit -m "Add PRISM NewPlace static scan (2026-10-02)"
git push
```

A few conventions keep the folders clean:

- **No compressed copies.** `*.csv.gz` and `*.zip` are git-ignored; commit the
  plain `.csv`.
- **No duplicates.** macOS "keep both" copies (`… 2.csv`) and `(1)`/`(2)`
  variants of PRISM files are read as *separate collections* and double-count,
  so delete them before committing.
- **File names are collection names** for PRISM and SigCap, so name a capture
  the way you want it to appear in the app.
- **Every file must be under 100 MB** (GitHub's hard limit). The largest QP
  export today is ~19 MB.

> **Privacy.** The QP exports carry GPS coordinates. That is why the repo is
> **private** — keep it that way, and share access (or any deploy) only with
> people who should see the measurement locations.

A parquet cache is built under `.cache/qpviz/` on first load and reused after.

## 4. Run it

```bash
streamlit run app.py
```

First load parses the CSVs (a few seconds) and caches them; later loads are
near-instant. Use **Reload data** in the sidebar after replacing a file.

## 5. Configuration

Point the app at data elsewhere with environment variables (all optional):

| Variable | Default | What it is |
|---|---|---|
| `QP_DATA_DIR` | `QP CSV` | Folder of Qualipoc exports |
| `PRISM_DATA_DIR` | `PRISM CSV` | Folder of PRISM scans |
| `SIGCAP_DATA_DIR` | `Sigcap CSV` | Folder of SigCap session CSVs |
| `TLE_DATA_DIR` | `TLE` | Folder of Starlink TLE files |
| `QP_CACHE_DIR` | `.cache/qpviz` | Parquet cache location |
| `CARTO_API_KEY` | *(unset)* | Free key for the CARTO Light/Dark basemaps (see below) |

```bash
QP_DATA_DIR="/data/QP CSV" PRISM_DATA_DIR="/data/PRISM CSV" streamlit run app.py
```

The **QP data folder** and **PRISM data folder** boxes in the sidebar override
these at runtime too.

**Map basemaps.** Since 25 Sept 2026 CARTO serves its Light (Positron) and Dark
(Dark Matter) basemaps only with an API key; without one they render as blank
"API KEY REQUIRED" tiles. The app therefore offers those two styles only when a
key is configured, and otherwise defaults to OpenStreetMap (Esri satellite also
works without a key). To enable them:

1. Request a free key at <https://carto.com/basemaps/apikey/> (no account needed;
   pick non-commercial for research use — 5M tile requests/month).
2. In <https://dashboard.basemaps.carto.com> restrict the key to your hosts
   (`localhost`, `127.0.0.1`, and your deploy's domain) — the key is visible in
   the browser, so the restriction is what stops reuse.
3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` (git-ignored)
   and set `CARTO_API_KEY = "…"`, or export `CARTO_API_KEY` in the environment
   (the environment wins). On Streamlit Community Cloud, paste it into the app's
   *Secrets* setting instead.

Keep the CARTO and OpenStreetMap attribution visible on the map — it is a
condition of the free tier.

Optional — satellite visibility needs Starlink TLEs. Fetch the current set with:

```bash
python -m qpviz.satellites fetch TLE
```

and, for the historical campaign dates, pull the Space-Track archive (see the
[TLE guide](docs/FETCHING_TLE.md)).

---

## 6. Share it with others (deploy)

The app is a normal Streamlit server, so anyone with the URL can use it — no
install on their side. Because the data is committed to the (private) repo, any
host that pulls the repo also gets the data — no separate upload step. Pick the
option that fits.

### A. Quick public link — tunnel your local app (easiest)

Run the app locally, then expose that port with a tunnel. Others open the URL;
nothing runs on their machine.

```bash
# terminal 1 — run the app
streamlit run app.py

# terminal 2 — expose port 8501 (pick one)
cloudflared tunnel --url http://localhost:8501      # Cloudflare, no account
ngrok http 8501                                      # ngrok (free account)
```

Great for demos and sharing with a few people. The link lives only while your
app + tunnel are running.

### B. Streamlit Community Cloud (free hosting from GitHub)

At <https://share.streamlit.io>, connect this repo and point at `app.py`. It
builds from `requirements.txt` automatically, and because the data is in the
repo, the deployed app has it — nothing extra to upload. Since the repo is
private, authorize Streamlit's access to it when prompted. Anyone you give the
**app URL** to can use the app; the repo itself stays private.

### C. Self-host with Docker (persistent, your own server)

A `Dockerfile` and `.dockerignore` are included. The image contains the **code
only**; you mount your data at run time, so nothing sensitive is baked into the
image.

```bash
# build once
docker build -t starlink-d2c .

# run — mount a folder that contains "QP CSV", "PRISM CSV", "Sigcap CSV" and "TLE"
docker run -p 8501:8501 -v "/absolute/path/to/data:/data" starlink-d2c
```

Then open `http://<server-ip>:8501`. On a cloud VM (EC2, a droplet, your lab
box), put it behind nginx/Caddy for HTTPS and a domain. Best for a stable,
always-on internal site — and keep it access-controlled, since the mounted data
includes GPS.

**Rule of thumb:** share a quick view → **tunnel (A)**; a hosted site straight
from GitHub → **Community Cloud (B)**; a stable, controlled internal site →
**Docker on a server (C)**.

---

## Features

**Three measurement sources, one canvas.** A *Measurement source* filter (QP /
PRISM / SigCap) pools or isolates them; every other filter is a multi-select
where empty means *all*. SigCap rows also carry a **device identity** (the phone
UUID + codename), so sessions logged by different phones at the same time stay
distinct.

**Filters** — Measurement source, Direction (DL/UL), **Traffic** (the generated
traffic — iperf DL / iperf UL, Ookla, Ping, HTTP, Idle; read **per row from the
data**: SigCap from its iperf/HTTP/ping columns, QP from its `Test Name` column
and per-channel throughput, so a combined session splits into its active phases;
`Mixed` = a combined session's rows that carry no bulk transfer). In the **D2C**
campaign QP records idle (PHY only) while SigCap generates the traffic, so those
QP rows are relabelled from the time-aligned SigCap traffic rather than QP's own
idle columns or the (sometimes mistyped) collection name), Collection, Band,
EARFCN/NR-ARFCN, PCI (union or intersection across collections), a **region drawn
on the map**, and a **time window** (with a per-session jump so
you can isolate a few minutes at a spot).

**Metrics** — each one is read from one fixed Qualipoc export, named above the
plots, so there is no export to pick and no sample is counted twice:

| Metric | Read from |
|---|---|
| RSRP, RSRQ, RSSI, SINR, **RSRQ N-factor** (`N = RSRQ·RSSI/RSRP`) | Radio [connected] — the serving cell only (plus the PRISM and SigCap readings when those sources are selected) |
| **DL throughput, all carriers** | PDSCH [aggregated] |
| **DL throughput, per carrier** | PDSCH [per-carrier] |
| **UL throughput, all carriers** and **per carrier** | PUSCH [per-carrier]; there is no aggregated uplink export, so the all-carriers figure is the sum of the carriers at each timestamp |
| MCS, BLER (DL only) | PDSCH / PUSCH [per-carrier] |
| **Spectral efficiency (bps/Hz)**, **Subframe usage (%)**, **RBs per used subframe**, **RBs per subframe, all subframes** | PDSCH / PUSCH [per-carrier] |
| **Ping RTT (ms)** | SigCap |

*Subframe usage* is the share of the available sub-frames (slots on NR) in which
the phone was scheduled. *RBs per used subframe* is the grant size in those
sub-frames (`Avg RBs per SF`, reaching the carrier's RB ceiling); *RBs per
subframe, all subframes* averages over idle sub-frames too, so it equals the
grant size times the usage.

| Tab | What it shows |
|---|---|
| 🗺️ Map | Each sample coloured by the metric, by serving PCI / eNB-gNB / sector (every cell its own colour), by how many distinct sector ids the serving eNB carries, or by D2C satellites overhead. |
| 🛰️ D2C visibility | How many Starlink Direct-to-Cell satellites were above the mask at each QP sample — distinct-in-window vs eNB counts, over time, distribution. Needs TLEs. |
| 📈 CDF curves | One ECDF per group on one canvas — or the same groups as a **PDF** (overlaid histograms). CDF / CCDF / PDF toggle. |
| 🛰️ Cell identity | Who is serving you and for how long: dwell timeline, eNB→sector→PCI tree, fan-out statistics. |
| 📊 Statistics | Per-group n / mean / std / percentiles, downloadable. |
| 🧾 Data | The filtered rows, downloadable as CSV. |

Deeper notes on cell identity, the QP↔PRISM differences, satellite visibility,
and how the exports are normalised live in the `qpviz/` module docstrings
(`schema.py`, `loader.py`, `identity.py`, `satellites.py`).

## Project layout

```
app.py               Streamlit entry point: sidebar, filters, header KPIs, tab wiring
qpviz/
  ui/                Streamlit-only code
    common.py        cached data loaders, satellite propagation cache, shared widgets
    context.py       Ctx - the sidebar choices + filtered data every tab reads
    tab_*.py         one module per tab (map, satellites, cdf, cells, stats, data),
                     each with render(ctx)
  schema.py          the exports and their column maps; metric registry
  loader.py          CSV -> canonical frame (QP + PRISM + SigCap), derived metrics, cache
  filters.py         filter set, point-in-polygon region test, data windows
  identity.py        serving-cell scope, dwell runs, hierarchy, fan-out
  stats.py           ECDF + grouped summary statistics
  charts.py          plotly figures: CDF/PDF, timeline, tree, fan-out, satellites
  mapview.py         folium map, draw control, categorical colouring
  satellites.py      TLE ingest, SGP4 propagation, D2C visibility
  theme.py           colour tokens
docs/                screenshot + TLE guide
tests/               pytest suite (synthetic data; no measurement CSVs needed)
Dockerfile           container image for self-hosting
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite builds small synthetic CSVs that reuse the real column names, so it
runs without any measurement data. `tests/test_app.py` executes `app.py`
headlessly; if a real `QP CSV/` folder is present it also smoke-tests against it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No QP CSV files found` | Confirm `QP CSV/` has the `… - Limited.csv` files (or set `QP_DATA_DIR`); check the sidebar folder box. |
| Satellite tab says *sgp4 not installed* | `pip install -r requirements.txt` (adds `sgp4`), restart. |
| TLE fetch `CERTIFICATE_VERIFY_FAILED` | macOS Python has no CA bundle — the `certifi` dependency fixes it; see the [TLE guide](docs/FETCHING_TLE.md). |
| Stale numbers after replacing a CSV | Click **Reload data**; the parquet cache re-keys on file size/mtime, or delete `.cache/`. |
| Port already in use | `streamlit run app.py --server.port 8502`. |
| `zsh: command not found: streamlit` (venv active) | Streamlit isn't installed in this venv: `pip install -r requirements.txt`, then `python -m streamlit run app.py`. |
| Map tiles say *API KEY REQUIRED* or are blank | Set `CARTO_API_KEY` (see [Configuration](#5-configuration)) or pick the OpenStreetMap basemap. |

More: **[docs/FETCHING_TLE.md](docs/FETCHING_TLE.md)** — getting Starlink TLEs
(daily + historical) for the satellite-visibility tab.
