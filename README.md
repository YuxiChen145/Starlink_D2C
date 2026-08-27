# Starlink D2C — Measurement Explorer

An interactive platform for exploring Qualipoc drive/walk-test exports from the
Starlink Direct-to-Cell measurement campaign: pick any combination of
collections, filter down to the cells you care about, and compare them on a map
or as CDF curves on one canvas.

![tabs: Map · CDF curves · Statistics · Data](docs/screenshot-cdf.png)

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app looks for the exports in `QP CSV/` next to `app.py`. Point it somewhere
else with the **Data folder** box in the sidebar, or with the `QP_DATA_DIR`
environment variable:

```bash
QP_DATA_DIR="/path/to/QP CSV" streamlit run app.py
```

The CSVs are **not** committed (see `.gitignore`) — drop your own Qualipoc
export into `QP CSV/` and the app picks up whichever of the ten known files are
present.

## What it does

**Multiple collections as one data source.** Every filter is a multi-select, and
an empty selection means *all*. Selecting several collections pools their
samples; splitting curves by `Collection` compares them side by side.

**No dataset picking.** You choose a **Technology** (All / LTE / 5G NR) and a
**Metric**; the app works out which of the ten exports can supply that metric
and pools them. The sidebar names what it read, e.g. *"Reading LTE PDSCH
[per-carrier], LTE PUSCH [per-carrier]"*.

**Filters** — Direction (DL / UL), Data source, Collection, Band,
EARFCN / NR-ARFCN, PCI, and a **region drawn on the map**. Every one is a
multi-select where empty means *all*, and each filter's options are derived
from what survives the filters above it, so you never see a PCI that no longer
has data behind it. When a metric exists in both directions — throughput and
MCS do — Direction defaults to **DL** rather than silently pooling uplink and
downlink into one distribution. Direction and Data source are also available
as CDF grouping dimensions, so you can put DL and UL, or per-carrier and
aggregated, on the same canvas deliberately.

**Region selection.** On the Map tab, use the rectangle or polygon tool at the
top-left. Every shape you draw is applied (as a union) to *all* tabs — map, CDF,
statistics and the raw data export. "Clear" in the sidebar removes them.
Point-in-polygon is an even-odd ray cast, so concave shapes work correctly.

**Visualisations**

| Tab | What it shows |
|---|---|
| Map | Each sample coloured by the chosen metric, on OSM / Carto / Esri satellite tiles. Diverging red↔blue about a midpoint you set, or a single-hue sequential ramp. Hover any point for its value, collection, PCI, carrier and timestamp. |
| CDF curves | One ECDF per group, all on the same canvas. CCDF toggle. |
| Statistics | Per-group n / mean / std / min / P5 / P10 / P50 / P90 / P95 / max, downloadable as CSV. |
| Data | The filtered rows, downloadable as CSV. |

**Metrics** — RSRP, RSRQ, RSSI, SINR, MCS, Throughput, plus BLER, CQI, Tx power
and DL pathloss where the export carries them. The metric picker only offers
what the selected datasets actually contain.

### CDF curves: grouping

"Split curves by" takes **any combination** of Collection, PCI, EARFCN/NR-ARFCN,
Band, Technology, Direction, Data source and Cell type:

- `PCI` — one curve per PCI, multiple PCIs on one canvas (select the PCIs you
  want in the sidebar, or leave it empty for all).
- `PCI` + `EARFCN / NR-ARFCN` — the same PCIs separated by carrier, when a PCI
  appears on more than one frequency.
- `Collection` — the same cells compared across collections.
- `Collection` + `PCI` — every combination, for a full cross-comparison.

Curves are ordered largest-sample-first; **Max curves** caps how many are drawn
and **Min samples per curve** (sidebar) drops groups too small to be meaningful.
The first eight curves take the eight categorical colours; beyond that the hues
repeat with a different dash pattern, so two curves never look identical.

## How the exports are normalised

Each Qualipoc file names its columns differently. `qpviz/schema.py` maps all ten
onto one canonical schema, so a filter or a curve never has to know which file a
row came from.

| Export | Tech | Dir | Metrics contributed |
|---|---|---|---|
| LTE Radio [connected] | LTE | DL | RSRP, RSRQ, RSSI, SINR |
| LTE Radio Neighbor | LTE | DL | RSRP, RSRQ, RSSI |
| LTE PDSCH [per-carrier] | LTE | DL | MCS, throughput, BLER |
| LTE PDSCH [aggregated] | LTE | DL | MCS, throughput, BLER |
| LTE PUSCH [per-carrier] | LTE | UL | MCS, throughput, Tx power, pathloss |
| LTE PUCCH | LTE | UL | CQI, Tx power |
| NR Radio [connected] | NR | DL | SS-RSRP, SS-RSRQ, SS-SINR |
| NR PDSCH [per-carrier] | NR | DL | MCS, throughput, SINR, BLER |
| NR PDSCH [aggregated] | NR | DL | MCS, throughput, BLER |
| NR PUSCH [per-carrier] | NR | UL | MCS, throughput, Tx power, pathloss |

Things worth knowing about the data:

- **Throughput is in kbps** — that is the unit Qualipoc reports (bytes per
  sample window). The `[aggregated]` exports sum all component carriers; the
  `[per-carrier]` ones do not.
- **Bands are normalised** to `B25` / `n25` whether the raw column said `25`,
  `LTE E-UTRA 25` or `5G NR n25`; anything unparseable becomes `Unknown`.
- **LTE PUCCH has no GPS columns.** Those rows never appear on the map, and a
  drawn region drops them unless you tick "Keep rows without GPS".
- **LTE Radio Neighbor leaves PCI and EARFCN blank** on most rows. "Require a
  known PCI and EARFCN" (on by default) keeps those rows from collapsing into
  one large `n/a` curve.
- Rows where every metric of a file is empty are dropped at load time.

### Caching

First load parses the CSVs (a few seconds for the full ~110 MB export) and
writes a normalised parquet cache under `.cache/qpviz/`, keyed by file size and
mtime — later loads are near-instant. Replace an export and the cache
invalidates itself. "Reload data" in the sidebar clears Streamlit's in-memory
cache; delete `.cache/` to force a full re-parse.

## Layout

```
app.py               Streamlit UI: filters, tabs, wiring
qpviz/
  schema.py          the ten exports and their column maps; metric registry
  loader.py          CSV -> canonical frame, dtype coercion, parquet cache
  filters.py         filter set, point-in-polygon region test
  stats.py           ECDF + grouped summary statistics
  charts.py          plotly CDF figure, map colour binning, legend
  mapview.py         folium map, draw control, GeoJSON -> polygons
  theme.py           colour tokens (categorical / sequential / diverging)
tests/               pytest suite, incl. a headless run of the whole app
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite builds small synthetic CSVs that reuse the real column names, so it
runs without the measurement data. `tests/test_app.py` executes `app.py`
headlessly through `streamlit.testing` and drives the filters; if a real
`QP CSV/` folder is present it additionally smoke-tests the app against it.
