"""Load the QP CSV exports and normalise them onto the canonical schema."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import (
    CARRIER_GROUP,
    D2C_CARRIERS,
    DATASETS,
    DATASETS_BY_KEY,
    DERIVED_METRIC,
    DIMENSION_COLUMNS,
    METRICS,
    NETWORK_D2C,
    NETWORK_TERRESTRIAL,
    PRISM_DATASETS,
    PRISM_DATASETS_BY_KEY,
    SIGCAP_DEVICE_COL,
    traffic_app,
    traffic_label,
    PRISM_GPS_TIME_COL,
    QP_TEST_NAME_COL,
    SIGCAP_DATASETS,
    THROUGHPUT_METRICS,
    Dataset,
)

DEFAULT_CACHE_DIR = os.environ.get("QP_CACHE_DIR", ".cache/qpviz")

log = logging.getLogger(__name__)

#: bump whenever normalisation changes, so stale parquet caches are ignored
CACHE_VERSION = 21

#: one LTE/NR resource block spans 12 subcarriers
_SUBCARRIERS_PER_RB = 12

_TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    # PRISM stamps ISO-8601 with a "T" separator
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
)

_BAND_RE = re.compile(r"(\d+)\s*$")

#: PRISM file names encode the collection after a ``date_TECH_ARFCN_`` prefix,
#: e.g. ``2026_09_01_ LTE_8665_Shoreham_Court_Kitchen_static`` -> the collection
#: is ``Shoreham Court Kitchen static``. The separators are whatever the person
#: typed that day: ``_``, a space, both, or a day-less ``2026_09`` date.
_PRISM_NAME_RE = re.compile(
    r"^\s*\d{4}_\d{2}(?:_\d{2})?[_\s]*(?:LTE|NR)[_\s]*\d+[_\s]+(.+)$", re.IGNORECASE
)

#: columns stored as pandas categoricals (kept object-backed so the parquet
#: cache round-trips to an identical dtype)
_CATEGORICAL = (
    "collection",
    "band",
    "tech",
    "link",
    "source",
    "source_label",
    "provider",
    "test_type",
    "traffic",
    "network",
)

#: free-text dimensions, always stored as pandas ``string``
_TEXT_DIMS = ("database", "operator", "model", "serial", "device", "ran_config", "cell_type")

#: nullable-integer dimensions and the dtype each is stored as
_INT_DIMS = {
    "pci": "Int32",
    "enb": "Int64",
    "sector": "Int32",
    "cell_id": "Int64",
    "ssb": "Int16",
    "freq": "Int64",
}

#: NR reports the gNB-Id and the full NCGI; the cell-local id (what LTE calls
#: the sector) is the remainder once the gNB-Id is shifted out
_NR_CELL_ID_SHIFT = 256


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #


def normalise_band(value, tech: str) -> str:
    """Turn any of ``25`` / ``"25"`` / ``"LTE E-UTRA 25"`` / ``"5G NR n25"`` into
    a short canonical label (``"B25"`` for LTE, ``"n25"`` for NR)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    text = str(value).strip()
    if not text or "unknown" in text.lower() or text.lower() in {"nan", "-", "none"}:
        return "Unknown"
    # a bare number ("25", or "25.0" when a NaN in the column floated the dtype)
    # is the band directly - take its integer part rather than letting the
    # trailing-digits regex below pick the "0" out of "25.0".
    try:
        num = float(text.replace(",", ""))
    except ValueError:
        num = None
    if num is not None and np.isfinite(num):
        number = str(int(num))
        return f"n{number}" if tech == "NR" else f"B{number}"
    match = _BAND_RE.search(text.replace("n", " n").strip()) or _BAND_RE.search(text)
    if not match:
        return text
    number = match.group(1)
    return f"n{number}" if tech == "NR" else f"B{number}"


def parse_time(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    for fmt in _TIME_FORMATS:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().any():
            missing = parsed.isna() & series.notna()
            if missing.any():
                parsed.loc[missing] = pd.to_datetime(
                    series[missing], errors="coerce", format="mixed"
                )
            return parsed
    return pd.to_datetime(series, errors="coerce", format="mixed")


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def split_enb_sector(series: pd.Series):
    """``"11311/73"`` -> ``(11311, 73)``. LTE reports the pair as one string."""
    text = series.astype("string")
    parts = text.str.split("/", n=1, expand=True)
    if parts.shape[1] < 2:
        parts[1] = pd.NA
    enb = pd.to_numeric(parts[0], errors="coerce")
    sector = pd.to_numeric(parts[1], errors="coerce")
    return enb, sector


def _add_efficiency(out: pd.DataFrame, raw: pd.DataFrame, ds: Dataset) -> None:
    """Fill the derived ``spec_eff`` (bps/Hz) column.

    It comes from the net throughput and the RBs allocated averaged over *all*
    slots/subframes (so numerator and denominator cover the same window):

        spec_eff = throughput_bps / (RB_all x 12 x SCS_Hz)          [bps/Hz]

    ``RB_all`` is the time-averaged occupancy, so ``RB_all x 12 x SCS`` is the
    average occupied bandwidth and the ratio is the spectral efficiency of the
    resource the scheduler actually granted (not the whole carrier). LTE fixes
    SCS at 15 kHz; NR reads it per row. Throughput is kbps in the export.
    """
    tput_col = ds.tput_metric
    if not ds.rb_all_col or tput_col is None or tput_col not in out.columns:
        return
    if ds.rb_all_col not in raw.columns:
        return
    rb_all = _to_numeric(raw[ds.rb_all_col]).to_numpy(dtype="float64")
    if ds.scs_col and ds.scs_col in raw.columns:
        scs_khz = _to_numeric(raw[ds.scs_col]).to_numpy(dtype="float64")
        # some exports report SCS in Hz (15000) rather than kHz (15)
        scs_khz = np.where(scs_khz >= 1000.0, scs_khz / 1000.0, scs_khz)
    else:
        scs_khz = np.full(len(out), 15.0)  # LTE
    tput_bps = out[tput_col].to_numpy(dtype="float64") * 1000.0

    valid = np.isfinite(rb_all) & (rb_all > 0) & np.isfinite(scs_khz) & (scs_khz > 0)
    rb_bw_hz = _SUBCARRIERS_PER_RB * scs_khz * 1000.0

    with np.errstate(invalid="ignore", divide="ignore"):
        spec = np.where(valid, tput_bps / (rb_all * rb_bw_hz), np.nan)
    out["spec_eff"] = spec


def _add_carrier_sum(out: pd.DataFrame, ds: Dataset) -> None:
    """Fill the derived all-carriers uplink throughput (``tput_ul``).

    Qualipoc exports an aggregated PDSCH file but no aggregated PUSCH one, so
    the uplink total is built here: the per-carrier throughputs of one phone at
    one timestamp (:data:`~qpviz.schema.CARRIER_GROUP`) are summed, and the sum
    is written on ONE row of the group - its primary carrier - leaving the
    others empty. That gives one sample per timestamp, filed under the primary
    cell's PCI and carrier, which is how the aggregated downlink export is laid
    out. A timestamp with a single carrier keeps its own value.
    """
    if ds.metrics.get("tput_ul") != DERIVED_METRIC or "tput_ul_cc" not in out.columns:
        return
    out["tput_ul"] = np.nan
    if out.empty:
        return
    keys = [out[c] if c in out.columns else pd.Series(pd.NA, index=out.index)
            for c in CARRIER_GROUP]
    gid = pd.Series(
        out.groupby(keys, dropna=False, sort=False).ngroup().to_numpy(), index=out.index
    )
    cc = out["tput_ul_cc"]
    total = cc.groupby(gid).transform("sum", min_count=1)

    kind = (out["cell_type"].astype("string").str.lower()
            if "cell_type" in out.columns else pd.Series("", index=out.index))
    # the primary carrier is the PCell, else the PSCell (NR as the secondary
    # cell group), else whichever carrier is logged
    rank = np.where(
        kind.str.contains("pcell", na=False).to_numpy(), 2,
        np.where(kind.str.contains("pscell", na=False).to_numpy(), 1, 0),
    )
    order = pd.DataFrame({"gid": gid, "rank": rank, "cc": cc.fillna(-np.inf)})
    # per timestamp: the primary carrier first, and of two records for the
    # same carrier the one that actually carried data
    anchor = order.sort_values(
        ["gid", "rank", "cc"], ascending=[True, False, False], kind="stable"
    ).drop_duplicates("gid").index
    out.loc[anchor, "tput_ul"] = total.loc[anchor]


def _network(out: pd.DataFrame, ds: Dataset) -> pd.Series:
    """``D2C`` for a row measured on the Direct-to-Cell carrier, ``Terrestrial``
    for any other known carrier, empty when the carrier never decoded.

    The carrier decides it (:data:`~qpviz.schema.D2C_CARRIERS`), never the
    collection name.
    """
    freq = (pd.to_numeric(out["freq"], errors="coerce") if "freq" in out.columns
            else pd.Series(np.nan, index=out.index))
    on_d2c = freq.isin(D2C_CARRIERS.get(ds.tech, ())).to_numpy()
    network = np.where(on_d2c, NETWORK_D2C, NETWORK_TERRESTRIAL).astype("object")
    network[freq.isna().to_numpy()] = None
    return pd.Series(network, index=out.index, dtype="object")


def _add_n_factor(out: pd.DataFrame, ds: Dataset) -> None:
    """Fill the derived ``n_factor`` from the 3GPP RSRQ identity.

        RSRQ = N · RSRP / RSSI   (linear)  ⇒  N = 10^((RSRQ − RSRP + RSSI)/10)

    with RSRP/RSSI in dBm and RSRQ in dB. N should be the RSSI measurement
    bandwidth in resource blocks (25 / 50 / 100 for 5 / 10 / 20 MHz LTE); the
    point of surfacing it is that measured N drifts from those round numbers.
    Needs all three of RSRP, RSRQ, RSSI on the same row.
    """
    if "n_factor" not in ds.metrics:
        return
    need = ("rsrp", "rsrq", "rssi")
    if not all(c in out.columns for c in need):
        out["n_factor"] = np.nan
        return
    rsrp = out["rsrp"].to_numpy(dtype="float64")
    rsrq = out["rsrq"].to_numpy(dtype="float64")
    rssi = out["rssi"].to_numpy(dtype="float64")
    valid = np.isfinite(rsrp) & np.isfinite(rsrq) & np.isfinite(rssi)
    with np.errstate(over="ignore", invalid="ignore"):
        n = np.where(valid, 10.0 ** ((rsrq - rsrp + rssi) / 10.0), np.nan)
    out["n_factor"] = n


#: a channel carrying more than this (Mbps) is a real bulk transfer; TCP ACKs on
#: the reverse channel of a one-way iperf peak well below it (~0.4), real bulk is
#: several Mbps, so this cleanly tells the active direction apart from ACKs
_QP_BULK_MBPS = 0.5


def _qp_traffic(out: pd.DataFrame, raw: pd.DataFrame, ds: Dataset) -> pd.Series:
    """Per-row QP traffic type from the columns, not just the session name.

    Direction-specific sessions (``...iperfDL`` / ``...iperfUL`` / ``Ookla`` /
    ``Ping`` / ``IDLE``) already name their traffic, so every row takes that
    label. Only a **combined** session (name maps to ``Mixed`` - e.g.
    ``PingiperfULDL``, ``PinghttpULDL``, or a bare ``_http``) is split per row:
    a channel actively transferring (> ``_QP_BULK_MBPS``, so ACKs on the reverse
    channel don't count) is that direction's traffic of the test the name
    mentions (``iperf DL`` / ``HTTP UL`` ...); an idle row stays ``Mixed``
    because QP can't tell a ping from a gap. ``Test Name`` (Qualipoc's
    active-test column) fills in the rows whose name carried no traffic keyword
    at all (``Other``).
    """
    idx = out.index
    coll_label = (
        out["collection"].astype("string").map(traffic_label)
        if "collection" in out.columns
        else pd.Series("Other", index=idx)
    )
    res = coll_label.astype("object")

    link = (out["link"].astype("string") if "link" in out.columns
            else pd.Series(ds.link, index=idx, dtype="string"))
    tput_col = ds.tput_metric
    tput = (pd.to_numeric(out[tput_col], errors="coerce")
            if tput_col in out.columns else pd.Series(np.nan, index=idx))
    bulk = tput.fillna(0.0) > _QP_BULK_MBPS

    # which transfer test the session ran, when the name mentions exactly one
    if "collection" in out.columns:
        names = out["collection"].astype("string")
        lookup = {n: traffic_app(n) for n in names.dropna().unique()}
        app = names.astype("object").map(lookup).astype("object")
    else:
        app = pd.Series(None, index=idx, dtype="object")

    # enrich rows whose session name had no traffic keyword, using Test Name;
    # an iperf test with no direction in sight is a combined session, split below
    if QP_TEST_NAME_COL in raw.columns:
        tn = raw[QP_TEST_NAME_COL].astype("string").str.strip().str.lower().reindex(idx)
        other = (pd.Series(res, index=idx) == "Other").to_numpy()
        res[other & tn.str.contains("ping", na=False).to_numpy()] = "Ping"
        res[other & tn.str.contains("ookla", na=False).to_numpy()] = "Ookla"
        iperf_test = other & tn.str.contains("iperf", na=False).to_numpy()
        res[iperf_test] = "Mixed"
        app[iperf_test] = "iperf"

    # split only the ambiguous combined ("Mixed") sessions, by active direction
    mixed = (pd.Series(res, index=idx) == "Mixed").to_numpy() & app.notna().to_numpy()
    for direction in ("DL", "UL"):
        active = mixed & (bulk & link.eq(direction)).to_numpy()
        res[active] = app[active].astype(str) + f" {direction}"

    return pd.Series(res, index=idx).astype("string")


def normalise(
    raw: pd.DataFrame,
    ds: Dataset,
    collection: Optional[str] = None,
    link=None,
    test_type: Optional[str] = None,
    traffic=None,
    time: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Map one raw CSV frame onto the canonical schema.

    ``collection`` overrides the collection column - PRISM and SigCap have no
    such column, so theirs comes from the file name instead.  ``link`` overrides
    the per-dataset direction: SigCap passes a per-row Series (DL/UL from
    ``iperf_direction``) since one session mixes downlink and uplink tests.
    ``test_type`` tags the workload (``iperf``/``http``) for SigCap rows.
    ``time`` replaces the parsed time column when the caller has already
    derived a better clock (PRISM, see :func:`prism_timestamps`).
    """
    out = pd.DataFrame(index=raw.index)

    for canon, source_col in ds.dims.items():
        col = raw[source_col] if source_col in raw.columns else pd.Series(
            np.nan, index=raw.index
        )
        if canon == "time":
            out["time"] = parse_time(col) if time is None else pd.Series(
                time, index=raw.index
            )
        elif canon in ("lat", "lon"):
            out[canon] = _to_numeric(col)
        elif canon == "enb_sector":
            enb, sector = split_enb_sector(col)
            out["enb"] = enb.astype("Int64")
            out["sector"] = sector.astype("Int32")
        elif canon in _INT_DIMS:
            out[canon] = _to_numeric(col).astype(_INT_DIMS[canon])
        elif canon == "band":
            # one call per distinct raw value, not per row
            lookup = {v: normalise_band(v, ds.tech) for v in col.dropna().unique()}
            out["band"] = col.map(lookup).fillna("Unknown").astype("object")
        else:
            out[canon] = col.astype("string").str.strip()

    for canon, source_col in ds.metrics.items():
        if source_col == DERIVED_METRIC:
            out[canon] = np.nan  # filled by _add_efficiency below
            continue
        out[canon] = (
            _to_numeric(raw[source_col]) if source_col in raw.columns else np.nan
        )

    _add_efficiency(out, raw, ds)
    _add_n_factor(out, ds)

    # QP logs net throughput in kbps; store it in Mbps to keep axes readable
    # (kbps renders as "5k"). Done AFTER _add_efficiency, which needs the kbps
    # value for its bps maths.
    for col in THROUGHPUT_METRICS:
        if ds.metrics.get(col, DERIVED_METRIC) != DERIVED_METRIC:
            out[col] = pd.to_numeric(out[col], errors="coerce") / 1000.0
    # The exports give sub-frame / slot usage and BLER as 0-1 fractions; the
    # metrics are in percent.
    for col in ("sf_usage", "bler"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * 100.0
    _add_carrier_sum(out, ds)

    # NR gives the gNB-Id and the full cell identity but no explicit sector;
    # the cell-local id is what corresponds to LTE's sector.
    if "sector" not in out.columns and {"enb", "cell_id"} <= set(out.columns):
        local = out["cell_id"].astype("Float64") - out["enb"].astype("Float64") * _NR_CELL_ID_SHIFT
        local = local.where((local >= 0) & (local < _NR_CELL_ID_SHIFT))
        out["sector"] = local.astype("Int32")

    out["tech"] = ds.tech
    out["network"] = _network(out, ds)
    out["link"] = link if link is not None else ds.link
    out["source"] = ds.key
    out["source_label"] = ds.label
    out["provider"] = ds.provider
    if collection is not None:
        out["collection"] = collection
    if test_type is not None:
        out["test_type"] = test_type
    # traffic type - one label per row. SigCap passes a per-row Series read from
    # its measurement columns (a bare-"iperf" session is really ping + iperf
    # UL/DL); QP / PRISM fall back to the collection-name classifier.
    if traffic is not None:
        out["traffic"] = pd.Series(traffic, index=out.index).astype("string")
    elif ds.provider == "QP":
        out["traffic"] = _qp_traffic(out, raw, ds)
    elif "collection" in out.columns:
        out["traffic"] = out["collection"].astype("string").map(traffic_label)
    if "lat" not in out.columns:
        out["lat"] = np.nan
        out["lon"] = np.nan
    for col in DIMENSION_COLUMNS:
        if col not in out.columns:
            dtype = _INT_DIMS.get(col, "string")
            out[col] = pd.Series(pd.NA, index=out.index, dtype=dtype)

    metric_cols = list(ds.metrics)
    if metric_cols:
        keep = out[metric_cols].notna().any(axis=1)
        if traffic is not None:
            # SigCap: a row logged while a test ran still dates that test phase
            # for the traffic reference (see align_d2c_traffic), even when it
            # carries no radio fix and so no metric value
            in_test = out["traffic"].notna() & out["traffic"].ne("Other")
            keep |= in_test.fillna(False).astype(bool)
        out = out[keep]

    ordered = DIMENSION_COLUMNS + metric_cols
    out = out[ordered].reset_index(drop=True)
    return coerce_dtypes(out)


# --------------------------------------------------------------------------- #
# Reading + caching
# --------------------------------------------------------------------------- #


def coerce_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Pin every canonical column to one dtype.

    Applied to freshly parsed frames *and* to frames read back from the parquet
    cache, so the two are always interchangeable.
    """
    for col in _TEXT_DIMS:
        if col in frame.columns:
            frame[col] = frame[col].astype("string")
    for col in _CATEGORICAL:
        if col in frame.columns:
            frame[col] = frame[col].astype("object").astype("category")
    for col in ("lat", "lon"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    for col, dtype in _INT_DIMS.items():
        if col in frame.columns:
            frame[col] = frame[col].astype(dtype)
    if "time" in frame.columns:
        # pin the resolution: pandas 2 infers ns and pandas 3 us, and a frame
        # that changes unit between versions changes every duration downstream
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce").astype(
            "datetime64[ns]"
        )
    for col in METRICS:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    return frame


def _fingerprint(path: Path, ds: Optional[Dataset] = None) -> str:
    """Cache key for one file: loader version, dataset column map, file name,
    size and mtime - any of them changing rebuilds the parquet."""
    stat = path.stat()
    sig = ds.signature() if ds is not None else "-"
    raw = f"v{CACHE_VERSION}:{sig}:{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _resolve_csv(root: Path, filename: str) -> Optional[Path]:
    """Locate a QP export by its exact file name, or ``None`` if absent."""
    p = root / filename
    return p if p.is_file() else None


def read_dataset(
    data_dir: os.PathLike | str,
    ds: Dataset,
    cache_dir: Optional[os.PathLike | str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Read + normalise one dataset, using a parquet cache when available."""
    path = _resolve_csv(Path(data_dir), ds.filename)
    if path is None:
        raise FileNotFoundError(Path(data_dir) / ds.filename)

    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{ds.key}-{_fingerprint(path, ds)}.parquet"
        if cache_path.exists():
            try:
                return coerce_dtypes(pd.read_parquet(cache_path))
            except Exception:  # pragma: no cover - corrupt cache, just rebuild
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    pass

    header = pd.read_csv(path, nrows=0)
    wanted = [c for c in ds.raw_columns() if c in header.columns]
    raw = pd.read_csv(path, usecols=wanted, low_memory=False)
    frame = normalise(raw, ds)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache_path, index=False)
        except Exception:  # pragma: no cover - caching is best effort
            pass
    return frame


def available_datasets(data_dir: os.PathLike | str) -> List[Dataset]:
    """Datasets whose CSV is present in ``data_dir``."""
    root = Path(data_dir)
    return [d for d in DATASETS if _resolve_csv(root, d.filename) is not None]


def load(
    data_dir: os.PathLike | str,
    dataset_keys: Sequence[str],
    cache_dir: Optional[os.PathLike | str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Load and concatenate several datasets into one canonical frame."""
    frames: List[pd.DataFrame] = []
    for key in dataset_keys:
        ds = DATASETS_BY_KEY.get(key)
        if ds is None:
            continue
        try:
            frames.append(read_dataset(data_dir, ds, cache_dir=cache_dir))
        except FileNotFoundError:
            continue
    return concat_frames(frames)


def concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Align columns across ``frames`` and concatenate into one canonical frame.

    Shared by the QP loader, the PRISM loader and the app's QP+PRISM pooling, so
    a mix of measurement sources always lands on one column set with one dtype
    per column.
    """
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return empty_frame()
    # Give every frame the same columns, with missing *metrics* typed as float
    # so the concat never has to guess a dtype from an all-NA object column.
    columns: List[str] = []
    for frame in frames:
        columns.extend(c for c in frame.columns if c not in columns)
    aligned = []
    for frame in frames:
        missing = {c: np.nan for c in columns if c not in frame.columns}
        if missing:
            frame = frame.assign(**missing)
        aligned.append(frame[columns])
    combined = pd.concat(aligned, ignore_index=True, sort=False)
    for col in _CATEGORICAL:
        if col in combined.columns:
            combined[col] = combined[col].astype("object").astype("category")
    return combined


# --------------------------------------------------------------------------- #
# PRISM (SDR scanner) reading
# --------------------------------------------------------------------------- #


def prism_collection(path: Path) -> str:
    """Collection name for a PRISM file, taken from its name.

    ``2026_09_01_ LTE_8665_Shoreham_Court_Kitchen_static.csv`` ->
    ``Shoreham Court Kitchen static``.
    """
    name = path.name
    if name.endswith(".csv"):
        name = name[:-4]
    stem = name.strip()
    match = _PRISM_NAME_RE.match(stem)
    core = match.group(1) if match else stem
    return core.replace("_", " ").strip()


#: a PRISM system clock this far from the GPS clock is unset, not merely drifting
_PRISM_CLOCK_SLACK = pd.Timedelta(hours=1)


def prism_timestamps(raw: pd.DataFrame, name: str = "") -> pd.Series:
    """Best wall-clock time for each PRISM row.

    PRISM stamps every row with its own ``system_date_time`` (millisecond
    resolution) and, when it has a fix, the GPS clock ``gps_date_time`` (whole
    seconds). Some captures were made with the system clock unset, which puts
    them in the year 2070. When the two clocks disagree by more than
    :data:`_PRISM_CLOCK_SLACK`, the median GPS-minus-system offset is added to
    the system clock, which keeps the millisecond cadence while moving the
    capture to the right day; rows without a fix are corrected too. A file with
    a bad clock and no GPS fix at all is moved onto the ``YYYY_MM_DD`` date its
    file name starts with (see :func:`_anchor_unset_clock`); without such a date
    it is left as recorded and logged.
    """
    sys_t = parse_time(raw["system_date_time"]) if "system_date_time" in raw else pd.Series(
        pd.NaT, index=raw.index
    )
    if PRISM_GPS_TIME_COL not in raw.columns:
        return sys_t
    gps_t = parse_time(raw[PRISM_GPS_TIME_COL])
    both = gps_t.notna() & sys_t.notna()
    if both.any():
        offset = (gps_t[both] - sys_t[both]).median()
        if abs(offset) > _PRISM_CLOCK_SLACK:
            log.info("PRISM %s: system clock off by %s, corrected from GPS time", name, offset)
            return sys_t + offset
        return sys_t
    # no row has both clocks: trust GPS where present, else the system clock
    out = sys_t.where(sys_t.notna(), gps_t)
    return _anchor_unset_clock(out, name)


#: ``YYYY_MM_DD`` at the start of a PRISM file name - the capture date
_PRISM_NAME_DATE_RE = re.compile(r"^\s*(\d{4})_(\d{2})_(\d{2})(?!\d)")


def prism_name_date(name: str) -> Optional[pd.Timestamp]:
    """The capture date a PRISM file name starts with, or ``None``.

    ``2026_09_30 LTE_8665_DC_Hotel_7thFloor_static.csv`` -> 2026-09-30. A name
    with only a year and month (``2026_09_LTE_...``) gives ``None``.
    """
    match = _PRISM_NAME_DATE_RE.match(name or "")
    if not match:
        return None
    try:
        return pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)),
                            day=int(match.group(3)))
    except ValueError:
        return None


def _anchor_unset_clock(times: pd.Series, name: str) -> pd.Series:
    """Move rows stamped by an unset clock (e.g. 2070-01-01 00:07) onto the
    capture date in the file name, keeping the elapsed time since the clock
    started: ``name date + (t - day the bad clock started)``.

    The date is right; the time of day is only the time since the scanner
    booted, because nothing in the file records the real one.
    """
    latest_plausible = pd.Timestamp.now().year + 1
    years = times.dt.year
    bad = times.notna() & ((years < 2020) | (years > latest_plausible))
    if not bad.any():
        return times
    day = prism_name_date(name)
    if day is None:
        log.warning("PRISM %s: system clock unset (year %d), no GPS fix and no "
                    "YYYY_MM_DD date in the file name to correct it",
                    name, int(years[bad].iloc[0]))
        return times
    start = times[bad].min().normalize()
    log.info("PRISM %s: system clock unset (year %d) and no GPS fix; placed on "
             "%s from the file name (time of day is time since boot)",
             name, int(years[bad].iloc[0]), day.date())
    return times.where(~bad, day + (times - start))


def _prism_key_for(path: Path) -> Optional[str]:
    """Which PRISM dataset a file belongs to, decided from its header columns."""
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:  # pragma: no cover - unreadable file
        return None
    cols = set(header.columns)
    if "nrarfcn" in cols or "ss_rsrp" in cols:
        return "prism_nr"
    if "earfcn" in cols or "rsrp" in cols:
        return "prism_lte"
    return None


def prism_files(prism_dir: os.PathLike | str) -> List[Path]:
    """Every PRISM scan (``*.csv``) in the folder, sorted by name."""
    root = Path(prism_dir)
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*.csv") if p.is_file())


def prism_available(prism_dir: os.PathLike | str) -> List[str]:
    """PRISM dataset keys that have at least one file present, in schema order."""
    keys = set()
    for path in prism_files(prism_dir):
        key = _prism_key_for(path)
        if key:
            keys.add(key)
    return [d.key for d in PRISM_DATASETS if d.key in keys]


def read_prism_file(
    path: Path,
    ds: Dataset,
    cache_dir: Optional[os.PathLike | str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Read + normalise one PRISM scan file (collection from its name)."""
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{ds.key}-{_fingerprint(path, ds)}.parquet"
        if cache_path.exists():
            try:
                return coerce_dtypes(pd.read_parquet(cache_path))
            except Exception:  # pragma: no cover - corrupt cache, just rebuild
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    pass

    header = pd.read_csv(path, nrows=0)
    wanted = [c for c in ds.raw_columns() if c in header.columns]
    raw = pd.read_csv(path, usecols=wanted, low_memory=False)
    frame = normalise(raw, ds, collection=prism_collection(path),
                      time=prism_timestamps(raw, name=path.name))

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache_path, index=False)
        except Exception:  # pragma: no cover - caching is best effort
            pass
    return frame


def load_prism(
    prism_dir: os.PathLike | str,
    dataset_keys: Sequence[str],
    cache_dir: Optional[os.PathLike | str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Read every PRISM file whose technology is in ``dataset_keys`` and pool them."""
    want = set(dataset_keys)
    frames: List[pd.DataFrame] = []
    for path in prism_files(prism_dir):
        key = _prism_key_for(path)
        if key is None or key not in want:
            continue
        ds = PRISM_DATASETS_BY_KEY[key]
        try:
            frames.append(read_prism_file(path, ds, cache_dir=cache_dir))
        except Exception as exc:  # skip a bad file, keep the rest - but say so
            log.warning("skipping PRISM file %s: %s: %s", path.name, type(exc).__name__, exc)
            continue
    return concat_frames(frames)


def empty_frame() -> pd.DataFrame:
    cols = DIMENSION_COLUMNS + list(METRICS)
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def freq_label_for(frame: pd.DataFrame) -> str:
    """``EARFCN``/``NR-ARFCN``/``EARFCN / NR-ARFCN`` depending on what is loaded."""
    if frame.empty or "tech" not in frame:
        return "EARFCN / NR-ARFCN"
    techs = set(pd.Series(frame["tech"]).dropna().astype(str).unique())
    if techs == {"LTE"}:
        return "EARFCN"
    if techs == {"NR"}:
        return "NR-ARFCN"
    return "EARFCN / NR-ARFCN"


def freq_prefix_for(frame: pd.DataFrame) -> str:
    """Short prefix used in curve names (``EARFCN``/``NR-ARFCN``/``ARFCN``)."""
    label = freq_label_for(frame)
    return "ARFCN" if "/" in label else label


def options(frame: pd.DataFrame, column: str) -> List:
    """Sorted unique non-null values of ``column`` - used to fill the filters."""
    if frame.empty or column not in frame.columns:
        return []
    values = pd.Series(frame[column]).dropna().unique().tolist()
    try:
        return sorted(values)
    except TypeError:  # pragma: no cover - mixed types
        return sorted(values, key=str)


# --------------------------------------------------------------------------- #
# SigCap (Android logger) reading
# --------------------------------------------------------------------------- #
#
# SigCap ships one wide file per test session, with both LTE and NR columns on
# every row and the collection/workload in the file name.  Each row is routed
# to the LTE or NR dataset by its *serving* technology, so a row is never
# double-counted.  The iperf / HTTP columns are read only to label which test a
# row belongs to; SigCap's throughput figure itself is not a metric.

#: trailing tz offset on an ISO-8601 stamp, e.g. ``...930-0400`` / ``...+05:30``
_TZ_OFFSET_RE = re.compile(r"([+-]\d{2}:?\d{2})\s*$")
_SIGCAP_SUFFIX_RE = re.compile(r"\s*\.debug[_ ]?general\s*$", re.IGNORECASE)
#: macOS "keep both" duplicates append a trailing " 2" / " 3" before the ext
_SIGCAP_DUP_RE = re.compile(r"\s+\d+$")
#: SigCap's per-cell ``.debug_cellular`` export - a different layout (one row per
#: heard cell, no serving-cell or app-layer columns), so it is not read
_SIGCAP_CELLULAR_RE = re.compile(r"\.debug[_ ]?cellular\s*\.csv$", re.IGNORECASE)


def _sigcap_serving_tech(raw: pd.DataFrame) -> pd.Series:
    """Per-row serving technology (``"LTE"``/``"NR"``).

    NR when an NR cell is reported (or the network type says NR); LTE when an
    LTE cell is reported (or the network type says LTE); otherwise the file's
    dominant technology, so app-only rows are not dropped.
    """
    idx = raw.index
    nt = (
        raw["network_type*"].astype("string").str.upper()
        if "network_type*" in raw.columns
        else pd.Series(pd.NA, index=idx, dtype="string")
    )
    lte_r = (
        _to_numeric(raw["lte_primary_rsrp_dbm"])
        if "lte_primary_rsrp_dbm" in raw.columns
        else pd.Series(np.nan, index=idx)
    )
    nr_r = (
        _to_numeric(raw["nr_first_ss_rsrp_dbm"])
        if "nr_first_ss_rsrp_dbm" in raw.columns
        else pd.Series(np.nan, index=idx)
    )
    is_nr = nr_r.notna() | nt.str.contains("NR", na=False)
    is_lte = lte_r.notna() | nt.str.contains("LTE", na=False)
    dominant = "NR" if bool(is_nr.any()) else "LTE"
    tech = np.where(is_nr, "NR", np.where(is_lte, "LTE", dominant))
    return pd.Series(tech, index=idx, dtype="object")


def _sigcap_link(raw: pd.DataFrame) -> pd.Series:
    """Per-row DL/UL direction from ``iperf_direction``; an HTTP test is an
    upload when its target is a POST endpoint, otherwise a download."""
    idx = raw.index
    d = (
        raw["iperf_direction"].astype("string").str.lower()
        if "iperf_direction" in raw.columns
        else pd.Series(pd.NA, index=idx, dtype="string")
    )
    http = (
        _to_numeric(raw["http_tput_mean_mbps"])
        if "http_tput_mean_mbps" in raw.columns
        else pd.Series(np.nan, index=idx)
    )
    link = pd.Series("-", index=idx, dtype="object")
    link[d.eq("download")] = "DL"
    link[d.eq("upload")] = "UL"
    is_http = (link == "-") & http.notna()
    upload = _sigcap_http_upload(raw)
    link[is_http & ~upload] = "DL"
    link[is_http & upload] = "UL"
    return link


#: an HTTP test whose target URL has one of these path words sends data up
_HTTP_UPLOAD_RE = re.compile(r"(?:^|[/._?=-])(?:post|put|upload)(?:$|[/._?=-])", re.IGNORECASE)


def _sigcap_http_upload(raw: pd.DataFrame) -> pd.Series:
    """Per row: True when the HTTP test on it uploads (``http_target`` is a
    POST / upload endpoint, e.g. ``https://httpbin.org/post``). SigCap has no
    direction column for HTTP; the target URL is what tells the two apart. A
    row without a target counts as a download."""
    if "http_target" not in raw.columns:
        return pd.Series(False, index=raw.index)
    target = raw["http_target"].astype("string")
    return target.str.contains(_HTTP_UPLOAD_RE, na=False).astype(bool)


def _sigcap_traffic(raw: pd.DataFrame) -> pd.Series:
    """Per-row traffic type from SigCap's own measurement columns.

    A SigCap session whose *name* is only "iperf" actually interleaves ping,
    iperf downlink and iperf uplink phases - and each row records which one was
    running (``iperf_tput_mean_mbps`` + ``iperf_direction``, ``http_tput_...``,
    ``ping_rtt_mean_ms``). So the label is read per row, not from the file name:
    iperf DL/UL first, then HTTP DL/UL (by its target, see
    :func:`_sigcap_http_upload`), then Ping; a radio-only row (no app sample)
    is ``Other``. An iperf row with no direction logged is ``Mixed``.
    """
    idx = raw.index

    def _num(col):
        return (_to_numeric(raw[col]) if col in raw.columns
                else pd.Series(np.nan, index=idx))

    iperf = _num("iperf_tput_mean_mbps")
    http = _num("http_tput_mean_mbps")
    ping = _num("ping_rtt_mean_ms")
    direction = (raw["iperf_direction"].astype("string").str.lower()
                 if "iperf_direction" in raw.columns
                 else pd.Series(pd.NA, index=idx, dtype="string"))

    out = pd.Series("Other", index=idx, dtype="object")
    has_iperf = iperf.notna()
    out[has_iperf & direction.eq("download")] = "iperf DL"
    out[has_iperf & direction.eq("upload")] = "iperf UL"
    out[has_iperf & ~direction.isin(["download", "upload"])] = "Mixed"
    is_http = (out == "Other") & http.notna()
    upload = _sigcap_http_upload(raw)
    out[is_http & ~upload] = "HTTP DL"
    out[is_http & upload] = "HTTP UL"
    out[(out == "Other") & ping.notna()] = "Ping"
    return out


def _prep_sigcap(raw: pd.DataFrame) -> pd.DataFrame:
    """In-place prep before normalisation: strip the tz offset and synthesise
    the per-phone device label."""
    if "timestamp" in raw.columns:
        raw["timestamp"] = (
            raw["timestamp"].astype("string").str.replace(_TZ_OFFSET_RE, "", regex=True)
        )
    raw[SIGCAP_DEVICE_COL] = _sigcap_device(raw)
    return raw


def _sigcap_device(raw: pd.DataFrame) -> pd.Series:
    """A stable, readable per-phone label: ``device_name (short-uuid)``.

    ``uuid`` is unique per phone and never blank in practice; ``device_name``
    (e.g. ``pa2q``) is the friendly Samsung codename.  We key on the uuid so two
    phones never collapse together even when a file's scenario label is wrong,
    but show the codename first because that is what a human recognises.
    """
    idx = raw.index
    uuid = (
        raw["uuid"].astype("string").str.strip()
        if "uuid" in raw.columns
        else pd.Series(pd.NA, index=idx, dtype="string")
    )
    name = (
        raw["device_name"].astype("string").str.strip()
        if "device_name" in raw.columns
        else pd.Series(pd.NA, index=idx, dtype="string")
    )
    short = uuid.str.slice(0, 8)
    out = []
    for nm, sh in zip(name, short):
        nm = None if (nm is pd.NA or nm in (None, "", "nan", "NaN")) else nm
        sh = None if (sh is pd.NA or sh in (None, "", "nan", "NaN")) else sh
        if nm and sh:
            out.append(f"{nm} ({sh})")
        elif sh:
            out.append(sh)
        elif nm:
            out.append(nm)
        else:
            out.append("unknown")
    return pd.Series(out, index=idx, dtype="object")


def _sigcap_stem(path: Path) -> str:
    """File name with the extension, ``.debug_general`` tag and any macOS
    ``" 2"`` duplicate marker stripped - the stable identity of a session."""
    name = path.name
    if name.endswith(".csv"):
        name = name[:-4]
    name = _SIGCAP_SUFFIX_RE.sub("", name)
    return _SIGCAP_DUP_RE.sub("", name.strip()).strip()


def _sigcap_meta(path: Path) -> tuple[str, str]:
    """``(collection, test_type)`` from a SigCap file name, tolerant of both
    naming conventions in the wild:

    * ``..._YC_910_iperf.debug_general.csv``        -> ``("910 iperf", "iperf")``
    * ``2026-08-14-14-23-06[D2C_Static_OpenField_SigcapiperfDL].csv``
      -> ``("D2C Static OpenField SigcapiperfDL", "iperf")``

    A bracketed descriptor is used verbatim; otherwise the location token plus
    the workload, so iperf and HTTP runs at one spot stay separable while still
    grouping by location.
    """
    stem = _sigcap_stem(path)
    low = stem.lower()
    test = "iperf" if "iperf" in low else "http" if "http" in low else "unknown"
    bracket = re.search(r"\[(.+)\]", stem)
    if bracket:
        label = bracket.group(1)
    else:
        tokens = [t for t in re.split(r"[_\s]+", stem) if t]
        loc = ""
        workload = test  # fall back to the bare "iperf"/"http"
        for i, tok in enumerate(tokens):
            # match "iperf", "http" and their direction variants ("iperfDL",
            # "iperfUL", "iperfdl", ...). Keep the FULL token as the workload so
            # a downlink and an uplink run at one spot stay separate collections
            # ("twoPhone2 iperfDL" vs "twoPhone2 iperfUL"), not merged.
            if re.match(r"^(iperf|http)", tok, re.IGNORECASE):
                workload = tok
                if i > 0:
                    loc = tokens[i - 1]
                break
        if not loc and tokens:
            loc = tokens[-1]
        label = workload if (not loc or loc == workload) else f"{loc} {workload}"
    label = re.sub(r"[_\s]+", " ", label).strip()
    return label, test


def _visible(path: Path, root: Path) -> bool:
    """False for anything under a hidden folder or a macOS ``__MACOSX`` folder."""
    parts = path.relative_to(root).parts
    return not any(p.startswith(".") or p == "__MACOSX" for p in parts)


def sigcap_files(sigcap_dir: os.PathLike | str) -> List[Path]:
    """Every SigCap session file (``*.csv``) in the folder or any subfolder.

    Sessions can be grouped into subfolders (e.g. one per campaign). Hidden
    and ``__MACOSX`` folders are ignored. Within one folder, a macOS "keep
    both" duplicate (``... 2.csv``) of a session that is also present under its
    canonical name is skipped. Files are returned sorted by relative path.
    """
    root = Path(sigcap_dir)
    if not root.exists():
        return []

    def _pref(p: Path):
        # lower is better: the canonical name over a " 2" duplicate
        base = p.name[:-4] if p.name.endswith(".csv") else p.name
        base = _SIGCAP_SUFFIX_RE.sub("", base)
        is_dup = 1 if _SIGCAP_DUP_RE.search(base.strip()) else 0
        return (is_dup, p.name)

    chosen: dict = {}
    for p in root.rglob("*.csv"):
        if not (p.is_file() and _visible(p, root)):
            continue
        if _SIGCAP_CELLULAR_RE.search(p.name):
            # per-cell export: no serving-cell or iperf/HTTP/ping columns
            log.info("SigCap %s: .debug_cellular export, not read (needs the "
                     ".debug_general export of the session)", p.name)
            continue
        key = (p.parent.relative_to(root).as_posix(), _sigcap_stem(p).lower())
        chosen.setdefault(key, []).append(p)
    return [min(v, key=_pref) for _, v in sorted(chosen.items())]


def _sigcap_techs_present(path: Path) -> set:
    """Serving technologies that appear in one SigCap file (cheap column read)."""
    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:  # pragma: no cover - unreadable file
        return set()
    routing = ["network_type*", "lte_primary_rsrp_dbm", "nr_first_ss_rsrp_dbm"]
    wanted = [c for c in routing if c in header.columns]
    if not wanted:
        return set()
    try:
        raw = pd.read_csv(path, usecols=wanted, low_memory=False)
    except Exception:  # pragma: no cover
        return set()
    return set(pd.Series(_sigcap_serving_tech(raw)).unique().tolist())


def sigcap_available(sigcap_dir: os.PathLike | str) -> List[str]:
    """SigCap dataset keys with at least one row present, in schema order."""
    techs: set = set()
    for path in sigcap_files(sigcap_dir):
        techs |= _sigcap_techs_present(path)
    return [d.key for d in SIGCAP_DATASETS if d.tech in techs]


def read_sigcap_file(
    path: Path,
    ds: Dataset,
    cache_dir: Optional[os.PathLike | str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Read + normalise one SigCap file for one dataset (LTE or NR rows only)."""
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{ds.key}-{_fingerprint(path, ds)}.parquet"
        if cache_path.exists():
            try:
                return coerce_dtypes(pd.read_parquet(cache_path))
            except Exception:  # pragma: no cover - corrupt cache, just rebuild
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    pass

    header = pd.read_csv(path, nrows=0)
    wanted = [c for c in ds.raw_columns() if c in header.columns]
    raw = pd.read_csv(path, usecols=wanted, low_memory=False)
    raw = _prep_sigcap(raw)

    serving = _sigcap_serving_tech(raw)
    raw = raw[serving == ds.tech]

    collection, test_type = _sigcap_meta(path)
    link = _sigcap_link(raw)
    traffic = _sigcap_traffic(raw)
    frame = normalise(raw, ds, collection=collection, link=link,
                      test_type=test_type, traffic=traffic)

    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(cache_path, index=False)
        except Exception:  # pragma: no cover - caching is best effort
            pass
    return frame


def load_sigcap(
    sigcap_dir: os.PathLike | str,
    dataset_keys: Sequence[str],
    cache_dir: Optional[os.PathLike | str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Read every SigCap file, route each row to its serving tech, and pool."""
    want = set(dataset_keys)
    frames: List[pd.DataFrame] = []
    for path in sigcap_files(sigcap_dir):
        techs = _sigcap_techs_present(path)
        for ds in SIGCAP_DATASETS:
            if ds.key not in want or ds.tech not in techs:
                continue
            try:
                frames.append(read_sigcap_file(path, ds, cache_dir=cache_dir))
            except Exception as exc:  # skip a bad file, keep the rest - but say so
                log.warning("skipping SigCap file %s: %s: %s", path.name, type(exc).__name__, exc)
                continue
    return concat_frames(frames)


def align_d2c_traffic(
    frame: pd.DataFrame,
    ref: Optional[pd.DataFrame],
    tolerance: str = "6s",
) -> pd.DataFrame:
    """Relabel QP ``D2C`` rows' traffic from a time-aligned SigCap reference.

    In the D2C campaign QP runs *idle* (extracting PHY only) while SigCap
    generates the traffic, so QP's own columns and its collection name (which
    even carries typos like ``PingiperULDL``) are not a reliable traffic label.
    ``ref`` is a ``time`` + ``traffic`` table taken from the SigCap rows (the
    ground truth); each QP D2C row inherits the SigCap traffic active within
    ``tolerance`` of its timestamp. A D2C row is one measured on the D2C carrier
    (the ``network`` column), whatever its collection is called. Rows with no
    SigCap match, and every non-D2C or non-QP row, keep the label they had.
    """
    if (frame is None or frame.empty or ref is None or getattr(ref, "empty", True)
            or "network" not in frame.columns or "traffic" not in frame.columns):
        return frame
    prov = frame["provider"].astype("string") if "provider" in frame.columns else None
    is_qp = prov.eq("QP").to_numpy() if prov is not None else np.ones(len(frame), bool)
    is_d2c = frame["network"].astype("string").eq(NETWORK_D2C).fillna(False).to_numpy() & is_qp
    if not is_d2c.any():
        return frame

    r = ref.dropna(subset=["time", "traffic"]).copy()
    r["_time"] = pd.to_datetime(r["time"], errors="coerce")
    r = r.dropna(subset=["_time"]).sort_values("_time")[["_time", "traffic"]]
    r = r.rename(columns={"traffic": "_ref"})
    if r.empty:
        return frame

    out = frame.copy()
    out["traffic"] = out["traffic"].astype("string")  # leave Categorical so we
    # can write SigCap labels that weren't already among QP's categories
    sub = out.loc[is_d2c, ["time"]].copy()
    sub["_idx"] = sub.index
    sub["_time"] = pd.to_datetime(sub["time"], errors="coerce")
    sub = sub.dropna(subset=["_time"]).sort_values("_time")
    if sub.empty:
        return frame

    merged = pd.merge_asof(
        sub, r, on="_time", direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
    matched = merged["_ref"].notna()
    if matched.any():
        idx = merged.loc[matched, "_idx"].to_numpy()
        out.loc[idx, "traffic"] = merged.loc[matched, "_ref"].to_numpy()
    return out
