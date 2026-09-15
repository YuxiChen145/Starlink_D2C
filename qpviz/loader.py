"""Load the QP CSV exports and normalise them onto the canonical schema."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .schema import (
    DATASETS,
    DATASETS_BY_KEY,
    DERIVED_METRIC,
    DIMENSION_COLUMNS,
    METRICS,
    PRISM_DATASETS,
    PRISM_DATASETS_BY_KEY,
    SIGCAP_APP_TPUT_COL,
    SIGCAP_DATASETS,
    SIGCAP_DATASETS_BY_KEY,
    Dataset,
)

DEFAULT_DATA_DIR = os.environ.get("QP_DATA_DIR", "QP CSV")
DEFAULT_PRISM_DIR = os.environ.get("PRISM_DATA_DIR", "PRISM CSV")
DEFAULT_SIGCAP_DIR = os.environ.get("SIGCAP_DATA_DIR", "Sigcap CSV")
DEFAULT_CACHE_DIR = os.environ.get("QP_CACHE_DIR", ".cache/qpviz")

#: bump whenever normalisation changes, so stale parquet caches are ignored
CACHE_VERSION = 7

#: one LTE/NR resource block spans 12 subcarriers
_SUBCARRIERS_PER_RB = 12
#: LTE subframe / NR 15-kHz slot cadence; NR scales with the subcarrier spacing
_LTE_TTIS_PER_SEC = 1000

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
#: is ``Shoreham Court Kitchen static``.
_PRISM_NAME_RE = re.compile(
    r"^\s*\d{4}_\d{2}_\d{2}_\s*(?:LTE|NR)_\d+_(.+)$", re.IGNORECASE
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
)

#: free-text dimensions, always stored as pandas ``string``
_TEXT_DIMS = ("database", "operator", "model", "serial", "ran_config", "cell_type")

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
    """Fill the derived ``spec_eff`` (bps/Hz) and ``tput_per_rb`` (bits/RB) columns.

    Both come from the net throughput and the RBs allocated averaged over *all*
    slots/subframes (so numerator and denominator cover the same window):

        spec_eff    = throughput_bps / (RB_all x 12 x SCS_Hz)          [bps/Hz]
        tput_per_rb = throughput_bps / (RB_all x TTIs_per_second)      [bits/RB]

    ``RB_all`` is the time-averaged occupancy, so ``RB_all x 12 x SCS`` is the
    average occupied bandwidth and the ratio is the spectral efficiency of the
    resource the scheduler actually granted (not the whole carrier). LTE fixes
    SCS at 15 kHz; NR reads it per row. Throughput is kbps in the export.
    """
    if not ds.rb_all_col or "throughput" not in out.columns:
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
    tput_bps = out["throughput"].to_numpy(dtype="float64") * 1000.0

    valid = np.isfinite(rb_all) & (rb_all > 0) & np.isfinite(scs_khz) & (scs_khz > 0)
    rb_bw_hz = _SUBCARRIERS_PER_RB * scs_khz * 1000.0
    ttis = _LTE_TTIS_PER_SEC * (scs_khz / 15.0)  # slots/s: 15 kHz->1000, 30->2000

    with np.errstate(invalid="ignore", divide="ignore"):
        spec = np.where(valid, tput_bps / (rb_all * rb_bw_hz), np.nan)
        per_rb = np.where(valid, tput_bps / (rb_all * ttis), np.nan)
    out["spec_eff"] = spec
    out["tput_per_rb"] = per_rb


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


def normalise(
    raw: pd.DataFrame,
    ds: Dataset,
    collection: Optional[str] = None,
    link=None,
    test_type: Optional[str] = None,
) -> pd.DataFrame:
    """Map one raw CSV frame onto the canonical schema.

    ``collection`` overrides the collection column - PRISM and SigCap have no
    such column, so theirs comes from the file name instead.  ``link`` overrides
    the per-dataset direction: SigCap passes a per-row Series (DL/UL from
    ``iperf_direction``) since one session mixes downlink and uplink tests.
    ``test_type`` tags the workload (``iperf``/``http``) for SigCap rows.
    """
    out = pd.DataFrame(index=raw.index)

    for canon, source_col in ds.dims.items():
        col = raw[source_col] if source_col in raw.columns else pd.Series(
            np.nan, index=raw.index
        )
        if canon == "time":
            out["time"] = parse_time(col)
        elif canon in ("lat", "lon"):
            out[canon] = _to_numeric(col)
        elif canon == "enb_sector":
            enb, sector = split_enb_sector(col)
            out["enb"] = enb.astype("Int64")
            out["sector"] = sector.astype("Int32")
        elif canon in _INT_DIMS:
            out[canon] = _to_numeric(col).astype(_INT_DIMS[canon])
        elif canon == "band":
            out["band"] = [normalise_band(v, ds.tech) for v in col]
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

    # NR gives the gNB-Id and the full cell identity but no explicit sector;
    # the cell-local id is what corresponds to LTE's sector.
    if "sector" not in out.columns and {"enb", "cell_id"} <= set(out.columns):
        local = out["cell_id"].astype("Float64") - out["enb"].astype("Float64") * _NR_CELL_ID_SHIFT
        local = local.where((local >= 0) & (local < _NR_CELL_ID_SHIFT))
        out["sector"] = local.astype("Int32")

    out["tech"] = ds.tech
    out["link"] = link if link is not None else ds.link
    out["source"] = ds.key
    out["source_label"] = ds.label
    out["provider"] = ds.provider
    if collection is not None:
        out["collection"] = collection
    if test_type is not None:
        out["test_type"] = test_type
    if "lat" not in out.columns:
        out["lat"] = np.nan
        out["lon"] = np.nan
    for col in DIMENSION_COLUMNS:
        if col not in out.columns:
            dtype = _INT_DIMS.get(col, "string")
            out[col] = pd.Series(pd.NA, index=out.index, dtype=dtype)

    metric_cols = list(ds.metrics)
    if metric_cols:
        out = out[out[metric_cols].notna().any(axis=1)]

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


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    raw = f"v{CACHE_VERSION}:{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _resolve_csv(root: Path, filename: str) -> Optional[Path]:
    """Locate an export as either ``name.csv`` or its committed ``name.csv.gz``.

    The raw CSVs stay local (git-ignored); gzipped copies travel in the repo so
    a plain clone or a cloud deploy has data. pandas reads ``.gz`` by extension,
    so only the path lookup needs to know about it. A local raw CSV wins over a
    gzip of the same name.
    """
    p = root / filename
    if p.exists():
        return p
    gz = root / (filename + ".gz")
    if gz.exists():
        return gz
    return None


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
        cache_path = Path(cache_dir) / f"{ds.key}-{_fingerprint(path)}.parquet"
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
    """Datasets whose CSV (or gzipped CSV) is present in ``data_dir``."""
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
    if name.endswith(".gz"):
        name = name[:-3]
    if name.endswith(".csv"):
        name = name[:-4]
    stem = name.strip()
    match = _PRISM_NAME_RE.match(stem)
    core = match.group(1) if match else stem
    return core.replace("_", " ").strip()


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
    """Every PRISM scan in the folder (``.csv`` or gzipped ``.csv.gz``).

    When both a raw ``.csv`` and its ``.csv.gz`` exist, the raw one wins.
    """
    root = Path(prism_dir)
    if not root.exists():
        return []
    chosen: dict = {}
    for p in sorted(root.glob("*.csv")) + sorted(root.glob("*.csv.gz")):
        if not p.is_file():
            continue
        base = p.name[:-3] if p.name.endswith(".csv.gz") else p.name
        chosen.setdefault(base, p)  # .csv globbed first -> preferred
    return [chosen[b] for b in sorted(chosen)]


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
        cache_path = Path(cache_dir) / f"{ds.key}-{_fingerprint(path)}.parquet"
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
    frame = normalise(raw, ds, collection=prism_collection(path))

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
        except Exception:  # pragma: no cover - skip a bad file, keep the rest
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
# to the LTE or NR dataset by its *serving* technology, so a row's app-layer
# throughput rides with the radio that was actually serving and is never
# double-counted.  The two app-throughput columns (iperf / HTTP) are coalesced
# into one before normalisation, and CQI's 2147483647 "unknown" sentinel is
# scrubbed to NaN.

#: SigCap logs Android's Integer.MAX_VALUE for an unavailable integer reading
_SIGCAP_INT_SENTINEL = 2147483647
#: integer-valued source columns that use the sentinel
_SIGCAP_SENTINEL_COLS = ("lte_primary_cqi",)
#: trailing tz offset on an ISO-8601 stamp, e.g. ``...930-0400`` / ``...+05:30``
_TZ_OFFSET_RE = re.compile(r"([+-]\d{2}:?\d{2})\s*$")
_SIGCAP_SUFFIX_RE = re.compile(r"\s*\.debug[_ ]?general\s*$", re.IGNORECASE)
#: macOS "keep both" duplicates append a trailing " 2" / " 3" before the ext
_SIGCAP_DUP_RE = re.compile(r"\s+\d+$")


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
    """Per-row DL/UL direction from ``iperf_direction`` (HTTP is a download)."""
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
    link[(link == "-") & http.notna()] = "DL"
    return link


def _prep_sigcap(raw: pd.DataFrame) -> pd.DataFrame:
    """In-place prep before normalisation: strip tz, scrub sentinels, coalesce
    the iperf/HTTP throughput columns into one app-throughput column."""
    if "timestamp" in raw.columns:
        raw["timestamp"] = (
            raw["timestamp"].astype("string").str.replace(_TZ_OFFSET_RE, "", regex=True)
        )
    for col in _SIGCAP_SENTINEL_COLS:
        if col in raw.columns:
            v = _to_numeric(raw[col])
            raw[col] = v.mask(v.abs() >= _SIGCAP_INT_SENTINEL)
    iperf = (
        _to_numeric(raw["iperf_tput_mean_mbps"])
        if "iperf_tput_mean_mbps" in raw.columns
        else pd.Series(np.nan, index=raw.index)
    )
    http = (
        _to_numeric(raw["http_tput_mean_mbps"])
        if "http_tput_mean_mbps" in raw.columns
        else pd.Series(np.nan, index=raw.index)
    )
    raw[SIGCAP_APP_TPUT_COL] = iperf.where(iperf.notna(), http)
    return raw


def _sigcap_stem(path: Path) -> str:
    """File name with the extension, ``.debug_general`` tag and any macOS
    ``" 2"`` duplicate marker stripped - the stable identity of a session."""
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
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
        for i, tok in enumerate(tokens):
            if tok.lower() in ("iperf", "http"):
                if i > 0:
                    loc = tokens[i - 1]
                break
        if not loc and tokens:
            loc = tokens[-1]
        label = f"{loc} {test}"
    label = re.sub(r"[_\s]+", " ", label).strip()
    return label, test


def sigcap_files(sigcap_dir: os.PathLike | str) -> List[Path]:
    """Every SigCap session file in the folder (``.csv`` or ``.csv.gz``).

    When both a raw ``.csv`` and its ``.csv.gz`` exist, the raw one wins.
    """
    root = Path(sigcap_dir)
    if not root.exists():
        return []

    def _pref(p: Path):
        # lower is better: prefer a raw .csv over its .csv.gz, and the canonical
        # name over a macOS "keep both" duplicate ("... 2.csv")
        base = p.name[:-3] if p.name.endswith(".csv.gz") else p.name
        base = base[:-4] if base.endswith(".csv") else base
        base = _SIGCAP_SUFFIX_RE.sub("", base)
        is_dup = 1 if _SIGCAP_DUP_RE.search(base.strip()) else 0
        is_gz = 1 if p.name.endswith(".csv.gz") else 0
        return (is_dup, is_gz, p.name)

    chosen: dict = {}
    for p in root.glob("*.csv"):
        if not p.is_file():
            continue
        chosen.setdefault(_sigcap_stem(p).lower(), []).append(p)
    for p in root.glob("*.csv.gz"):
        if not p.is_file():
            continue
        chosen.setdefault(_sigcap_stem(p).lower(), []).append(p)
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
        cache_path = Path(cache_dir) / f"{ds.key}-{_fingerprint(path)}.parquet"
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
    frame = normalise(raw, ds, collection=collection, link=link, test_type=test_type)

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
            except Exception:  # pragma: no cover - skip a bad file, keep the rest
                continue
    return concat_frames(frames)
