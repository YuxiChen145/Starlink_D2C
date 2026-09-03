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
    Dataset,
)

DEFAULT_DATA_DIR = os.environ.get("QP_DATA_DIR", "QP CSV")
DEFAULT_PRISM_DIR = os.environ.get("PRISM_DATA_DIR", "PRISM CSV")
DEFAULT_CACHE_DIR = os.environ.get("QP_CACHE_DIR", ".cache/qpviz")

#: bump whenever normalisation changes, so stale parquet caches are ignored
CACHE_VERSION = 6

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
    raw: pd.DataFrame, ds: Dataset, collection: Optional[str] = None
) -> pd.DataFrame:
    """Map one raw CSV frame onto the canonical schema.

    ``collection`` overrides the collection column - PRISM has no such column,
    so its collection comes from the file name instead.
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
    out["link"] = ds.link
    out["source"] = ds.key
    out["source_label"] = ds.label
    out["provider"] = ds.provider
    if collection is not None:
        out["collection"] = collection
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
