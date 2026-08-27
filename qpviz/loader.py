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
    DIMENSION_COLUMNS,
    METRICS,
    Dataset,
)

DEFAULT_DATA_DIR = os.environ.get("QP_DATA_DIR", "QP CSV")
DEFAULT_CACHE_DIR = os.environ.get("QP_CACHE_DIR", ".cache/qpviz")

#: bump whenever normalisation changes, so stale parquet caches are ignored
CACHE_VERSION = 3

_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")

_BAND_RE = re.compile(r"(\d+)\s*$")

#: columns stored as pandas categoricals (kept object-backed so the parquet
#: cache round-trips to an identical dtype)
_CATEGORICAL = ("collection", "band", "tech", "link", "source", "source_label")

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


def normalise(raw: pd.DataFrame, ds: Dataset) -> pd.DataFrame:
    """Map one raw CSV frame onto the canonical schema."""
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
        out[canon] = (
            _to_numeric(raw[source_col]) if source_col in raw.columns else np.nan
        )

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


def read_dataset(
    data_dir: os.PathLike | str,
    ds: Dataset,
    cache_dir: Optional[os.PathLike | str] = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Read + normalise one dataset, using a parquet cache when available."""
    path = Path(data_dir) / ds.filename
    if not path.exists():
        raise FileNotFoundError(path)

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
    """Datasets whose CSV is actually present in ``data_dir``."""
    root = Path(data_dir)
    return [d for d in DATASETS if (root / d.filename).exists()]


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
    frames = aligned
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for col in _CATEGORICAL:
        combined[col] = combined[col].astype("object").astype("category")
    return combined


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
