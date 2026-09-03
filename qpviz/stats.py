"""ECDF computation and grouped summary statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: dimensions the CDF curves may be split by
GROUP_DIMENSIONS: Dict[str, str] = {
    "collection": "Collection",
    "provider": "Measurement source",
    "pci": "PCI",
    "freq": "EARFCN / NR-ARFCN",
    "band": "Band",
    "tech": "Technology",
    "link": "Direction",
    "source_label": "Data source",
    "cell_type": "Cell type",
}


@dataclass
class Curve:
    """One ECDF ready to plot."""

    key: Tuple
    name: str
    x: np.ndarray
    y: np.ndarray
    n: int


def ecdf(values: Sequence[float], max_points: Optional[int] = 2000):
    """Empirical CDF.

    Returns ``(x, y)`` with ``y_i = i / n`` for the i-th smallest value, i.e.
    ``y`` is the fraction of samples <= ``x``.  When ``max_points`` is set and
    the sample is larger, the curve is thinned to that many evenly spaced
    points (endpoints always kept) - the drawn line is unchanged at screen
    resolution but plotting stays fast.
    """
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.array([]), np.array([])
    arr.sort()
    n = arr.size
    y = np.arange(1, n + 1, dtype="float64") / n
    if max_points and n > max_points:
        idx = np.unique(
            np.round(np.linspace(0, n - 1, max_points)).astype(int)
        )
        arr, y = arr[idx], y[idx]
    return arr, y


def _format_value(dimension: str, value, freq_prefix: str = "ARFCN") -> str:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return f"{GROUP_DIMENSIONS.get(dimension, dimension)} n/a"
    if dimension == "pci":
        return f"PCI {int(value)}"
    if dimension == "freq":
        return f"{freq_prefix} {int(value)}"
    if dimension == "band":
        return f"Band {value}"
    return str(value)


def group_label(
    dimensions: Sequence[str], values: Sequence, freq_prefix: str = "ARFCN"
) -> str:
    return " · ".join(
        _format_value(d, v, freq_prefix) for d, v in zip(dimensions, values)
    )


def _grouped(frame: pd.DataFrame, dimensions: Sequence[str]):
    if not dimensions:
        yield (), frame
        return
    grouped = frame.groupby(list(dimensions), dropna=False, observed=True, sort=True)
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        yield key, sub


def cdf_curves(
    frame: pd.DataFrame,
    metric: str,
    group_by: Sequence[str] = ("pci",),
    min_samples: int = 1,
    max_points: Optional[int] = 2000,
    freq_prefix: str = "ARFCN",
) -> List[Curve]:
    """Build one ECDF per group, largest sample first."""
    if frame.empty or metric not in frame.columns:
        return []

    dimensions = [d for d in group_by if d in frame.columns]
    values = pd.to_numeric(frame[metric], errors="coerce")
    work = frame.loc[values.notna().to_numpy()].copy()
    work[metric] = values[values.notna()]
    if work.empty:
        return []

    curves: List[Curve] = []
    for key, sub in _grouped(work, dimensions):
        series = sub[metric].to_numpy(dtype="float64")
        if series.size < max(1, min_samples):
            continue
        x, y = ecdf(series, max_points=max_points)
        if x.size == 0:
            continue
        name = group_label(dimensions, key, freq_prefix) if dimensions else "All data"
        curves.append(Curve(key=key, name=name, x=x, y=y, n=int(series.size)))

    curves.sort(key=lambda c: (-c.n, c.name))
    return curves


PERCENTILES = (5, 10, 50, 90, 95)


def summary_table(
    frame: pd.DataFrame,
    metric: str,
    group_by: Sequence[str] = ("pci",),
    min_samples: int = 1,
    freq_prefix: str = "ARFCN",
) -> pd.DataFrame:
    """Per-group descriptive statistics for ``metric``."""
    if frame.empty or metric not in frame.columns:
        return pd.DataFrame()

    dimensions = [d for d in group_by if d in frame.columns]
    values = pd.to_numeric(frame[metric], errors="coerce")
    work = frame.loc[values.notna().to_numpy()].copy()
    work[metric] = values[values.notna()]
    if work.empty:
        return pd.DataFrame()

    rows = []
    for key, sub in _grouped(work, dimensions):
        series = sub[metric].to_numpy(dtype="float64")
        if series.size < max(1, min_samples):
            continue
        row = {
            "Group": group_label(dimensions, key, freq_prefix) if dimensions else "All data",
            "Samples": int(series.size),
            "Mean": float(np.mean(series)),
            "Std": float(np.std(series, ddof=1)) if series.size > 1 else float("nan"),
            "Min": float(np.min(series)),
        }
        for p in PERCENTILES:
            row[f"P{p}"] = float(np.percentile(series, p))
        row["Max"] = float(np.max(series))
        rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("Samples", ascending=False).reset_index(drop=True)
    return table
