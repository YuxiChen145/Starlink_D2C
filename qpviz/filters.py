"""Filtering: collections, PCI, EARFCN/ARFCN, band and a drawn map region."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

LatLon = Tuple[float, float]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def points_in_polygon(
    lat: Sequence[float], lon: Sequence[float], polygon: Sequence[LatLon]
) -> np.ndarray:
    """Vectorised even-odd (ray casting) point-in-polygon test.

    ``polygon`` is a sequence of ``(lat, lon)`` vertices; it may or may not
    repeat the first vertex at the end.  Points with NaN coordinates are False.
    """
    lat_arr = np.asarray(lat, dtype="float64")
    lon_arr = np.asarray(lon, dtype="float64")
    inside = np.zeros(lat_arr.shape, dtype=bool)

    verts = [tuple(map(float, v)) for v in polygon]
    if len(verts) > 1 and verts[0] == verts[-1]:
        verts = verts[:-1]
    if len(verts) < 3:
        return inside

    valid = np.isfinite(lat_arr) & np.isfinite(lon_arr)
    x = lon_arr
    y = lat_arr

    n = len(verts)
    for i in range(n):
        y1, x1 = verts[i]
        y2, x2 = verts[(i + 1) % n]
        if y1 == y2:
            continue
        straddles = (y1 > y) != (y2 > y)
        with np.errstate(invalid="ignore", divide="ignore"):
            x_cross = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= straddles & (x < x_cross)

    return inside & valid


def bbox_polygon(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> List[LatLon]:
    return [
        (lat_min, lon_min),
        (lat_min, lon_max),
        (lat_max, lon_max),
        (lat_max, lon_min),
    ]


# --------------------------------------------------------------------------- #
# Filter set
# --------------------------------------------------------------------------- #


@dataclass
class Filters:
    """A ``None`` (or empty) list means "do not filter on this dimension"."""

    collections: Optional[Sequence[str]] = None
    bands: Optional[Sequence[str]] = None
    freqs: Optional[Sequence[int]] = None
    pcis: Optional[Sequence[int]] = None
    techs: Optional[Sequence[str]] = None
    links: Optional[Sequence[str]] = None
    traffic: Optional[Sequence[str]] = None
    sources: Optional[Sequence[str]] = None
    #: list of polygons, each a list of ``(lat, lon)`` vertices; OR-ed together
    regions: List[List[LatLon]] = field(default_factory=list)
    time_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    #: drop rows whose ``metric`` value is NaN
    metric: Optional[str] = None
    #: columns that must be non-null (e.g. drop rows with an unknown PCI)
    require: Sequence[str] = ()
    #: keep rows without GPS even when a region is drawn
    keep_ungeolocated: bool = False

    def describe(self) -> str:
        bits = []
        for name, values in (
            ("collection", self.collections),
            ("band", self.bands),
            ("freq", self.freqs),
            ("PCI", self.pcis),
        ):
            if values:
                bits.append(f"{len(values)} {name}(s)")
        if self.regions:
            bits.append(f"{len(self.regions)} region(s)")
        return ", ".join(bits) if bits else "no filters"


def _isin(series: pd.Series, values: Sequence) -> pd.Series:
    return pd.Series(series).isin(list(values)).to_numpy()


def apply_filters(frame: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    """Apply ``filters`` to a canonical frame and return the surviving rows.

    When every row survives, ``frame`` itself comes back rather than a copy, so
    treat the result as read-only. The sidebar applies one filter per step, and
    copying the full frame at each step held several copies of it at once.
    """
    if frame.empty:
        return frame

    mask = np.ones(len(frame), dtype=bool)

    for column, values in (
        ("collection", filters.collections),
        ("traffic", filters.traffic),
        ("band", filters.bands),
        ("freq", filters.freqs),
        ("pci", filters.pcis),
        ("tech", filters.techs),
        ("link", filters.links),
        ("source", filters.sources),
    ):
        if values:
            mask &= _isin(frame[column], values)

    if filters.metric and filters.metric in frame.columns:
        mask &= frame[filters.metric].notna().to_numpy()

    for column in filters.require or ():
        if column in frame.columns:
            mask &= frame[column].notna().to_numpy()

    if filters.time_range and "time" in frame.columns:
        start, end = filters.time_range
        times = pd.to_datetime(frame["time"], errors="coerce")
        mask &= ((times >= start) & (times <= end)).to_numpy()

    if filters.regions:
        lat = frame["lat"].to_numpy(dtype="float64", na_value=np.nan)
        lon = frame["lon"].to_numpy(dtype="float64", na_value=np.nan)
        region_mask = np.zeros(len(frame), dtype=bool)
        for polygon in filters.regions:
            region_mask |= points_in_polygon(lat, lon, polygon)
        if filters.keep_ungeolocated:
            region_mask |= ~(np.isfinite(lat) & np.isfinite(lon))
        mask &= region_mask

    if mask.all():
        return frame
    return frame.loc[mask]


def shared_values(
    frame: pd.DataFrame, column: str, group: str = "collection"
) -> List:
    """Values of ``column`` that appear in **every** group.

    The union of PCIs across several collections answers "what did we see?";
    the intersection answers "which cells can I compare like for like?" - only
    those appear in all of them.

    A group that carries no value at all for ``column`` is skipped rather than
    emptying the result, so one PCI-less collection cannot silently zero the
    shared set.
    """
    if frame.empty or column not in frame.columns or group not in frame.columns:
        return []

    sets = []
    for _, sub in frame.groupby(group, observed=True, sort=True):
        values = set(pd.Series(sub[column]).dropna().unique().tolist())
        if values:
            sets.append(values)
    if not sets:
        return []

    common = set.intersection(*sets)
    try:
        return sorted(common)
    except TypeError:  # pragma: no cover - mixed types
        return sorted(common, key=str)


def geolocated(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry usable GPS coordinates."""
    if frame.empty:
        return frame
    lat = pd.to_numeric(frame["lat"], errors="coerce")
    lon = pd.to_numeric(frame["lon"], errors="coerce")
    keep = lat.notna() & lon.notna() & (lat.abs() > 1e-9) & (lon.abs() > 1e-9)
    return frame.loc[keep.to_numpy()]


def data_windows(times, gap_seconds: float = 300.0):
    """Contiguous time spans that actually contain samples.

    The measurements come in short sessions on different days, so the raw
    min→max span is mostly empty. This splits the timestamps wherever there is a
    gap longer than ``gap_seconds`` and returns one ``(start, end, count)`` per
    session, chronologically — what the time-window control shows so you can land
    a window on real data instead of a dead gap.
    """
    t = pd.to_datetime(pd.Series(list(times)), errors="coerce").dropna().sort_values()
    if t.empty:
        return []
    gap = t.diff() > pd.Timedelta(seconds=gap_seconds)
    group = gap.cumsum()
    out = []
    for _, g in t.groupby(group):
        out.append((g.iloc[0], g.iloc[-1], int(len(g))))
    return out
