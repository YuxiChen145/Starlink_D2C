"""Cell-identity analysis: who is serving us, and for how long.

Terrestrial cells are effectively static at a fixed spot; the Starlink D2C
cells are not - standing still on D2C touches dozens of PCIs, eNB-Ids and
sectors in half an hour.  The helpers here turn the raw per-sample identity
columns into the two things needed to see that: **dwell runs** (one row per
continuous stretch on a single identity) and a **hierarchy** of dwell time
across eNB / sector / PCI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

#: identity columns the UI offers, in drill-down order
IDENTITY_LEVELS: Dict[str, str] = {
    "enb": "eNB / gNB-Id",
    "sector": "Sector Id",
    "pci": "PCI",
    "ssb": "SSB Index",
}

#: the cell that is actually serving the connection; SCell rows describe
#: aggregated carriers that come and go independently and would otherwise
#: inflate every churn number
SERVING_CELL_TYPES = ("MCG PCell", "SCG PSCell")

#: a bigger gap than this means the log was paused, not that a cell was held
DEFAULT_GAP_SECONDS = 60.0


def serving_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only the serving cell (PCell / PSCell) rows."""
    if frame.empty or "cell_type" not in frame.columns:
        return frame
    keep = frame["cell_type"].astype("string").isin(SERVING_CELL_TYPES)
    return frame.loc[keep.fillna(False).to_numpy()]


def available_levels(frame: pd.DataFrame) -> List[str]:
    """Identity levels that actually carry values in ``frame``."""
    if frame.empty:
        return []
    return [
        level
        for level in IDENTITY_LEVELS
        if level in frame.columns and frame[level].notna().any()
    ]


def label_for(level: str) -> str:
    return IDENTITY_LEVELS.get(level, level)


# --------------------------------------------------------------------------- #
# Dwell
# --------------------------------------------------------------------------- #


def _effective_dt(times: pd.Series, gap_seconds: float) -> np.ndarray:
    """Seconds each sample 'owns' - the step to the next sample, with pauses
    and the final sample charged the collection's median step instead.

    Uses pandas' own timedelta conversion rather than the raw int64: the
    datetime resolution is nanoseconds on pandas 2 but microseconds on
    pandas 3, so dividing the integer view by a hard-coded 1e9 is wrong on
    one of them.
    """
    if len(times) == 0:
        return np.array([])
    step_to_next = times.diff().shift(-1).dt.total_seconds().to_numpy(dtype="float64")
    positive = step_to_next[np.isfinite(step_to_next) & (step_to_next > 0)]
    typical = float(np.median(positive)) if positive.size else 1.0
    out = np.where(
        np.isfinite(step_to_next) & (step_to_next > 0) & (step_to_next <= gap_seconds),
        step_to_next,
        typical,
    )
    return out


def dwell_runs(
    frame: pd.DataFrame,
    level: str,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> pd.DataFrame:
    """One row per continuous stretch on a single identity.

    Columns: ``collection, value, start, end, duration_s, samples,
    elapsed_start, elapsed_end, order``.  ``duration_s`` is the time that cell
    actually owned, and the elapsed values are *active* seconds within the
    collection - time the log was paused is excluded - so several collections
    can share one axis without a paused one stretching it.
    """
    empty = pd.DataFrame(
        columns=[
            "collection", "value", "start", "end", "duration_s",
            "samples", "elapsed_start", "elapsed_end", "order",
        ]
    )
    if frame.empty or level not in frame.columns or "time" not in frame.columns:
        return empty

    work = frame[["time", "collection", level]].copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work.dropna(subset=["time", level])
    if work.empty:
        return empty

    rows = []
    for collection, group in work.groupby("collection", observed=True, sort=True):
        group = group.sort_values("time")
        times = group["time"]
        values = group[level].to_numpy()
        dt = _effective_dt(times, gap_seconds)
        # Elapsed is *active* time, not wall-clock since the first sample: a
        # collection logged across two days would otherwise stretch the axis
        # by the hours it spent switched off.
        active = np.concatenate([[0.0], np.cumsum(dt)[:-1]]) if len(dt) else dt

        since_previous = times.diff().dt.total_seconds().to_numpy()
        changed = np.ones(len(group), dtype=bool)
        changed[1:] = (values[1:] != values[:-1]) | (
            np.nan_to_num(since_previous[1:], nan=np.inf) > gap_seconds
        )
        run_id = np.cumsum(changed)

        detail = pd.DataFrame(
            {
                "run": run_id,
                "value": values,
                "time": times.to_numpy(),
                "dt": dt,
                "active": active,
            }
        )
        for _, run in detail.groupby("run", sort=True):
            start = pd.Timestamp(run["time"].iloc[0])
            duration = float(run["dt"].sum())
            elapsed_start = float(run["active"].iloc[0])
            rows.append(
                {
                    "collection": collection,
                    "value": run["value"].iloc[0],
                    "start": start,
                    "end": start + pd.Timedelta(seconds=duration),
                    "duration_s": duration,
                    "samples": int(len(run)),
                    "elapsed_start": elapsed_start,
                    "elapsed_end": elapsed_start + duration,
                }
            )

    runs = pd.DataFrame(rows)
    if runs.empty:
        return empty
    runs = runs.sort_values(["collection", "start"]).reset_index(drop=True)
    runs["order"] = runs.groupby("collection", observed=True).cumcount()
    return runs


def order_identities(
    runs: pd.DataFrame, by: str = "first seen", limit: Optional[int] = None
) -> List:
    """The identity values to draw, in axis order.

    ``first seen`` produces the staircase that makes a satellite pass obvious;
    ``dwell`` and ``value`` are the alternatives.
    """
    if runs.empty:
        return []
    grouped = runs.groupby("value", observed=True)
    if by == "dwell":
        ranked = grouped["duration_s"].sum().sort_values(ascending=False)
    elif by == "value":
        ranked = pd.Series(0, index=sorted(grouped.groups)).sort_index()
    else:
        ranked = grouped["start"].min().sort_values()
    values = list(ranked.index)
    if limit is not None and len(values) > limit:
        # keep the longest-dwelling ones, then restore the requested order
        keep = set(grouped["duration_s"].sum().sort_values(ascending=False).index[:limit])
        values = [v for v in values if v in keep]
    return values


# --------------------------------------------------------------------------- #
# Hierarchy
# --------------------------------------------------------------------------- #


def hierarchy(
    frame: pd.DataFrame,
    levels: Sequence[str],
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> pd.DataFrame:
    """Dwell seconds and sample counts per ``levels`` path.

    Rows missing any level in the path are dropped - a partial path cannot be
    placed in the tree.
    """
    levels = [l for l in levels if l in frame.columns]
    if frame.empty or not levels:
        return pd.DataFrame(columns=list(levels) + ["dwell_s", "samples"])

    work = frame[["time", "collection", *levels]].copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work.dropna(subset=["time", *levels])
    if work.empty:
        return pd.DataFrame(columns=list(levels) + ["dwell_s", "samples"])

    parts = []
    for _, group in work.groupby("collection", observed=True, sort=True):
        group = group.sort_values("time").copy()
        group["dwell_s"] = _effective_dt(group["time"], gap_seconds)
        parts.append(group)
    work = pd.concat(parts, ignore_index=True)

    out = (
        work.groupby(levels, observed=True)
        .agg(dwell_s=("dwell_s", "sum"), samples=("time", "size"))
        .reset_index()
        .sort_values("dwell_s", ascending=False)
        .reset_index(drop=True)
    )
    return out
