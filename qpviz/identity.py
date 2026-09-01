"""Cell-identity analysis: who is serving us, and for how long.

Terrestrial cells are effectively static at a fixed spot; the Starlink D2C
cells are not - standing still on D2C touches dozens of PCIs, eNB-Ids and
sectors in half an hour.  The helpers here turn the raw per-sample identity
columns into the two things needed to see that: **dwell runs** (one row per
continuous stretch on a single identity) and a **hierarchy** of dwell time
across eNB / sector / PCI.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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


# --------------------------------------------------------------------------- #
# Fan-out: how many children each parent carries
# --------------------------------------------------------------------------- #


def relationships(levels: Sequence[str]) -> List[Tuple[Tuple[str, ...], str]]:
    """The parent/child pairs implied by a level path.

    Each adjacent pair, with the parent scoped by the whole prefix above it
    (a sector id only means something inside its eNB), plus the outermost to
    the innermost when the path is deeper than two.
    """
    levels = list(levels)
    pairs = [(tuple(levels[: i + 1]), levels[i + 1]) for i in range(len(levels) - 1)]
    if len(levels) > 2:
        pairs.append(((levels[0],), levels[-1]))
    return pairs


def relationship_label(parent_levels: Sequence[str], child_level: str) -> str:
    return f"{label_for(child_level)} per {label_for(parent_levels[-1])}"


def fan_out_counts(
    frame: pd.DataFrame,
    levels: Sequence[str],
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """One row per parent: how many distinct children it carries.

    Columns: ``relationship, group, children``.  ``group_by`` splits the
    population (by technology, operator, collection, ...); without it
    everything lands in one group called "All".
    """
    empty = pd.DataFrame(columns=["relationship", "group", "children"])
    levels = [l for l in levels if l in frame.columns]
    if frame.empty or len(levels) < 2:
        return empty

    rows = []
    for parent_levels, child_level in relationships(levels):
        needed = list(dict.fromkeys([*parent_levels, child_level]))
        keys = list(parent_levels)
        if group_by and group_by in frame.columns:
            needed = list(dict.fromkeys([group_by, *needed]))
            keys = [group_by, *keys]
        work = frame[needed].dropna()
        if work.empty:
            continue
        counts = work.groupby(keys, observed=True)[child_level].nunique()
        if counts.empty:
            continue
        group = (
            counts.index.get_level_values(0).astype(str)
            if group_by and group_by in frame.columns
            else "All"
        )
        rows.append(
            pd.DataFrame(
                {
                    "relationship": relationship_label(parent_levels, child_level),
                    "group": group,
                    "children": counts.to_numpy(dtype="int64"),
                }
            )
        )

    if not rows:
        return empty
    out = pd.concat(rows, ignore_index=True)
    # Keep tree order (outer pair first), not alphabetical: "Sector per eNB"
    # is the relationship worth looking at, and it should lead.
    order = list(dict.fromkeys(out["relationship"]))
    out["relationship"] = pd.Categorical(out["relationship"], categories=order, ordered=True)
    return out


FAN_OUT_PERCENTILES = (50, 90)


def fan_out_summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Descriptive statistics per relationship (and group)."""
    if counts.empty:
        return pd.DataFrame()
    out = (
        counts.groupby(["relationship", "group"], observed=True)["children"]
        .agg(
            Parents="size",
            Children="sum",
            Min="min",
            Median="median",
            Mean="mean",
            Max="max",
        )
        .reset_index()
    )
    p90 = (
        counts.groupby(["relationship", "group"], observed=True)["children"]
        .quantile(0.9)
        .reset_index(name="P90")
    )
    out = out.merge(p90, on=["relationship", "group"])
    out = out.rename(columns={"relationship": "Relationship", "group": "Group"})
    columns = ["Relationship", "Group", "Parents", "Children", "Min", "Median", "P90", "Max", "Mean"]
    out = out.sort_values(["Relationship", "Group"]).reset_index(drop=True)
    out["Relationship"] = out["Relationship"].astype(str)
    return out[columns]


def fan_out_distribution(counts: pd.DataFrame) -> pd.DataFrame:
    """How many parents carry exactly N children.

    Columns: ``relationship, group, children, parents, share``.
    """
    if counts.empty:
        return pd.DataFrame(columns=["relationship", "group", "children", "parents", "share"])
    out = (
        counts.groupby(["relationship", "group", "children"], observed=True)
        .size()
        .reset_index(name="parents")
    )
    totals = out.groupby(["relationship", "group"], observed=True)["parents"].transform("sum")
    out["share"] = out["parents"] / totals
    return out.sort_values(["relationship", "group", "children"]).reset_index(drop=True)


def relationship_order(counts: pd.DataFrame) -> List[str]:
    """Relationship labels in tree order (outermost pair first)."""
    if counts.empty or "relationship" not in counts.columns:
        return []
    column = counts["relationship"]
    if isinstance(column.dtype, pd.CategoricalDtype):
        return [c for c in column.cat.categories if c in set(column)]
    return list(dict.fromkeys(column))
