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

#: identity columns the UI offers, in drill-down order (for a 5G NR cell the
#: "eNB ID" is its gNB ID, and the sector its cell-local id)
IDENTITY_LEVELS: Dict[str, str] = {
    "enb": "eNB ID",
    "sector": "Sector ID",
    "pci": "PCI",
}

#: the one path the Hierarchy and Structure views draw
TREE_LEVELS: Tuple[str, ...] = ("enb", "sector", "pci")

#: how the eNBs of the per-eNB summary can be ordered
ENB_ORDERS: Dict[str, str] = {
    "sectors": "Number of sectors",
    "enb": "eNB ID",
}

_SINGULAR = {"enb": "eNB", "sector": "sector", "pci": "PCI"}
_PLURAL = {"enb": "eNBs", "sector": "sectors", "pci": "PCIs"}

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
# Serving-cell change timeline: one segment per Cell Id, coloured by whether the
# eNB changed too
# --------------------------------------------------------------------------- #


def serving_segments(
    frame: pd.DataFrame,
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> pd.DataFrame:
    """One row per continuous stretch on a single **Cell Id** (beam/sector).

    A new segment begins whenever the Cell Id changes; the ``change`` column says
    what kind of change it was, by comparing the eNB to the previous segment:

    * ``start`` - the first segment of the collection,
    * ``enb``   - the eNB changed too (a site / satellite change),
    * ``cell``  - same eNB, new Cell Id (a beam / sector change).

    Columns: ``collection, enb, cell_id, sector, pci, start, elapsed_start,
    elapsed_end, duration_s, samples, change``. Elapsed values are *active*
    seconds within the collection (paused time excluded), so every panel starts
    at zero. When no Cell Id is decoded it falls back to segmenting on
    ``eNB + sector``.
    """
    carry = [c for c in ("enb", "cell_id", "sector", "pci") if c in frame.columns]
    cols = ["collection", *carry, "start", "elapsed_start", "elapsed_end",
            "duration_s", "samples", "change"]
    empty = pd.DataFrame(columns=cols)
    if frame.empty or "time" not in frame.columns or "enb" not in frame.columns:
        return empty
    if "cell_id" in frame.columns:
        key_cols = ["cell_id"]
    else:
        key_cols = [c for c in ("enb", "sector") if c in frame.columns]
    if not key_cols:
        return empty

    work = frame[["time", "collection", *carry]].copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work.dropna(subset=["time", "enb"])
    if work.empty:
        return empty

    rows = []
    for collection, group in work.groupby("collection", observed=True, sort=True):
        group = group.sort_values("time")
        times = group["time"]
        dt = _effective_dt(times, gap_seconds)
        active = np.concatenate([[0.0], np.cumsum(dt)[:-1]]) if len(dt) else dt
        key = (
            group[key_cols].astype("string").fillna("~").agg("|".join, axis=1).to_numpy()
        )
        since_previous = times.diff().dt.total_seconds().to_numpy()
        changed = np.ones(len(group), dtype=bool)
        if len(group) > 1:
            changed[1:] = (key[1:] != key[:-1]) | (
                np.nan_to_num(since_previous[1:], nan=np.inf) > gap_seconds
            )
        run_id = np.cumsum(changed)

        detail = group.assign(_run=run_id, _dt=dt, _active=active)
        detail["_time"] = times.to_numpy()
        seg_rows = []
        for _, run in detail.groupby("_run", sort=True):
            start = pd.Timestamp(run["_time"].iloc[0])
            duration = float(run["_dt"].sum())
            elapsed_start = float(run["_active"].iloc[0])
            r = {
                "collection": collection,
                "start": start,
                "elapsed_start": elapsed_start,
                "elapsed_end": elapsed_start + duration,
                "duration_s": duration,
                "samples": int(len(run)),
            }
            for c in carry:
                r[c] = run[c].iloc[0]
            seg_rows.append(r)

        # classify each boundary by whether the eNB moved as well
        prev_enb = None
        for r in seg_rows:
            enb = r.get("enb")
            if prev_enb is None:
                r["change"] = "start"
            elif pd.isna(enb) or pd.isna(prev_enb) or enb != prev_enb:
                r["change"] = "enb"
            else:
                r["change"] = "cell"
            prev_enb = enb
        rows.extend(seg_rows)

    segs = pd.DataFrame(rows, columns=cols)
    if segs.empty:
        return empty
    return segs.sort_values(["collection", "elapsed_start"]).reset_index(drop=True)


def segment_summary(segments: pd.DataFrame) -> pd.DataFrame:
    """Per-collection counts for the change timeline.

    Columns: ``collection, n_enb, n_cell, n_pci, segments, cell_changes,
    enb_changes, span_min, median_dwell_s, cell_per_min, enb_per_min``. Every
    segment after the first is a Cell-Id change; the eNB changes are the subset
    marked ``enb``.
    """
    cols = ["collection", "n_enb", "n_cell", "n_pci", "segments", "cell_changes",
            "enb_changes", "span_min", "median_dwell_s", "cell_per_min", "enb_per_min"]
    if segments.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for collection, g in segments.groupby("collection", observed=True, sort=True):
        span_min = max(float(g["elapsed_end"].max()) / 60.0, 1e-9)
        cell_changes = int(len(g) - 1)
        enb_changes = int((g["change"] == "enb").sum())
        rows.append(
            {
                "collection": collection,
                "n_enb": int(g["enb"].nunique()) if "enb" in g.columns else 0,
                "n_cell": int(g["cell_id"].nunique()) if "cell_id" in g.columns else int(len(g)),
                "n_pci": int(g["pci"].nunique()) if "pci" in g.columns else 0,
                "segments": int(len(g)),
                "cell_changes": cell_changes,
                "enb_changes": enb_changes,
                "span_min": span_min,
                "median_dwell_s": float(g["duration_s"].median()),
                "cell_per_min": cell_changes / span_min,
                "enb_per_min": enb_changes / span_min,
            }
        )
    return pd.DataFrame(rows, columns=cols)


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


def paths(
    frame: pd.DataFrame,
    by: Optional[str] = "network",
    gap_seconds: float = DEFAULT_GAP_SECONDS,
) -> pd.DataFrame:
    """Every eNB / sector / PCI path seen, with its time held and samples.

    One row per path, split by ``by`` (the network) when the frame carries it,
    so the same table feeds one tree per network, the per-eNB summary, the
    Structure counts and the download.
    """
    levels = list(TREE_LEVELS)
    if by and by in frame.columns:
        levels = [by, *levels]
    if not set(TREE_LEVELS) <= set(frame.columns):
        # a partial path cannot be placed in the tree
        return pd.DataFrame(columns=levels + ["dwell_s", "samples"])
    table = hierarchy(frame, levels, gap_seconds=gap_seconds)
    if by and by in table.columns:
        table[by] = table[by].astype(str)
    return table


def enb_summary(table: pd.DataFrame, order: str = "sectors") -> pd.DataFrame:
    """One row per eNB of a :func:`paths` table.

    Columns: ``enb, sectors, pcis, minutes`` - how many sectors and PCIs were
    seen under the eNB and for how long it was held.  ``order`` is a key of
    :data:`ENB_ORDERS`: most sectors first (then most PCIs, then eNB ID), or
    by eNB ID.  Never by time held: that says where the phone sat, not how the
    network is built.
    """
    columns = ["enb", "sectors", "pcis", "minutes"]
    if table.empty or not set(TREE_LEVELS) <= set(table.columns):
        return pd.DataFrame(columns=columns)
    out = (
        table.groupby("enb", observed=True)
        .agg(
            sectors=("sector", "nunique"),
            pcis=("pci", "nunique"),
            minutes=("dwell_s", "sum"),
        )
        .reset_index()
    )
    out["minutes"] = out["minutes"] / 60.0
    if order == "enb":
        out = out.sort_values("enb")
    else:
        out = out.sort_values(
            ["sectors", "pcis", "enb"], ascending=[False, False, True]
        )
    return out[columns].reset_index(drop=True)


def network_summary(table: pd.DataFrame, by: str = "network") -> pd.DataFrame:
    """One row per network of a :func:`paths` table: what was seen, and how
    wide the tree is (median and largest count of sectors per eNB and of PCIs
    per sector)."""
    columns = [
        by, "enbs", "sectors", "pcis",
        "sectors_per_enb_median", "sectors_per_enb_max",
        "pcis_per_sector_median", "pcis_per_sector_max",
    ]
    if table.empty or not {by, *TREE_LEVELS} <= set(table.columns):
        return pd.DataFrame(columns=columns)
    rows = []
    for name, group in table.groupby(by, observed=True, sort=False):
        per_enb = group.groupby("enb", observed=True)["sector"].nunique()
        per_sector = group.groupby(["enb", "sector"], observed=True)["pci"].nunique()
        rows.append(
            {
                by: name,
                "enbs": int(len(per_enb)),
                "sectors": int(len(per_sector)),
                "pcis": int(group["pci"].nunique()),
                "sectors_per_enb_median": float(per_enb.median()),
                "sectors_per_enb_max": int(per_enb.max()),
                "pcis_per_sector_median": float(per_sector.median()),
                "pcis_per_sector_max": int(per_sector.max()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


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
    """Plain name of a count, e.g. "Sectors seen per eNB"."""
    parent = parent_levels[-1]
    children = _PLURAL.get(child_level, label_for(child_level))
    return (
        f"{children[0].upper()}{children[1:]} seen per "
        f"{_SINGULAR.get(parent, label_for(parent))}"
    )


def parents_label(parent_levels: Sequence[str]) -> str:
    """What a bar of the fan-out chart counts, e.g. "Number of eNBs"."""
    parent = parent_levels[-1]
    return f"Number of {_PLURAL.get(parent, label_for(parent))}"


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
