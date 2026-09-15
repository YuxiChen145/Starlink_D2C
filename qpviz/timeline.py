"""Multi-device timeline analysis.

Several SigCap sessions - one per phone - can run at the same time on the same
spot.  This module aligns those phones onto a shared wall-clock timeline so the
platform can answer two questions the single-source tabs cannot:

* **Contention.** How does each phone's throughput change between the stretches
  when it is the *only* phone using Starlink D2C and the stretches when a second
  phone is transmitting at the same time?  "How many phones are using it" is
  inferred from the data - the number of distinct devices with a sample in the
  same time bin - never from a file-name label (those are hand-typed and, in
  this campaign, sometimes wrong).

* **Shared cell.** When two phones camp on the *same* PCI, does throughput move
  together?  We mark the bins where the concurrent phones share a serving PCI
  and compare throughput in the shared-PCI bins against the rest.

Everything is built on one intermediate: :func:`bin_devices`, which collapses
the raw per-sample rows to one row per (device, time-bin) carrying that bin's
mean RSRP, its downlink and uplink app-throughput, and its dominant serving
PCI.  The rest of the module - concurrency, shared-PCI, the comparison tables -
derives from that single frame so the numbers on the plot and in the tables can
never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "bin_devices",
    "device_order",
    "concurrency_segments",
    "shared_pci_segments",
    "compare_by_concurrency",
    "compare_by_shared_pci",
    "throughput_drivers",
    "overlay_by_condition",
    "phone_slot",
    "device_slots",
    "TimelineResult",
    "build",
]

# Aggregations offered to the user; applied to RSRP and throughput within a bin.
AGGS = ("mean", "median", "max", "min")


import re as _re


def phone_slot(name) -> Optional[int]:
    """The phone number encoded in a collection / session name.

    Both sources tag the experiment phone in the name - SigCap ``twoPhone1``,
    QP ``D2C_OpenField_Static_twoPhone1_iperfDL`` - so the trailing digit after
    ``phone`` is the shared key that lets QP and SigCap line up by phone even
    though their device IDs (uuid vs serial) are unrelated. ``None`` when the
    name carries no ``phoneN`` token.
    """
    if name is None:
        return None
    m = _re.search(r"phone\s*_?\s*(\d+)", str(name), _re.IGNORECASE)
    return int(m.group(1)) if m else None


def device_slots(frame: pd.DataFrame) -> Dict[str, int]:
    """Map each SigCap ``device`` to its phone slot via the collection token.

    A phone only ever plays one slot in the campaign, so the mapping is taken
    from the most common ``phoneN`` token across that device's rows.
    """
    if frame is None or frame.empty or "device" not in frame.columns \
            or "collection" not in frame.columns:
        return {}
    out: Dict[str, int] = {}
    for dev, g in frame.dropna(subset=["device"]).groupby("device"):
        slots = [phone_slot(c) for c in g["collection"].dropna().unique()]
        slots = [x for x in slots if x is not None]
        if slots:
            out[str(dev)] = max(set(slots), key=slots.count)
    return out


def _agg_fn(agg: str):
    return {"mean": np.nanmean, "median": np.nanmedian,
            "max": np.nanmax, "min": np.nanmin}.get(agg, np.nanmean)


def _mode(series: pd.Series):
    """Most common non-null value (the bin's dominant serving cell)."""
    s = series.dropna()
    if s.empty:
        return pd.NA
    m = s.mode()
    return m.iloc[0] if not m.empty else pd.NA


def bin_devices(
    frame: pd.DataFrame,
    bin: str = "5s",
    agg: str = "mean",
) -> pd.DataFrame:
    """One row per (device, time-bin).

    Columns: ``time`` (bin start), ``device``, ``rsrp``, ``sinr``,
    ``app_tput_dl``, ``app_tput_ul``, ``app_tput`` (either direction),
    ``rtt``, ``pci`` (dominant), ``n_samples``.  Directions are split so a
    downlink test and an uplink test in the same bin never average together.
    """
    cols = frame.columns
    need_time = "time" in cols and "device" in cols
    if not need_time or frame.empty:
        return pd.DataFrame(
            columns=["time", "device", "rsrp", "sinr", "app_tput_dl",
                     "app_tput_ul", "app_tput", "rtt", "pci", "n_samples"]
        )
    df = frame.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df[df["time"].notna() & df["device"].notna()]
    if df.empty:
        return bin_devices(pd.DataFrame(), bin=bin)
    df["__bin__"] = df["time"].dt.floor(bin)
    fn = _agg_fn(agg)

    def _num(s):
        return pd.to_numeric(s, errors="coerce") if s is not None else np.nan

    rows: List[dict] = []
    link = df["link"].astype("string") if "link" in cols else pd.Series("-", index=df.index)
    for (dev, b), g in df.groupby(["device", "__bin__"], observed=True):
        gl = link.loc[g.index]
        app = _num(g["app_tput"]) if "app_tput" in cols else pd.Series(np.nan, index=g.index)
        dl = app[gl.eq("DL")]
        ul = app[gl.eq("UL")]
        with np.errstate(all="ignore"):
            rec = {
                "time": b,
                "device": dev,
                "rsrp": fn(_num(g["rsrp"])) if "rsrp" in cols and g["rsrp"].notna().any() else np.nan,
                "sinr": fn(_num(g["sinr"])) if "sinr" in cols and g["sinr"].notna().any() else np.nan,
                "app_tput_dl": fn(dl) if dl.notna().any() else np.nan,
                "app_tput_ul": fn(ul) if ul.notna().any() else np.nan,
                "app_tput": fn(app) if app.notna().any() else np.nan,
                "rtt": fn(_num(g["rtt"])) if "rtt" in cols and g["rtt"].notna().any() else np.nan,
                "pci": _mode(g["pci"]) if "pci" in cols else pd.NA,
                "n_samples": int(len(g)),
            }
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["time", "device"]).reset_index(drop=True)
    out["pci"] = out["pci"].astype("Int64")
    return out


def device_order(binned: pd.DataFrame) -> List[str]:
    """Devices, most-sampled first (stable colour / legend order)."""
    if binned.empty:
        return []
    return (
        binned.groupby("device")["n_samples"].sum()
        .sort_values(ascending=False).index.astype(str).tolist()
    )


def _active_by_bin(binned: pd.DataFrame) -> pd.DataFrame:
    """Per bin: how many distinct devices are present, and which."""
    if binned.empty:
        return pd.DataFrame(columns=["time", "n_active", "devices"])
    g = binned.groupby("time")["device"].agg(
        n_active="nunique", devices=lambda s: tuple(sorted(s.unique()))
    ).reset_index()
    return g


def _segments_from_flags(times: pd.Series, values: pd.Series) -> List[dict]:
    """Collapse a per-bin step signal into contiguous [start, end, value] runs.

    ``end`` is exclusive-ish: it is the last bin's start plus the bin width, so
    a shaded band reaches the right edge of the final bin it covers.
    """
    times = list(times)
    values = list(values)
    if not times:
        return []
    width = (times[1] - times[0]) if len(times) > 1 else pd.Timedelta(seconds=5)
    segs: List[dict] = []
    start = times[0]
    cur = values[0]
    for i in range(1, len(times)):
        # a gap larger than 3 bins breaks the run (separate sessions)
        gap = times[i] - times[i - 1]
        if values[i] != cur or gap > 3 * width:
            segs.append({"start": start, "end": times[i - 1] + width, "value": cur})
            start = times[i]
            cur = values[i]
    segs.append({"start": start, "end": times[-1] + width, "value": cur})
    return segs


def concurrency_segments(binned: pd.DataFrame) -> List[dict]:
    """Contiguous stretches labelled by how many phones were active."""
    act = _active_by_bin(binned)
    if act.empty:
        return []
    return _segments_from_flags(act["time"], act["n_active"])


def _shared_pci_by_bin(binned: pd.DataFrame) -> pd.DataFrame:
    """Per bin (with >=2 active devices): do they share one serving PCI?

    Returns columns ``time``, ``n_active``, ``shared`` (bool), ``pci`` (the
    shared PCI when they agree, else <NA>).
    """
    if binned.empty:
        return pd.DataFrame(columns=["time", "n_active", "shared", "pci"])
    rows = []
    for t, g in binned.groupby("time"):
        pcis = g["pci"].dropna().unique().tolist()
        n_active = int(g["device"].nunique())
        # every active device reported a PCI, and they all match
        all_have = g["device"].nunique() == g.loc[g["pci"].notna(), "device"].nunique()
        shared = bool(n_active >= 2 and len(pcis) == 1 and all_have)
        rows.append({
            "time": t, "n_active": n_active, "shared": shared,
            "pci": (pcis[0] if shared else pd.NA),
        })
    out = pd.DataFrame(rows)
    out["pci"] = out["pci"].astype("Int64")
    return out


def shared_pci_segments(binned: pd.DataFrame) -> List[dict]:
    """Contiguous stretches where the concurrent phones shared a serving PCI."""
    sp = _shared_pci_by_bin(binned)
    sp = sp[sp["n_active"] >= 2]
    if sp.empty:
        return []
    segs = _segments_from_flags(sp["time"], sp["shared"])
    return [s for s in segs if s["value"]]


# --------------------------------------------------------------------------- #
# Quantification
# --------------------------------------------------------------------------- #


def _describe(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype="float64")
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "std": np.nan}
    return {"n": int(v.size), "mean": float(np.mean(v)),
            "median": float(np.median(v)), "std": float(np.std(v, ddof=1) if v.size > 1 else 0.0)}


def compare_by_concurrency(
    binned: pd.DataFrame, metric: str = "app_tput"
) -> pd.DataFrame:
    """For each device, ``metric`` when it is the only active phone vs when a
    second phone is also active.  ``metric`` is one of the binned throughput
    columns (``app_tput``, ``app_tput_dl``, ``app_tput_ul``) or ``rsrp``.
    """
    if binned.empty or metric not in binned.columns:
        return pd.DataFrame(
            columns=["device", "solo_n", "solo_mean", "concurrent_n",
                     "concurrent_mean", "delta_pct"]
        )
    act = _active_by_bin(binned)[["time", "n_active"]]
    df = binned.merge(act, on="time", how="left")
    out = []
    for dev, g in df.groupby("device"):
        solo = _describe(g.loc[g["n_active"] <= 1, metric].to_numpy())
        conc = _describe(g.loc[g["n_active"] >= 2, metric].to_numpy())
        delta = (
            100.0 * (conc["mean"] - solo["mean"]) / solo["mean"]
            if solo["mean"] not in (0.0, np.nan) and np.isfinite(solo["mean"]) and solo["mean"] != 0
            else np.nan
        )
        out.append({
            "device": dev,
            "solo_n": solo["n"], "solo_mean": solo["mean"],
            "concurrent_n": conc["n"], "concurrent_mean": conc["mean"],
            "delta_pct": delta,
        })
    return pd.DataFrame(out)


def compare_by_shared_pci(
    binned: pd.DataFrame, metric: str = "app_tput"
) -> pd.DataFrame:
    """Among concurrent bins (>=2 phones), ``metric`` per device when the phones
    shared a serving PCI vs when they were on different PCIs."""
    if binned.empty or metric not in binned.columns:
        return pd.DataFrame(
            columns=["device", "shared_n", "shared_mean", "diff_n",
                     "diff_mean", "delta_pct"]
        )
    sp = _shared_pci_by_bin(binned)
    conc = sp[sp["n_active"] >= 2][["time", "shared"]]
    df = binned.merge(conc, on="time", how="inner")
    out = []
    for dev, g in df.groupby("device"):
        sh = _describe(g.loc[g["shared"], metric].to_numpy())
        di = _describe(g.loc[~g["shared"], metric].to_numpy())
        delta = (
            100.0 * (sh["mean"] - di["mean"]) / di["mean"]
            if np.isfinite(di["mean"]) and di["mean"] != 0 else np.nan
        )
        out.append({
            "device": dev,
            "shared_n": sh["n"], "shared_mean": sh["mean"],
            "diff_n": di["n"], "diff_mean": di["mean"],
            "delta_pct": delta,
        })
    return pd.DataFrame(out)


def throughput_drivers(
    binned: pd.DataFrame, metric: str = "app_tput"
) -> pd.DataFrame:
    """Per device: Pearson & Spearman of throughput vs RSRP, and the serving PCI
    that carried the highest mean throughput (the "best cell")."""
    if binned.empty or metric not in binned.columns:
        return pd.DataFrame(
            columns=["device", "n", "pearson_rsrp", "spearman_rsrp",
                     "best_pci", "best_pci_mean", "n_pci"]
        )
    out = []
    for dev, g in binned.groupby("device"):
        x = pd.to_numeric(g["rsrp"], errors="coerce")
        y = pd.to_numeric(g[metric], errors="coerce")
        ok = x.notna() & y.notna()
        n = int(ok.sum())
        if n >= 2 and x[ok].std() > 0 and y[ok].std() > 0:
            xa, ya = x[ok], y[ok]
            pear = float(np.corrcoef(xa, ya)[0, 1])
            # Spearman = Pearson of the ranks, computed by hand so scipy stays
            # out of the dependency set (pandas method="spearman" imports it)
            spear = float(xa.rank().corr(ya.rank(), method="pearson"))
        else:
            pear = spear = np.nan
        by_pci = (
            g.dropna(subset=["pci", metric])
            .groupby("pci")[metric].mean().sort_values(ascending=False)
        )
        best_pci = int(by_pci.index[0]) if not by_pci.empty else pd.NA
        best_mean = float(by_pci.iloc[0]) if not by_pci.empty else np.nan
        out.append({
            "device": dev, "n": n,
            "pearson_rsrp": pear, "spearman_rsrp": spear,
            "best_pci": best_pci, "best_pci_mean": best_mean,
            "n_pci": int(g["pci"].dropna().nunique()),
        })
    return pd.DataFrame(out)


def overlay_by_condition(
    binned: pd.DataFrame, overlay: pd.DataFrame, col: str
) -> pd.DataFrame:
    """A per-bin overlay signal (e.g. QP ``rb_dl``) averaged under each timeline
    condition: one phone vs two active, and - among concurrent bins - same PCI
    vs different PCI.

    The overlay is QP PHY data binned to the same grid as the SigCap phones, so
    it merges to the concurrency / shared-PCI flags on the bin timestamp. This is
    the RB sanity check: if a shared cell splits its resource blocks, the QP
    phone's RB count should be lower under the two-phone and same-PCI conditions.
    Returns a one-row-per-condition-pair frame with n / mean for each side.
    """
    empty = pd.DataFrame(
        columns=["comparison", "a_label", "a_n", "a_mean",
                 "b_label", "b_n", "b_mean", "delta_pct"]
    )
    if (
        binned is None or binned.empty or overlay is None or overlay.empty
        or col not in overlay.columns or "time" not in overlay.columns
    ):
        return empty
    o = overlay[["time", col]].dropna()
    if o.empty:
        return empty
    act = _active_by_bin(binned)[["time", "n_active"]]
    sp = _shared_pci_by_bin(binned)[["time", "shared", "n_active"]]

    rows = []

    def _pair(comparison, a_label, a_vals, b_label, b_vals):
        a, b = _describe(a_vals), _describe(b_vals)
        delta = (
            100.0 * (a["mean"] - b["mean"]) / b["mean"]
            if np.isfinite(b["mean"]) and b["mean"] != 0 else np.nan
        )
        rows.append({
            "comparison": comparison,
            "a_label": a_label, "a_n": a["n"], "a_mean": a["mean"],
            "b_label": b_label, "b_n": b["n"], "b_mean": b["mean"],
            "delta_pct": delta,
        })

    ma = o.merge(act, on="time", how="inner")
    _pair(
        "phones", "2 phones", ma.loc[ma["n_active"] >= 2, col].to_numpy(),
        "1 phone", ma.loc[ma["n_active"] <= 1, col].to_numpy(),
    )
    mc = o.merge(sp[sp["n_active"] >= 2], on="time", how="inner")
    _pair(
        "pci", "same PCI", mc.loc[mc["shared"], col].to_numpy(),
        "diff PCI", mc.loc[~mc["shared"], col].to_numpy(),
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# One-shot bundle for the app
# --------------------------------------------------------------------------- #


@dataclass
class TimelineResult:
    binned: pd.DataFrame
    devices: List[str]
    concurrency: List[dict]
    shared_pci: List[dict]
    by_concurrency: pd.DataFrame
    by_shared_pci: pd.DataFrame
    drivers: pd.DataFrame
    bin: str = "5s"
    max_active: int = 0
    n_shared_bins: int = 0

    @property
    def has_concurrency(self) -> bool:
        return self.max_active >= 2


def build(
    frame: pd.DataFrame, bin: str = "5s", agg: str = "mean",
    metric: str = "app_tput",
) -> TimelineResult:
    """Bin the frame once and derive every timeline product from it."""
    binned = bin_devices(frame, bin=bin, agg=agg)
    act = _active_by_bin(binned)
    max_active = int(act["n_active"].max()) if not act.empty else 0
    shared = shared_pci_segments(binned)
    n_shared_bins = int(_shared_pci_by_bin(binned)["shared"].sum()) if not binned.empty else 0
    return TimelineResult(
        binned=binned,
        devices=device_order(binned),
        concurrency=concurrency_segments(binned),
        shared_pci=shared,
        by_concurrency=compare_by_concurrency(binned, metric),
        by_shared_pci=compare_by_shared_pci(binned, metric),
        drivers=throughput_drivers(binned, metric),
        bin=bin, max_active=max_active, n_shared_bins=n_shared_bins,
    )
