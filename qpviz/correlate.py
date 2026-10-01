"""Time-align two measurement sources on the same phone and correlate them.

The Qualipoc (QP) logs carry the PHY layer in depth - MCS, PDSCH throughput,
spectral efficiency, BLER, RB allocation - while SigCap, running on the *same
phone at the same time*, carries the application layer the PHY logs never see:
iperf / HTTP throughput and ping RTT.  Lining the two up in time turns "the
radio looked like X" and "the app got Y" into one paired observation, so you can
ask how PHY conditions translate into what the application actually experienced.

The join is by time, not by row: QP samples about once a second and SigCap about
every five, so both are resampled onto a common bin (default 5 s) and matched
bin-to-bin over the window where the two recordings overlap.  Everything here is
pure pandas so it can be unit-tested without Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

#: default common bin - SigCap's own cadence; the UI lets the user change it
DEFAULT_BIN = "5s"

#: fewest paired bins before a coefficient is reported. Pearson on two points is
#: always +/-1 and on a handful is dominated by one bin; ten is the floor below
#: which the number says more about the sample than the relationship.
MIN_PAIRS = 10

#: directions a row can carry; ``"-"`` marks rows with no direction (ping, idle)
_DIRECTIONS = ("DL", "UL")


def _binned(
    frame: Optional[pd.DataFrame],
    metric: str,
    bin: str,
    agg: str = "mean",
    time_col: str = "time",
) -> pd.Series:
    """Resample one metric onto ``bin`` time buckets (mean by default).

    Returns an empty float Series (DatetimeIndex) when the frame is missing, has
    no such metric, or has nothing non-null to bin.
    """
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([], name=time_col))
    if frame is None or frame.empty or metric not in frame.columns:
        return empty
    if time_col not in frame.columns:
        return empty
    df = frame[[time_col, metric]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[time_col, metric])
    if df.empty:
        return empty
    series = df.set_index(time_col)[metric].sort_index().resample(bin).agg(agg)
    return series.dropna()


def overlap_window(
    a: Optional[pd.DataFrame],
    b: Optional[pd.DataFrame],
    time_col: str = "time",
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """The wall-clock window both frames cover, or ``None`` if they never do."""
    def span(f):
        if f is None or f.empty or time_col not in f.columns:
            return None
        t = pd.to_datetime(f[time_col], errors="coerce").dropna()
        return (t.min(), t.max()) if not t.empty else None

    sa, sb = span(a), span(b)
    if sa is None or sb is None:
        return None
    lo, hi = max(sa[0], sb[0]), min(sa[1], sb[1])
    return (lo, hi) if lo <= hi else None


@dataclass(frozen=True)
class Correlation:
    """One resample-and-align of a QP PHY metric against a SigCap app metric."""

    merged: pd.DataFrame  # columns: time (bin start), x (QP PHY), y (SigCap app)
    x_metric: str
    y_metric: str
    bin: str
    n: int
    pearson: float
    spearman: float
    overlap: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
    #: direction both sides were restricted to, or ``None`` for pooled
    link: Optional[str] = None
    #: True when the QP side pooled DL and UL rows into one series - the
    #: coefficient then compares a direction mix, not a PHY/app relationship
    mixed_directions: bool = False

    @property
    def has_overlap(self) -> bool:
        return self.overlap is not None

    @property
    def has_pairs(self) -> bool:
        return self.n >= MIN_PAIRS


def _restrict_link(frame: Optional[pd.DataFrame], link: Optional[str],
                   keep_undirected: bool) -> Optional[pd.DataFrame]:
    """Rows of one direction (plus, optionally, rows that carry none)."""
    if link is None or frame is None or frame.empty or "link" not in frame.columns:
        return frame
    col = frame["link"].astype("string")
    keep = col.eq(link)
    if keep_undirected:
        keep |= col.eq("-") | col.isna()
    return frame[keep]


def _directions_present(frame: Optional[pd.DataFrame], metric: str) -> set:
    if frame is None or frame.empty or "link" not in frame.columns or metric not in frame.columns:
        return set()
    has = pd.to_numeric(frame[metric], errors="coerce").notna()
    present = set(frame.loc[has, "link"].astype("string").dropna().unique())
    return present & set(_DIRECTIONS)


def correlate(
    qp_frame: Optional[pd.DataFrame],
    qp_metric: str,
    sig_frame: Optional[pd.DataFrame],
    sig_metric: str,
    bin: str = DEFAULT_BIN,
    qp_agg: str = "mean",
    sig_agg: str = "mean",
    time_col: str = "time",
    link: Optional[str] = None,
) -> Correlation:
    """Resample both sources to ``bin`` and pair them on the bins they share.

    Only bins present in *both* recordings survive, so the result is exactly the
    overlapping window sampled at ``bin``.  Pearson and Spearman coefficients are
    NaN until there are at least :data:`MIN_PAIRS` paired bins.

    ``link`` (``"DL"`` / ``"UL"``) restricts both sides to one direction: the QP
    PHY exports are per direction (PDSCH is DL, PUSCH is UL) and averaging the
    two into one series would correlate a direction mix. SigCap rows that carry
    no direction (ping, idle) are kept, so RTT can still be paired. With
    ``link=None`` nothing is filtered and ``mixed_directions`` reports whether
    the QP side actually pooled both.
    """
    qp_frame = _restrict_link(qp_frame, link, keep_undirected=False)
    sig_frame = _restrict_link(sig_frame, link, keep_undirected=True)
    mixed = link is None and len(_directions_present(qp_frame, qp_metric)) > 1

    x = _binned(qp_frame, qp_metric, bin, qp_agg, time_col).rename("x")
    y = _binned(sig_frame, sig_metric, bin, sig_agg, time_col).rename("y")
    merged = pd.concat([x, y], axis=1, join="inner").dropna()
    merged = merged.sort_index()
    merged.index.name = time_col
    n = int(len(merged))

    if n >= MIN_PAIRS and merged["x"].std() > 0 and merged["y"].std() > 0:
        pearson = float(merged["x"].corr(merged["y"], method="pearson"))
        # Spearman = Pearson of the ranks; done by hand so scipy is not a
        # dependency (pandas' method="spearman" imports scipy.stats)
        spearman = float(merged["x"].rank().corr(merged["y"].rank(), method="pearson"))
    else:
        pearson = spearman = float("nan")

    # the window is [first bin start, last bin end)
    win = (
        (merged.index.min(), merged.index.max() + pd.Timedelta(bin))
        if n
        else overlap_window(qp_frame, sig_frame, time_col)
    )

    return Correlation(
        merged=merged.reset_index(),
        x_metric=qp_metric,
        y_metric=sig_metric,
        bin=bin,
        n=n,
        pearson=pearson,
        spearman=spearman,
        overlap=win,
        link=link,
        mixed_directions=mixed,
    )
