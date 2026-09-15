"""Time-alignment and correlation of QP PHY vs SigCap application metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from qpviz import correlate


def _qp(start="2026-09-12 16:11:00", n=120, freq="1s", slope=0.15):
    t = pd.date_range(start, periods=n, freq=freq)
    mcs = 2.0 + slope * np.arange(n)
    return pd.DataFrame({"time": t, "mcs": mcs, "link": "DL"})


def _sig_from(qp, offset="0s", n=30, freq="5s", gain=0.5, noise=0.0, seed=0):
    t0 = qp["time"].min() + pd.Timedelta(offset)
    t = pd.date_range(t0, periods=n, freq=freq)
    base = qp["time"].min()
    secs = [(x - base).total_seconds() for x in t]
    mcs = np.interp(secs, np.arange(len(qp)), qp["mcs"].to_numpy())
    rng = np.random.default_rng(seed)
    app = gain * mcs + rng.normal(0, noise, n)
    return pd.DataFrame({"time": t, "app_tput": app, "link": "DL"})


def test_positive_relationship_scores_high():
    qp = _qp()
    sig = _sig_from(qp, offset="30s", n=30)
    r = correlate.correlate(qp, "mcs", sig, "app_tput", bin="5s")
    assert r.n >= 2 and r.has_pairs
    assert r.pearson > 0.9 and r.spearman > 0.9
    assert list(r.merged.columns) == ["time", "x", "y"]


def test_overlap_window_is_the_shared_span():
    qp = _qp(n=120)                       # 16:11:00 .. 16:12:59
    sig = _sig_from(qp, offset="30s", n=30)  # 16:11:30 .. 16:13:55
    r = correlate.correlate(qp, "mcs", sig, "app_tput", bin="5s")
    assert r.overlap is not None
    lo, hi = r.overlap
    assert lo >= pd.Timestamp("2026-09-12 16:11:30")
    assert hi <= pd.Timestamp("2026-09-12 16:12:59")


def test_no_temporal_overlap_yields_no_pairs():
    qp = _qp()
    sig = _sig_from(qp, offset="0s")
    sig["time"] = sig["time"] + pd.Timedelta("6h")
    r = correlate.correlate(qp, "mcs", sig, "app_tput", bin="5s")
    assert r.n == 0
    assert not r.has_overlap
    assert np.isnan(r.pearson) and np.isnan(r.spearman)


def test_missing_or_empty_frames_are_safe():
    sig = _sig_from(_qp())
    r = correlate.correlate(None, "mcs", sig, "app_tput")
    assert r.n == 0 and not r.has_pairs
    r2 = correlate.correlate(pd.DataFrame(), "mcs", sig, "app_tput")
    assert r2.n == 0


def test_bin_width_changes_pair_count():
    qp = _qp(n=300, freq="1s")
    sig = _sig_from(qp, offset="0s", n=60, freq="5s")
    fine = correlate.correlate(qp, "mcs", sig, "app_tput", bin="5s")
    coarse = correlate.correlate(qp, "mcs", sig, "app_tput", bin="30s")
    assert fine.n > coarse.n


def test_constant_series_gives_nan_correlation():
    qp = _qp(n=60)
    sig = _sig_from(qp, offset="0s", n=12)
    sig["app_tput"] = 5.0  # no variance -> correlation undefined
    r = correlate.correlate(qp, "mcs", sig, "app_tput", bin="5s")
    assert r.n >= 2
    assert np.isnan(r.pearson)


def test_overlap_window_helper():
    qp = _qp()
    sig = _sig_from(qp, offset="30s")
    win = correlate.overlap_window(qp, sig)
    assert win is not None and win[0] <= win[1]
    assert correlate.overlap_window(qp, None) is None
