"""Multi-device timeline analysis: concurrency, shared-PCI, comparisons."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qpviz import loader, timeline

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture()
def multi(sigcap_multi_dir, cache_dir):
    return loader.load_sigcap(sigcap_multi_dir, ["sigcap_lte", "sigcap_nr"], cache_dir)


# --------------------------------------------------------------------------- #
# Device identity carried through the loader
# --------------------------------------------------------------------------- #


def test_two_distinct_devices_are_recovered(multi):
    devs = set(multi["device"].dropna().unique())
    assert len(devs) == 2
    assert any("pa2q" in d for d in devs) and any("e2q" in d for d in devs)
    # the label carries the short uuid so two phones never collapse
    assert any("aaaa1111" in d for d in devs)


# --------------------------------------------------------------------------- #
# Binning + concurrency
# --------------------------------------------------------------------------- #


def test_bin_devices_splits_dl_ul(multi):
    b = timeline.bin_devices(multi, bin="5s")
    assert not b.empty
    assert {"time", "device", "rsrp", "app_tput_dl", "app_tput_ul", "pci"} <= set(b.columns)
    # every bin belongs to exactly one device+time
    assert not b.duplicated(subset=["time", "device"]).any()


def test_concurrency_inferred_from_overlap(multi):
    res = timeline.build(multi, bin="5s")
    assert res.max_active == 2
    # the solo stretch (only phone A) must contribute n_active==1 somewhere
    vals = {seg["value"] for seg in res.concurrency}
    assert 1 in vals and 2 in vals


def test_solo_vs_concurrent_partition(multi):
    res = timeline.build(multi, bin="5s")
    cmp = res.by_concurrency.set_index("device")
    # phone A has both solo bins (its 19:00 session) and concurrent bins
    a = [d for d in cmp.index if "pa2q" in d][0]
    assert cmp.loc[a, "solo_n"] > 0
    assert cmp.loc[a, "concurrent_n"] > 0
    # phone B only ever recorded during the concurrent window
    b = [d for d in cmp.index if "e2q" in d][0]
    assert cmp.loc[b, "solo_n"] == 0
    assert cmp.loc[b, "concurrent_n"] > 0


# --------------------------------------------------------------------------- #
# Shared PCI
# --------------------------------------------------------------------------- #


def test_shared_pci_detected_only_when_pcis_match(multi):
    res = timeline.build(multi, bin="5s")
    # the first concurrent bins share PCI 100; later bins diverge (200 vs 300)
    assert res.n_shared_bins > 0
    seg = res.shared_pci[0]
    assert seg["start"] <= seg["end"]

    # a bin where the two phones are on different PCIs is NOT shared
    sp = timeline._shared_pci_by_bin(res.binned)
    concurrent = sp[sp["n_active"] >= 2]
    assert (~concurrent["shared"]).any()  # some divergent bins exist
    assert concurrent["shared"].any()     # and some shared bins exist
    # a shared bin's recorded pci is the common one (100 in the fixture)
    shared_rows = concurrent[concurrent["shared"]]
    assert (shared_rows["pci"] == 100).all()


def test_compare_by_shared_pci_runs(multi):
    res = timeline.build(multi, bin="5s")
    df = res.by_shared_pci
    assert set(["device", "shared_mean", "diff_mean"]).issubset(df.columns)
    assert (df["shared_n"] > 0).any()


# --------------------------------------------------------------------------- #
# Drivers (throughput vs RSRP / PCI)
# --------------------------------------------------------------------------- #


def test_drivers_reports_correlation_and_best_pci(multi):
    res = timeline.build(multi, bin="5s")
    dv = res.drivers
    assert (dv["n"] > 0).all()
    assert dv["best_pci"].notna().any()
    # correlations are finite or NaN, never out of range
    for col in ("pearson_rsrp", "spearman_rsrp"):
        v = pd.to_numeric(dv[col], errors="coerce").dropna()
        assert ((v >= -1.0001) & (v <= 1.0001)).all()


def test_spearman_computed_without_scipy(monkeypatch, multi):
    # guard against reintroducing a scipy dependency for Spearman
    import builtins
    real_import = builtins.__import__

    def _blocked(name, *a, **k):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("scipy is intentionally blocked in this test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    dv = timeline.throughput_drivers(timeline.bin_devices(multi, bin="5s"))
    assert "spearman_rsrp" in dv.columns


# --------------------------------------------------------------------------- #
# Empty / single-device safety
# --------------------------------------------------------------------------- #


def test_single_device_has_no_concurrency(sigcap_dir, cache_dir):
    frame = loader.load_sigcap(sigcap_dir, ["sigcap_lte"], cache_dir)
    res = timeline.build(frame, bin="5s")
    assert res.max_active <= 1
    assert not res.has_concurrency
    assert res.n_shared_bins == 0


def test_empty_frame_is_safe():
    res = timeline.build(pd.DataFrame(), bin="5s")
    assert res.binned.empty and res.devices == [] and res.max_active == 0


# --------------------------------------------------------------------------- #
# End-to-end app
# --------------------------------------------------------------------------- #


def _run(sigcap_dir, qp_dir=None, timeout=200):
    os.environ["QP_DATA_DIR"] = str(qp_dir) if qp_dir else "/no-qp-dir"
    os.environ["PRISM_DATA_DIR"] = "/no-prism-dir"
    os.environ["SIGCAP_DATA_DIR"] = str(sigcap_dir)
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.run()
    return app


def test_timeline_tab_present_and_computes(sigcap_multi_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(sigcap_multi_dir)
    assert not app.exception, [str(e.value) for e in app.exception]
    assert any("Timeline analysis" in t.label for t in app.tabs)
    assert any(m.key == "tl_devices" for m in app.multiselect)
    # the concurrency metric reflects the two overlapping phones
    conc = [m for m in app.metric if m.label == "Max concurrent"]
    assert conc and conc[0].value == "2"
