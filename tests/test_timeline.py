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


# --------------------------------------------------------------------------- #
# QP RB overlay: metric exposure + condition breakdown
# --------------------------------------------------------------------------- #


def test_rb_metric_is_exposed_and_loads_from_qp(fixture_dir, cache_dir):
    from qpviz.schema import METRICS, DATASETS_BY_KEY
    assert "rb" in METRICS and METRICS["rb"].unit == "RB"
    # per-carrier datasets carry it (PDSCH = DL, PUSCH = UL)
    assert "rb" in DATASETS_BY_KEY["lte_pdsch_pc"].metrics
    frame = loader.load(fixture_dir, ["lte_pdsch_pc", "lte_pusch_pc"], cache_dir)
    assert "rb" in frame.columns and frame["rb"].notna().any()
    # DL RBs come from PDSCH, UL RBs from PUSCH
    assert frame.loc[frame["link"] == "DL", "rb"].notna().any()
    assert frame.loc[frame["link"] == "UL", "rb"].notna().any()


def test_overlay_by_condition_partitions_solo_concurrent_and_pci(multi):
    res = timeline.build(multi, bin="5s")
    # a synthetic QP overlay: RB count keyed to each bin, lower where the two
    # phones are active AND share a PCI (what the sanity check expects)
    binned = res.binned
    import numpy as np
    times = sorted(binned["time"].unique())
    sp = timeline._shared_pci_by_bin(binned).set_index("time")
    act = timeline._active_by_bin(binned).set_index("time")
    rows = []
    for t in times:
        n_active = int(act.loc[t, "n_active"]) if t in act.index else 1
        shared = bool(sp.loc[t, "shared"]) if t in sp.index else False
        rb = 20.0
        if n_active >= 2:
            rb -= 6.0
        if shared:
            rb -= 6.0
        rows.append({"time": t, "rb_dl": rb})
    overlay = pd.DataFrame(rows)

    out = timeline.overlay_by_condition(binned, overlay, "rb_dl")
    assert set(out["comparison"]) == {"phones", "pci"}
    phones = out[out["comparison"] == "phones"].iloc[0]
    # a_label is "2 phones", b_label "1 phone" -> two phones should be lower
    assert phones["a_mean"] < phones["b_mean"]
    assert phones["delta_pct"] < 0
    pci = out[out["comparison"] == "pci"].iloc[0]
    # same-PCI lower than different-PCI
    assert pci["a_mean"] < pci["b_mean"]
    assert pci["delta_pct"] < 0


def test_overlay_by_condition_is_empty_without_data(multi):
    res = timeline.build(multi, bin="5s")
    assert timeline.overlay_by_condition(res.binned, pd.DataFrame(), "rb_dl").empty
    assert timeline.overlay_by_condition(pd.DataFrame(), pd.DataFrame(
        {"time": [], "rb_dl": []}), "rb_dl").empty


def test_qp_overlay_scopes_to_matching_collection(fixture_dir, sigcap_multi_dir,
                                                   monkeypatch, tmp_path):
    """Selecting a SigCap session scopes the QP overlay to the QP collection
    whose name contains the same tokens (token match, since QP and SigCap spell
    the collection differently)."""
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    os.environ["QP_DATA_DIR"] = str(fixture_dir)
    os.environ["PRISM_DATA_DIR"] = "/no-prism-dir"
    os.environ["SIGCAP_DATA_DIR"] = str(sigcap_multi_dir)
    app = AppTest.from_file(APP, default_timeout=200).run()
    # the synthetic QP fixture uses collections Collection_A/B/C, and the SigCap
    # multi fixture uses onephoneA/twoPhoneA/twoPhoneB - no token overlap, so the
    # scope is empty and the overlay falls back to time-only without error
    assert not app.exception, [str(e.value) for e in app.exception]
    assert "_tl_qp_scope" in app.session_state


# --------------------------------------------------------------------------- #
# Phone-slot mapping (aligns QP and SigCap by the collection token)
# --------------------------------------------------------------------------- #


def test_phone_slot_extracts_the_number():
    assert timeline.phone_slot("twoPhone1 iperfDL") == 1
    assert timeline.phone_slot("D2C_OpenField_Static_twoPhone2_iperfDL") == 2
    assert timeline.phone_slot("onephone1 iperf") == 1
    assert timeline.phone_slot("910 iperf") is None
    assert timeline.phone_slot(None) is None


def test_device_slots_maps_each_phone(multi):
    slots = timeline.device_slots(multi)
    # the multi fixture: device A on onephoneA/twoPhoneA -> slot ?, but the
    # tokens are letters not digits, so no numeric slot -> empty is acceptable;
    # assert it never raises and returns a dict
    assert isinstance(slots, dict)


def test_device_slots_maps_numeric_phone_tokens():
    frame = pd.DataFrame({
        "device": ["A", "A", "B", "B"],
        "collection": ["twoPhone1 iperfDL", "twoPhone1 iperfUL",
                       "twoPhone2 iperfDL", "twoPhone2 iperfUL"],
    })
    assert timeline.device_slots(frame) == {"A": 1, "B": 2}


# --------------------------------------------------------------------------- #
# Detection is decoupled from the plot bin (shading survives a fine bin)
# --------------------------------------------------------------------------- #


def test_shading_survives_a_bin_finer_than_the_cadence(multi):
    # the multi fixture logs ~every 5 s; a 1 s plot bin must NOT collapse the
    # concurrency / shared-PCI detection
    coarse = timeline.build(multi, bin="5s")
    fine = timeline.build(multi, bin="1s")
    assert fine.detect_bin != "1s"          # detection fell back to the cadence
    assert fine.max_active == coarse.max_active == 2
    assert fine.n_shared_bins == coarse.n_shared_bins
    assert len(fine.shared_pci) == len(coarse.shared_pci)
    # the plotted frame is still at the requested fine resolution
    gaps = fine.binned["time"].sort_values().diff().dropna().dt.total_seconds()
    assert gaps.min() < 5.0


def test_detection_bin_matches_plot_bin_when_coarser(multi):
    r = timeline.build(multi, bin="30s")
    assert r.detect_bin == "30s"            # coarser than cadence -> use plot bin


# --------------------------------------------------------------------------- #
# Activity means carrying traffic, not merely logging
# --------------------------------------------------------------------------- #


def _two_phones(idle_second=True):
    """Phone A runs iperf DL for 60 s on PCI 100. Phone B logs the whole time
    on PCI 100 but only carries traffic during the last 30 s."""
    t = pd.date_range("2026-09-14 19:35:00", periods=60, freq="1s")
    a = pd.DataFrame({"time": t, "device": "A", "app_tput": 10.0, "link": "DL",
                      "pci": 100, "rsrp": -100.0})
    b = pd.DataFrame({"time": t, "device": "B", "app_tput": np.nan, "link": "-",
                      "pci": 100, "rsrp": -105.0})
    if idle_second:
        b.loc[b.index >= 30, ["app_tput", "link"]] = [5.0, "DL"]
    return pd.concat([a, b], ignore_index=True)


def test_idle_logger_is_not_a_contender():
    res = timeline.build(_two_phones(), bin="5s")
    # B is present in every bin but only active in the second half
    act = timeline._active_by_bin(res.binned_detect)
    assert (act["n_present"] == 2).all()
    assert act["n_active"].tolist() == [1] * 6 + [2] * 6
    cmp = res.by_concurrency.set_index("device")
    assert cmp.loc["A", "solo_n"] == 6 and cmp.loc["A", "concurrent_n"] == 6
    # shared PCI only counts while both carry traffic
    assert res.n_shared_bins == 6
    assert {"solo_std", "concurrent_std"} <= set(res.by_concurrency.columns)


def test_direction_tables_are_separate():
    res = timeline.build(_two_phones(), bin="5s")
    assert set(res.by_concurrency_dir) == {"DL", "UL", "both"}
    dl = res.by_concurrency_dir["DL"].set_index("device")
    ul = res.by_concurrency_dir["UL"].set_index("device")
    assert dl.loc["A", "concurrent_n"] == 6
    assert ul["solo_n"].sum() == 0 and ul["concurrent_n"].sum() == 0


def test_segments_use_the_declared_bin_width():
    t = pd.to_datetime(["2026-09-14 19:00:00", "2026-09-14 19:00:20", "2026-09-14 19:00:25"])
    # first two populated bins are 20 s apart: inferring the width from them
    # would make a 5 s bin look 20 s wide
    segs = timeline._segments_from_flags(pd.Series(t), pd.Series([1, 1, 1]),
                                         width=pd.Timedelta("5s"))
    assert segs[-1]["end"] == pd.Timestamp("2026-09-14 19:00:30")
    # a gap of more than three bins breaks the run
    assert len(segs) == 2


def test_overlay_by_condition_keeps_slots_apart():
    frame = _two_phones()
    res = timeline.build(frame, bin="5s")
    t = res.binned_detect["time"].unique()
    # slot 1 gets fewer RBs under contention, slot 2 is constant: pooled they
    # would hide each other
    ov = pd.concat([
        pd.DataFrame({"time": t, "slot": 1, "rb_dl": [20.0] * 6 + [10.0] * 6}),
        pd.DataFrame({"time": t, "slot": 2, "rb_dl": 12.0}),
    ])
    out = timeline.overlay_by_condition(res.binned_detect, ov, "rb_dl", bin="5s")
    assert "slot" in out.columns and set(out["slot"]) == {1, 2}
    s1 = out[(out["slot"] == 1) & (out["comparison"] == "phones")].iloc[0]
    assert s1["a_mean"] == 10.0 and s1["b_mean"] == 20.0 and s1["delta_pct"] == -50.0
    s2 = out[(out["slot"] == 2) & (out["comparison"] == "phones")].iloc[0]
    assert s2["delta_pct"] == 0.0


def test_best_pci_needs_enough_bins():
    frame = _two_phones()
    # one stray bin on PCI 999 with a huge throughput must not win
    stray = frame.iloc[[0]].copy()
    stray["pci"] = 999
    stray["app_tput"] = 1000.0
    stray["time"] = pd.Timestamp("2026-09-14 19:40:00")
    res = timeline.build(pd.concat([frame, stray], ignore_index=True), bin="5s")
    best = res.drivers.set_index("device").loc["A", "best_pci"]
    assert best == 100
