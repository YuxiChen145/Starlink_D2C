"""SigCap (Android logger) integration: loader + end-to-end app behaviour."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qpviz import loader
from qpviz.schema import SIGCAP_APP_TPUT_COL

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


# --------------------------------------------------------------------------- #
# File discovery + name parsing
# --------------------------------------------------------------------------- #


def test_sigcap_available_lists_lte_only_when_no_nr(sigcap_dir):
    assert loader.sigcap_available(sigcap_dir) == ["sigcap_lte"]
    assert loader.sigcap_available("/does-not-exist") == []


def test_sigcap_available_detects_nr(sigcap_nr_dir):
    assert loader.sigcap_available(sigcap_nr_dir) == ["sigcap_nr"]


def test_sigcap_files_dedupes_macos_copies(sigcap_dir):
    files = loader.sigcap_files(sigcap_dir)
    names = [p.name for p in files]
    # the " 2" duplicate of the OpenField session is dropped
    assert not any(n.endswith(" 2.csv") for n in names)
    assert len(files) == 3


def test_sigcap_meta_parses_both_naming_conventions():
    coll, test = loader._sigcap_meta(
        Path("2026_09_S12_1611_E12_1620Z-0400_outdoor_T-Mobile_dev_D2C_Sta_YC_910_iperf.debug_general.csv")
    )
    assert coll == "910 iperf" and test == "iperf"
    coll, test = loader._sigcap_meta(
        Path("2026_09_S12_1633_E12_1640Z-0400_outdoor_T-Mobile_dev_D2C_Sta_YC_1011_http .debug_general.csv")
    )
    assert coll == "1011 http" and test == "http"
    coll, test = loader._sigcap_meta(
        Path("2026-08-14-14-23-06[D2C_Static_OpenField_SigcapiperfDL].csv")
    )
    assert coll == "D2C Static OpenField SigcapiperfDL" and test == "iperf"


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_load_sigcap_normalises_onto_the_canonical_schema(sigcap_dir, cache_dir):
    frame = loader.load_sigcap(sigcap_dir, ["sigcap_lte", "sigcap_nr"], cache_dir)
    assert not frame.empty
    assert set(frame["provider"].unique()) == {"SIGCAP"}
    assert set(frame["tech"].unique()) == {"LTE"}
    # canonical dimension + metric columns are present
    for col in ("time", "collection", "pci", "band", "test_type", "link"):
        assert col in frame.columns
    for metric in ("rsrp", "rsrq", "rssi", "sinr", "app_tput", "rtt", "n_factor"):
        assert metric in frame.columns


def test_sigcap_coalesces_iperf_and_http_into_app_tput(sigcap_dir, cache_dir):
    frame = loader.load_sigcap(sigcap_dir, ["sigcap_lte"], cache_dir)
    by_test = frame.groupby("test_type", observed=True)["app_tput"].apply(
        lambda s: s.notna().any()
    )
    assert bool(by_test.get("iperf")) and bool(by_test.get("http"))
    # the internal synthesised column never leaks into the canonical frame
    assert SIGCAP_APP_TPUT_COL not in frame.columns


def test_sigcap_scrubs_the_cqi_sentinel(sigcap_dir, cache_dir):
    frame = loader.load_sigcap(sigcap_dir, ["sigcap_lte"], cache_dir)
    if "cqi" in frame.columns:
        assert (frame["cqi"].dropna() < 2_000_000_000).all()
        # only the i%5==0 rows had a real CQI of 9
        assert frame["cqi"].dropna().eq(9).all()


def test_sigcap_direction_is_per_row_from_iperf_direction(sigcap_dir, cache_dir):
    frame = loader.load_sigcap(sigcap_dir, ["sigcap_lte"], cache_dir)
    links = set(frame["link"].dropna().astype(str).unique())
    # iperf sessions alternate download/upload; http rows are downlink
    assert {"DL", "UL"} <= links


def test_sigcap_timestamp_is_naive_local_wall_clock(sigcap_dir, cache_dir):
    frame = loader.load_sigcap(sigcap_dir, ["sigcap_lte"], cache_dir)
    t = frame["time"].dropna()
    assert t.dt.tz is None  # tz offset stripped, kept as local wall clock
    # the 910 iperf session starts at 16:11 local
    assert (t.dt.hour == 16).any()


def test_sigcap_band_is_b25_not_b0(sigcap_dir, cache_dir):
    frame = loader.load_sigcap(sigcap_dir, ["sigcap_lte"], cache_dir)
    bands = set(frame["band"].dropna().astype(str).unique())
    assert "B25" in bands
    assert "B0" not in bands


def test_sigcap_app_only_rows_are_kept(sigcap_dir, cache_dir):
    # rows with throughput but no serving-cell fix must survive (routed to LTE)
    frame = loader.load_sigcap(sigcap_dir, ["sigcap_lte"], cache_dir)
    app_no_radio = frame[frame["app_tput"].notna() & frame["rsrp"].isna()]
    assert len(app_no_radio) > 0


def test_sigcap_routes_nr_rows_to_the_nr_dataset(sigcap_nr_dir, cache_dir):
    frame = loader.load_sigcap(sigcap_nr_dir, ["sigcap_lte", "sigcap_nr"], cache_dir)
    assert set(frame["tech"].unique()) == {"NR"}
    assert frame["rsrp"].notna().any()


def test_sigcap_cache_roundtrips(sigcap_dir, cache_dir):
    first = loader.load_sigcap(sigcap_dir, ["sigcap_lte"], cache_dir)
    second = loader.load_sigcap(sigcap_dir, ["sigcap_lte"], cache_dir)  # from cache
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# End-to-end app
# --------------------------------------------------------------------------- #


def _run(qp_dir, sigcap_dir, timeout=180):
    os.environ["QP_DATA_DIR"] = str(qp_dir) if qp_dir else "/no-qp-dir"
    os.environ["PRISM_DATA_DIR"] = "/no-prism-dir"
    os.environ["SIGCAP_DATA_DIR"] = str(sigcap_dir)
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.run()
    return app


def test_app_offers_sigcap_source_and_app_metrics(sigcap_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(None, sigcap_dir)
    assert not app.exception, [str(e.value) for e in app.exception]
    metric_labels = app.selectbox(key="metric_choice").options
    assert "App throughput (Mbps)" in metric_labels
    assert "Ping RTT (ms)" in metric_labels


def test_correlation_tab_present_only_with_both_sources(
    fixture_dir, sigcap_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    # SigCap alone: no correlation tab
    app = _run(None, sigcap_dir)
    assert not any("Correlation" in t.label for t in app.tabs)
    # QP + SigCap: correlation tab appears
    app = _run(fixture_dir, sigcap_dir)
    assert not app.exception, [str(e.value) for e in app.exception]
    assert any("Correlation" in t.label for t in app.tabs)
    assert app.selectbox(key="corr_phy") is not None
    assert app.selectbox(key="corr_app") is not None
