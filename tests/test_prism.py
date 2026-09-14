"""PRISM (SDR scanner) integration: loader + end-to-end app behaviour."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from qpviz import loader

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def test_prism_available_lists_both_technologies(prism_dir):
    assert loader.prism_available(prism_dir) == ["prism_lte", "prism_nr"]
    assert loader.prism_available("/does-not-exist") == []


def test_prism_collection_comes_from_the_filename():
    got = loader.prism_collection(
        Path("2026_09_01_ LTE_8665_Shoreham_Court_Kitchen_static.csv")
    )
    assert got == "Shoreham Court Kitchen static"
    # no leading space, no ARFCN, underscores turned to spaces
    assert loader.prism_collection(
        Path("2026_09_01_NR_397970_Cushing202_static.csv")
    ) == "Cushing202 static"


def test_load_prism_normalises_onto_the_canonical_schema(prism_dir, cache_dir):
    frame = loader.load_prism(prism_dir, ["prism_lte", "prism_nr"], cache_dir)
    from qpviz.schema import DIMENSION_COLUMNS

    for column in DIMENSION_COLUMNS:
        assert column in frame.columns
    assert set(frame["provider"].unique()) == {"PRISM"}
    assert set(frame["tech"].unique()) == {"LTE", "NR"}
    # LTE + NR at two locations -> LTE files share a collection name per spot
    assert "Test Kitchen static" in set(frame["collection"].unique())
    assert "Test Bedroom static" in set(frame["collection"].unique())
    # bands normalised per technology
    assert set(frame["band"].unique()) == {"B25", "n25"}
    # PRISM only carries the radio-signal metrics, never throughput/MCS
    for metric in ("rsrp", "rsrq", "rssi", "sinr"):
        assert frame[metric].notna().any()


def test_load_prism_only_the_requested_technology(prism_dir, cache_dir):
    lte = loader.load_prism(prism_dir, ["prism_lte"], cache_dir)
    assert set(lte["tech"].unique()) == {"LTE"}
    assert lte["provider"].unique().tolist() == ["PRISM"]


def test_prism_cache_round_trip(prism_dir, cache_dir):
    import pandas as pd

    first = loader.load_prism(prism_dir, ["prism_lte"], cache_dir)
    assert any(cache_dir.glob("prism_lte-*.parquet"))
    second = loader.load_prism(prism_dir, ["prism_lte"], cache_dir)
    pd.testing.assert_frame_equal(
        first.sort_values(["collection", "time"]).reset_index(drop=True),
        second.sort_values(["collection", "time"]).reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# App integration
# --------------------------------------------------------------------------- #


def _run(qp_dir, prism_dir, tmp_path, monkeypatch, timeout=120):
    monkeypatch.setenv("QP_DATA_DIR", str(qp_dir))
    monkeypatch.setenv("PRISM_DATA_DIR", str(prism_dir))
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.run()
    return app


def _clean(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_measurement_source_filter_appears_with_both_sources(
    fixture_dir, prism_dir, tmp_path, monkeypatch
):
    app = _run(fixture_dir, prism_dir, tmp_path, monkeypatch)
    _clean(app)
    source = app.multiselect(key="f_providers")
    assert source.options == ["QP", "PRISM"]


def test_prism_only_selection_changes_metric_options(
    fixture_dir, prism_dir, tmp_path, monkeypatch
):
    app = _run(fixture_dir, prism_dir, tmp_path, monkeypatch)
    _clean(app)
    # pool of both offers throughput (QP PDSCH); PRISM alone does not
    app.multiselect(key="f_providers").select("PRISM").run()
    _clean(app)
    options = app.selectbox(key="metric_choice").options
    assert "Throughput (kbps)" not in options
    assert "RSRP (dBm)" in options


def test_prism_only_pooling_changes_the_sample_count(
    fixture_dir, prism_dir, tmp_path, monkeypatch
):
    app = _run(fixture_dir, prism_dir, tmp_path, monkeypatch)
    _clean(app)
    both = int(app.metric[0].value.replace(",", ""))
    app.multiselect(key="f_providers").select("PRISM").run()
    _clean(app)
    prism_only = int(app.metric[0].value.replace(",", ""))
    assert 0 < prism_only < both


def test_identity_tab_opens_on_prism_decoded_cells(fixture_dir, prism_dir, tmp_path, monkeypatch):
    app = _run(fixture_dir, prism_dir, tmp_path, monkeypatch)
    _clean(app)
    app.multiselect(key="f_providers").select("PRISM").run()
    _clean(app)
    # PRISM-only now opens the Cell identity tab on its decoded eNB/sector/PCI,
    # offering the relationship views (no serving-cell timelines, no QP prompt)
    view = app.radio(key="identity_view")
    assert set(view.options) == {"Hierarchy", "Structure"}
    assert not any("connected-mode" in i.value for i in app.info)


def test_map_can_colour_by_enb(fixture_dir, prism_dir, tmp_path, monkeypatch):
    app = _run(fixture_dir, prism_dir, tmp_path, monkeypatch)
    _clean(app)
    # the "Colour by" selectbox offers the identity levels including eNB / gNB
    options = None
    for sb in app.selectbox:
        if sb.label == "Colour by":
            options = sb.options
    assert options is not None
    assert "Metric" in options and "eNB / gNB-Id" in options
