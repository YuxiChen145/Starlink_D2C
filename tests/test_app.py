"""End-to-end smoke tests: run the real Streamlit script headlessly."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(data_dir, timeout=120):
    os.environ["QP_DATA_DIR"] = str(data_dir)
    app = AppTest.from_file(APP, default_timeout=timeout)
    app.run()
    return app


def _assert_clean(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_app_starts_and_renders_every_tab(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    labels = [t.label for t in app.tabs]
    assert any("Map" in l for l in labels)
    assert any("CDF" in l for l in labels)
    assert app.multiselect(key="f_collections") is not None
    assert app.selectbox(key="metric_choice") is not None
    # datasets are resolved from technology + metric, never picked by hand
    assert not [m for m in app.multiselect if m.label == "Datasets"]
    # header metrics + at least one plotly chart rendered
    assert len(app.metric) >= 4


def test_app_reacts_to_filter_changes(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    before = app.metric[0].value

    app.multiselect(key="f_collections").select("Collection_A").run()
    _assert_clean(app)
    after = app.metric[0].value
    assert after != before

    app.multiselect(key="f_pcis").select(123).run()
    _assert_clean(app)


def test_app_switches_metric_and_technology(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    assert app.radio(key="tech_choice").options == ["All", "LTE", "5G NR"]

    app.radio(key="tech_choice").set_value("5G NR").run()
    _assert_clean(app)
    metric = app.selectbox(key="metric_choice")
    # only the NR radio export is present in the fixtures, so only its metrics
    assert metric.options == ["RSRP (dBm)", "RSRQ (dB)", "SINR (dB)"]
    metric.set_value("sinr").run()
    _assert_clean(app)
    assert app.selectbox(key="metric_choice").value == "sinr"


def test_metric_with_both_directions_defaults_to_downlink(
    fixture_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    app.radio(key="tech_choice").set_value("LTE").run()
    app.selectbox(key="metric_choice").set_value("throughput").run()
    _assert_clean(app)

    direction = app.multiselect(key="f_links_DL-UL")
    assert direction.options == ["DL", "UL"]
    assert direction.value == ["DL"]

    downlink_only = int(app.metric[0].value.replace(",", ""))
    direction.select("UL").run()
    _assert_clean(app)
    assert int(app.metric[0].value.replace(",", "")) > downlink_only


def test_data_source_filter_narrows_the_pooled_exports(
    fixture_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    app.radio(key="tech_choice").set_value("LTE").run()
    app.selectbox(key="metric_choice").set_value("throughput").run()
    _assert_clean(app)

    sources = app.multiselect(key="f_sources")
    # the DL default leaves only the PDSCH export in play; options render as labels
    assert sources.options == ["LTE PDSCH [per-carrier]"]
    sources.select("lte_pdsch_pc").run()
    _assert_clean(app)
    assert app.multiselect(key="f_sources").value == ["lte_pdsch_pc"]


REAL_DATA = Path(__file__).resolve().parents[1] / "QP CSV"


@pytest.mark.skipif(not REAL_DATA.exists(), reason="real QP CSV export not present")
def test_app_starts_against_the_real_export():
    app = _run(REAL_DATA, timeout=300)
    _assert_clean(app)
    assert int(app.metric[0].value.replace(",", "")) > 0


def test_cell_identity_tab_renders_both_views(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)

    labels = [t.label for t in app.tabs]
    assert any("Cell identity" in l for l in labels)

    # timeline is the default view, with its four summary metrics
    view = app.radio(key="identity_view")
    assert view.value == "Timeline"
    assert any("Distinct" in m.label for m in app.metric)

    level = app.selectbox(key="identity_level")
    # technology is "All", so the NR-only SSB index is offered too
    assert level.options == ["eNB / gNB-Id", "Sector Id", "PCI", "SSB Index"]
    level.set_value("pci").run()
    _assert_clean(app)

    app.selectbox(key="identity_order").set_value("dwell").run()
    _assert_clean(app)
    app.selectbox(key="identity_x").set_value("absolute").run()
    _assert_clean(app)

    view.set_value("Hierarchy").run()
    _assert_clean(app)
    hierarchy = app.multiselect(key="identity_hierarchy")
    assert hierarchy.value == ["enb", "sector", "pci"]
    assert app.number_input(key="identity_leaves").value == 150


def test_cell_identity_has_no_collection_picker_of_its_own(
    fixture_dir, monkeypatch, tmp_path
):
    """The sidebar Collections filter governs this tab like every other."""
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    assert not [m for m in app.multiselect if m.label == "Collections to draw"]

    runs_before = int(next(m for m in app.metric if m.label == "Dwell runs").value)
    app.multiselect(key="f_collections").select("Collection_A").run()
    _assert_clean(app)
    runs_after = int(next(m for m in app.metric if m.label == "Dwell runs").value)
    assert runs_after < runs_before


def test_cell_identity_survives_a_metric_with_no_identity_columns(
    fixture_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    # throughput comes from the PDSCH/PUSCH exports, which carry no eNB-Id -
    # the tab must still work, reading the Radio exports underneath
    app.radio(key="tech_choice").set_value("LTE").run()
    app.selectbox(key="metric_choice").set_value("throughput").run()
    _assert_clean(app)
    assert app.selectbox(key="identity_level").options == [
        "eNB / gNB-Id",
        "Sector Id",
        "PCI",
    ]
