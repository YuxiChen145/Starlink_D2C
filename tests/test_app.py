"""End-to-end smoke tests: run the real Streamlit script headlessly."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from qpviz import schema  # noqa: E402
from tests.conftest import open_tab  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _run(data_dir, timeout=120, prism_dir=None, sigcap_dir=None):
    os.environ["QP_DATA_DIR"] = str(data_dir)
    # Isolate from any real PRISM / SigCap folder next to app.py: unless a test
    # opts in, point them at directories with no CSVs so these QP assertions stay
    # QP-only.
    os.environ["PRISM_DATA_DIR"] = str(prism_dir) if prism_dir else "/no-prism-dir"
    os.environ["SIGCAP_DATA_DIR"] = str(sigcap_dir) if sigcap_dir else "/no-sigcap-dir"
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


def test_only_the_open_tab_runs(fixture_dir, monkeypatch, tmp_path):
    """Tabs are lazy: the Map is open on arrival and the others stay unbuilt
    until opened, so a first visit does not load every tab's data."""
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    assert app.session_state["main_tab"] == "🗺️ Map"
    assert not [w for w in app.multiselect if w.key == "stat_group"]

    open_tab(app, "Statistics")
    _assert_clean(app)
    assert app.multiselect(key="stat_group") is not None
    # a sidebar change keeps the same tab open
    app.multiselect(key="f_collections").select("Collection_A").run()
    _assert_clean(app)
    assert app.session_state["main_tab"] == "📊 Statistics"
    assert app.multiselect(key="stat_group") is not None


def test_load_frame_pools_one_shared_copy_per_export(fixture_dir, tmp_path):
    import pandas as pd

    from qpviz import loader
    from qpviz.ui import common

    common._qp_dataset.clear()
    keys = ("lte_radio", "nr_radio")
    pooled = common.load_frame(str(fixture_dir), keys)
    pd.testing.assert_frame_equal(
        pooled, loader.load(fixture_dir, list(keys), cache_dir=tmp_path / "cache")
    )
    # each export is held once and shared by every caller...
    first = common._qp_dataset(str(fixture_dir), "lte_radio")
    assert common._qp_dataset(str(fixture_dir), "lte_radio") is first
    # ...while each caller gets a frame of its own to work on
    assert common.load_frame(str(fixture_dir), keys) is not pooled


def test_every_tab_opens_cleanly(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    for label in [t.label for t in app.tabs]:
        open_tab(app, label)
        _assert_clean(app)


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
    app.selectbox(key="metric_choice").set_value("mcs").run()
    _assert_clean(app)

    direction = app.multiselect(key="f_links_DL-UL")
    assert direction.options == ["DL", "UL"]
    assert direction.value == ["DL"]

    downlink_only = int(app.metric[0].value.replace(",", ""))
    direction.select("UL").run()
    _assert_clean(app)
    assert int(app.metric[0].value.replace(",", "")) > downlink_only


def _captions(app):
    return [c.value for c in app.caption]


@pytest.mark.parametrize(
    "metric, export, direction",
    [
        ("rsrp", "LTE Radio [connected]", ["DL"]),
        ("tput_dl", "LTE PDSCH [aggregated]", ["DL"]),
        ("tput_dl_cc", "LTE PDSCH [per-carrier]", ["DL"]),
        ("tput_ul", "LTE PUSCH [per-carrier]", ["UL"]),
        ("tput_ul_cc", "LTE PUSCH [per-carrier]", ["UL"]),
        ("bler", "LTE PDSCH [per-carrier]", ["DL"]),
        ("sf_usage", "LTE PDSCH [per-carrier]", ["DL", "UL"]),
    ],
)
def test_each_metric_reads_one_fixed_export_named_with_the_plots(
    fixture_dir, monkeypatch, tmp_path, metric, export, direction
):
    """There is no Data source filter: the metric decides the export, and the
    page says which one it is."""
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    app.radio(key="tech_choice").set_value("LTE").run()
    app.selectbox(key="metric_choice").set_value(metric).run()
    _assert_clean(app)

    assert "f_sources" not in app.session_state
    assert "Data source" not in [m.label for m in app.multiselect]
    label = schema.METRICS[metric].label
    assert f"{label} is read from **{export}**." in _captions(app)
    assert app.multiselect(key=f"f_links_{'-'.join(direction)}").options == direction


REAL_DATA = Path(__file__).resolve().parents[1] / "QP CSV"


@pytest.mark.skipif(not REAL_DATA.exists(), reason="real QP CSV export not present")
def test_app_starts_against_the_real_export():
    app = _run(REAL_DATA, timeout=300)
    _assert_clean(app)
    assert int(app.metric[0].value.replace(",", "")) > 0


def test_cell_identity_opens_on_the_hierarchy(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)

    labels = [t.label for t in app.tabs]
    assert any("Cell identity" in l for l in labels)
    open_tab(app, "Cell identity")
    _assert_clean(app)

    # the node tree leads, the counts behind it come second
    view = app.radio(key="identity_view")
    assert view.options == ["Hierarchy", "Structure"]
    assert view.value == "Hierarchy"

    # D2C and terrestrial are separate trees, D2C first
    network = app.radio(key="identity_network")
    assert network.options == ["D2C", "Terrestrial"]
    assert network.value == "D2C"

    # the levels are fixed to eNB, sector, PCI: nothing to pick
    assert not [m for m in app.multiselect if "Levels" in m.label]
    assert not [n for n in app.number_input if n.label in ("Max leaves drawn", "Columns")]

    # the tab opens on the per-eNB summary
    summary = next(
        d.value for d in app.dataframe if "Sectors seen" in list(d.value.columns)
    )
    assert list(summary.columns) == ["eNB ID", "Sectors seen", "PCIs seen", "Minutes"]
    assert len(summary) > 1

    # and one download serves the whole tab
    assert len(app.get("download_button")) == 1


def test_hierarchy_orders_enbs_by_sectors_or_by_id(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    open_tab(app, "Cell identity")
    _assert_clean(app)

    def _summary():
        return next(
            d.value for d in app.dataframe if "Sectors seen" in list(d.value.columns)
        )

    order = app.radio(key="identity_enb_order")
    assert order.options == ["Number of sectors", "eNB ID"]   # not by time held
    assert order.value == "sectors"
    sectors = _summary()["Sectors seen"].tolist()
    assert sectors == sorted(sectors, reverse=True)

    order.set_value("enb").run()
    _assert_clean(app)
    ids = _summary()["eNB ID"].tolist()
    assert ids == sorted(ids)


def test_hierarchy_expands_one_enb_into_its_tree(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    open_tab(app, "Cell identity")
    _assert_clean(app)

    summary = next(
        d.value for d in app.dataframe if "Sectors seen" in list(d.value.columns)
    )
    focus = app.selectbox(key="identity_focus_D2C")
    # no "All": the table is the overview, the tree is always one eNB,
    # offered in the order of the table and opening on its first row
    assert [int(o) for o in focus.options] == summary["eNB ID"].tolist()
    assert focus.value == summary["eNB ID"].iloc[0]
    focus.set_value(int(focus.options[1])).run()
    _assert_clean(app)

    # the other network keeps its own choice of eNB
    app.radio(key="identity_network").set_value("Terrestrial").run()
    _assert_clean(app)
    assert app.selectbox(key="identity_focus_Terrestrial").value is not None


def test_cell_identity_follows_the_sidebar_collections(
    fixture_dir, monkeypatch, tmp_path
):
    """The sidebar Collections filter governs this tab like every other."""
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    open_tab(app, "Cell identity")
    _assert_clean(app)
    assert not [m for m in app.multiselect if "Collections" in m.label
                and m.key != "f_collections"]

    def _minutes():
        table = next(
            d.value for d in app.dataframe if "Sectors seen" in list(d.value.columns)
        )
        return table["Minutes"].sum()

    before = _minutes()
    app.multiselect(key="f_collections").select("Collection_A").run()
    _assert_clean(app)
    assert _minutes() < before


def test_cell_identity_survives_a_metric_with_no_identity_columns(
    fixture_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    open_tab(app, "Cell identity")
    _assert_clean(app)
    # throughput comes from the PDSCH/PUSCH exports, which carry no eNB-Id -
    # the tab must still work, reading the Radio exports underneath
    app.radio(key="tech_choice").set_value("LTE").run()
    app.selectbox(key="metric_choice").set_value("tput_dl_cc").run()
    _assert_clean(app)
    assert app.radio(key="identity_view").value == "Hierarchy"
    assert app.selectbox(key="identity_focus_D2C").options


def test_cell_identity_has_no_time_views(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    open_tab(app, "Cell identity")
    _assert_clean(app)
    # Timeline and Change timeline followed one session, not the network
    assert app.radio(key="identity_view").options == ["Hierarchy", "Structure"]
    assert not [m for m in app.metric if m.label == "Dwell runs"]
    # the NR-only SSB index is no longer a level anywhere in the tab
    assert not [b for b in app.selectbox if "SSB Index" in b.options]


def test_cell_identity_shows_prism_decoded_identity(prism_dir, monkeypatch, tmp_path):
    # PRISM decodes eNB/sector/PCI on some scan rows; with no QP present the
    # Cell identity tab must still open on those.
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    empty_qp = tmp_path / "no_qp"
    empty_qp.mkdir()
    app = _run(empty_qp, prism_dir=prism_dir)
    _assert_clean(app)

    assert any("Cell identity" in t.label for t in app.tabs)
    open_tab(app, "Cell identity")
    _assert_clean(app)
    view = app.radio(key="identity_view")
    assert view.options == ["Hierarchy", "Structure"]
    # PRISM's decoded eNB / sector / PCI drive the hierarchy
    summary = next(
        d.value for d in app.dataframe if "Sectors seen" in list(d.value.columns)
    )
    assert len(summary) >= 1
    view.set_value("Structure").run()
    _assert_clean(app)


def test_cell_identity_pools_qp_serving_and_prism_scans(
    fixture_dir, prism_dir, monkeypatch, tmp_path
):
    # with both sources, the QP serving cells and the PRISM scans feed one tree
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir, prism_dir=prism_dir)
    _assert_clean(app)
    open_tab(app, "Cell identity")
    _assert_clean(app)
    source = next(c.value for c in app.caption if c.value.startswith("Cell identity from"))
    assert "QP serving cell" in source and "PRISM decoded scans" in source
    assert "SigCap" not in source   # PCI only: it cannot be placed in the tree


def test_metric_picker_offers_only_the_metrics_in_metric_order(
    fixture_dir, monkeypatch, tmp_path
):
    from qpviz.schema import METRIC_ORDER, METRICS

    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    options = app.selectbox(key="metric_choice").options
    assert set(options) <= {METRICS[m].axis_label for m in METRIC_ORDER}
    for cut in ("CQI (index)", "Tx Power (dBm)", "DL Pathloss (dB)",
                "Throughput per RB (bits/RB)", "App throughput (Mbps)"):
        assert cut not in options


def test_pci_set_modes_offer_the_union_and_the_intersection(
    fixture_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)

    mode = app.radio(key="f_pci_mode")
    assert mode.value == "Any collection"
    # AppTest renders option labels, so these come back as strings
    assert app.multiselect(key="f_pcis").options == ["123", "456", "790"]

    # every fixture collection carries all three PCIs, so the shared set matches
    mode.set_value("Shared by all").run()
    _assert_clean(app)
    assert app.multiselect(key="f_pcis").options == ["123", "456", "790"]

    # drop a collection down to a single PCI and only that one stays shared
    app.multiselect(key="f_collections").select("Collection_A").run()
    _assert_clean(app)
    assert app.multiselect(key="f_pcis").options == ["123", "456", "790"]


def test_shared_mode_restricts_the_data_to_the_intersection(
    fixture_dir, monkeypatch, tmp_path
):
    """Shared mode filters even with nothing picked - that is the point."""
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    everything = int(app.metric[0].value.replace(",", ""))

    app.radio(key="f_pci_mode").set_value("Shared by all").run()
    _assert_clean(app)
    # all three PCIs are shared here, so the sample count is unchanged...
    assert int(app.metric[0].value.replace(",", "")) == everything

    # ...but narrowing to one PCI in one collection shrinks it
    app.multiselect(key="f_pcis").select(123).run()
    _assert_clean(app)
    assert int(app.metric[0].value.replace(",", "")) < everything


def test_structure_view_compares_the_two_networks(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    open_tab(app, "Cell identity")
    _assert_clean(app)

    app.radio(key="identity_view").set_value("Structure").run()
    _assert_clean(app)
    # nothing to set: the levels are fixed and the split is always by network
    assert not [m for m in app.multiselect if "Levels" in m.label]
    assert not [b for b in app.selectbox if b.label in ("Split by", "Chart")]
    assert not [c for c in app.checkbox if c.label == "Show as CDF"]

    summary = next(d.value for d in app.dataframe if "Network" in list(d.value.columns))
    assert summary["Network"].tolist() == ["D2C", "Terrestrial"]
    assert list(summary.columns) == [
        "Network", "eNBs seen", "Sectors seen", "PCIs seen",
        "Sectors per eNB (median)", "Sectors per eNB (most)",
        "PCIs per sector (median)", "PCIs per sector (most)",
    ]
    # one chart per count, each named in plain words above it
    titles = [m.value for m in app.markdown]
    assert "**Sectors seen per eNB**" in titles
    assert "**PCIs seen per sector**" in titles
    assert len(app.get("download_button")) == 1


def test_cell_identity_drops_a_network_the_filters_remove(
    fixture_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    open_tab(app, "Cell identity")
    _assert_clean(app)
    app.radio(key="identity_network").set_value("Terrestrial").run()
    _assert_clean(app)

    # keep the D2C carrier only: the stored "Terrestrial" choice must not break
    app.multiselect(key="f_freqs").select(8665).run()
    _assert_clean(app)
    assert not [r for r in app.radio if r.key == "identity_network"]
    assert app.selectbox(key="identity_focus_D2C").options


def test_time_window_filter_narrows_samples(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    before = int(app.metric[0].value.replace(",", ""))

    app.checkbox(key="use_time_window").set_value(True).run()
    _assert_clean(app)
    # a 1-minute window is shorter than the fixture's ~2-minute span
    app.number_input(key="time_len").set_value(1).run()
    _assert_clean(app)
    after = int(app.metric[0].value.replace(",", ""))
    assert after < before


def test_later_tabs_survive_a_missing_identity_source(tmp_path, monkeypatch):
    """With only a PDSCH export there is no cell-identity source; the Cell
    identity tab must say so and leave Statistics / Data rendering (it used to
    call st.stop() and blank every tab after it)."""
    from qpviz.schema import DATASETS_BY_KEY
    from tests.conftest import write_dataset
    d = tmp_path / "qp_pdsch_only"
    d.mkdir()
    write_dataset(d, DATASETS_BY_KEY["lte_pdsch_pc"])
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(d)
    _assert_clean(app)
    open_tab(app, "Cell identity")
    _assert_clean(app)
    assert any("Cell identity needs" in i.value for i in app.info)
    # the Statistics tab (after Cell identity) still renders its controls
    open_tab(app, "Statistics")
    _assert_clean(app)
    assert app.multiselect(key="stat_group") is not None


def test_filter_selection_is_pruned_when_options_cascade(fixture_dir, monkeypatch, tmp_path):
    """A value picked in a downstream filter must not linger as an invisible
    filter once an upstream change (here: Collections) no longer offers it."""
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)
    # pick a PCI with every collection selected
    first = int(app.multiselect(key="f_pcis").options[0])
    app.multiselect(key="f_pcis").select(first).run()
    _assert_clean(app)
    assert app.multiselect(key="f_pcis").value == [first]
    # now restrict collections so the option lists downstream shrink
    colls = app.multiselect(key="f_collections").options
    app.multiselect(key="f_collections").set_value(colls[:1]).run()
    _assert_clean(app)
    opts = [int(o) for o in app.multiselect(key="f_pcis").options]
    value = app.multiselect(key="f_pcis").value
    # invariant the fix guarantees: whatever is selected is in the options
    assert all(v in opts for v in value)


# --------------------------------------------------------------------------- #
# The trimmed sidebar and in-tab controls
# --------------------------------------------------------------------------- #


def _labels(widgets):
    return [w.label for w in widgets]


def test_sidebar_keeps_min_samples_and_drops_the_cut_options(
    fixture_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)

    assert "Min samples per curve" in _labels(app.sidebar.number_input)
    for gone in ("Max points per curve", "Max points on map"):
        assert gone not in _labels(app.sidebar.number_input)
    for gone in ("Map colour scale", "Basemap"):
        assert gone not in _labels(app.sidebar.selectbox)
    for gone in ("Require a known PCI and carrier", "Keep rows without GPS"):
        assert gone not in _labels(app.sidebar.checkbox)
    # the region is cleared from beside the map, not from the sidebar
    assert "Clear" not in _labels(app.sidebar.button)


def test_rows_without_a_pci_or_carrier_are_always_left_out(
    fixture_dir, monkeypatch, tmp_path
):
    from qpviz import loader

    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    blank = tmp_path / "qp"
    blank.mkdir()
    for path in Path(fixture_dir).iterdir():
        frame = pd.read_csv(path)
        if path.name.startswith("LTE Radio"):
            frame.loc[frame.index[:30], "PCI LTE"] = None   # PCI never decoded
            frame.loc[frame.index[30:50], "DL EARFCN"] = None  # nor the carrier
        frame.to_csv(blank / path.name, index=False)

    app = _run(blank)
    _assert_clean(app)
    app.radio(key="tech_choice").set_value("LTE").run()
    _assert_clean(app)
    radio = loader.read_dataset(blank, schema.DATASETS_BY_KEY["lte_radio"], None)
    known = radio[radio["rsrp"].notna() & radio["pci"].notna() & radio["freq"].notna()]
    assert 0 < len(known) < int(radio["rsrp"].notna().sum())
    assert int(app.metric[0].value.replace(",", "")) == len(known)


def test_map_controls_are_colour_by_and_clear_region(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    _assert_clean(app)

    colour_by = next(s for s in app.selectbox if s.label == "Colour by")
    assert colour_by.options == ["Metric", "eNB ID"]
    assert "Colour scheme" not in _labels(app.radio)
    assert not {"Marker size", "Coloured cells"} & set(_labels(app.slider))
    assert "Diverging midpoint" not in _labels(app.number_input)
    assert "Apply region to map" not in _labels(app.checkbox)
    # the fixed colour range is stated next to the map
    assert any("always span -140 to -80 dBm" in c.value for c in app.caption)

    # with nothing drawn there is nothing to clear
    clear = next(b for b in app.button if b.label == "Clear region")
    assert clear.disabled
    everything = int(app.metric[0].value.replace(",", ""))

    # A region stays put while the Map tab is open (the map reporting no shapes
    # must not remove it), and "Clear region" brings all the data back.
    app.session_state["regions"] = [[(41.699, -86.2405), (41.699, -86.235),
                                     (41.705, -86.235), (41.705, -86.2405)]]
    app.run()
    _assert_clean(app)
    inside = int(app.metric[0].value.replace(",", ""))
    assert 0 < inside < everything
    app.run()
    assert int(app.metric[0].value.replace(",", "")) == inside
    clear = next(b for b in app.button if b.label == "Clear region")
    assert not clear.disabled
    clear.click().run()
    _assert_clean(app)
    assert app.session_state["regions"] == []
    assert int(app.metric[0].value.replace(",", "")) == everything
    assert next(b for b in app.button if b.label == "Clear region").disabled


def test_a_drawn_region_narrows_the_other_tabs(fixture_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    open_tab(app, "Data")
    _assert_clean(app)
    everything = int(app.metric[0].value.replace(",", ""))
    app.session_state["regions"] = [[(41.699, -86.2405), (41.699, -86.235),
                                     (41.705, -86.235), (41.705, -86.2405)]]
    app.run()
    _assert_clean(app)
    assert 0 < int(app.metric[0].value.replace(",", "")) < everything


def test_map_colours_by_enb_only_when_the_metric_carries_it(
    fixture_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    app.radio(key="tech_choice").set_value("LTE").run()
    app.selectbox(key="metric_choice").set_value("tput_dl").run()
    _assert_clean(app)
    colour_by = next(s for s in app.selectbox if s.label == "Colour by")
    assert colour_by.options == ["Metric"]   # the PDSCH export has no eNB-Id


def test_cdf_splits_by_network_first_and_plots_cdf_only(
    fixture_dir, monkeypatch, tmp_path
):
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = _run(fixture_dir)
    open_tab(app, "CDF")
    _assert_clean(app)

    split = next(m for m in app.multiselect if m.label == "Split curves by")
    assert split.options == ["Network", "Collection", "Measurement source", "eNB / gNB"]
    assert split.value == ["network"]
    assert "Plot as" not in _labels(app.radio)
    assert "Histogram bins" not in _labels(app.slider)
    # D2C (EARFCN 8665) against terrestrial: two curves
    assert any("2 group(s)" in c.value for c in app.caption)

    # a metric whose export has no eNB-Id does not offer the eNB split
    app.radio(key="tech_choice").set_value("LTE").run()
    app.selectbox(key="metric_choice").set_value("tput_dl").run()
    _assert_clean(app)
    split = next(m for m in app.multiselect if m.label == "Split curves by")
    assert split.options == ["Network", "Collection", "Measurement source"]
    assert split.value == ["network"]
