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
    # space-separated and day-less variants seen in the wild
    assert loader.prism_collection(
        Path("2026_09_30 LTE_8665 ORD to DCA in flight 0.csv")
    ) == "ORD to DCA in flight 0"
    assert loader.prism_collection(
        Path("2026_09_LTE_8665_Chicago_indoors_static_41806989La_-87594799Lo.csv")
    ) == "Chicago indoors static 41806989La -87594799Lo"
    assert loader.prism_collection(Path("2026_09_18LTE_8665_emptySta.csv")) == "emptySta"


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


# --------------------------------------------------------------------------- #
# Timestamps: the scanner clock vs the GPS clock
# --------------------------------------------------------------------------- #


def _prism_rows(sys_times, gps_times, with_geo=True):
    import pandas as pd
    raw = pd.DataFrame({
        "system_date_time": sys_times,
        "gps_date_time": gps_times,
        "physical_cell_id": [1] * len(sys_times),
        "rsrp": [-100.0] * len(sys_times),
    })
    if with_geo:
        raw["latitude"] = 41.7
        raw["longitude"] = -86.2
    return raw


def test_prism_timestamps_keep_system_clock_when_it_agrees_with_gps():
    import pandas as pd
    sys_t = ["2026-09-15T17:15:13.157", "2026-09-15T17:15:13.650", "2026-09-15T17:15:14.100"]
    gps_t = ["2026-09-15T17:15:12.0", "", "2026-09-15T17:15:13.0"]
    out = loader.prism_timestamps(_prism_rows(sys_t, gps_t))
    # a ~1 s lag is normal: keep the millisecond system clock untouched
    assert out.tolist() == pd.to_datetime(sys_t).tolist()


def test_prism_timestamps_correct_an_unset_system_clock_from_gps():
    import pandas as pd
    # the real failure mode: system clock reads 2070, GPS has the right day
    sys_t = ["2070-01-01T00:03:52.659", "2070-01-01T00:03:54.595", "2070-01-01T00:03:56.000"]
    gps_t = ["2026-09-30T23:14:38.0", "2026-09-30T23:14:40.0", ""]
    out = loader.prism_timestamps(_prism_rows(sys_t, gps_t))
    assert (out.dt.year == 2026).all()
    # cadence (and the row without a fix) preserved: shifted by one offset
    deltas = out.diff().dropna()
    assert deltas.tolist() == pd.to_datetime(sys_t).to_series().diff().dropna().tolist()
    assert abs((out.iloc[0] - pd.Timestamp("2026-09-30T23:14:38")).total_seconds()) < 2


def test_prism_timestamps_without_any_gps_fix_are_left_as_recorded(caplog):
    import logging
    sys_t = ["2070-01-01T00:07:46.000", "2070-01-01T00:07:47.000"]
    with caplog.at_level(logging.WARNING, logger="qpviz.loader"):
        out = loader.prism_timestamps(_prism_rows(sys_t, ["", ""], with_geo=False), name="x.csv")
    assert (out.dt.year == 2070).all()
    assert any("x.csv" in r.message and "no GPS" in r.message for r in caplog.records)


def test_load_prism_uses_corrected_time(prism_dir, cache_dir, tmp_path):
    """A 2070-clock file lands on its GPS day in the pooled frame."""
    import pandas as pd
    from tests.conftest import _PRISM_LTE_COLUMNS, _write_prism
    d = tmp_path / "prism"
    d.mkdir()
    path = _write_prism(d, "2026_09_30_LTE_8665_Bad_Clock.csv", _PRISM_LTE_COLUMNS,
                        "lte", 8665, rows=20, seed=9)
    raw = pd.read_csv(path)
    raw["gps_date_time"] = pd.to_datetime(raw["system_date_time"]).dt.strftime("%Y-%m-%dT%H:%M:%S.0")
    raw["system_date_time"] = (
        pd.to_datetime(raw["system_date_time"]) + pd.DateOffset(years=44)
    ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]
    raw.loc[raw.index % 3 == 0, "gps_date_time"] = ""   # some rows without a fix
    raw.to_csv(path, index=False)
    frame = loader.load_prism(d, ["prism_lte"], cache_dir=None)
    assert len(frame) == 20
    assert (frame["time"].dt.year == 2026).all()
