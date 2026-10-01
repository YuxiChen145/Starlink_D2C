import numpy as np
import pandas as pd
import pytest

from qpviz import loader
from qpviz.schema import DATASETS_BY_KEY, DIMENSION_COLUMNS


@pytest.mark.parametrize(
    "raw,tech,expected",
    [
        (25, "LTE", "B25"),
        ("25", "LTE", "B25"),
        ("LTE E-UTRA 25", "LTE", "B25"),
        ("66", "LTE", "B66"),
        (25, "NR", "n25"),
        ("5G NR n41", "NR", "n41"),
        ("Unknown", "NR", "Unknown"),
        ("", "LTE", "Unknown"),
        (None, "LTE", "Unknown"),
        (float("nan"), "LTE", "Unknown"),
    ],
)
def test_normalise_band(raw, tech, expected):
    assert loader.normalise_band(raw, tech) == expected


def test_parse_time_handles_both_formats():
    parsed = loader.parse_time(
        pd.Series(["2026-07-22 14:17:33.598", "2026-07-22 14:17:34", "", None])
    )
    assert parsed.iloc[0] == pd.Timestamp("2026-07-22 14:17:33.598")
    assert parsed.iloc[1] == pd.Timestamp("2026-07-22 14:17:34")
    assert parsed.isna().sum() == 2


def test_read_dataset_produces_canonical_columns(lte_radio):
    for column in DIMENSION_COLUMNS:
        assert column in lte_radio.columns
    for metric in ("rsrp", "rsrq", "rssi", "sinr"):
        assert metric in lte_radio.columns
    assert lte_radio["tech"].unique().tolist() == ["LTE"]
    assert lte_radio["source"].unique().tolist() == ["lte_radio"]
    assert set(lte_radio["band"].unique()) == {"B25", "B66"}
    assert lte_radio["pci"].dtype == "Int32"
    assert lte_radio["freq"].dtype == "Int64"
    assert pd.api.types.is_datetime64_any_dtype(lte_radio["time"])


def test_rows_without_any_metric_are_dropped(lte_radio):
    metrics = ["rsrp", "rsrq", "rssi", "sinr"]
    assert lte_radio[metrics].notna().any(axis=1).all()
    # the fixture blanks every 17th row of all four metrics at once
    assert len(lte_radio) < 240


def test_cache_round_trip_is_identical(fixture_dir, cache_dir):
    ds = DATASETS_BY_KEY["lte_radio"]
    first = loader.read_dataset(fixture_dir, ds, cache_dir)
    assert any(cache_dir.glob("lte_radio-*.parquet"))
    second = loader.read_dataset(fixture_dir, ds, cache_dir)
    pd.testing.assert_frame_equal(first, second)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        loader.read_dataset(tmp_path, DATASETS_BY_KEY["lte_radio"], None)


def test_available_datasets_lists_only_present_files(fixture_dir):
    keys = {d.key for d in loader.available_datasets(fixture_dir)}
    assert keys == {
        "lte_radio",
        "nr_radio",
        "lte_pdsch_pc",
        "lte_pusch_pc",
        "lte_pucch",
    }


def test_load_concatenates_and_skips_missing(fixture_dir, cache_dir):
    frame = loader.load(fixture_dir, ["lte_radio", "nr_radio", "lte_pdsch_agg"], cache_dir)
    assert set(frame["tech"].unique()) == {"LTE", "NR"}
    assert set(frame["source"].unique()) == {"lte_radio", "nr_radio"}
    # NR contributes no RSSI, LTE contributes no NR-only column -> NaN filled
    assert frame.loc[frame["tech"] == "NR", "rssi"].isna().all()


def test_load_with_no_valid_keys_returns_empty_frame(fixture_dir):
    frame = loader.load(fixture_dir, ["does_not_exist"], None)
    assert frame.empty
    assert "rsrp" in frame.columns


def test_pucch_gets_null_geo_columns(fixture_dir, cache_dir):
    frame = loader.read_dataset(fixture_dir, DATASETS_BY_KEY["lte_pucch"], cache_dir)
    assert frame["lat"].isna().all() and frame["lon"].isna().all()


def test_freq_label_for():
    lte = pd.DataFrame({"tech": ["LTE", "LTE"]})
    nr = pd.DataFrame({"tech": ["NR"]})
    both = pd.DataFrame({"tech": ["LTE", "NR"]})
    assert loader.freq_label_for(lte) == "EARFCN"
    assert loader.freq_label_for(nr) == "NR-ARFCN"
    assert loader.freq_label_for(both) == "EARFCN / NR-ARFCN"
    assert loader.freq_label_for(loader.empty_frame()) == "EARFCN / NR-ARFCN"


def test_options_are_sorted_and_null_free(lte_radio):
    assert loader.options(lte_radio, "pci") == [123, 456, 790]
    assert loader.options(lte_radio, "band") == ["B25", "B66"]
    assert loader.options(lte_radio, "nope") == []


def test_freq_prefix_for_is_short():
    assert loader.freq_prefix_for(pd.DataFrame({"tech": ["LTE"]})) == "EARFCN"
    assert loader.freq_prefix_for(pd.DataFrame({"tech": ["NR"]})) == "NR-ARFCN"
    assert loader.freq_prefix_for(pd.DataFrame({"tech": ["LTE", "NR"]})) == "ARFCN"


def test_spectral_efficiency_derived_metric(fixture_dir, cache_dir):
    ds = DATASETS_BY_KEY["lte_pdsch_pc"]
    frame = loader.read_dataset(fixture_dir, ds, cache_dir)
    # both derived metrics exist and are finite where throughput + RBs are present
    assert "spec_eff" in frame.columns and "tput_per_rb" in frame.columns
    ok = frame["throughput"].notna() & frame["spec_eff"].notna()
    assert ok.any()
    # exact relation: bits/RB = spec_eff x 12 x 15kHz x 1ms = spec_eff x 180 (LTE)
    sub = frame.loc[ok]
    ratio = (sub["tput_per_rb"] / sub["spec_eff"]).round(3)
    assert (ratio == 180.0).all()
    # and they are positive
    assert (sub["spec_eff"] > 0).all()


def test_spectral_efficiency_only_on_pdsch_pusch(fixture_dir, cache_dir):
    # the RSRP radio export carries no throughput/RB, so no spectral efficiency;
    # the PDSCH export does. After pooling, radio rows are NaN, PDSCH rows filled.
    frame = loader.load(fixture_dir, ["lte_radio", "lte_pdsch_pc"], cache_dir)
    assert frame.loc[frame["source"] == "lte_radio", "spec_eff"].isna().all()
    assert frame.loc[frame["source"] == "lte_pdsch_pc", "spec_eff"].notna().any()


def test_n_factor_derived_metric(fixture_dir, cache_dir):
    frame = loader.read_dataset(fixture_dir, DATASETS_BY_KEY["lte_radio"], cache_dir)
    assert "n_factor" in frame.columns
    ok = frame[["rsrp", "rsrq", "rssi"]].notna().all(axis=1)
    # N = 10^((RSRQ - RSRP + RSSI)/10), exactly
    import numpy as np
    expect = 10.0 ** ((frame.loc[ok, "rsrq"] - frame.loc[ok, "rsrp"] + frame.loc[ok, "rssi"]) / 10.0)
    assert np.allclose(frame.loc[ok, "n_factor"], expect, rtol=1e-9)
    assert (frame.loc[ok, "n_factor"] > 0).all()


def test_n_factor_absent_without_rssi(fixture_dir, cache_dir):
    # NR radio has RSRP/RSRQ/SINR but no RSSI, so it must not declare N-factor
    assert "n_factor" not in DATASETS_BY_KEY["nr_radio"].metrics
    frame = loader.load(fixture_dir, ["nr_radio", "lte_radio"], cache_dir)
    assert frame.loc[frame["source"] == "nr_radio", "n_factor"].isna().all()
    assert frame.loc[frame["source"] == "lte_radio", "n_factor"].notna().any()


def test_reads_gzipped_csv(fixture_dir, tmp_path):
    # gzip a copy of an export into a fresh dir; the loader must read it as if raw
    import gzip, shutil
    src = fixture_dir / DATASETS_BY_KEY["lte_radio"].filename
    d = tmp_path / "gz"
    d.mkdir()
    with open(src, "rb") as f, gzip.open(d / (src.name + ".gz"), "wb") as g:
        shutil.copyfileobj(f, g)
    # only the .gz is present
    assert loader._resolve_csv(d, DATASETS_BY_KEY["lte_radio"].filename).name.endswith(".gz")
    assert {x.key for x in loader.available_datasets(d)} == {"lte_radio"}
    frame = loader.read_dataset(d, DATASETS_BY_KEY["lte_radio"], cache_dir=None)
    assert len(frame) > 0 and "rsrp" in frame.columns


def test_prism_collection_strips_gz_suffix():
    from pathlib import Path
    a = loader.prism_collection(Path("2026_09_01_ LTE_8665_Cushing202_static.csv"))
    b = loader.prism_collection(Path("2026_09_01_ LTE_8665_Cushing202_static.csv.gz"))
    assert a == b == "Cushing202 static"


def test_qp_traffic_splits_only_mixed_sessions_by_throughput():
    """QP direction-specific sessions keep their name label; a combined 'Mixed'
    session is split per row by which channel actually carried bulk traffic
    (ACKs on the reverse channel stay below the threshold)."""
    import pandas as pd
    from qpviz import loader
    from qpviz.schema import DATASETS_BY_KEY

    ds = DATASETS_BY_KEY["lte_pdsch_pc"]  # DL channel
    out = pd.DataFrame({
        "collection": [
            "D2C_OpenField_Static_twoPhone1_iperfDL",   # direction-specific
            "D2C_OpenField_Static_twoPhone1_iperfDL",
            "D2C_OpenField_Static_onePhone1_PingiperfULDL",  # Mixed, DL bulk
            "D2C_OpenField_Static_onePhone1_PingiperfULDL",  # Mixed, idle/ping
        ],
        "link": ["DL", "DL", "DL", "DL"],
        "throughput": [3.0, 0.1, 4.0, 0.02],   # Mbps
    })
    raw = pd.DataFrame({"Test Name": ["-", "-", "-", "-"]})
    traffic = loader._qp_traffic(out, raw, ds)
    # a DL-only session is iperf DL on every row, even the low-throughput one
    assert traffic.iloc[0] == "iperf DL"
    assert traffic.iloc[1] == "iperf DL"
    # the Mixed session: bulk DL row -> iperf DL; idle row -> stays Mixed
    assert traffic.iloc[2] == "iperf DL"
    assert traffic.iloc[3] == "Mixed"


def test_qp_traffic_uses_test_name_for_unnamed_sessions():
    import pandas as pd
    from qpviz import loader
    from qpviz.schema import DATASETS_BY_KEY

    ds = DATASETS_BY_KEY["lte_pdsch_pc"]
    out = pd.DataFrame({
        "collection": ["D2C_Boulder_MostStatic_EARFCN8665"] * 3,  # -> "Other"
        "link": ["DL", "DL", "DL"],
        "throughput": [1.0, 1.0, 0.0],
    })
    raw = pd.DataFrame({"Test Name": ["Ookla(R)", "Ping", "-"]})
    traffic = loader._qp_traffic(out, raw, ds)
    assert traffic.iloc[0] == "Ookla"
    assert traffic.iloc[1] == "Ping"
    assert traffic.iloc[2] == "Other"


def test_align_d2c_traffic_relabels_from_sigcap():
    """QP D2C rows (QP-idle PHY captures) inherit the SigCap traffic active at
    the same time; non-D2C and non-QP rows are untouched."""
    import pandas as pd
    from qpviz import loader

    t0 = pd.Timestamp("2026-09-14 19:47:00")
    qp = pd.DataFrame({
        "time": [t0, t0 + pd.Timedelta("2s"), t0 + pd.Timedelta("30s")],
        "collection": ["D2C_..._twoPhone1_PingiperULDL"] * 2 + ["NoForcing_Static_Ookla"],
        "provider": ["QP", "QP", "QP"],
        "traffic": pd.Series(["Ping", "Ping", "Ookla"], dtype="category"),
    })
    ref = pd.DataFrame({
        "time": [t0, t0 + pd.Timedelta("2s")],
        "traffic": ["iperf DL", "iperf UL"],
    })
    out = loader.align_d2c_traffic(qp, ref, tolerance="6s")
    # the two D2C rows are relabelled to the aligned SigCap traffic
    assert out.loc[0, "traffic"] == "iperf DL"
    assert out.loc[1, "traffic"] == "iperf UL"
    # the non-D2C Ookla row (and any far-from-ref row) keeps its own label
    assert out.loc[2, "traffic"] == "Ookla"


def test_align_d2c_traffic_keeps_unmatched_and_nonqp_rows():
    import pandas as pd
    from qpviz import loader
    t0 = pd.Timestamp("2026-09-14 19:47:00")
    frame = pd.DataFrame({
        "time": [t0, t0],
        "collection": ["D2C_x", "D2C_x"],
        "provider": ["QP", "SIGCAP"],   # SigCap D2C row must not be overwritten
        "traffic": pd.Series(["Mixed", "iperf DL"], dtype="category"),
    })
    ref = pd.DataFrame({"time": [t0 + pd.Timedelta("1h")], "traffic": ["Ping"]})  # too far
    out = loader.align_d2c_traffic(frame, ref, tolerance="6s")
    assert out.loc[0, "traffic"] == "Mixed"       # QP row, no match in tolerance
    assert out.loc[1, "traffic"] == "iperf DL"    # SigCap row untouched
