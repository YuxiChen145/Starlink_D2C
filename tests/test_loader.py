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
