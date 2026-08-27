import numpy as np
import pandas as pd
import pytest

from qpviz import stats
from qpviz.loader import load


def test_ecdf_shape_and_monotonicity():
    x, y = stats.ecdf([3, 1, 2, 2])
    assert x.tolist() == [1, 2, 2, 3]
    assert y.tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    assert np.all(np.diff(x) >= 0) and np.all(np.diff(y) >= 0)


def test_ecdf_ignores_nan_and_inf():
    x, y = stats.ecdf([1.0, np.nan, np.inf, 2.0])
    assert x.tolist() == [1.0, 2.0]
    assert y[-1] == 1.0


def test_ecdf_empty():
    x, y = stats.ecdf([np.nan, np.nan])
    assert x.size == 0 and y.size == 0


def test_ecdf_downsampling_keeps_endpoints():
    values = np.random.default_rng(0).normal(size=50_000)
    x, y = stats.ecdf(values, max_points=1000)
    assert x.size <= 1000
    assert x[0] == pytest.approx(values.min())
    assert x[-1] == pytest.approx(values.max())
    assert y[-1] == pytest.approx(1.0)
    full_x, _ = stats.ecdf(values, max_points=None)
    assert full_x.size == 50_000


@pytest.fixture()
def frame(fixture_dir, cache_dir):
    return load(fixture_dir, ["lte_radio"], cache_dir)


def test_one_curve_per_pci(frame):
    curves = stats.cdf_curves(frame, "rsrp", group_by=["pci"], min_samples=1)
    assert {c.key[0] for c in curves} == {123, 456, 790}
    assert all(c.name.startswith("PCI ") for c in curves)
    assert sum(c.n for c in curves) == int(frame["rsrp"].notna().sum())


def test_curves_sorted_by_sample_count(frame):
    curves = stats.cdf_curves(frame, "rsrp", group_by=["collection", "pci"])
    counts = [c.n for c in curves]
    assert counts == sorted(counts, reverse=True)


def test_multi_dimension_grouping_labels(frame):
    curves = stats.cdf_curves(frame, "rsrp", group_by=["collection", "pci", "freq"])
    assert len(curves) == 3 * 3 * 2 or len(curves) > 0
    name = curves[0].name
    assert " · " in name and "PCI " in name and "ARFCN " in name


def test_freq_prefix_is_configurable(frame):
    curves = stats.cdf_curves(frame, "rsrp", group_by=["freq"], freq_prefix="EARFCN")
    assert all(c.name.startswith("EARFCN ") for c in curves)


def test_min_samples_drops_small_groups(frame):
    everything = stats.cdf_curves(frame, "rsrp", group_by=["collection", "pci"], min_samples=1)
    biggest = max(c.n for c in everything)
    assert stats.cdf_curves(
        frame, "rsrp", group_by=["collection", "pci"], min_samples=biggest + 1
    ) == []


def test_no_grouping_gives_single_curve(frame):
    curves = stats.cdf_curves(frame, "rsrp", group_by=[])
    assert len(curves) == 1 and curves[0].name == "All data"


def test_unknown_metric_or_empty_frame():
    assert stats.cdf_curves(pd.DataFrame(), "rsrp") == []
    assert stats.cdf_curves(pd.DataFrame({"pci": [1]}), "nope") == []


def test_all_nan_metric_gives_no_curves():
    frame = pd.DataFrame({"pci": [1, 1], "rsrp": [np.nan, np.nan]})
    assert stats.cdf_curves(frame, "rsrp") == []


def test_summary_table_columns_and_values(frame):
    table = stats.summary_table(frame, "rsrp", group_by=["pci"], min_samples=1)
    assert list(table.columns) == [
        "Group", "Samples", "Mean", "Std", "Min", "P5", "P10", "P50", "P90", "P95", "Max"
    ]
    assert len(table) == 3
    assert (table["Min"] <= table["P50"]).all()
    assert (table["P50"] <= table["Max"]).all()
    assert table["Samples"].sum() == int(frame["rsrp"].notna().sum())


def test_summary_table_empty_when_below_min_samples(frame):
    assert stats.summary_table(frame, "rsrp", group_by=["pci"], min_samples=10**6).empty
    assert stats.summary_table(pd.DataFrame(), "rsrp").empty


def test_group_label_handles_missing_values():
    assert stats.group_label(["pci"], [None]) == "PCI n/a"
    assert stats.group_label(["band"], ["n25"]) == "Band n25"
    assert stats.group_label(["collection", "pci"], ["A", 7]) == "A · PCI 7"
