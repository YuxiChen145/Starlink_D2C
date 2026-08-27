import numpy as np
import pandas as pd
import pytest

from qpviz import filters as flt
from qpviz.loader import load


SQUARE = flt.bbox_polygon(0.0, 10.0, 0.0, 10.0)


def test_points_in_polygon_basic():
    lat = [5.0, 15.0, 0.5, -1.0, np.nan]
    lon = [5.0, 5.0, 9.5, 5.0, 5.0]
    got = flt.points_in_polygon(lat, lon, SQUARE)
    assert got.tolist() == [True, False, True, False, False]


def test_points_in_polygon_closed_ring_and_degenerate():
    closed = SQUARE + [SQUARE[0]]
    assert flt.points_in_polygon([5.0], [5.0], closed).tolist() == [True]
    assert flt.points_in_polygon([5.0], [5.0], SQUARE[:2]).tolist() == [False]


def test_points_in_polygon_concave():
    # An L in (lat, lon): the full lon span only exists for lat <= 4, so the
    # notch at lat 8 / lon 2 must fall outside.
    poly = [(0, 0), (0, 10), (10, 10), (10, 6), (4, 6), (4, 0)]
    got = flt.points_in_polygon([2.0, 2.0, 8.0, 8.0], [2.0, 8.0, 8.0, 2.0], poly)
    assert got.tolist() == [True, True, True, False]


@pytest.fixture()
def frame(fixture_dir, cache_dir):
    return load(fixture_dir, ["lte_radio", "nr_radio"], cache_dir)


def test_no_filters_is_identity(frame):
    assert len(flt.apply_filters(frame, flt.Filters())) == len(frame)


def test_filters_compose(frame):
    out = flt.apply_filters(
        frame,
        flt.Filters(collections=["Collection_A"], techs=["LTE"], pcis=[123]),
    )
    assert set(out["collection"].unique()) == {"Collection_A"}
    assert set(out["tech"].unique()) == {"LTE"}
    assert set(out["pci"].dropna().unique()) == {123}


def test_metric_filter_drops_nan(frame):
    out = flt.apply_filters(frame, flt.Filters(metric="rssi"))
    assert out["rssi"].notna().all()
    assert set(out["tech"].unique()) == {"LTE"}  # NR export has no RSSI


def test_band_and_freq_filters(frame):
    out = flt.apply_filters(frame, flt.Filters(bands=["n25"], freqs=[394850]))
    assert set(out["band"].unique()) == {"n25"}
    assert set(out["freq"].dropna().unique()) == {394850}


def test_region_filter_and_union(frame):
    inside = flt.bbox_polygon(41.6980, 41.7000, -86.2410, -86.2380)
    one = flt.apply_filters(frame, flt.Filters(regions=[inside]))
    assert 0 < len(one) < len(frame)
    far_away = flt.bbox_polygon(0, 1, 0, 1)
    two = flt.apply_filters(frame, flt.Filters(regions=[inside, far_away]))
    assert len(two) == len(one)
    assert len(flt.apply_filters(frame, flt.Filters(regions=[far_away]))) == 0


def test_region_can_keep_ungeolocated_rows(fixture_dir, cache_dir):
    mixed = load(fixture_dir, ["lte_radio", "lte_pucch"], cache_dir)
    far_away = flt.bbox_polygon(0, 1, 0, 1)
    dropped = flt.apply_filters(mixed, flt.Filters(regions=[far_away]))
    kept = flt.apply_filters(
        mixed, flt.Filters(regions=[far_away], keep_ungeolocated=True)
    )
    assert len(dropped) == 0
    assert len(kept) > 0
    assert set(kept["source"].unique()) == {"lte_pucch"}


def test_time_range_filter(frame):
    times = pd.to_datetime(frame["time"])
    start, end = times.min(), times.min() + pd.Timedelta(seconds=10)
    out = flt.apply_filters(frame, flt.Filters(time_range=(start, end)))
    assert 0 < len(out) < len(frame)
    assert pd.to_datetime(out["time"]).max() <= end


def test_geolocated_drops_null_and_zero_islands(frame):
    padded = pd.concat(
        [frame, frame.head(3).assign(lat=0.0, lon=0.0), frame.head(2).assign(lat=None)]
    )
    assert len(flt.geolocated(padded)) == len(flt.geolocated(frame))


def test_describe_is_human_readable():
    text = flt.Filters(collections=["a"], pcis=[1, 2], regions=[SQUARE]).describe()
    assert "1 collection(s)" in text and "2 PCI(s)" in text and "1 region(s)" in text
    assert flt.Filters().describe() == "no filters"


def test_empty_frame_passes_through():
    empty = pd.DataFrame(columns=["collection"])
    assert flt.apply_filters(empty, flt.Filters(collections=["x"])).empty


def test_require_drops_rows_with_null_dimensions(fixture_dir, cache_dir):
    from qpviz.loader import load as _load

    mixed = _load(fixture_dir, ["lte_radio", "lte_pucch"], cache_dir)
    patched = mixed.copy()
    patched.loc[patched.index[:5], "pci"] = None
    out = flt.apply_filters(patched, flt.Filters(require=["pci"]))
    assert out["pci"].notna().all()
    assert len(out) == len(patched) - 5
    assert len(flt.apply_filters(patched, flt.Filters(require=["nope"]))) == len(patched)


def test_direction_filter(fixture_dir, cache_dir):
    from qpviz.loader import load as _load

    both = _load(fixture_dir, ["lte_pdsch_pc", "lte_pusch_pc"], cache_dir)
    assert set(both["link"].unique()) == {"DL", "UL"}
    downlink = flt.apply_filters(both, flt.Filters(links=["DL"]))
    assert set(downlink["link"].unique()) == {"DL"}
    assert 0 < len(downlink) < len(both)
    assert len(flt.apply_filters(both, flt.Filters(links=["DL", "UL"]))) == len(both)


def test_source_filter(fixture_dir, cache_dir):
    from qpviz.loader import load as _load

    both = _load(fixture_dir, ["lte_pdsch_pc", "lte_pusch_pc"], cache_dir)
    out = flt.apply_filters(both, flt.Filters(sources=["lte_pusch_pc"]))
    assert set(out["source"].unique()) == {"lte_pusch_pc"}
