import pandas as pd
import pytest

from qpviz import mapview
from qpviz.loader import load


def _feature(coords, gtype="Polygon"):
    return {"type": "Feature", "geometry": {"type": gtype, "coordinates": coords}}


def test_regions_from_drawings_converts_lon_lat_to_lat_lon():
    ring = [[-86.24, 41.69], [-86.23, 41.69], [-86.23, 41.70], [-86.24, 41.70], [-86.24, 41.69]]
    got = mapview.regions_from_drawings([_feature([ring])])
    assert got == [[(41.69, -86.24), (41.69, -86.23), (41.70, -86.23), (41.70, -86.24), (41.69, -86.24)]]


def test_regions_from_drawings_handles_multipolygon_and_junk():
    ring = [[0, 0], [1, 0], [1, 1], [0, 0]]
    got = mapview.regions_from_drawings(
        [
            _feature([[ring]], gtype="MultiPolygon"),
            _feature([[[0, 0], [1, 1]]]),  # too few vertices
            _feature(None, gtype="LineString"),
            None,
        ]
    )
    assert len(got) == 1


def test_regions_from_drawings_empty():
    assert mapview.regions_from_drawings(None) == []
    assert mapview.regions_from_drawings([]) == []


@pytest.fixture()
def frame(fixture_dir, cache_dir):
    return load(fixture_dir, ["lte_radio"], cache_dir)


def test_bounds(frame):
    box = mapview.bounds(frame)
    assert box is not None
    (lat_min, lon_min), (lat_max, lon_max) = box
    assert lat_min < lat_max and lon_min < lon_max
    assert mapview.bounds(frame.assign(lat=None, lon=None)) is None


def test_build_map_renders_markers_and_legend(frame):
    fmap, edges, colors = mapview.build_map(
        frame, "rsrp", "RSRP (dBm)", max_points=50
    )
    html = fmap.get_root().render()
    assert "circleMarker" in html
    assert "RSRP (dBm)" in html
    assert "L.Control.Draw" in html or "drawnItems" in html
    assert len(edges) == len(colors) + 1


def test_build_map_without_geo_returns_bare_map(fixture_dir, cache_dir):
    pucch = load(fixture_dir, ["lte_pucch"], cache_dir)
    fmap, edges, colors = mapview.build_map(pucch, "cqi", "CQI")
    assert edges == [] and colors == []
    assert "circleMarker" not in fmap.get_root().render()


def test_build_map_draws_the_selected_region(frame):
    region = [(41.69, -86.24), (41.69, -86.23), (41.71, -86.23), (41.71, -86.24)]
    fmap, _, _ = mapview.build_map(frame, "rsrp", "RSRP", max_points=20, regions=[region])
    assert "polygon" in fmap.get_root().render().lower()
