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
    no_gps = load(fixture_dir, ["lte_radio"], cache_dir).copy()
    no_gps[["lat", "lon"]] = float("nan")
    fmap, edges, colors = mapview.build_map(no_gps, "rsrp", "RSRP")
    assert edges == [] and colors == []
    assert "circleMarker" not in fmap.get_root().render()


def test_build_map_draws_the_selected_region(frame):
    region = [(41.69, -86.24), (41.69, -86.23), (41.71, -86.23), (41.71, -86.24)]
    fmap, _, _ = mapview.build_map(frame, "rsrp", "RSRP", max_points=20, regions=[region])
    assert "polygon" in fmap.get_root().render().lower()


def test_basemap_is_carto_light_with_a_key_and_openstreetmap_without(monkeypatch):
    monkeypatch.delenv(mapview.CARTO_KEY_NAME, raising=False)
    assert mapview.carto_api_key() is None
    assert mapview.carto_api_key({}) is None
    assert mapview.default_tiles(None) == "OpenStreetMap"
    assert mapview.default_tiles("k") in mapview.CARTO_TILES


def test_build_map_uses_the_fixed_colour_range(frame):
    _, edges, colors = mapview.build_map(
        frame, "rsrp", "RSRP (dBm)", max_points=50, color_range=(-140.0, -80.0)
    )
    assert edges[0] == -140.0 and edges[-1] == -80.0
    assert len(edges) == len(colors) + 1
    # a subset of the data is coloured on the very same scale
    _, subset_edges, _ = mapview.build_map(
        frame.head(20), "rsrp", "RSRP (dBm)", color_range=(-140.0, -80.0)
    )
    assert subset_edges == edges


def test_build_map_colours_the_ten_busiest_cells_and_folds_the_rest(frame):
    many = frame.copy()
    many["enb"] = [20000 + (i % 25) for i in range(len(many))]
    html = mapview.build_map(
        many, "rsrp", "RSRP", max_points=200, color_by="enb", color_by_label="eNB / gNB-Id",
    )[0].get_root().render()
    assert "Other" in html


def test_carto_key_read_from_env_then_secrets(monkeypatch):
    monkeypatch.delenv(mapview.CARTO_KEY_NAME, raising=False)
    assert mapview.carto_api_key({mapview.CARTO_KEY_NAME: " abc "}) == "abc"
    monkeypatch.setenv(mapview.CARTO_KEY_NAME, "env-key")
    assert mapview.carto_api_key({mapview.CARTO_KEY_NAME: "abc"}) == "env-key"


def test_carto_tiles_carry_key_or_fall_back_to_osm(frame):
    with_key = mapview.build_map(
        frame, "rsrp", "RSRP", max_points=5,
        tiles="Light (CartoDB positron)", carto_key="SECRET123",
    )[0].get_root().render()
    assert "basemaps.cartocdn.com" in with_key and "key=SECRET123" in with_key
    no_key = mapview.build_map(
        frame, "rsrp", "RSRP", max_points=5, tiles="Dark (CartoDB dark_matter)",
    )[0].get_root().render()
    assert "cartocdn" not in no_key and "openstreetmap" in no_key.lower()


def test_new_regions_only_returns_shapes_not_already_in_the_region():
    a = [(41.69, -86.24), (41.69, -86.23), (41.70, -86.23)]
    b = [(41.71, -86.24), (41.71, -86.23), (41.72, -86.23)]
    assert mapview.new_regions([a], []) == [a]
    # the map re-reports a shape it already gave: nothing new, so no rerun
    assert mapview.new_regions([a], [a]) == []
    assert mapview.new_regions([[list(p) for p in a]], [a]) == []
    # a report with no shapes never removes the region
    assert mapview.new_regions([], [a]) == []
    assert mapview.new_regions([a, b, b], [a]) == [b]


def test_the_draw_toolbar_only_adds_shapes(frame):
    html = mapview.build_map(frame, "rsrp", "RSRP", max_points=5)[0].get_root().render()
    compact = html.replace(" ", "").replace("\n", "")
    assert '"edit":false' in compact and '"remove":false' in compact
