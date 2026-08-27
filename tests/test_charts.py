import numpy as np
import pytest

from qpviz import charts, theme
from qpviz.stats import Curve, cdf_curves
from qpviz.loader import load


VALUES = np.linspace(-120, -70, 500)


@pytest.mark.parametrize("scale", ["sequential", "diverging"])
def test_color_bins_edges_are_sorted_and_one_longer(scale):
    edges, colors = charts.color_bins(VALUES, scale=scale)
    assert len(edges) == len(colors) + 1
    assert all(edges[i] <= edges[i + 1] for i in range(len(edges) - 1))


def test_color_bins_handles_constant_and_empty_input():
    edges, colors = charts.color_bins([5.0] * 10)
    assert len(edges) == len(colors) + 1
    edges, colors = charts.color_bins([np.nan, np.nan])
    assert len(colors) == 1


def test_color_for_clamps_and_handles_nan():
    edges, colors = charts.color_bins(VALUES, scale="sequential")
    assert charts.color_for(-1000, edges, colors) == colors[0]
    assert charts.color_for(1000, edges, colors) == colors[-1]
    assert charts.color_for(np.nan, edges, colors) == "#898781"


def test_diverging_midpoint_splits_red_and_blue():
    edges, colors = charts.color_bins(VALUES, scale="diverging", midpoint=-95.0)
    assert charts.color_for(-115, edges, colors) in theme.DIVERGING_RED_BLUE[:4]
    assert charts.color_for(-75, edges, colors) in theme.DIVERGING_RED_BLUE[5:]


def test_series_style_uses_fixed_slots_then_changes_dash():
    first, dash0 = theme.series_style(0)
    assert first == theme.CATEGORICAL["light"][0] and dash0 == "solid"
    ninth, dash8 = theme.series_style(8)
    assert ninth == first and dash8 != dash0
    assert theme.series_style(0, "dark")[0] == theme.CATEGORICAL["dark"][0]


def test_downsample_thins_but_never_grows():
    import pandas as pd

    frame = pd.DataFrame({"a": range(1000)})
    assert len(charts.downsample(frame, 100)) <= 100
    assert len(charts.downsample(frame, 5000)) == 1000


def test_legend_html_lists_every_bin():
    edges, colors = charts.color_bins(VALUES, scale="sequential")
    html = charts.legend_html(edges, colors, "RSRP (dBm)")
    assert html.count("border-radius:2px") == len(colors)
    assert "RSRP (dBm)" in html


def _curves(n):
    return [
        Curve(key=(i,), name=f"PCI {i}", x=np.linspace(0, 1, 10), y=np.linspace(0.1, 1, 10), n=10)
        for i in range(n)
    ]


def test_cdf_figure_one_trace_per_curve():
    fig = charts.cdf_figure(_curves(3), "RSRP (dBm)")
    assert len(fig.data) == 3
    assert fig.layout.showlegend is True
    assert {t.line.width for t in fig.data} == {2}


def test_cdf_figure_hides_legend_for_a_single_series():
    fig = charts.cdf_figure(_curves(1), "RSRP (dBm)")
    assert fig.layout.showlegend is False


def test_cdf_figure_direct_labels_only_for_small_series_counts():
    assert len(charts.cdf_figure(_curves(3), "x").layout.annotations) == 3
    assert len(charts.cdf_figure(_curves(9), "x").layout.annotations) == 0


def test_ccdf_inverts_the_y_axis_values():
    normal = charts.cdf_figure(_curves(1), "x")
    inverted = charts.cdf_figure(_curves(1), "x", complementary=True)
    assert np.allclose(inverted.data[0].y, 1 - np.asarray(normal.data[0].y))
    assert "P(X > x)" in inverted.layout.yaxis.title.text


def test_figure_uses_theme_surfaces():
    light = charts.cdf_figure(_curves(2), "x", mode="light")
    dark = charts.cdf_figure(_curves(2), "x", mode="dark")
    assert light.layout.plot_bgcolor == theme.CHROME["light"]["surface"]
    assert dark.layout.plot_bgcolor == theme.CHROME["dark"]["surface"]


def test_real_curves_render(fixture_dir, cache_dir):
    frame = load(fixture_dir, ["lte_radio"], cache_dir)
    fig = charts.cdf_figure(cdf_curves(frame, "rsrp", ["pci"]), "RSRP (dBm)")
    assert len(fig.data) == 3
