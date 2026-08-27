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


def test_legend_is_a_colour_bar_with_one_stop_per_bin():
    edges, colors = charts.color_bins(VALUES, scale="sequential")
    html = charts.legend_html(edges, colors, "RSRP (dBm)")
    assert "linear-gradient(to top," in html
    for color in colors:
        assert color in html
    # hard stops, not a smooth blend: every band declares a start and an end
    assert html.count("%,") + 1 >= len(colors)
    assert "RSRP (dBm)" in html


def test_legend_bands_are_proportional_to_their_value_range():
    edges, colors = charts.color_bins(VALUES, scale="diverging", midpoint=-95.0)
    html = charts.legend_html(edges, colors, "RSRP")
    gradient = html.split("linear-gradient(to top,")[1].split(")")[0]
    stops = [s.strip().split() for s in gradient.split(",")]
    starts = [float(p[1].rstrip("%")) for p in stops]
    ends = [float(p[2].rstrip("%")) for p in stops]
    assert starts[0] == pytest.approx(0.0)
    assert ends[-1] == pytest.approx(100.0)
    # contiguous, ascending, and matching the width of each bin
    span = edges[-1] - edges[0]
    for i in range(len(colors)):
        assert ends[i] == pytest.approx(starts[i + 1] if i + 1 < len(colors) else 100.0)
        expected = (edges[i + 1] - edges[i]) / span * 100.0
        # the CSS carries three decimals, so compare at that resolution
        assert ends[i] - starts[i] == pytest.approx(expected, abs=0.01)


def test_legend_ticks_are_round_numbers_inside_the_range():
    edges, colors = charts.color_bins(VALUES, scale="sequential")
    html = charts.legend_html(edges, colors, "RSRP")
    assert 2 <= html.count("translateY(50%)") <= 7
    # VALUES spans -120..-70, so the bar lands on a step of 10
    values, step = charts.nice_ticks(float(edges[0]), float(edges[-1]))
    assert values == [-110.0, -100.0, -90.0, -80.0]
    assert step == 10.0
    for value in values:
        assert charts._tick_text(value, step) in html


@pytest.mark.parametrize(
    "lo,hi",
    [
        (-131.3, -98.6),  # RSRP
        (-20.3, -7.8),  # RSRQ
        (-2.1, 26.0),  # SINR
        (0.5, 27.4),  # MCS
        (0.0, 12.1),  # BLER
        (1296.8, 35867.8),  # throughput in kbps
        (0.0, 0.004),  # a degenerate, very small range
    ],
)
def test_nice_ticks_stay_inside_the_range_and_are_evenly_stepped(lo, hi):
    values, step = charts.nice_ticks(lo, hi)
    assert 2 <= len(values) <= 7
    assert all(lo <= v <= hi for v in values)
    gaps = [b - a for a, b in zip(values, values[1:])]
    assert all(g == pytest.approx(step) for g in gaps)


def test_nice_ticks_survives_a_zero_width_range():
    values, step = charts.nice_ticks(5.0, 5.0)
    assert step > 0
    assert all(v == pytest.approx(5.0) for v in values)


def test_a_two_and_a_half_step_keeps_one_decimal():
    assert charts._tick_text(12.5, 2.5) == "12.5"
    assert charts._tick_text(12.0, 5.0) == "12"
    assert charts._tick_text(0.25, 0.05) == "0.25"


@pytest.mark.parametrize(
    "value,expected",
    [
        (-113.25, "-113"),  # RSRP: three significant digits, no decimal noise
        (12.0, "12.0"),  # SINR: one decimal
        (-3.75, "-3.8"),
        (305.0, "305"),
        (1234.0, "1.2k"),  # throughput in kbps
        (40500.0, "40k"),
    ],
)
def test_tick_text_stays_compact(value, expected):
    assert charts._tick_text(value) == expected


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


# --------------------------------------------------------------------------- #
# Cell identity figures
# --------------------------------------------------------------------------- #


def _runs():
    import pandas as pd
    from qpviz import identity

    times = pd.date_range("2026-07-22 14:00:00", periods=12, freq="1s")
    frame = pd.DataFrame(
        {
            "time": list(times) + list(times),
            "collection": ["A"] * 12 + ["B"] * 12,
            "pci": pd.array([1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4] * 2, dtype="Int32"),
        }
    )
    return identity.dwell_runs(frame, "pci")


def test_timeline_draws_one_trace_per_collection():
    runs = _runs()
    fig = charts.timeline_figure(runs, [1, 2, 3, 4], "PCI")
    assert len(fig.data) == 2
    assert {t.name for t in fig.data} == {"A", "B"}
    assert all(t.orientation == "h" for t in fig.data)
    assert fig.layout.showlegend is True


def test_timeline_row_order_and_reversed_axis():
    fig = charts.timeline_figure(_runs(), [3, 1, 4], "PCI")
    assert list(fig.layout.yaxis.categoryarray) == ["3", "1", "4"]
    # first identity at the top, so the staircase reads downward
    assert fig.layout.yaxis.autorange == "reversed"
    # identities not in the list are not drawn
    assert all("2" not in list(t.y) for t in fig.data)


def test_timeline_elapsed_and_absolute_axes():
    runs = _runs()
    elapsed = charts.timeline_figure(runs, [1, 2, 3, 4], "PCI", x_axis="elapsed")
    assert "Elapsed" in elapsed.layout.xaxis.title.text
    assert max(elapsed.data[0].x) == pytest.approx(4 / 60)  # 4 s in minutes
    absolute = charts.timeline_figure(runs, [1, 2, 3, 4], "PCI", x_axis="absolute")
    assert absolute.layout.xaxis.title.text == "Time"


def test_timeline_height_grows_with_the_number_of_rows():
    runs = _runs()
    short = charts.timeline_figure(runs, [1, 2], "PCI").layout.height
    tall = charts.timeline_figure(runs, list(range(1, 60)), "PCI").layout.height
    assert tall > short


def test_timeline_handles_no_runs():
    import pandas as pd

    fig = charts.timeline_figure(pd.DataFrame(columns=["collection", "value"]), [], "PCI")
    assert len(fig.data) == 0


def test_treemap_builds_a_node_per_path_segment():
    import pandas as pd

    table = pd.DataFrame(
        {
            "enb": [10, 10, 11],
            "sector": [1, 2, 3],
            "pci": [100, 200, 300],
            "dwell_s": [30.0, 20.0, 10.0],
            "samples": [30, 20, 10],
        }
    )
    fig = charts.treemap_figure(table, ["enb", "sector", "pci"], ["eNB", "Sector", "PCI"])
    node = fig.data[0]
    assert node.ids[0] == "All" and node.parents[0] == ""
    # 1 root + 2 eNBs + 3 sectors + 3 PCIs
    assert len(node.ids) == 9
    assert sum(node.values) == pytest.approx(60.0)
    assert "eNB 10" in node.labels and "PCI 300" in node.labels


def test_treemap_shades_by_depth_never_by_cycled_hue():
    import pandas as pd

    table = pd.DataFrame(
        {"enb": [10, 11], "pci": [1, 2], "dwell_s": [1.0, 2.0], "samples": [1, 2]}
    )
    fig = charts.treemap_figure(table, ["enb", "pci"], ["eNB", "PCI"])
    used = set(fig.data[0].marker.colors)
    assert used <= set(charts._DEPTH_COLORS)
    assert not used & set(theme.CATEGORICAL["light"])


def test_treemap_handles_empty_input():
    import pandas as pd

    assert len(charts.treemap_figure(pd.DataFrame(), ["pci"], ["PCI"]).data) == 0


# --------------------------------------------------------------------------- #
# Categorical map colouring
# --------------------------------------------------------------------------- #


def test_categorical_bins_takes_the_eight_most_common_values():
    values = [1] * 10 + [2] * 9 + [3] * 8 + [4] * 7 + [5] * 6 + [6] * 5 + [7] * 4
    values += [8] * 3 + [9] * 2 + [10] * 1
    order, colors, other = charts.categorical_bins(values)
    assert order == [1, 2, 3, 4, 5, 6, 7, 8]
    assert other == 3  # the 9th and 10th values fold into "Other"
    # hues are never cycled, so no two identities share a colour
    assert len(set(colors.values())) == len(colors)
    assert list(colors.values()) == theme.CATEGORICAL["light"][:8]


def test_categorical_bins_with_few_values_has_no_other_bucket():
    order, colors, other = charts.categorical_bins([1, 1, 2])
    assert order == [1, 2] and other == 0
    assert charts.categorical_bins([]) == ([], {}, 0)


def test_categorical_legend_lists_every_slot_and_the_other_bucket():
    order, colors, other = charts.categorical_bins([1] * 3 + [2] * 2)
    html = charts.categorical_legend_html(order, colors, "PCI", other_count=17)
    assert html.count("border-radius:3px") == 3  # two slots + Other
    assert "Other (17 pts)" in html
    assert charts.OTHER_COLOR in html
