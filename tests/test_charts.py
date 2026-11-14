import numpy as np
import pytest

from qpviz import charts, theme
from qpviz.stats import Curve, cdf_curves
from qpviz.loader import load


VALUES = np.linspace(-120, -70, 500)


@pytest.mark.parametrize("value_range", [None, (-140.0, -80.0)])
def test_color_bins_edges_are_sorted_and_one_longer(value_range):
    edges, colors = charts.color_bins(VALUES, value_range=value_range)
    assert len(edges) == len(colors) + 1
    assert all(edges[i] <= edges[i + 1] for i in range(len(edges) - 1))


def test_color_bins_handles_constant_and_empty_input():
    edges, colors = charts.color_bins([5.0] * 10)
    assert len(edges) == len(colors) + 1
    edges, colors = charts.color_bins([np.nan, np.nan])
    assert len(colors) == 1


def test_color_for_clamps_and_handles_nan():
    edges, colors = charts.color_bins(VALUES)
    assert charts.color_for(-1000, edges, colors) == colors[0]
    assert charts.color_for(1000, edges, colors) == colors[-1]
    assert charts.color_for(np.nan, edges, colors) == "#898781"


def test_a_fixed_range_gives_the_same_colours_whatever_the_data():
    """With a fixed range a colour means the same value on every map; the data
    only decides the colours when no range is given."""
    fixed = (-140.0, -80.0)
    weak, strong = np.linspace(-135, -120, 50), np.linspace(-95, -85, 50)
    edges_weak, colors = charts.color_bins(weak, value_range=fixed)
    edges_strong, _ = charts.color_bins(strong, value_range=fixed)
    assert edges_weak == edges_strong
    assert edges_weak[0] == -140.0 and edges_weak[-1] == -80.0
    assert charts.color_for(-110.0, edges_weak, colors) == charts.color_for(
        -110.0, edges_strong, colors
    )
    # values outside the range take the end colours
    assert charts.color_for(-150.0, edges_weak, colors) == colors[0]
    assert charts.color_for(-60.0, edges_weak, colors) == colors[-1]
    # without a range the scale follows the data
    assert charts.color_bins(weak)[0] != charts.color_bins(strong)[0]


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
    edges, colors = charts.color_bins(VALUES)
    html = charts.legend_html(edges, colors, "RSRP (dBm)")
    assert "linear-gradient(to top," in html
    for color in colors:
        assert color in html
    # hard stops, not a smooth blend: every band declares a start and an end
    assert html.count("%,") + 1 >= len(colors)
    assert "RSRP (dBm)" in html


def test_legend_bands_are_proportional_to_their_value_range():
    edges, colors = charts.color_bins(VALUES, value_range=(-140.0, -80.0))
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
    edges, colors = charts.color_bins(VALUES)
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


def _hierarchy():
    import pandas as pd

    return pd.DataFrame(
        {
            "enb": [10, 10, 11],
            "sector": [1, 2, 3],
            "pci": [100, 200, 300],
            "dwell_s": [30.0, 20.0, 10.0],
            "samples": [30, 20, 10],
        }
    )


def test_tree_builds_a_node_per_path_segment():
    nodes, edges = charts.build_tree(_hierarchy(), ["enb", "sector", "pci"])
    # 1 root + 2 eNBs + 3 sectors + 3 PCIs
    assert len(nodes) == 9
    assert len(edges) == 8
    # a node carries its value only; the level is named once, in the header
    labels = {n["label"] for n in nodes}
    assert labels == {"All", "10", "11", "1", "2", "3", "100", "200", "300"}
    assert {n["level"] for n in nodes} == {None, "enb", "sector", "pci"}


def test_tree_can_drop_the_synthetic_root():
    nodes, edges = charts.build_tree(_hierarchy(), ["enb", "sector", "pci"], include_root=False)
    assert len(nodes) == 8
    assert all(n["depth"] > 0 for n in nodes)
    # the two edges leaving the root go with it; row positions are unchanged
    assert len(edges) == 6
    assert sorted(n["y"] for n in nodes if not n["children"]) == [0.0, 1.0, 2.0]


def test_tree_rolls_dwell_up_to_every_ancestor():
    nodes, _ = charts.build_tree(_hierarchy(), ["enb", "sector", "pci"])
    by_label = {n["label"]: n for n in nodes}
    assert by_label["All"]["dwell"] == pytest.approx(60.0)
    assert by_label["10"]["dwell"] == pytest.approx(50.0)
    assert by_label["11"]["dwell"] == pytest.approx(10.0)
    assert by_label["100"]["dwell"] == pytest.approx(30.0)


def test_tree_gives_every_leaf_a_row_and_centres_the_parents():
    nodes, _ = charts.build_tree(_hierarchy(), ["enb", "sector", "pci"])
    leaves = sorted(n["y"] for n in nodes if not n["children"])
    assert leaves == [0.0, 1.0, 2.0]
    by_label = {n["label"]: n for n in nodes}
    # eNB 10 owns the first two leaves, so it sits between them
    assert by_label["10"]["y"] == pytest.approx(0.5)
    assert by_label["All"]["y"] == pytest.approx(1.25)
    # depth becomes the x coordinate
    assert by_label["All"]["x"] == 0.0 and by_label["300"]["x"] == 3.0


def test_tree_orders_branches_by_identity_value_not_by_time_held():
    import pandas as pd

    table = pd.DataFrame(
        {
            "enb": [11, 10, 10, 10],
            "pci": [2, 30, 4, 100],
            "dwell_s": [50.0, 5.0, 1.0, 9.0],  # the longest-held is listed first
            "samples": [50, 5, 1, 9],
        }
    )
    nodes, _ = charts.build_tree(table, ["enb", "pci"])
    root = next(n for n in nodes if n["depth"] == 0)
    assert [c["label"] for c in root["children"]] == ["10", "11"]
    # numeric order, not text order ("100" would sort before "30" as text)
    assert [c["label"] for c in root["children"][0]["children"]] == ["4", "30", "100"]


def test_tree_edges_are_elbows_between_parent_and_child():
    _, edges = charts.build_tree(_hierarchy(), ["enb", "sector", "pci"])
    for path in edges:
        assert len(path) == 4
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = path
        assert y0 == y1 and y2 == y3  # horizontal in, horizontal out
        assert x1 == x2  # vertical trunk between them
        assert x3 > x0


_HEADERS = ["eNB ID", "Sector ID", "PCI"]


def test_tree_figure_draws_edges_nodes_and_the_time_held_column():
    fig = charts.tree_figure(_hierarchy(), ["enb", "sector", "pci"], _HEADERS)
    assert len(fig.data) == 3
    edges, nodes, held = fig.data
    assert edges.mode == "lines" and None in list(edges.x)
    assert nodes.mode == "markers+text"
    assert len(nodes.x) == 8  # the root is not drawn
    # one time per leaf, in a column right of the PCI column
    assert held.mode == "text"
    assert sorted(held.text) == ["10 s", "20 s", "30 s"]
    assert set(held.x) == {3.9}
    # rows run top to bottom
    top, bottom = fig.layout.yaxis.range
    assert top > bottom


def test_tree_names_each_level_once_in_a_header_row():
    fig = charts.tree_figure(_hierarchy(), ["enb", "sector", "pci"], _HEADERS)
    # the nodes carry bare values ...
    assert set(fig.data[1].text) == {"10", "11", "1", "2", "3", "100", "200", "300"}
    # ... and the level names sit once on top, one per column
    axis = fig.layout.xaxis
    assert axis.side == "top"
    assert list(axis.tickvals) == [1.0, 2.0, 3.0, 3.9]
    assert list(axis.ticktext) == [
        "<b>eNB ID</b>", "<b>Sector ID</b>", "<b>PCI</b>", "<b>Time held</b>",
    ]
    # the hover still says which level a value belongs to
    assert ["eNB ID", "50 s"] in [list(c) for c in fig.data[1].customdata]


def test_tree_markers_are_one_size():
    fig = charts.tree_figure(_hierarchy(), ["enb", "sector", "pci"], _HEADERS)
    assert fig.data[1].marker.size == 10


def test_tree_shades_by_depth_never_by_cycled_hue():
    fig = charts.tree_figure(_hierarchy(), ["enb", "pci"], ["eNB ID", "PCI"])
    used = set(fig.data[1].marker.color)
    assert used <= set(charts._DEPTH_COLORS)
    assert not used & set(theme.CATEGORICAL["light"])


def test_tree_caps_the_number_of_leaves():
    import pandas as pd

    table = pd.DataFrame(
        {
            "enb": list(range(20)),
            "pci": list(range(20)),
            "dwell_s": [float(i) for i in range(20)],
            "samples": [1] * 20,
        }
    )
    fig = charts.tree_figure(table, ["enb", "pci"], ["eNB ID", "PCI"], max_leaves=5)
    assert len(fig.data[1].x) == 5 + 5


def test_tree_height_follows_the_leaf_count():
    import pandas as pd

    def _table(n):
        return pd.DataFrame(
            {"enb": [1] * n, "pci": list(range(n)), "dwell_s": [1.0] * n,
              "samples": [1] * n}
        )

    short = charts.tree_figure(_table(2), ["enb", "pci"], ["eNB ID", "PCI"])
    tall = charts.tree_figure(_table(40), ["enb", "pci"], ["eNB ID", "PCI"])
    assert short.layout.height < tall.layout.height


def test_tree_handles_empty_input():
    import pandas as pd

    assert len(charts.tree_figure(pd.DataFrame(), ["pci"], ["PCI"]).data) == 0


@pytest.mark.parametrize(
    "seconds,expected", [(12.0, "12 s"), (89.0, "89 s"), (90.0, "1.5 min"), (600.0, "10.0 min")]
)
def test_held_text(seconds, expected):
    assert charts._held_text(seconds) == expected


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


# --------------------------------------------------------------------------- #
# Fan-out figure
# --------------------------------------------------------------------------- #


_REL = "Sectors seen per eNB"


def _dist(groups=("All",)):
    import pandas as pd

    rows = []
    for g in groups:
        for children, parents in [(1, 10), (2, 6), (3, 4)]:
            rows.append(
                {"relationship": _REL, "group": g,
                 "children": children, "parents": parents, "share": parents / 20}
            )
    return pd.DataFrame(rows)


def test_fan_out_bars_carry_plain_titles():
    fig = charts.fan_out_figure(_dist(), _REL, y_title="Number of eNBs")
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"
    assert list(fig.data[0].x) == [1, 2, 3]
    # a single series gets its values written on the bars
    assert list(fig.data[0].text) == [10, 6, 4]
    assert fig.layout.xaxis.title.text == "Sectors seen per eNB"
    assert fig.layout.yaxis.title.text == "Number of eNBs"
    assert "Number of eNBs: %{y}" in fig.data[0].hovertemplate
    assert "Parents" not in fig.data[0].hovertemplate


def test_fan_out_puts_the_networks_side_by_side():
    fig = charts.fan_out_figure(_dist(("Terrestrial", "D2C")), _REL)
    assert len(fig.data) == 2
    assert fig.layout.barmode == "group"
    assert fig.layout.showlegend is True
    assert {t.marker.color for t in fig.data} == set(theme.CATEGORICAL["light"][:2])
    # no value labels once there is more than one series
    assert all(t.text is None for t in fig.data)


def test_fan_out_groups_keep_their_order_and_colour():
    order = ["D2C", "Terrestrial"]
    both = charts.fan_out_figure(_dist(("Terrestrial", "D2C")), _REL, groups=order)
    assert [t.name for t in both.data] == order
    # with D2C filtered out, terrestrial keeps the second colour
    alone = charts.fan_out_figure(_dist(("Terrestrial",)), _REL, groups=order)
    assert [t.name for t in alone.data] == ["Terrestrial"]
    assert alone.data[0].marker.color == both.data[1].marker.color


def test_fan_out_ignores_other_relationships_and_empty_input():
    import pandas as pd

    assert len(charts.fan_out_figure(_dist(), "PCIs seen per sector").data) == 0
    assert len(charts.fan_out_figure(pd.DataFrame(), "anything").data) == 0
