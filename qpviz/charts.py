"""Figure builders: CDF curves (plotly) and the measurement map (folium)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .stats import Curve
from .theme import (
    CATEGORICAL,
    CHROME,
    FONT_FAMILY,
    SEQUENTIAL_BLUE,
    series_style,
)

# --------------------------------------------------------------------------- #
# CDF
# --------------------------------------------------------------------------- #


def cdf_figure(
    curves: Sequence[Curve],
    axis_label: str,
    mode: str = "light",
    show_direct_labels: bool = True,
    height: int = 560,
) -> go.Figure:
    """One ECDF line per curve, all on the same canvas."""
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()

    y_title = "P(X ≤ x)"

    for i, curve in enumerate(curves):
        color, dash = series_style(i, mode)
        y = curve.y
        fig.add_trace(
            go.Scatter(
                x=curve.x,
                y=y,
                mode="lines",
                name=f"{curve.name}  (n={curve.n:,})",
                line=dict(color=color, width=2, dash=dash),
                hovertemplate=(
                    f"<b>{curve.name}</b><br>{axis_label}: %{{x:.2f}}"
                    f"<br>{y_title}: %{{y:.3f}}<extra></extra>"
                ),
            )
        )

    # Direct labels for a small number of series (identity never colour-alone).
    if show_direct_labels and 2 <= len(curves) <= 4:
        for i, curve in enumerate(curves):
            color, _ = series_style(i, mode)
            y = curve.y
            fig.add_annotation(
                x=float(curve.x[-1]),
                y=float(y[-1]),
                text=curve.name,
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(color=chrome["text_secondary"], size=11, family=FONT_FAMILY),
            )

    fig.update_layout(
        height=height,
        template=None,
        paper_bgcolor=chrome["surface"],
        plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=64, r=24, t=16, b=56),
        hovermode="x unified" if 0 < len(curves) <= 8 else "closest",
        hoverlabel=dict(font_family=FONT_FAMILY),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            font=dict(size=11, color=chrome["text_secondary"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=len(curves) >= 2,
    )
    fig.update_xaxes(
        title=dict(text=axis_label, font=dict(color=chrome["text_secondary"], size=12)),
        gridcolor=chrome["grid"],
        zeroline=False,
        linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11),
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=chrome["text_secondary"], size=12)),
        range=[0, 1.02],
        gridcolor=chrome["grid"],
        zeroline=False,
        linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11),
        tickformat=".0%",
    )
    return fig


# --------------------------------------------------------------------------- #
# Map colouring
# --------------------------------------------------------------------------- #


def _clean(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def color_bins(
    values: Sequence[float],
    value_range: Optional[Tuple[float, float]] = None,
    n_bins: int = 9,
) -> Tuple[List[float], List[str]]:
    """Return ``(edges, colors)`` where ``len(edges) == len(colors) + 1``.

    A single blue hue, light -> dark, in equal steps. With ``value_range`` the
    steps span that fixed range, so a colour means the same value on every map;
    values outside it take the end colours. Without it they span the 2nd-98th
    percentile of ``values`` (so a few outliers do not wash out the scale).
    """
    colors = SEQUENTIAL_BLUE[:n_bins]
    if value_range is not None:
        lo, hi = float(value_range[0]), float(value_range[1])
        return list(np.linspace(lo, hi, len(colors) + 1)), colors

    arr = _clean(values)
    if arr.size == 0:
        return [0.0, 1.0], [SEQUENTIAL_BLUE[-1]]

    lo, hi = float(np.percentile(arr, 2)), float(np.percentile(arr, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1.0
    return list(np.linspace(lo, hi, len(colors) + 1)), colors


def color_for(value: float, edges: Sequence[float], colors: Sequence[str]) -> str:
    if value is None or not np.isfinite(value):
        return "#898781"
    idx = int(np.searchsorted(np.asarray(edges[1:-1], dtype="float64"), value, side="right"))
    return colors[min(max(idx, 0), len(colors) - 1)]


def _nice_step(span: float, max_ticks: int = 7) -> float:
    """The finest round step (1, 2, 2.5 or 5 x a power of ten) that still fits
    within ``max_ticks`` labels across ``span``."""
    if span <= 0 or not math.isfinite(span):
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(span))
    for scale in (magnitude / 100, magnitude / 10, magnitude, magnitude * 10):
        for multiple in (1.0, 2.0, 2.5, 5.0):
            step = multiple * scale
            if step > 0 and math.floor(span / step) + 1 <= max_ticks:
                return step
    return span


def nice_ticks(lo: float, hi: float, max_ticks: int = 7) -> Tuple[List[float], float]:
    """Round tick values inside ``[lo, hi]``, plus the step used."""
    step = _nice_step(hi - lo, max_ticks)
    first = math.ceil(lo / step) * step
    ticks: List[float] = []
    value = first
    while value <= hi + step * 1e-9:
        # snap away -0.0 and floating-point dust like 4.999999999
        ticks.append(round(value / step) * step + 0.0)
        value += step
    return ticks, step


def _tick_text(value: float, step: Optional[float] = None) -> str:
    """Compact tick label; values in the thousands (e.g. ARFCNs, cell ids,
    sample counts) get a ``k`` suffix."""
    magnitude = abs(value)
    if magnitude >= 10_000:
        return f"{value / 1000:.0f}k"
    if magnitude >= 1_000:
        text = f"{value / 1000:.1f}k"
        return text.replace(".0k", "k")
    if step is not None:
        if step < 1:
            decimals = min(3, int(math.ceil(-math.log10(step))))
        else:
            # a 2.5-style step needs one decimal; 1/2/5 steps do not
            decimals = 1 if abs(step - round(step)) > 1e-9 else 0
        return f"{value:.{decimals}f}"
    if magnitude >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def legend_html(
    edges: Sequence[float],
    colors: Sequence[str],
    title: str,
    mode: str = "light",
    height: int = 168,
) -> str:
    """A vertical colour bar with round-number ticks.

    The bar uses hard gradient stops rather than a smooth blend, and each band
    is as tall as the share of the value range it covers - so the bar is a
    faithful picture of the bins the markers are actually coloured by.
    """
    chrome = CHROME.get(mode, CHROME["light"])
    lo, hi = float(edges[0]), float(edges[-1])
    span = (hi - lo) or 1.0

    stops = []
    for i, color in enumerate(colors):
        start = (float(edges[i]) - lo) / span * 100.0
        end = (float(edges[i + 1]) - lo) / span * 100.0
        stops.append(f"{color} {start:.3f}% {end:.3f}%")
    gradient = "linear-gradient(to top, " + ", ".join(stops) + ")"

    values, step = nice_ticks(lo, hi)
    ticks = []
    for value in values:
        position = (value - lo) / span * 100.0
        ticks.append(
            f'<div style="position:absolute;bottom:{position:.3f}%;'
            f'transform:translateY(50%);white-space:nowrap;line-height:1">'
            f'<span style="display:inline-block;width:4px;height:1px;'
            f'background:{chrome["axis"]};vertical-align:middle;margin-right:5px">'
            f'</span>{_tick_text(value, step)}</div>'
        )

    return (
        f'<div style="position:fixed;bottom:24px;left:12px;z-index:9999;'
        f'background:{chrome["surface"]};color:{chrome["text_primary"]};'
        f'border:1px solid rgba(11,11,11,.12);border-radius:8px;padding:10px 12px;'
        f'font-family:{FONT_FAMILY};font-size:11px;line-height:1.4;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.12);'
        f'font-variant-numeric:tabular-nums">'
        f'<div style="font-weight:600;font-size:12px;margin-bottom:8px">{title}</div>'
        f'<div style="display:flex;gap:2px;height:{height}px">'
        f'<div style="width:12px;height:100%;border-radius:3px;background:{gradient};'
        f'box-shadow:0 0 0 1px rgba(11,11,11,.12)"></div>'
        f'<div style="position:relative;height:100%;color:{chrome["text_secondary"]}">'
        + "".join(ticks)
        + "</div></div></div>"
    )


#: everything past the coloured slots folds into one muted "Other"
OTHER_COLOR = "#898781"

#: how many of the validated categorical hues exist before the palette has to be
#: generated; also the point past which the legend gains a count and scrolls
_BASE_SLOTS = 8


def generate_palette(n: int, mode: str = "light") -> List[str]:
    """``n`` visually distinct colours, so a categorical map never runs out.

    The first eight are the validated categorical hues; beyond that, colours are
    generated on a golden-angle hue rotation with rotating lightness/saturation
    tiers, which keeps neighbours in the sequence far apart. Hues are never
    cycled, so two colours always mean two different identities.
    """
    if n <= 0:
        return []
    base = CATEGORICAL.get(mode, CATEGORICAL["light"])
    if n <= len(base):
        return list(base[:n])

    import colorsys

    dark = mode == "dark"
    lightness = [0.60, 0.74, 0.46] if dark else [0.50, 0.66, 0.38]
    saturation = [0.72, 0.52, 0.88]
    golden = 0.6180339887498949
    colours = list(base)  # keep the eight validated hues up front
    for i in range(len(base), n):
        hue = (i * golden) % 1.0
        light = lightness[i % len(lightness)]
        sat = saturation[(i // len(lightness)) % len(saturation)]
        r, g, b = colorsys.hls_to_rgb(hue, light, sat)
        colours.append(f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}")
    return colours[:n]


def categorical_bins(values, max_slots: int = 8, mode: str = "light"):
    """Assign the ``max_slots`` most common values a distinct colour each.

    Hues are never cycled: a value past ``max_slots`` is not given a repeated
    colour, it folds into "Other" - so two differently coloured points always
    mean two different identities. Raise ``max_slots`` up to the number of
    distinct values to colour every cell instead of folding the tail into grey.
    """
    series = pd.Series(values).dropna()
    if series.empty:
        return [], {}, 0
    counts = series.value_counts()
    slots = max(1, int(max_slots))
    top = list(counts.index[:slots])
    palette = generate_palette(len(top), mode)
    colors = {value: palette[i] for i, value in enumerate(top)}
    others = int(counts.iloc[slots:].sum()) if len(counts) > slots else 0
    return top, colors, others


def categorical_counts(values) -> Dict:
    """How many points carry each identity value - used for the legend counts."""
    series = pd.Series(values).dropna()
    if series.empty:
        return {}
    return {k: int(v) for k, v in series.value_counts().to_dict().items()}


def categorical_legend_html(
    order: Sequence,
    colors: dict,
    title: str,
    mode: str = "light",
    other_count: int = 0,
    value_prefix: str = "",
    counts: Optional[dict] = None,
) -> str:
    """Swatch legend for a categorically coloured map.

    ``counts`` (value -> points) writes a per-cell tally beside each swatch. Once
    there are many entries - colouring every cell rather than the top eight - the
    list scrolls inside a fixed height instead of running off the map.
    """
    chrome = CHROME.get(mode, CHROME["light"])

    def _row(swatch: str, label: str) -> str:
        return (
            f'<div style="display:flex;align-items:center;gap:6px;line-height:1.6">'
            f'<span style="flex:0 0 auto;width:12px;height:12px;border-radius:3px;'
            f'background:{swatch};box-shadow:0 0 0 1px rgba(11,11,11,.12)">'
            f"</span><span>{label}</span></div>"
        )

    rows = []
    for value in order:
        label = f"{value_prefix}{value}"
        if counts and value in counts:
            label += (
                f'<span style="color:{chrome["muted"]}"> · '
                f"{counts[value]:,}</span>"
            )
        rows.append(_row(colors[value], label))
    if other_count:
        rows.append(_row(OTHER_COLOR, f"Other ({other_count:,} pts)"))

    # a header count so "colour all" is legible at a glance
    subtitle = ""
    n_shown = len(order)
    if n_shown > _BASE_SLOTS:
        subtitle = (
            f'<div style="color:{chrome["text_secondary"]};font-size:10px;'
            f'margin:-2px 0 6px">{n_shown} coloured</div>'
        )

    # scroll the swatch list once it is long, so a busy map keeps its legend
    scroll = "max-height:min(48vh,360px);overflow-y:auto;" if n_shown > 14 else ""
    return (
        f'<div style="position:fixed;bottom:24px;left:12px;z-index:9999;'
        f'background:{chrome["surface"]};color:{chrome["text_primary"]};'
        f'border:1px solid rgba(11,11,11,.12);border-radius:8px;padding:10px 12px;'
        f'font-family:{FONT_FAMILY};font-size:11px;line-height:1.4;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.12);font-variant-numeric:tabular-nums">'
        f'<div style="font-weight:600;font-size:12px;margin-bottom:6px">{title}</div>'
        f"{subtitle}"
        f'<div style="{scroll}">' + "".join(rows) + "</div>"
        + "</div>"
    )


def downsample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Evenly thin a frame so the map stays responsive."""
    if max_points and len(frame) > max_points:
        idx = np.round(np.linspace(0, len(frame) - 1, max_points)).astype(int)
        return frame.iloc[np.unique(idx)]
    return frame


# --------------------------------------------------------------------------- #
# Cell identity
# --------------------------------------------------------------------------- #


#: one step of the sequential ramp per tree depth (an ordinal encoding - never
#: a cycled categorical hue, which would imply groups that do not exist)
_DEPTH_COLORS = ["#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]


def _node_text(value) -> str:
    """Identity values are whole numbers - never render them as ``10.0``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _held_text(seconds: float) -> str:
    if seconds >= 90:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.0f} s"


def build_tree(
    table: pd.DataFrame,
    levels: Sequence[str],
    include_root: bool = True,
):
    """Lay an identity hierarchy out as a left-to-right node-link tree.

    Returns ``(nodes, edges)``: nodes carry ``x`` (depth), ``y`` (row), the
    identity value as ``label``, the level it belongs to, and the time held
    under it (``dwell``); edges are elbow paths from parent to child.  Children
    are ordered by identity value - never by time held, which says where the
    phone sat rather than how the network is built - and a parent sits at the
    mean row of its children.

    ``include_root`` keeps the synthetic "All" node.  Drawing leaves it out - a
    single root just adds a column of dead space.
    """
    root = {
        "id": "", "label": "All", "value": None, "level": None, "depth": 0,
        "dwell": 0.0, "children": [],
    }
    nodes = {"": root}

    cols = [table[level].tolist() for level in levels]
    dwells = table["dwell_s"].tolist()

    for row in range(len(table)):
        path = ""
        for depth, level in enumerate(levels, start=1):
            value = cols[depth - 1][row]
            path_next = f"{path}/{depth}={value}"
            node = nodes.get(path_next)
            if node is None:
                node = {
                    "id": path_next,
                    "label": _node_text(value),
                    "value": value,
                    "level": level,
                    "depth": depth,
                    "dwell": 0.0,
                    "children": [],
                }
                nodes[path_next] = node
                nodes[path]["children"].append(node)
            node["dwell"] += float(dwells[row])
            path = path_next
        root["dwell"] += float(dwells[row])

    def _by_value(node):
        value = node["value"]
        try:
            return (0, float(value), "")
        except (TypeError, ValueError):
            return (1, 0.0, str(value))

    order: List[dict] = []
    counter = [0]

    def place(node):
        if not node["children"]:
            node["y"] = float(counter[0])
            counter[0] += 1
        else:
            node["children"].sort(key=_by_value)
            for child in node["children"]:
                place(child)
            node["y"] = float(
                sum(c["y"] for c in node["children"]) / len(node["children"])
            )
        node["x"] = float(node["depth"])
        order.append(node)

    place(root)
    if not include_root:
        order = [n for n in order if n["depth"] > 0]

    edges = []
    for node in order:
        for child in node["children"]:
            # elbow: out from the parent, down the trunk, in to the child
            trunk = child["x"] - 0.45
            edges.append(
                [
                    (node["x"], node["y"]),
                    (trunk, node["y"]),
                    (trunk, child["y"]),
                    (child["x"], child["y"]),
                ]
            )
    return order, edges


def tree_figure(
    table: pd.DataFrame,
    levels: Sequence[str],
    level_labels: Sequence[str],
    mode: str = "light",
    max_leaves: int = 150,
    height: Optional[int] = None,
) -> go.Figure:
    """Node-link tree over ``levels``.

    Each level is a column, named once in a header row ("eNB ID", "Sector ID",
    "PCI"), so a node carries only its value.  A last column gives the time
    each leaf was held.  ``max_leaves`` is a safety cap for a table too large
    to read: the longest-held paths are kept.
    """
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()
    if table.empty or not levels:
        return fig

    nodes, edges = build_tree(
        table.nlargest(max_leaves, "dwell_s"), levels, include_root=False
    )
    if not nodes:
        return fig
    label_of = dict(zip(levels, level_labels))
    leaves = [n for n in nodes if not n["children"]]
    deepest = len(levels)

    edge_x: List[Optional[float]] = []
    edge_y: List[Optional[float]] = []
    for path in edges:
        for x, y in path:
            edge_x.append(x)
            edge_y.append(y)
        edge_x.append(None)
        edge_y.append(None)
    fig.add_trace(
        go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color=chrome["axis"], width=1),
            hoverinfo="skip", showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[n["x"] for n in nodes],
            y=[n["y"] for n in nodes],
            mode="markers+text",
            text=[n["label"] for n in nodes],
            # Leaves read best beside the marker. The outermost node has
            # nothing to its left, so its label goes there. A node in between
            # has a branch entering and leaving it, so its label sits above.
            textposition=[
                "middle right"
                if not n["children"]
                else ("middle left" if n["depth"] == 1 else "top center")
                for n in nodes
            ],
            textfont=dict(family=FONT_FAMILY, size=12, color=chrome["text_primary"]),
            marker=dict(
                size=10,
                color=[
                    _DEPTH_COLORS[min(n["depth"], len(_DEPTH_COLORS) - 1)]
                    for n in nodes
                ],
                line=dict(color=chrome["surface"], width=1.5),
            ),
            customdata=[
                [label_of.get(n["level"], n["level"]), _held_text(n["dwell"])]
                for n in nodes
            ],
            hovertemplate=(
                "<b>%{customdata[0]} %{text}</b>"
                "<br>held %{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    # time held, once per leaf, in a column of its own
    held_x = deepest + 0.9
    fig.add_trace(
        go.Scatter(
            x=[held_x] * len(leaves),
            y=[n["y"] for n in leaves],
            mode="text",
            text=[_held_text(n["dwell"]) for n in leaves],
            textposition="middle center",
            textfont=dict(family=FONT_FAMILY, size=12, color=chrome["text_secondary"]),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # one row per leaf, plus room for the header row
    if height is None:
        height = int(min(11000, max(170, 30 * len(leaves) + 90)))

    fig.update_layout(
        height=height,
        paper_bgcolor=chrome["surface"],
        plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=8, r=8, t=34, b=8),
        hoverlabel=dict(font_family=FONT_FAMILY),
    )
    # the header row: the x axis, drawn on top, with one tick per column
    fig.update_xaxes(
        side="top",
        tickmode="array",
        tickvals=[float(d) for d in range(1, deepest + 1)] + [held_x],
        ticktext=[f"<b>{label}</b>" for label in level_labels] + ["<b>Time held</b>"],
        tickfont=dict(family=FONT_FAMILY, size=12, color=chrome["text_secondary"]),
        ticks="",
        showgrid=False,
        zeroline=False,
        showline=False,
        range=[0.2, held_x + 0.6],
        fixedrange=True,
    )
    fig.update_yaxes(
        visible=False, range=[len(leaves) - 0.4, -0.9], fixedrange=True
    )
    return fig


def fan_out_figure(
    dist: pd.DataFrame,
    relationship: str,
    y_title: str = "Count",
    groups: Optional[Sequence[str]] = None,
    mode: str = "light",
    height: int = 380,
) -> go.Figure:
    """How many parents show exactly N children, one bar group per network.

    ``relationship`` names the count on the x axis ("Sectors seen per eNB")
    and ``y_title`` what a bar counts ("Number of eNBs").  ``groups`` fixes the
    order and the colour of the groups (the networks), so D2C keeps its colour
    whichever network has more eNBs; without it the largest group leads.
    """
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()
    sub = dist[dist["relationship"] == relationship] if not dist.empty else dist
    if sub.empty:
        return fig

    present = set(sub["group"])
    if groups is None:
        ranked = (
            sub.groupby("group", observed=True)["parents"]
            .sum()
            .sort_values(ascending=False)
        )
        groups = list(ranked.index)
    slots = [(i, g) for i, g in enumerate(groups) if g in present]
    single = len(slots) == 1

    for i, group in slots:
        part = sub[sub["group"] == group].sort_values("children")
        color = series_style(i, mode)[0]
        fig.add_trace(
            go.Bar(
                x=part["children"],
                y=part["parents"],
                name=str(group),
                marker=dict(color=color, cornerradius=4, line=dict(width=0)),
                text=part["parents"] if single else None,
                textposition="outside",
                textfont=dict(color=chrome["text_secondary"], size=11),
                hovertemplate=(
                    f"<b>{group}</b><br>{relationship}: %{{x}}"
                    f"<br>{y_title}: %{{y}}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=height,
        barmode="group",
        bargap=0.28,
        bargroupgap=0.08,
        paper_bgcolor=chrome["surface"],
        plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=64, r=24, t=16, b=52),
        hovermode="closest",
        hoverlabel=dict(font_family=FONT_FAMILY),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=11, color=chrome["text_secondary"]), bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        title=dict(text=relationship, font=dict(color=chrome["text_secondary"], size=12)),
        dtick=1, gridcolor="rgba(0,0,0,0)", zeroline=False,
        linecolor=chrome["axis"], tickfont=dict(color=chrome["muted"], size=11),
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=chrome["text_secondary"], size=12)),
        gridcolor=chrome["grid"], zeroline=False, linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11),
    )
    return fig


# --------------------------------------------------------------------------- #
# D2C satellite visibility
# --------------------------------------------------------------------------- #


def _sat_layout(fig, chrome, x_title, y_title, height=460, legend=False):
    fig.update_layout(
        height=height,
        paper_bgcolor=chrome["surface"],
        plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=64, r=24, t=16, b=52),
        hovermode="closest",
        hoverlabel=dict(font_family=FONT_FAMILY),
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=11, color=chrome["text_secondary"]), bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(color=chrome["text_secondary"], size=12)),
        gridcolor=chrome["grid"], zeroline=False, linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11),
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=chrome["text_secondary"], size=12)),
        gridcolor=chrome["grid"], zeroline=False, linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11),
    )
    return fig


def sat_timeseries_figure(
    ts: pd.DataFrame, count_col: str, mask: int, mode: str = "light", height: int = 460
) -> go.Figure:
    """Visible-satellite count over time, one trace per collection.

    ``ts`` has columns ``time``, ``collection`` and ``count_col`` and is already
    time-sorted. The eight busiest collections take the categorical hues; the
    rest fold into one muted "Other", so hues are never cycled.
    """
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()
    if ts.empty:
        return _sat_layout(fig, chrome, "Time", f"D2C satellites ≥{mask}°", height)

    ranked = ts["collection"].value_counts().index.tolist()
    top, folded = ranked[:8], ranked[8:]

    def _add(sub, name, color):
        if sub.empty:
            return
        fig.add_trace(
            go.Scattergl(
                x=sub["time"], y=sub[count_col], mode="markers",
                name=str(name),
                marker=dict(color=color, size=5, opacity=0.7, line=dict(width=0)),
                hovertemplate=(
                    f"<b>{name}</b><br>%{{x|%Y-%m-%d %H:%M:%S}}"
                    f"<br>%{{y}} sats ≥{mask}°<extra></extra>"
                ),
            )
        )

    for i, coll in enumerate(top):
        _add(ts[ts["collection"] == coll], coll, series_style(i, mode)[0])
    if folded:
        _add(ts[ts["collection"].isin(folded)], f"Other ({len(folded)})", OTHER_COLOR)

    return _sat_layout(
        fig, chrome, "Time", f"D2C satellites ≥{mask}°", height,
        legend=len(top) + (1 if folded else 0) >= 2,
    )


def sat_histogram_figure(
    counts: pd.Series, mask: int, mode: str = "light", height: int = 420
) -> go.Figure:
    """Distribution of the visible-satellite count across samples."""
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()
    vals = pd.to_numeric(counts, errors="coerce").dropna().astype(int)
    if vals.empty:
        return _sat_layout(fig, chrome, f"D2C satellites ≥{mask}°", "Samples", height)
    dist = vals.value_counts().sort_index()
    fig.add_trace(
        go.Bar(
            x=dist.index, y=dist.values,
            marker=dict(color=series_style(0, mode)[0], cornerradius=4, line=dict(width=0)),
            hovertemplate=f"%{{x}} sats ≥{mask}°<br>%{{y}} samples<extra></extra>",
        )
    )
    fig.update_xaxes(dtick=1)
    return _sat_layout(fig, chrome, f"D2C satellites visible (≥{mask}°)", "Samples", height)
