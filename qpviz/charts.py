"""Figure builders: CDF curves (plotly) and the measurement map (folium)."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .stats import Curve
from .theme import (
    CHROME,
    DIVERGING_RED_BLUE,
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
    complementary: bool = False,
    show_direct_labels: bool = True,
    height: int = 560,
) -> go.Figure:
    """One ECDF line per curve, all on the same canvas."""
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()

    y_title = "P(X > x)" if complementary else "P(X ≤ x)"

    for i, curve in enumerate(curves):
        color, dash = series_style(i, mode)
        y = 1.0 - curve.y if complementary else curve.y
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
            y = 1.0 - curve.y if complementary else curve.y
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
    scale: str = "diverging",
    midpoint: Optional[float] = None,
    n_bins: int = 9,
) -> Tuple[List[float], List[str]]:
    """Return ``(edges, colors)`` where ``len(edges) == len(colors) + 1``.

    ``sequential`` uses a single blue hue light -> dark over the 5th-95th
    percentile range.  ``diverging`` uses red <-> blue with a neutral grey
    midpoint and an equal number of steps per arm.
    """
    arr = _clean(values)
    if arr.size == 0:
        return [0.0, 1.0], [SEQUENTIAL_BLUE[-1]]

    lo, hi = float(np.percentile(arr, 2)), float(np.percentile(arr, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        hi = lo + 1.0

    if scale == "sequential":
        colors = SEQUENTIAL_BLUE[:n_bins]
        edges = list(np.linspace(lo, hi, len(colors) + 1))
        return edges, colors

    mid = float(np.median(arr)) if midpoint is None else float(midpoint)
    mid = min(max(mid, lo + 1e-9), hi - 1e-9)
    colors = list(DIVERGING_RED_BLUE)  # 4 red + neutral + 4 blue
    arm = (len(colors) - 1) // 2
    half = min((mid - lo) / (arm + 0.5), (hi - mid) / (arm + 0.5)) / 2.0
    half = max(half, 1e-9)
    lower = list(np.linspace(lo, mid - half, arm + 1))
    upper = list(np.linspace(mid + half, hi, arm + 1))
    edges = lower + upper  # len == len(colors) + 1
    return edges, colors


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
    """Compact tick label - throughput runs to tens of thousands of kbps."""
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


#: everything past the eight categorical slots folds into one muted "Other"
OTHER_COLOR = "#898781"


def categorical_bins(values, max_slots: int = 8, mode: str = "light"):
    """Assign the ``max_slots`` most common values a categorical hue each.

    Hues are never cycled: a ninth value is not given a repeated colour, it
    folds into "Other" - so two differently coloured points always mean two
    different identities.
    """
    series = pd.Series(values).dropna()
    if series.empty:
        return [], {}, 0
    counts = series.value_counts()
    top = list(counts.index[:max_slots])
    colors = {value: series_style(i, mode)[0] for i, value in enumerate(top)}
    others = int(counts.iloc[max_slots:].sum()) if len(counts) > max_slots else 0
    return top, colors, others


def categorical_legend_html(
    order: Sequence,
    colors: dict,
    title: str,
    mode: str = "light",
    other_count: int = 0,
    value_prefix: str = "",
) -> str:
    """Swatch legend for a categorically coloured map."""
    chrome = CHROME.get(mode, CHROME["light"])
    rows = []
    for value in order:
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;line-height:1.6">'
            f'<span style="width:12px;height:12px;border-radius:3px;'
            f'background:{colors[value]};box-shadow:0 0 0 1px rgba(11,11,11,.12)">'
            f"</span><span>{value_prefix}{value}</span></div>"
        )
    if other_count:
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;line-height:1.6">'
            f'<span style="width:12px;height:12px;border-radius:3px;'
            f'background:{OTHER_COLOR};box-shadow:0 0 0 1px rgba(11,11,11,.12)">'
            f'</span><span>Other ({other_count:,} pts)</span></div>'
        )
    return (
        f'<div style="position:fixed;bottom:24px;left:12px;z-index:9999;'
        f'background:{chrome["surface"]};color:{chrome["text_primary"]};'
        f'border:1px solid rgba(11,11,11,.12);border-radius:8px;padding:10px 12px;'
        f'font-family:{FONT_FAMILY};font-size:11px;line-height:1.4;'
        f'box-shadow:0 2px 8px rgba(0,0,0,.12);font-variant-numeric:tabular-nums">'
        f'<div style="font-weight:600;font-size:12px;margin-bottom:6px">{title}</div>'
        + "".join(rows)
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


def timeline_figure(
    runs: pd.DataFrame,
    identities: Sequence,
    level_label: str,
    mode: str = "light",
    x_axis: str = "elapsed",
    height: Optional[int] = None,
) -> go.Figure:
    """A swimlane of serving-cell dwell runs: one bar per continuous stretch.

    ``identities`` fixes the row order (first-seen order draws the staircase a
    satellite pass makes). ``x_axis`` is ``elapsed`` minutes from each
    collection's first sample, or ``absolute`` wall-clock time.
    """
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()
    labels = [str(v) for v in identities]
    keep = set(identities)

    # Colour by collection, but never cycle hues: the eight with the most runs
    # take the categorical slots and the remainder share one muted "Other".
    if runs.empty:
        collections, folded = [], []
    else:
        ranked = runs["collection"].value_counts().index.tolist()
        collections, folded = ranked[:8], ranked[8:]

    def _add(sub, name, color):
        if sub.empty:
            return
        if x_axis == "absolute":
            base = sub["start"]
            width = (sub["end"] - sub["start"]).dt.total_seconds() * 1000.0
            hover_x = "%{base|%Y-%m-%d %H:%M:%S}"
        else:
            base = sub["elapsed_start"] / 60.0
            width = sub["duration_s"] / 60.0
            hover_x = "%{base:.2f} min"
        fig.add_trace(
            go.Bar(
                x=width,
                y=[str(v) for v in sub["value"]],
                base=base,
                orientation="h",
                name=str(name),
                marker=dict(color=color, line=dict(width=0)),
                width=0.72,
                customdata=np.stack(
                    [
                        sub["duration_s"].to_numpy(),
                        sub["samples"].to_numpy(),
                        sub["collection"].astype(str).to_numpy(),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    f"<b>{level_label} %{{y}}</b><br>%{{customdata[2]}}"
                    f"<br>start {hover_x}"
                    "<br>held %{customdata[0]:.1f}s (%{customdata[1]} samples)"
                    "<extra></extra>"
                ),
            )
        )

    for i, collection in enumerate(collections):
        _add(
            runs[(runs["collection"] == collection) & (runs["value"].isin(keep))],
            collection,
            series_style(i, mode)[0],
        )
    if folded:
        _add(
            runs[runs["collection"].isin(folded) & runs["value"].isin(keep)],
            f"Other ({len(folded)} collections)",
            OTHER_COLOR,
        )
        collections = collections + ["other"]

    if height is None:
        height = int(min(1400, max(320, 18 * len(labels) + 120)))

    fig.update_layout(
        barmode="overlay",
        bargap=0.15,
        height=height,
        paper_bgcolor=chrome["surface"],
        plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=8, r=16, t=8, b=48),
        hovermode="closest",
        hoverlabel=dict(font_family=FONT_FAMILY),
        showlegend=len(collections) >= 2,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(size=11, color=chrome["text_secondary"]), bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        title=dict(
            text="Time" if x_axis == "absolute" else "Elapsed (minutes)",
            font=dict(color=chrome["text_secondary"], size=12),
        ),
        gridcolor=chrome["grid"], zeroline=False, linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11),
    )
    fig.update_yaxes(
        title=dict(text=level_label, font=dict(color=chrome["text_secondary"], size=12)),
        type="category", categoryorder="array", categoryarray=labels,
        autorange="reversed",  # first identity at the top, so time reads downward
        gridcolor="rgba(0,0,0,0)", linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=10),
    )
    return fig


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
    level_labels: Sequence[str],
    include_root: bool = True,
):
    """Lay a dwell hierarchy out as a left-to-right node-link tree.

    Returns ``(nodes, edges)``: nodes carry ``x`` (depth), ``y`` (row), label,
    dwell and depth; edges are elbow paths from parent to child.  Children are
    ordered by dwell, and a parent sits at the mean row of its children.

    ``include_root`` keeps the synthetic "All" node.  Drawing leaves it out -
    a single root just adds a column of dead space, and the running total is
    already in the caption above the figure.
    """
    root = {"id": "", "label": "All", "depth": 0, "dwell": 0.0, "children": []}
    nodes = {"": root}

    columns = [table[level].tolist() for level in levels]
    dwells = table["dwell_s"].tolist()

    for row in range(len(table)):
        path = ""
        for depth, label in enumerate(level_labels, start=1):
            value = columns[depth - 1][row]
            path_next = f"{path}/{depth}={value}"
            node = nodes.get(path_next)
            if node is None:
                node = {
                    "id": path_next,
                    "label": f"{label} {_node_text(value)}",
                    "depth": depth,
                    "dwell": 0.0,
                    "children": [],
                }
                nodes[path_next] = node
                nodes[path]["children"].append(node)
            node["dwell"] += float(dwells[row])
            path = path_next
        root["dwell"] += float(dwells[row])

    # depth-first, biggest branch first, assigning each leaf its own row
    order: List[dict] = []
    counter = [0]

    def place(node):
        if not node["children"]:
            node["y"] = float(counter[0])
            counter[0] += 1
        else:
            node["children"].sort(key=lambda c: -c["dwell"])
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
        if not include_root and node["depth"] == 0:
            continue
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
    """Node-link tree over ``levels``, with dwell time on every node."""
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()
    if table.empty or not levels:
        return fig

    nodes, edges = build_tree(
        table.nlargest(max_leaves, "dwell_s"),
        levels,
        level_labels,
        include_root=False,
    )
    if not nodes:
        return fig

    # Held time is worth reading off the outermost grouping and off the leaves;
    # putting it on every intermediate node just crowds the picture.
    outermost = min(n["depth"] for n in nodes)
    labels = [
        f"{n['label']} · {_held_text(n['dwell'])}"
        if n["depth"] == outermost or not n["children"]
        else n["label"]
        for n in nodes
    ]

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

    biggest = max((n["dwell"] for n in nodes), default=1.0) or 1.0
    depth_of = [n["depth"] for n in nodes]
    fig.add_trace(
        go.Scatter(
            x=[n["x"] for n in nodes],
            y=[n["y"] for n in nodes],
            mode="markers+text",
            text=labels,
            # A node with children has a branch line leaving it at its own
            # height, so its label goes above the line; leaves have no
            # outgoing line and read better beside the marker.
            textposition=[
                "middle right" if not n["children"] else "top right" for n in nodes
            ],
            textfont=dict(family=FONT_FAMILY, size=11, color=chrome["text_primary"]),
            marker=dict(
                size=[7 + 11 * (n["dwell"] / biggest) ** 0.5 for n in nodes],
                color=[
                    _DEPTH_COLORS[min(d, len(_DEPTH_COLORS) - 1)] for d in depth_of
                ],
                line=dict(color=chrome["surface"], width=1.5),
            ),
            customdata=[[n["dwell"]] for n in nodes],
            hovertemplate="<b>%{text}</b><br>held %{customdata[0]:.0f}s<extra></extra>",
            showlegend=False,
        )
    )

    leaves = sum(1 for n in nodes if not n["children"])
    if height is None:
        height = int(min(2800, max(320, 26 * leaves + 80)))

    fig.update_layout(
        height=height,
        paper_bgcolor=chrome["surface"],
        plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=8, r=8, t=8, b=8),
        hoverlabel=dict(font_family=FONT_FAMILY),
    )
    fig.update_xaxes(
        visible=False, range=[0.6, len(levels) + 2.1], fixedrange=False
    )
    fig.update_yaxes(
        visible=False, autorange="reversed", range=None
    )
    return fig


def fan_out_figure(
    dist: pd.DataFrame,
    relationship: str,
    mode: str = "light",
    cumulative: bool = False,
    normalise: bool = False,
    height: int = 420,
) -> go.Figure:
    """How many parents carry N children.

    The counts are small integers, so bars are the honest default; the
    cumulative form is the same population read as "P(children <= n)".
    """
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()
    sub = dist[dist["relationship"] == relationship] if not dist.empty else dist
    if sub.empty:
        return fig

    ranked = (
        sub.groupby("group", observed=True)["parents"].sum().sort_values(ascending=False)
    )
    groups, folded = list(ranked.index[:8]), list(ranked.index[8:])
    if folded:
        sub = sub.copy()
        sub.loc[sub["group"].isin(folded), "group"] = f"Other ({len(folded)})"
        sub = (
            sub.groupby(["relationship", "group", "children"], observed=True)
            .agg(parents=("parents", "sum"))
            .reset_index()
        )
        totals = sub.groupby("group", observed=True)["parents"].transform("sum")
        sub["share"] = sub["parents"] / totals
        groups = groups + [f"Other ({len(folded)})"]

    single = len(groups) == 1
    y_title = "Share of parents" if (normalise or cumulative) else "Parents"

    for i, group in enumerate(groups):
        part = sub[sub["group"] == group].sort_values("children")
        if part.empty:
            continue
        color = OTHER_COLOR if str(group).startswith("Other (") else series_style(i, mode)[0]
        if cumulative:
            fig.add_trace(
                go.Scatter(
                    x=part["children"],
                    y=part["share"].cumsum(),
                    mode="lines+markers",
                    line=dict(color=color, width=2, shape="hv"),
                    marker=dict(size=8, color=color),
                    name=str(group),
                    hovertemplate=(
                        f"<b>{group}</b><br>≤ %{{x}} children"
                        "<br>%{y:.1%} of parents<extra></extra>"
                    ),
                )
            )
        else:
            values = part["share"] if normalise else part["parents"]
            fig.add_trace(
                go.Bar(
                    x=part["children"],
                    y=values,
                    name=str(group),
                    marker=dict(color=color, cornerradius=4, line=dict(width=0)),
                    text=part["parents"] if single else None,
                    textposition="outside",
                    textfont=dict(color=chrome["text_secondary"], size=11),
                    hovertemplate=(
                        f"<b>{group}</b><br>%{{x}} children"
                        "<br>%{y}<extra></extra>"
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
        showlegend=len(groups) >= 2,
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
        tickformat=".0%" if (normalise or cumulative) else None,
        range=[0, 1.02] if cumulative else None,
    )
    return fig
