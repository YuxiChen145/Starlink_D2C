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

    collections = list(dict.fromkeys(runs["collection"].tolist())) if not runs.empty else []
    for i, collection in enumerate(collections):
        color, _ = series_style(i, mode)
        sub = runs[(runs["collection"] == collection) & (runs["value"].isin(keep))]
        if sub.empty:
            continue
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
                name=str(collection),
                marker=dict(color=color, line=dict(width=0)),
                width=0.72,
                customdata=np.stack(
                    [sub["duration_s"].to_numpy(), sub["samples"].to_numpy()], axis=-1
                ),
                hovertemplate=(
                    f"<b>{level_label} %{{y}}</b><br>{collection}"
                    f"<br>start {hover_x}"
                    "<br>held %{customdata[0]:.1f}s (%{customdata[1]} samples)"
                    "<extra></extra>"
                ),
            )
        )

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


#: one step of the sequential ramp per hierarchy depth (an ordinal encoding -
#: never a cycled categorical hue, which would imply groups that do not exist)
_DEPTH_COLORS = ["#f0efec", "#5598e7", "#9ec5f4", "#cde2fb", "#e4eefc"]


def _node_text(value) -> str:
    """Identity values are whole numbers - never render them as ``10.0``."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def treemap_figure(
    table: pd.DataFrame,
    levels: Sequence[str],
    level_labels: Sequence[str],
    mode: str = "light",
    max_leaves: int = 400,
    height: int = 620,
) -> go.Figure:
    """Nested rectangles over ``levels``, sized by dwell time."""
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()
    if table.empty or not levels:
        return fig

    table = table.nlargest(max_leaves, "dwell_s")
    # .tolist() per column, never iterrows(): a mixed-dtype row would coerce
    # integer cell ids to floats and label the tiles "eNB 10.0"
    columns = [table[level].tolist() for level in levels]
    dwells = table["dwell_s"].tolist()

    ids, labels, parents, values, colors = [], [], [], [], []
    index: Dict[str, int] = {}

    root = "All"
    index[root] = 0
    ids.append(root)
    labels.append(root)
    parents.append("")
    values.append(0.0)
    colors.append(_DEPTH_COLORS[0])

    for row in range(len(table)):
        path = root
        for depth, label in enumerate(level_labels, start=1):
            value = columns[depth - 1][row]
            path_next = f"{path}/{levels[depth - 1]}={value}"
            if path_next not in index:
                index[path_next] = len(ids)
                ids.append(path_next)
                labels.append(f"{label} {_node_text(value)}")
                parents.append(path)
                values.append(0.0)
                colors.append(_DEPTH_COLORS[min(depth, len(_DEPTH_COLORS) - 1)])
            if depth == len(levels):
                values[index[path_next]] = float(dwells[row])
            path = path_next

    fig.add_trace(
        go.Treemap(
            ids=ids, labels=labels, parents=parents, values=values,
            branchvalues="remainder",
            marker=dict(colors=colors, line=dict(color=chrome["surface"], width=2)),
            textfont=dict(family=FONT_FAMILY, size=12, color=chrome["text_primary"]),
            tiling=dict(pad=2),
            hovertemplate="<b>%{label}</b><br>%{value:.0f}s held<extra></extra>",
            texttemplate="%{label}",
        )
    )
    fig.update_layout(
        height=height,
        paper_bgcolor=chrome["surface"],
        plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=4, r=4, t=4, b=4),
        hoverlabel=dict(font_family=FONT_FAMILY),
    )
    return fig
