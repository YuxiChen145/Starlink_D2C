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


def pdf_figure(
    curves: Sequence[Curve],
    axis_label: str,
    mode: str = "light",
    bins: int = 60,
    height: int = 560,
) -> go.Figure:
    """Overlaid histograms (a PDF per group) sharing one set of bins.

    Each group is normalised to a probability (its bars sum to 1), so groups of
    different sizes are comparable and the mass piling up at particular values
    (e.g. the RSRQ N-factor at 25 / 50 / 100 RBs) is easy to read.
    """
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()

    allx = np.concatenate(
        [np.asarray(c.x, dtype="float64") for c in curves if len(c.x)]
    ) if curves else np.array([])
    allx = allx[np.isfinite(allx)]
    if allx.size:
        lo, hi = float(allx.min()), float(allx.max())
        if hi <= lo:
            hi = lo + 1.0
        size = (hi - lo) / max(1, int(bins))
        xbins = dict(start=lo, end=hi + size, size=size)
    else:
        xbins = None

    for i, curve in enumerate(curves):
        color, _ = series_style(i, mode)
        fig.add_trace(
            go.Histogram(
                x=curve.x,
                name=f"{curve.name}  (n={curve.n:,})",
                histnorm="probability",
                xbins=xbins,
                marker=dict(color=color, line=dict(width=0)),
                opacity=0.55 if len(curves) > 1 else 0.85,
                hovertemplate=(
                    f"<b>{curve.name}</b><br>{axis_label}: %{{x}}"
                    "<br>%{y:.1%} of samples<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=height,
        barmode="overlay",
        bargap=0.02,
        paper_bgcolor=chrome["surface"],
        plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=64, r=24, t=16, b=56),
        hovermode="closest",
        hoverlabel=dict(font_family=FONT_FAMILY),
        showlegend=len(curves) >= 2,
        legend=dict(
            orientation="v", yanchor="top", y=1, xanchor="left", x=1.01,
            font=dict(size=11, color=chrome["text_secondary"]), bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(
        title=dict(text=axis_label, font=dict(color=chrome["text_secondary"], size=12)),
        gridcolor=chrome["grid"], zeroline=False, linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11),
    )
    fig.update_yaxes(
        title=dict(text="Fraction of samples", font=dict(color=chrome["text_secondary"], size=12)),
        gridcolor=chrome["grid"], zeroline=False, linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11), tickformat=".0%",
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


def identity_tracks_figure(
    frame: pd.DataFrame,
    levels: Sequence[str] = ("pci", "sector", "enb"),
    labels: Optional[Sequence[str]] = None,
    mode: str = "light",
    x_axis: str = "absolute",
    height: Optional[int] = None,
) -> go.Figure:
    """Stacked, time-aligned step tracks of several identity levels.

    One row per level, all sharing the time axis, so a change in one line can be
    read straight down against the others: at a PCI change, did the sector or eNB
    change too? At a sector change, did the eNB? Each track is a step line with a
    larger dot at every transition, and hovering a time reads all three values at
    once. ``x_axis`` is wall-clock ``absolute`` or ``elapsed`` minutes from the
    first sample (intended for one collection at a time).
    """
    from plotly.subplots import make_subplots

    chrome = CHROME.get(mode, CHROME["light"])
    levels = [l for l in levels if l in frame.columns]
    n = len(levels)
    if frame.empty or n == 0:
        return go.Figure()
    label_map = {
        l: (labels[i] if labels is not None and i < len(labels) else l)
        for i, l in enumerate(levels)
    }

    work = frame[["time", *levels]].copy()
    work["time"] = pd.to_datetime(work["time"], errors="coerce")
    work = work.dropna(subset=["time"]).sort_values("time")
    if work.empty:
        return go.Figure()

    if x_axis == "elapsed":
        x = ((work["time"] - work["time"].iloc[0]).dt.total_seconds() / 60.0).to_numpy()
        x_title = "Elapsed (minutes)"
    else:
        x = work["time"].to_numpy()
        x_title = "Time"

    palette = generate_palette(n, mode)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.06)

    for i, level in enumerate(levels):
        colour = palette[i]
        vals = work[level].astype("string")
        order = list(dict.fromkeys(vals.dropna().tolist()))
        yv = vals.where(vals.notna(), None).to_numpy()
        # step line: the value held until it jumps, so a change is a visible step
        fig.add_trace(
            go.Scatter(
                x=x, y=yv, mode="lines",
                line=dict(shape="hv", color=colour, width=2),
                connectgaps=False, name=label_map[level],
                showlegend=False,
                hovertemplate=f"<b>{label_map[level]}</b> %{{y}}<extra></extra>",
            ),
            row=i + 1, col=1,
        )
        # a bigger dot at each transition, to draw the eye to the changes and let
        # coincident changes across rows line up vertically
        changed = vals.ne(vals.shift()) & vals.notna()
        if len(changed):
            changed.iloc[0] = False
        if changed.any():
            cx = np.asarray(x)[changed.to_numpy()]
            fig.add_trace(
                go.Scatter(
                    x=cx, y=vals[changed].to_numpy(), mode="markers",
                    marker=dict(size=9, color=colour,
                                line=dict(width=1.5, color=chrome["surface"])),
                    showlegend=False, hoverinfo="skip",
                ),
                row=i + 1, col=1,
            )
        fig.update_yaxes(
            type="category", categoryorder="array", categoryarray=order,
            title=dict(text=label_map[level],
                       font=dict(color=chrome["text_secondary"], size=12)),
            gridcolor=chrome["grid"], linecolor=chrome["axis"],
            tickfont=dict(color=chrome["muted"], size=10),
            row=i + 1, col=1,
        )

    if height is None:
        height = int(min(1200, 150 * n + 120))
    fig.update_layout(
        height=height,
        paper_bgcolor=chrome["surface"], plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=8, r=16, t=10, b=48),
        hovermode="x unified",
        hoverlabel=dict(font_family=FONT_FAMILY),
    )
    fig.update_xaxes(
        gridcolor=chrome["grid"], zeroline=False, linecolor=chrome["axis"],
        tickfont=dict(color=chrome["muted"], size=11),
    )
    # x title only under the bottom track
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(color=chrome["text_secondary"], size=12)),
        row=n, col=1,
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


def _balanced_columns(branches: List[dict], n: int) -> List[List[dict]]:
    """Split dwell-ordered branches into ``n`` contiguous groups of roughly
    equal leaf count, so the columns come out about the same height while the
    biggest-dwell branches stay in the first column."""
    if n <= 1 or len(branches) <= 1:
        return [branches]
    total = sum(b["_leaves"] for b in branches) or 1
    groups: List[List[dict]] = []
    current: List[dict] = []
    acc = 0
    for i, branch in enumerate(branches):
        current.append(branch)
        acc += branch["_leaves"]
        remaining = len(branches) - i - 1
        # close this column once it holds its share, provided enough branches
        # are left to fill the columns still to come
        if (
            len(groups) < n - 1
            and acc >= total * (len(groups) + 1) / n
            and remaining >= (n - 1 - len(groups))
        ):
            groups.append(current)
            current = []
    groups.append(current)
    return [g for g in groups if g]


def build_tree(
    table: pd.DataFrame,
    levels: Sequence[str],
    level_labels: Sequence[str],
    include_root: bool = True,
    columns: int = 1,
):
    """Lay a dwell hierarchy out as a left-to-right node-link tree.

    Returns ``(nodes, edges)``: nodes carry ``x`` (depth, offset by column),
    ``y`` (row within the column), ``col`` (column index), label, dwell and
    depth; edges are elbow paths from parent to child.  Children are ordered by
    dwell, and a parent sits at the mean row of its children.

    ``columns`` breaks the outermost level (the eNBs) across that many
    side-by-side sub-trees, balanced by leaf count, so a tree of hundreds of
    paths is a few short columns instead of one unreadable strip.  Columns only
    apply with ``include_root=False`` (the root joins every branch, so it
    cannot sit in one column); the root is placed at column 0 otherwise.

    ``include_root`` keeps the synthetic "All" node.  Drawing leaves it out - a
    single root just adds a column of dead space, and the running total is
    already in the caption above the figure.
    """
    root = {"id": "", "label": "All", "depth": 0, "dwell": 0.0, "children": []}
    nodes = {"": root}

    cols = [table[level].tolist() for level in levels]
    dwells = table["dwell_s"].tolist()

    for row in range(len(table)):
        path = ""
        for depth, label in enumerate(level_labels, start=1):
            value = cols[depth - 1][row]
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

    # dwell-order every sibling list, and count the leaves under each node
    def count_leaves(node) -> int:
        if not node["children"]:
            node["_leaves"] = 1
        else:
            node["children"].sort(key=lambda c: -c["dwell"])
            node["_leaves"] = sum(count_leaves(c) for c in node["children"])
        return node["_leaves"]

    count_leaves(root)

    n_cols = 1
    if not include_root and root["children"]:
        n_cols = max(1, min(int(columns), len(root["children"])))
    groups = (
        _balanced_columns(root["children"], n_cols)
        if not include_root
        else [root["children"]]
    )

    # one column occupies the depth span plus a gutter wide enough for the
    # rightmost (leaf) labels of the column to its left
    col_width = len(levels) + 2.6

    order: List[dict] = []

    def place(node, counter, x_offset, col_index):
        if not node["children"]:
            node["y"] = float(counter[0])
            counter[0] += 1
        else:
            for child in node["children"]:
                place(child, counter, x_offset, col_index)
            node["y"] = float(
                sum(c["y"] for c in node["children"]) / len(node["children"])
            )
        node["x"] = float(node["depth"]) + x_offset
        node["col"] = col_index
        order.append(node)

    for col_index, group in enumerate(groups):
        counter = [0]
        x_offset = col_index * col_width
        for branch in group:
            place(branch, counter, x_offset, col_index)

    if include_root:
        top = root["children"]
        root["x"] = 0.0
        root["y"] = (
            float(sum(c["y"] for c in top) / len(top)) if top else 0.0
        )
        root["col"] = 0
        order.append(root)
    else:
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
    columns: int = 1,
    height: Optional[int] = None,
) -> go.Figure:
    """Node-link tree over ``levels``, with dwell time on every node.

    ``columns`` splits the eNBs across that many side-by-side sub-trees, so a
    tall strip of hundreds of paths becomes a few short, readable columns.
    """
    chrome = CHROME.get(mode, CHROME["light"])
    fig = go.Figure()
    if table.empty or not levels:
        return fig

    nodes, edges = build_tree(
        table.nlargest(max_leaves, "dwell_s"),
        levels,
        level_labels,
        include_root=False,
        columns=columns,
    )
    if not nodes:
        return fig

    # Held time is worth reading off the leaves, and off the outermost grouping
    # too when there is horizontal room for it. Split into columns the room is
    # gone - the eNB label would run into the sector level - so drop it there
    # and leave it to the hover; the leaves keep it either way.
    outermost = min(n["depth"] for n in nodes)
    show_outer_time = columns <= 1
    labels = [
        f"{n['label']} · {_held_text(n['dwell'])}"
        if (not n["children"]) or (n["depth"] == outermost and show_outer_time)
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
            # Leaves read best beside the marker. Inner nodes have a branch
            # line leaving them, so their label goes above it. The outermost
            # (eNB) node in a multi-column layout puts its label to the LEFT,
            # into the empty gutter, so a long gNB-Id cannot run into the
            # sector level of a narrow column.
            textposition=[
                "middle right"
                if not n["children"]
                else ("middle left"
                      if n["depth"] == outermost and columns > 1
                      else "top right")
                for n in nodes
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

    # Height follows the tallest column, not the total leaf count - that is the
    # whole point of columns. Each leaf needs its own row plus a line for the
    # sector label above it, so the figure GROWS with the leaf count (the page
    # scrolls) rather than compressing rows into an unreadable smear once a
    # column passes ~100 leaves. The ceiling is a safety limit, not a target:
    # past it, focus on one eNB or lower the leaf cap.
    per_column: Dict[int, int] = {}
    for n in nodes:
        if not n["children"]:
            per_column[n["col"]] = per_column.get(n["col"], 0) + 1
    tallest = max(per_column.values(), default=1)
    if height is None:
        height = int(min(11000, max(320, 30 * tallest + 80)))

    fig.update_layout(
        height=height,
        paper_bgcolor=chrome["surface"],
        plot_bgcolor=chrome["surface"],
        font=dict(family=FONT_FAMILY, color=chrome["text_primary"], size=13),
        margin=dict(l=8, r=8, t=8, b=8),
        hoverlabel=dict(font_family=FONT_FAMILY),
    )
    # left labels of column 0 spill left of its eNB nodes (x=1) in multi-column
    # mode, so widen the left margin to keep them on canvas
    right = max((n["x"] for n in nodes), default=float(len(levels)))
    left = -1.0 if columns > 1 else 0.6
    fig.update_xaxes(
        visible=False, range=[left, right + 1.8], fixedrange=False
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
