"""Figure builders: CDF curves (plotly) and the measurement map (folium)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

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


def legend_html(
    edges: Sequence[float],
    colors: Sequence[str],
    title: str,
    mode: str = "light",
) -> str:
    chrome = CHROME.get(mode, CHROME["light"])
    rows = []
    for i, color in enumerate(colors):
        lo, hi = edges[i], edges[i + 1]
        rows.append(
            f'<div style="display:flex;align-items:center;gap:6px;line-height:1.5">'
            f'<span style="width:14px;height:10px;border-radius:2px;background:{color};'
            f'box-shadow:0 0 0 1px rgba(0,0,0,.12)"></span>'
            f'<span style="font-variant-numeric:tabular-nums">{lo:.1f} – {hi:.1f}</span>'
            f"</div>"
        )
    return (
        f'<div style="position:fixed;bottom:24px;left:12px;z-index:9999;'
        f'background:{chrome["surface"]};color:{chrome["text_primary"]};'
        f'border:1px solid rgba(11,11,11,.12);border-radius:8px;padding:8px 10px;'
        f'font:11px {FONT_FAMILY};box-shadow:0 2px 8px rgba(0,0,0,.12)">'
        f'<div style="font-weight:600;margin-bottom:4px">{title}</div>'
        + "".join(rows)
        + "</div>"
    )


def downsample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    """Evenly thin a frame so the map stays responsive."""
    if max_points and len(frame) > max_points:
        idx = np.round(np.linspace(0, len(frame) - 1, max_points)).astype(int)
        return frame.iloc[np.unique(idx)]
    return frame
