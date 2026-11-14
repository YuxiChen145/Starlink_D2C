"""CDF curves tab of the Starlink D2C explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qpviz import charts, stats
from qpviz.ui.context import Ctx


#: each curve is thinned to at most this many points before it is drawn
MAX_CURVE_POINTS = 2000

#: what the curves can be split by, in menu order. eNB / gNB is offered only
#: when the metric's export carries it (the signal metrics).
SPLIT_OPTIONS = ("network", "collection", "provider", "enb")


def render(ctx: Ctx) -> None:
    """Render the CDF curves tab."""
    MODE = ctx.mode
    active = ctx.active
    freq_prefix = ctx.freq_prefix
    metric = ctx.metric
    metric_meta = ctx.metric_meta
    min_samples = ctx.min_samples

    options = [
        d for d in SPLIT_OPTIONS
        if d != "enb" or ("enb" in active.columns and active["enb"].notna().any())
    ]
    controls = st.columns([3, 2])
    group_by = controls[0].multiselect(
        "Split curves by",
        options=options,
        default=["network"],
        format_func=lambda d: stats.GROUP_DIMENSIONS[d],
        # the eNB / gNB option comes and goes with the metric
        key=f"cdf_split_{'-'.join(options)}",
        help=(
            "Network gives one curve for D2C and one for terrestrial. Add "
            "Collection to compare sessions, Measurement source to compare "
            "instruments, or eNB / gNB to compare cells. Leave empty for one "
            "curve of everything."
        ),
    )
    max_curves = controls[1].number_input(
        "Max curves", min_value=1, max_value=60, value=12, step=1,
        help="Largest samples first.",
    )
    sort_note = st.empty()

    curves = stats.cdf_curves(
        active,
        metric,
        group_by=group_by,
        min_samples=int(min_samples),
        max_points=MAX_CURVE_POINTS,
        freq_prefix=freq_prefix,
    )
    total_curves = len(curves)
    curves = curves[: int(max_curves)]
    sort_note.caption(
        f"{total_curves} group(s) ≥ {int(min_samples)} samples; showing {len(curves)}."
    )

    if not curves:
        st.info(
            "No group has enough samples. Lower “Min samples per curve” in the "
            "sidebar or widen the filters."
        )
    else:
        fig = charts.cdf_figure(curves, metric_meta.axis_label, mode=MODE)
        st.plotly_chart(fig, width="stretch")
        with st.expander("Curve values as a table"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Curve": c.name,
                            "Samples": c.n,
                            "P5": float(pd.Series(c.x).quantile(0.05)),
                            "Median": float(pd.Series(c.x).quantile(0.50)),
                            "P95": float(pd.Series(c.x).quantile(0.95)),
                        }
                        for c in curves
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
