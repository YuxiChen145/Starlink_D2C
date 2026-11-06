"""CDF curves tab of the Starlink D2C explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qpviz import charts, stats
from qpviz.ui.context import Ctx


def render(ctx: Ctx) -> None:
    """Render the CDF curves tab."""
    MODE = ctx.mode
    active = ctx.active
    freq_label = ctx.freq_label
    freq_prefix = ctx.freq_prefix
    max_curve_points = ctx.max_curve_points
    metric = ctx.metric
    metric_meta = ctx.metric_meta
    min_samples = ctx.min_samples

    controls = st.columns([3, 2, 2])
    group_by = controls[0].multiselect(
        "Split curves by",
        options=list(stats.GROUP_DIMENSIONS),
        default=["pci"],
        format_func=lambda d: stats.GROUP_DIMENSIONS[d],
        help=(
            "Pick one or more dimensions. e.g. PCI alone gives one curve per PCI; "
            "add Collection to compare the same PCI across collections; add "
            f"{freq_label} to separate carriers."
        ),
    )
    max_curves = controls[1].number_input(
        "Max curves", min_value=1, max_value=60, value=12, step=1,
        help="Largest samples first.",
    )
    plot_as = controls[2].radio(
        "Plot as", ["CDF", "CCDF", "PDF"], horizontal=True, key="cdf_plot_as",
        help="CDF/CCDF are cumulative curves; PDF is an overlaid histogram — best "
             "for seeing where the mass piles up (e.g. the N-factor at 25/50/100).",
    )
    complementary = plot_as == "CCDF"
    pdf_bins = 60
    if plot_as == "PDF":
        pdf_bins = st.slider(
            "Histogram bins", min_value=10, max_value=200, value=60, step=5,
            key="cdf_pdf_bins",
            help="More bins → finer resolution near the integer values.",
        )
    sort_note = st.empty()

    curves = stats.cdf_curves(
        active,
        metric,
        group_by=group_by,
        min_samples=int(min_samples),
        # the PDF is binned from the raw samples; only the CDF lines are thinned
        max_points=None if plot_as == "PDF" else int(max_curve_points),
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
        if plot_as == "PDF":
            fig = charts.pdf_figure(
                curves, metric_meta.axis_label, mode=MODE, bins=int(pdf_bins)
            )
        else:
            fig = charts.cdf_figure(
                curves, metric_meta.axis_label, mode=MODE, complementary=complementary
            )
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
