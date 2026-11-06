"""Statistics tab of the Starlink D2C explorer."""

from __future__ import annotations

import streamlit as st

from qpviz import stats
from qpviz.ui.context import Ctx


def render(ctx: Ctx) -> None:
    """Render the Statistics tab."""
    active = ctx.active
    freq_prefix = ctx.freq_prefix
    metric = ctx.metric
    min_samples = ctx.min_samples

    stat_group = st.multiselect(
        "Group statistics by",
        options=list(stats.GROUP_DIMENSIONS),
        default=["collection", "pci"],
        format_func=lambda d: stats.GROUP_DIMENSIONS[d],
        key="stat_group",
    )
    table = stats.summary_table(
        active,
        metric,
        group_by=stat_group,
        min_samples=int(min_samples),
        freq_prefix=freq_prefix,
    )
    if table.empty:
        st.info("No group meets the minimum sample count.")
    else:
        st.dataframe(
            table.round(2),
            width="stretch",
            hide_index=True,
            height=520,
        )
        st.download_button(
            "Download statistics (CSV)",
            table.to_csv(index=False).encode(),
            file_name=f"{metric}_summary.csv",
            mime="text/csv",
        )
