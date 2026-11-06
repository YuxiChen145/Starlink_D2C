"""Raw data tab of the Starlink D2C explorer."""

from __future__ import annotations

import streamlit as st

from qpviz.ui.context import Ctx


def render(ctx: Ctx) -> None:
    """Render the Raw data tab."""
    active = ctx.active
    metric = ctx.metric
    sat_active = ctx.sat_active
    sat_vis_col = ctx.sat_vis_col

    show_cols = [
        "time",
        "collection",
        "source_label",
        "tech",
        "link",
        "band",
        "freq",
        "pci",
        "lat",
        "lon",
        metric,
    ]
    if sat_active and sat_vis_col in active.columns:
        show_cols += [sat_vis_col, "sat_max_elev"]
    show_cols = [c for c in show_cols if c in active.columns]
    st.dataframe(active[show_cols].head(5000), width="stretch", height=520)
    st.caption(f"Showing the first 5,000 of {len(active):,} filtered rows.")
    st.download_button(
        "Download filtered rows (CSV)",
        active[show_cols].to_csv(index=False).encode(),
        file_name=f"{metric}_filtered.csv",
        mime="text/csv",
    )
