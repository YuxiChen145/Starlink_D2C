"""Map tab of the Starlink D2C explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qpviz import filters as flt, identity, mapview
from qpviz.ui.context import Ctx


def render(ctx: Ctx) -> None:
    """Render the Map tab."""
    MODE = ctx.mode
    active = ctx.active
    carto_key = ctx.carto_key
    map_scale = ctx.map_scale
    max_map_points = ctx.max_map_points
    metric = ctx.metric
    metric_meta = ctx.metric_meta
    sat_mask = ctx.sat_mask
    sat_vis_col = ctx.sat_vis_col
    show_sat_kpi = ctx.show_sat_kpi
    tiles = ctx.tiles
    unregioned = ctx.unregioned

    from streamlit_folium import st_folium

    left, right = st.columns([4, 1])
    with right:
        _SAT_OPT = "D2C satellites visible"
        _SEC_OPT = identity.SECTORS_PER_ENB_LABEL
        # How many distinct sector ids each eNB carries, counted over the
        # filtered selection so the number describes what is on screen.
        sector_counts = identity.sectors_per_enb(active)
        color_options = ["Metric"] + [
            l for l in identity.available_levels(active) if l != "ssb"
        ]
        if not sector_counts.empty:
            color_options.append(_SEC_OPT)
        if show_sat_kpi:
            color_options.append(_SAT_OPT)
        _CONTINUOUS = ("Metric", _SAT_OPT, _SEC_OPT)
        color_choice = st.selectbox(
            "Colour by",
            color_options,
            format_func=lambda c: c if c in _CONTINUOUS else identity.label_for(c),
            help="Colour each point by the metric, categorically by serving PCI / "
                 "eNB-gNB / Sector, by how many distinct sector ids the serving "
                 "eNB carries, or (when enabled) by how many D2C satellites "
                 "were overhead at that instant.",
        )
        if color_choice == _SEC_OPT:
            st.caption(
                f"{len(sector_counts):,} eNBs in the current selection carry "
                f"{int(sector_counts.min())}–{int(sector_counts.max())} distinct "
                f"sector ids (median {sector_counts.median():.0f}). Points whose "
                "eNB or sector never decoded are not drawn."
            )

        # When colouring by an identity there can be dozens of PCIs or eNBs.
        # "Every cell" gives each its own generated colour (no grey dump);
        # "Top N + Other" keeps the busiest N and folds the rest into grey.
        color_all = False
        color_max_slots = 8
        if color_choice not in _CONTINUOUS:
            n_ids = int(active[color_choice].nunique())
            cat_scheme = st.radio(
                "Colour scheme",
                ["Every cell", "Top N + Other"],
                key="map_cat_scheme",
                help="“Every cell” gives all "
                     f"{n_ids} {identity.label_for(color_choice)} their own "
                     "colour so none are lost to grey; the legend scrolls. "
                     "“Top N + Other” keeps only the busiest N.",
            )
            if cat_scheme == "Every cell":
                color_all = True
                st.caption(
                    f"{n_ids:,} distinct {identity.label_for(color_choice)}, "
                    "each its own colour"
                    + (" (capped at 60)." if n_ids > 60 else ".")
                )
            else:
                color_max_slots = st.slider(
                    "Coloured cells",
                    min_value=2,
                    max_value=int(min(40, max(2, n_ids))),
                    value=int(min(8, max(2, n_ids))),
                    key="map_cat_slots",
                    help="The rest fold into a single grey “Other”.",
                )

        apply_region_to_map = st.checkbox("Apply region to map", value=False)
        marker_radius = st.slider("Marker size", 2, 12, 4)
        midpoint = st.number_input(
            "Diverging midpoint",
            value=float(pd.to_numeric(active[metric], errors="coerce").median()),
            help="Values below this are red, above are blue.",
            disabled=map_scale != "diverging" or color_choice != "Metric",
        )
        st.caption(
            "Use the rectangle or polygon tool (top-left of the map) to select a "
            "region. It applies to every tab."
        )

    # Colouring by satellite count is a continuous scale over the sat_vis column,
    # not the signal metric; everything else colours by the metric.
    sat_color = color_choice == _SAT_OPT
    sector_color = color_choice == _SEC_OPT
    map_metric = (
        identity.SECTORS_PER_ENB_COL
        if sector_color
        else (sat_vis_col if sat_color else metric)
    )
    map_metric_label = (
        _SEC_OPT
        if sector_color
        else (f"D2C satellites ≥{int(sat_mask)}°" if sat_color else metric_meta.axis_label)
    )
    map_source = active if apply_region_to_map else unregioned
    if sector_color:
        # Counts come from the filtered set, so an eNB keeps the same colour
        # whether or not the drawn region is applied to the map.
        map_source = identity.with_sectors_per_enb(map_source, sector_counts)
        map_source = map_source[map_source[identity.SECTORS_PER_ENB_COL].notna()]
    fmap, _, _ = mapview.build_map(
        map_source,
        map_metric,
        map_metric_label,
        scale="sequential" if (sat_color or sector_color) else map_scale,
        midpoint=midpoint,
        max_points=int(max_map_points),
        tiles=tiles,
        carto_key=carto_key,
        mode=MODE,
        regions=st.session_state["regions"],
        radius=int(marker_radius),
        color_by=None if color_choice in _CONTINUOUS else color_choice,
        color_by_label="" if color_choice in _CONTINUOUS else identity.label_for(color_choice),
        color_max_slots=int(color_max_slots),
        color_all=bool(color_all),
    )
    with left:
        result = st_folium(
            fmap,
            use_container_width=True,
            height=620,
            returned_objects=["all_drawings"],
            key="qp_map",
        )

    drawn = mapview.regions_from_drawings((result or {}).get("all_drawings"))
    if drawn != st.session_state["regions"]:
        st.session_state["regions"] = drawn
        st.rerun()

    geo_rows = len(flt.geolocated(map_source))
    st.caption(
        f"{geo_rows:,} geolocated samples · showing up to {int(max_map_points):,} "
        f"(evenly thinned). Rows without GPS are never drawn."
    )
    if "provider" in map_source.columns:
        prism_rows = int((map_source["provider"] == "PRISM").sum())
        prism_geo = int((flt.geolocated(map_source)["provider"] == "PRISM").sum())
        if prism_rows and not prism_geo:
            st.info(
                f"{prism_rows:,} PRISM scan rows carry no GPS fix, so they are not "
                "drawn on the map — see the CDF curves, Statistics and Data tabs "
                "to compare them by location and cell."
            )
