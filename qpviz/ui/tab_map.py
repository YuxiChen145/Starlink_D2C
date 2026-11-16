"""Map tab of the Starlink D2C explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qpviz import filters as flt, identity, mapview
from qpviz.ui.context import Ctx


#: fixed display settings (they were sidebar options once)
MAX_MAP_POINTS = 3000
MARKER_RADIUS = 4
#: colouring by eNB / gNB: the busiest this many get a colour, the rest are "Other"
COLOURED_CELLS = 10


def render(ctx: Ctx) -> None:
    """Render the Map tab."""
    MODE = ctx.mode
    active = ctx.active
    metric = ctx.metric
    metric_meta = ctx.metric_meta
    sat_mask = ctx.sat_mask
    sat_vis_col = ctx.sat_vis_col
    show_sat_kpi = ctx.show_sat_kpi
    unregioned = ctx.unregioned

    from streamlit_folium import st_folium

    carto_key = mapview.carto_api_key(st.secrets)
    tiles = mapview.default_tiles(carto_key)

    left, right = st.columns([4, 1])
    with right:
        _SAT_OPT = "D2C satellites visible"
        color_options = ["Metric"]
        if "enb" in identity.available_levels(active):
            color_options.append("enb")
        if show_sat_kpi:
            color_options.append(_SAT_OPT)
        _CONTINUOUS = ("Metric", _SAT_OPT)
        color_choice = st.selectbox(
            "Colour by",
            color_options,
            format_func=lambda c: c if c in _CONTINUOUS else identity.label_for(c),
            help="Colour each point by the metric, by its serving eNB / gNB, or "
                 "(when satellite visibility is on) by how many D2C satellites "
                 "were overhead at that instant.",
        )
        if color_choice == "Metric" and metric_meta.map_range:
            lo, hi = metric_meta.map_range
            st.caption(
                f"The colours always span {lo:g} to {hi:g} {metric_meta.unit}, so "
                "two maps can be compared. Values outside take the end colours."
            )
        elif color_choice == "enb":
            n_ids = int(active["enb"].nunique())
            st.caption(
                f"{n_ids:,} distinct {identity.label_for('enb')} in the selection. "
                + (
                    f"The {COLOURED_CELLS} with the most samples get a colour; "
                    "the rest are grey “Other”."
                    if n_ids > COLOURED_CELLS else "Each has its own colour."
                )
            )

        st.caption(
            "Use the rectangle or polygon tool (top-left of the map) to select a "
            "region. It applies to every tab. Draw more shapes to add to it."
        )
        n_regions = len(st.session_state["regions"])
        if st.button(
            "Clear region", disabled=not n_regions, width="stretch",
            help="Remove the drawn region(s) and show all the data again.",
        ):
            st.session_state["regions"] = []
            # The map component remembers the shapes drawn on it and would hand
            # them straight back; a new key gives a fresh map with none.
            st.session_state["map_epoch"] = st.session_state.get("map_epoch", 0) + 1
            st.rerun()

    # Colouring by satellite count is a continuous scale over the sat_vis column,
    # not the signal metric; everything else colours by the metric.
    sat_color = color_choice == _SAT_OPT
    map_metric = sat_vis_col if sat_color else metric
    map_metric_label = (
        f"D2C satellites ≥{int(sat_mask)}°" if sat_color else metric_meta.axis_label
    )
    # The map always shows every filtered point, with the drawn region outlined
    # on top, so the region can be seen and redrawn in its surroundings.
    map_source = unregioned
    fmap, _, _ = mapview.build_map(
        map_source,
        map_metric,
        map_metric_label,
        color_range=None if sat_color else metric_meta.map_range,
        max_points=MAX_MAP_POINTS,
        tiles=tiles,
        carto_key=carto_key,
        mode=MODE,
        regions=st.session_state["regions"],
        radius=MARKER_RADIUS,
        color_by=None if color_choice in _CONTINUOUS else color_choice,
        color_by_label="" if color_choice in _CONTINUOUS else identity.label_for(color_choice),
        color_max_slots=COLOURED_CELLS,
    )
    with left:
        result = st_folium(
            fmap,
            use_container_width=True,
            height=620,
            returned_objects=["all_drawings"],
            key=f"qp_map_{st.session_state.get('map_epoch', 0)}",
        )

    # A shape drawn on the map is added to the region. The component only ever
    # adds: it re-reports its shapes (or none at all) every time the map is
    # redrawn, so treating its report as the whole truth made the region flip
    # on and off. "Clear region" is the one way to remove it.
    drawn = mapview.regions_from_drawings((result or {}).get("all_drawings"))
    added = mapview.new_regions(drawn, st.session_state["regions"])
    if added:
        st.session_state["regions"] = st.session_state["regions"] + added
        st.rerun()

    geo_rows = len(flt.geolocated(map_source))
    st.caption(
        f"{geo_rows:,} geolocated samples · showing up to {MAX_MAP_POINTS:,} "
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
