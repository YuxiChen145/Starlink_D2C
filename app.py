"""Starlink D2C - Qualipoc measurement visualisation platform.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import streamlit as st

from qpviz import charts, filters as flt, identity, loader, mapview, stats
from qpviz.schema import (
    DATASETS_BY_KEY,
    METRICS,
    TECHNOLOGIES,
    Dataset,
    datasets_for,
    metrics_available,
)

APP_TITLE = "Starlink D2C — Measurement Explorer"

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📡")


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Loading measurement data…")
def load_frame(data_dir: str, dataset_keys: Tuple[str, ...]) -> pd.DataFrame:
    return loader.load(data_dir, list(dataset_keys))


@st.cache_data(show_spinner=False)
def available(data_dir: str) -> List[str]:
    return [d.key for d in loader.available_datasets(data_dir)]


def theme_mode() -> str:
    try:
        return "dark" if st.get_option("theme.base") == "dark" else "light"
    except Exception:  # pragma: no cover
        return "light"


def multiselect_all(label: str, options: Sequence, key: str, help: str = ""):
    """Multiselect where an empty selection means 'everything'."""
    chosen = st.multiselect(
        label,
        options=list(options),
        default=[],
        key=key,
        help=help or "Leave empty to include all.",
        placeholder=f"All ({len(options)})",
    )
    return chosen


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

MODE = theme_mode()
st.session_state.setdefault("regions", [])

with st.sidebar:
    st.markdown(f"### {APP_TITLE}")

    default_dir = os.environ.get("QP_DATA_DIR", str(Path(__file__).parent / "QP CSV"))
    data_dir = st.text_input("Data folder", value=default_dir)
    if st.button("Reload data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    present = available(data_dir)
    if not present:
        st.error(f"No QP CSV files found in `{data_dir}`.")
        st.stop()

    st.divider()
    st.markdown("#### 1 · Data source")

    tech = st.radio(
        "Technology",
        list(TECHNOLOGIES),
        horizontal=True,
        key="tech_choice",
        help="Which radio technology to read. The matching Qualipoc exports are "
             "selected automatically from the metric you pick below.",
    )
    tech_value = TECHNOLOGIES[tech]
    in_tech = [
        k for k in present if tech_value is None or DATASETS_BY_KEY[k].tech == tech_value
    ]
    if not in_tech:
        st.warning(f"No {tech} exports in this folder.")
        st.stop()

    metric_keys = metrics_available(in_tech)
    metric = st.selectbox(
        "Metric",
        options=metric_keys,
        format_func=lambda m: METRICS[m].axis_label,
        index=0,
        key="metric_choice",
    )

    source_keys = [d.key for d in datasets_for(metric, tech_value, available=in_tech)]
    st.caption(
        "Reading "
        + ", ".join(f"**{DATASETS_BY_KEY[k].label}**" for k in source_keys)
        + ". Narrow this with the Direction and Data source filters below."
    )

frame = load_frame(data_dir, tuple(sorted(source_keys)))
frame = frame[frame[metric].notna()] if metric in frame.columns else frame.iloc[0:0]
freq_label = loader.freq_label_for(frame)
freq_prefix = loader.freq_prefix_for(frame)

with st.sidebar:
    st.divider()
    st.markdown("#### 2 · Filters")

    link_options = loader.options(frame, "link")
    links = st.multiselect(
        "Direction",
        options=link_options,
        # a metric that exists in both directions defaults to downlink rather
        # than silently pooling DL and UL into one distribution
        default=["DL"] if set(link_options) == {"DL", "UL"} else [],
        key=f"f_links_{'-'.join(link_options)}",
        help="Leave empty to include all.",
        placeholder=f"All ({len(link_options)})",
    )
    step0 = flt.apply_filters(frame, flt.Filters(links=links))

    source_options = [k for k in source_keys if k in set(step0["source"].dropna())]
    sources = st.multiselect(
        "Data source",
        options=source_options,
        default=[],
        format_func=lambda k: DATASETS_BY_KEY[k].label,
        key="f_sources",
        help="Leave empty to include all of the exports listed above.",
        placeholder=f"All ({len(source_options)})",
    )
    step0 = flt.apply_filters(step0, flt.Filters(sources=sources))

    collections = multiselect_all(
        "Collections", loader.options(step0, "collection"), "f_collections"
    )
    step1 = flt.apply_filters(step0, flt.Filters(collections=collections))

    bands = multiselect_all("Band", loader.options(step1, "band"), "f_bands")
    step2 = flt.apply_filters(step1, flt.Filters(bands=bands))

    freqs = multiselect_all(freq_label, loader.options(step2, "freq"), "f_freqs")
    step3 = flt.apply_filters(step2, flt.Filters(freqs=freqs))

    pci_options = loader.options(step3, "pci")
    pcis = multiselect_all(f"PCI ({len(pci_options)} available)", pci_options, "f_pcis")

    drop_unknown = st.checkbox(
        f"Require a known PCI and {freq_prefix}",
        value=True,
        help=(
            "Some exports (notably LTE Radio Neighbor) leave PCI and the "
            "carrier blank; those rows would otherwise collapse into one "
            "large “n/a” curve."
        ),
    )

    st.markdown("**Region**")
    n_regions = len(st.session_state["regions"])
    cols = st.columns([3, 2])
    cols[0].caption(
        f"{n_regions} shape(s) drawn" if n_regions else "Draw on the map tab to limit by area"
    )
    if cols[1].button("Clear", disabled=not n_regions, width="stretch"):
        st.session_state["regions"] = []
        st.rerun()
    keep_ungeo = st.checkbox(
        "Keep rows without GPS",
        value=False,
        help="LTE PUCCH has no GPS columns; enable to keep such rows when a region is drawn.",
    )

    st.divider()
    with st.expander("Display options"):
        min_samples = st.number_input(
            "Min samples per curve", min_value=1, max_value=100000, value=30, step=10
        )
        max_curve_points = st.number_input(
            "Max points per curve", min_value=100, max_value=100000, value=2000, step=500
        )
        complementary = st.checkbox("Plot CCDF (1 − CDF)", value=False)
        max_map_points = st.number_input(
            "Max points on map", min_value=200, max_value=20000, value=3000, step=500
        )
        map_scale = st.selectbox(
            "Map colour scale", ["diverging", "sequential"], index=0
        )
        tiles = st.selectbox(
            "Basemap",
            list(mapview.TILE_CHOICES),
            index=1,
        )

active = flt.apply_filters(
    frame,
    flt.Filters(
        links=links,
        sources=sources,
        collections=collections,
        bands=bands,
        freqs=freqs,
        pcis=pcis,
        regions=st.session_state["regions"],
        metric=metric,
        require=("pci", "freq") if drop_unknown else (),
        keep_ungeolocated=keep_ungeo,
    ),
)

# Same filters minus the region - used to draw the map you select the region on.
unregioned = flt.apply_filters(
    frame,
    flt.Filters(
        links=links,
        sources=sources,
        collections=collections,
        bands=bands,
        freqs=freqs,
        pcis=pcis,
        metric=metric,
        require=("pci", "freq") if drop_unknown else (),
    ),
)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

metric_meta = METRICS[metric]
head = st.columns([2, 1, 1, 1, 1])
head[0].markdown(f"### {metric_meta.label} · {tech}")
head[1].metric("Samples", f"{len(active):,}")
head[2].metric("Collections", f"{active['collection'].nunique():,}")
head[3].metric("PCIs", f"{active['pci'].nunique():,}")
head[4].metric(freq_label, f"{active['freq'].nunique():,}")

if active.empty:
    st.info("No samples match the current filters.")
    st.stop()

tab_map, tab_cdf, tab_cells, tab_stats, tab_data = st.tabs(
    ["🗺️ Map", "📈 CDF curves", "🛰️ Cell identity", "📊 Statistics", "🧾 Data"]
)


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #

with tab_map:
    from streamlit_folium import st_folium

    left, right = st.columns([4, 1])
    with right:
        color_options = ["Metric"] + [
            l for l in identity.available_levels(active) if l != "ssb"
        ]
        color_choice = st.selectbox(
            "Colour by",
            color_options,
            format_func=lambda c: "Metric" if c == "Metric" else identity.label_for(c),
            help="Colouring by an identity shows how the serving cell changes "
                 "along the route: terrestrial cells hold whole stretches, "
                 "satellite cells interleave.",
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

    map_source = active if apply_region_to_map else unregioned
    fmap, _, _ = mapview.build_map(
        map_source,
        metric,
        metric_meta.axis_label,
        scale=map_scale,
        midpoint=midpoint,
        max_points=int(max_map_points),
        tiles=tiles,
        mode=MODE,
        regions=st.session_state["regions"],
        radius=int(marker_radius),
        color_by=None if color_choice == "Metric" else color_choice,
        color_by_label="" if color_choice == "Metric" else identity.label_for(color_choice),
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


# --------------------------------------------------------------------------- #
# CDF
# --------------------------------------------------------------------------- #

with tab_cdf:
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
    sort_note = controls[2].empty()

    curves = stats.cdf_curves(
        active,
        metric,
        group_by=group_by,
        min_samples=int(min_samples),
        max_points=int(max_curve_points),
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
        st.plotly_chart(
            charts.cdf_figure(
                curves,
                metric_meta.axis_label,
                mode=MODE,
                complementary=complementary,
            ),
            width="stretch",
        )
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


# --------------------------------------------------------------------------- #
# Cell identity
# --------------------------------------------------------------------------- #

with tab_cells:
    # Serving-cell identity is a property of the connection, not of the metric
    # you happen to be plotting - and only the Radio exports carry eNB/gNB-Id
    # and Sector Id. So this tab reads those directly, under the same filters.
    radio_keys = tuple(
        k
        for k in sorted(present)
        if k in ("lte_radio", "nr_radio")
        and (tech_value is None or DATASETS_BY_KEY[k].tech == tech_value)
    )
    if not radio_keys:
        st.info("No Radio [connected] export for this technology, so no cell identity.")
        st.stop()

    radio = load_frame(data_dir, radio_keys)

    def _intersecting(values, column):
        """Only carry a filter across if it means anything in the radio frame."""
        if not values:
            return []
        present_values = set(loader.options(radio, column))
        return [v for v in values if v in present_values]

    cells = flt.apply_filters(
        radio,
        flt.Filters(
            collections=collections,
            bands=_intersecting(bands, "band"),
            freqs=_intersecting(freqs, "freq"),
            pcis=_intersecting(pcis, "pci"),
            regions=st.session_state["regions"],
            keep_ungeolocated=keep_ungeo,
            require=("pci",) if drop_unknown else (),
        ),
    )
    cells = identity.serving_only(cells)
    levels = identity.available_levels(cells)

    if cells.empty or not levels:
        st.info("No serving-cell rows match the current filters.")
    else:
        st.caption(
            "Serving cell (PCell / PSCell) only, from "
            + ", ".join(DATASETS_BY_KEY[k].label for k in radio_keys)
            + f" — {len(cells):,} samples."
        )
        view = st.radio(
            "View", ["Timeline", "Hierarchy"], horizontal=True, key="identity_view"
        )

        if view == "Timeline":
            row = st.columns([2, 2, 2, 2])
            level = row[0].selectbox(
                "Identity", levels, format_func=identity.label_for, key="identity_level"
            )
            order_by = row[1].selectbox(
                "Row order", ["first seen", "dwell", "value"], key="identity_order",
                help="First seen draws the staircase a moving constellation makes; "
                     "dwell puts the cells you spent longest on at the top.",
            )
            max_rows = row[2].number_input(
                "Max rows", min_value=5, max_value=400, value=60, step=5,
                help="Largest total dwell wins when there are more.",
            )
            x_axis = row[3].selectbox(
                "X axis", ["elapsed", "absolute"], key="identity_x",
                help="Elapsed is active minutes within each collection (time "
                     "the log was paused is excluded), so several collections "
                     "share one axis.",
            )

            # One lane per identity per collection: drawing all 28 at once is
            # unreadable and the legend swamps the plot, so focus on a few.
            by_size = (
                cells["collection"].value_counts().index.tolist()
            )
            focus = st.multiselect(
                "Collections to draw",
                options=by_size,
                default=by_size[:1],
                max_selections=6,
                key="identity_focus",
                help="Up to six at a time — each gets its own colour.",
            )
            if not focus:
                st.info("Pick at least one collection to draw.")
                st.stop()

            runs = identity.dwell_runs(cells[cells["collection"].isin(focus)], level)
            if runs.empty:
                st.info("Not enough samples to build dwell runs.")
            else:
                shown = identity.order_identities(runs, order_by, limit=int(max_rows))
                span_min = max(runs["elapsed_end"].max() / 60.0, 1e-9)
                handovers = len(runs) - runs["collection"].nunique()
                summary = st.columns(4)
                summary[0].metric(f"Distinct {identity.label_for(level)}", f"{runs['value'].nunique():,}")
                summary[1].metric("Dwell runs", f"{len(runs):,}")
                summary[2].metric("Median dwell", f"{runs['duration_s'].median():.1f} s")
                summary[3].metric("Changes / min", f"{handovers / span_min:.2f}")

                st.plotly_chart(
                    charts.timeline_figure(
                        runs, shown, identity.label_for(level), mode=MODE, x_axis=x_axis
                    ),
                    width="stretch",
                )
                if len(shown) < runs["value"].nunique():
                    st.caption(
                        f"Showing the {len(shown)} longest-held of "
                        f"{runs['value'].nunique()} identities."
                    )
                with st.expander("Dwell runs as a table"):
                    table = runs[
                        ["collection", "value", "start", "duration_s", "samples"]
                    ].rename(
                        columns={
                            "collection": "Collection",
                            "value": identity.label_for(level),
                            "start": "Start",
                            "duration_s": "Held (s)",
                            "samples": "Samples",
                        }
                    )
                    st.dataframe(table, width="stretch", hide_index=True, height=320)
                    st.download_button(
                        "Download dwell runs (CSV)",
                        table.to_csv(index=False).encode(),
                        file_name=f"{level}_dwell_runs.csv",
                        mime="text/csv",
                    )
        else:
            default_levels = [l for l in ("enb", "sector", "pci") if l in levels]
            chosen = st.multiselect(
                "Levels, outer → inner",
                options=levels,
                default=default_levels or levels[:1],
                format_func=identity.label_for,
                key="identity_hierarchy",
                help="Rectangles are sized by how long that cell served.",
            )
            if not chosen:
                st.info("Pick at least one level.")
            else:
                table = identity.hierarchy(cells, chosen)
                if table.empty:
                    st.info(
                        "No rows carry every level in that path — try dropping "
                        "a level."
                    )
                else:
                    counts = " · ".join(
                        f"{table[l].nunique():,} {identity.label_for(l)}" for l in chosen
                    )
                    st.caption(f"{counts} · {table['dwell_s'].sum() / 60:.1f} minutes held in total.")
                    st.plotly_chart(
                        charts.treemap_figure(
                            table, chosen,
                            [identity.label_for(l) for l in chosen],
                            mode=MODE,
                        ),
                        width="stretch",
                    )
                    with st.expander("Hierarchy as a table"):
                        pretty = table.rename(
                            columns={l: identity.label_for(l) for l in chosen}
                        ).rename(columns={"dwell_s": "Held (s)", "samples": "Samples"})
                        st.dataframe(pretty.round(1), width="stretch", hide_index=True, height=320)
                        st.download_button(
                            "Download hierarchy (CSV)",
                            pretty.to_csv(index=False).encode(),
                            file_name="cell_hierarchy.csv",
                            mime="text/csv",
                        )


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

with tab_stats:
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


# --------------------------------------------------------------------------- #
# Raw data
# --------------------------------------------------------------------------- #

with tab_data:
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
    show_cols = [c for c in show_cols if c in active.columns]
    st.dataframe(active[show_cols].head(5000), width="stretch", height=520)
    st.caption(f"Showing the first 5,000 of {len(active):,} filtered rows.")
    st.download_button(
        "Download filtered rows (CSV)",
        active[show_cols].to_csv(index=False).encode(),
        file_name=f"{metric}_filtered.csv",
        mime="text/csv",
    )
