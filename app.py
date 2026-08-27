"""Starlink D2C - Qualipoc measurement visualisation platform.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import streamlit as st

from qpviz import charts, filters as flt, loader, mapview, stats
from qpviz.schema import (
    DATASETS_BY_KEY,
    METRICS,
    Dataset,
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
        "Technology", ["All", "LTE", "NR"], horizontal=True, key="tech_choice"
    )
    candidates = [
        DATASETS_BY_KEY[k]
        for k in present
        if tech == "All" or DATASETS_BY_KEY[k].tech == tech
    ]
    default_sources = [
        d.key for d in candidates if d.key in ("lte_radio", "nr_radio")
    ] or [candidates[0].key]

    source_keys = st.multiselect(
        "Datasets",
        options=[d.key for d in candidates],
        default=default_sources,
        format_func=lambda k: DATASETS_BY_KEY[k].label,
        help="Each Qualipoc export carries a different set of metrics.",
    )
    if not source_keys:
        st.warning("Select at least one dataset.")
        st.stop()

    metric_keys = metrics_available(source_keys)
    if not metric_keys:
        st.warning("The selected datasets share no plottable metric.")
        st.stop()

    metric = st.selectbox(
        "Metric",
        options=metric_keys,
        format_func=lambda m: METRICS[m].axis_label,
        index=0,
        key="metric_choice",
    )

frame = load_frame(data_dir, tuple(sorted(source_keys)))
frame = frame[frame[metric].notna()] if metric in frame.columns else frame.iloc[0:0]
freq_label = loader.freq_label_for(frame)
freq_prefix = loader.freq_prefix_for(frame)

with st.sidebar:
    st.divider()
    st.markdown("#### 2 · Filters")

    collections = multiselect_all(
        "Collections", loader.options(frame, "collection"), "f_collections"
    )
    step1 = flt.apply_filters(frame, flt.Filters(collections=collections))

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
head[0].markdown(f"### {metric_meta.label} — {len(source_keys)} dataset(s)")
head[1].metric("Samples", f"{len(active):,}")
head[2].metric("Collections", f"{active['collection'].nunique():,}")
head[3].metric("PCIs", f"{active['pci'].nunique():,}")
head[4].metric(freq_label, f"{active['freq'].nunique():,}")

if active.empty:
    st.info("No samples match the current filters.")
    st.stop()

tab_map, tab_cdf, tab_stats, tab_data = st.tabs(
    ["🗺️ Map", "📈 CDF curves", "📊 Statistics", "🧾 Data"]
)


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #

with tab_map:
    from streamlit_folium import st_folium

    left, right = st.columns([4, 1])
    with right:
        apply_region_to_map = st.checkbox("Apply region to map", value=False)
        marker_radius = st.slider("Marker size", 2, 12, 4)
        midpoint = st.number_input(
            "Diverging midpoint",
            value=float(pd.to_numeric(active[metric], errors="coerce").median()),
            help="Values below this are red, above are blue.",
            disabled=map_scale != "diverging",
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
