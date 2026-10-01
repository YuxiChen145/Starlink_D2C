"""Starlink D2C - Qualipoc measurement visualisation platform.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import re
from datetime import timedelta
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import streamlit as st

from qpviz import (
    charts,
    correlate,
    filters as flt,
    identity,
    loader,
    mapview,
    satellites,
    stats,
    timeline,
)
from qpviz.schema import (
    ALL_DATASETS_BY_KEY,
    DATASETS_BY_KEY,
    METRICS,
    PRISM_DATASETS_BY_KEY,
    SIGCAP_DATASETS_BY_KEY,
    TECHNOLOGIES,
    TRAFFIC_ORDER,
    Dataset,
    datasets_for,
    metrics_available,
    prism_datasets_for,
    sigcap_datasets_for,
)

APP_TITLE = "Starlink D2C — Measurement Explorer"

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📡")

# Collection names run to 60 characters, which the default sidebar truncates in
# both the dropdown and the selected chips. Widen it a little and let long
# labels wrap rather than clip. These selectors reach into Streamlit/BaseWeb
# internals; if a future version renames them the app simply falls back to the
# old truncation, so nothing breaks.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"], div[data-testid="stSidebar"] {
        width: 24rem !important;
        min-width: 24rem !important;
    }
    /* Collection names run to 60 characters, which the sidebar cannot show.
       The dropdown is portalled to <body>, so let it spill past the sidebar
       and give every option one full line. */
    div[role="listbox"] {
        min-width: 30rem !important;
        max-width: none !important;
        overflow-x: visible !important;
    }
    /* Its positioning wrapper has to grow too, or it clips the list. Kept in
       its own rule: an engine without :has() would otherwise drop the lot. */
    div:has(> div[role="listbox"]),
    div:has(> div > div[role="listbox"]) {
        min-width: 30rem !important;
        max-width: none !important;
        width: auto !important;
        overflow-x: visible !important;
    }
    /* Fallback for anything longer still: the clipping is on the div *inside*
       each option, which ships with white-space:nowrap and overflow:hidden. */
    div[role="option"], div[role="option"] > div,
    div[data-baseweb="popover"] li, div[data-baseweb="popover"] li > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        height: auto !important;
        max-height: none !important;
        line-height: 1.35;
        word-break: break-word;
    }
    /* Selected chips live inside the sidebar and cannot spill, so they wrap.
       (data-baseweb is the older Streamlit build; role=group the current one.) */
    [data-testid="stMultiSelect"] span[role="group"],
    [data-testid="stMultiSelect"] span[role="group"] span,
    [data-testid="stMultiSelect"] [data-baseweb="tag"],
    [data-testid="stMultiSelect"] [data-baseweb="tag"] * {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        height: auto !important;
        max-width: 100% !important;
        word-break: break-word;
    }
    [data-testid="stMultiSelect"] span[role="group"] {
        padding-top: 1px;
        padding-bottom: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached data access
# --------------------------------------------------------------------------- #


@st.cache_data(show_spinner="Loading measurement data…")
def load_frame(data_dir: str, dataset_keys: Tuple[str, ...]) -> pd.DataFrame:
    return loader.load(data_dir, list(dataset_keys))


@st.cache_data(show_spinner="Loading PRISM data…")
def load_prism_frame(prism_dir: str, dataset_keys: Tuple[str, ...]) -> pd.DataFrame:
    return loader.load_prism(prism_dir, list(dataset_keys))


@st.cache_data(show_spinner="Loading SigCap data…")
def load_sigcap_frame(sigcap_dir: str, dataset_keys: Tuple[str, ...]) -> pd.DataFrame:
    return loader.load_sigcap(sigcap_dir, list(dataset_keys))


@st.cache_data(show_spinner=False)
def sigcap_traffic_ref(sigcap_dir: str, dataset_keys: Tuple[str, ...]) -> pd.DataFrame:
    """time + traffic from SigCap - the ground truth for labelling QP D2C rows,
    which are QP-idle PHY captures of SigCap-generated traffic."""
    sc = loader.load_sigcap(sigcap_dir, list(dataset_keys))
    if sc.empty or "traffic" not in sc.columns:
        return sc.iloc[0:0][["time"]].assign(traffic=[])
    keep = sc["traffic"].isin(["iperf DL", "iperf UL", "HTTP", "Ping"])
    return sc.loc[keep, ["time", "traffic"]].reset_index(drop=True)


def _align_qp_d2c(fr: pd.DataFrame) -> pd.DataFrame:
    """Relabel QP D2C traffic from the SigCap reference, when SigCap is present."""
    if not sigcap_present:
        return fr
    ref = sigcap_traffic_ref(sigcap_dir, tuple(sorted(sigcap_present)))
    return loader.align_d2c_traffic(fr, ref)


@st.cache_data(show_spinner=False)
def available(data_dir: str) -> List[str]:
    return [d.key for d in loader.available_datasets(data_dir)]


@st.cache_data(show_spinner=False)
def prism_present_keys(prism_dir: str) -> List[str]:
    return loader.prism_available(prism_dir)


@st.cache_data(show_spinner=False)
def sigcap_present_keys(sigcap_dir: str) -> List[str]:
    return loader.sigcap_available(sigcap_dir)


@st.cache_data(show_spinner=False)
def tle_summary(tle_dir: str, tle_fp: str) -> dict:
    return satellites.archive_summary(satellites.load_tle_dir(tle_dir))


@st.cache_data(show_spinner="Propagating D2C satellites (SGP4)…")
def visibility_columns(
    _frame: pd.DataFrame,
    cache_key: Tuple,
    tle_dir: str,
    tle_fp: str,
    offset_hours: float,
    masks: Tuple[int, ...],
) -> pd.DataFrame:
    """Per-sample D2C visibility columns for ``_frame`` (heavy → cached).

    ``_frame`` is not part of the cache key (leading underscore); ``cache_key``
    carries the identifiers that actually determine its rows, plus the TLE
    fingerprint and UTC offset, so the propagation reruns only when those change.
    """
    sats = satellites.filter_dtc(satellites.load_tle_dir(tle_dir))
    annotated = satellites.annotate_frame(
        _frame.reset_index(drop=True),
        sats,
        masks=list(masks),
        utc_offset_hours=offset_hours,
    )
    cols = [f"sat_vis_{m}" for m in masks] + ["sat_max_elev"]
    return annotated[cols].reset_index(drop=True)


@st.cache_data(show_spinner="Counting distinct satellites…")
def distinct_sats(
    _sel: pd.DataFrame, cache_key: Tuple, tle_dir: str, tle_fp: str,
    mask: int, offset_hours: float,
) -> int:
    """Distinct D2C satellites ever above ``mask`` across the current selection.

    Unlike the per-sample count (how many are up at an instant), this counts the
    different satellites that passed through the whole window. Cached on a
    fingerprint of the selection so it only recomputes when the filter changes.
    """
    sats = satellites.filter_dtc(satellites.load_tle_dir(tle_dir))
    return satellites.distinct_visible_count(
        _sel, sats, float(mask), utc_offset_hours=float(offset_hours)
    )


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

    default_qp = os.environ.get("QP_DATA_DIR", str(Path(__file__).parent / "QP CSV"))
    default_prism = os.environ.get(
        "PRISM_DATA_DIR", str(Path(__file__).parent / "PRISM CSV")
    )
    default_sigcap = os.environ.get(
        "SIGCAP_DATA_DIR", str(Path(__file__).parent / "Sigcap CSV")
    )
    qp_dir = st.text_input("QP data folder", value=default_qp)
    prism_dir = st.text_input(
        "PRISM data folder",
        value=default_prism,
        help="PRISM SDR scan exports. Leave as-is if you have none — the app "
             "simply reads whichever sources are present.",
    )
    sigcap_dir = st.text_input(
        "SigCap data folder",
        value=default_sigcap,
        help="SigCap session logs (one file per iperf/HTTP run). Leave as-is if "
             "you have none — the app reads whichever sources are present.",
    )
    if st.button("Reload data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    qp_present = available(qp_dir)
    prism_present = prism_present_keys(prism_dir)
    sigcap_present = sigcap_present_keys(sigcap_dir)
    providers_present = (
        (["QP"] if qp_present else [])
        + (["PRISM"] if prism_present else [])
        + (["SIGCAP"] if sigcap_present else [])
    )
    if not providers_present:
        st.error(
            f"No QP CSV files in `{qp_dir}`, no PRISM CSV files in `{prism_dir}`, "
            f"and no SigCap CSV files in `{sigcap_dir}`."
        )
        st.stop()

    st.divider()
    st.markdown("#### 1 · Data source")

    # QP (Qualipoc phone) vs PRISM (SDR scanner) are genuinely different
    # measurements, so they get their own top-level filter. Empty means pool all
    # present sources; the same idiom as every other filter below.
    if len(providers_present) > 1:
        provider_sel = st.multiselect(
            "Measurement source",
            options=providers_present,
            default=[],
            key="f_providers",
            help="QP is the Qualipoc phone (connected-mode logs, one serving "
                 "cell); PRISM is the SDR scanner (every audible cell, signal "
                 "only); SigCap runs on the same phone and adds app-layer "
                 "iperf/HTTP throughput and ping RTT. Leave empty to pool all; "
                 "split CDF/Statistics by “Measurement source” to compare them.",
            placeholder=f"All ({len(providers_present)})",
        )
    else:
        provider_sel = []
        st.caption(f"Measurement source: **{providers_present[0]}**")
    providers_active = provider_sel or providers_present

    tech = st.radio(
        "Technology",
        list(TECHNOLOGIES),
        horizontal=True,
        key="tech_choice",
        help="Which radio technology to read. The matching exports are selected "
             "automatically from the metric you pick below.",
    )
    tech_value = TECHNOLOGIES[tech]

    qp_in_tech = (
        [k for k in qp_present if tech_value is None or DATASETS_BY_KEY[k].tech == tech_value]
        if "QP" in providers_active
        else []
    )
    prism_in_tech = (
        [
            k
            for k in prism_present
            if tech_value is None or PRISM_DATASETS_BY_KEY[k].tech == tech_value
        ]
        if "PRISM" in providers_active
        else []
    )
    sigcap_in_tech = (
        [
            k
            for k in sigcap_present
            if tech_value is None or SIGCAP_DATASETS_BY_KEY[k].tech == tech_value
        ]
        if "SIGCAP" in providers_active
        else []
    )
    if not qp_in_tech and not prism_in_tech and not sigcap_in_tech:
        st.warning(f"No {tech} exports for the selected measurement source(s).")
        st.stop()

    metric_keys = metrics_available(qp_in_tech + prism_in_tech + sigcap_in_tech)
    if not metric_keys:
        st.warning("No offered metric is available for this selection.")
        st.stop()
    metric = st.selectbox(
        "Metric",
        options=metric_keys,
        format_func=lambda m: METRICS[m].axis_label,
        index=0,
        key="metric_choice",
    )

    qp_source_keys = [d.key for d in datasets_for(metric, tech_value, available=qp_in_tech)]
    prism_source_keys = [
        d.key for d in prism_datasets_for(metric, tech_value, available=prism_in_tech)
    ]
    sigcap_source_keys = [
        d.key for d in sigcap_datasets_for(metric, tech_value, available=sigcap_in_tech)
    ]
    reading = qp_source_keys + prism_source_keys + sigcap_source_keys
    st.caption(
        "Reading "
        + ", ".join(f"**{ALL_DATASETS_BY_KEY[k].label}**" for k in reading)
        + ". Narrow this with the Direction and Data source filters below."
    )

_parts = []
if qp_source_keys:
    _parts.append(load_frame(qp_dir, tuple(sorted(qp_source_keys))))
if prism_source_keys:
    _parts.append(load_prism_frame(prism_dir, tuple(sorted(prism_source_keys))))
if sigcap_source_keys:
    _parts.append(load_sigcap_frame(sigcap_dir, tuple(sorted(sigcap_source_keys))))
frame = loader.concat_frames(_parts)
frame = frame[frame[metric].notna()] if metric in frame.columns else frame.iloc[0:0]
# D2C QP rows are QP-idle PHY captures of SigCap-generated traffic -> label their
# traffic from the time-aligned SigCap ground truth, not QP's own columns/name
frame = _align_qp_d2c(frame)
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

    _traffic_present = loader.options(step0, "traffic")
    traffic_options = (
        [t for t in TRAFFIC_ORDER if t in _traffic_present]
        + [t for t in _traffic_present if t not in TRAFFIC_ORDER]
    )
    traffic = st.multiselect(
        "Traffic",
        options=traffic_options,
        default=[],
        key="f_traffic",
        help="Traffic type read from the session name (iperf DL/UL, Ookla, Ping, "
             "HTTP, Idle). 'Mixed' = several traffic types in one session; a bare "
             "'iperf' has no direction in its name (use Direction). Empty = all.",
        placeholder=f"All ({len(traffic_options)})",
    )
    step0 = flt.apply_filters(step0, flt.Filters(traffic=traffic))

    source_options = [k for k in reading if k in set(step0["source"].dropna())]
    sources = st.multiselect(
        "Data source",
        options=source_options,
        default=[],
        format_func=lambda k: ALL_DATASETS_BY_KEY[k].label,
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

    # Two ways to look at PCIs across several collections: everything that
    # turned up anywhere, or only the cells common to all of them - the set you
    # can compare like for like.
    union_pcis = loader.options(step3, "pci")
    shared_pcis = flt.shared_values(step3, "pci")
    n_collections = int(step3["collection"].nunique()) if not step3.empty else 0

    pci_mode = st.radio(
        "PCI set",
        ["Any collection", "Shared by all"],
        horizontal=True,
        key="f_pci_mode",
        help="“Any collection” lists every PCI seen in the selected "
             "collections. “Shared by all” lists only those present in every "
             "one of them, and restricts the data to that set even with "
             "nothing picked below.",
    )
    shared_mode = pci_mode == "Shared by all"
    pci_options = shared_pcis if shared_mode else union_pcis

    pcis = multiselect_all("PCI", pci_options, "f_pcis")
    st.caption(
        f"{len(union_pcis)} PCI across {n_collections} collection"
        f"{'' if n_collections == 1 else 's'} · {len(shared_pcis)} shared by all"
    )

    if shared_mode and not shared_pcis:
        st.warning("No PCI appears in every selected collection.")
    # PCIs are non-negative, so this matches nothing: "shared by all" with an
    # empty shared set has to show no data, not all of it.
    _NO_PCI = [-1]
    effective_pcis = pcis or (shared_pcis or _NO_PCI if shared_mode else [])

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

    # ------------------------------------------------------------------- #
    # Time window
    # ------------------------------------------------------------------- #
    # Data spans many days, so a fixed slider over the whole span is useless
    # for picking a few minutes. Instead the window is bounded by the data that
    # survives every *other* filter (collection, band, PCI, region) — draw a
    # spot or pick a collection and the slider auto-scales to that session, so a
    # 10-minute window is easy to select.
    st.markdown("**Time window**")
    _time_scope = flt.apply_filters(
        frame,
        flt.Filters(
            links=links, sources=sources, collections=collections, traffic=traffic,
            bands=bands, freqs=freqs, pcis=effective_pcis,
            regions=st.session_state["regions"],
            require=("pci", "freq") if drop_unknown else (),
            keep_ungeolocated=keep_ungeo,
        ),
    )
    _tt = pd.to_datetime(_time_scope["time"], errors="coerce").dropna()
    time_range = None
    _windows = flt.data_windows(_tt, gap_seconds=300.0)
    if _tt.empty or _tt.min() == _tt.max():
        st.caption("Not enough time span in the current selection to window.")
    else:
        tmin = _tt.min().to_pydatetime()
        tmax = _tt.max().to_pydatetime()
        # actual data coverage (sum of session lengths), not the empty raw span
        data_min = sum(
            (e - s).total_seconds() for s, e, _ in _windows
        ) / 60.0
        # A readable list of the sessions that actually hold data, so you can put
        # the window on one instead of a dead gap between days.
        def _win_label(s, e, n):
            same_day = s.date() == e.date()
            tail = f"{e:%H:%M}" if same_day else f"{e:%m/%d %H:%M}"
            return f"{s:%m/%d %H:%M}–{tail} ({(e - s).total_seconds() / 60:.0f}m)"

        _labels = [_win_label(s, e, n) for s, e, n in _windows]
        shown = " · ".join(_labels[:6]) + (
            f"  +{len(_labels) - 6} more" if len(_labels) > 6 else ""
        )
        st.caption(
            f"**{len(_windows)} data window(s), {data_min:.0f} min total** "
            f"across {(tmax - tmin).total_seconds() / 86400:.1f} days:  {shown}"
        )

        use_time = st.checkbox(
            "Limit to a time window",
            value=False,
            key="use_time_window",
            help="Count what you saw at a spot over a few minutes — pair it with "
                 "a region drawn on the map.",
        )
        if use_time:
            # Jump straight to a session so you never land in an empty gap.
            if len(_labels) > 1:
                jump = st.selectbox(
                    "Jump to a data window",
                    ["(manual)"] + _labels,
                    key="time_jump",
                    help="Sets the window start to that session's first sample.",
                )
                if jump != "(manual)" and st.session_state.get("_time_jump_prev") != jump:
                    st.session_state["time_start"] = _windows[_labels.index(jump)][0].to_pydatetime()
                    st.session_state["_time_jump_prev"] = jump
                    st.rerun()
                if jump == "(manual)":
                    st.session_state["_time_jump_prev"] = "(manual)"
            # Keep the stored start inside the current bounds (they change when
            # you pick a collection or draw a region).
            if "time_start" not in st.session_state or not (
                tmin <= st.session_state["time_start"] <= tmax
            ):
                st.session_state["time_start"] = tmin
            start = st.slider(
                "Window start",
                min_value=tmin, max_value=tmax,
                step=timedelta(minutes=1), format="MM/DD HH:mm", key="time_start",
            )
            dur = st.number_input(
                "Window length (minutes)",
                min_value=1, max_value=int(max(2, data_min)) + 1,
                value=int(min(10, max(1, data_min))), step=1, key="time_len",
            )
            end = min(start + timedelta(minutes=int(dur)), tmax)
            time_range = (pd.Timestamp(start), pd.Timestamp(end))
            _in_win = int(((_tt >= time_range[0]) & (_tt <= time_range[1])).sum())
            st.caption(
                f"{start:%m/%d %H:%M} → {end:%m/%d %H:%M} · "
                f"**{_in_win:,} samples** in window"
                + ("" if _in_win else " — this window is empty; pick a data window above.")
            )

    # When the time window changes, the satellite estimate (computed for the
    # previous window) is stale, so clear the "Estimate visible D2C satellites"
    # toggle rather than show misleading numbers. The sat checkbox is rendered
    # below, so unsetting its state here makes it come up unchecked; a hint there
    # invites re-enabling to recompute for the new window.
    _wr = (str(time_range[0]), str(time_range[1])) if time_range else None
    if "_last_time_range" in st.session_state and st.session_state["_last_time_range"] != _wr:
        if st.session_state.get("sat_enable"):
            st.session_state["sat_enable"] = False
            st.session_state["_sat_window_dropped"] = True
    st.session_state["_last_time_range"] = _wr

    st.divider()
    with st.expander("Display options"):
        min_samples = st.number_input(
            "Min samples per curve", min_value=1, max_value=100000, value=30, step=10
        )
        max_curve_points = st.number_input(
            "Max points per curve", min_value=100, max_value=100000, value=2000, step=500
        )
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

    st.divider()
    with st.expander("🛰️ Satellite visibility (D2C)"):
        sat_enable = st.checkbox(
            "Estimate visible D2C satellites",
            value=False,
            key="sat_enable",
            help="Propagate Starlink Direct-to-Cell TLEs with SGP4 and count how "
                 "many were above the elevation mask at each QP sample's time and "
                 "location. Needs TLE files (see below); QP GPS rows only.",
        )
        if sat_enable:
            st.session_state["_sat_window_dropped"] = False
        elif st.session_state.get("_sat_window_dropped"):
            st.info("Time window changed — tick above to recompute for the new window.")
        default_tle = os.environ.get(
            "TLE_DATA_DIR", str(Path(__file__).parent / "TLE")
        )
        tle_dir = st.text_input("TLE folder", value=default_tle)
        tle_fp = satellites.tle_fingerprint(tle_dir)
        summary = tle_summary(tle_dir, tle_fp)
        if summary["objects"]:
            emin, emax = summary["epoch_min"], summary["epoch_max"]
            span = (
                f"{emin:%Y-%m-%d} → {emax:%Y-%m-%d}" if emin and emax else "—"
            )
            st.caption(
                f"**{summary['dtc_unique']}** D2C satellites "
                f"({summary['dtc']} TLEs) of {summary['objects']} objects · "
                f"epochs {span}"
            )
        else:
            st.caption(
                "No TLE files found. Drop `*.tle` snapshots into this folder "
                "(D2C sats carry a `[DTC]` name tag), or fetch below."
            )
        sat_mask = st.select_slider(
            "Elevation mask (°)",
            options=list(satellites.DEFAULT_MASKS),
            value=25,
            help="A satellite counts as visible at or above this elevation angle. "
                 "25° is typical for direct-to-cell.",
        )
        sat_offset = st.number_input(
            "Measurement clock → UTC offset (hours)",
            value=0.0, step=1.0, format="%.1f",
            help="Set to your logs' UTC offset so satellite geometry lines up "
                 "with real time. e.g. −4 if the timestamps are US/Eastern (EDT). "
                 "0 means the timestamps are already UTC.",
        )
        if st.button("Fetch today's Starlink TLEs", width="stretch"):
            try:
                path = satellites.fetch_celestrak(tle_dir)
                st.success(f"Fetched → {Path(path).name}")
                tle_summary.clear()
                st.rerun()
            except Exception as exc:  # network is often unavailable in-session
                st.warning(f"Could not fetch: {exc}")

# --------------------------------------------------------------------------- #
# Satellite visibility annotation (before filtering, so every tab inherits it)
# --------------------------------------------------------------------------- #

sat_columns = [f"sat_vis_{m}" for m in satellites.DEFAULT_MASKS] + ["sat_max_elev"]
sat_missing_dep = bool(sat_enable) and not satellites.HAVE_SGP4
if sat_missing_dep:
    st.warning(
        "Satellite visibility needs the **sgp4** package, which isn't installed "
        "in this environment. Install it and restart:  `pip install sgp4`  "
        "(or `pip install -r requirements.txt`)."
    )
sat_active = (
    bool(sat_enable)
    and satellites.HAVE_SGP4
    and tle_summary(tle_dir, tle_fp)["dtc"] > 0
)
sat_vis_col = f"sat_vis_{int(sat_mask)}" if sat_enable else None
if sat_active and not frame.empty:
    frame = frame.reset_index(drop=True)
    _vis_key = (
        qp_dir, tuple(sorted(qp_source_keys)), prism_dir,
        tuple(sorted(prism_source_keys)), metric, float(sat_offset), tle_fp,
    )
    _vis = visibility_columns(
        frame, _vis_key, tle_dir, tle_fp, float(sat_offset),
        tuple(satellites.DEFAULT_MASKS),
    )
    for c in sat_columns:
        frame[c] = _vis[c].to_numpy()

active = flt.apply_filters(
    frame,
    flt.Filters(
        links=links,
        sources=sources,
        collections=collections,
        traffic=traffic,
        bands=bands,
        freqs=freqs,
        pcis=effective_pcis,
        regions=st.session_state["regions"],
        time_range=time_range,
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
        traffic=traffic,
        bands=bands,
        freqs=freqs,
        pcis=effective_pcis,
        time_range=time_range,
        metric=metric,
        require=("pci", "freq") if drop_unknown else (),
    ),
)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #

metric_meta = METRICS[metric]
show_sat_kpi = sat_active and sat_vis_col in active.columns and active[sat_vis_col].notna().any()
show_enb_kpi = "enb" in active.columns and active["enb"].notna().any()

# Distinct D2C satellites across the whole selection — shown in the header so it
# is visible on every tab. It is a propagation, so only compute it when the
# selection is a window or otherwise small; on the full multi-day set it is both
# slow and not meaningful (it would approach the whole catalogue).
hdr_distinct = None
if show_sat_kpi:
    _hgeo = active[active[sat_vis_col].notna()]
    _windowed = time_range is not None
    if _windowed or len(_hgeo) <= 6000:
        _hlat = pd.to_numeric(_hgeo["lat"], errors="coerce")
        _hlon = pd.to_numeric(_hgeo["lon"], errors="coerce")
        _ht = pd.to_datetime(_hgeo["time"], errors="coerce")
        _hk = (
            len(_hgeo), str(_ht.min()), str(_ht.max()),
            round(float(_hlat.sum()), 2), round(float(_hlon.sum()), 2),
            int(sat_mask), tle_fp, float(sat_offset),
        )
        hdr_distinct = distinct_sats(_hgeo, _hk, tle_dir, tle_fp, int(sat_mask), float(sat_offset))

widths = (
    [2, 1, 1, 1, 1]
    + ([1] if show_enb_kpi else [])
    + ([1] if show_sat_kpi else [])
)
head = st.columns(widths)
head[0].markdown(f"### {metric_meta.label} · {tech}")
head[1].metric("Samples", f"{len(active):,}")
head[2].metric("Collections", f"{active['collection'].nunique():,}")
head[3].metric("PCIs", f"{active['pci'].nunique():,}")
head[4].metric(freq_label, f"{active['freq'].nunique():,}")
_next = 5
if show_enb_kpi:
    head[_next].metric("eNB / gNB", f"{active['enb'].nunique():,}")
    _next += 1
if show_sat_kpi:
    head[_next].metric(
        f"Distinct sats ≥{int(sat_mask)}°",
        f"{hdr_distinct:,}" if hdr_distinct is not None else "—",
        help="Distinct D2C satellites that passed through the whole selection "
             "(grows with the window — compare against the eNB count). "
             "Shown for a time window or a small selection; set a Time window in "
             "the sidebar to compute it on the full data." if hdr_distinct is None
             else "Distinct D2C satellites that passed through the whole selection "
                  "(grows with the window — compare against the eNB count).",
    )

if active.empty:
    st.info("No samples match the current filters.")
    st.stop()

# Correlation needs both a QP source (PHY layer) and a SigCap source (app layer)
corr_enable = bool(qp_present and sigcap_present)
# Timeline analysis is a SigCap feature (per-phone device identity + app tput)
timeline_enable = bool(sigcap_present)

tab_labels = ["🗺️ Map", "📈 CDF curves", "🛰️ Cell identity", "📊 Statistics", "🧾 Data"]
if sat_enable:
    tab_labels.insert(1, "🛰️ D2C visibility")
if timeline_enable:
    tab_labels.append("⏱️ Timeline analysis")
if corr_enable:
    tab_labels.append("🔗 Correlation")
_tabs = st.tabs(tab_labels)
# map tabs by label so the optional tabs never throw the positional unpacking off
_tab = dict(zip(tab_labels, _tabs))
tab_map = _tab["🗺️ Map"]
tab_cdf = _tab["📈 CDF curves"]
tab_cells = _tab["🛰️ Cell identity"]
tab_stats = _tab["📊 Statistics"]
tab_data = _tab["🧾 Data"]
tab_sat = _tab.get("🛰️ D2C visibility")
tab_corr = _tab.get("🔗 Correlation")
tab_timeline = _tab.get("⏱️ Timeline analysis")


# --------------------------------------------------------------------------- #
# Map
# --------------------------------------------------------------------------- #

with tab_map:
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


# --------------------------------------------------------------------------- #
# D2C satellite visibility
# --------------------------------------------------------------------------- #

if tab_sat is not None:
    with tab_sat:
        if not sat_active:
            st.info(
                "No Direct-to-Cell TLEs loaded. Put dated Starlink TLE files "
                "(objects tagged `[DTC]`) in the **TLE folder** set in the sidebar "
                "— historical snapshots for the campaign dates, refreshed daily "
                "going forward — then this tab estimates how many D2C satellites "
                "were overhead at each QP sample."
            )
        else:
            vis = active[active[sat_vis_col].notna()].copy()
            if vis.empty:
                st.info(
                    "No geolocated QP samples in the current filter to compute "
                    "visibility for (PRISM rows have no GPS and are excluded)."
                )
            else:
                counts = pd.to_numeric(vis[sat_vis_col], errors="coerce")
                elev = pd.to_numeric(vis["sat_max_elev"], errors="coerce")
                summ = tle_summary(tle_dir, tle_fp)
                emin, emax = summ["epoch_min"], summ["epoch_max"]
                dmin = pd.to_datetime(vis["time"]).min()
                dmax = pd.to_datetime(vis["time"]).max()

                # Distinct satellites that passed through the *whole* selection —
                # this grows with the window, unlike the instantaneous count.
                _vlat = pd.to_numeric(vis["lat"], errors="coerce")
                _vlon = pd.to_numeric(vis["lon"], errors="coerce")
                _dk = (
                    len(vis), str(dmin), str(dmax),
                    round(float(_vlat.sum()), 2), round(float(_vlon.sum()), 2),
                    int(sat_mask), tle_fp, float(sat_offset),
                )
                n_distinct = distinct_sats(
                    vis, _dk, tle_dir, tle_fp, int(sat_mask), float(sat_offset)
                )
                win_min = (dmax - dmin).total_seconds() / 60.0 if pd.notna(dmin) else 0.0

                k = st.columns(6)
                k[0].metric(f"Distinct in window ≥{int(sat_mask)}°", f"{n_distinct:,}")
                k[1].metric("Median at once", f"{counts.median():.0f}")
                k[2].metric("Max at once", f"{int(counts.max())}")
                k[3].metric("Mean", f"{counts.mean():.1f}")
                k[4].metric("Samples with ≥1", f"{100*(counts>=1).mean():.0f}%")
                k[5].metric("D2C sats in TLEs", f"{summ['dtc_unique']}")

                st.markdown(
                    f"**{n_distinct} distinct D2C satellites** rose above "
                    f"{int(sat_mask)}° across the selection "
                    f"({win_min:.0f} min, {len(vis):,} samples) — versus a peak of "
                    f"**{int(counts.max())}** overhead at any one instant."
                )
                if emin and emax:
                    st.caption(
                        f"TLE epochs span {emin:%Y-%m-%d}→{emax:%Y-%m-%d}; data here "
                        f"spans {dmin:%Y-%m-%d}→{dmax:%Y-%m-%d}. Each sample is "
                        "propagated from the nearest-epoch TLE — the closer the "
                        "epoch, the more accurate the count. Mask and UTC offset "
                        "are set in the sidebar."
                    )

                view = st.radio(
                    "View",
                    ["Cells vs satellites", "Over time", "Distribution"],
                    horizontal=True,
                    key="sat_view",
                )

                if view == "Cells vs satellites":
                    have_enb = "enb" in vis.columns and vis["enb"].notna().any()
                    d_enb = int(vis["enb"].nunique()) if have_enb else 0
                    d_pci = int(vis["pci"].nunique())
                    kk = st.columns(5)
                    kk[0].metric("Distinct eNB/gNB", f"{d_enb:,}" if have_enb else "—")
                    kk[1].metric("Distinct PCI", f"{d_pci:,}")
                    kk[2].metric(f"Distinct D2C sats ≥{int(sat_mask)}°", f"{n_distinct:,}")
                    kk[3].metric(
                        "eNB per satellite",
                        f"{d_enb / n_distinct:.1f}" if (have_enb and n_distinct > 0) else "—",
                    )
                    kk[4].metric(
                        "PCI per satellite",
                        f"{d_pci / n_distinct:.1f}" if n_distinct > 0 else "—",
                    )
                    st.caption(
                        "**Draw a region on the Map and set a Time window in the "
                        "sidebar** to read this for one spot over a few minutes. "
                        "eNB identity needs the connected radio, so use the "
                        "**RSRP · LTE** (or NR) metric. The ratios divide the "
                        "*distinct* eNB/PCI seen by the *distinct* satellites that "
                        "passed through the window (not the ~handful up at any "
                        "instant) — a rough sense of how many cells ride one "
                        "spacecraft."
                    )
                    # Per-collection (per-session) breakdown, so it is useful even
                    # before you window down to a single spot. Distinct-per-session
                    # is a propagation per group, so only do it for a few sessions.
                    groups = list(vis.groupby("collection", observed=True))
                    per_distinct = len(groups) <= 12
                    _dtc = (
                        satellites.filter_dtc(satellites.load_tle_dir(tle_dir))
                        if per_distinct else None
                    )
                    rows = []
                    for coll, g in groups:
                        c = pd.to_numeric(g[sat_vis_col], errors="coerce")
                        gt = pd.to_datetime(g["time"], errors="coerce")
                        mins = (gt.max() - gt.min()).total_seconds() / 60 if gt.notna().any() else 0.0
                        e = int(g["enb"].nunique()) if have_enb else 0
                        p = int(g["pci"].nunique())
                        peak = int(c.max()) if c.notna().any() else 0
                        row = {
                            "Collection": coll, "Minutes": round(mins, 1),
                            "Samples": len(g), "eNB": e, "PCI": p, "Peak sats": peak,
                        }
                        if per_distinct:
                            ds = satellites.distinct_visible_count(
                                g, _dtc, int(sat_mask), utc_offset_hours=float(sat_offset)
                            )
                            row["Distinct sats"] = ds
                            row["eNB/sat"] = round(e / ds, 2) if (ds and have_enb) else None
                            row["PCI/sat"] = round(p / ds, 2) if ds else None
                        rows.append(row)
                    bd = pd.DataFrame(rows).sort_values("Samples", ascending=False)
                    with st.expander("Per-collection breakdown", expanded=True):
                        if not per_distinct:
                            st.caption(
                                f"{len(groups)} collections selected — narrow the "
                                "filters (or window down) to under 12 to get the "
                                "per-session distinct-satellite count and ratios."
                            )
                        st.dataframe(bd, width="stretch", hide_index=True, height=300)
                        st.download_button(
                            "Download (CSV)", bd.to_csv(index=False).encode(),
                            file_name="cells_vs_satellites.csv", mime="text/csv",
                        )
                elif view == "Over time":
                    ts = vis[["time", "collection", sat_vis_col]].copy()
                    ts["time"] = pd.to_datetime(ts["time"], errors="coerce")
                    ts = ts.dropna(subset=["time"]).sort_values("time")
                    fig = charts.sat_timeseries_figure(
                        ts, sat_vis_col, int(sat_mask), mode=MODE
                    )
                    st.plotly_chart(fig, width="stretch")
                elif view == "Distribution":
                    fig = charts.sat_histogram_figure(counts, int(sat_mask), mode=MODE)
                    st.plotly_chart(fig, width="stretch")

                with st.expander("Per-sample visibility as a table"):
                    tbl = vis[["time", "collection", "pci", sat_vis_col, "sat_max_elev"]].rename(
                        columns={sat_vis_col: f"visible≥{int(sat_mask)}°",
                                 "sat_max_elev": "max elev (°)"}
                    )
                    st.dataframe(tbl.head(5000), width="stretch", height=340, hide_index=True)
                    st.download_button(
                        "Download visibility (CSV)",
                        vis[["time", "collection", "lat", "lon", "pci", sat_vis_col,
                             "sat_max_elev"]].to_csv(index=False).encode(),
                        file_name=f"d2c_visibility_{int(sat_mask)}deg.csv",
                        mime="text/csv",
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


# --------------------------------------------------------------------------- #
# Cell identity
# --------------------------------------------------------------------------- #

with tab_cells:
    # Cell identity from whichever active sources carry it. QP connected-mode
    # Radio exports give a *serving* cell over time (dwell, hand-overs). PRISM is
    # an SDR scanner: its decoded eNB / sector / PCI are identity observations
    # (every audible cell), not a serving sequence — so they feed the relationship
    # views (Hierarchy, Structure), not the serving-cell timelines.
    qp_radio_keys = tuple(
        k
        for k in sorted(qp_present)
        if k in ("lte_radio", "nr_radio")
        and (tech_value is None or DATASETS_BY_KEY[k].tech == tech_value)
    ) if "QP" in providers_active else ()
    prism_id_keys = tuple(
        k
        for k in sorted(prism_present)
        if tech_value is None or PRISM_DATASETS_BY_KEY[k].tech == tech_value
    ) if "PRISM" in providers_active else ()
    # SigCap logs the serving cell too (its "primary" cell), so it feeds the
    # serving-cell timelines alongside QP — it just carries PCI/Cell Id, not the
    # eNB/sector split, so it adds to the PCI view rather than the eNB hierarchy.
    sigcap_id_keys = tuple(
        k
        for k in sorted(sigcap_present)
        if tech_value is None or SIGCAP_DATASETS_BY_KEY[k].tech == tech_value
    ) if "SIGCAP" in providers_active else ()

    if not qp_radio_keys and not prism_id_keys and not sigcap_id_keys:
        st.info(
            "Cell identity needs the QP Radio [connected] export, a PRISM scan "
            "that decodes eNB / PCI, or a SigCap session — none is available for "
            "this selection."
        )
        st.stop()

    _id_parts = []
    if qp_radio_keys:
        _id_parts.append(load_frame(qp_dir, qp_radio_keys))
    if prism_id_keys:
        _id_parts.append(load_prism_frame(prism_dir, prism_id_keys))
    if sigcap_id_keys:
        _id_parts.append(load_sigcap_frame(sigcap_dir, sigcap_id_keys))
    radio = loader.concat_frames(_id_parts)
    radio = _align_qp_d2c(radio)

    def _intersecting(values, column):
        """Only carry a filter across if it means anything in the identity frame."""
        if not values:
            return []
        present_values = set(loader.options(radio, column))
        return [v for v in values if v in present_values]

    ident = flt.apply_filters(
        radio,
        flt.Filters(
            collections=collections,
            traffic=traffic,
            bands=_intersecting(bands, "band"),
            freqs=_intersecting(freqs, "freq"),
            pcis=_intersecting(effective_pcis, "pci"),
            regions=st.session_state["regions"],
            time_range=time_range,
            keep_ungeolocated=keep_ungeo,
            require=("pci",) if drop_unknown else (),
        ),
    )
    # QP rows: keep only the serving cell (PCell / PSCell). PRISM rows have no
    # serving concept, so keep every decoded identity row.
    if "provider" in ident.columns:
        prov = ident["provider"].astype("string")
        is_prism = (prov == "PRISM").to_numpy()
        is_sigcap = (prov == "SIGCAP").to_numpy()
        # QP marks the serving cell with a cell_type; keep only those. SigCap's
        # primary cell is already the serving one (no cell_type to filter on), so
        # keep it as-is. PRISM scans have no serving concept — keep every row.
        cells = pd.concat(
            [
                identity.serving_only(ident.loc[~is_prism & ~is_sigcap]),
                ident.loc[is_sigcap],
                ident.loc[is_prism],
            ],
            ignore_index=True,
        )
    else:
        cells = identity.serving_only(ident)
    # a cell needs at least a PCI to place it; drop identity-less rows
    if "pci" in cells.columns:
        cells = cells[cells["pci"].notna()]

    # the QP serving-cell subset drives the time views; everything (QP + PRISM)
    # drives the relationship views
    if "provider" in cells.columns:
        serving = cells[cells["provider"].isin(["QP", "SIGCAP"])]
    else:
        serving = cells
    levels = identity.available_levels(cells)
    serving_levels = identity.available_levels(serving)

    if cells.empty or not levels:
        st.info("No cell-identity rows match the current filters.")
    else:
        has_prov = "provider" in cells.columns
        n_prism = int((cells["provider"] == "PRISM").sum()) if has_prov else 0
        n_qp = int((serving["provider"] == "QP").sum()) if has_prov else len(serving)
        n_sigcap = int((serving["provider"] == "SIGCAP").sum()) if has_prov else 0
        src_parts = []
        if n_qp and qp_radio_keys:
            src_parts.append(
                f"QP serving cell ({n_qp:,} samples, "
                + ", ".join(DATASETS_BY_KEY[k].label for k in qp_radio_keys) + ")"
            )
        if n_sigcap:
            src_parts.append(
                f"SigCap serving cell ({n_sigcap:,} samples — PCI / Cell Id only)"
            )
        if n_prism:
            src_parts.append(
                f"PRISM decoded scans ({n_prism:,} rows — every audible cell, "
                "in the Hierarchy / Structure views)"
            )
        st.caption("Cell identity from " + " + ".join(src_parts) + ".")

        # serving-cell time views need QP rows; the relationship views work for
        # either source — that is where PRISM's decoded identity appears
        view_options = (["Timeline", "Change timeline"] if not serving.empty else []) + [
            "Hierarchy", "Structure",
        ]
        if st.session_state.get("identity_view") not in view_options:
            st.session_state.pop("identity_view", None)
        view = st.radio(
            "View", view_options, horizontal=True, key="identity_view",
        )

        if view == "Change timeline":
            st.caption(
                "**Serving-cell timeline** with **eNB Id on the y-axis** and one "
                "segment per Cell Id. A segment is **orange** when the **eNB** "
                "changed (a site/satellite change) and **blue** when only the "
                "**Cell Id** changed under the same eNB (a beam/sector change). Each "
                "segment is labelled with its **PCI above** and **sector below** "
                "(`s#`), and dotted lines connect consecutive cells so an eNB jump "
                "is visible. One panel per collection."
            )
            if "enb" not in serving_levels:
                st.info(
                    "This needs the connected radio to carry an eNB/gNB-Id. It is "
                    "not decoded in the current selection."
                )
            else:
                segments = identity.serving_segments(serving)
                summary = identity.segment_summary(segments)
                if segments.empty:
                    st.info("Not enough serving-cell samples to build segments.")
                else:
                    by_size = serving.groupby("collection", observed=True).size()
                    all_colls = by_size.sort_values(ascending=False).index.tolist()
                    default = all_colls[:4]
                    chosen = st.multiselect(
                        "Collections (one panel each)",
                        options=all_colls, default=default, key="chg_collections",
                        help="Each collection is a separate session, drawn as its "
                             "own panel from elapsed time 0.",
                    )
                    if not chosen:
                        st.info("Pick at least one collection.")
                    else:
                        srow = {r["collection"]: r for _, r in summary.iterrows()}

                        def _subtitle(coll):
                            r = srow.get(coll)
                            if r is None:
                                return str(coll)
                            return (
                                f"{coll}<br>{r['n_enb']} eNB Ids · {r['n_cell']} "
                                f"Cell Ids · {r['n_pci']} PCIs in {r['span_min']:.0f} min"
                                f"   |   {r['cell_changes']} cell changes "
                                f"({r['cell_per_min']:.1f}/min), {r['enb_changes']} "
                                f"eNB changes ({r['enb_per_min']:.1f}/min), median "
                                f"dwell {r['median_dwell_s']:.0f} s"
                            )

                        st.plotly_chart(
                            charts.serving_timeline_figure(
                                segments[segments["collection"].isin(chosen)],
                                collections=chosen,
                                labels=[str(c) for c in chosen],
                                subtitles=[_subtitle(c) for c in chosen],
                                mode=MODE,
                            ),
                            width="stretch",
                        )
                        with st.expander("Segments as a table"):
                            cols = [c for c in ("collection", "enb", "cell_id", "sector",
                                                "pci", "change", "elapsed_start",
                                                "duration_s", "samples")
                                    if c in segments.columns]
                            table = segments.loc[
                                segments["collection"].isin(chosen), cols
                            ].rename(
                                columns={
                                    "collection": "Collection", "enb": "eNB",
                                    "cell_id": "Cell Id", "sector": "Sector",
                                    "pci": "PCI", "change": "Change",
                                    "elapsed_start": "Elapsed start (s)",
                                    "duration_s": "Held (s)", "samples": "Samples",
                                }
                            )
                            st.dataframe(table, width="stretch", hide_index=True, height=320)
                            st.download_button(
                                "Download segments (CSV)",
                                table.to_csv(index=False).encode(),
                                file_name="serving_segments.csv",
                                mime="text/csv",
                            )

        elif view == "Timeline":
            row = st.columns([2, 2, 2, 2])
            level = row[0].selectbox(
                "Identity", serving_levels, format_func=identity.label_for,
                key="identity_level",
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

            runs = identity.dwell_runs(serving, level)
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
        elif view == "Hierarchy":
            default_levels = [l for l in ("enb", "sector", "pci") if l in levels]
            chosen = st.multiselect(
                "Levels, outer → inner",
                options=levels,
                default=default_levels or levels[:1],
                format_func=identity.label_for,
                key="identity_hierarchy",
                help="Nodes are sized by how long that cell served; branches "
                     "are ordered longest-held first.",
            )
            if not chosen:
                st.info("Pick at least one level.")
            else:
                table = identity.hierarchy(cells, chosen)
                # A tree of a thousand leaves is a wall, not a picture. Focus on
                # one branch to read a subtree whole; the download is always the
                # complete set either way.
                outer = chosen[0]
                branches = (
                    table.groupby(outer, observed=True)["dwell_s"]
                    .sum()
                    .sort_values(ascending=False)
                )
                focus = st.selectbox(
                    f"Focus on one {identity.label_for(outer)}",
                    options=["All"] + [str(v) for v in branches.index],
                    key="identity_focus_branch",
                    help="Longest-held first. Picking one shows that branch "
                         "complete, however many leaves it has.",
                )
                if focus != "All":
                    table = table[table[outer].astype(str) == focus]
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
                    tree_cols = st.columns([2, 2])
                    max_leaves = tree_cols[0].number_input(
                        "Max leaves drawn",
                        min_value=10, max_value=5000, value=150, step=10,
                        key="identity_leaves",
                        help="Longest-held paths win when there are more. The "
                             "download below is always the complete set.",
                    )
                    n_columns = tree_cols[1].number_input(
                        "Columns",
                        min_value=1, max_value=4, value=2, step=1,
                        key="identity_columns",
                        help="Break the eNBs across this many side-by-side "
                             "sub-trees so each row has room to breathe. The "
                             "columns are balanced by leaf count.",
                    )
                    st.plotly_chart(
                        charts.tree_figure(
                            table, chosen,
                            [identity.label_for(l) for l in chosen],
                            mode=MODE, max_leaves=int(max_leaves),
                            columns=int(n_columns),
                        ),
                        width="stretch",
                    )
                    if len(table) > max_leaves:
                        drawn = table.nlargest(int(max_leaves), "dwell_s")
                        covered = drawn["dwell_s"].sum() / table["dwell_s"].sum()
                        st.caption(
                            f"Drawing the {int(max_leaves):,} longest-held of "
                            f"{len(table):,} paths — {covered:.0%} of the dwell "
                            f"time in view. The table and CSV below are complete."
                        )
                    with st.expander("Hierarchy as a table"):
                        pretty = table.rename(
                            columns={l: identity.label_for(l) for l in chosen}
                        ).rename(columns={"dwell_s": "Held (s)", "samples": "Samples"})
                        st.dataframe(pretty.round(1), width="stretch", hide_index=True, height=320)
                        st.download_button(
                            f"Download all {len(pretty):,} paths (CSV)",
                            pretty.to_csv(index=False).encode(),
                            file_name="cell_hierarchy.csv",
                            mime="text/csv",
                        )


        if view == "Structure":
            st.caption(
                "How wide the identity tree is: how many sectors an eNB / gNB "
                "carries, and how many PCIs a sector carries."
            )
            row = st.columns([2, 2, 2])
            struct_levels = row[0].multiselect(
                "Levels, outer → inner",
                options=levels,
                default=[l for l in ("enb", "sector", "pci") if l in levels] or levels[:2],
                format_func=identity.label_for,
                key="structure_levels",
            )
            split_by = row[1].selectbox(
                "Split by",
                options=["None", "tech", "link", "collection", "operator"],
                format_func=lambda c: "Nothing" if c == "None" else stats.GROUP_DIMENSIONS.get(c, c.title()),
                key="structure_split",
                help="Compare populations — terrestrial and satellite fan out "
                     "very differently.",
            )
            as_cdf = row[2].checkbox(
                "Show as CDF", value=False, key="structure_cdf",
                help="The same population read as P(children ≤ n).",
            )

            if len(struct_levels) < 2:
                st.info("Pick at least two levels — fan-out needs a parent and a child.")
            else:
                counts = identity.fan_out_counts(
                    cells,
                    struct_levels,
                    group_by=None if split_by == "None" else split_by,
                )
                summary = identity.fan_out_summary(counts)
                if summary.empty:
                    st.info("No rows carry two complete levels.")
                else:
                    st.dataframe(
                        summary.round(2), width="stretch", hide_index=True
                    )
                    st.download_button(
                        "Download fan-out summary (CSV)",
                        summary.to_csv(index=False).encode(),
                        file_name="cell_fan_out_summary.csv",
                        mime="text/csv",
                    )

                    dist = identity.fan_out_distribution(counts)
                    relationship = st.selectbox(
                        "Chart",
                        options=identity.relationship_order(counts),
                        key="structure_relationship",
                        help="Tree order: the outermost pair first, which is "
                             "usually the one that varies.",
                    )
                    st.plotly_chart(
                        charts.fan_out_figure(
                            dist, relationship, mode=MODE, cumulative=as_cdf
                        ),
                        width="stretch",
                    )
                    with st.expander("Distribution as a table"):
                        pretty = dist.rename(
                            columns={
                                "relationship": "Relationship",
                                "group": "Group",
                                "children": "Children",
                                "parents": "Parents",
                                "share": "Share",
                            }
                        )
                        st.dataframe(
                            pretty.round(4), width="stretch", hide_index=True, height=320
                        )
                        st.download_button(
                            "Download distribution (CSV)",
                            pretty.to_csv(index=False).encode(),
                            file_name="cell_fan_out_distribution.csv",
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


# --------------------------------------------------------------------------- #
# Correlation  (QP PHY layer  x  SigCap application layer, same phone)
# --------------------------------------------------------------------------- #

if tab_corr is not None:
    with tab_corr:
        st.markdown(
            "**PHY vs. application, on the same phone.** Qualipoc logs the PHY "
            "layer; SigCap, recording at the same time, logs what the application "
            "actually got — iperf / HTTP throughput and ping RTT. Both are "
            "averaged into a common time bin and paired bin-to-bin over the window "
            "where the two recordings overlap, so you can see how the radio "
            "conditions translated into delivered performance."
        )

        qp_keys_tech = [
            k for k in qp_present
            if tech_value is None or DATASETS_BY_KEY[k].tech == tech_value
        ]
        sig_keys_tech = [
            k for k in sigcap_present
            if tech_value is None or SIGCAP_DATASETS_BY_KEY[k].tech == tech_value
        ]
        # QP contributes the PHY metric; app_tput never comes from QP
        phy_options = [m for m in metrics_available(qp_keys_tech) if m != "app_tput"]
        app_present = metrics_available(sig_keys_tech)
        app_options = [m for m in ("app_tput", "rtt") if m in app_present]

        if not phy_options or not app_options:
            st.info(
                "Correlation needs a QP PHY metric and a SigCap app-layer metric "
                f"for {tech}. QP PHY metrics available: "
                f"{len(phy_options)}; SigCap app metrics available: "
                f"{len(app_options)}."
            )
        else:
            c1, c2, c3, c4 = st.columns(4)
            phy_metric = c1.selectbox(
                "QP PHY metric", phy_options,
                index=phy_options.index("throughput") if "throughput" in phy_options else 0,
                format_func=lambda m: METRICS[m].axis_label, key="corr_phy",
                help="From the Qualipoc PHY-layer exports (MCS, PDSCH throughput, "
                     "spectral efficiency, BLER, RSRP, …).",
            )
            app_metric = c2.selectbox(
                "SigCap app metric", app_options,
                format_func=lambda m: METRICS[m].axis_label, key="corr_app",
                help="Application-layer performance SigCap measured (iperf / HTTP "
                     "throughput, ping RTT).",
            )
            bin_choice = c3.selectbox(
                "Time bin", ["1s", "2s", "5s", "10s", "15s", "30s", "60s"],
                index=2, key="corr_bin",
                help="Both sources are averaged into buckets this wide, then paired "
                     "bucket-to-bucket. SigCap samples about every 5 s.",
            )
            agg_choice = c4.selectbox(
                "Aggregate", ["mean", "median", "max", "min"], index=0, key="corr_agg",
                help="How samples within a bucket are combined before pairing.",
            )

            qp_corr_keys = [
                d.key for d in datasets_for(phy_metric, tech_value, available=qp_keys_tech)
            ]
            sig_corr_keys = [
                d.key for d in sigcap_datasets_for(app_metric, tech_value, available=sig_keys_tech)
            ]
            qp_cf = (
                load_frame(qp_dir, tuple(sorted(qp_corr_keys)))
                if qp_corr_keys else loader.empty_frame()
            )
            sig_cf = (
                load_sigcap_frame(sigcap_dir, tuple(sorted(sig_corr_keys)))
                if sig_corr_keys else loader.empty_frame()
            )
            # honour the sidebar time window and any region drawn on the map, so a
            # session can be isolated the same way as everywhere else
            _cf_filters = flt.Filters(
                traffic=traffic,
                time_range=time_range,
                regions=st.session_state["regions"],
                keep_ungeolocated=keep_ungeo,
            )
            qp_cf = flt.apply_filters(qp_cf, _cf_filters)
            sig_cf = flt.apply_filters(sig_cf, _cf_filters)

            # tab-local narrowing of the SigCap side: workload, direction, session
            fc1, fc2, fc3 = st.columns(3)
            work_opts = loader.options(sig_cf, "test_type")
            workloads = fc1.multiselect(
                "SigCap workload", work_opts, default=[], key="corr_work",
                placeholder=f"All ({len(work_opts)})",
                help="iperf vs. HTTP. Leave empty to include all.",
            )
            dir_opts = loader.options(sig_cf, "link")
            dirs = fc2.multiselect(
                "Direction", dir_opts, default=[], key="corr_dir",
                placeholder=f"All ({len(dir_opts)})",
                help="Applied to both sources (DL/UL). Leave empty to include all.",
            )
            sess_opts = loader.options(sig_cf, "collection")
            sessions = fc3.multiselect(
                "SigCap session", sess_opts, default=[], key="corr_sess",
                placeholder=f"All ({len(sess_opts)})",
                help="Restrict to one or more SigCap capture sessions.",
            )
            if workloads:
                sig_cf = sig_cf[sig_cf["test_type"].astype("string").isin(workloads)]
            if sessions:
                sig_cf = sig_cf[sig_cf["collection"].astype("string").isin(sessions)]
            if dirs:
                sig_cf = sig_cf[sig_cf["link"].astype("string").isin(dirs)]
                if "link" in qp_cf.columns:
                    qp_cf = qp_cf[qp_cf["link"].astype("string").isin(dirs)]

            result = correlate.correlate(
                qp_cf, phy_metric, sig_cf, app_metric,
                bin=bin_choice, qp_agg=agg_choice, sig_agg=agg_choice,
            )

            def _span(fr):
                if fr is None or fr.empty or "time" not in fr.columns:
                    return None
                t = pd.to_datetime(fr["time"], errors="coerce").dropna()
                return (t.min(), t.max()) if not t.empty else None

            qp_span, sig_span = _span(qp_cf), _span(sig_cf)
            phy_label = METRICS[phy_metric].axis_label
            app_label = METRICS[app_metric].axis_label

            if sig_span is None:
                st.info("No SigCap samples for this metric and the current filters.")
            elif qp_span is None:
                st.info(
                    "No QP PHY samples yet for this selection. Upload the QP logs "
                    "captured alongside these SigCap sessions, then pick a PHY "
                    "metric here to correlate. SigCap currently covers "
                    f"**{sig_span[0]:%Y-%m-%d %H:%M}** – **{sig_span[1]:%Y-%m-%d %H:%M}**."
                )
            elif not result.has_overlap:
                st.warning(
                    "QP and SigCap don't overlap in time, so there is nothing to "
                    f"pair. QP covers **{qp_span[0]:%Y-%m-%d %H:%M}** – "
                    f"**{qp_span[1]:%Y-%m-%d %H:%M}**; SigCap covers "
                    f"**{sig_span[0]:%Y-%m-%d %H:%M}** – **{sig_span[1]:%Y-%m-%d %H:%M}**. "
                    "Correlation needs QP and SigCap recorded at the same time on "
                    "the same phone."
                )
            elif not result.has_pairs:
                st.warning(
                    f"Only {result.n} paired {bin_choice} bin(s) in the overlap — "
                    "not enough to correlate. Try a coarser time bin or widen the "
                    "time window."
                )
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Pearson r", f"{result.pearson:+.2f}")
                m2.metric("Spearman ρ", f"{result.spearman:+.2f}")
                m3.metric("Paired bins", f"{result.n:,}")
                m4.metric(
                    "Overlap",
                    f"{(result.overlap[1] - result.overlap[0]).total_seconds() / 60:.0f} min",
                    help=f"{result.overlap[0]:%Y-%m-%d %H:%M:%S} – "
                         f"{result.overlap[1]:%H:%M:%S}",
                )
                st.plotly_chart(
                    charts.correlation_timeline_figure(
                        result.merged, phy_label, app_label, mode=MODE
                    ),
                    width="stretch",
                )
                st.plotly_chart(
                    charts.correlation_scatter_figure(
                        result.merged, phy_label, app_label, mode=MODE
                    ),
                    width="stretch",
                )
                st.caption(
                    f"Each point is one {bin_choice} bin ({agg_choice}) present in "
                    "both recordings. Pearson measures a linear relationship; "
                    "Spearman a monotonic one. Correlation is not causation — "
                    "confounders (distance, load, beam changes) move both."
                )


# --------------------------------------------------------------------------- #
# Timeline analysis  (multi-phone contention + shared-PCI, SigCap)
# --------------------------------------------------------------------------- #

if tab_timeline is not None:
    with tab_timeline:
        st.markdown(
            "**One phone or two, on the same spot.** SigCap sessions from "
            "different phones can run at the same time. This aligns them on one "
            "wall clock so you can see how each phone's throughput changes when a "
            "second phone starts using Starlink D2C, and whether throughput moves "
            "together when both phones camp on the **same PCI**. How many phones "
            "are \"using\" it at any instant is counted from the data (distinct "
            "device UUIDs with a sample in the same bin), not from the file names."
        )

        sig_keys_tech = [
            k for k in sigcap_present
            if tech_value is None or SIGCAP_DATASETS_BY_KEY[k].tech == tech_value
        ]
        tframe = (
            load_sigcap_frame(sigcap_dir, tuple(sorted(sig_keys_tech)))
            if sig_keys_tech else loader.empty_frame()
        )
        # Honour the sidebar Collections / time-window / region filters so
        # selecting the one-phone & two-phone sessions scopes the analysis - but
        # NOT the Direction filter (the timeline splits DL vs UL itself) or the
        # PCI filter (it analyses which PCIs the phones used).
        tframe = flt.apply_filters(
            tframe,
            flt.Filters(
                collections=collections,
                traffic=traffic,
                time_range=time_range,
                regions=st.session_state["regions"],
                keep_ungeolocated=keep_ungeo,
            ),
        )
        has_dev = (
            not tframe.empty
            and "device" in tframe.columns
            and tframe["device"].notna().any()
        )
        if not has_dev:
            st.info(
                "Timeline analysis needs SigCap sessions that carry a device id. "
                "None are available for the current technology and filters."
            )
        else:
            dev_all = sorted(tframe["device"].dropna().astype(str).unique())
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            picked = c1.multiselect(
                "Phones", dev_all, default=dev_all, key="tl_devices",
                help="Each SigCap device UUID is one physical phone.",
            )
            bin_choice = c2.selectbox(
                "Time bin", ["1s", "2s", "5s", "10s", "15s", "30s"],
                index=2, key="tl_bin",
                help="Samples are averaged into buckets this wide before aligning "
                     "the phones. SigCap samples about every 5 s.",
            )
            agg_choice = c3.selectbox(
                "Aggregate", list(timeline.AGGS), index=0, key="tl_agg",
            )
            overlay_on = c4.checkbox(
                "QP PHY overlay", value=bool(qp_present), key="tl_overlay",
                help="Draw QP PHY-layer throughput / RSRP where a QP log overlaps "
                     "these sessions in time. Usually empty unless QP was recording "
                     "at the same time.",
            )
            if picked:
                tframe = tframe[tframe["device"].astype(str).isin(picked)]

            res = timeline.build(
                tframe, bin=bin_choice, agg=agg_choice, metric="app_tput"
            )

            if res.binned.empty:
                st.info("No SigCap samples in the current window.")
            else:
                span = (res.binned["time"].min(), res.binned["time"].max())
                mcols = st.columns(4)
                mcols[0].metric("Phones", f"{len(res.devices)}")
                mcols[1].metric("Max concurrent", f"{res.max_active}")
                mcols[2].metric("Shared-PCI bins", f"{res.n_shared_bins:,}")
                _span_min = (span[1] - span[0]).total_seconds() / 60.0
                _span_txt = (
                    f"{_span_min:.0f} min" if _span_min < 90
                    else f"{_span_min / 60:.1f} h" if _span_min < 1440
                    else f"{_span_min / 1440:.1f} d"
                )
                mcols[3].metric(
                    "Span", _span_txt,
                    help=f"{span[0]:%Y-%m-%d %H:%M} – {span[1]:%Y-%m-%d %H:%M}. "
                         "Set a time window in the sidebar to zoom to one session.",
                )

                # QP PHY overlay, binned to the same grid and split PER PHONE.
                # QP recorded both phones (one collection each), so every QP
                # series - PHY throughput, RSRP and RBs - is keyed by the phone
                # slot in the collection name and coloured to match that phone,
                # exactly like the SigCap traces. Falls back to a single pooled
                # "QP" series (slot 0) when a collection carries no phoneN token.
                qp_overlay = None     # long: time, slot, phy_tput_dl/ul, rsrp, rb_dl/ul
                sig_slots = timeline.device_slots(tframe)   # SigCap device -> phone slot
                slot_labels = {}      # phone slot -> a SigCap device label (legend)
                for _dev, _sl in sig_slots.items():
                    slot_labels.setdefault(_sl, _dev)
                if overlay_on and qp_present:
                    qp_keys_tech = [
                        k for k in qp_present
                        if tech_value is None or DATASETS_BY_KEY[k].tech == tech_value
                    ]
                    ov_keys = set()
                    for m in ("throughput", "rsrp", "rb"):
                        ov_keys |= {
                            d.key for d in datasets_for(m, tech_value, available=qp_keys_tech)
                        }
                    if ov_keys:
                        qf = load_frame(qp_dir, tuple(sorted(ov_keys)))
                        qf = qf[(qf["time"] >= span[0]) & (qf["time"] <= span[1])]
                        # Scope QP to the SAME sessions as the SigCap timeline via
                        # the shared collection tokens (see below).
                        _sel = [c for c in (collections or []) if c]
                        if _sel and "collection" in qf.columns:
                            _qp_colls = set(qf["collection"].dropna().astype(str).unique())
                            _scope = {c for c in _sel if c in _qp_colls}
                            for _sc in _sel:
                                if _sc in _qp_colls:
                                    continue
                                _toks = [t for t in re.split(r"[\s_]+", _sc.lower()) if t]
                                if not _toks:
                                    continue
                                for _qc in _qp_colls:
                                    if all(t in _qc.lower() for t in _toks):
                                        _scope.add(_qc)
                            if _scope:
                                qf = qf[qf["collection"].astype(str).isin(_scope)]
                                st.session_state["_tl_qp_scope"] = sorted(_scope)
                            else:
                                st.session_state["_tl_qp_scope"] = []
                        else:
                            st.session_state["_tl_qp_scope"] = []
                        if not qf.empty:
                            qf = qf.assign(_b=qf["time"].dt.floor(bin_choice))
                            qf["slot"] = qf["collection"].map(timeline.phone_slot)
                            if qf["slot"].isna().all():
                                qf["slot"] = 0   # no phoneN token -> one pooled series
                            qf = qf.dropna(subset=["slot"])
                            if not qf.empty:
                                qf["slot"] = qf["slot"].astype(int)
                                _link = qf["link"].astype("string")

                                def _qcol(name):
                                    return (pd.to_numeric(qf[name], errors="coerce")
                                            if name in qf.columns
                                            else pd.Series(float("nan"), index=qf.index))

                                _tp, _rs, _rb = _qcol("throughput"), _qcol("rsrp"), _qcol("rb")
                                recs = []
                                for (b, slot), g in qf.groupby(["_b", "slot"]):
                                    idx = g.index
                                    gl = _link.loc[idx]

                                    def _m(series, mask=None, scale=1.0):
                                        x = series.loc[idx]
                                        if mask is not None:
                                            x = x[mask]
                                        return float(x.mean() * scale) if x.notna().any() else float("nan")

                                    recs.append({
                                        "time": b, "slot": int(slot),
                                        "phy_tput_dl": _m(_tp, gl.eq("DL")),
                                        "phy_tput_ul": _m(_tp, gl.eq("UL")),
                                        "rsrp": _m(_rs),
                                        "rb_dl": _m(_rb, gl.eq("DL")),
                                        "rb_ul": _m(_rb, gl.eq("UL")),
                                    })
                                if recs:
                                    qp_overlay = pd.DataFrame(recs).sort_values("time")
                if overlay_on and (qp_overlay is None or qp_overlay.empty):
                    st.caption(
                        "No QP PHY samples overlap this window — the overlay is "
                        "empty (QP wasn't recording during these sessions)."
                    )

                st.plotly_chart(
                    charts.timeline_multi_figure(
                        res.binned, res.devices, res.concurrency,
                        res.shared_pci, qp_overlay=qp_overlay,
                        device_slots=sig_slots, slot_labels=slot_labels, mode=MODE,
                    ),
                    width="stretch",
                )
                st.caption(
                    "Downlink and uplink throughput are in separate panels; each "
                    "phone keeps its colour down every panel. Grey bands = two "
                    "phones active at once; orange bands = the concurrent phones "
                    "shared one serving PCI."
                )
                if res.detect_bin != bin_choice:
                    st.caption(
                        f"Phones sample about every {res.detect_bin}, so the grey "
                        f"/ orange bands (and the comparison tables) are detected "
                        f"at {res.detect_bin} even though the lines are drawn at "
                        f"{bin_choice} — otherwise the two phones rarely land in "
                        "the same fine bin and the shading would vanish."
                    )
                _qp_scope = st.session_state.get("_tl_qp_scope") or []
                if overlay_on and _qp_scope:
                    st.caption(
                        "QP overlay scoped to the matching QP collection(s): "
                        + ", ".join(f"`{c}`" for c in _qp_scope) + "."
                    )

                # ---- quantification --------------------------------------- #
                def _fmt(df, rename, tputcols):
                    d = df.copy()
                    for c in tputcols:
                        if c in d.columns:
                            d[c] = d[c].map(
                                lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                            )
                    if "delta_pct" in d.columns:
                        d["delta_pct"] = d["delta_pct"].map(
                            lambda v: f"{v:+.0f}%" if pd.notna(v) else "—"
                        )
                    return d.rename(columns=rename)

                if res.has_concurrency:
                    st.markdown("#### 1 phone vs 2 phones active")
                    st.caption(
                        "Mean throughput (Mbps) for each phone when it was the "
                        "only one transmitting vs when a second phone was active "
                        "at the same instant. Δ is the change under contention."
                    )
                    dl = timeline.compare_by_concurrency(res.binned, "app_tput_dl")
                    ul = timeline.compare_by_concurrency(res.binned, "app_tput_ul")
                    both = timeline.compare_by_concurrency(res.binned, "app_tput")
                    tabs_dir = st.tabs(["Both directions", "Downlink", "Uplink"])
                    for tb, df in zip(tabs_dir, (both, dl, ul)):
                        with tb:
                            st.dataframe(
                                _fmt(
                                    df,
                                    {"device": "Phone", "solo_n": "solo bins",
                                     "solo_mean": "solo mean", "concurrent_n": "2-phone bins",
                                     "concurrent_mean": "2-phone mean", "delta_pct": "Δ"},
                                    ["solo_mean", "concurrent_mean"],
                                ),
                                hide_index=True, width="stretch",
                            )
                else:
                    st.info(
                        "Only one phone is active in this window, so there is no "
                        "1-vs-2-phone contention to compare. Widen the time window "
                        "or clear filters to include the concurrent sessions."
                    )

                if res.n_shared_bins > 0:
                    st.markdown("#### Same PCI vs different PCI")
                    st.caption(
                        "Among the bins where both phones were active, mean "
                        "throughput (Mbps) when they shared one serving PCI vs "
                        "when they were on different PCIs."
                    )
                    sp_both = res.by_shared_pci
                    st.dataframe(
                        _fmt(
                            sp_both,
                            {"device": "Phone", "shared_n": "same-PCI bins",
                             "shared_mean": "same-PCI mean", "diff_n": "diff-PCI bins",
                             "diff_mean": "diff-PCI mean", "delta_pct": "Δ"},
                            ["shared_mean", "diff_mean"],
                        ),
                        hide_index=True, width="stretch",
                    )

                # ---- QP RB sanity check (per phone) ----------------------- #
                _rb_present = qp_overlay is not None and any(
                    c in qp_overlay.columns and qp_overlay[c].notna().any()
                    for c in ("rb_dl", "rb_ul")
                )
                if _rb_present:
                    st.markdown("#### QP RBs / subframe — sanity check")
                    st.caption(
                        "QP PHY resource blocks per phone (scheduled sub-frames — "
                        "*Avg RBs per SF*, the grant size, up to the carrier's "
                        "25-RB ceiling on 5 MHz) under each condition. If a shared "
                        "cell splits its RBs between two UEs, each phone should get "
                        "**fewer** RBs when two phones are active and when they "
                        "share a PCI (a negative Δ). The *all-SF* occupancy is the "
                        "separate `RBs per subframe (all SF)` metric."
                    )

                    def _cell(mean, n):
                        return f"{mean:.1f} RB (n={int(n)})" if pd.notna(mean) else "—"

                    def _cond_rows(phone, series_df, col, lbl):
                        cond = timeline.overlay_by_condition(
                            res.binned_detect, series_df, col, bin=res.detect_bin)
                        out = []
                        for _, r in cond.iterrows():
                            out.append({
                                "Phone": phone, "RBs": lbl,
                                "Comparison": f"{r['a_label']} vs {r['b_label']}",
                                "Higher load": _cell(r["a_mean"], r["a_n"]),
                                "Baseline": _cell(r["b_mean"], r["b_n"]),
                                "Δ": f"{r['delta_pct']:+.0f}%" if pd.notna(r["delta_pct"]) else "—",
                            })
                        return out

                    rb_rows = []
                    for slot, g in qp_overlay.groupby("slot"):
                        phone = (slot_labels.get(int(slot), f"phone{int(slot)}")
                                 if int(slot) >= 1 else "QP (pooled)")
                        for col, lbl in (("rb_dl", "DL"), ("rb_ul", "UL")):
                            if col in g.columns and g[col].notna().any():
                                rb_rows += _cond_rows(phone, g[["time", col]], col, lbl)

                    if rb_rows:
                        st.dataframe(
                            pd.DataFrame(rb_rows, columns=[
                                "Phone", "RBs", "Comparison", "Higher load",
                                "Baseline", "Δ"]),
                            hide_index=True, width="stretch",
                        )
                    else:
                        st.caption("No overlapping QP RB samples for these bins.")

                st.markdown("#### Throughput vs RSRP & serving PCI")
                st.caption(
                    "Per phone: correlation of app throughput with RSRP over the "
                    "window, and the serving PCI that carried the highest mean "
                    "throughput."
                )
                dv = res.drivers.copy()
                for c in ("pearson_rsrp", "spearman_rsrp"):
                    dv[c] = dv[c].map(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")
                dv["best_pci_mean"] = dv["best_pci_mean"].map(
                    lambda v: f"{v:.2f}" if pd.notna(v) else "—"
                )
                st.dataframe(
                    dv.rename(columns={
                        "device": "Phone", "n": "bins",
                        "pearson_rsrp": "r (tput,RSRP)",
                        "spearman_rsrp": "ρ (tput,RSRP)",
                        "best_pci": "best PCI",
                        "best_pci_mean": "best-PCI mean",
                        "n_pci": "# PCIs",
                    }),
                    hide_index=True, width="stretch",
                )
                st.caption(
                    "Concurrency is inferred from overlapping sample timestamps "
                    "across device UUIDs. Correlation is not causation — distance, "
                    "beam scheduling and load move several of these together."
                )
