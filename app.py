"""Starlink D2C - Qualipoc measurement visualisation platform.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from qpviz import filters as flt, loader, mapview, satellites
from qpviz.schema import (
    ALL_DATASETS_BY_KEY,
    DATASETS_BY_KEY,
    METRICS,
    PRISM_DATASETS_BY_KEY,
    SIGCAP_DATASETS_BY_KEY,
    TECHNOLOGIES,
    TRAFFIC_ORDER,
    datasets_for,
    metrics_available,
    prism_datasets_for,
    sigcap_datasets_for,
)

from qpviz.ui import (
    tab_cdf,
    tab_cells,
    tab_data,
    tab_map,
    tab_satellites,
    tab_stats,
)
from qpviz.ui.common import (
    align_qp_d2c,
    available,
    distinct_sats,
    load_frame,
    load_prism_frame,
    load_sigcap_frame,
    multiselect_all,
    prism_present_keys,
    release_memory,
    sigcap_present_keys,
    theme_mode,
    tle_summary,
    visibility_columns,
)
from qpviz.ui.context import Ctx

APP_TITLE = "Starlink D2C — Measurement Explorer"

st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="📡")
release_memory()  # what the previous rerun freed, before this one allocates

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
        st.cache_resource.clear()
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
                 "only); SigCap runs on the same phone and adds ping RTT. "
                 "Leave empty to pool all; "
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
        help="Which radio technology to read. Each metric is read from one "
             "fixed export per technology, named above the plots.",
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

_parts = []
if qp_source_keys:
    _parts.append(load_frame(qp_dir, tuple(sorted(qp_source_keys))))
if prism_source_keys:
    _parts.append(load_prism_frame(prism_dir, tuple(sorted(prism_source_keys))))
if sigcap_source_keys:
    _parts.append(load_sigcap_frame(sigcap_dir, tuple(sorted(sigcap_source_keys))))
frame = loader.concat_frames(_parts)
del _parts  # its frames are pooled into `frame` now
frame = frame[frame[metric].notna()] if metric in frame.columns else frame.iloc[0:0]
# D2C QP rows are QP-idle PHY captures of SigCap-generated traffic -> label their
# traffic from the time-aligned SigCap ground truth, not QP's own columns/name
frame = align_qp_d2c(frame, sigcap_dir, sigcap_present)
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
        "Require a known PCI and carrier",
        value=True,
        key="f_drop_unknown",
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
            links=links, collections=collections, traffic=traffic,
            bands=bands, freqs=freqs, pcis=effective_pcis,
            regions=st.session_state["regions"],
            require=("pci", "freq") if drop_unknown else (),
            keep_ungeolocated=keep_ungeo,
        ),
    )
    _tt = pd.to_datetime(_time_scope["time"], errors="coerce").dropna()
    # only its timestamps are needed; drop the frame and the per-step ones above
    # so they are not held next to `active` for the rest of the run
    del _time_scope, step0, step1, step2, step3
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
        carto_key = mapview.carto_api_key(st.secrets)
        tiles = st.selectbox(
            "Basemap",
            mapview.tile_choices(carto_key),
            # Light CARTO when a key is configured, otherwise OpenStreetMap.
            index=1 if carto_key else 0,
            help=None if carto_key else (
                "The CARTO Light/Dark basemaps need a free API key: set "
                "CARTO_API_KEY in .streamlit/secrets.toml or the environment."
            ),
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
    # every input that changes `frame`'s rows must be in here, or a cached
    # visibility column of the wrong length comes back
    _vis_key = (
        qp_dir, tuple(sorted(qp_source_keys)),
        prism_dir, tuple(sorted(prism_source_keys)),
        sigcap_dir, tuple(sorted(sigcap_source_keys)),
        tuple(providers_active), metric, float(sat_offset), tle_fp, len(frame),
    )
    _vis = visibility_columns(
        frame, _vis_key, tle_dir, tle_fp, float(sat_offset),
        tuple(satellites.DEFAULT_MASKS),
    )
    for c in sat_columns:
        frame[c] = _vis[c].to_numpy()

# Every sidebar filter except the drawn region - the map draws this so you can
# select a region with the context around it.
unregioned = flt.apply_filters(
    frame,
    flt.Filters(
        links=links,
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
# ...and the region on top. With no region drawn this is `unregioned` itself,
# not a second copy of it.
active = flt.apply_filters(
    unregioned,
    flt.Filters(regions=st.session_state["regions"], keep_ungeolocated=keep_ungeo),
)
del frame  # everything below reads `active` / `unregioned`


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

# Which export the numbers on screen come from - fixed per metric, so it is
# stated with the plots rather than offered as a filter.
_shown = set(active["source"].dropna().unique())
reading_note = (
    f"{metric_meta.label} is read from "
    + ", ".join(f"**{ALL_DATASETS_BY_KEY[k].label}**" for k in reading if k in _shown)
    + "."
)
st.caption(reading_note)

tab_labels = ["🗺️ Map", "📈 CDF curves", "🛰️ Cell identity", "📊 Statistics", "🧾 Data"]
if sat_enable:
    tab_labels.insert(1, "🛰️ D2C visibility")


# --------------------------------------------------------------------------- #
# Tabs - each one lives in qpviz/ui/tab_*.py and reads the shared state in ctx
# --------------------------------------------------------------------------- #

ctx = Ctx(
    mode=MODE,
    qp_dir=qp_dir,
    prism_dir=prism_dir,
    sigcap_dir=sigcap_dir,
    qp_present=qp_present,
    prism_present=prism_present,
    sigcap_present=sigcap_present,
    providers_active=providers_active,
    tech=tech,
    tech_value=tech_value,
    metric=metric,
    metric_meta=metric_meta,
    freq_label=freq_label,
    freq_prefix=freq_prefix,
    traffic=traffic,
    collections=collections,
    bands=bands,
    freqs=freqs,
    effective_pcis=effective_pcis,
    drop_unknown=drop_unknown,
    keep_ungeo=keep_ungeo,
    time_range=time_range,
    active=active,
    unregioned=unregioned,
    min_samples=min_samples,
    max_curve_points=max_curve_points,
    max_map_points=max_map_points,
    map_scale=map_scale,
    tiles=tiles,
    carto_key=carto_key,
    tle_dir=tle_dir,
    tle_fp=tle_fp,
    sat_mask=sat_mask,
    sat_offset=sat_offset,
    sat_active=sat_active,
    sat_vis_col=sat_vis_col,
    show_sat_kpi=show_sat_kpi,
)

# label -> renderer, in the order the tabs have always been drawn
TAB_RENDERERS = {
    "🗺️ Map": tab_map.render,
    "🛰️ D2C visibility": tab_satellites.render,
    "📈 CDF curves": tab_cdf.render,
    "🛰️ Cell identity": tab_cells.render,
    "📊 Statistics": tab_stats.render,
    "🧾 Data": tab_data.render,
}
# Only the open tab runs. Switching tabs reruns the script (on_change="rerun"),
# so a first visit builds the Map alone instead of every tab's frames at once,
# which is what took the hosted app over its memory limit.
for label, tab in zip(tab_labels, st.tabs(tab_labels, key="main_tab", on_change="rerun")):
    if tab.open:
        with tab:
            TAB_RENDERERS[label](ctx)
