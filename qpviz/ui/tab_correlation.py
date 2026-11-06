"""Correlation tab of the Starlink D2C explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qpviz import charts, correlate, filters as flt, loader
from qpviz.schema import (
    DATASETS_BY_KEY,
    METRICS,
    SIGCAP_DATASETS_BY_KEY,
    datasets_for,
    metrics_available,
    sigcap_datasets_for,
)
from qpviz.ui.common import load_frame, load_sigcap_frame
from qpviz.ui.context import Ctx


def render(ctx: Ctx) -> None:
    """Render the Correlation tab."""
    MODE = ctx.mode
    keep_ungeo = ctx.keep_ungeo
    qp_dir = ctx.qp_dir
    qp_present = ctx.qp_present
    sigcap_dir = ctx.sigcap_dir
    sigcap_present = ctx.sigcap_present
    tech = ctx.tech
    tech_value = ctx.tech_value
    time_range = ctx.time_range
    traffic = ctx.traffic

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
        dir_opts = [d for d in ("DL", "UL") if d in set(loader.options(sig_cf, "link"))
                    or d in set(loader.options(qp_cf, "link"))] or ["DL", "UL"]
        corr_link = fc2.selectbox(
            "Direction", dir_opts + ["Both (pooled)"], index=0, key="corr_dir",
            help="QP PHY exports are per direction (PDSCH = DL, PUSCH = UL), so "
                 "both sources are restricted to one direction before pairing. "
                 "SigCap rows with no direction (ping) are kept. 'Both' pools "
                 "DL and UL into one series - only meaningful for direction-"
                 "free PHY metrics such as RSRP.",
        )
        corr_link = None if corr_link == "Both (pooled)" else corr_link
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
        result = correlate.correlate(
            qp_cf, phy_metric, sig_cf, app_metric,
            bin=bin_choice, qp_agg=agg_choice, sig_agg=agg_choice,
            link=corr_link,
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
                f"at least {correlate.MIN_PAIRS} are needed before a coefficient "
                "means anything. Try a finer time bin or widen the time window."
            )
        else:
            if result.mixed_directions:
                st.warning(
                    "DL and UL QP rows are pooled into one series for "
                    f"**{phy_label}** — pick a direction unless the metric is "
                    "direction-free."
                )
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
