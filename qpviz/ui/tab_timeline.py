"""Timeline analysis tab of the Starlink D2C explorer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import re
import streamlit as st

from qpviz import charts, filters as flt, loader, timeline
from qpviz.schema import DATASETS_BY_KEY, SIGCAP_DATASETS_BY_KEY, datasets_for
from qpviz.ui.common import load_frame, load_sigcap_frame
from qpviz.ui.context import Ctx


def render(ctx: Ctx) -> None:
    """Render the Timeline analysis tab."""
    MODE = ctx.mode
    collections = ctx.collections
    keep_ungeo = ctx.keep_ungeo
    qp_dir = ctx.qp_dir
    qp_present = ctx.qp_present
    sigcap_dir = ctx.sigcap_dir
    sigcap_present = ctx.sigcap_present
    tech_value = ctx.tech_value
    time_range = ctx.time_range
    traffic = ctx.traffic

    st.markdown(
        "**One phone or two, on the same spot.** SigCap sessions from "
        "different phones can run at the same time. This aligns them on one "
        "wall clock so you can see how each phone's throughput changes when a "
        "second phone starts using Starlink D2C, and whether throughput moves "
        "together when both phones camp on the **same PCI**. How many phones "
        "are \"using\" it at any instant is counted from the data (distinct "
        "device UUIDs carrying iperf / HTTP traffic in the same bin — a phone "
        "that is only logging its serving cell is not a contender), not from "
        "the file names."
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
            mcols[1].metric(
                "Max concurrent", f"{res.max_active}",
                help="Most phones carrying app traffic in one bin.",
            )
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
            st.session_state["_tl_qp_scope"] = []   # set below only when scoped
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
                        sd = d[c.replace("_mean", "_std")] if c.replace("_mean", "_std") in d.columns else None
                        d[c] = [
                            (f"{v:.2f} ± {s_:.2f}" if pd.notna(v) and pd.notna(s_)
                             else f"{v:.2f}" if pd.notna(v) else "—")
                            for v, s_ in zip(d[c], sd if sd is not None else [np.nan] * len(d))
                        ]
                d = d.drop(columns=[c for c in d.columns if c.endswith("_std")])
                if "delta_pct" in d.columns:
                    d["delta_pct"] = d["delta_pct"].map(
                        lambda v: f"{v:+.0f}%" if pd.notna(v) else "—"
                    )
                return d.rename(columns=rename)

            if res.has_concurrency:
                st.markdown("#### 1 phone vs 2 phones active")
                st.caption(
                    "Mean ± std throughput (Mbps) for each phone when it was "
                    "the only one carrying traffic vs when a second phone was "
                    "active at the same instant, on the detection grid "
                    f"({res.detect_bin} bins). Δ is the change under "
                    "contention. Downlink and uplink are compared separately "
                    "— pooling them would confound direction with contention."
                )
                dl = res.by_concurrency_dir["DL"]
                ul = res.by_concurrency_dir["UL"]
                both = res.by_concurrency_dir["both"]
                tabs_dir = st.tabs(["Downlink", "Uplink", "Both (pooled)"])
                for tb, df in zip(tabs_dir, (dl, ul, both)):
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
                    "Among the bins where both phones were carrying traffic, "
                    "mean ± std throughput (Mbps) when they shared one serving "
                    "PCI vs when they were on different PCIs."
                )
                tabs_pci = st.tabs(["Downlink", "Uplink", "Both (pooled)"])
                for tb, df in zip(tabs_pci, (res.by_shared_pci_dir["DL"],
                                             res.by_shared_pci_dir["UL"],
                                             res.by_shared_pci_dir["both"])):
                    with tb:
                        st.dataframe(
                            _fmt(
                                df,
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
