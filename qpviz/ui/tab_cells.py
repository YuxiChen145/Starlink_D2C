"""Cell identity tab of the Starlink D2C explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qpviz import charts, filters as flt, identity, loader, stats
from qpviz.schema import (
    DATASETS_BY_KEY,
    PRISM_DATASETS_BY_KEY,
    SIGCAP_DATASETS_BY_KEY,
)
from qpviz.ui.common import (
    align_qp_d2c,
    load_frame,
    load_prism_frame,
    load_sigcap_frame,
)
from qpviz.ui.context import Ctx


def render(ctx: Ctx) -> None:
    """Render the Cell identity tab. Returns early (instead of st.stop(),
    which would also blank every tab after this one) when no source carries
    cell identity for the current selection."""
    MODE = ctx.mode
    bands = ctx.bands
    collections = ctx.collections
    drop_unknown = ctx.drop_unknown
    effective_pcis = ctx.effective_pcis
    freqs = ctx.freqs
    keep_ungeo = ctx.keep_ungeo
    prism_dir = ctx.prism_dir
    prism_present = ctx.prism_present
    providers_active = ctx.providers_active
    qp_dir = ctx.qp_dir
    qp_present = ctx.qp_present
    sigcap_dir = ctx.sigcap_dir
    sigcap_present = ctx.sigcap_present
    tech_value = ctx.tech_value
    time_range = ctx.time_range
    traffic = ctx.traffic

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
        return

    _id_parts = []
    if qp_radio_keys:
        _id_parts.append(load_frame(qp_dir, qp_radio_keys))
    if prism_id_keys:
        _id_parts.append(load_prism_frame(prism_dir, prism_id_keys))
    if sigcap_id_keys:
        _id_parts.append(load_sigcap_frame(sigcap_dir, sigcap_id_keys))
    radio = loader.concat_frames(_id_parts)
    radio = align_qp_d2c(radio, sigcap_dir, sigcap_present)

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
