"""Cell identity tab of the Starlink D2C explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qpviz import charts, filters as flt, identity, loader
from qpviz.schema import (
    DATASETS_BY_KEY,
    NETWORKS,
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


#: the tree is always eNB, then sector, then PCI
LEVELS = identity.TREE_LEVELS
LEVEL_LABELS = [identity.label_for(level) for level in LEVELS]

#: the time views draw at most this many collections (the largest ones)
MAX_TIMELINE_PANELS = 4


def _networks(table: pd.DataFrame) -> list:
    """Networks in ``table``: D2C first, then terrestrial, then anything else."""
    present = list(dict.fromkeys(table["network"]))
    return [n for n in NETWORKS if n in present] + sorted(
        n for n in present if n not in NETWORKS
    )


def _keep_choice(key: str, options) -> None:
    """Forget a stored widget choice that the current filters no longer offer."""
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)


def _hierarchy_view(table: pd.DataFrame, mode: str) -> None:
    """A per-eNB summary of one network, and the tree of one eNB."""
    networks = _networks(table)
    row = st.columns([2, 3])
    if len(networks) > 1:
        _keep_choice("identity_network", networks)
        network = row[0].radio(
            "Network", networks, horizontal=True, key="identity_network",
            help="D2C and terrestrial cells are kept apart: each network has "
                 "its own table and its own trees.",
        )
    else:
        network = networks[0]
    order = row[1].radio(
        "Order eNBs by", list(identity.ENB_ORDERS),
        format_func=identity.ENB_ORDERS.get, horizontal=True,
        key="identity_enb_order",
        help="Most sectors first, or by eNB ID.",
    )

    part = table[table["network"] == network]
    summary = identity.enb_summary(part, order)
    sectors = part.groupby(["enb", "sector"], observed=True).ngroups
    st.caption(
        f"**{network}**: {len(summary):,} eNBs, {sectors:,} sectors and "
        f"{part['pci'].nunique():,} PCIs seen in "
        f"{part['dwell_s'].sum() / 60:,.0f} minutes of measurement."
    )
    st.dataframe(
        summary.rename(
            columns={
                "enb": "eNB ID", "sectors": "Sectors seen",
                "pcis": "PCIs seen", "minutes": "Minutes",
            }
        ),
        width="stretch",
        hide_index=True,
        height=min(38 + 35 * len(summary), 285),
        column_config={
            "eNB ID": st.column_config.NumberColumn(
                format="%d", help="For a 5G NR cell this is the gNB ID."
            ),
            "Minutes": st.column_config.NumberColumn(
                format="%.1f", help="Time this eNB was held."
            ),
        },
    )

    enbs = [int(v) for v in summary["enb"]]
    focus_key = f"identity_focus_{network}"
    _keep_choice(focus_key, enbs)
    focus = st.selectbox(
        "Show the tree of one eNB", options=enbs, key=focus_key,
        help="In the order of the table above. The tree shows every sector "
             "and PCI seen under this eNB.",
    )
    one = part[part["enb"] == focus]
    st.plotly_chart(
        charts.tree_figure(one, LEVELS, LEVEL_LABELS, mode=mode),
        width="stretch",
        key="identity_tree",
    )


def _structure_view(table: pd.DataFrame, mode: str) -> None:
    """How wide the tree is, D2C and terrestrial side by side."""
    st.caption(
        "How wide the tree is on each network: how many sectors an eNB "
        "showed, and how many PCIs a sector showed."
    )
    networks = _networks(table)
    summary = identity.network_summary(table).set_index("network").loc[networks]
    st.dataframe(
        summary.reset_index().rename(
            columns={
                "network": "Network",
                "enbs": "eNBs seen",
                "sectors": "Sectors seen",
                "pcis": "PCIs seen",
                "sectors_per_enb_median": "Sectors per eNB (median)",
                "sectors_per_enb_max": "Sectors per eNB (most)",
                "pcis_per_sector_median": "PCIs per sector (median)",
                "pcis_per_sector_max": "PCIs per sector (most)",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    counts = identity.fan_out_counts(table, LEVELS, group_by="network")
    dist = identity.fan_out_distribution(counts)
    # eNB to sector, then sector to PCI
    for parent_levels, child in identity.relationships(LEVELS)[:2]:
        label = identity.relationship_label(parent_levels, child)
        st.markdown(f"**{label}**")
        st.plotly_chart(
            charts.fan_out_figure(
                dist, label, y_title=identity.parents_label(parent_levels),
                groups=networks, mode=mode,
            ),
            width="stretch",
            key=f"structure_{child}",
        )


def _download(table: pd.DataFrame) -> None:
    """The one download of the tab: every eNB / sector / PCI path seen."""
    pretty = (
        table[["network", *LEVELS, "dwell_s", "samples"]]
        .sort_values(["network", *LEVELS])
        .rename(columns=dict(zip(LEVELS, LEVEL_LABELS)))
        .rename(
            columns={"network": "Network", "dwell_s": "Held (s)", "samples": "Samples"}
        )
    )
    st.download_button(
        f"Download all {len(pretty):,} cells seen (CSV)",
        pretty.round(1).to_csv(index=False).encode(),
        file_name="cell_identity.csv",
        mime="text/csv",
        key="identity_download",
    )


def _change_timeline_view(serving: pd.DataFrame, mode: str) -> None:
    st.caption(
        "**Serving-cell timeline** with **eNB ID on the y-axis** and one "
        "segment per Cell Id. A segment is **orange** when the **eNB** "
        "changed (a site/satellite change) and **blue** when only the "
        "**Cell Id** changed under the same eNB (a beam/sector change). Each "
        "segment is labelled with its **PCI above** and **sector below** "
        "(`s#`), and dotted lines connect consecutive cells so an eNB jump "
        "is visible. One panel per collection."
    )
    if "enb" not in identity.available_levels(serving):
        st.info(
            "This needs the connected radio to carry an eNB ID. It is not "
            "decoded in the current selection."
        )
        return
    segments = identity.serving_segments(serving)
    summary = identity.segment_summary(segments)
    if segments.empty:
        st.info("Not enough serving-cell samples to build segments.")
        return
    by_size = serving.groupby("collection", observed=True).size()
    all_colls = by_size.sort_values(ascending=False).index.tolist()
    chosen = all_colls[:MAX_TIMELINE_PANELS]
    if len(all_colls) > len(chosen):
        st.caption(
            f"Showing the {len(chosen)} largest of {len(all_colls)} collections. "
            "Pick collections in the sidebar to see others."
        )
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
            mode=mode,
        ),
        width="stretch",
    )


def _timeline_view(serving: pd.DataFrame, mode: str) -> None:
    serving_levels = identity.available_levels(serving)
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
        return
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
            runs, shown, identity.label_for(level), mode=mode, x_axis=x_axis
        ),
        width="stretch",
    )
    if len(shown) < runs["value"].nunique():
        st.caption(
            f"Showing the {len(shown)} longest-held of "
            f"{runs['value'].nunique()} identities."
        )


def render(ctx: Ctx) -> None:
    """Render the Cell identity tab. Returns early (instead of st.stop(),
    which would also blank every tab after this one) when no source carries
    cell identity for the current selection."""
    MODE = ctx.mode
    bands = ctx.bands
    collections = ctx.collections
    effective_pcis = ctx.effective_pcis
    freqs = ctx.freqs
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
            require=("pci",),
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
    if cells.empty or not identity.available_levels(cells):
        st.info("No cell-identity rows match the current filters.")
        return

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
            f"PRISM decoded scans ({n_prism:,} rows — every audible cell)"
        )
    st.caption("Cell identity from " + " + ".join(src_parts) + ".")

    # the tree views need the whole eNB / sector / PCI path; the serving-cell
    # time views need QP or SigCap rows
    table = identity.paths(cells)
    if not table.empty and "network" not in table.columns:
        table["network"] = "All"
    view_options = ["Hierarchy", "Structure"] + (
        ["Timeline", "Change timeline"] if not serving.empty else []
    )
    _keep_choice("identity_view", view_options)
    view = st.radio("View", view_options, horizontal=True, key="identity_view")

    if view in ("Hierarchy", "Structure"):
        if table.empty:
            st.info(
                "The tree needs cells with an eNB ID, a sector ID and a PCI. "
                "None match the current filters (SigCap reports the PCI only)."
            )
        elif view == "Hierarchy":
            _hierarchy_view(table, MODE)
            _download(table)
        else:
            _structure_view(table, MODE)
            _download(table)
    elif view == "Change timeline":
        _change_timeline_view(serving, MODE)
    else:
        _timeline_view(serving, MODE)
