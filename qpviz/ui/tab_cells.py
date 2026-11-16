"""Cell identity tab of the Starlink D2C explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qpviz import charts, filters as flt, identity, loader
from qpviz.schema import (
    DATASETS_BY_KEY,
    NETWORKS,
    PRISM_DATASETS_BY_KEY,
)
from qpviz.ui.common import align_qp_d2c, load_frame, load_prism_frame
from qpviz.ui.context import Ctx


#: the tree is always eNB, then sector, then PCI
LEVELS = identity.TREE_LEVELS
LEVEL_LABELS = [identity.label_for(level) for level in LEVELS]

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


def render(ctx: Ctx) -> None:
    """Render the Cell identity tab. Returns early (instead of st.stop(),
    which would also blank every tab after this one) when no source carries
    cell identity for the current selection."""
    providers_active = ctx.providers_active
    tech_value = ctx.tech_value

    # Cell identity from the sources that decode the whole eNB / sector / PCI
    # path. The QP connected-mode Radio exports give the serving cell; PRISM is
    # an SDR scanner whose decoded identities are every audible cell. SigCap
    # reports the PCI only, which cannot be placed in the tree.
    qp_radio_keys = tuple(
        k
        for k in sorted(ctx.qp_present)
        if k in ("lte_radio", "nr_radio")
        and (tech_value is None or DATASETS_BY_KEY[k].tech == tech_value)
    ) if "QP" in providers_active else ()
    prism_id_keys = tuple(
        k
        for k in sorted(ctx.prism_present)
        if tech_value is None or PRISM_DATASETS_BY_KEY[k].tech == tech_value
    ) if "PRISM" in providers_active else ()

    if not qp_radio_keys and not prism_id_keys:
        st.info(
            "Cell identity needs the QP Radio [connected] export or a PRISM "
            "scan that decodes eNB / PCI. Neither is available for this "
            "selection."
        )
        return

    parts = []
    if qp_radio_keys:
        parts.append(load_frame(ctx.qp_dir, qp_radio_keys))
    if prism_id_keys:
        parts.append(load_prism_frame(ctx.prism_dir, prism_id_keys))
    radio = loader.concat_frames(parts)
    radio = align_qp_d2c(radio, ctx.sigcap_dir, ctx.sigcap_present)

    def _intersecting(values, column):
        """Only carry a filter across if it means anything in the identity frame."""
        if not values:
            return []
        present_values = set(loader.options(radio, column))
        return [v for v in values if v in present_values]

    ident = flt.apply_filters(
        radio,
        flt.Filters(
            collections=ctx.collections,
            traffic=ctx.traffic,
            bands=_intersecting(ctx.bands, "band"),
            freqs=_intersecting(ctx.freqs, "freq"),
            pcis=_intersecting(ctx.effective_pcis, "pci"),
            regions=st.session_state["regions"],
            time_range=ctx.time_range,
            require=("pci",),
        ),
    )
    # QP marks the serving cell with a cell_type: keep only those rows (an
    # aggregated carrier comes and goes on its own). PRISM scans have no
    # serving concept, so every decoded row is kept.
    if "provider" in ident.columns:
        is_prism = (ident["provider"].astype("string") == "PRISM").to_numpy()
        cells = pd.concat(
            [identity.serving_only(ident.loc[~is_prism]), ident.loc[is_prism]],
            ignore_index=True,
        )
    else:
        is_prism = None
        cells = identity.serving_only(ident)

    table = identity.paths(cells)
    if table.empty:
        st.info(
            "No cell with an eNB ID, a sector ID and a PCI matches the current "
            "filters."
        )
        return
    if "network" not in table.columns:
        table["network"] = "All"

    n_prism = int((cells["provider"] == "PRISM").sum()) if is_prism is not None else 0
    n_qp = len(cells) - n_prism
    src_parts = []
    if n_qp and qp_radio_keys:
        src_parts.append(
            f"QP serving cell ({n_qp:,} samples, "
            + ", ".join(DATASETS_BY_KEY[k].label for k in qp_radio_keys) + ")"
        )
    if n_prism:
        src_parts.append(
            f"PRISM decoded scans ({n_prism:,} rows — every audible cell)"
        )
    st.caption("Cell identity from " + " + ".join(src_parts) + ".")

    view = st.radio(
        "View", ["Hierarchy", "Structure"], horizontal=True, key="identity_view"
    )
    if view == "Hierarchy":
        _hierarchy_view(table, ctx.mode)
    else:
        _structure_view(table, ctx.mode)
    _download(table)
