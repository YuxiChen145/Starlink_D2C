"""D2C satellite visibility tab of the Starlink D2C explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from qpviz import charts, satellites
from qpviz.ui.common import distinct_sats, tle_summary
from qpviz.ui.context import Ctx


def render(ctx: Ctx) -> None:
    """Render the D2C satellite visibility tab."""
    MODE = ctx.mode
    active = ctx.active
    sat_active = ctx.sat_active
    sat_mask = ctx.sat_mask
    sat_offset = ctx.sat_offset
    sat_vis_col = ctx.sat_vis_col
    tle_dir = ctx.tle_dir
    tle_fp = ctx.tle_fp

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
