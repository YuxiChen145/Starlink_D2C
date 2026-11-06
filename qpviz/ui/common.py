"""Cached data access and small widgets shared by app.py and the tab modules.

Everything here is wrapped in ``st.cache_data`` or touches Streamlit widgets, so
it lives under ``qpviz.ui`` rather than in the pure-pandas ``qpviz`` modules.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import pandas as pd
import streamlit as st

from qpviz import loader, satellites


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


def align_qp_d2c(fr: pd.DataFrame, sigcap_dir: str,
                 sigcap_keys: Sequence[str]) -> pd.DataFrame:
    """Relabel QP D2C traffic from the SigCap reference, when SigCap is present."""
    if not sigcap_keys:
        return fr
    ref = sigcap_traffic_ref(sigcap_dir, tuple(sorted(sigcap_keys)))
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
    """Multiselect where an empty selection means 'everything'.

    The options cascade (collections -> band -> carrier -> PCI), so a value
    picked earlier can disappear from the list after an upstream change; it is
    pruned from the stored selection before the widget renders rather than
    lingering as an invisible filter.
    """
    if key in st.session_state:
        opts = set(options)
        kept = [v for v in st.session_state[key] if v in opts]
        if len(kept) != len(st.session_state[key]):
            st.session_state[key] = kept
    chosen = st.multiselect(
        label,
        options=list(options),
        default=[],
        key=key,
        help=help or "Leave empty to include all.",
        placeholder=f"All ({len(options)})",
    )
    return chosen
