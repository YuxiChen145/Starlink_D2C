"""The state every tab reads, built once per rerun by app.py."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd

from qpviz.schema import Metric


@dataclass(frozen=True)
class Ctx:
    """Sidebar choices plus the filtered data, passed to each ``tab_*.render``."""

    #: "light" | "dark" - the Streamlit theme, for chart colours
    mode: str

    # -- data sources ---------------------------------------------------------
    qp_dir: str
    prism_dir: str
    sigcap_dir: str
    #: dataset keys found in each folder
    qp_present: List[str]
    prism_present: List[str]
    sigcap_present: List[str]
    #: measurement sources selected (QP / PRISM / SIGCAP)
    providers_active: List[str]

    # -- metric ---------------------------------------------------------------
    tech: str  # sidebar label ("All", "LTE", "5G NR")
    tech_value: Optional[str]  # schema value for ``tech``
    metric: str
    metric_meta: Metric
    freq_label: str  # "EARFCN", "NR-ARFCN" or both
    freq_prefix: str

    # -- filters --------------------------------------------------------------
    traffic: Sequence[str]
    collections: Sequence[str]
    bands: Sequence[Any]
    freqs: Sequence[Any]
    effective_pcis: Sequence[Any]
    drop_unknown: bool
    keep_ungeo: bool
    time_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]]

    # -- filtered data --------------------------------------------------------
    #: every filter applied, including the drawn map region
    active: pd.DataFrame
    #: every filter except the drawn region (the map can show context around it)
    unregioned: pd.DataFrame

    # -- display options ------------------------------------------------------
    min_samples: int
    max_curve_points: int
    max_map_points: int
    map_scale: str
    tiles: str
    carto_key: Optional[str]

    # -- D2C satellite visibility ---------------------------------------------
    tle_dir: str
    tle_fp: str
    sat_mask: int
    sat_offset: float
    sat_active: bool
    sat_vis_col: Optional[str]
    show_sat_kpi: bool
