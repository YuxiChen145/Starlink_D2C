"""Schema definitions for the Qualipoc (QP) CSV exports.

Each raw CSV has its own column naming.  This module maps every file onto one
canonical schema so the rest of the app never has to know which file a row came
from.

Canonical dimension columns
---------------------------
time, collection, database, operator, model, serial, ran_config, cell_type,
lat, lon, pci, freq, band, tech, link, source, source_label

Canonical metric columns
------------------------
rsrp, rsrq, rssi, sinr, mcs, throughput, bler, cqi, tx_power, pathloss
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Metric:
    key: str
    label: str
    unit: str
    #: True when a *higher* value is better (used only for wording in the UI).
    higher_is_better: bool = True

    @property
    def axis_label(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label


METRICS: Dict[str, Metric] = {
    m.key: m
    for m in [
        Metric("rsrp", "RSRP", "dBm"),
        Metric("rsrq", "RSRQ", "dB"),
        Metric("rssi", "RSSI", "dBm"),
        Metric("sinr", "SINR", "dB"),
        Metric("mcs", "MCS", "index"),
        Metric("throughput", "Throughput", "kbps"),
        # Derived (computed in the loader from throughput + allocated RBs + SCS):
        Metric("spec_eff", "Spectral efficiency", "bps/Hz"),
        Metric("tput_per_rb", "Throughput per RB", "bits/RB"),
        # Derived from the RSRQ = N·RSRP/RSSI identity: N should be the RSSI
        # measurement bandwidth in RBs (25/50/100), but empirically it drifts.
        Metric("n_factor", "RSRQ N-factor", "RB"),
        Metric("bler", "BLER", "%", higher_is_better=False),
        Metric("cqi", "CQI", "index"),
        Metric("tx_power", "Tx Power", "dBm"),
        Metric("pathloss", "DL Pathloss", "dB", higher_is_better=False),
    ]
}

#: Metrics offered in the picker, in display order.
METRIC_ORDER: List[str] = [
    "rsrp",
    "rsrq",
    "rssi",
    "sinr",
    "mcs",
    "throughput",
    "spec_eff",
    "tput_per_rb",
    "bler",
    "n_factor",
]

#: metrics computed in the loader rather than read from a raw column; their
#: value in a dataset's ``metrics`` map is this sentinel, not a column name
DERIVED_METRIC = "@derived"

#: Still parsed, cached and filterable, but not offered in the metric picker
#: for now. Moving a key back into METRIC_ORDER is all it takes to restore it -
#: the datasets already map the underlying columns.
HIDDEN_METRICS: List[str] = [
    "cqi",
    "tx_power",
    "pathloss",
]

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

#: canonical cell-identity columns (see qpviz.identity)
IDENTITY_COLUMNS: List[str] = ["pci", "enb", "sector", "cell_id", "ssb"]

DIMENSION_COLUMNS: List[str] = [
    "time",
    "collection",
    "database",
    "operator",
    "model",
    "serial",
    "ran_config",
    "cell_type",
    "lat",
    "lon",
    "pci",
    "enb",
    "sector",
    "cell_id",
    "ssb",
    "freq",
    "band",
    "tech",
    "link",
    "source",
    "source_label",
    "provider",
]

#: measurement sources the platform can pool.  "QP" is the Qualipoc phone
#: (connected-mode drive/walk logs); "PRISM" is the PRISM SDR scanner, which
#: sweeps every cell it can hear rather than logging one serving cell.
PROVIDERS: List[str] = ["QP", "PRISM"]


@dataclass(frozen=True)
class Dataset:
    """One QP CSV export and how its columns map onto the canonical schema."""

    key: str
    filename: str
    label: str
    tech: str  # "LTE" | "NR"
    link: str  # "DL" | "UL" | "-"
    #: canonical dimension name -> raw CSV column name
    dims: Dict[str, str]
    #: canonical metric name -> raw CSV column name
    metrics: Dict[str, str]
    #: extra notes surfaced in the UI
    note: str = ""
    #: which measurement source this export belongs to (see PROVIDERS)
    provider: str = "QP"
    #: PRISM exports are read by globbing a folder (many single-tech files),
    #: not by matching one fixed filename like the QP exports
    is_glob: bool = False
    #: raw column giving the RBs allocated averaged over *all* slots/subframes
    #: (incl. idle) - the denominator for the derived spectral-efficiency and
    #: throughput-per-RB metrics. ``None`` when the export can't supply them.
    rb_all_col: Optional[str] = None
    #: raw column giving the subcarrier spacing in kHz (NR); ``None`` -> 15 (LTE)
    scs_col: Optional[str] = None

    @property
    def freq_label(self) -> str:
        return "NR-ARFCN" if self.tech == "NR" else "EARFCN"

    @property
    def has_geo(self) -> bool:
        return "lat" in self.dims and "lon" in self.dims

    def raw_columns(self) -> List[str]:
        cols = list(self.dims.values()) + [
            v for v in self.metrics.values() if v != DERIVED_METRIC
        ]
        if self.rb_all_col:
            cols.append(self.rb_all_col)
        if self.scs_col:
            cols.append(self.scs_col)
        return list(dict.fromkeys(cols))


_COMMON = {
    "time": "Time",
    "collection": "Collection",
    "database": "Database",
    "operator": "Operator",
    "model": "Model",
    "serial": "Serial Number",
    "ran_config": "RAN Configuration",
}

_GEO = {"lat": "Latitude", "lon": "Longitude"}


DATASETS: List[Dataset] = [
    Dataset(
        key="lte_radio",
        filename="LTE Radio [connected] - Limited.csv",
        label="LTE Radio [connected]",
        tech="LTE",
        link="DL",
        dims={
            **_COMMON,
            **_GEO,
            "pci": "PCI LTE",
            "enb_sector": "eNB-Id / Sector Id",
            "cell_id": "Cell Id",
            "freq": "DL EARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={
            "rsrp": "RSRP",
            "rsrq": "RSRQ",
            "rssi": "RSSI (LTE)",
            "sinr": "SINR",
            "n_factor": DERIVED_METRIC,
        },
    ),
    Dataset(
        key="lte_neighbor",
        filename="LTE Radio Neighbor - Limited.csv",
        label="LTE Radio Neighbor",
        tech="LTE",
        link="DL",
        dims={
            **_COMMON,
            **_GEO,
            "pci": "PCI LTE",
            "freq": "DL EARFCN",
            "band": "Band Number",
        },
        metrics={
            "rsrp": "RSRP", "rsrq": "RSRQ", "rssi": "RSSI",
            "n_factor": DERIVED_METRIC,
        },
        note="Neighbour cells; PCI/EARFCN are often blank in the export.",
    ),
    Dataset(
        key="nr_radio",
        filename="NR Radio [connected] - Limited.csv",
        label="NR Radio [connected]",
        tech="NR",
        link="DL",
        dims={
            **_COMMON,
            **_GEO,
            "pci": "PCI 5G NR",
            "enb": "NR gNB-Id",
            "cell_id": "Cell Id",
            "ssb": "SSB Index",
            "freq": "DL NR-ARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={"rsrp": "SS-RSRP", "rsrq": "SS-RSRQ", "sinr": "SS-SINR"},
    ),
    Dataset(
        key="lte_pdsch_pc",
        filename="LTE PDSCH [per-carrier] - Limited.csv",
        label="LTE PDSCH [per-carrier]",
        tech="LTE",
        link="DL",
        dims={
            **_COMMON,
            **_GEO,
            "pci": "PCI LTE",
            "freq": "DL EARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={
            "mcs": "MCS",
            "throughput": "LTE Net PDSCH Throughput",
            "spec_eff": DERIVED_METRIC,
            "tput_per_rb": DERIVED_METRIC,
            "bler": "BLER",
        },
        rb_all_col="PDSCH Avg RBs all SF",  # LTE SCS is 15 kHz -> scs_col=None
    ),
    Dataset(
        key="lte_pdsch_agg",
        filename="LTE PDSCH [aggregated] - Limited.csv",
        label="LTE PDSCH [aggregated]",
        tech="LTE",
        link="DL",
        dims={
            **_COMMON,
            **_GEO,
            "pci": "PCI LTE",
            "freq": "DL EARFCN",
            "band": "PCell Band",
        },
        metrics={
            "mcs": "MCS",
            "throughput": "LTE Net PDSCH Throughput",
            "bler": "BLER",
        },
        note="Carrier-aggregated totals - throughput sums all component carriers.",
    ),
    Dataset(
        key="lte_pusch_pc",
        filename="LTE PUSCH [per-carrier] - Limited.csv",
        label="LTE PUSCH [per-carrier]",
        tech="LTE",
        link="UL",
        dims={
            **_COMMON,
            **_GEO,
            "pci": "PCI LTE",
            "freq": "UL EARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={
            "mcs": "MCS",
            "throughput": "LTE Net PUSCH Throughput",
            "spec_eff": DERIVED_METRIC,
            "tput_per_rb": DERIVED_METRIC,
            "tx_power": "PUSCH Tx Power",
            "pathloss": "DL Pathloss",
        },
        rb_all_col="PUSCH Avg RBs all SF",
    ),
    Dataset(
        key="lte_pucch",
        filename="LTE PUCCH - Limited.csv",
        label="LTE PUCCH",
        tech="LTE",
        link="UL",
        dims={
            **_COMMON,
            "pci": "PCI LTE",
            "enb_sector": "eNB-Id / Sector Id",
            "freq": "DL EARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={"cqi": "CQI0", "tx_power": "TxPower"},
        note="No GPS columns in this export - excluded from the map and from any drawn-region filter.",
    ),
    Dataset(
        key="nr_pdsch_pc",
        filename="NR PDSCH [per-carrier] - Limited.csv",
        label="NR PDSCH [per-carrier]",
        tech="NR",
        link="DL",
        dims={
            **_COMMON,
            **_GEO,
            "pci": "PCI 5G NR",
            "ssb": "SSB Index",
            "freq": "DL NR-ARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={
            "mcs": "MCS",
            "throughput": "5G NR Net PDSCH Throughput",
            "spec_eff": DERIVED_METRIC,
            "tput_per_rb": DERIVED_METRIC,
            "sinr": "PDSCH SINR",
            "bler": "PDSCH BLER",
        },
        rb_all_col="PDSCH Avg RBs all Slots",
        scs_col="DL Subcarrier Spacing",
    ),
    Dataset(
        key="nr_pdsch_agg",
        filename="NR PDSCH [aggregated] - Limited.csv",
        label="NR PDSCH [aggregated]",
        tech="NR",
        link="DL",
        dims={
            **_COMMON,
            **_GEO,
            "pci": "PCI 5G NR",
            "freq": "DL NR-ARFCN",
            "band": "PCell Band",
        },
        metrics={
            "mcs": "MCS",
            "throughput": "5G NR Net PDSCH Throughput",
            "bler": "PDSCH BLER",
        },
        note="Carrier-aggregated totals - throughput sums all component carriers.",
    ),
    Dataset(
        key="nr_pusch_pc",
        filename="NR PUSCH [per-carrier] - Limited.csv",
        label="NR PUSCH [per-carrier]",
        tech="NR",
        link="UL",
        dims={
            **_COMMON,
            **_GEO,
            "pci": "PCI 5G NR",
            "ssb": "SSB Index",
            "freq": "UL NR-ARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={
            "mcs": "MCS",
            "throughput": "5G NR Net PUSCH Throughput",
            "spec_eff": DERIVED_METRIC,
            "tput_per_rb": DERIVED_METRIC,
            "tx_power": "PUSCH Tx Power",
            "pathloss": "DL Pathloss",
        },
        rb_all_col="PUSCH Avg RBs all Slots",
        scs_col="UL Subcarrier Spacing",
    ),
]

DATASETS_BY_KEY: Dict[str, Dataset] = {d.key: d for d in DATASETS}


def datasets_with_metric(metric: str, tech: Optional[str] = None) -> List[Dataset]:
    """Datasets that carry ``metric`` (optionally restricted to one technology)."""
    out = [d for d in DATASETS if metric in d.metrics]
    if tech in ("LTE", "NR"):
        out = [d for d in out if d.tech == tech]
    return out


#: UI label -> the ``tech`` value stored on each dataset (``None`` = no filter)
TECHNOLOGIES: Dict[str, Optional[str]] = {"All": None, "LTE": "LTE", "5G NR": "NR"}


def datasets_for(
    metric: str, tech: Optional[str] = None, available: Optional[List[str]] = None
) -> List[Dataset]:
    """Every export that can supply ``metric`` for ``tech``.

    This is what replaces picking datasets by hand: the technology and the
    metric together determine which files are read.
    """
    out = datasets_with_metric(metric, tech)
    if available is not None:
        keep = set(available)
        out = [d for d in out if d.key in keep]
    return out


def metrics_available(dataset_keys: List[str]) -> List[str]:
    """Metric keys present in at least one of ``dataset_keys``, in display order."""
    present = set()
    for k in dataset_keys:
        ds = ALL_DATASETS_BY_KEY.get(k)
        if ds:
            present.update(ds.metrics)
    return [m for m in METRIC_ORDER if m in present]


# --------------------------------------------------------------------------- #
# PRISM (SDR scanner) datasets
# --------------------------------------------------------------------------- #
#
# PRISM is a software-defined-radio scanner.  Unlike the Qualipoc phone, which
# logs the one cell it is connected to, PRISM sweeps and reports every LTE / NR
# cell it can hear, so it only carries the radio-signal metrics (RSRP, RSRQ,
# RSSI, SINR) - never throughput / MCS / BLER, and it has no serving-cell vs
# neighbour distinction.  Every PRISM row is downlink.
#
# The exports also differ structurally from QP: there is no "Collection" column
# (the collection is encoded in the file name), the columns are lower-case, and
# there is one file per location + carrier rather than one file per metric
# family.  So each PRISM technology is one *glob-based* dataset - the loader
# reads every matching file in the PRISM folder and routes it by its own
# ``tech`` column.

#: columns every PRISM export shares
_PRISM_COMMON = {
    "time": "system_date_time",
    "lat": "latitude",
    "lon": "longitude",
}

PRISM_DATASETS: List[Dataset] = [
    Dataset(
        key="prism_lte",
        filename="PRISM LTE scan (*.csv)",
        label="PRISM LTE",
        tech="LTE",
        link="DL",
        provider="PRISM",
        is_glob=True,
        dims={
            **_PRISM_COMMON,
            "pci": "physical_cell_id",
            "enb": "e_node_b",
            "sector": "sector_id",
            "cell_id": "cell_identity",
            "freq": "earfcn",
            "band": "band",
        },
        metrics={
            "rsrp": "rsrp", "rsrq": "rsrq", "rssi": "rssi", "sinr": "sinr",
            "n_factor": DERIVED_METRIC,
        },
        note="PRISM SDR scan - every audible LTE cell, not just the serving one.",
    ),
    Dataset(
        key="prism_nr",
        filename="PRISM NR scan (*.csv)",
        label="PRISM NR",
        tech="NR",
        link="DL",
        provider="PRISM",
        is_glob=True,
        dims={
            **_PRISM_COMMON,
            "pci": "physical_cell_id",
            "cell_id": "cell_identity",
            "ssb": "issb",
            "freq": "nrarfcn",
            "band": "band",
        },
        metrics={
            "rsrp": "ss_rsrp",
            "rsrq": "ss_rsrq",
            "rssi": "ss_rssi",
            "sinr": "ss_sinr",
            "n_factor": DERIVED_METRIC,
        },
        note="PRISM SDR scan - every audible 5G NR cell, not just the serving one.",
    ),
]

PRISM_DATASETS_BY_KEY: Dict[str, Dataset] = {d.key: d for d in PRISM_DATASETS}

#: every dataset the platform knows about, QP and PRISM together - use this for
#: label / metadata look-ups by key; the metric->dataset resolvers stay split
#: so the two sources never get pooled by accident.
ALL_DATASETS_BY_KEY: Dict[str, Dataset] = {**DATASETS_BY_KEY, **PRISM_DATASETS_BY_KEY}


def prism_datasets_for(
    metric: str, tech: Optional[str] = None, available: Optional[List[str]] = None
) -> List[Dataset]:
    """PRISM exports that can supply ``metric`` for ``tech`` (and are present)."""
    out = [d for d in PRISM_DATASETS if metric in d.metrics]
    if tech in ("LTE", "NR"):
        out = [d for d in out if d.tech == tech]
    if available is not None:
        keep = set(available)
        out = [d for d in out if d.key in keep]
    return out
