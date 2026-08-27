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
        Metric("bler", "BLER", "%", higher_is_better=False),
        Metric("cqi", "CQI", "index"),
        Metric("tx_power", "Tx Power", "dBm"),
        Metric("pathloss", "DL Pathloss", "dB", higher_is_better=False),
    ]
}

#: Order the metric picker is shown in.
METRIC_ORDER: List[str] = [
    "rsrp",
    "rsrq",
    "rssi",
    "sinr",
    "mcs",
    "throughput",
    "bler",
    "cqi",
    "tx_power",
    "pathloss",
]

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

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
    "freq",
    "band",
    "tech",
    "link",
    "source",
    "source_label",
]


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

    @property
    def freq_label(self) -> str:
        return "NR-ARFCN" if self.tech == "NR" else "EARFCN"

    @property
    def has_geo(self) -> bool:
        return "lat" in self.dims and "lon" in self.dims

    def raw_columns(self) -> List[str]:
        return list(dict.fromkeys(list(self.dims.values()) + list(self.metrics.values())))


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
            "freq": "DL EARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={
            "rsrp": "RSRP",
            "rsrq": "RSRQ",
            "rssi": "RSSI (LTE)",
            "sinr": "SINR",
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
        metrics={"rsrp": "RSRP", "rsrq": "RSRQ", "rssi": "RSSI"},
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
            "bler": "BLER",
        },
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
            "tx_power": "PUSCH Tx Power",
            "pathloss": "DL Pathloss",
        },
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
            "freq": "DL NR-ARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={
            "mcs": "MCS",
            "throughput": "5G NR Net PDSCH Throughput",
            "sinr": "PDSCH SINR",
            "bler": "PDSCH BLER",
        },
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
            "freq": "UL NR-ARFCN",
            "band": "Band Number",
            "cell_type": "Cell Type",
        },
        metrics={
            "mcs": "MCS",
            "throughput": "5G NR Net PUSCH Throughput",
            "tx_power": "PUSCH Tx Power",
            "pathloss": "DL Pathloss",
        },
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
        ds = DATASETS_BY_KEY.get(k)
        if ds:
            present.update(ds.metrics)
    return [m for m in METRIC_ORDER if m in present]
