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
rsrp, rsrq, rssi, sinr, n_factor, tput_dl, tput_dl_cc, tput_ul, tput_ul_cc,
mcs, bler, spec_eff, sf_usage, rb, rb_occ, rtt

Each metric reads one fixed Qualipoc export (the PRISM scanner and SigCap add
their own serving-cell readings of the signal metrics): the signal metrics the
Radio [connected] export, the all-carriers downlink throughput the PDSCH
[aggregated] export, and everything else the per-carrier PDSCH / PUSCH exports.
No metric is pooled from two exports of the same phone.
"""

from __future__ import annotations

import hashlib
import json
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
        # Throughput is four metrics, not one: downlink and uplink, each as the
        # total over every carrier the phone used and as one carrier's share.
        # Each reads one fixed export (see Dataset.metrics):
        #   tput_dl     PDSCH [aggregated]   - one row per timestamp, all carriers
        #   tput_dl_cc  PDSCH [per-carrier]  - one row per carrier
        #   tput_ul_cc  PUSCH [per-carrier]  - one row per carrier
        #   tput_ul     PUSCH [per-carrier], summed over the carriers of each
        #               timestamp (Qualipoc has no aggregated uplink export)
        Metric("tput_dl", "DL throughput, all carriers", "Mbps"),
        Metric("tput_dl_cc", "DL throughput, per carrier", "Mbps"),
        Metric("tput_ul", "UL throughput, all carriers", "Mbps"),
        Metric("tput_ul_cc", "UL throughput, per carrier", "Mbps"),
        # Ping round-trip time, logged by SigCap (the only source that pings):
        Metric("rtt", "Ping RTT", "ms", higher_is_better=False),
        # Derived (computed in the loader from throughput + allocated RBs + SCS):
        Metric("spec_eff", "Spectral efficiency", "bps/Hz"),
        # PHY resource allocation, from the per-carrier exports:
        #   sf_usage = share of the available sub-frames (NR: slots) in which
        #              the phone was scheduled at all ("SF Utilization").
        #   rb       = Avg RBs *per SF* - the grant size averaged over the USED
        #              sub-frames/slots only (reaches the carrier's RB ceiling).
        #   rb_occ   = Avg RBs *all SF* - averaged over ALL sub-frames incl.
        #              idle ones, so rb_occ = rb x sf_usage.
        # A shared cell splitting its RBs between two UEs should show fewer RBs.
        Metric("sf_usage", "Subframe usage", "%"),
        Metric("rb", "RBs per used subframe", "RB"),
        Metric("rb_occ", "RBs per subframe, all subframes", "RB"),
        # Derived from the RSRQ = N·RSRP/RSSI identity: N should be the RSSI
        # measurement bandwidth in RBs (25/50/100), but empirically it drifts.
        Metric("n_factor", "RSRQ N-factor", "RB"),
        Metric("bler", "BLER", "%", higher_is_better=False),
    ]
}

#: Metrics offered in the picker, in display order.
METRIC_ORDER: List[str] = [
    "rsrp",
    "rsrq",
    "rssi",
    "sinr",
    "n_factor",
    "tput_dl",
    "tput_dl_cc",
    "tput_ul",
    "tput_ul_cc",
    "mcs",
    "bler",
    "spec_eff",
    "sf_usage",
    "rb",
    "rb_occ",
    "rtt",
]

#: the throughput metrics, as a group (the raw exports log them in kbps)
THROUGHPUT_METRICS = ("tput_dl", "tput_dl_cc", "tput_ul", "tput_ul_cc")

#: metrics computed in the loader rather than read from a raw column; their
#: value in a dataset's ``metrics`` map is this sentinel, not a column name
DERIVED_METRIC = "@derived"

#: rows of one phone that share a timestamp are the carriers of one sample
CARRIER_GROUP = ("collection", "serial", "time")

# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

QP_TEST_NAME_COL = "Test Name"

DIMENSION_COLUMNS: List[str] = [
    "time",
    "collection",
    "database",
    "operator",
    "model",
    "serial",
    "device",
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
    "test_type",
    "traffic",
    "network",
]

#: The traffic types the platform shows, in the order of the Traffic filter.
#: ``Mixed`` is a row of a session that ran several tests (or both directions of
#: one) and that cannot be put down to a single one of them.
TRAFFIC_ORDER: List[str] = [
    "Idle", "HTTP UL", "HTTP DL", "iperf DL", "iperf UL", "Mixed",
]

#: Labels a row can also carry but the Traffic filter does not offer: a ping-
#: or Ookla-only test, and a session whose name says nothing about its traffic.
#: Those rows are in the data whenever no traffic type is picked.
TRAFFIC_UNLISTED: List[str] = ["Ping", "Ookla", "Other"]

_TRAFFIC_APPS = ("iperf", "http", "ping", "ookla", "idle")
_APP_LABEL = {"iperf": "iperf", "http": "HTTP"}


def _name_apps(name) -> tuple:
    """``(apps, raw)``: the traffic keywords in a session name, lower-cased."""
    if name is None:
        return (), ""
    raw = str(name).lower().replace(" ", "")
    return tuple(a for a in _TRAFFIC_APPS if a in raw), raw


def traffic_label(name) -> str:
    """Classify a collection / session name into a single traffic type.

    The measurement campaign encodes the generated traffic in the collection
    name - ``iperfDL`` / ``iperfUL`` / ``httpDL`` / ``httpUL`` / ``Ookla`` /
    ``Ping`` / ``IDLE`` - so we read it back into one label per row:

    * one test in one direction -> ``iperf DL`` / ``iperf UL`` / ``HTTP DL`` /
      ``HTTP UL``; an idle session -> ``Idle``;
    * several tests in one session (``PingiperfULDL``), both directions of one
      (``iperfULDL``), or an iperf / HTTP session whose name gives no direction
      (``_http``) -> ``Mixed``. The loader then splits those per row by which
      channel was carrying the transfer (see :func:`traffic_app`);
    * a ping- or Ookla-only session -> ``Ping`` / ``Ookla``; no recognisable
      keyword -> ``Other``.
    """
    apps, raw = _name_apps(name)
    if not apps:
        return "Other"
    if len(apps) > 1:
        return "Mixed"
    app = apps[0]
    if app in _APP_LABEL:
        seg = raw.split(app, 1)[1][:6]
        dl, ul = "dl" in seg, "ul" in seg
        if dl == ul:  # both directions, or none named
            return "Mixed"
        return f"{_APP_LABEL[app]} {'DL' if dl else 'UL'}"
    return {"ping": "Ping", "ookla": "Ookla", "idle": "Idle"}[app]


def traffic_app(name) -> Optional[str]:
    """The one transfer test a session name mentions: ``"iperf"`` or ``"HTTP"``.

    Used to name the active rows of a ``Mixed`` session: in ``PinghttpULDL`` a
    row carrying a downlink transfer is ``HTTP DL``, not iperf. ``None`` when
    the name mentions neither or both, so the transfer cannot be named.
    """
    apps, _ = _name_apps(name)
    found = [a for a in apps if a in _APP_LABEL]
    return _APP_LABEL[found[0]] if len(found) == 1 else None


# --------------------------------------------------------------------------- #
# D2C vs terrestrial
# --------------------------------------------------------------------------- #

#: The Starlink Direct-to-Cell carrier, per technology: LTE Band 25, downlink
#: EARFCN 8665; the uplink exports record its paired uplink EARFCN, 26665. A
#: row is D2C because of the carrier it was measured on, not because of what
#: its collection is called: sessions named D2C also hold rows on terrestrial
#: carriers the phone fell back to.
D2C_CARRIERS: Dict[str, tuple] = {"LTE": (8665, 26665)}

NETWORK_D2C = "D2C"
NETWORK_TERRESTRIAL = "Terrestrial"
#: display order
NETWORKS: List[str] = [NETWORK_D2C, NETWORK_TERRESTRIAL]


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
    #: which measurement source this export belongs to: "QP" is the Qualipoc
    #: phone (connected-mode drive/walk logs); "PRISM" is the PRISM SDR
    #: scanner, which sweeps every cell it can hear rather than logging one
    #: serving cell; "SIGCAP" is the SigCap logger on the same phone, which
    #: runs the iperf/HTTP/ping tests and adds ping RTT.
    provider: str = "QP"
    #: raw column giving the RBs allocated averaged over *all* slots/subframes
    #: (incl. idle) - the denominator for the derived spectral-efficiency
    #: metric, and the ``rb_occ`` metric. ``None`` when the export can't supply
    #: them.
    rb_all_col: Optional[str] = None
    #: raw column giving the RBs averaged over *scheduled* slots/subframes only
    #: (the grant size) - the ``rb`` metric. ``None`` when unavailable.
    rb_sched_col: Optional[str] = None
    #: raw column giving the subcarrier spacing in kHz (NR); ``None`` -> 15 (LTE)
    scs_col: Optional[str] = None
    #: extra raw CSV columns to read beyond dims/metrics - used to synthesise
    #: values before normalisation (e.g. SigCap reads the serving tech to route
    #: each row and the iperf/HTTP columns to label its traffic)
    extra_cols: List[str] = field(default_factory=list)

    @property
    def freq_label(self) -> str:
        return "NR-ARFCN" if self.tech == "NR" else "EARFCN"

    @property
    def has_geo(self) -> bool:
        return "lat" in self.dims and "lon" in self.dims

    @property
    def tput_metric(self) -> Optional[str]:
        """The throughput metric this export logs in a raw column (the
        per-carrier one where the all-carriers figure is derived), if any."""
        for key in THROUGHPUT_METRICS:
            if self.metrics.get(key, DERIVED_METRIC) != DERIVED_METRIC:
                return key
        return None

    def __post_init__(self):
        # Expose the RB counts a dataset reports as canonical metrics (frozen
        # dataclass: mutating the dict is allowed - we are not rebinding it).
        # ``rb`` is the scheduled-subframe grant (matches the CSV "per SF"),
        # ``rb_occ`` the all-subframe occupancy.
        if self.rb_sched_col and "rb" not in self.metrics:
            self.metrics["rb"] = self.rb_sched_col
        if self.rb_all_col and "rb_occ" not in self.metrics:
            self.metrics["rb_occ"] = self.rb_all_col

    def signature(self) -> str:
        """Stable digest of the column mapping - part of the parquet cache key,
        so editing a dataset definition invalidates its cached frames."""
        payload = json.dumps(
            {
                "key": self.key, "tech": self.tech, "link": self.link,
                "provider": self.provider, "dims": self.dims,
                "metrics": self.metrics, "rb_all": self.rb_all_col,
                "rb_sched": self.rb_sched_col, "scs": self.scs_col,
                "extra": self.extra_cols,
            },
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode()).hexdigest()[:10]

    def raw_columns(self) -> List[str]:
        cols = list(self.dims.values()) + [
            v for v in self.metrics.values() if v != DERIVED_METRIC
        ]
        if self.rb_all_col:
            cols.append(self.rb_all_col)
        if self.rb_sched_col:
            cols.append(self.rb_sched_col)
        if self.scs_col:
            cols.append(self.scs_col)
        if self.provider == "QP":
            cols.append(QP_TEST_NAME_COL)  # Qualipoc's per-row active-test label
        cols.extend(self.extra_cols)
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
            "tput_dl_cc": "LTE Net PDSCH Throughput",
            "spec_eff": DERIVED_METRIC,
            "bler": "BLER",
            "sf_usage": "PDSCH SF Utilization",
        },
        rb_all_col="PDSCH Avg RBs all SF",  # LTE SCS is 15 kHz -> scs_col=None
        rb_sched_col="PDSCH Avg RBs per SF",
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
        # MCS and BLER are in this export too, but they are read per carrier
        # (above) so that no sample is counted twice
        metrics={"tput_dl": "LTE Net PDSCH Throughput"},
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
            "tput_ul_cc": "LTE Net PUSCH Throughput",
            # no aggregated uplink export: the loader sums the carriers
            "tput_ul": DERIVED_METRIC,
            "spec_eff": DERIVED_METRIC,
            "sf_usage": "PUSCH SF Utilization",
        },
        rb_all_col="PUSCH Avg RBs all SF",
        rb_sched_col="PUSCH Avg RBs per SF",
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
        # "PDSCH SINR" is in this export too; SINR is read from Radio
        # [connected] only, like the other signal metrics
        metrics={
            "mcs": "MCS",
            "tput_dl_cc": "5G NR Net PDSCH Throughput",
            "spec_eff": DERIVED_METRIC,
            "bler": "PDSCH BLER",
            "sf_usage": "PDSCH Slot Utilization",
        },
        rb_all_col="PDSCH Avg RBs all Slots",
        rb_sched_col="PDSCH Avg RBs per Slot",
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
        metrics={"tput_dl": "5G NR Net PDSCH Throughput"},
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
            "tput_ul_cc": "5G NR Net PUSCH Throughput",
            "tput_ul": DERIVED_METRIC,
            "spec_eff": DERIVED_METRIC,
            "sf_usage": "PUSCH Slot Utilization",
        },
        rb_all_col="PUSCH Avg RBs all Slots",
        rb_sched_col="PUSCH Avg RBs per Slot",
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

    The technology and the metric together determine which files are read:
    one export per technology, never a pool of two (see the module docstring).
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

#: columns every PRISM export shares.  ``system_date_time`` is the scanner's
#: own clock, which is sometimes unset (it then reads year 2070); the loader
#: also reads ``gps_date_time`` (see PRISM_GPS_TIME_COL) and uses it to put the
#: system clock right when the two disagree.
_PRISM_COMMON = {
    "time": "system_date_time",
    "lat": "latitude",
    "lon": "longitude",
}
#: PRISM's GPS-derived wall clock (whole seconds; blank without a fix)
PRISM_GPS_TIME_COL = "gps_date_time"

PRISM_DATASETS: List[Dataset] = [
    Dataset(
        key="prism_lte",
        filename="PRISM LTE scan (*.csv)",
        label="PRISM LTE",
        tech="LTE",
        link="DL",
        provider="PRISM",
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
        extra_cols=[PRISM_GPS_TIME_COL],
    ),
    Dataset(
        key="prism_nr",
        filename="PRISM NR scan (*.csv)",
        label="PRISM NR",
        tech="NR",
        link="DL",
        provider="PRISM",
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
        extra_cols=[PRISM_GPS_TIME_COL],
    ),
]

PRISM_DATASETS_BY_KEY: Dict[str, Dataset] = {d.key: d for d in PRISM_DATASETS}



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


# --------------------------------------------------------------------------- #
# SigCap (Android logger) datasets
# --------------------------------------------------------------------------- #
#
# SigCap runs on the *same phone* as the Qualipoc (QP) logs.  Each file is one
# test session (one location + one workload) sampled every ~5 s, and it carries
# two very different kinds of data at once:
#
#   * the phone's serving-cell radio snapshot - primary LTE cell (RSRP / RSRQ /
#     RSSI, PCI, EARFCN, band) or the first NR cell - directly comparable to QP
#     and PRISM; and
#   * the tests it ran: which of iperf DL / iperf UL / HTTP / ping was active on
#     each row, and the ping round-trip time (ms).
#
# SigCap's own iperf/HTTP throughput figure is deliberately not a metric: it is
# not the throughput through the chip and transmitter. Qualipoc's PDSCH/PUSCH
# throughput is the one the platform shows. The iperf/HTTP columns are still
# read, only to tell which test was running (the traffic label and direction).
#
# So SigCap is a third measurement source.  Like PRISM it is read by globbing a
# folder, but structurally it differs again: one wide file per session (not one
# per metric family or per carrier), the collection/location/workload encoded
# in the file name, and both LTE and NR columns side by side in every row.  The
# loader routes each row to the LTE or NR dataset by its serving technology, so
# a row is never double-counted across the two. The per-row uplink/downlink
# direction comes from ``iperf_direction``.

#: columns every SigCap export shares
_SIGCAP_COMMON = {
    "time": "timestamp",
    "lat": "latitude",
    "lon": "longitude",
    "operator": "operator",
}

#: synthesised per-phone identity: device_name + short uuid (see _prep_sigcap).
#: SigCap sessions from several phones can run concurrently, so the physical
#: phone - not the file - is the unit the timeline analysis groups by.
SIGCAP_DEVICE_COL = "__device__"
#: raw columns the loader needs for routing / synthesis beyond the mapped ones
_SIGCAP_EXTRA = [
    "network_type*",
    "iperf_tput_mean_mbps",
    "http_tput_mean_mbps",
    "http_target",
    "iperf_direction",
    "lte_primary_rsrp_dbm",
    "nr_first_ss_rsrp_dbm",
    "uuid",
    "device_name",
]

SIGCAP_DATASETS: List[Dataset] = [
    Dataset(
        key="sigcap_lte",
        filename="SigCap session (*.csv)",
        label="SigCap LTE",
        tech="LTE",
        link="DL",  # per-row link (DL/UL) is filled from iperf_direction
        provider="SIGCAP",
        dims={
            **_SIGCAP_COMMON,
            "device": SIGCAP_DEVICE_COL,
            "pci": "lte_primary_pci",
            "cell_id": "lte_primary_ci",
            "freq": "lte_primary_earfcn",
            "band": "lte_primary_band*",
        },
        metrics={
            "rsrp": "lte_primary_rsrp_dbm",
            "rsrq": "lte_primary_rsrq_db",
            "rssi": "lte_primary_rssi_dbm",
            "rtt": "ping_rtt_mean_ms",
            "n_factor": DERIVED_METRIC,
        },
        extra_cols=_SIGCAP_EXTRA,
        note="SigCap on the phone - serving LTE cell (RSRP, RSRQ, RSSI) and ping RTT.",
    ),
    Dataset(
        key="sigcap_nr",
        filename="SigCap session (*.csv)",
        label="SigCap NR",
        tech="NR",
        link="DL",
        provider="SIGCAP",
        dims={
            **_SIGCAP_COMMON,
            "device": SIGCAP_DEVICE_COL,
            "pci": "nr_first_pci",
            "cell_id": "nr_first_nci",
            "freq": "nr_first_arfcn",
            "band": "nr_first_band*",
        },
        metrics={
            "rsrp": "nr_first_ss_rsrp_dbm",
            "rsrq": "nr_first_ss_rsrq_db",
            "rtt": "ping_rtt_mean_ms",
        },
        extra_cols=_SIGCAP_EXTRA,
        note="SigCap on the phone - serving NR cell (RSRP, RSRQ) and ping RTT.",
    ),
]

SIGCAP_DATASETS_BY_KEY: Dict[str, Dataset] = {d.key: d for d in SIGCAP_DATASETS}

#: every dataset the platform knows about (QP, PRISM and SigCap) - use this
#: for label / metadata look-ups by key; the metric->dataset resolvers stay
#: split so the sources never get pooled by accident.
ALL_DATASETS_BY_KEY: Dict[str, Dataset] = {
    **DATASETS_BY_KEY,
    **PRISM_DATASETS_BY_KEY,
    **SIGCAP_DATASETS_BY_KEY,
}


def sigcap_datasets_for(
    metric: str, tech: Optional[str] = None, available: Optional[List[str]] = None
) -> List[Dataset]:
    """SigCap datasets that can supply ``metric`` for ``tech`` (and are present)."""
    out = [d for d in SIGCAP_DATASETS if metric in d.metrics]
    if tech in ("LTE", "NR"):
        out = [d for d in out if d.tech == tech]
    if available is not None:
        keep = set(available)
        out = [d for d in out if d.key in keep]
    return out
