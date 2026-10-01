"""Synthetic QP-shaped CSV fixtures.

The real exports are ~110 MB and are not committed, so the tests build small
CSVs that use the *same* raw column names taken straight from the schema.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pandas as pd
import pytest

from qpviz.schema import DATASETS_BY_KEY, Dataset

METRIC_RANGES = {
    "rsrp": (-125.0, -75.0),
    "rsrq": (-20.0, -6.0),
    "rssi": (-100.0, -55.0),
    "sinr": (-8.0, 25.0),
    "mcs": (0.0, 28.0),
    "throughput": (0.0, 90000.0),
    "bler": (0.0, 30.0),
    "cqi": (0.0, 15.0),
    "tx_power": (-10.0, 23.0),
    "pathloss": (80.0, 150.0),
    "rb": (1.0, 25.0),      # RBs per subframe, scheduled (rb_sched_col)
    "rb_occ": (0.0, 14.0),  # RBs per subframe, all SF (rb_all_col)
}

COLLECTIONS = ["Collection_A", "Collection_B", "Collection_C"]
PCIS = [123, 456, 790]
FREQS = {"LTE": [8665, 2175], "NR": [394850, 520050]}
BANDS = {"LTE": ["25", "66"], "NR": ["25", "41"]}


def write_dataset(directory: Path, ds: Dataset, rows: int = 240, seed: int = 7) -> Path:
    rng = random.Random(seed + len(ds.key))
    columns = ds.raw_columns()
    path = directory / ds.filename
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        base = pd.Timestamp("2026-07-22 14:00:00")
        for i in range(rows):
            row = {c: "" for c in columns}
            row[ds.dims["time"]] = (
                base + pd.Timedelta(milliseconds=500 * i)
            ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            row[ds.dims["collection"]] = COLLECTIONS[i % len(COLLECTIONS)]
            row[ds.dims["database"]] = "Starlink_D2C_2026"
            row[ds.dims["operator"]] = "T-Mobile"
            row[ds.dims["model"]] = "Samsung Galaxy S24+ (SM-S926U1)"
            row[ds.dims["serial"]] = "R5CX32LGQBD"
            row[ds.dims["ran_config"]] = "5G SA" if ds.tech == "NR" else "LTE"
            if ds.has_geo:
                row[ds.dims["lat"]] = f"{41.6990 + (i % 40) * 0.0005:.8f}"
                row[ds.dims["lon"]] = f"{-86.2400 + (i % 40) * 0.0005:.8f}"
            # Step the PCI on a period coprime with the collection cycle, so
            # every collection sees every PCI. Anything sharing a factor with
            # 3 pins each collection to its own subset and makes the shared-PCI
            # set empty for reasons that are an artefact of the fixture.
            row[ds.dims["pci"]] = PCIS[(i // 5) % len(PCIS)]
            row[ds.dims["freq"]] = FREQS[ds.tech][i % 2]
            band_value = BANDS[ds.tech][i % 2]
            row[ds.dims["band"]] = (
                f"{'5G NR n' if ds.tech == 'NR' else 'LTE E-UTRA '}{band_value}"
                if ds.dims["band"] in ("PCell Band",)
                else band_value
            )
            if "cell_type" in ds.dims:
                # every 9th row is an aggregated carrier, not the serving cell
                row[ds.dims["cell_type"]] = "MCG SCell1" if i % 9 == 8 else "MCG PCell"
            # cell identity: a handful of eNBs, each with a couple of sectors
            enb = 11000 + (i // 7) % 5
            sector = 10 + (i // 3) % 4
            if "enb_sector" in ds.dims:
                row[ds.dims["enb_sector"]] = f"{enb}/{sector}"
            if "enb" in ds.dims:
                row[ds.dims["enb"]] = enb
            if "cell_id" in ds.dims:
                # NR reports the full NCGI; LTE an ECI. Keep NR consistent with
                # gNB-Id so the derived sector round-trips.
                row[ds.dims["cell_id"]] = (
                    enb * 256 + sector if ds.tech == "NR" else 2895000 + enb
                )
            if "ssb" in ds.dims:
                row[ds.dims["ssb"]] = (i % 2) * 2
            for metric, source_col in ds.metrics.items():
                if source_col == "@derived":
                    continue  # spec_eff / tput_per_rb are computed, not read
                lo, hi = METRIC_RANGES[metric]
                # every 17th row leaves the metric blank, like the real exports
                row[source_col] = "" if i % 17 == 0 else f"{rng.uniform(lo, hi):.3f}"
            # RBs-allocated + SCS feed the derived spectral-efficiency metrics
            if ds.rb_all_col:
                row[ds.rb_all_col] = f"{rng.uniform(0.05, 8.0):.4f}"
            if ds.scs_col:
                row[ds.scs_col] = 15 if i % 2 == 0 else 30
            writer.writerow(row)
    return path


@pytest.fixture(autouse=True)
def _isolated_environ():
    """Several app tests point the data folders at fixtures via ``os.environ``
    directly; restore the environment after every test so one file's choice
    of SigCap / PRISM folder cannot leak into the next."""
    import os
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("qp_csv")
    for key in ("lte_radio", "nr_radio", "lte_pdsch_pc", "lte_pusch_pc", "lte_pucch"):
        write_dataset(directory, DATASETS_BY_KEY[key])
    return directory


@pytest.fixture()
def cache_dir(tmp_path) -> Path:
    return tmp_path / "cache"


@pytest.fixture()
def lte_radio(fixture_dir, cache_dir):
    from qpviz import loader

    return loader.read_dataset(fixture_dir, DATASETS_BY_KEY["lte_radio"], cache_dir)


# --------------------------------------------------------------------------- #
# Synthetic PRISM (SDR scanner) fixtures
# --------------------------------------------------------------------------- #

# Real column headers taken from the PRISM exports. PRISM has no Collection
# column (the collection is in the file name), lower-case names, ISO timestamps,
# and one file per technology + carrier.
_PRISM_LTE_COLUMNS = [
    "tech", "band", "earfcn", "frequency_mhz", "bandwidth_mhz", "rssi", "rsrp",
    "rsrq", "sinr", "physical_cell_id", "mcc", "mnc", "tac", "cell_identity",
    "e_node_b", "sector_id", "mib", "sib1", "sib2", "sib3", "sib4", "latitude",
    "longitude", "altitude", "fix_type", "satellite_count",
    "horizontal_dilution_of_precision", "gps_date_time", "system_date_time",
    "cp", "rcs_cp", "survey_id", "scan_iteration",
]
_PRISM_NR_COLUMNS = [
    "tech", "band", "nrarfcn", "frequency_mhz", "ssb_subcarrier_spacing_hz",
    "physical_cell_id", "ss_rssi", "ss_sinr", "ss_rsrp", "ss_rsrq", "issb",
    "mib", "mcc", "mnc", "tac", "cell_identity", "dl_freq_band_indicator",
    "latitude", "longitude", "altitude", "fix_type", "satellite_count",
    "horizontal_dilution_of_precision", "gps_date_time", "system_date_time",
    "cp", "rcs_cp", "survey_id", "scan_iteration",
]

_PRISM_PCIS = [281, 163, 305, 118, 536, 742, 3, 269, 90, 131, 160, 200]


def _write_prism(directory, filename, columns, tech, arfcn, rows, seed, geo=True):
    rng = random.Random(seed)
    path = directory / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        base = pd.Timestamp("2026-09-01 19:00:00")
        for i in range(rows):
            row = {c: "" for c in columns}
            row["tech"] = tech
            row["band"] = "25"
            row["physical_cell_id"] = _PRISM_PCIS[i % len(_PRISM_PCIS)]
            row["system_date_time"] = (
                base + pd.Timedelta(seconds=i)
            ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            if geo:
                row["latitude"] = f"{41.6990 + (i % 30) * 0.0004:.6f}"
                row["longitude"] = f"{-86.2400 + (i % 30) * 0.0004:.6f}"
            if tech == "lte":
                row["earfcn"] = arfcn
                row["rssi"] = f"{rng.uniform(-100, -60):.2f}"
                row["rsrp"] = f"{rng.uniform(-125, -80):.2f}"
                row["rsrq"] = f"{rng.uniform(-18, -6):.2f}"
                row["sinr"] = f"{rng.uniform(-8, 20):.2f}"
                # eNB / sector only decoded on a fraction of scans, as in reality
                if i % 4 == 0:
                    row["e_node_b"] = 11000 + (i // 8) % 6
                    row["sector_id"] = 10 + (i // 3) % 3
            else:
                row["nrarfcn"] = arfcn
                row["ss_rssi"] = f"{rng.uniform(-90, -60):.2f}"
                row["ss_rsrp"] = f"{rng.uniform(-125, -95):.2f}"
                row["ss_rsrq"] = f"{rng.uniform(-16, -5):.2f}"
                row["ss_sinr"] = f"{rng.uniform(-10, 12):.2f}"
                row["issb"] = i % 4
            writer.writerow(row)
    return path


@pytest.fixture(scope="session")
def prism_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("prism_csv")
    _write_prism(
        directory, "2026_09_01_ LTE_8665_Test_Kitchen_static.csv",
        _PRISM_LTE_COLUMNS, "lte", 8665, rows=200, seed=1,
    )
    _write_prism(
        directory, "2026_09_01_ NR_397970_Test_Kitchen_static.csv",
        _PRISM_NR_COLUMNS, "nr", 397970, rows=160, seed=2,
    )
    _write_prism(
        directory, "2026_09_02_LTE_8665_Test_Bedroom_static.csv",
        _PRISM_LTE_COLUMNS, "lte", 8665, rows=180, seed=3,
    )
    return directory


# --------------------------------------------------------------------------- #
# Synthetic SigCap (Android logger) fixtures
# --------------------------------------------------------------------------- #

# A realistic subset of the ~983 SigCap columns - the loader only reads the ones
# a dataset maps, so a subset header round-trips exactly like the full export.
_SIGCAP_COLUMNS = [
    "timestamp", "latitude", "longitude", "operator", "network_type*",
    "uuid", "device_name",
    "iperf_tput_mean_mbps", "iperf_direction", "iperf_protocol",
    "http_tput_mean_mbps", "ping_rtt_mean_ms",
    "lte_primary_pci", "lte_primary_ci", "lte_primary_earfcn",
    "lte_primary_band*", "lte_primary_rsrp_dbm", "lte_primary_rsrq_db",
    "lte_primary_rssi_dbm", "lte_primary_rssnr_db", "lte_primary_cqi",
    "nr_first_pci", "nr_first_nci", "nr_first_arfcn", "nr_first_band*",
    "nr_first_ss_rsrp_dbm", "nr_first_ss_rsrq_db", "nr_first_ss_sinr_db",
]

_SIGCAP_INT_SENTINEL = 2147483647


def _write_sigcap(directory, filename, workload, rows=60, seed=11, tech="LTE",
                  start="2026-09-12T16:11:00", uuid="dev0uuid-aaaa",
                  device_name="pa2q", pci_seq=None):
    """Write one synthetic SigCap session file.

    ``workload`` is ``"iperf"`` or ``"http"``. iperf rows alternate download /
    upload; every fifth radio row carries the CQI "unknown" sentinel and a couple
    of rows are app-only (throughput but no serving-cell fix) to exercise routing.
    """
    rng = random.Random(seed)
    path = directory / filename
    base = pd.Timestamp(start)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SIGCAP_COLUMNS)
        writer.writeheader()
        for i in range(rows):
            row = {c: "NaN" for c in _SIGCAP_COLUMNS}
            # ISO-8601 with a -0400 offset, exactly as SigCap stamps it
            row["timestamp"] = (base + pd.Timedelta(seconds=5 * i)).strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )[:-3] + "-0400"
            row["latitude"] = f"{41.6990 + (i % 20) * 0.0002:.7f}"
            row["longitude"] = f"{-86.2400 + (i % 20) * 0.0002:.7f}"
            row["operator"] = "T-Mobile SpaceX"
            row["network_type*"] = tech
            row["uuid"] = uuid
            row["device_name"] = device_name
            app_only = i in (3, 4)  # throughput logged but no serving-cell fix
            if not app_only:
                if tech == "LTE":
                    row["lte_primary_pci"] = (
                        pci_seq[i % len(pci_seq)] if pci_seq else 100 + (i % 7)
                    )
                    row["lte_primary_ci"] = 200000 + i
                    row["lte_primary_earfcn"] = 8665
                    row["lte_primary_band*"] = "25"
                    row["lte_primary_rsrp_dbm"] = f"{rng.uniform(-125, -95):.0f}"
                    row["lte_primary_rsrq_db"] = f"{rng.uniform(-18, -5):.0f}"
                    row["lte_primary_rssi_dbm"] = f"{rng.uniform(-95, -70):.0f}"
                    row["lte_primary_rssnr_db"] = f"{rng.uniform(-8, 15):.0f}"
                    # Android reports Integer.MAX_VALUE when CQI is unavailable
                    row["lte_primary_cqi"] = _SIGCAP_INT_SENTINEL if i % 5 else 9
                else:
                    row["nr_first_pci"] = 300 + (i % 5)
                    row["nr_first_nci"] = 500000 + i
                    row["nr_first_arfcn"] = 397970
                    row["nr_first_band*"] = "25"
                    row["nr_first_ss_rsrp_dbm"] = f"{rng.uniform(-120, -90):.0f}"
                    row["nr_first_ss_rsrq_db"] = f"{rng.uniform(-16, -5):.0f}"
                    row["nr_first_ss_sinr_db"] = f"{rng.uniform(-6, 18):.0f}"
            # application-layer workload - phases are interleaved and mutually
            # exclusive, like a real SigCap run (some rows ping, some iperf, ...)
            if i % 6 == 0:
                row["ping_rtt_mean_ms"] = f"{rng.uniform(50, 400):.1f}"
            elif workload == "iperf":
                row["iperf_tput_mean_mbps"] = f"{rng.uniform(0, 12):.3f}"
                row["iperf_direction"] = "download" if i % 2 == 0 else "upload"
                row["iperf_protocol"] = "TCP"
            else:
                row["http_tput_mean_mbps"] = f"{rng.uniform(0, 3):.3f}"
            writer.writerow(row)
    return path


@pytest.fixture(scope="session")
def sigcap_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("sigcap_csv")
    _write_sigcap(
        directory,
        "2026_09_S12_1611_E12_1620Z-0400_outdoor_T-Mobile_dev_D2C_Sta_YC_910_iperf.debug_general.csv",
        "iperf", rows=60, seed=1, start="2026-09-12T16:11:00",
    )
    _write_sigcap(
        directory,
        "2026_09_S12_1633_E12_1640Z-0400_outdoor_T-Mobile_dev_D2C_Sta_YC_1011_http .debug_general.csv",
        "http", rows=50, seed=2, start="2026-09-12T16:33:00",
    )
    # August bracket-style name, plus a macOS "keep both" duplicate to de-dup
    _write_sigcap(
        directory,
        "2026-08-14-14-23-06[D2C_Static_OpenField_SigcapiperfDL].csv",
        "iperf", rows=40, seed=3, start="2026-08-14T14:23:00",
    )
    _write_sigcap(
        directory,
        "2026-08-14-14-23-06[D2C_Static_OpenField_SigcapiperfDL] 2.csv",
        "iperf", rows=40, seed=3, start="2026-08-14T14:23:00",
    )
    return directory


@pytest.fixture(scope="session")
def sigcap_nr_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("sigcap_nr_csv")
    _write_sigcap(
        directory,
        "2026_09_S20_1000_E20_1010Z-0400_outdoor_T-Mobile_dev_D2C_Sta_YC_lab_iperf.debug_general.csv",
        "iperf", rows=40, seed=5, tech="NR", start="2026-09-20T10:00:00",
    )
    return directory


@pytest.fixture(scope="session")
def sigcap_multi_dir(tmp_path_factory) -> Path:
    """Two phones on the same spot: a solo stretch for phone A, then a window
    where A and B record concurrently. PCI sequences are chosen so the first
    concurrent bins share a PCI and later ones differ - deterministic input for
    the shared-PCI logic."""
    directory = tmp_path_factory.mktemp("sigcap_multi")
    # phone A alone, 19:00:00-19:02:25 (30 rows @5s)
    _write_sigcap(
        directory,
        "2026_09_S14_1900_E14_1902Z-0400_outdoor_T-Mobile_aaaa1111_D2C_OpenField_Static_onephoneA_iperf.debug_general.csv",
        "iperf", rows=30, seed=1, start="2026-09-14T19:00:00",
        uuid="aaaa1111-eeee", device_name="pa2q",
        pci_seq=[100],
    )
    # phone A concurrent, 19:05:00.., PCI 100 then 200
    _write_sigcap(
        directory,
        "2026_09_S14_1905_E14_1908Z-0400_outdoor_T-Mobile_aaaa1111_D2C_OpenField_Static_twoPhoneA_iperf.debug_general.csv",
        "iperf", rows=40, seed=2, start="2026-09-14T19:05:00",
        uuid="aaaa1111-eeee", device_name="pa2q",
        pci_seq=[100, 100, 100, 100, 100, 100, 200, 200, 200, 200],
    )
    # phone B concurrent, same 19:05:00 start, PCI 100 then 300
    _write_sigcap(
        directory,
        "2026_09_S14_1905_E14_1908Z-0400_outdoor_T-Mobile_bbbb2222_D2C_OpenField_Static_twoPhoneB_iperf.debug_general.csv",
        "iperf", rows=40, seed=3, start="2026-09-14T19:05:00",
        uuid="bbbb2222-ffff", device_name="e2q",
        pci_seq=[100, 100, 100, 100, 100, 100, 300, 300, 300, 300],
    )
    return directory
