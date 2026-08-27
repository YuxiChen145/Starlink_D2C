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
            row[ds.dims["pci"]] = PCIS[i % len(PCIS)]
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
                lo, hi = METRIC_RANGES[metric]
                # every 17th row leaves the metric blank, like the real exports
                row[source_col] = "" if i % 17 == 0 else f"{rng.uniform(lo, hi):.3f}"
            writer.writerow(row)
    return path


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
