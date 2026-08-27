"""Qualipoc measurement visualisation toolkit for the Starlink D2C project."""

from .schema import DATASETS, DATASETS_BY_KEY, METRICS, METRIC_ORDER, Dataset, Metric

__all__ = [
    "DATASETS",
    "DATASETS_BY_KEY",
    "METRICS",
    "METRIC_ORDER",
    "Dataset",
    "Metric",
]

__version__ = "0.1.0"
