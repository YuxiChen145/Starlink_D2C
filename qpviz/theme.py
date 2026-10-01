"""Colour tokens for the app.

Values come from the validated reference palette: categorical hues are used in
fixed slot order (never cycled by rank), sequential encoding is a single blue
hue light -> dark, and diverging encoding is blue <-> red with a neutral grey
midpoint.
"""

from __future__ import annotations

from typing import Dict, List

CATEGORICAL: Dict[str, List[str]] = {
    "light": [
        "#2a78d6",  # blue
        "#eb6834",  # orange
        "#1baf7a",  # aqua
        "#eda100",  # yellow
        "#e87ba4",  # magenta
        "#008300",  # green
        "#4a3aa7",  # violet
        "#e34948",  # red
    ],
    "dark": [
        "#3987e5",
        "#d95926",
        "#199e70",
        "#c98500",
        "#d55181",
        "#008300",
        "#9085e9",
        "#e66767",
    ],
}

#: shaded-band accents on the multi-phone timeline. Phones take the categorical
#: slots (blue, orange, ...), so the bands use hues no phone can get: a neutral
#: grey for "two phones active" and the yellow for "shared PCI".
BAND_CONCURRENT: Dict[str, str] = {"light": "#6b6b6b", "dark": "#a0a0a0"}
BAND_SHARED_PCI: Dict[str, str] = {"light": "#eda100", "dark": "#c98500"}

#: used as a secondary encoding once the 8 categorical slots are exhausted
DASHES: List[str] = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]

SEQUENTIAL_BLUE: List[str] = [
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#0d366b",
]

DIVERGING_RED_BLUE: List[str] = [
    "#8f1f1f",
    "#b32e2e",
    "#d03b3b",
    "#e57070",
    "#c3c2b7",  # neutral midpoint
    "#86b6ef",
    "#3987e5",
    "#256abf",
    "#0d366b",
]

CHROME: Dict[str, Dict[str, str]] = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "text_primary": "#0b0b0b",
        "text_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
    },
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def series_style(index: int, mode: str = "light"):
    """Colour + dash for the *index*-th series.

    The first eight series take the eight categorical slots solid; beyond that
    the hues repeat but the dash pattern changes, so identity never rests on a
    repeated colour alone.
    """
    palette = CATEGORICAL.get(mode, CATEGORICAL["light"])
    color = palette[index % len(palette)]
    dash = DASHES[(index // len(palette)) % len(DASHES)]
    return color, dash
