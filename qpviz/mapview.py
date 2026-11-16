"""Folium map with a draw control used both to show data and to pick a region."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .charts import (
    OTHER_COLOR,
    categorical_bins,
    categorical_counts,
    categorical_legend_html,
    color_bins,
    color_for,
    downsample,
    legend_html,
)
from .filters import LatLon, geolocated

#: Environment variable / Streamlit secret holding the CARTO basemaps API key.
#: CARTO stopped serving its raster basemaps without a key on 2026-09-25 (a
#: keyless request now gets an "API KEY REQUIRED" tile), so the CARTO styles are
#: only offered when a key is configured. Get a free key at
#: https://carto.com/basemaps/apikey/.
CARTO_KEY_NAME = "CARTO_API_KEY"

#: Basemap choices backed by CARTO, mapped to their ``xyzservices`` provider.
CARTO_TILES: Dict[str, str] = {
    "Light (CartoDB positron)": "CartoDB.Positron",
    "Dark (CartoDB dark_matter)": "CartoDB.DarkMatter",
}

TILE_CHOICES: Dict[str, Tuple[str, Optional[str]]] = {
    "OpenStreetMap": ("OpenStreetMap", None),
    "Light (CartoDB positron)": ("CartoDB positron", None),
    "Dark (CartoDB dark_matter)": ("CartoDB dark_matter", None),
    "Satellite (Esri)": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Tiles &copy; Esri",
    ),
}


def carto_api_key(secrets=None) -> Optional[str]:
    """The CARTO key from the environment, else from ``secrets`` (``st.secrets``)."""
    key = os.environ.get(CARTO_KEY_NAME, "").strip()
    if not key and secrets is not None:
        try:
            key = str(secrets.get(CARTO_KEY_NAME, "") or "").strip()
        except Exception:  # no secrets.toml -> Streamlit raises on access
            key = ""
    return key or None


def default_tiles(carto_key: Optional[str] = None) -> str:
    """The basemap the map is drawn on: light CARTO when a key is configured,
    otherwise OpenStreetMap (which needs none)."""
    return "Light (CartoDB positron)" if carto_key else "OpenStreetMap"


def resolve_tiles(tiles: str, carto_key: Optional[str] = None):
    """``(tiles, attr)`` for ``folium.Map``; a CARTO choice without a key falls
    back to OpenStreetMap rather than rendering blank "API KEY REQUIRED" tiles."""
    if tiles in CARTO_TILES:
        if not carto_key:
            return TILE_CHOICES["OpenStreetMap"]
        import xyzservices

        provider = xyzservices.providers.query_name(CARTO_TILES[tiles])
        return provider(apikey=carto_key), None
    return TILE_CHOICES.get(tiles, TILE_CHOICES["OpenStreetMap"])


def regions_from_drawings(drawings) -> List[List[LatLon]]:
    """Convert ``st_folium``'s ``all_drawings`` GeoJSON into (lat, lon) polygons."""
    polygons: List[List[LatLon]] = []
    for feature in drawings or []:
        geometry = (feature or {}).get("geometry") or {}
        gtype = geometry.get("type")
        coords = geometry.get("coordinates")
        if gtype == "Polygon" and coords:
            ring = coords[0]
        elif gtype == "MultiPolygon" and coords:
            for poly in coords:
                if poly:
                    polygons.append([(float(lat), float(lon)) for lon, lat in poly[0]])
            continue
        else:
            continue
        polygons.append([(float(lat), float(lon)) for lon, lat in ring])
    return [p for p in polygons if len(p) >= 3]


def new_regions(
    drawn: Sequence[Sequence[LatLon]], existing: Sequence[Sequence[LatLon]]
) -> List[List[LatLon]]:
    """The polygons in ``drawn`` that are not already in ``existing``."""
    known = [list(map(tuple, p)) for p in existing]
    out: List[List[LatLon]] = []
    for polygon in drawn:
        shape = list(map(tuple, polygon))
        if shape not in known and shape not in out:
            out.append(shape)
    return out


def bounds(frame: pd.DataFrame) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    geo = geolocated(frame)
    if geo.empty:
        return None
    lat = pd.to_numeric(geo["lat"], errors="coerce")
    lon = pd.to_numeric(geo["lon"], errors="coerce")
    return ((float(lat.min()), float(lon.min())), (float(lat.max()), float(lon.max())))


def build_map(
    frame: pd.DataFrame,
    metric: str,
    metric_label: str,
    *,
    color_range: Optional[Tuple[float, float]] = None,
    max_points: int = 3000,
    tiles: str = "OpenStreetMap",
    carto_key: Optional[str] = None,
    mode: str = "light",
    regions: Sequence[Sequence[LatLon]] = (),
    radius: int = 4,
    color_by: Optional[str] = None,
    color_by_label: str = "",
    color_max_slots: int = 10,
):
    """Build a folium map with a rectangle/polygon draw control.

    Points are coloured by ``metric`` on a continuous scale - spanning
    ``color_range`` when given, so the colours mean the same on every map - or,
    when ``color_by`` names an identity column such as ``enb``, categorically:
    the ``color_max_slots`` most common values are coloured and the rest fold
    into a muted "Other".
    """
    import folium
    from folium.plugins import Draw

    tile_value, attribution = resolve_tiles(tiles, carto_key)
    geo = geolocated(frame)
    box = bounds(geo)
    center = (41.7001, -86.2371)  # ND campus - sensible default when empty
    if box:
        center = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)

    fmap = folium.Map(
        location=center,
        zoom_start=14,
        tiles=tile_value,
        attr=attribution,
        control_scale=True,
    )
    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
            "rectangle": True,
            "polygon": True,
        },
        # shapes are only added on the map; the region is removed with the
        # "Clear region" button beside it
        edit_options={"edit": False, "remove": False},
    ).add_to(fmap)

    if geo.empty or metric not in geo.columns:
        return fmap, [], []

    plotted = downsample(geo.sort_values("time", na_position="last"), max_points)
    values = pd.to_numeric(plotted[metric], errors="coerce").to_numpy(dtype="float64")

    categorical = bool(color_by) and color_by in plotted.columns
    legend_counts = None
    if categorical:
        identities = plotted[color_by].to_numpy()
        order, palette, other_count = categorical_bins(
            identities, max_slots=int(color_max_slots), mode=mode
        )
        legend_counts = categorical_counts(identities)
        edges, colors = [], []
    else:
        edges, colors = color_bins(values, value_range=color_range)

    layer = folium.FeatureGroup(name=metric_label, show=True)
    lat = pd.to_numeric(plotted["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(plotted["lon"], errors="coerce").to_numpy()
    pci = plotted["pci"].astype("object").to_numpy()
    freq = plotted["freq"].astype("object").to_numpy()
    coll = plotted["collection"].astype("object").to_numpy()
    enb = (
        plotted["enb"].astype("object").to_numpy()
        if "enb" in plotted.columns
        else np.full(len(plotted), None)
    )
    sector = (
        plotted["sector"].astype("object").to_numpy()
        if "sector" in plotted.columns
        else np.full(len(plotted), None)
    )
    times = pd.to_datetime(plotted["time"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    ).to_numpy()

    def _fmt(value) -> str:
        return "n/a" if value is None or pd.isna(value) else str(value)

    for i in range(len(plotted)):
        value = values[i]
        # counts (satellites visible, sectors per eNB) read better whole
        shown = (
            f"{value:.0f}"
            if np.isfinite(value) and float(value).is_integer()
            else f"{value:.2f}"
        )
        tooltip = (
            f"<b>{metric_label}: {shown}</b><br>{coll[i]}<br>"
            f"PCI {_fmt(pci[i])} · eNB/gNB {_fmt(enb[i])} · "
            f"Sector {_fmt(sector[i])} · ARFCN {_fmt(freq[i])}"
            f"<br>{times[i]}"
        )
        if categorical:
            fill = palette.get(identities[i], OTHER_COLOR)
        else:
            fill = color_for(value, edges, colors)
        folium.CircleMarker(
            location=(lat[i], lon[i]),
            radius=radius,
            color=None,
            weight=0,
            fill=True,
            fill_color=fill,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(tooltip, sticky=True),
        ).add_to(layer)
    layer.add_to(fmap)

    for polygon in regions:
        folium.Polygon(
            locations=[(la, lo) for la, lo in polygon],
            color="#0b0b0b",
            weight=2,
            dash_array="6,4",
            fill=True,
            fill_opacity=0.05,
        ).add_to(fmap)

    import branca

    if categorical:
        legend = categorical_legend_html(
            order, palette, color_by_label or color_by, mode, other_count,
            value_prefix=f"{color_by_label or color_by} " if color_by_label else "",
            counts=legend_counts,
        )
    else:
        legend = legend_html(edges, colors, metric_label, mode)
    fmap.get_root().html.add_child(branca.element.Element(legend))
    if box:
        fmap.fit_bounds([box[0], box[1]])
    return fmap, edges, colors
