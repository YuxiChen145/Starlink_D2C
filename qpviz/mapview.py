"""Folium map with a draw control used both to show data and to pick a region."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .charts import (
    OTHER_COLOR,
    categorical_bins,
    categorical_legend_html,
    color_bins,
    color_for,
    downsample,
    legend_html,
)
from .filters import LatLon, geolocated

TILE_CHOICES: Dict[str, Tuple[str, Optional[str]]] = {
    "OpenStreetMap": ("OpenStreetMap", None),
    "Light (CartoDB positron)": ("CartoDB positron", None),
    "Dark (CartoDB dark_matter)": ("CartoDB dark_matter", None),
    "Satellite (Esri)": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "Tiles &copy; Esri",
    ),
}


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
    scale: str = "diverging",
    midpoint: Optional[float] = None,
    max_points: int = 3000,
    tiles: str = "Light (CartoDB positron)",
    mode: str = "light",
    regions: Sequence[Sequence[LatLon]] = (),
    radius: int = 4,
    color_by: Optional[str] = None,
    color_by_label: str = "",
):
    """Build a folium map with a rectangle/polygon draw control.

    Points are coloured by ``metric`` on a continuous scale, or - when
    ``color_by`` names an identity column such as ``pci`` - categorically, with
    the eight most common values taking the categorical slots and the rest
    folding into a muted "Other".
    """
    import folium
    from folium.plugins import Draw

    tile_value, attribution = TILE_CHOICES.get(tiles, TILE_CHOICES["OpenStreetMap"])
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
        edit_options={"edit": True, "remove": True},
    ).add_to(fmap)

    if geo.empty or metric not in geo.columns:
        return fmap, [], []

    plotted = downsample(geo.sort_values("time", na_position="last"), max_points)
    values = pd.to_numeric(plotted[metric], errors="coerce").to_numpy(dtype="float64")

    categorical = bool(color_by) and color_by in plotted.columns
    if categorical:
        identities = plotted[color_by].to_numpy()
        order, palette, other_count = categorical_bins(identities, mode=mode)
        edges, colors = [], []
    else:
        edges, colors = color_bins(values, scale=scale, midpoint=midpoint)

    layer = folium.FeatureGroup(name=metric_label, show=True)
    lat = pd.to_numeric(plotted["lat"], errors="coerce").to_numpy()
    lon = pd.to_numeric(plotted["lon"], errors="coerce").to_numpy()
    pci = plotted["pci"].astype("object").to_numpy()
    freq = plotted["freq"].astype("object").to_numpy()
    coll = plotted["collection"].astype("object").to_numpy()
    times = pd.to_datetime(plotted["time"], errors="coerce").dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    ).to_numpy()

    for i in range(len(plotted)):
        value = values[i]
        tooltip = (
            f"<b>{metric_label}: {value:.2f}</b><br>{coll[i]}<br>"
            f"PCI {pci[i]} · ARFCN {freq[i]}<br>{times[i]}"
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
        )
    else:
        legend = legend_html(edges, colors, metric_label, mode)
    fmap.get_root().html.add_child(branca.element.Element(legend))
    if box:
        fmap.fit_bounds([box[0], box[1]])
    return fmap, edges, colors
