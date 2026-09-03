"""Starlink Direct-to-Cell satellite visibility from TLEs + SGP4.

The goal is one number per measurement sample: **how many Starlink D2C
satellites were above the horizon (by some elevation mask) at that instant, seen
from that spot on the ground.**

Pipeline
--------
1. **TLEs.** Two-line element sets are read from a folder (`TLE/`). Dated
   snapshots dropped in by hand (e.g. Space-Track exports) give accurate
   historical epochs; a daily CelesTrak fetch (`fetch_celestrak`) keeps it
   current going forward. Direct-to-Cell satellites are identified by a
   ``[DTC]`` tag in the TLE name line, the CelesTrak convention.
2. **Nearest epoch.** SGP4 is only accurate near a TLE's epoch, so for a given
   instant each satellite is propagated from *its* TLE whose epoch is closest in
   time. Samples are bucketed by UTC date and each bucket uses the snapshot
   nearest that date.
3. **Propagation + geometry.** SGP4 gives a TEME position; a GMST rotation takes
   it to Earth-fixed (ECEF); a topocentric ENU rotation about the observer gives
   the elevation angle. A satellite counts as visible when its elevation is at
   or above the mask.

No network is required to *use* a TLE folder — only `fetch_celestrak` reaches
out, and only when asked. The heavy step is propagation; everything is
vectorised over samples (one pass per satellite), so tens of thousands of
samples against a few hundred satellites take a couple of seconds.
"""

from __future__ import annotations

import http.cookiejar
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # sgp4 is an optional runtime dependency; the app degrades gracefully.
    from sgp4.api import Satrec, jday

    HAVE_SGP4 = True
except Exception:  # pragma: no cover - exercised only where sgp4 is absent
    Satrec = None  # type: ignore
    jday = None  # type: ignore
    HAVE_SGP4 = False

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

#: CelesTrak's "Direct to Cell" satellites carry a ``[DTC]`` tag in the name.
DTC_RE = re.compile(r"\[?\bDTC\b\]?", re.IGNORECASE)

#: WGS-84 ellipsoid
_WGS84_A = 6378.137  # km, semi-major axis
_WGS84_E2 = 6.69437999014e-3  # first eccentricity squared

CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle"
)

SPACETRACK_BASE = "https://www.space-track.org"

#: elevation masks pre-computed so the UI can switch threshold without a
#: re-propagation; the app snaps its slider to these
DEFAULT_MASKS: Tuple[int, ...] = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)


# --------------------------------------------------------------------------- #
# TLE ingest
# --------------------------------------------------------------------------- #


@dataclass
class Sat:
    """One satellite at one TLE epoch."""

    name: str
    norad: int
    epoch: datetime  # UTC
    line1: str
    line2: str

    def satrec(self):
        if not HAVE_SGP4:
            raise RuntimeError("sgp4 is not installed")
        return Satrec.twoline2rv(self.line1, self.line2)

    @property
    def is_dtc(self) -> bool:
        return bool(DTC_RE.search(self.name))


def _tle_epoch(line1: str) -> datetime:
    """UTC epoch encoded in columns 19-32 of TLE line 1 (YYDDD.DDDDDDDD)."""
    raw = line1[18:32]
    yy = int(raw[0:2])
    year = 2000 + yy if yy < 57 else 1900 + yy
    doy = float(raw[2:])
    # day-of-year is 1-based; seconds since Jan 1 00:00 = (doy - 1) days
    base = datetime(year, 1, 1, tzinfo=timezone.utc)
    return base + _timedelta_days(doy - 1.0)


def _timedelta_days(days: float):
    from datetime import timedelta

    return timedelta(days=days)


def parse_tle_text(text: str) -> List[Sat]:
    """Parse TLE text (2-line, or 3-line with a name) into ``Sat`` records.

    Robust to blank lines and a leading ``0 `` on name lines.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    sats: List[Sat] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        name = ""
        if not line.startswith("1 ") and not line.startswith("2 "):
            name = line[2:].strip() if line.startswith("0 ") else line.strip()
            i += 1
        if i + 1 >= n:
            break
        l1, l2 = lines[i], lines[i + 1]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            i += 1
            continue
        try:
            norad = int(l1[2:7])
            epoch = _tle_epoch(l1)
        except Exception:
            i += 2
            continue
        if not name:
            name = f"NORAD {norad}"
        sats.append(Sat(name=name, norad=norad, epoch=epoch, line1=l1, line2=l2))
        i += 2
    return sats


def load_tle_dir(tle_dir: os.PathLike | str) -> List[Sat]:  # noqa: F821
    """Every TLE from every ``*.tle`` / ``*.txt`` file in ``tle_dir``.

    Duplicates (same NORAD id and epoch across snapshots) are collapsed, so
    dropping overlapping daily files in is harmless.
    """
    root = Path(tle_dir)
    if not root.exists():
        return []
    seen = set()
    out: List[Sat] = []
    for path in sorted(list(root.glob("*.tle")) + list(root.glob("*.txt"))):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for sat in parse_tle_text(text):
            key = (sat.norad, sat.line1[18:32])
            if key in seen:
                continue
            seen.add(key)
            out.append(sat)
    return out


def filter_dtc(sats: Sequence[Sat]) -> List[Sat]:
    """Keep only Direct-to-Cell satellites (``[DTC]`` in the name)."""
    return [s for s in sats if s.is_dtc]


def tle_fingerprint(tle_dir: os.PathLike | str) -> str:
    """Cheap signature of a TLE folder's contents, for cache invalidation."""
    import hashlib

    root = Path(tle_dir)
    if not root.exists():
        return "none"
    parts = []
    for path in sorted(list(root.glob("*.tle")) + list(root.glob("*.txt"))):
        try:
            st = path.stat()
            parts.append(f"{path.name}:{st.st_size}:{int(st.st_mtime)}")
        except OSError:
            continue
    if not parts:
        return "empty"
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def nearest_by_epoch(sats: Sequence[Sat], when: datetime) -> List[Sat]:
    """For each NORAD id, the single ``Sat`` whose epoch is closest to ``when``."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    best: Dict[int, Sat] = {}
    for s in sats:
        cur = best.get(s.norad)
        if cur is None or abs((s.epoch - when).total_seconds()) < abs(
            (cur.epoch - when).total_seconds()
        ):
            best[s.norad] = s
    return list(best.values())


def _ssl_context():
    """An SSL context with a working CA bundle.

    The python.org macOS builds ship without system CA certificates unless
    "Install Certificates.command" was run, which makes HTTPS verification fail
    with ``CERTIFICATE_VERIFY_FAILED``. Prefer certifi's bundle when present so
    the fetch works out of the box; fall back to the stdlib default otherwise.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # certifi missing → use whatever the platform provides
        return ssl.create_default_context()


def fetch_celestrak(
    tle_dir: os.PathLike | str,  # noqa: F821
    url: str = CELESTRAK_URL,
    when: Optional[datetime] = None,
    force: bool = False,
    timeout: int = 30,
) -> Path:
    """Download the current Starlink TLE set to ``tle_dir/celestrak_YYYY-MM-DD.tle``.

    A no-op if today's file already exists (unless ``force``) — call it daily and
    it fetches at most once a day. Requires network access; raises otherwise.
    """
    when = when or datetime.now(timezone.utc)
    root = Path(tle_dir)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"celestrak_{when:%Y-%m-%d}.tle"
    if dest.exists() and not force:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "qpviz/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        data = resp.read().decode("utf-8", errors="ignore")
    if "1 " not in data:  # crude sanity check on the payload
        raise RuntimeError("CelesTrak returned no TLE data")
    dest.write_text(data)
    return dest


# --------------------------------------------------------------------------- #
# Space-Track historical archive
# --------------------------------------------------------------------------- #
#
# CelesTrak only serves *today's* elements, so accurate SGP4 for a past
# measurement date needs a TLE from around then. Those come from Space-Track's
# ``gp_history`` archive (free account). Two wrinkles this handles:
#   * Space-Track has no "D2C" flag, and its names carry no ``[DTC]`` tag. So we
#     first learn which Starlink NORAD ids are D2C from CelesTrak's current tags,
#     then pull only those ids' history — and re-tag the names ``[DTC]`` on the
#     way out, so the rest of the app (which filters on the tag) just works.
#   * The id list is long, so queries are chunked and paced politely.


def dtc_norad_ids(tle_text: str) -> List[int]:
    """NORAD ids of the D2C satellites in a CelesTrak Starlink TLE dump."""
    return sorted({s.norad for s in filter_dtc(parse_tle_text(tle_text))})


def dtc_norad_ids_from_dir(tle_dir: os.PathLike | str) -> List[int]:  # noqa: F821
    """D2C NORAD ids already present in a TLE folder (no network)."""
    return sorted({s.norad for s in filter_dtc(load_tle_dir(tle_dir))})


def dtc_norad_ids_from_celestrak(
    url: str = CELESTRAK_URL, timeout: int = 30, retries: int = 3
) -> List[int]:
    """Fetch CelesTrak's Starlink set and return the current D2C NORAD ids.

    CelesTrak rate-limits and answers 403 when hit too often; retry with backoff
    and a browser-like User-Agent. Prefer :func:`dtc_norad_ids_from_dir` when a
    recent local snapshot already exists.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; qpviz/1.0)"}
    last: Exception = RuntimeError("no attempt")
    for attempt in range(max(1, retries)):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
                return dtc_norad_ids(resp.read().decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as exc:  # 403/429 are transient rate limits
            last = exc
            if exc.code in (403, 429) and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except Exception as exc:  # pragma: no cover - network variability
            last = exc
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
    raise last


def spacetrack_query_url(
    norad_ids: Sequence[int], start: str, end: str, fmt: str = "3le"
) -> str:
    """Build a Space-Track ``gp_history`` REST query URL.

    ``start``/``end`` are ``YYYY-MM-DD`` (inclusive EPOCH range). ``3le`` returns
    three-line sets (a ``0 NAME`` line + the two element lines).
    """
    ids = ",".join(str(int(n)) for n in norad_ids)
    return (
        f"{SPACETRACK_BASE}/basicspacedata/query/class/gp_history"
        f"/NORAD_CAT_ID/{ids}"
        f"/EPOCH/{start}--{end}"
        f"/orderby/NORAD_CAT_ID,EPOCH/format/{fmt}"
    )


def tag_dtc_names(text: str) -> str:
    """Ensure every ``0 NAME`` line carries a ``[DTC]`` tag.

    Space-Track names have no tag; since we only ever fetch known-D2C ids, we add
    it so the app's ``[DTC]`` filter keeps them.
    """
    out = []
    for line in text.splitlines():
        if line.startswith("0 ") and not DTC_RE.search(line):
            out.append(line.rstrip() + " [DTC]")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def _spacetrack_opener(user: str, password: str, timeout: int = 60):
    """A cookie-carrying urllib opener, logged in to Space-Track."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        urllib.request.HTTPSHandler(context=_ssl_context()),
    )
    data = urllib.parse.urlencode(
        {"identity": user, "password": password}
    ).encode()
    with opener.open(f"{SPACETRACK_BASE}/ajaxauth/login", data=data, timeout=timeout) as r:
        body = r.read().decode("utf-8", errors="ignore")
    if "Failed" in body or not any(c.name == "chocolatechip" for c in jar):
        raise RuntimeError("Space-Track login failed — check identity/password.")
    return opener


def fetch_spacetrack_archive(
    user: str,
    password: str,
    start: str,
    end: str,
    tle_dir: os.PathLike | str,  # noqa: F821
    norad_ids: Optional[Sequence[int]] = None,
    chunk: int = 120,
    pause: float = 3.0,
    timeout: int = 90,
) -> Path:
    """Download D2C ``gp_history`` for ``start``→``end`` into ``tle_dir``.

    If ``norad_ids`` is None the current D2C set is learned from CelesTrak.
    Queries are chunked (``chunk`` ids each) and paced (``pause`` seconds between)
    to respect Space-Track's rate limits. Returns the written file.
    """
    ids = list(norad_ids) if norad_ids is not None else dtc_norad_ids_from_celestrak()
    if not ids:
        raise RuntimeError("No D2C NORAD ids to query.")
    root = Path(tle_dir)
    root.mkdir(parents=True, exist_ok=True)
    opener = _spacetrack_opener(user, password, timeout=timeout)

    pieces: List[str] = []
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        url = spacetrack_query_url(batch, start, end)
        with opener.open(url, timeout=timeout) as r:
            pieces.append(r.read().decode("utf-8", errors="ignore"))
        if i + chunk < len(ids):
            time.sleep(pause)

    text = tag_dtc_names("\n".join(p for p in pieces if p.strip()))
    dest = root / f"spacetrack_{start}_{end}.tle"
    dest.write_text(text)
    return dest


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def _gmst_rad(jd_ut1: np.ndarray) -> np.ndarray:
    """Greenwich Mean Sidereal Time (radians) from Julian date (IAU-82).

    UT1 ≈ UTC here (sub-second difference is negligible for a satellite
    elevation to a fraction of a degree).
    """
    tut1 = (jd_ut1 - 2451545.0) / 36525.0
    seconds = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 0.093104 * tut1 * tut1
        - 6.2e-6 * tut1 * tut1 * tut1
    )
    return np.deg2rad((seconds % 86400.0) / 240.0)  # 1 s = 1/240 deg


def observer_ecef(lat_deg, lon_deg, alt_km=0.0):
    """WGS-84 geodetic → ECEF (km). Accepts scalars or arrays."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype="float64"))
    lon = np.deg2rad(np.asarray(lon_deg, dtype="float64"))
    alt = np.asarray(alt_km, dtype="float64")
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - _WGS84_E2) + alt) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def ecef_to_geodetic(r_ecef: np.ndarray):
    """ECEF (km) → (lat_deg, lon_deg, alt_km) on WGS-84 (Bowring's method)."""
    r = np.atleast_2d(np.asarray(r_ecef, dtype="float64"))
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    b = _WGS84_A * np.sqrt(1.0 - _WGS84_E2)
    ep2 = (_WGS84_A ** 2 - b ** 2) / b ** 2
    theta = np.arctan2(z * _WGS84_A, p * b)
    lat = np.arctan2(
        z + ep2 * b * np.sin(theta) ** 3,
        p - _WGS84_E2 * _WGS84_A * np.cos(theta) ** 3,
    )
    n = _WGS84_A / np.sqrt(1.0 - _WGS84_E2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    out = (np.degrees(lat), np.degrees(lon), alt)
    if np.asarray(r_ecef).ndim == 1:
        return float(out[0][0]), float(out[1][0]), float(out[2][0])
    return out


def teme_to_ecef(r_teme: np.ndarray, jd: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """Rotate TEME position(s) (km) to ECEF about the pole by GMST.

    ``r_teme`` is ``(N, 3)``; ``jd + fr`` are the matching Julian dates.
    """
    theta = _gmst_rad(jd + fr)
    c, s = np.cos(theta), np.sin(theta)
    x, y, z = r_teme[:, 0], r_teme[:, 1], r_teme[:, 2]
    xe = c * x + s * y
    ye = -s * x + c * y
    return np.stack([xe, ye, z], axis=-1)


def elevation_deg(
    sat_ecef: np.ndarray, obs_ecef: np.ndarray, lat_deg, lon_deg
) -> np.ndarray:
    """Topocentric elevation (degrees) of ``sat_ecef`` seen from ``obs_ecef``.

    All arrays are ``(N, 3)`` / ``(N,)`` and paired element-wise.
    """
    lat = np.deg2rad(np.asarray(lat_deg, dtype="float64"))
    lon = np.deg2rad(np.asarray(lon_deg, dtype="float64"))
    d = sat_ecef - obs_ecef
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_lon, cos_lon = np.sin(lon), np.cos(lon)
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    rng = np.sqrt(dx * dx + dy * dy + dz * dz)
    with np.errstate(invalid="ignore", divide="ignore"):
        el = np.degrees(np.arcsin(np.clip(up / rng, -1.0, 1.0)))
    return el


# --------------------------------------------------------------------------- #
# Visibility
# --------------------------------------------------------------------------- #


def _jday_arrays(times_utc: Sequence[datetime]) -> Tuple[np.ndarray, np.ndarray]:
    jd = np.empty(len(times_utc), dtype="float64")
    fr = np.empty(len(times_utc), dtype="float64")
    for i, t in enumerate(times_utc):
        if t.tzinfo is not None:
            t = t.astimezone(timezone.utc).replace(tzinfo=None)
        j, f = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
        jd[i], fr[i] = j, f
    return jd, fr


@dataclass
class VisibilityResult:
    """Per-sample visibility. Arrays are aligned with the input samples."""

    max_elev: np.ndarray  # highest satellite elevation (deg), NaN if none
    counts: Dict[int, np.ndarray]  # mask (deg) -> count of sats at/above it
    n_sats: int  # how many satellites were propagated
    masks: Tuple[int, ...]


def visibility_counts(
    times_utc: Sequence[datetime],
    lats: Sequence[float],
    lons: Sequence[float],
    sats: Sequence[Sat],
    masks: Sequence[int] = DEFAULT_MASKS,
    alt_km: float = 0.0,
) -> VisibilityResult:
    """Count satellites at/above each mask for every sample.

    ``sats`` should already be the set to count (e.g. D2C only) and already
    reduced to one TLE per satellite (see :func:`nearest_by_epoch`). Propagation
    is vectorised over all samples, one satellite at a time.
    """
    if not HAVE_SGP4:
        raise RuntimeError("sgp4 is not installed")
    n = len(times_utc)
    masks = tuple(int(m) for m in masks)
    counts = {m: np.zeros(n, dtype="int32") for m in masks}
    max_elev = np.full(n, -np.inf, dtype="float64")
    if n == 0 or not sats:
        return VisibilityResult(
            max_elev=np.where(np.isfinite(max_elev), max_elev, np.nan),
            counts=counts, n_sats=len(sats), masks=masks,
        )

    jd, fr = _jday_arrays(times_utc)
    lat = np.asarray(lats, dtype="float64")
    lon = np.asarray(lons, dtype="float64")
    obs = observer_ecef(lat, lon, alt_km)

    for sat in sats:
        rec = sat.satrec()
        err, r, _v = rec.sgp4_array(jd, fr)  # r: (N,3) TEME km
        r = np.asarray(r, dtype="float64")
        ok = np.asarray(err) == 0
        r_ecef = teme_to_ecef(r, jd, fr)
        el = elevation_deg(r_ecef, obs, lat, lon)
        el = np.where(ok, el, -np.inf)
        np.maximum(max_elev, el, out=max_elev)
        for m in masks:
            counts[m] += (el >= m).astype("int32")

    max_elev = np.where(np.isfinite(max_elev), max_elev, np.nan)
    return VisibilityResult(max_elev=max_elev, counts=counts, n_sats=len(sats), masks=masks)


def distinct_visible_ids(
    times_utc: Sequence[datetime],
    lats: Sequence[float],
    lons: Sequence[float],
    sats: Sequence[Sat],
    mask_deg: float,
    alt_km: float = 0.0,
) -> set:
    """NORAD ids of satellites that rose to/above ``mask_deg`` at *any* sample.

    Where :func:`visibility_counts` answers "how many are up right now", this
    answers "how many *different* satellites passed through the window" — the
    number that grows with the window length even though only a handful are up at
    once. ``sats`` should be one TLE per satellite (see :func:`nearest_by_epoch`).
    """
    if not HAVE_SGP4:
        raise RuntimeError("sgp4 is not installed")
    if len(times_utc) == 0 or not sats:
        return set()
    jd, fr = _jday_arrays(times_utc)
    lat = np.asarray(lats, dtype="float64")
    lon = np.asarray(lons, dtype="float64")
    obs = observer_ecef(lat, lon, alt_km)
    mask = float(mask_deg)
    seen = set()
    for sat in sats:
        rec = sat.satrec()
        err, r, _v = rec.sgp4_array(jd, fr)
        r = np.asarray(r, dtype="float64")
        ok = np.asarray(err) == 0
        el = elevation_deg(teme_to_ecef(r, jd, fr), obs, lat, lon)
        if np.any(ok & (el >= mask)):
            seen.add(sat.norad)
    return seen


def distinct_visible_count(
    frame,
    sats: Sequence[Sat],
    mask_deg: float,
    alt_km: float = 0.0,
    utc_offset_hours: float = 0.0,
    max_points: int = 4000,
) -> int:
    """How many *distinct* D2C satellites were ever visible across ``frame``.

    Operates on the geolocated rows, bucketed by UTC date (each bucket uses the
    nearest-epoch TLE), and unions the satellites seen. To stay fast on a large
    selection the rows are evenly thinned to ``max_points`` first — a satellite
    stays above the horizon for minutes, so a thinned time grid still catches
    every distinct pass.
    """
    import pandas as pd

    if not sats:
        return 0
    t = pd.to_datetime(frame["time"], errors="coerce")
    lat = pd.to_numeric(frame["lat"], errors="coerce")
    lon = pd.to_numeric(frame["lon"], errors="coerce")
    good = t.notna() & lat.notna() & lon.notna() & (lat.abs() > 1e-9) & (lon.abs() > 1e-9)
    idx = np.where(good.to_numpy())[0]
    if idx.size == 0:
        return 0
    if idx.size > max_points:
        idx = idx[np.round(np.linspace(0, idx.size - 1, max_points)).astype(int)]
        idx = np.unique(idx)

    t_utc = (t - pd.Timedelta(hours=utc_offset_hours)).iloc[idx]
    lat_s = lat.iloc[idx]
    lon_s = lon.iloc[idx]
    seen = set()
    for day, day_pos in pd.Series(range(len(idx))).groupby(t_utc.dt.floor("D").to_numpy()):
        pos = day_pos.to_numpy()
        when = pd.Timestamp(day).to_pydatetime().replace(tzinfo=timezone.utc)
        day_sats = nearest_by_epoch(sats, when)
        times = [ts.to_pydatetime() for ts in t_utc.iloc[pos]]
        seen |= distinct_visible_ids(
            times, lat_s.iloc[pos].to_numpy(), lon_s.iloc[pos].to_numpy(),
            day_sats, mask_deg, alt_km=alt_km,
        )
    return len(seen)


def annotate_frame(
    frame,
    sats: Sequence[Sat],
    masks: Sequence[int] = DEFAULT_MASKS,
    alt_km: float = 0.0,
    utc_offset_hours: float = 0.0,
):
    """Add ``sat_vis_<mask>`` + ``sat_max_elev`` columns to a geolocated frame.

    Only rows with a finite lat/lon and time are computed (others get NaN).
    Samples are bucketed by UTC date and each bucket is propagated from the TLE
    snapshot nearest that date, so a mixed-date frame still gets epoch-matched
    geometry. ``utc_offset_hours`` converts the measurement clock to UTC (e.g.
    -4 if the logs are in US/Eastern daylight time).
    """
    import pandas as pd

    out = frame.copy()
    masks = tuple(int(m) for m in masks)
    for m in masks:
        out[f"sat_vis_{m}"] = pd.array([pd.NA] * len(out), dtype="Int32")
    out["sat_max_elev"] = np.nan
    if len(out) == 0 or not sats:
        return out

    t = pd.to_datetime(out["time"], errors="coerce")
    lat = pd.to_numeric(out["lat"], errors="coerce")
    lon = pd.to_numeric(out["lon"], errors="coerce")
    good = t.notna() & lat.notna() & lon.notna() & (lat.abs() > 1e-9) & (lon.abs() > 1e-9)
    if not good.any():
        return out

    # measurement clock -> UTC
    t_utc = t - pd.Timedelta(hours=utc_offset_hours)
    idx_all = np.where(good.to_numpy())[0]
    dates = t_utc.dt.floor("D")

    for day, day_idx in pd.Series(idx_all).groupby(dates.iloc[idx_all].to_numpy()):
        rows = day_idx.to_numpy()
        when = pd.Timestamp(day).to_pydatetime().replace(tzinfo=timezone.utc)
        day_sats = nearest_by_epoch(sats, when)
        times = [ts.to_pydatetime() for ts in t_utc.iloc[rows]]
        res = visibility_counts(
            times, lat.iloc[rows].to_numpy(), lon.iloc[rows].to_numpy(),
            day_sats, masks=masks, alt_km=alt_km,
        )
        for m in masks:
            col = out.columns.get_loc(f"sat_vis_{m}")
            out.iloc[rows, col] = res.counts[m].astype("int64")
        out.iloc[rows, out.columns.get_loc("sat_max_elev")] = res.max_elev

    return out


def archive_summary(sats: Sequence[Sat]) -> dict:
    """Small dict describing a loaded TLE set - for the sidebar caption."""
    dtc = filter_dtc(sats)
    epochs = [s.epoch for s in sats]
    return {
        "objects": len(sats),
        "dtc": len(dtc),
        "dtc_unique": len({s.norad for s in dtc}),
        "epoch_min": min(epochs) if epochs else None,
        "epoch_max": max(epochs) if epochs else None,
    }


# --------------------------------------------------------------------------- #
# CLI - for a daily "fetch the latest TLEs" cron / launchd job
# --------------------------------------------------------------------------- #


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m qpviz.satellites",
        description="Fetch Starlink TLEs (daily) or summarise a TLE folder.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download today's Starlink TLEs into a folder")
    f.add_argument("tle_dir", nargs="?", default=os.environ.get("TLE_DATA_DIR", "TLE"))
    f.add_argument("--force", action="store_true", help="re-download even if today's file exists")
    s = sub.add_parser("summary", help="report how many D2C satellites a folder holds")
    s.add_argument("tle_dir", nargs="?", default=os.environ.get("TLE_DATA_DIR", "TLE"))

    a = sub.add_parser(
        "archive",
        help="download D2C history from Space-Track for a date range (needs a login)",
    )
    a.add_argument("--start", required=True, help="first date, YYYY-MM-DD")
    a.add_argument("--end", required=True, help="last date, YYYY-MM-DD")
    a.add_argument("--tle-dir", default=os.environ.get("TLE_DATA_DIR", "TLE"))
    a.add_argument("--user", default=os.environ.get("SPACETRACK_USER"),
                   help="Space-Track identity (or set SPACETRACK_USER)")
    a.add_argument("--password", default=os.environ.get("SPACETRACK_PASS"),
                   help="Space-Track password (or set SPACETRACK_PASS; prompts if omitted)")
    a.add_argument("--pad-days", type=int, default=1,
                   help="widen the epoch range this many days each side (default 1)")
    args = parser.parse_args(argv)

    if args.cmd == "fetch":
        path = fetch_celestrak(args.tle_dir, force=args.force)
        info = archive_summary(load_tle_dir(args.tle_dir))
        print(f"Saved {path}")
        print(f"{info['dtc_unique']} D2C satellites of {info['objects']} objects in {args.tle_dir}")
    elif args.cmd == "summary":
        info = archive_summary(load_tle_dir(args.tle_dir))
        span = (
            f"{info['epoch_min']:%Y-%m-%d} → {info['epoch_max']:%Y-%m-%d}"
            if info["epoch_min"] else "—"
        )
        print(f"{info['objects']} objects, {info['dtc_unique']} D2C satellites, epochs {span}")
    elif args.cmd == "archive":
        user = args.user or input("Space-Track identity (email): ").strip()
        password = args.password
        if not password:
            import getpass

            password = getpass.getpass("Space-Track password: ")
        start = (datetime.strptime(args.start, "%Y-%m-%d") - _timedelta_days(args.pad_days)).strftime("%Y-%m-%d")
        end = (datetime.strptime(args.end, "%Y-%m-%d") + _timedelta_days(args.pad_days)).strftime("%Y-%m-%d")
        # Prefer the D2C ids already in the TLE folder (from a recent `fetch`);
        # only hit CelesTrak if the folder has none, to dodge its rate limit.
        ids = dtc_norad_ids_from_dir(args.tle_dir)
        source = "local TLE folder"
        if not ids:
            print("Learning D2C satellites from CelesTrak…")
            ids = dtc_norad_ids_from_celestrak()
            source = "CelesTrak"
        print(f"{len(ids)} D2C satellites (from {source}); "
              f"querying Space-Track {start}→{end} (chunked)…")
        path = fetch_spacetrack_archive(user, password, start, end, args.tle_dir, norad_ids=ids)
        info = archive_summary(load_tle_dir(args.tle_dir))
        print(f"Saved {path}")
        print(f"TLE folder now holds {info['dtc_unique']} D2C satellites, "
              f"{info['dtc']} element sets.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
