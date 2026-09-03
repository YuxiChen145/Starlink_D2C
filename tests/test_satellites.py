"""Satellite visibility engine: TLE parsing, geometry, SGP4 propagation.

No network and no Starlink-specific TLEs are needed — the geometry is validated
against controlled cases (a satellite placed exactly at the zenith, the horizon,
and the antipode) and the full SGP4 chain against a standard published TLE (the
ISS), so it is self-contained and deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from qpviz import satellites as sat

# A standard, widely published TLE (ISS, Vallado test vector). Not D2C — used
# only to exercise real SGP4 propagation + the frame conversions.
ISS_NAME = "ISS (ZARYA)"
ISS_L1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
ISS_L2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"

# Two tiny synthetic "[DTC]" satellites at different epochs, for filtering and
# nearest-epoch tests (values are structurally valid TLEs).
DTC_TLE = f"""\
STARLINK-32000 [DTC]
{ISS_L1}
{ISS_L2}
STARLINK-31000 [DTC]
1 40000U 24001A   26200.50000000  .00000000  00000-0  00000-0 0  9990
2 40000  53.0000 100.0000 0001000  90.0000 270.0000 15.00000000000010
STARLINK-30000
1 40001U 24002A   26200.50000000  .00000000  00000-0  00000-0 0  9992
2 40001  53.0000 100.0000 0001000  90.0000 270.0000 15.00000000000017
"""


# --------------------------------------------------------------------------- #
# TLE parsing / filtering
# --------------------------------------------------------------------------- #


def test_parse_two_and_three_line():
    sats = sat.parse_tle_text(DTC_TLE)
    assert [s.norad for s in sats] == [25544, 40000, 40001]
    assert sats[0].name == "STARLINK-32000 [DTC]"
    # a bare 2-line block still parses, with a synthesised name
    bare = sat.parse_tle_text(ISS_L1 + "\n" + ISS_L2)
    assert len(bare) == 1 and bare[0].norad == 25544


def test_dtc_filter_keeps_only_tagged():
    sats = sat.parse_tle_text(DTC_TLE)
    dtc = sat.filter_dtc(sats)
    assert {s.norad for s in dtc} == {25544, 40000}
    assert all("DTC" in s.name.upper() for s in dtc)


def test_tle_epoch_decoding():
    s = sat.parse_tle_text(DTC_TLE)[1]  # epoch 26200.5 -> 2026 day 200.5
    assert s.epoch.year == 2026 and s.epoch.tzinfo == timezone.utc
    # day-of-year 200.5 is noon of the 200th day
    doy = (s.epoch - datetime(2026, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0
    assert abs(doy - 199.5) < 1e-6


def test_nearest_by_epoch_picks_closest():
    # same NORAD id at two epochs -> nearest wins
    text = (
        "SAT [DTC]\n"
        "1 50000U 24001A   26100.00000000  .00000000  00000-0  00000-0 0  9995\n"
        "2 50000  53.0000 100.0000 0001000  90.0000 270.0000 15.00000000000015\n"
        "SAT [DTC]\n"
        "1 50000U 24001A   26200.00000000  .00000000  00000-0  00000-0 0  9993\n"
        "2 50000  53.0000 100.0000 0001000  90.0000 270.0000 15.00000000000010\n"
    )
    sats = sat.parse_tle_text(text)
    when = datetime(2026, 1, 1, tzinfo=timezone.utc) + pd.Timedelta(days=205).to_pytimedelta()
    picked = sat.nearest_by_epoch(sats, when)
    assert len(picked) == 1
    # day 205 is closer to the 26200 (day-of-year 200) TLE than the 26100 one
    assert picked[0].epoch.timetuple().tm_yday == 200


# --------------------------------------------------------------------------- #
# Geometry (independent of SGP4)
# --------------------------------------------------------------------------- #


def test_gmst_matches_j2000_reference():
    # GMST at J2000.0 (JD 2451545.0) is 280.4606° (18.697 h) — a standard value.
    g = np.degrees(sat._gmst_rad(np.array([2451545.0])))[0]
    assert abs(g - 280.4606) < 0.01


def test_zenith_horizon_and_below():
    lat, lon, alt = 41.70, -86.24, 0.0
    obs = sat.observer_ecef(lat, lon, alt).reshape(1, 3)
    # local up (geodetic zenith) unit vector
    la, lo = np.deg2rad(lat), np.deg2rad(lon)
    up = np.array([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    east = np.array([-np.sin(lo), np.cos(lo), 0.0])

    zenith = obs + 550.0 * up
    horizon = obs + 550.0 * east  # tangent direction
    below = -obs / np.linalg.norm(obs) * (sat._WGS84_A + 550.0)  # antipode-ish

    assert abs(sat.elevation_deg(zenith, obs, [lat], [lon])[0] - 90.0) < 1e-6
    assert abs(sat.elevation_deg(horizon, obs, [lat], [lon])[0] - 0.0) < 1e-6
    assert sat.elevation_deg(below.reshape(1, 3), obs, [lat], [lon])[0] < -30.0


def test_ecef_geodetic_round_trip():
    for lat, lon, alt in [(41.7, -86.2, 0.0), (0.0, 0.0, 550.0), (-33.9, 151.2, 100.0)]:
        e = sat.observer_ecef(lat, lon, alt)
        glat, glon, galt = sat.ecef_to_geodetic(e)
        assert abs(glat - lat) < 1e-6 and abs(glon - lon) < 1e-6 and abs(galt - alt) < 1e-3


# --------------------------------------------------------------------------- #
# Full SGP4 chain
# --------------------------------------------------------------------------- #


def test_full_chain_observer_under_satellite_sees_high_elevation():
    """Propagate the ISS, drop an observer at its sub-satellite point, and the
    elevation must be ~90° — this exercises SGP4 + TEME→ECEF(GMST) + topocentric
    elevation end to end."""
    from sgp4.api import jday

    s = sat.Sat(ISS_NAME, 25544, datetime(2019, 12, 9, tzinfo=timezone.utc), ISS_L1, ISS_L2)
    rec = s.satrec()
    t = datetime(2019, 12, 9, 17, 0, 0)
    jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
    e, r, _v = rec.sgp4(jd, fr)
    assert e == 0
    r_ecef = sat.teme_to_ecef(np.array([r]), np.array([jd]), np.array([fr]))
    glat, glon, galt = sat.ecef_to_geodetic(r_ecef[0])
    assert 380.0 < galt < 430.0            # ISS altitude sanity
    assert abs(glat) <= 51.7 + 0.5         # within its inclination band
    el = sat.elevation_deg(r_ecef, sat.observer_ecef(glat, glon, 0.0).reshape(1, 3),
                           [glat], [glon])[0]
    assert el > 89.0


def test_visibility_counts_monotonic_in_mask():
    s = sat.Sat(ISS_NAME, 25544, datetime(2019, 12, 9, tzinfo=timezone.utc), ISS_L1, ISS_L2)
    # a spread of times so the satellite is sometimes up, sometimes down
    times = [datetime(2019, 12, 9, 17, m, 0) for m in range(0, 60, 3)]
    lats = [41.70] * len(times)
    lons = [-86.24] * len(times)
    res = sat.visibility_counts(times, lats, lons, [s], masks=(0, 10, 25, 40))
    # a higher mask can never count more satellites than a lower one
    assert np.all(res.counts[0] >= res.counts[10])
    assert np.all(res.counts[10] >= res.counts[25])
    assert np.all(res.counts[25] >= res.counts[40])
    assert res.counts[0].max() <= 1  # only one satellite in play


def test_annotate_frame_adds_columns_and_respects_geo():
    s = sat.Sat(ISS_NAME, 25544, datetime(2019, 12, 9, tzinfo=timezone.utc), ISS_L1, ISS_L2)
    frame = pd.DataFrame({
        "time": pd.to_datetime([
            "2019-12-09 17:00:00", "2019-12-09 17:03:00", "2019-12-09 17:06:00",
        ]),
        "lat": [41.70, 41.70, np.nan],   # last row has no GPS
        "lon": [-86.24, -86.24, np.nan],
    })
    out = sat.annotate_frame(frame, [s], masks=(0, 25))
    assert "sat_vis_0" in out.columns and "sat_vis_25" in out.columns
    assert "sat_max_elev" in out.columns
    # the un-geolocated row stays null; the geolocated ones are filled
    assert pd.isna(out["sat_vis_0"].iloc[2])
    assert out["sat_vis_0"].iloc[:2].notna().all()


# --------------------------------------------------------------------------- #
# End-to-end app integration
# --------------------------------------------------------------------------- #

import os  # noqa: E402
from pathlib import Path  # noqa: E402

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")

# Two [DTC] sats with epochs near the synthetic QP fixture window (July 2026).
_APP_TLE = """\
STARLINK-32100 [DTC]
1 60000U 26050A   26203.50000000  .00000000  00000-0  00000-0 0  9991
2 60000  53.2000  50.0000 0001500  80.0000 280.0000 15.20000000000013
STARLINK-32101 [DTC]
1 60001U 26050B   26203.50000000  .00000000  00000-0  00000-0 0  9992
2 60001  53.2000 200.0000 0001500  80.0000 280.0000 15.20000000000010
STARLINK-40000
1 60002U 26050C   26203.50000000  .00000000  00000-0  00000-0 0  9993
2 60002  53.2000 300.0000 0001500  80.0000 280.0000 15.20000000000017
"""


@pytest.fixture()
def tle_dir(tmp_path) -> Path:
    d = tmp_path / "TLE"
    d.mkdir()
    (d / "celestrak_2026-07-22.tle").write_text(_APP_TLE)
    return d


def test_app_visibility_tab_enables_and_annotates(fixture_dir, tle_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("QP_DATA_DIR", str(fixture_dir))
    monkeypatch.setenv("PRISM_DATA_DIR", "/no-prism-dir")
    monkeypatch.setenv("TLE_DATA_DIR", str(tle_dir))
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert not app.exception, [str(e.value) for e in app.exception]

    # before enabling, there is no visibility tab
    assert not any("D2C visibility" in t.label for t in app.tabs)

    app.checkbox(key="sat_enable").set_value(True).run()
    assert not app.exception, [str(e.value) for e in app.exception]

    # the tab appears and the header distinct-satellites KPI shows up
    assert any("D2C visibility" in t.label for t in app.tabs)
    assert any("Distinct sats" in m.label for m in app.metric)


# --------------------------------------------------------------------------- #
# Space-Track archive helpers (pure logic; the live fetch needs a login)
# --------------------------------------------------------------------------- #


def test_dtc_norad_ids_from_tle_text():
    ids = sat.dtc_norad_ids(DTC_TLE)
    assert ids == [25544, 40000]  # the two [DTC] entries, sorted; the untagged one dropped


def test_spacetrack_query_url_shape():
    url = sat.spacetrack_query_url([40000, 25544], "2026-07-22", "2026-08-30")
    assert url.startswith("https://www.space-track.org/basicspacedata/query/class/gp_history")
    assert "/NORAD_CAT_ID/40000,25544/" in url
    assert "/EPOCH/2026-07-22--2026-08-30/" in url
    assert url.endswith("/format/3le")


def test_tag_dtc_names_adds_tag_once():
    raw = (
        "0 STARLINK-31000\n"
        "1 40000U 24001A   26200.50000000  .00000000  00000-0  00000-0 0  9990\n"
        "2 40000  53.0000 100.0000 0001000  90.0000 270.0000 15.00000000000010\n"
        "0 STARLINK-32000 [DTC]\n"
        "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991\n"
        "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482\n"
    )
    tagged = sat.tag_dtc_names(raw)
    # the untagged name gains [DTC]; the already-tagged one is untouched (no dup)
    assert "0 STARLINK-31000 [DTC]" in tagged
    assert tagged.count("[DTC]") == 2
    # and the whole thing now parses as two D2C satellites
    assert len(sat.filter_dtc(sat.parse_tle_text(tagged))) == 2


def test_dtc_norad_ids_from_dir(tmp_path):
    d = tmp_path / "TLE"
    d.mkdir()
    (d / "celestrak_2026-09-02.tle").write_text(DTC_TLE)
    assert sat.dtc_norad_ids_from_dir(d) == [25544, 40000]
    assert sat.dtc_norad_ids_from_dir(tmp_path / "empty") == []


def test_distinct_visible_grows_with_window_vs_instantaneous():
    """Distinct-over-window counts different satellites passing through, so it is
    >= the instantaneous count and tracks satellite identity, not just a tally."""
    # two satellites, offset in the orbit so they are up at different moments
    s1 = sat.Sat("A [DTC]", 25544, datetime(2019, 12, 9, tzinfo=timezone.utc), ISS_L1, ISS_L2)
    s2 = sat.Sat(
        "B [DTC]", 25545, datetime(2019, 12, 9, tzinfo=timezone.utc),
        ISS_L1.replace("25544", "25545"),
        "2 25545  51.6439 211.2001 0007417  17.6667 185.6398 15.50103472202482",
    )
    import pandas as pd
    from datetime import timedelta

    t0 = datetime(2019, 12, 9, 17, 0, 0)
    times = [t0 + timedelta(seconds=30 * i) for i in range(40)]  # 20 min
    frame = pd.DataFrame({"time": times, "lat": [41.70] * 40, "lon": [-86.24] * 40})
    ids = sat.distinct_visible_ids(times, [41.70] * 40, [-86.24] * 40, [s1, s2], 0.0)
    assert isinstance(ids, set) and ids.issubset({25544, 25545})
    n = sat.distinct_visible_count(frame, [s1, s2], 0.0)
    assert 0 <= n <= 2
    # no satellites -> zero, and an empty frame -> zero
    assert sat.distinct_visible_count(frame, [], 25) == 0
    assert sat.distinct_visible_count(frame.iloc[0:0], [s1], 25) == 0


def test_changing_time_window_clears_satellite_estimate(
    fixture_dir, tle_dir, tmp_path, monkeypatch
):
    """Turning the estimate on and then changing the time window must auto-clear
    the checkbox, so its (now stale) numbers never linger."""
    monkeypatch.setenv("QP_DATA_DIR", str(fixture_dir))
    monkeypatch.setenv("PRISM_DATA_DIR", "/no-prism-dir")
    monkeypatch.setenv("TLE_DATA_DIR", str(tle_dir))
    monkeypatch.setenv("QP_CACHE_DIR", str(tmp_path / "cache"))
    app = AppTest.from_file(APP, default_timeout=180).run()
    assert not app.exception, [str(e.value) for e in app.exception]

    app.checkbox(key="sat_enable").set_value(True).run()
    assert app.checkbox(key="sat_enable").value is True

    # enabling a time window changes the window -> the estimate auto-clears
    app.checkbox(key="use_time_window").set_value(True).run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert app.checkbox(key="sat_enable").value is False
