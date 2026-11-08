from astropy.time import Time
from astropy.coordinates import get_sun
import astropy.units as u

import numpy as np


R_SUN_KM = 695_700.0

def unit(x):
    return x / np.linalg.norm(x, axis=0)

def sep_deg(sun_v, sat_v):
    """Angle (deg) between Sun and satellite direction vectors, both (3, N)."""
    c = np.sum(unit(sun_v) * unit(sat_v), axis=0)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

def sky_rate(pos, vel):
    """Angular speed across the sky (deg/s) from topocentric position and velocity, (3, N)."""
    r = np.linalg.norm(pos, axis=0)
    return np.degrees(np.linalg.norm(np.cross(pos, vel, axis=0), axis=0) / r**2)

def sun_at(tt, eph, observer):
    app = (eph["earth"] + observer).at(tt).observe(eph["sun"]).apparent()
    return app.position.km, app.distance().km


def refine(sat, s0, t0, DT, FINE_DT, TOL_DEG, THETA_BOUND, observer, eph):
    fine = np.arange(s0 - DT, s0 + DT, FINE_DT)
    tf = t0 + fine / 86400

    sun_v, sun_dist = sun_at(tf, eph, observer)
    topo = (sat - observer).at(tf)
    v = topo.position.km
    sep = sep_deg(sun_v, v)

    j = int(np.argmin(sep))
    sun_r = np.degrees(np.arcsin(R_SUN_KM / sun_dist[j]))   # apparent solar radius at that time
    if sep[j] > sun_r + TOL_DEG:
        return None

    alt, az, rng = topo.altaz()
    rng_m = rng.km[j] * 1e3
    return {
        "time_utc":     tf[j].utc_iso(places=2),
        "name":         sat.name,
        "norad_id":     sat.model.satnum,
        "min_sep_deg":  sep[j],
        "sun_r_deg":    sun_r,
        "in_disc":      sep[j] < sun_r,
        "duration_s":   np.sum(sep < sun_r) * FINE_DT,
        "alt_deg":      alt.degrees[j],
        "az_deg":       az.degrees[j],
        "range_km":     rng.km[j],
        "rate_deg_s":   sky_rate(v[:, [j]], topo.velocity.km_per_s[:, [j]])[0],
        # smallest feature you could resolve at this range, using THETA_BOUND from your notebook
        "min_resolvable_m": 2 * rng_m * np.tan(np.radians(THETA_BOUND / 3600) / 2),
        "tle_age_days": tf[j] - sat.epoch,
    }

