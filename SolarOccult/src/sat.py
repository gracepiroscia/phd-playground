import numpy as np
import json

from skyfield.api import load, wgs84
from spacetrack import SpaceTrackClient


MAX_PERIOD_MIN = 128      # LEO cutoff (minutes)
ST_USER =  "pirosciagrace@gmail.com"
ST_PASS = "vw.q_QsB2e!8kuN"

TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"

def norad_ids_by_size(sizes):
    """NORAD IDs of on-orbit objects with the given Space-Track RCS_SIZE values."""

    st = SpaceTrackClient(ST_USER, ST_PASS)
    rows = st.satcat(current="Y", decay=None, rcs_size=sizes, format="json")
    if isinstance(rows, str):
        rows = json.loads(rows)
    return {int(r["NORAD_CAT_ID"]) for r in rows}

def load_leo_satellites(RCS_SIZES: list[str] = None):
    filename="active.tle"
    stale = (not load.exists(filename)) or load.days_old(filename) > 0.25
    sats = load.tle_file(TLE_URL, reload=stale, filename=filename)

    sats = [s for s in sats if 2 * np.pi / s.model.no_kozai < MAX_PERIOD_MIN]  # no_kozai is rad/min
    if RCS_SIZES:
        keep = norad_ids_by_size(RCS_SIZES)
        sats = [s for s in sats if s.model.satnum in keep]
    return sats