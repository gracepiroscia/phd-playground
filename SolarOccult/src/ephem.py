from astropy.time import Time
from astropy.coordinates import get_sun
import astropy.units as u

ANGULAR_DIAM = 0.53  # degrees, average angular diameter of the Sun

def sun_ra_dec(date_string, latitude=-33.8596, elevation_limit=30):
    """
    Return the Sun's RA and Dec throughout
    the day, as visible by given observer. 

    Parameters
    ----------
    date_string : str
        In format 'YYYY-MM-DD', e.g.
        '2026-05-25'
    latitude : float
        Observer's latitude in degrees. Default
        is for Sydney Observatory= -33.8596
    elevation_limit : float
        Minimum elevation (in degrees) for the Sun
        to be considered "visible". Default is 30°.

    Returns
    -------
    dict
        {
            'ra_hours': Array,
            'dec_degrees': Array,
        }
    For viewable locations, given observer loc.
    """

    # Convert input time string into Astropy Time object
    t = Time(date_string)

    # Get Sun coordinates
    sun = get_sun(t)

    # RA/Dec
    ra = sun.ra
    dec = sun.dec

    return {
        "ra_hours": ra.hour,
        "ra_hms": ra.to_string(unit=u.hour, sep=':'),
        "dec_degrees": dec.degree,
        "dec_dms": dec.to_string(unit=u.degree, sep=':')
    }




# class of objects we will be able to see: