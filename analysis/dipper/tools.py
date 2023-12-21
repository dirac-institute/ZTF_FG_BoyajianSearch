"""
Tools for analysis.
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u


def prepare_lc(time, mag, mag_err, flag, band, band_of_study='r', flag_good=0, q=None, custom_q=False):
    """
    Prepare the light curve for analysis.
    
    Parameters:
    -----------
    time (array-like): Input time values.
    mag (array-like): Input magnitude values.
    mag_err (array-like): Input magnitude error values.
    flag (array-like): Input flag values.
    band (array-like): Input band values.
    band_of_study (str): Band to study. Default is 'r' band
    flag_good (int): Flag value for good detections. Default is 0 (see ZTF documentation)

    Returns:
    --------
    time (array-like): Output time values.
    mag (array-like): Output magnitude values.
    mag_err (array-like): Output magnitude error values.
    """
    
    if custom_q:
        rmv = q
    else:
        # Selection and preparation of the light curve (default selection on )
        rmv = (flag == flag_good) & (band==band_of_study')
    
    # TODO: I'm working with the numpy values because we had some unexpected issues. Check if stable.
    time, mag, mag_err = time[rmv].values, mag[rmv].values, mag_err[rmv].values
    
    # sort time
    srt = np.argsort(time)

    return time[srt], mag[srt], mag_err[srt]

def estimate_gaiadr3_density(ra_target, dec_target, radius=0.01667, gaia_lite_table=tbl):
    """
    Estimate the density of stars in the Gaia DR3 catalog around a given target.

    Parameters:
    -----------
    ra_target (float): Right ascension of the target in degrees.
    dec_target (float): Declination of the target in degrees.
    radius (float): Radius of the cone search in degrees. Default is 1 arcmin (1arcmin ~0.01667 deg).
    gaia_lite_table (LSDB hipscat table): Gaia DR3 table. Default is the one loaded in the notebook.

    Returns:
    --------
    closest_star_arcsec (float): Separation in arcseconds of the closest star.
    closest_star_mag (float): Magnitude of the closest star.
    density_arcsec2 (float): Density of stars in arcsec^-2.
    """

    sky_table = gaia_lite_table.cone_search(ra=ra_target, dec=dec_target, radius=radius).compute()
    sky = SkyCoord(ra=sky_table['ra'].values*u.deg, dec=sky_table['dec'].values*u.deg, frame='icrs')
    sky_target = SkyCoord(ra=ra_target*u.deg, dec=dec_target*u.deg, frame='icrs') # sky coord of object of interest

    delta_sep = sky.separation(sky_target).to(u.arcsec).value # separation in arcseconds

    return {"closest_star_arcsec": np.min(delta_sep),
        "closest_star_mag": sky_table['phot_g_mean_mag'][np.argmin(delta_sep)], 
        "density_arcsec2": len(delta_sep)/np.pi/radius**2}







