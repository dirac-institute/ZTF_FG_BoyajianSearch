"""
Tools for analysis.
"""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.modeling.models import Gaussian1D
from astropy.modeling import fitting


def prepare_lc(time, mag, mag_err, flag, band, band_of_study='r', flag_good=0, q=None, custom_q=False):
    """
    Prepare the light curve for analysis - specifically for the ZTF data.
    
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
        rmv = (flag == flag_good) & (band==band_of_study)
    
    # TODO: I'm working with the numpy values because we had some unexpected issues. Check if stable.
    time, mag, mag_err = time[rmv].values, mag[rmv].values, mag_err[rmv].values
    
    # sort time
    srt = np.argsort(time)

    return time[srt], mag[srt], mag_err[srt]

#TODO: fill the seasonal gaps...
def fill_gaps(time, mag, mag_err):
    """Fill the seasonal gaps of my data to the mean including the scatter of the observations..."""

def digest_the_peak(peak_dict, time, mag, mag_err, expandby=0):
    """Given the peak dictionary and data - prepare my light curve for GP analysis and integration.
    
    Parameters:
    -----------
    peak_dict (dict): Dictionary of the peak.
    time (array-like): Input time values.
    mag (array-like): Input magnitude values.
    mag_err (array-like): Input magnitude error values.
    expandby (float): Number of days to expand the window by. Default is 0 days.

    Returns:
    --------
    time (array-like): Output time values.
    mag (array-like): Output magnitude values.
    mag_err (array-like): Output magnitude error values.
    """

    # Define starting pontnts
    # TODO: make sure correct order
    start, end = peak_dict['window_start'].values[0], peak_dict['window_end'].values[0]

    # select
    selection = np.where((time > end-expandby) & (time < start+expandby) & (~np.isnan(time)) & (~np.isnan(mag)) & (~np.isnan(mag_err)))

    return time[selection], mag[selection], mag_err[selection]

def estimate_gaiadr3_density(ra_target, dec_target, gaia_lite_table, radius=0.01667):
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
    assert len(sky_table) > 0, "No stars found in the Gaia DR3 catalog around the target."

    sky = SkyCoord(ra=sky_table['ra'].values*u.deg, dec=sky_table['dec'].values*u.deg, frame='icrs')
    sky_target = SkyCoord(ra=ra_target*u.deg, dec=dec_target*u.deg, frame='icrs') # sky coord of object of interest

    # Find sky separation in arcseconds
    delta_sep = sky.separation(sky_target).to(u.arcsec).value

    # Find the separation to the cloest star.
    sep_sorted = np.argsort(delta_sep) 

    return {"closest_bright_star_arcsec": delta_sep[np.argmax(sky_table['phot_g_mean_mag'].values)],
        "closest_bright_star_mag": sky_table['phot_g_mean_mag'].values[np.argmin(sky_table['phot_g_mean_mag'].values)], 
        "closest_star_arcsec": delta_sep[sep_sorted][0],
        "closest_star_mag": sky_table['phot_g_mean_mag'].values[sep_sorted][0],
        "density_arcsec2": len(delta_sep)/np.pi/radius**2}


def bin_counter(xdat, ydat, bin_width=3):
    """Calculate the number of points per bin"""
    bins = np.arange(min(xdat), max(xdat), step=bin_width)
    value_count = []
    bin_centroid = []
    for i in range(0, len(bins)-1):
        _cond = np.where((xdat > bins[i]) & (xdat < bins[i+1]))
        value_count.append(len(_cond[0]))
        bin_centroid.append(0.5*(bins[i] + bins[i+1]))
    return np.array(bin_centroid), np.array(value_count)
   

def bin_counter(xdat, ydat, bin_width=3):
    """
    Calculate the number of points per bin and the running median of y-values.

    Parameters:
    -----------
    xdat (array-like): Array of x-values.
    ydat (array-like): Array of y-values.
    bin_width (float): Width of each bin. Default is 3.

    Returns:
    --------
    bin_centroid (ndarray): Array of bin centroids.
    value_count (ndarray): Array of counts per bin.
    running_median (ndarray): Array of running medians of y-values.
    """
    bins = np.arange(min(xdat), max(xdat), step=bin_width)
    value_count, bin_edges = np.histogram(xdat, bins=bins)
    bin_centroid = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    running_median = []
    for i in range(len(bin_centroid)):
        bin_indices = np.where((xdat > bin_edges[i]) & (xdat < bin_edges[i+1]))
        y_values = ydat[bin_indices]
        median = np.median(y_values)
        running_median.append(median)
    
    return bin_centroid, value_count, running_median


def quick_Gaussian_fit(time, running_deviation):
    """LevMarSQFitter Gaussian fit to the running deviation.
    
    Parameters:
    -----------
    time (array-like): Array of time values.
    running_deviation (array-like): Array of running deviation values.

    Returns:
    --------
    summary (dict): Dictionary of the Gaussian fit parameters.
    """
    t_init = Gaussian1D(amplitude=50., mean=time[np.argmax(running_deviation)]-1, stddev=1, 
                          bounds={"amplitude": (3, max(running_deviation)), 
                                 "mean": (time[np.argmax(running_deviation)]-535,
                                          time[np.argmax(running_deviation)]+535), 
                                 "stddev": (0.1, 1_000)})
    fit_t = fitting.LevMarLSQFitter()
    t = fit_t(t_init, time, running_deviation, maxiter=2_000)
    return dict(amplitude=t.amplitude.value, mean=t.mean.value, stddev=t.stddev.value)