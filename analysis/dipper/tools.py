"""Tools and code to find dippers in ZTF data."""

from math import gamma
from scipy.special import erf
import astropy.stats as astro_stats
import numpy as np


def deviation(mag, mag_err):
    """Calculate the running deviation of a light curve for outburst or dip detection.
    
    Parameters:
    -----------
    mag (array-like): Magnitude values of the light curve.
    mag_err (array-like): Magnitude errors of the light curve.

    Returns:
    --------
    dev (array-like): Deviation values of the light curve.
    """
    # Calculate biweight estimators
    R, S = astro_stats.biweight_location(mag), astro_stats.biweight_scale(mag)

    return (mag - R) / np.sqrt(mag_err**2 + S**2)
    


def assymetry_yso_M(mag):
    """Calculate the magnitude assymetry score defined by Hillenbrand et al. 2022 (https://iopscience.iop.org/article/10.3847/1538-3881/ac62d8/pdf).

    Described in the paper: 
    Objects that have M values <0 are predominately brightening,
    objects with M values >0 are predominantly dimming, and
    objects with M values near 0 have symmetric light curves.

    Parameters:
    -----------
    mag (array-like): Magnitude values of the light curve.
    
    Returns:
    --------
    assymetry (float): Assymetry score.
    """
    mag_decile = np.percentile(mag, 10)

    return (mag_decile - np.nanmedian(mag))/np.nanstd(mag)


def ggd(x, mu, alpha, beta, ofs, amp):
    """
    Generalized Gaussian Distribution modified with an offset term and amplitude to fit the light curves.
    
    Parameters:
    -----------
    x (array-like): Input time values.
    mu (float): Mean of the distribution.
    alpha (float): Scale parameter.
    beta (float): Shape parameter.
    ofs (float): Offset term.
    amp (float): Amplitude term.

    Returns:
    --------
    y (array-like): Output magnitude values.
    """

    term_1 = beta/(2*alpha*gamma(1/beta))
    abs_term = ((abs(x-mu))/alpha)**beta
    return amp * term_1 * np.exp(-abs_term) + ofs


def depth(m1, mode='min'): 
    """
    Normalized magnitude depth on the minimum magnitude.
    
    Parameters:
    ----------
    m1 (array-like): Magnitude values of the light curve.

    Returns:
    --------
    depth (array-like): Depth values of the light curve.
    """
    return 10**((m1.min() - m1) / 2.5)


def skew_norm(x, mu, sigma, alpha, ofs, amp):
    """
    Skew-normal distribution modified with an offset term and amplitude to fit the light curves.
    
    Parameters:
    -----------
    x (array-like): Input time values.
    mu (float): Mean of the distribution.
    sigma (float): Standard deviation of the distribution.
    alpha (float): Shape parameter controlling the skewness of the distribution.
    ofs (float): Offset term.

    Returns:
    --------
    y (array-like): Output magnitude values.
    """
    gaus = 1/np.sqrt(2*np.pi) * np.exp(-(x-mu)**2/sigma**2)
    FI = 0.5 * (1 + erf(alpha*(x-mu)/np.sqrt(2)))
    return ofs + amp*2*gaus*FI