"""Dipper detection algorithm


Computes: 
- Detected dipper 
- Centroid
- Window
- Score
"""

import numpy as np
from scipy.signal import find_peaks
import astropy.stats as astro_stats


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


def calc_dip_edges(xx, yy, _cent, atol=0.2):
    """
    Calculate the edges of a dip in a light curve.

    Parameters:
    -----------
    xx (array-like): The x-values of the light curve.
    yy (array-like): The y-values of the light curve.
    _cent (float): The central value around which to search for the dip edges.
    atol (float, optional): The absolute tolerance for comparing y-values to the median. Default is 0.2.

    Returns:
    --------
    t_forward (float): The time of the first data point after the dip.
    t_back (float): The time of the last data point before the dip.
    delta_t (float): The time difference between the central value and the first data point after the dip.
    N_thresh_1 (int): The number of detections above the median threshold in the given window.
    """
    
    indices_forward = np.where((xx > _cent) & np.isclose(yy, np.median(yy), atol=atol))[0]
    t_forward = xx[indices_forward[0]] if indices_forward.size > 0 else 0
    
    indices_back = np.where((xx < _cent) & np.isclose(yy, np.median(yy), atol=atol))[0]
    if indices_back.size > 0:
        t_back = xx[indices_back[-1]]
    else:
        t_back = 0
        
    # How many detections above the median thresh in the given window?
    _window_ = (xx>t_back) & (xx<t_forward)
    sel_1_sig = (yy[_window_]>np.median(yy) + np.std(yy)) # detections above 1 sigma
    N_thresh_1 = len((yy[_window_])[sel_1_sig])
    
    return t_forward, t_back, (t_forward-_cent), N_thresh_1


def summarize_dev_dips(times, dips, power_thresh=3, loc_peak_thresh=6):
    """
    Summarizes the developer dips based on the given times and dips arrays.

    Parameters:
    -----------
    times (array-like): Array of time values.
    dips (array-like): Array of dip values.
    power_thresh (float, optional): Threshold for dip power. Defaults to 3.
    loc_peak_thresh (float, optional): Threshold for peak location. Defaults to 6.

    Returns:
    --------
    tuple: A tuple containing the number of peaks with removed dips and a dictionary summarizing the peak information.
    """
    # Scipy peak finding algorithm
    pks, _ = find_peaks(dips, height=loc_peak_thresh)
    pks = np.sort(pks)[::-1] # sort the reverse peaks
    
    # Time of peaks and dev of peaks
    t_pks, p_pks = times[pks], dips[pks]
    
    # remove peaks that are too close to each other
    t_pks = np.array([t_pks[i] for i in range(-1, len(t_pks)-1) if ~np.isclose(t_pks[i],
                                                                         t_pks[i+1],
                                                                         atol=5)])
    
    p_pks = np.array([p_pks[i] for i in range(-1, len(t_pks)-1) if ~np.isclose(t_pks[i],
                                                                        t_pks[i+1],
                                                                        atol=5)])
    srt = np.argsort(t_pks) # argsort the t_pks
    
    t_pks, p_pks = t_pks[srt], p_pks[srt] # rename variables...
    
    N_peaks = len(t_pks) # number of peaks with removed
    
    # summarize peak information
    dip_summary = {}
    
    i = 0
    for time_ppk, ppk in zip(t_pks, p_pks):
        _edges = calc_dip_edges(times, dips, time_ppk, atol=0.2)
        
        dip_summary[f'dip_{i}'] = {
            "peak_loc": time_ppk,
            'window_start': _edges[0],
            'window_end': _edges[1],
            "N_1sig_in_dip": _edges[-1], 
            'loc_forward_dur': _edges[2],
            "dip_power":ppk
        }
        
        i+=1
    
    return N_peaks, dip_summary

