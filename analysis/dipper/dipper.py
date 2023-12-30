"""
Dipper detection algorithm developed in https://github.com/AndyTza/little-dip
"""


import numpy as np
#Gaussian process modules
from george import kernels
import george
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared 
from scipy import interpolate
import astropy.stats as astro_stats
from scipy.optimize import curve_fit
from astropy.io import ascii
from scipy.signal import find_peaks
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import pandas as pd
import matplotlib.pyplot as plt


_all_funcs = ["deviation", 
                "calc_dip_edges", 
                "GaussianProcess_dip", 
                "calculate_integral",
                 "calculate_assymetry_score", 
                 "evaluate_dip", 
                 "light_curve_ens"]


def deviation(mag, mag_err, R, S):
    """Calculate the running deviation of a light curve for outburst or dip detection.
    
    d >> 0 will be dimming
    d << 0 (or negative) will be brightenning
    
    
    Parameters:
    -----------
    mag (array-like): Magnitude values of the light curve.
    mag_err (array-like): Magnitude errors of the light curve.
    R (float): Biweight location of the light curve (global).
    S (float): Biweight scale of the light curve (global).

    Returns:
    --------
    dev (array-like): Deviation values of the light curve.
    """
    # Calculate biweight estimators
    return (mag - R) / np.sqrt(mag_err**2 + S**2)  

def calc_dip_edges(xx, yy, _cent, atol=0.01):
    """ Calculate the edges of a dip given the center dip time. 

    Parameters:
    -----------
    xx (array-like): Time values of the light curve.
    yy (array-like): Magnitude values of the light curve.
    _cent (float): Center time of the dip.
    atol (float): Tolerance for the edge calculation. Default is 0.01.

    Returns:
    --------
    t_forward (float): Forward edge of the dip.
    t_back (float): Backward edge of the dip.
    time forward difference (float): Time difference between the forward edge and the center.
    time backward difference (float): Time difference between the backward edge and the center.
    N_thresh_1 (int): Number of detections above the median threshold in the given window.
    t_in_window (float): Average time difference in the given window.
    """
    indices_forward = np.where((xx > _cent) & np.isclose(yy, np.mean(yy) - 0.7*np.std(yy), atol=atol))[0]
    t_forward = xx[indices_forward[0]] if indices_forward.size > 0 else 0
    
    # select indicies close to the center (negative)
    indices_back = np.where((xx < _cent) & np.isclose(yy, np.mean(yy) - 0.7*np.std(yy), atol=atol))[0]
    if indices_back.size > 0:
        t_back = xx[indices_back[-1]]
    else:
        t_back = 0
        
    # Diagnostics numbers
    # How many detections above the median thresh in the given window?
    _window_ = (xx>t_back) & (xx<t_forward)
    sel_1_sig = (yy[_window_]>np.median(yy) + 1*np.std(yy)) # detections above 1 sigma
    N_thresh_1 = len((yy[_window_])[sel_1_sig])
    N_in_dip = len((yy[_window_]))

    # select times inside window and compute the average distance
    t_in_window = np.nanmean(np.diff(xx[_window_]))

    return t_forward, t_back, t_forward-_cent, _cent-t_back, N_thresh_1, N_in_dip, t_in_window


def GaussianProcess_dip(x, y, yerr, length_scale=0.01, error_penalty=0.5):
    """Perform a Gaussian Process interpolation on the light curve dip.
    
    Parameters:
    -----------
    x (array-like): Time values of the light curve.
    y (array-like): Magnitude values of the light curve.
    yerr (array-like): Magnitude errors of the light curve.
    length_scale (float): Length scale of the Gaussian Process kernel. Default is 0.01.

    Returns:
    --------
    x_pred (array-like): Time values of the interpolated light curve.
    pred (array-like): Magnitude values of the interpolated light curve.
    pred_var (array-like): Magnitude variance of the interpolated light curve. (multiply by 1.96 for 1-sigma)
    summary dictionary (dict): Summary of the GP interpolation. Including initial and final log likelihoods, and the success status. TODO: femove features.
    """

    # Penalize my errors for 
    if np.mean(yerr) * 100 > 1:
        yerr = yerr * error_penalty

    kernel = 1 * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e3)) + ExpSineSquared(length_scale=0.5,
    periodicity=0,
    length_scale_bounds=(1e-05, 100000.0),
    periodicity_bounds=(1e100, 1e200),) #TODO: these priors and and bounds work with current examples... be careful with the outsde observation windows with large gaps in the time series...
    
    noise_std = yerr
    gaussian_process = GaussianProcessRegressor(
    kernel=kernel, alpha=noise_std**2, n_restarts_optimizer=100)

    gaussian_process.fit(x.reshape(-1, 1), y)

    X = np.linspace(min(x), max(x), 2_000)

    # predict my work 
    mean_prediction, std_prediction = gaussian_process.predict(X.reshape(-1, 1), return_std=True)
    
    mean_prediction_dat, _ = gaussian_process.predict(x.reshape(-1, 1), return_std=True)
        
    return X.ravel(), mean_prediction, std_prediction, {"init_log_L": 0, 
                                "final_log_L": 0, 
                                "success_status": True, 
                                'chi-square': np.sum((y-mean_prediction_dat)**2/yerr**2)}

def GaussianProcess_dip_old(x, y, yerr, alpha=0.5, metric=100):
    """ (DEPRICATED)Perform a Gaussian Process interpolation on the light curve dip.

    Parameters:
    -----------
    x (array-like): Time values of the light curve.
    y (array-like): Magnitude values of the light curve.
    yerr (array-like): Magnitude errors of the light curve.

    Returns:
    --------
    x_pred (array-like): Time values of the interpolated light curve.
    pred (array-like): Magnitude values of the interpolated light curve.
    pred_var (array-like): Magnitude variance of the interpolated light curve.
    summary dictionary (dict): Summary of the GP interpolation. Including initial and final log likelihoods, and the success status.
    """
    # Standard RationalQuadraticKernel kernel from George
    # TODO: are my alpha and metric values correct?

    #TODO: Currently working with the scipy GP version - and happy with this kernel behavior.
    
    #try: # if the GP is failing it's likely an issue with the fitting. TODO: is this the best way to handle this?
    #kernel = kernels.RationalQuadraticKernel(log_alpha=alpha, metric=metric)
    

    # Standard GP proceedure using scipy.minimize the log likelihood (following https://george.readthedocs.io/en/latest/tutorials/)
    gp = george.GP(kernel)
    gp.compute(x, yerr)
    x_pred = np.linspace(min(x), max(x), 5_000)
    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    
    init_log_L = gp.log_likelihood(y)
    
    from scipy.optimize import minimize

    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)

    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)

    result = minimize(neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like)

    gp.set_parameter_vector(result.x)
    
    final_log_L = gp.log_likelihood(y)
    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    pred_on_data, pred_var_on_data = gp.predict(y, x, return_var=True) # make the GP prediction based on the data..
    
    return x_pred, pred, pred_var, {"init_log_L": init_log_L, 
                                "final_log_L": final_log_L, 
                                "success_status":result['success'], 
                                'chi-square': np.sum((y-pred_on_data)**2/yerr**2)}
    #except:
        #print ("GP failed!")
        #return x, y, yerr, {"init_log_L": np.nan, 
                                    #"final_log_L": np.nan, 
                                    #"success_status":False, 
                                    #'chi-square': np.nan}

def calculate_integral(x0, y0, yerr0, R, S):
    """Calculate the integral and integral error of a light curve.
    
    Parameters:
    -----------
    x0 (array-like): Time values of the light curve dip.
    y0 (array-like): Magnitude values of the light curve dip.
    yerr0 (array-like): Magnitude errors of the light curve dip.
    R (float): Biweight location of the light curve (global).
    S (float): Biweight scale of the light curve (global).

    Returns:
    --------
    integral (float): Integral of the light curve.
    integral_error (float): Integral error of the light curve.
    """
    # Integral equation
    #TODO: check that this calculation is right.
    integral = np.sum((y0[1::] - R) * (np.diff(x0)/2))
    integral_error = np.sum((yerr0[1::]**2 + S**2) * (np.diff(x0)/2)**2)
    
    return integral, np.sqrt(integral_error)

def calculate_assymetry_score(Int_left, Int_right, Int_err_left, Int_err_right):
    """Calculate the assymetry score of a light curve using the GP mean curve minmum as the zero-phase of the identified dip.
    
    Parameters:
    -----------
    Int_left (float): Integral of the light curve to the left of the center.
    Int_right (float): Integral of the light curve to the right of the center.
    Int_err_left (float): Integral error of the light curve to the left of the center.
    Int_err_right (float): Integral error of the light curve to the right of the center.

    Returns:
    --------
    assymetry_score (float): Assymetry score of the light curve.
    """
    # Assymetry score
    assymetry_score = (Int_left - Int_right) / (np.sqrt(Int_err_left**2 + Int_err_right**2))
    
    return assymetry_score  

def evaluate_dip(gp_model, x0, y0, yerr0, R, S, peak_loc, diagnostic=False):
    """Evaluate the dip and calculate its significance and error.

    Parameters:
    -----------
    gp_model (tuple): Gaussian Process model of the light curve dip.
    x0 (array-like): Time values of the light curve dip.
    y0 (array-like): Magnitude values of the light curve dip.
    yerr0 (array-like): Magnitude errors of the light curve dip.
    R (float): Biweight location of the light curve (global).
    S (float): Biweight scale of the light curve (global).
    diagnostic (bool): If True, plot the diagnostic plots. Default is False.
    peak_loc (float): Location of the peak of the light curve dip.

    Returns:
    --------
    summary (dict): Summary of the dip. Including the assymetry score, the left and right integral errors, the log sum error, and the chi-square.
    """
    # unpack the Gaussian Process model
    gpx, gpy, gpyerr, gp_info = gp_model
    
    # Find the peak of the GP curve
    #TODO: GP peak is causing issues withthe phase. Let keep it at the centroid of the window finder.
    #loc_gp = peak_loc # maximum magnitude...

    #TODO alternative peak finder - this might be better since we want the integral to be symmetric wrt to the gp peak...
    loc_gp = gpx[np.argmax(gpy)] # maximum magnitude...

    # If the difference is too large, let's use the peak finder peak...
    if not np.isclose(loc_gp, peak_loc, atol=3): #TODO: is a 3 day tolerance window too small or large? My guess is that this is fine since we want the GP to be near the peak of the dip finder...
        loc_gp = peak_loc
    
    # Let's try to find the edges of the dip using the GP fit...
    try:
        # Create an array of the global biweight location
        M = np.zeros(len(gpy)) + R # TODO: is global median the best choice here?

        # Find the indexes where the two functions intersect
        #TODO: this featue is struggling to find the edges of the dip
        idx = np.argwhere(np.diff(np.sign(M - gpy))).flatten()
        
        #GP model times where they intersect
        tdx = gpx[idx]
        tdx_phase = loc_gp - tdx # normalize wrt to peak loc
        
        # Select either positive or negative phase of the GP fit
        w_pos = tdx_phase>0
        w_neg = tdx_phase<0

        w_end = min(tdx[w_neg])
        w_start = max(tdx[w_pos])
    except:
        # If the GP is failing, let's use the edges from the dip finder...
        find = calc_dip_edges(x0, y0, peak_loc, atol=0.2)
        w_start, w_end = find[0], find[1]
    
    # Select the GP area of interest
    sel_gp = np.where((gpx>=w_start) & (gpx<=w_end))
    
    # The selected gaussian process curves.... (all dip)
    _gpx, _gpy, _gpyerr = gpx[sel_gp], gpy[sel_gp], gpyerr[sel_gp]
    _gpyerr2 = np.sqrt(_gpyerr)
    
    gp_left = _gpx <= loc_gp # left side indicies
    gp_right = _gpx >= loc_gp # right side indicies
    
    # left gp 
    left_gpx, left_gpy, left_gpyerr2 =  _gpx[gp_left], _gpy[gp_left], _gpyerr[gp_left]
    
    # right gp
    right_gpx, right_gpy, right_gpyerr2 =  _gpx[gp_right], _gpy[gp_right], _gpyerr[gp_right]
    
    # Calculate left integral
    integral_left = calculate_integral(left_gpx, left_gpy, left_gpyerr2, R, S)
    
    #Calculate right integral
    integral_right = calculate_integral(right_gpx, right_gpy, right_gpyerr2, R, S)

    # Calculate assymetry score
    IScore = calculate_assymetry_score(integral_left[0], integral_right[0], # left right integrals
                                        integral_left[1], integral_right[1]) # left right integral errors


    summary = {"assymetry_score": IScore, 
              "left_error": integral_left[1],
              "right_error": integral_right[1], 
              "log_sum_error": np.log10(sum(_gpy/_gpyerr2**2)), 
              "chi-square": gp_info['chi-square'], 
              "separation_btw_peaks": loc_gp-peak_loc} # check the separation between the peaks of the GP and the dip finder...
    
    if diagnostic:
        #### Diagnotstics for fitting ####
        # diagnostic fits
        plt.figure(figsize=(3,3))
        plt.axvline(w_start, color='red')
        plt.axvline(w_end, color='green')

        plt.scatter(gpx[idx], gpy[idx])
        plt.plot(gpx, gpy)
        plt.axhline(astro_stats.biweight_location(y0))
        plt.plot(gpx[sel_gp], gpy[sel_gp])
        plt.axvline(loc_gp, color='k', ls='--')
        plt.ylim(plt.ylim()[::-1])
        plt.errorbar(x0, y0, yerr0, color='k', fmt='.')
        plt.axhline(np.mean(y0), color='green', lw=2)
        plt.fill_between(_gpx, _gpy-np.sqrt(_gpyerr), _gpy+np.sqrt(_gpyerr), alpha=0.4)
        plt.title(f"{summary}, and logsum-err: {np.log10(sum(_gpy/(_gpyerr)))}")
        
    return summary

def peak_detector(times, dips, power_thresh=3, peak_close_rmv=15, pk_2_pk_cut=30):
    """
    Run and compute dip detection algorithm on a light curve.
    
    Parameters:
    -----------
    times (array-like): Time values of the light curve.
    dips (array-like): Deviation values of the light curve.
    power_thresh (float): Threshold for the peak detection. Default is 3.
    peak_close_rmv (float): Tolerance for removing peaks that are too close to each other. Default is 15.
    pk_2_pk_cut (float): Minimum peak to peak separation. Default is 30 days.

    Returns:
    --------
    N_peaks (int): Number of peaks detected.
    dip_summary (dict): Summary of the dip. Including the peak location, the window start and end, the number of 1 sigma detections in the dip, the number of detections in the dip, the forward and backward duration of the dip, and the dip power.
    """
    
    # Smooth the deviation dips with a savgol filter
    yht = savgol_filter(dips, 11, 8) # TODO: is this reccomended?
    
    # Scipy peak finding algorithm
    pks, _ = find_peaks(yht, height=power_thresh, distance=pk_2_pk_cut) #TODO: is 100 days peak separation too aggresive?

    # Reverse sort the peak values
    pks = np.sort(pks)[::-1]
    
    # Time of peaks and dev of peaks
    t_pks, p_pks = times[pks], dips[pks]
    
    # TODO: this section is likely not needed because scipy peak finder is good enough?
    # If we have more than one peak, remove peaks that are too close to each other?
    #if len(pks)>1:
    #    # remove peaks that are too close to each other
    #    t_pks = np.array([t_pks[i] for i in range(-1, len(t_pks)-1) if ~np.isclose(t_pks[i],
    #                                                                         t_pks[i+1],
    #                                                                         atol=peak_close_rmv)]) # 5 day tolerance window...

    #    p_pks = np.array([p_pks[i] for i in range(-1, len(t_pks)-1) if ~np.isclose(t_pks[i],
    #                                                                        t_pks[i+1],
    #                                                                        atol=peak_close_rmv)])
    #    srt = np.argsort(t_pks) # argsort the t_pks
    #
    #    t_pks, p_pks = t_pks[srt], p_pks[srt] # rename variables...
    
    # Number of peaks
    N_peaks = len(t_pks)
    
    dip_summary = {}
    for i, (time_ppk, ppk) in enumerate(zip(t_pks, p_pks)):
        _edges = calc_dip_edges(times, dips, time_ppk, atol=0.2)
        
        dip_summary[f'dip_{i}'] = {
            "peak_loc": time_ppk,
            'window_start': _edges[0],
            'window_end': _edges[1],
            "N_1sig_in_dip": _edges[-3], # number of 1 sigma detections in the dip
            "N_in_dip": _edges[-2], # number of detections in the dip
            'loc_forward_dur': _edges[2],
            "loc_backward_dur": _edges[3],
            "dip_power":ppk,
            "average_dt_dif": _edges[-1]
        }
                
    return N_peaks, dip_summary


def best_peak_detector(peak_dictionary, min_in_dip=1):
    """Chose the best peak from the peak detector with a minimum number of detections threshold. 
    
    Parameters:
    -----------
    peak_dictionary (dict): Dictionary of the peaks.
    min_in_dip (int): Minimum number of detections in the dip. Default is 3 detections.

    Returns:
    --------
    pd.DataFrame: Table of the best dip properties.
    
    """
    # unpack dictionary
    N_peaks, dict_summary = peak_dictionary
    
    summary_matrix = np.zeros(shape=(N_peaks, 9)) # TODO: add more columns to this matrix
    for i, info in enumerate(dict_summary.keys()):
       summary_matrix[i,:] = np.array(list(dict_summary[f'{info}'].values()))

    dip_table = pd.DataFrame(summary_matrix, columns=['peak_loc', 'window_start', 'window_end', 'N_1sig_in_dip', 'N_in_dip', 'loc_forward_dur', 'loc_backward_dur', 'dip_power', 'average_dt_dif'])
    dip_table_q = dip_table['N_in_dip'] >= min_in_dip # must contain at least one detection at the bottom

    if len(dip_table_q) == 0:
        print ("No dip is found within the minimum number of detections.")
        return None

    return dip_table.iloc[dip_table[dip_table_q]['dip_power'].idxmax()]
