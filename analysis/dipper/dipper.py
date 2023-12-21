"""
Dipper detection algorithm developed in https://github.com/AndyTza/little-dip
"""

def deviation(mag, mag_err):
    """Calculate the running deviation of a light curve for outburst or dip detection.
    
    d >> 0 will be dimming
    d << 0 (or negative) will be brightenning
    
    
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
    """ Calculate the edges of a dip given the center dip time. 

    Parameters:
    -----------
    xx (array-like): Time values of the light curve.
    yy (array-like): Magnitude values of the light curve.
    _cent (float): Center time of the dip.
    atol (float): Tolerance for the edge calculation. Default is 0.2.

    Returns:
    --------
    t_forward (float): Forward edge of the dip.
    t_back (float): Backward edge of the dip.
    time forward difference (float): Time difference between the forward edge and the center.
    time backward difference (float): Time difference between the backward edge and the center.
    N_thresh_1 (int): Number of detections above the median threshold in the given window.
    """
    
    # select indicies close to the center (positive)
    indices_forward = np.where((xx > _cent) & np.isclose(yy, np.median(yy) - 0.02*np.median(yy), atol=atol))[0]
    t_forward = xx[indices_forward[0]] if indices_forward.size > 0 else 0
    
    # select indicies close to the center (negative)
    indices_back = np.where((xx < _cent) & np.isclose(yy, np.median(yy) - 0.02*np.median(yy), atol=atol))[0]
    if indices_back.size > 0:
        t_back = xx[indices_back[-1]]
    else:
        t_back = 0
        
    # Diagnostics numbers
    # How many detections above the median thresh in the given window?
    _window_ = (xx>t_back) & (xx<t_forward)
    sel_1_sig = (yy[_window_]>np.median(yy) + 1*np.std(yy)) # detections above 1 sigma
    N_thresh_1 = len((yy[_window_])[sel_1_sig])
    
    return t_forward, t_back, t_forward-_cent, _cent-t_back, N_thresh_1

def GaussianProcess_dip(x, y, yerr, alpha=0.5, metric=100):
    """ Perform a Gaussian Process interpolation on the light curve dip.

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
    kernel = kernels.RationalQuadraticKernel(log_alpha=alpha, metric=metric)
    
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
    """Calculate the assymetry score of a light curve.
    
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

def evaluate_dip(gp_model, x0, y0, yerr0, R, S, diagnostic=False):
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

    Returns:
    --------
    summary (dict): Summary of the dip. Including the assymetry score, the left and right integral errors, the log sum error, and the chi-square.
    """
    # unpack the Gaussian Process model
    gpx, gpy, gpyerr, gp_info = gp_model
    
    # Find the peak of the GP curve
    #TODO: should this be the the center found from the dip finder or the GP minimum?
    loc_gp = gpx[np.argmax(gpy)] # maximum magnitude...
    
    # Create an array of the global biweight location
    M = np.zeros(len(gpy)) + R
    
    # Find the indexes where the two functions intersect
    idx = np.argwhere(np.diff(np.sign(M - gpy))).flatten()
    
    #GP model times where they intersect
    tdx = gpx[idx]
    tdx_phase = loc_gp - tdx # normalize wrt to peak loc
    
    # Select either positive or negative phase of the GP fit
    w_pos = tdx_phase>0
    w_neg = tdx_phase<0
    
    # Select the edges of the phase
    try:
        w_end = min(tdx[w_neg])
        w_start = max(tdx[w_pos])
    except:
        #assert ("Failed to find start or end")
        return False
    
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
              "chi-square": gp_info['chi-square']"}
    
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