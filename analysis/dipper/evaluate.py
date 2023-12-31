from dipper import *
from tools import *
import astropy.stats as astro_stats
from gpmcmc import *

column_names = [
    'biweight_scale',
    'frac_above_2_sigma',
    'Ndips',
    'rate',
    'chi2dof',
    'ADF_const',
    'ADF_const_trend',
    'ADF_pval_const',
    'ADF_pval_const_trend',
    'best_dip_power',
    'best_dip_time_loc',
    'best_dip_start',
    'best_dip_end',
    'best_dip_dt',
    'best_dip_ndet',
    'best_dip_score',
    'left_error',
    'right_error',
    'chi_square_gp',
    'gp_fun',
    'gp_status',
    'separation_btw_peaks',
    'closest_bright_star_arcsec',
    'closest_bright_star_mag',
    'closest_star_arcsec',
    'closest_star_mag',
    'density_arcsec2'
]

def evaluate(time, mag, mag_err, flag, band, ra, dec, custom_cols=column_names):
    """Evaluate time series..."""

    # Summary information
    summary_ = {}
    
    # Digest my light curve. Select band, good detections & sort
    time, mag, mag_err = prepare_lc(time, mag, mag_err, flag, band,  band_of_study='r', flag_good=0, q=None, custom_q=False)

    # Fill in observational gaps...
    time, mag, mag_err = fill_gaps(time, mag, mag_err, num_points=25, max_gap_days=95)

    # Evaluate biweight location and scale & other obvious statistics
    R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)
    adf = adf_tests(mag) # ADF test for stationarity
    chi2dof = chidof(mag) # chi2dof

    # Running deviation
    running_deviation = deviation(mag, mag_err, R, S)

    # Peak detection summary per light curve
    peak_detections = peak_detector(time, running_deviation, power_thresh=3, peak_close_rmv=15, pk_2_pk_cut=30)
    
    if peak_detections[0] > 0:
        # Select best peak candidate with at least 3 points in the dip
        bp = best_peak_detector(peak_detections, min_in_dip=3)
    
    if peak_detections[0]==0 or len(time)==0:

        summary_['biweight_scale'] = S

        if len(running_deviation)==0:
            summary_['frac_above_2_sigma'] = 0
        else:
            summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
        
        summary_['Ndips'] = 0
        summary_['rate'] = 0
        summary_['chi2dof'] = chi2dof
        summary_['ADF_const'] = adf['ADF-const']
        summary_['ADF_const_trend'] = adf['ADF-const-trend']
        summary_['ADF_pval_const'] = adf['p-const']
        summary_['ADF_pval_const_trend'] = adf['p-const-trend']

        # If failing; set all values to NaN
        for col in custom_cols[9::]:
            summary_[col] = np.nan
    else: 
        # prepare the dip for the GP analysis... expand by 15 days should be fine...
        x, y, yerr = digest_the_peak(bp, time, mag, mag_err, expandby=15) # expand by 15 days is usually a good choice

        # GP analysis
        try:
            gp =  simple_GP(x, y, yerr, ell=10)
        except: # if GP fails for some reason...
            gp = [None, None, None]

        if gp[0] is not None:
            # GP assesment of quality 
            gp_quality = evaluate_dip(gp, x, y, yerr, R, S, bp['peak_loc'], diagnostic=False)

            summary_['biweight_scale'] = S
            summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
            summary_['Ndips'] = peak_detections[0] # number of peaks
            summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
            summary_['chi2dof'] = chi2dof
            summary_['ADF_const'] = adf['ADF-const']
            summary_['ADF_const_trend'] = adf['ADF-const-trend']
            summary_['ADF_pval_const'] = adf['p-const']
            summary_['ADF_pval_const_trend'] = adf['p-const-trend']

            summary_['best_dip_power'] = bp['dip_power'].values[0]
            summary_['best_dip_time_loc'] = bp['peak_loc'].values[0]
            summary_['best_dip_start'] = bp['window_start'].values[0]
            summary_['best_dip_end'] = bp['window_end'].values[0]
            summary_['best_dip_dt'] = bp['average_dt_dif'].values[0]
            summary_['best_dip_ndet'] = bp['N_in_dip'].values[0]

            # Gaussian Process metrics
            summary_["best_dip_score"]=gp_quality["assymetry_score"]
            summary_["left_error"]=gp_quality["left_error"]
            summary_["right_error"]=gp_quality["right_error"]
            summary_['gp_fun'] = gp_quality['gp-fun']
            summary_['gp_status'] = gp_quality['gp_status']
            summary_['chi_square_gp'] = gp_quality['chi-square-gp']
            summary_['separation_btw_peaks'] = gp_quality['separation_btw_peaks']

            # evaluate Gaia close star statistics
            _ra, _dec = np.median(ra), np.median(dec)
            gaia_info = estimate_gaiadr3_density_async(_ra, _dec, radius=0.01667)

            summary_['closest_bright_star_arcsec'] = gaia_info['closest_bright_star_arcsec']
            summary_['closest_bright_star_mag'] = gaia_info['closest_bright_star_mag']
            summary_['closest_star_arcsec'] = gaia_info['closest_star_arcsec']
            summary_['closest_star_mag'] = gaia_info['closest_star_mag']
            summary_['density_arcsec2'] = gaia_info['density_arcsec2']
        else:
            summary_['biweight_scale'] = S
            if len(running_deviation)==0:
                summary_['frac_above_2_sigma'] = 0
            else:
                summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
            summary_['Ndips'] = peak_detections[0] # number of peaks
            summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
            summary_['chi2dof'] = chi2dof
            summary_['ADF_const'] = adf['ADF-const']
            summary_['ADF_const_trend'] = adf['ADF-const-trend']
            summary_['ADF_pval_const'] = adf['p-const']
            summary_['ADF_pval_const_trend'] = adf['p-const-trend']

            # If failing; set all values to NaN
            for col in custom_cols[9::]:
                summary_[col] = np.nan


    return pd.Series(list(summary_.values()), index=custom_cols)





        