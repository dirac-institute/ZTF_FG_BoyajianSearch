from dipper import *
from tools import *
import astropy.stats as astro_stats
from gpmcmc import *
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')

import warnings
warnings.filterwarnings('ignore')

# feature evaluation 
column_names = ['Nphot',
    'biweight_scale',
    'frac_above_2_sigma', # in deviation
    'Ndips',
    'rate',
    'chi2dof',
    'skew', 
    'kurtosis',
    'mad',
    'stetson_i',
    'stetson_j',
    'stetson_k',
    'invNeumann',    
    'best_dip_power',
    'best_dip_time_loc',
    'best_dip_start',
    'best_dip_end',
    'best_dip_dt',
    'best_dip_ndet',
    'lc_score']

def half_eval(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, ra_cat, dec_cat, custom_cols=column_names, min_phot=10):
    # Digest my light curve. Select band, good detections & sort
    time, mag, mag_err = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat,  band_of_study='r', flag_good=0, q=None, custom_q=False)

    # Evaluate biweight location and scale & other obvious statistics
    R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)
    adf = adf_tests(mag) # ADF test for stationarity
    chi2dof = chidof(mag) # chi2dof

    # Running deviation
    running_deviation = deviation(mag, mag_err, R, S)

    # Peak detection summary per light curve
    peak_detections = peak_detector(time, running_deviation, power_thresh=3, peak_close_rmv=1, pk_2_pk_cut=1)

    return peak_detections


def evaluate(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, ra_cat, dec_cat, custom_cols=column_names, min_phot=10):
    """Evaluate time series."""

    # Summary information
    summary_ = {}
    
    # Digest my light curve. Select band, good detections & sort
    time, mag, mag_err = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat,  band_of_study='r', flag_good=0, q=None, custom_q=False)

    # Don't evaluate if there are less than 10 detections
    if len(time) < min_phot:
        summary_['Nphot'] = len(time)
        for col in custom_cols[1::]:
            summary_[col] = np.nan
    else:
        # Evaluate biweight location and scale & other obvious statistics
        R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)
        adf = adf_tests(mag) # ADF test for stationarity
        chi2dof = chidof(mag) # chi2dof

        # Running deviation
        running_deviation = deviation(mag, mag_err, R, S)

        # Peak detection summary per light curve
        peak_detections = peak_detector(time, running_deviation, power_thresh=3, peak_close_rmv=1, pk_2_pk_cut=1)

        # Calculate other summary statistics
        other_stats = other_summary_stats(time, mag, mag_err, len(mag), R, S)
        
        if peak_detections[0] > 0:
            # Select best peak candidate with at least 3 points in the dip
            bp = best_peak_detector(peak_detections, min_in_dip=3)
        
        if peak_detections[0]==0 or len(time)==0:

            summary_['Nphot'] = len(time)
            summary_['biweight_scale'] = S

            if len(running_deviation)==0:
                summary_['frac_above_2_sigma'] = 0
            else:
                summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
            
            summary_['Ndips'] = 0
            summary_['rate'] = 0
            summary_['chi2dof'] = chi2dof
            summary_['ADF_const'] = adf['ADF-const']
            summary_['ADF_const_trend'] = adf['ADF-const-trend']
            summary_['ADF_pval_const'] = adf['p-const']
            summary_['ADF_pval_const_trend'] = adf['p-const-trend']
            summary_['skew'] = other_stats['skew']
            summary_['kurtosis'] = other_stats['kurtosis']
            summary_['mad'] = other_stats['mad']
            summary_['stetson_i'] = other_stats['stetson_I']
            summary_['stetson_j'] = other_stats['stetson_J']
            summary_['stetson_k'] = other_stats['stetson_K']
            summary_['invNeumann'] = other_stats['invNeumann']

            # If failing; set all values to NaN
            for col in custom_cols[17::]:
                summary_[col] = np.nan
        else: 
            # If there are significant peaks...

            #TODO: validate the peak by first checking if it exists in the g-band
            g_validate = False
            out_g = 0
            time_g, mag_g, mag_err_g = prepare_lc(time_cat, mag_cat, mag_err_cat,
                                                   flag_cat, band_cat, band_of_study='g', flag_good=0, q=None, custom_q=False)
            if len(time_g) > 10:
                g_validate = True
            Rg, Sg = astro_stats.biweight.biweight_location(mag_g), astro_stats.biweight.biweight_scale(mag_g)
            running_deviation_g = deviation(mag_g, mag_err_g, Rg, Sg)
            xg, yg, yerrg = digest_the_peak(bp, time_g, running_deviation_g, mag_err_g, expandby=3) 
            if len(xg) == 0:
                g_validate = False
            else:
                g_validate = True
                out_g = (np.nanmean(yg)-np.nanmean(running_deviation_g))/(np.nanstd(running_deviation_g))

            #print (bp)
            #TODO: check if 1.5 sigma is okay for now...
            if g_validate and out_g >1.5:
                #TODO: this step might be unnecessary...
                # prepare the dip for the GP analysis... expand by 15 days should be fine...
                x, y, yerr = digest_the_peak(bp, time, mag, mag_err, expandby=0) # expand by 1 days is usually a good choice

                # GP analysis
                try:
                    gp =  simple_GP(x, y, yerr, ell=10)
                except: # if GP fails for some reason...
                    gp = [None, None, None]

                if (gp[0] is not None) and (x[0] > 0):
                    # GP assesment of quality 
                    gp_quality = evaluate_dip(gp, x, y, yerr, R, S, bp['peak_loc'], diagnostic=False)

                    summary_['Nphot'] = len(time)
                    summary_['biweight_scale'] = S
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                    summary_['chi2dof'] = chi2dof
                    summary_['ADF_const'] = adf['ADF-const']
                    summary_['ADF_const_trend'] = adf['ADF-const-trend']
                    summary_['ADF_pval_const'] = adf['p-const']
                    summary_['ADF_pval_const_trend'] = adf['p-const-trend']
                    summary_['skew'] = other_stats['skew']
                    summary_['kurtosis'] = other_stats['kurtosis']
                    summary_['mad'] = other_stats['mad']
                    summary_['stetson_i'] = other_stats['stetson_I']
                    summary_['stetson_j'] = other_stats['stetson_J']
                    summary_['stetson_k'] = other_stats['stetson_K']
                    summary_['invNeumann'] = other_stats['invNeumann']

                    summary_['Ndips'] = peak_detections[0] # number of peaks
                    summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
                    summary_['best_dip_power'] = bp['dip_power'].values[0]
                    summary_['best_dip_time_loc'] = bp['peak_loc'].values[0]
                    summary_['best_dip_start'] = bp['window_start'].values[0]
                    summary_['best_dip_end'] = bp['window_end'].values[0]
                    summary_['best_dip_dt'] = bp['average_dt_dif'].values[0]
                    summary_['best_dip_ndet'] = bp['N_in_dip'].values[0]
                    # change default

                    # Gaussian Process metrics
                    summary_["best_dip_score"]=gp_quality["assymetry_score"]
                    summary_["left_error"]=gp_quality["left_error"]
                    summary_["right_error"]=gp_quality["right_error"]
                    summary_['gp_fun'] = gp_quality['gp-fun']
                    summary_['gp_status'] = gp_quality['gp_status']
                    summary_['chi_square_gp'] = gp_quality['chi-square-gp']
                    summary_['separation_btw_peaks'] = gp_quality['separation_btw_peaks']

                    # TODO: need better way to handle this
                    # evaluate Gaia close star statistics
                    _ra, _dec = np.nanmedian(ra_cat), np.nanmedian(dec_cat)
                    #gaia_info = estimate_gaiadr3_density_async(_ra, _dec, radius=0.01667)

                    summary_['closest_bright_star_arcsec'] = 0  #gaia_info['closest_bright_star_arcsec']
                    summary_['closest_bright_star_mag'] =0 #gaia_info['closest_bright_star_mag']
                    summary_['closest_star_arcsec'] = 0 #gaia_info['closest_star_arcsec']
                    summary_['closest_star_mag'] = 0 #gaia_info['closest_star_mag']
                    summary_['density_arcsec2'] = 0 #gaia_info['density_arcsec2']

                    # Calculate now the total score
                    Iscors = []
                    for j in range(peak_detections[0]):
                        xt1, yt1, yerrt1 = digest_the_peak(peak_detections[1][f'dip_{j}'], 
                                                           time, 
                                                           running_deviation, mag_err, expandby=7)
                        gp_inter = simple_linear_interp(xt1, yt1, yerrt1)
                        _s = evaluate_dip(gp_inter, xt1,
                                           yt1, yerrt1, 
                                           R, S,
                                             peak_detections[1][f'dip_{j}']['peak_loc'], diagnostic=False)['assymetry_score']
                        Iscors.append(_s)
                    
                    C = 0 
                    for jj in range(peak_detections[0]):
                        DT = peak_detections[1][f'dip_{jj}']['window_end'] - peak_detections[1][f'dip_{jj}']['window_start']
                        N = peak_detections[1][f'dip_{jj}']['N_1sig_in_dip']
                        C += peak_detections[1][f'dip_{jj}']['dip_power'] * DT * N * abs(Iscors[jj])
                    
                    summary_['total_score'] = C

                else:
                    summary_['Nphot'] = len(time)
                    summary_['biweight_scale'] = S
                    if len(running_deviation)==0:
                        summary_['frac_above_2_sigma'] = 0
                    else:
                        summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                    summary_['Ndips'] = peak_detections[0] # number of peaks
                    summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
                    summary_['chi2dof'] = chi2dof
                    summary_['ADF_const'] = adf['ADF-const']
                    summary_['ADF_const_trend'] = adf['ADF-const-trend']
                    summary_['ADF_pval_const'] = adf['p-const']
                    summary_['ADF_pval_const_trend'] = adf['p-const-trend']
                    summary_['skew'] = other_stats['skew']
                    summary_['kurtosis'] = other_stats['kurtosis']
                    summary_['mad'] = other_stats['mad']
                    summary_['stetson_i'] = other_stats['stetson_I']
                    summary_['stetson_j'] = other_stats['stetson_J']
                    summary_['stetson_k'] = other_stats['stetson_K']
                    summary_['invNeumann'] = other_stats['invNeumann']

                    # If failing; set all values to NaN
                    for col in custom_cols[17::]:
                        summary_[col] = np.nan
            else:
                summary_['Nphot'] = len(time)
                summary_['biweight_scale'] = S
                if len(running_deviation)==0:
                    summary_['frac_above_2_sigma'] = 0
                else:
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                summary_['Ndips'] = peak_detections[0] # number of peaks
                summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
                summary_['chi2dof'] = chi2dof
                summary_['ADF_const'] = adf['ADF-const']
                summary_['ADF_const_trend'] = adf['ADF-const-trend']
                summary_['ADF_pval_const'] = adf['p-const']
                summary_['ADF_pval_const_trend'] = adf['p-const-trend']
                summary_['skew'] = other_stats['skew']
                summary_['kurtosis'] = other_stats['kurtosis']
                summary_['mad'] = other_stats['mad']
                summary_['stetson_i'] = other_stats['stetson_I']
                summary_['stetson_j'] = other_stats['stetson_J']
                summary_['stetson_k'] = other_stats['stetson_K']
                summary_['invNeumann'] = other_stats['invNeumann']

                # If failing; set all values to NaN
                for col in custom_cols[17::]:
                    summary_[col] = np.nan


    return pd.Series(list(summary_.values()), index=custom_cols)

def evaluate_updated(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, custom_cols=column_names, min_phot=10):
    """Evaluate time series as of April 2024."""

    # Summary information
    summary_ = {}
    
    # Digest my light curve. Select band, good detections & sort
    time, mag, mag_err = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, 
                                    band_of_study='r', flag_good=0, q=None, custom_q=False)

    # Don't evaluate if there are less than 10 detections
    if len(time) < min_phot:
        summary_['Nphot'] = len(time)
        for col in custom_cols[1::]:
            summary_[col] = np.nan
    else:
        # Evaluate biweight location and scale & other obvious statistics
        R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)
        chi2dof = chidof(mag) # chi2dof

        # Running deviation
        running_deviation = deviation(mag, mag_err, R, S)

        # Peak detection summary per light curve
        peak_detections = peak_detector(time, running_deviation, power_thresh=4, peak_close_rmv=20, pk_2_pk_cut=20)

        # Calculate other summary statistics
        other_stats = other_summary_stats(time, mag, mag_err, len(mag), R, S)
            
        # If there's no detected peaks or time array is empty or no peaks detected...
        if peak_detections[0]==0 or len(time)==0 or peak_detections[0]==0:
            # If failing; set all values to NaN
            for col in custom_cols:
                summary_[col] = np.nan
            
            # Replace nan's with values
            summary_['Nphot'] = len(time)
            summary_['biweight_scale'] = S

            if len(running_deviation)==0:
                summary_['frac_above_2_sigma'] = 0
            else:
                summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
            
            summary_['Ndips'] = 0
            summary_['rate'] = 0
            summary_['chi2dof'] = chi2dof
            summary_['skew'] = other_stats['skew']
            summary_['kurtosis'] = other_stats['kurtosis']
            summary_['mad'] = other_stats['mad']
            summary_['stetson_i'] = other_stats['stetson_I']
            summary_['stetson_j'] = other_stats['stetson_J']
            summary_['stetson_k'] = other_stats['stetson_K']
            summary_['invNeumann'] = other_stats['invNeumann']
        else: # If there are significant peaks...
            
            # From the r-band data select a good peak...
            bp = best_peak_detector(peak_detections, min_in_dip=3)
            
            # Investigate the g-band data and ensure we see a ~significant~ event 
            g_validate, out_g = False, 0
            
            time_g, mag_g, mag_err_g = prepare_lc(time_cat, mag_cat, mag_err_cat,
                                                   flag_cat, band_cat, band_of_study='g', flag_good=0, q=None, custom_q=False)
            
            # minimum number of g-band detections after processing
            if len(time_g) > 10:
                g_validate = True
                
            Rg, Sg = astro_stats.biweight.biweight_location(mag_g), astro_stats.biweight.biweight_scale(mag_g)
            
            running_deviation_g = deviation(mag_g, mag_err_g, Rg, Sg)

            try:
                best_peak_time = bp['peak_loc'].values[0]
                sel_g = np.where((time_g > best_peak_time-3) & (time_g < best_peak_time+3)) # peak within +/- 3 days
                xg, yg, yerrg = time_g[sel_g], mag_g[sel_g], mag_err_g[sel_g]

                Rg_mod, Sg_mod = astro_stats.biweight.biweight_location(yg), astro_stats.biweight.biweight_scale(yg)

                yg_dev = deviation(yg, yerrg, Rg, Sg)
            except:
                g_validate = False
                xg = [] # empty array...
            
            # Select g-band detections at bp and expand by ~3 days
            #xg, yg, yerrg = digest_the_peak(bp, time_g, running_deviation_g, mag_err_g, expandby=0) # do not expand...
            
            if (len(xg) == 0) or (g_validate==False): # reject if there's no detections...
                g_validate = False
            else:
                g_validate = True
                # Calculate the significance of this g-band bump...
                out_g = (np.nanmean(yg_dev)-np.nanmean(running_deviation_g))/(np.nanstd(running_deviation_g))

            #TODO: check if 1.5 sigma is okay for now...
            if g_validate and out_g >1.5: # both r-band and g-band data show similar peaks...
        
                _score_ = calc_sum_score(time, mag, peak_detections, R, S)

                # If failing; set all values to NaN
                for col in custom_cols:
                    summary_[col] = np.nan

                ######## Final appending data ########
                summary_['Nphot'] = len(time)
                summary_['biweight_scale'] = S
                if len(running_deviation)==0:
                    summary_['frac_above_2_sigma'] = 0
                else:
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                summary_['Ndips'] = peak_detections[0] # number of peaks
                summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
                summary_['chi2dof'] = chi2dof
                summary_['skew'] = other_stats['skew']
                summary_['kurtosis'] = other_stats['kurtosis']
                summary_['mad'] = other_stats['mad']
                summary_['stetson_i'] = other_stats['stetson_I']
                summary_['stetson_j'] = other_stats['stetson_J']
                summary_['stetson_k'] = other_stats['stetson_K']
                summary_['invNeumann'] = other_stats['invNeumann']
                summary_['best_dip_power'] = bp['dip_power'].values[0]
                summary_['best_dip_time_loc'] = bp['peak_loc'].values[0]
                summary_['best_dip_start'] = bp['window_start'].values[0]
                summary_['best_dip_end'] = bp['window_end'].values[0]
                summary_['best_dip_dt'] = bp['average_dt_dif'].values[0]
                summary_['best_dip_ndet'] = bp['N_in_dip'].values[0]
                summary_['lc_score'] = _score_
            
            else:
                # If failing; set all values to NaN
                for col in custom_cols:
                    summary_[col] = np.nan
                    
                summary_['Nphot'] = len(time)
                summary_['biweight_scale'] = S
                if len(running_deviation)==0:
                    summary_['frac_above_2_sigma'] = 0
                else:
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.nanmean(running_deviation)+2*np.nanstd(running_deviation)])/len(running_deviation)
                summary_['Ndips'] = peak_detections[0] # number of peaks
                summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
                summary_['chi2dof'] = chi2dof
                summary_['skew'] = other_stats['skew']
                summary_['kurtosis'] = other_stats['kurtosis']
                summary_['mad'] = other_stats['mad']
                summary_['stetson_i'] = other_stats['stetson_I']
                summary_['stetson_j'] = other_stats['stetson_J']
                summary_['stetson_k'] = other_stats['stetson_K']
                summary_['invNeumann'] = other_stats['invNeumann']
                
    return pd.Series(list(summary_.values()), index=custom_cols)