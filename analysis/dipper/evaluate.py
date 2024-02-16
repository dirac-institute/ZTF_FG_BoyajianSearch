from dipper import *
from tools import *
import astropy.stats as astro_stats
from gpmcmc import *

# col names
column_names = ['Nphot',
    'biweight_scale',
    'frac_above_2_sigma',
    'Ndips',
    'rate',
    'chi2dof',
    'ADF_const',
    'ADF_const_trend',
    'ADF_pval_const',
    'ADF_pval_const_trend',
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
    'density_arcsec2']

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
        peak_detections = peak_detector(time, running_deviation, power_thresh=3, peak_close_rmv=15, pk_2_pk_cut=30)

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
                summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
            
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

            #TODO: validate the peak by first checking if it exists in the g-band
            g_validate = False
            out_g = 0
            time_g, mag_g, mag_err_g = prepare_lc(time_cat, mag_cat, mag_err_cat, flag_cat, band_cat, band_of_study='g', flag_good=0, q=None, custom_q=False)
            if len(time_g) > 10:
                g_validate = True
            Rg, Sg = astro_stats.biweight.biweight_location(mag_g), astro_stats.biweight.biweight_scale(mag_g)
            running_deviation_g = deviation(mag_g, mag_err_g, Rg, Sg)
            xg, yg, yerrg = digest_the_peak(bp, time_g, running_deviation_g, mag_err_g, expandby=3) 
            if len(xg) == 0:
                g_validate = False
            else:
                g_validate = True
                out_g = (np.mean(yg)-np.mean(running_deviation_g))/(np.std(running_deviation_g))

            #print (out_g, np.mean(running_deviation_g), np.mean(yg))
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
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
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
                    _ra, _dec = np.median(ra_cat), np.median(dec_cat)
                    #gaia_info = estimate_gaiadr3_density_async(_ra, _dec, radius=0.01667)

                    summary_['closest_bright_star_arcsec'] = 0  #gaia_info['closest_bright_star_arcsec']
                    summary_['closest_bright_star_mag'] =0 #gaia_info['closest_bright_star_mag']
                    summary_['closest_star_arcsec'] = 0 #gaia_info['closest_star_arcsec']
                    summary_['closest_star_mag'] = 0 #gaia_info['closest_star_mag']
                    summary_['density_arcsec2'] = 0 #gaia_info['density_arcsec2']
                else:
                    summary_['Nphot'] = len(time)
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
                    summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
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





        


        