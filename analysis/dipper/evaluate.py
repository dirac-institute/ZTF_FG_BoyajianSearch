from dipper import *
from tools import *
import astropy.stats as astro_stats
from gpmcmc import *

column_names = [
    'biweight_scale',
    'frac_above_2_sigma',
    'Ndips',
    'rate',
    'best_dip_power',
    'best_dip_time_loc',
    'best_dip_start',
    'best_dip_end',
    'best_dip_dt',
    'best_dip_ndet',
    'best_dip_score',
    'left_error',
    'right_error',
    'log_sum_error',
    'logL_best_dip',
    'amp_median',
    'amp_std',
    'location_median',
    'location_std',
    'sigma_median',
    'sigma_std',
    'log_sigma2_median',
    'log_sigma2_std',
    'm_median',
    'm_std',
    'b_median',
    'b_std',
    'closest_bright_star_arcsec',
    'closest_bright_star_mag',
    'closest_star_arcsec',
    'closest_star_mag',
    'density_arcsec2'
]

def evaluate(time, mag, mag_err, flag, band, ra, dec, custom_cols=column_names):

    # Summary information
    summary_ = {}
    
    # Digest my light curve. Select band, good detections & sort
    time, mag, mag_err = prepare_lc(time, mag, mag_err, flag, band,  band_of_study='r', flag_good=0, q=None, custom_q=False)

    # Fill in observational gaps...
    time, mag, mag_err = fill_gaps(time, mag, mag_err, num_points=25, max_gap_days=95)

    # Evaluate biweight location and scale
    R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)

    # Running deviation
    running_deviation = deviation(mag, mag_err, R, S)

    # Peak detection summary per light curve
    peak_detections = peak_detector(time, running_deviation, power_thresh=3, peak_close_rmv=15, pk_2_pk_cut=30)

    # Select best peak candidate with at least 3 points in the dip
    bp = best_peak_detector(peak_detections, min_in_dip=3)

    if bp is None or len(bp)==0:
        bp = None
    else:
        # TODO: reject peaks that are too close to the edges of the light curve
        try:
            if bp['loc_forward_dur'].values[0]<2 or bp['loc_backward_dur'].values[0]<2:
                bp = None
        except:
           if bp['loc_forward_dur']<2 or bp['loc_backward_dur']<2:
                bp = None 

    # If no peaks found...
    if peak_detections[0]==0 or bp is None:
        print ("No peaks found!")
        summary_['biweight_scale'] = S

        if len(running_deviation)==0:
            summary_['frac_above_2_sigma'] = 0
        else:
            summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
        
        summary_['Ndips'] = 0
        summary_['rate'] = 0
        summary_['best_dip_power'] = np.nan
        summary_['best_dip_time_loc'] = np.nan
        summary_['best_dip_start'] = np.nan
        summary_['best_dip_end'] = np.nan
        summary_['best_dip_dt'] = np.nan
        summary_['best_dip_ndet'] = np.nan

        # Gaussian Process metrics
        summary_["best_dip_score"] = np.nan
        summary_["left_error"] = np.nan
        summary_["right_error"] = np.nan
        summary_["log_sum_error"]=np.nan
        summary_["logL_best_dip"]=np.nan
        summary_["amp_median"]=np.nan
        summary_["amp_std"]=np.nan
        summary_["location_median"]=np.nan
        summary_["location_std"]=np.nan
        summary_["sigma_median"]=np.nan
        summary_["sigma_std"]=np.nan
        summary_["log_sigma2_median"]=np.nan
        summary_["log_sigma2_std"]=np.nan
        summary_["m_median"]=np.nan
        summary_["m_std"]=np.nan
        summary_["b_median"]=np.nan
        summary_["b_std"]=np.nan

        summary_['closest_bright_star_arcsec'] = np.nan
        summary_['closest_bright_star_mag'] = np.nan
        summary_['closest_star_arcsec'] = np.nan
        summary_['closest_star_mag'] = np.nan
        summary_['density_arcsec2'] = np.nan

    else: 
        # prepare the dip for the GP analysis...
        x, y, yerr = digest_the_peak(bp, time, mag, mag_err, expandby=15) # expand by 15 days is usually a good choice

        astropy_fit = quick_Gaussian_fit(time, running_deviation) # run astropy fit on the running deviation

        # prepare the GP model initial guesses using astropy model fitting...
        init = dict(amp=astropy_fit['amplitude'],
            location=bp['peak_loc'],
            sigma=astropy_fit['stddev'], log_sigma2=np.log(0.4))
        
        try:
            we, ws = bp['window_end'].values[0], bp['window_start'].values[0]
        except:
            we, ws = bp['window_end'], bp['window_start']

        gp =  model_gp(x, y, yerr, we, ws, init)

        try:
            # GP assesment of quality 
            gp_quality = evaluate_dip(gp, x, y, yerr, R, S, bp['peak_loc'], diagnostic=False)

            summary_['biweight_scale'] = S
            summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
            summary_['Ndips'] = peak_detections[0] # number of peaks
            summary_['rate'] = peak_detections[0]/(time[-1]-time[0])
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
            summary_["log_sum_error"]=gp_quality["log_sum_error"]
            summary_["logL_best_dip"]=gp_quality["log-like"]
            summary_["amp_median"]=gp_quality["amp_median"]
            summary_["amp_std"]=gp_quality["amp_std"]
            summary_["location_median"]=gp_quality["location_median"]
            summary_["location_std"]=gp_quality["location_std"]
            summary_["sigma_median"]=gp_quality["sigma_median"]
            summary_["sigma_std"]=gp_quality["sigma_std"]
            summary_["log_sigma2_median"]=gp_quality["log_sigma2_median"]
            summary_["log_sigma2_std"]=gp_quality["log_sigma2_std"]
            summary_["m_median"]=gp_quality["m_median"]
            summary_["m_std"]=gp_quality["m_std"]
            summary_["b_median"]=gp_quality["b_median"]
            summary_["b_std"]=gp_quality["b_std"]

            # evaluate Gaia close star statistics
            _ra, _dec = np.median(ra), np.median(dec)
            gaia_info = estimate_gaiadr3_density_async(_ra, _dec, radius=0.01667)

            summary_['closest_bright_star_arcsec'] = gaia_info['closest_bright_star_arcsec']
            summary_['closest_bright_star_mag'] = gaia_info['closest_bright_star_mag']
            summary_['closest_star_arcsec'] = gaia_info['closest_star_arcsec']
            summary_['closest_star_mag'] = gaia_info['closest_star_mag']
            summary_['density_arcsec2'] = gaia_info['density_arcsec2']

            return pd.Series(list(summary_.values()), index=custom_cols)
        except:
            print ("Issue with fitting")
            summary_['biweight_scale'] = S
            summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
            summary_['Ndips'] = 0
            summary_['rate'] = 0
            summary_['best_dip_power'] = np.nan
            summary_['best_dip_time_loc'] = np.nan
            summary_['best_dip_start'] = np.nan
            summary_['best_dip_end'] = np.nan
            summary_['best_dip_dt'] = np.nan
            summary_['best_dip_ndet'] = np.nan

            # Gaussian Process metrics
            summary_["best_dip_score"] = np.nan
            summary_["left_error"] = np.nan
            summary_["right_error"] = np.nan
            summary_["log_sum_error"]=np.nan
            summary_["logL_best_dip"]=np.nan
            summary_["amp_median"]=np.nan
            summary_["amp_std"]=np.nan
            summary_["location_median"]=np.nan
            summary_["location_std"]=np.nan
            summary_["sigma_median"]=np.nan
            summary_["sigma_std"]=np.nan
            summary_["log_sigma2_median"]=np.nan
            summary_["log_sigma2_std"]=np.nan
            summary_["m_median"]=np.nan
            summary_["m_std"]=np.nan
            summary_["b_median"]=np.nan
            summary_["b_std"]=np.nan

            summary_['closest_bright_star_arcsec'] = np.nan
            summary_['closest_bright_star_mag'] = np.nan
            summary_['closest_star_arcsec'] = np.nan
            summary_['closest_star_mag'] = np.nan
            summary_['density_arcsec2'] = np.nan




        