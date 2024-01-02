from dipper import *
from tools import *
import astropy.stats as astro_stats

def evaluate(time, mag, mag_err, flag, band, ra, dec, gaia_lite, custom_cols):

    # Summary information
    summary_ = {}
    
    # Digest my light curve. Select band, good detections & sort
    time, mag, mag_err = prepare_lc(time, mag, mag_err, flag, band,  band_of_study='r', flag_good=0, q=None, custom_q=False)

    # Evaluate biweight location and scale
    R, S = astro_stats.biweight.biweight_location(mag), astro_stats.biweight.biweight_scale(mag)

    # Running deviation
    running_deviation = deviation(mag, mag_err, R, S)

    # Peak detection summary per light curve
    peak_detections = peak_detector(time, running_deviation, power_thresh=3, peak_close_rmv=15, pk_2_pk_cut=30)

    # Select best peak candidate with at least 3 points in the dip
    bp = best_peak_detector(peak_detections, min_in_dip=3)

    # If no peaks found...
    if peak_detections[0]==0 or bp is None:
        print ("No peaks found!")
        summary_['biweight_scale'] = S
        summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
        summary_['Ndips'] = peak_detections[0] # number of peaks
        summary_['best_dip_power'] = np.nan
        summary_['best_dip_time'] = np.nan
        summary_['best_dip_start'] = np.nan
        summary_['best_dip_end'] = np.nan
        summary_['best_dip_dt'] = np.nan
        summary_['best_dip_ndet'] = np.nan
        summary_['best_dip_integral_score'] = np.nan
        summary_['chi-square-gp'] = np.nan
        summary_['closest_bright_star_arcsec'] = np.nan
        summary_['closest_bright_star_mag'] = np.nan
        summary_['closest_star_arcsec'] = np.nan
        summary_['closest_star_mag'] = np.nan
        summary_['density_arcsec2'] = np.nan
    else: # ready for a GP...
        # prepare the dip for the GP analysis...
        x, y, yerr = digest_the_peak(bp, time, mag, mag_err, expandby=0)

        # feed to the GP
        gp = GaussianProcess_dip(x, y, yerr, length_scale=0.01)

        # GP assesment of quality 
        gp_quality = evaluate_dip(gp, x, y, yerr, R, S, bp['peak_loc'], diagnostic=False)

        if gp_quality is not None:
            summary_['biweight_scale'] = S
            summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
            summary_['Ndips'] = peak_detections[0] # number of peaks
            summary_['best_dip_power'] = bp['dip_power']
            summary_['best_dip_time_loc'] = bp['peak_loc']
            summary_['best_dip_start'] = bp['window_start']
            summary_['best_dip_end'] = bp['window_end']
            summary_['best_dip_dt'] = bp['average_dt_dif']
            summary_['best_dip_ndet'] = bp['N_in_dip']
            summary_['best_dip_integral_score'] = gp_quality['assymetry_score']
            summary_['chi-square-gp'] = gp_quality['chi-square']

            # evaluate Gaia close star statistics
            _ra, _dec = np.median(ra), np.median(dec)
            gaia_info = estimate_gaiadr3_density(_ra, _dec, gaia_lite, radius=0.01667)

            summary_['closest_bright_star_arcsec'] = gaia_info['closest_bright_star_arcsec']
            summary_['closest_bright_star_mag'] = gaia_info['closest_bright_star_mag']
            summary_['closest_star_arcsec'] = gaia_info['closest_star_arcsec']
            summary_['closest_star_mag'] = gaia_info['closest_star_mag']
            summary_['density_arcsec2'] = gaia_info['density_arcsec2']

            return pd.Series(list(summary_.values()), index=custom_cols)
        else:
            print ("No peaks found!")
            summary_['biweight_scale'] = S
            summary_['frac_above_2_sigma'] = len(running_deviation[running_deviation>np.mean(running_deviation)+2*np.std(running_deviation)])/len(running_deviation)
            summary_['Ndips'] = peak_detections[0] # number of peaks
            summary_['best_dip_power'] = np.nan
            summary_['best_dip_time'] = np.nan
            summary_['best_dip_start'] = np.nan
            summary_['best_dip_end'] = np.nan
            summary_['best_dip_dt'] = np.nan
            summary_['best_dip_ndet'] = np.nan
            summary_['best_dip_integral_score'] = np.nan
            summary_['chi-square-gp'] = np.nan
            summary_['closest_bright_star_arcsec'] = np.nan
            summary_['closest_bright_star_mag'] = np.nan
            summary_['closest_star_arcsec'] = np.nan
            summary_['closest_star_mag'] = np.nan
            summary_['density_arcsec2'] = np.nan




    