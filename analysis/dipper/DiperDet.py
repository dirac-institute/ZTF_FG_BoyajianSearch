
# Original code source from https://github.com/dirac-institute/ZTF_Boyajian/blob/master/dipper.py

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import biweight_location, biweight_scale
from IPython.core.display import display, HTML

import pyspark.sql.functions as sparkfunc
import pyspark.sql.types as stypes

from functools import partial
from collections import defaultdict


def calculate_reference_statistics(times, mags):
    """Calculate reference statistics for a light curve

    We do this using biweight estimators for both the location and scale.  One challenge
    with this is that if there are many repeated observations, which is the case for
    some of the ZTF deep fields, they can overwhelm all of the other observations. To
    mitigate this, we only use the first observation from each night when calculating
    the statistics.

    Parameters
    ----------
    times : ndarray
        Time of each observation.
    mags : ndarray
        Magnitude of each observation.

    Returns
    -------
    reference_magnitude : float
        Reference magnitude
    reference_scale : float
        Reference scale
    """
    # Consider at most one observation from each night.
    _, indices = np.unique(times.astype(int), return_index=True)
    use_mags = mags[indices]

    # Use a robust estimator of the reference time and standard deviation.
    reference_magnitude = biweight_location(use_mags)
    reference_scale = biweight_scale(use_mags)

    return reference_magnitude, reference_scale


def filter_ztf_observations(mjd, mag, magerr, xpos, ypos, catflags):
    """Identify and reject any bad ZTF observations, and return the valid ones.

    We reject any observations with the following characteristics:
    - Any processing flags set
    - Duplicate observations
    - Observations near the edge of a chip
    - Observations near an unflagged bad column.

    We return the resulting magnitude differences along with uncertainties, sorted by
    MJD.

    Parameters
    ----------
    mjd : list of floats
        A list of the mjd times for each observation.
    mag : list of floats
        A list of the observed magnitudes.
    magerr : list of floats
        A list of the observed magnitude uncertanties.
    xpos : list of floats
        A list of the x positions on the CCD for each observation.
    ypos : list of floats
        A list of the y positions on the CCD for each observation.
    catflags : list of floats
        A list of the processing flags for each observation.

    Returns
    -------
    parsed_mjd : ndarray
        Sorted array of parsed MJDs.
    parsed_mag : ndarray
        Corresponding magnitude differences relative to the median flux
    parsed_magerr : ndarray
        Magnitude uncertainties, including contributions from the intrinsic dispersion
        if applicable.
    """
    if len(mjd) == 0:
        return [], [], []

    mjd = np.array(mjd)
    order = np.argsort(mjd)

    # Convert everything to numpy arrays and sort them by MJD
    sort_mjd = mjd[order]
    sort_mag = np.array(mag)[order]
    sort_magerr = np.array(magerr)[order]
    sort_xpos = np.array(xpos)[order]
    sort_ypos = np.array(ypos)[order]
    sort_catflags = np.array(catflags)[order]

    # Mask out bad observations.
    pad_width = 20
    x_border = 3072
    y_border = 3080

    mask = (
        # Remove repeated observations
        (np.abs(sort_mjd - np.roll(sort_mjd, 1)) > 1e-5)

        # Remove anything that is too close to the chip edge. There are lots of weird
        # systematics there.
        & (sort_xpos > pad_width)
        & (sort_xpos < x_border - pad_width)
        & (sort_ypos > pad_width)
        & (sort_ypos < y_border - pad_width)

        # Remove observations that have any flagged data quality problems.
        & (sort_catflags == 0)

        # In the oct19 data, some observations have a magerr of 0 and aren't flagged.
        # This causes a world of problems, so throw them out.
        & (sort_magerr > 0)
    )

    parsed_mjd = sort_mjd[mask]
    parsed_mag = sort_mag[mask]
    parsed_magerr = sort_magerr[mask]

    return parsed_mjd, parsed_mag, parsed_magerr


def _interpolate_target(bin_edges, y_vals, idx, target):
    """Helper to identify when a function y that has been discretized hits value target.
    idx is the first index where y is greater than the target
    """
    if idx == 0:
        y_1 = 0.
    else:
        y_1 = y_vals[idx - 1]
    y_2 = y_vals[idx]

    edge_1 = bin_edges[idx]
    edge_2 = bin_edges[idx + 1]

    frac = (target - y_1) / (y_2 - y_1)
    x = edge_1 + frac * (edge_2 - edge_1)

    return x


def _measure_windowed_dip(mjd, mag, magerr, window_start_mjd, window_end_mjd,
                          dip_edge_percentile=0.05):
    """Measure the properties of a windowed dip, assuming that there is a single dip in
    the given range.

    Parameters
    ----------
    mjd : ndarray
        Observation times. Must be sorted.
    mag : ndarray
        Corresponding measured magnitudes. The reference level must be subtracted, see
        `_calculate_reference_magnitude` and `_combine_band_light_curves` for details.
    magerr : ndarray
        Corresponding measured magnitude uncertainties.
    window_start_mjd : float
        The start of the window to use.
    window_end_mjd : float
        The end of the window to use.
    """
    window_mask = (mjd > window_start_mjd) & (mjd < window_end_mjd)

    window_mjd = mjd[window_mask]
    window_mag = mag[window_mask]
    window_magerr = magerr[window_mask]

    if len(window_mjd) == 0:
        # No light curve to work with
        return None

    # Treat each observation as a bin in time, and assume that the flux remains the same
    # in that bin over its entire interval. We assume that the bin edges are the
    # midpoints of the MJDs of adjacent observations.
    bin_edges = np.hstack([window_start_mjd, (window_mjd[1:] + window_mjd[:-1]) / 2.,
                           window_end_mjd])
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Integrate the dip
    dip_integral = np.sum(window_mag * bin_widths)
    dip_integral_uncertainty = np.sqrt(np.sum(window_magerr**2 * bin_widths**2))

    # Figure out percentiles for the dip
    dip_percentiles = np.cumsum(window_mag * bin_widths) / dip_integral

    # Figure out where the 50th percentile of the dip is.
    dip_idx_center = np.argmax(dip_percentiles > 0.5)
    dip_mjd_center = _interpolate_target(bin_edges, dip_percentiles, dip_idx_center,
                                         0.5)

    # Figure out where the edges of the dip are. If the threshold is crossed multiple
    # times, we choose the time closest to the center of the dip.
    # dip_idx_start = 
    if dip_percentiles[0] > dip_edge_percentile:
        dip_idx_start = 0
    else:
        dip_idx_start = len(dip_percentiles) - np.argmax(dip_percentiles[::-1] <
                                                         dip_edge_percentile)
    dip_mjd_start = _interpolate_target(bin_edges, dip_percentiles,
                                        dip_idx_start, dip_edge_percentile)
    dip_idx_end = np.argmax(dip_percentiles > 1 - dip_edge_percentile)
    dip_mjd_end = _interpolate_target(bin_edges, dip_percentiles, dip_idx_end, 1 -
                                      dip_edge_percentile)

    result = {
        'integral': dip_integral,
        'integral_uncertainty': dip_integral_uncertainty,
        'significance': dip_integral / dip_integral_uncertainty,
        'start_mjd': dip_mjd_start,
        'center_mjd': dip_mjd_center,
        'end_mjd': dip_mjd_end,
        'length': dip_mjd_end - dip_mjd_start,
        'window_start_mjd': window_start_mjd,
        'window_end_mjd': window_end_mjd,
    }

    return result


def parse_light_curve(mjd, mag, magerr, mask_window=None, min_observations=10):
    """Parse a light curve in a single band.

    This subtracts out a reference magnitude, and returns the light curve along with an
    estimate of the reference scale.

    Parameters
    ----------
    mjd : iterable
        Observation times
    mag : iterable
        Corresponding measured magnitudes.
    magerr : iterable
        Corresponding measured magnitude uncertainties.
    mask_window : tuple of two floats or None
        Start and end times of a window to mask out when estimating the background
        level. If None, no masking is performed.
    min_observations : int
        Minimum number of observations required for the background estimation. If there
        are fewer observations than this, we return an empty light curve.

    Returns
    -------
    parsed_mjd : ndarray
        Parsed observation times
    parsed_mag : ndarray
        Magnitudes with the reference subtracted
    parsed_magerr : ndarray
        Magnitude uncertainties
    reference_scale : float
        Scale of deviations in the reference data
    """
    def default_return():
        return np.array([]), np.array([]), np.array([]), -1

    mjd = np.asarray(mjd)
    mag = np.asarray(mag)
    magerr = np.asarray(magerr)

    if len(mjd) < min_observations:
        return default_return()

    # Ensure that everything is sorted.
    if not np.all(np.diff(mjd) >= 0):
        # Arrays aren't sorted, fix that.
        order = np.argsort(mjd)
        mjd = mjd[order]
        mag = mag[order]
        magerr = magerr[order]

    # Mask out a window if desired.
    if mask_window is not None:
        mask = (mjd < mask_window[0]) | (mjd > mask_window[1])
        mask_mjd = mjd[mask]
        mask_mag = mag[mask]

        if len(mask_mjd) < min_observations:
            return default_return()
    else:
        mask_mjd = mjd
        mask_mag = mag

    ref_mag, ref_scale = calculate_reference_statistics(mask_mjd, mask_mag)
    sub_mag = mag - ref_mag

    return mjd, sub_mag, magerr, ref_scale


def find_dip(pulls):
    """Find the longest sequence of significant observations in the data"""
    significant = pulls > 3.

    if np.sum(significant) == 0:
        return 0, 0

    # Find indices of start and end of each significant sequence
    changes = np.diff(np.hstack(([False], significant, [False])))
    significant_sequences = np.where(changes)[0].reshape(-1, 2)

    # Find the index of the longest sequence
    longest_idx = np.argmax(np.diff(significant_sequences, axis=1))
    seq_start, seq_end = significant_sequences[longest_idx]

    return seq_start, seq_end


def _get_dip_window(dip_start, dip_end, pad_fraction, min_pad_length):
    dip_length = dip_end - dip_start
    pad_length = max(min_pad_length, pad_fraction * dip_length)

    window_start = dip_start - pad_length
    window_end = dip_end + pad_length

    return window_start, window_end


def integrate_dip(mjd, mag, magerr, start_time, end_time, method='center'):
    try:
        start_idx = np.where(mjd >= start_time)[0][0]
        end_idx = np.where(mjd <= end_time)[0][-1]
    except IndexError:
        return np.nan, np.nan

    if start_idx == 0 or end_idx == len(mag) - 1:
        # Goes past the end of the light curve, can't handle.
        return np.nan, np.nan

    if method == 'center':
        use_mjd = mjd[start_idx-1:end_idx+2]
        bin_edges = (use_mjd[1:] + use_mjd[:-1]) / 2.
        if bin_edges[0] <= start_time:
            bin_edges[0] = start_time

        if bin_edges[-1] >= end_time:
            bin_edges[-1] = end_time

        bin_edges = np.hstack([start_time, bin_edges, end_time])

        use_mag = mag[start_idx-1:end_idx+2]
        use_magerr = magerr[start_idx-1:end_idx+2]
    elif method in ['left', 'right', 'max', 'min']:
        bin_edges = np.hstack([start_time, mjd[start_idx:end_idx+1], end_time])

        left_mag = mag[start_idx-1:end_idx+1]
        right_mag = mag[start_idx:end_idx+2]
        left_magerr = magerr[start_idx-1:end_idx+1]
        right_magerr = magerr[start_idx:end_idx+2]

        if method == 'left':
            use_mag = left_mag
            use_magerr = left_magerr
        elif method == 'right':
            use_mag = right_mag
            use_magerr = right_magerr
        elif method == 'max':
            use_mag = np.max([left_mag, right_mag], axis=0)
            use_magerr = np.max([left_magerr, right_magerr], axis=0)
        elif method == 'min':
            use_mag = np.min([left_mag, right_mag], axis=0)
            use_magerr = np.min([left_magerr, right_magerr], axis=0)
    else:
        raise ValueError(f"Unknown method {method}")

    bin_widths = bin_edges[1:] - bin_edges[:-1]
    dip_integral = np.sum(use_mag * bin_widths)
    dip_integral_uncertainty = np.sqrt(np.sum(use_magerr**2 * bin_widths**2))

    return dip_integral, dip_integral_uncertainty


def measure_dip(mjd, mag, magerr, min_num_observations=20, min_significant_count=3,
                significant_threshold=3., window_pad_fraction=1.,
                min_window_pad_length=5., min_significance=5.,
                verbose=False, apply_cuts=True):
    """Measure the properties of the largest dip in a light curve.

    This function should be applied to each band separately. The reference level will be
    identified and subtracted before performing any detection.

    Parameters
    ----------
    mjd : iterable
        Observation times
    mag : iterable
        Corresponding measured magnitudes.
    magerr : iterable
        Corresponding measured magnitude uncertainties.
    """
    fail_return = defaultdict(lambda: 0)
    result = {}

    if len(mjd) < min_num_observations:
        if verbose:
            print("Failed to measure dip: not enough observations.")
        return fail_return

    # Subtract the baseline level from each band, and combine their observations into a
    # single light curve.
    initial_mjd, initial_mag, initial_magerr, initial_scale = \
        parse_light_curve(mjd, mag, magerr)

    if len(initial_mjd) < min_num_observations:
        if verbose:
            print("Failed to measure dip: not enough observations.")
        return fail_return

    initial_pulls = initial_mag / np.sqrt(initial_magerr**2 + initial_scale**2)

    if apply_cuts:
        # Check if we have enough significant observations in the full light curve to
        # even bother looking for a dip.
        total_significant_count = np.sum(initial_pulls > significant_threshold)
        if total_significant_count < min_significant_count:
            if verbose:
                print(f"Failed dip cuts: only {total_significant_count} observations "
                      f"of at least {significant_threshold} sigma, require "
                      f"{min_significant_count}.")
            return fail_return

    # Estimate where the dip is.
    initial_dip_start_idx, initial_dip_end_idx = find_dip(initial_pulls)
    initial_significant_count = initial_dip_end_idx - initial_dip_start_idx

    if apply_cuts and initial_significant_count < min_significant_count:
        if verbose:
            print(f"Not enough sequential significant observations "
                  f"({initial_significant_count} < {min_significant_count}")
        return fail_return

    # Get a window around the dip.
    window_start, window_end = _get_dip_window(
        initial_mjd[initial_dip_start_idx], initial_mjd[initial_dip_end_idx-1],
        window_pad_fraction, min_window_pad_length
    )
    result['initial_window_start_mjd'] = window_start
    result['initial_window_end_mjd'] = window_end

    # Redo our estimates of the background levels with the dip masked out.
    mjd, mag, magerr, scale = parse_light_curve(
        mjd, mag, magerr, mask_window=(window_start, window_end)
    )
    result['noise_scale'] = scale

    if len(mjd) < min_num_observations:
        if verbose:
            print("Failed to measure dip: not enough observations in second pass.")
        return fail_return

    inflated_magerr = np.sqrt(magerr**2 + scale**2)
    pulls = mag / inflated_magerr

    # Redo our estimate of where the dip is.
    dip_start_idx, dip_end_idx = find_dip(pulls)
    dip_start_mjd = mjd[dip_start_idx]
    dip_end_mjd = mjd[dip_end_idx-1]
    result['dip_start_mjd'] = dip_start_mjd
    result['dip_end_mjd'] = dip_end_mjd

    # Count how many significant observations there are on different nights
    dip_night_count = len(np.unique(mjd[dip_start_idx:dip_end_idx].astype(int)))
    result['dip_night_count'] = dip_night_count

    # Get a window around the dip.
    window_start, window_end = _get_dip_window(
        mjd[dip_start_idx], mjd[dip_end_idx-1],
        window_pad_fraction, min_window_pad_length
    )
    result['window_start_mjd'] = window_start
    result['window_end_mjd'] = window_end

    significant_count = dip_end_idx - dip_start_idx
    result['significant_count'] = significant_count
    if apply_cuts and significant_count < min_significant_count:
        if verbose:
            print(f"Not enough sequential significant observations in second pass. "
                  f"({initial_significant_count} < {min_significant_count}")
        return fail_return

    # Calculate the total significance of the dip.
    significance = np.sqrt(np.sum(pulls[dip_start_idx:dip_end_idx]**2))
    result['significance'] = significance

    if apply_cuts and significance < min_significance:
        if verbose:
            print(f"Failed dip cuts: significance < {min_significance}")
        return fail_return

    # Integrate the dip.
    integral, integral_err = integrate_dip(
        mjd, mag, inflated_magerr, window_start, window_end)
    result['integral'] = integral
    result['integral_err'] = integral_err

    # Do a "minimum integral" where we use the smallest value of each bin edge for the
    # integral. This is much less affected by very poorly measured light curves.
    min_integral, min_integral_err = integrate_dip(
        mjd, mag, inflated_magerr, window_start, window_end, method='min')
    result['min_integral'] = min_integral
    result['min_integral_err'] = min_integral_err

    # TODO: UPDATE THIS DESCRIPTION
    # Now look for asymmetric dips. We find the peak of the light curve, and then
    # integrate it in both directions so that we can compare the integral on either
    # side. We do these integrals assuming the worst possible sampling. If looking for
    # an asymmetric dip that is larger on the right, we take the maximum of adjacent
    # observations when doing the integral on the left, and the minimum on the right.
    # For well-sampled light curves this doesn't make much of a difference, but for
    # poorly sampled ones it does. We also choose the center time for the comparison
    # to be the right or left-most observation with at least 80% of the maximum depth.
    window_mask = (mjd >= window_start) & (mjd <= window_end)

    # Find the worst case for the time of maximum.
    window_mjd = mjd[window_mask]
    window_mag = mag[window_mask]
    window_magerr = magerr[window_mask]
    max_threshold = np.max(window_mag) * 0.9
    # max_threshold = np.max(window_mag - 2 * window_magerr)
    # max_loc = np.argmax(window_mag)
    # max_time = mjd[max_loc]

    max_locs = np.where(window_mag > max_threshold)[0]
    # max_locs = [np.argmax(window_mag)]
    if max_locs[0] == 0 or max_locs[-1] == len(window_mag) - 1:
        return fail_return

    # left_max_time = window_mjd[max_locs[0] - 1]
    # right_max_time = window_mjd[max_locs[-1] + 1]
    left_max_time = window_mjd[max_locs[0]]
    right_max_time = window_mjd[max_locs[-1]]
    # left_max_time = max_time
    # right_max_time = max_time
    result['left_max_time'] = left_max_time
    result['right_max_time'] = right_max_time

    # Dip to the left.
    min_left_integral, min_left_integral_err = integrate_dip(
        mjd, mag, inflated_magerr, window_start, left_max_time)
    max_right_integral, max_right_integral_err = integrate_dip(
        mjd, mag, inflated_magerr, left_max_time, window_end)
    result['min_left_integral'] = min_left_integral
    result['min_left_integral_err'] = min_left_integral_err
    result['max_right_integral'] = max_right_integral
    result['max_right_integral_err'] = max_right_integral_err
    result['left_dip_diff'] = min_left_integral - max_right_integral
    result['left_dip_diff_err'] = np.sqrt(min_left_integral_err**2 +
                                          max_right_integral_err**2)

    # Dip to the right.
    max_left_integral, max_left_integral_err = integrate_dip(
        mjd, mag, inflated_magerr, window_start, right_max_time)
    min_right_integral, min_right_integral_err = integrate_dip(
        mjd, mag, inflated_magerr, right_max_time, window_end)
    result['max_left_integral'] = max_left_integral
    result['max_left_integral_err'] = max_left_integral_err
    result['min_right_integral'] = min_right_integral
    result['min_right_integral_err'] = min_right_integral_err
    result['right_dip_diff'] = min_right_integral - max_left_integral
    result['right_dip_diff_err'] = np.sqrt(min_right_integral_err**2 +
                                           max_left_integral_err**2)

    if result['left_dip_diff'] > result['right_dip_diff']:
        asymmetry_score = result['left_dip_diff']
        asymmetry_score_err = result['left_dip_diff_err']
    else:
        asymmetry_score = result['right_dip_diff']
        asymmetry_score_err = result['right_dip_diff_err']

    result['asymmetry_score'] = asymmetry_score
    result['asymmetry_score_err'] = asymmetry_score_err
    result['asymmetry_significance'] = asymmetry_score / asymmetry_score_err

    result['outside_window_med_magerr'] = np.median(magerr[~window_mask])
    result['noise_ratio'] = result['noise_scale'] / result['outside_window_med_magerr']

    # Measure stability
    diff_sum = np.sum(np.abs(np.diff(window_mag)))
    exp_diff_sum = np.sum(np.sqrt(window_magerr[1:]**2 + window_magerr[:-1]**2)
                          * np.sqrt(2) / np.sqrt(np.pi))
    result['diff_sum'] = diff_sum
    result['exp_diff_sum'] = exp_diff_sum
    result['diff_ratio'] = (diff_sum - exp_diff_sum) / np.max(window_mag)

    # Time coverage
    window_length = window_end - window_start
    max_gap = np.max(np.diff(np.hstack([window_start, window_mjd, window_end])))
    result['max_gap'] = max_gap
    result['max_gap_ratio'] = max_gap / window_length

    window_mjd = mjd[window_mask]
    window_mag = mag[window_mask]

    def rms(x):
        return np.sqrt(np.mean(x**2))

    result['rms_ratio'] = rms(mag[window_mask]) / rms(mag[~window_mask])

    result['frac_10'] = np.mean(np.abs(mag[~window_mask]) > 0.1 *
                                np.max(mag[window_mask]))
    result['frac_20'] = np.mean(np.abs(mag[~window_mask]) > 0.2 *
                                np.max(mag[window_mask]))
    result['frac_50'] = np.mean(np.abs(mag[~window_mask]) > 0.5 *
                                np.max(mag[window_mask]))

    return result


def measure_dip_ztf(mjds, mags, magerrs, all_xpos, all_ypos, all_catflags, **kwargs):
    """Find the largest dip in any band."""
    best_result = None
    for args in zip(mjds, mags, magerrs, all_xpos, all_ypos, all_catflags):
        valid_mjd, valid_mag, valid_magerr = filter_ztf_observations(*args)
        result = measure_dip(valid_mjd, valid_mag, valid_magerr, **kwargs)
        if best_result is None:
            best_result = result
        elif (result['significance'] > 0
              and result['asymmetry_score'] > best_result['asymmetry_score']):
            best_result = result

    return best_result


def measure_dip_row(row, *args, **kwargs):
    """Wrapper to run measure_dip_ztf on a Spark or pandas row.

    See `measure_dip_ztf` for details.

    Parameters
    ----------
    row : Spark row, pandas row, or dict
        The row containing all of the observation data required for `analyze_dip`
    *args
        Additional arguments to pass to `measure_dip_ztf`
    **kwargs
        Additional keyword arguments to pass to `measure_dip_ztf`

    Returns
    -------
    result : dict
        A dictionary containing the result from `measure_dip_ztf`.
    """
    result = measure_dip_ztf(
        (row['mjd_g'], row['mjd_r'], row['mjd_i']),
        (row['mag_g'], row['mag_r'], row['mag_i']),
        (row['magerr_g'], row['magerr_r'], row['magerr_i']),
        (row['xpos_g'], row['xpos_r'], row['xpos_i']),
        (row['ypos_g'], row['ypos_r'], row['ypos_i']),
        (row['catflags_g'], row['catflags_r'], row['catflags_i']),
        *args,
        **kwargs
    )

    return result


def build_measure_dip_udf(**kwargs):
    """Build a Spark UDF to run `measure_single_dip_ztf`.

    Parameters
    ----------
    **kwargs
        Keyword arguments to pass to `measure_single_dip_ztf`.

    Returns
    -------
    analyze_dip_udf : function
        A wrapped function around `analyze_dip` that uses the given kwargs and that
        can be run in Spark.
    """
    use_keys = {
        'initial_window_start_mjd': float,
        'initial_window_end_mjd': float,
        'dip_start_mjd': float,
        'dip_end_mjd': float,
        'window_start_mjd': float,
        'window_end_mjd': float,
        'significant_count': int,
        'significance': float,
        'integral': float,
        'integral_err': float,
        'min_integral': float,
        'min_integral_err': float,

        'left_max_time': float,
        'right_max_time': float,

        'min_left_integral': float,
        'min_left_integral_err': float,
        'max_right_integral': float,
        'max_right_integral_err': float,
        'left_dip_diff': float,
        'left_dip_diff_err': float,
        'max_left_integral': float,
        'max_left_integral_err': float,
        'min_right_integral': float,
        'min_right_integral_err': float,
        'right_dip_diff': float,
        'right_dip_diff_err': float,
        'asymmetry_score': float,
        'asymmetry_score_err': float,
        'asymmetry_significance': float,
        'dip_night_count': int,

        'noise_scale': float,
        'outside_window_med_magerr': float,
        'noise_ratio': float,
        'rms_ratio': float,

        'diff_sum': float,
        'diff_ratio': float,
        'max_gap': float,
        'max_gap_ratio': float,
        
        'frac_10': float,
        'frac_20': float,
        'frac_50': float,
    }

    sparktype_map = {
        float: stypes.FloatType,
        int: stypes.IntegerType,
    }

    spark_fields = [stypes.StructField(key, sparktype_map[use_type](), True) for key,
                    use_type in use_keys.items()]
    schema = stypes.StructType(spark_fields)

    def _measure_dip_udf(
            mjd_g, mag_g, magerr_g, xpos_g, ypos_g, catflags_g,
            mjd_r, mag_r, magerr_r, xpos_r, ypos_r, catflags_r,
            mjd_i, mag_i, magerr_i, xpos_i, ypos_i, catflags_i,
            **kwargs
    ):
        result = measure_dip_ztf(
            (mjd_g, mjd_r, mjd_i),
            (mag_g, mag_r, mag_i),
            (magerr_g, magerr_r, magerr_i),
            (xpos_g, xpos_r, xpos_i),
            (ypos_g, ypos_r, ypos_i),
            (catflags_g, catflags_r, catflags_i),
            **kwargs
        )

        return [use_type(result[key]) for key, use_type in use_keys.items()]

    dip_udf = sparkfunc.udf(partial(_measure_dip_udf, **kwargs), schema)

    return dip_udf


def _plot_light_curve(row, parsed=True, show_bins=False):
    """Helper for `plot_light_curve` to do the actual work of plotting a light curve.

    Parameters
    ----------
    row : Spark row, pandas row, or dict
        The row containing all of the observation data for a light curve.
    parsed : bool
        If True, the observations in each band will be passed through
        `parse_observations` before plotting them which rejects bad observations and
        subtracts the median magnitude from each filter. Otherwise, the raw observations
        are plotted.
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    band_colors = {
        'g': 'tab:green',
        'r': 'tab:red',
        'i': 'tab:purple'
    }

    if parsed:
        # Run the dipper finder to find the region of the light curve that gets
        # masked out.
        dip_result = measure_dip_row(row)
        mask_window = (dip_result['window_start_mjd'], dip_result['window_end_mjd'])

    for band in ['g', 'r', 'i']:
        if parsed:
            mjd, mag, magerr = filter_ztf_observations(
                row[f'mjd_{band}'],
                row[f'mag_{band}'],
                row[f'magerr_{band}'],
                row[f'xpos_{band}'],
                row[f'ypos_{band}'],
                row[f'catflags_{band}'],
            )
            if len(mjd) < 1:
                continue

            # Subtract out the zeropoint
            mjd, mag, magerr = parse_light_curve(mjd, mag, magerr, mask_window)
        else:
            mask = (
                (np.array(row[f'catflags_{band}']) == 0.)
            )

            mjd = np.array(row[f'mjd_{band}'])[mask]
            mag = np.array(row[f'mag_{band}'])[mask]
            magerr = np.array(row[f'magerr_{band}'])[mask]

        ax.errorbar(mjd, mag, magerr, fmt='o', c=band_colors[band], label=f'ZTF-{band}')

    ax.set_xlabel('MJD')
    if parsed:
        ax.set_ylabel('Magnitude + offset')
    else:
        ax.set_ylabel('Magnitude')
    ax.legend()
    ax.set_title('objid %d' % row['ps1_objid'])
    ax.invert_yaxis()

    return ax


def _print_light_curve_info(row):
    """Plot information about a light curve.

    This outputs a simbad link 

    Parameters
    ----------
    row : Spark row, pandas row, or dict
        The row containing all of the observation data for a light curve.
    """
    display(HTML("<a href='http://simbad.u-strasbg.fr/simbad/sim-coo?Coord=%.6f%+.6f&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius=20&Radius.unit=arcsec&submit=submit+query&CoordList='>SIMBAD</link>" % (row['ra'], row['dec'])))
    print("RA+Dec: %.6f%+.6f" % (row['ra'], row['dec']))


def plot_light_curve(row, parsed=True, label_dip=True, zoom=False, verbose=True):
    """Plot a light curve.

    Parameters
    ----------
    row : Spark row, pandas row, or dict
        The row containing all of the observation data for a light curve.
    parsed : bool
        If True, the observations in each band will be passed through
        `parse_observations` before plotting them which rejects bad observations and
        subtracts the median magnitude from each filter. Otherwise, the raw observations
        are plotted.
    label_dip : bool
        If True, the dip is labeled on the plot with vertical lines. This is ignored if
        the dip statistics haven't been calculated yet.
    zoom : bool
        If True, the plot is zoomed in around the dip.
    verbose : bool
        If True, print out information about the dip.
    """
    if label_dip:
        dip_result = measure_dip_row(row)

    if verbose:
        _print_light_curve_info(row)

        if label_dip:
            print("")
            print("Dip details:")
            for key, value in dip_result.items():
                print(f"{key:11s}: {value}")

    ax = _plot_light_curve(row, parsed)

    if label_dip:
        window_start_mjd = dip_result['window_start_mjd']
        window_end_mjd = dip_result['window_end_mjd']

        ax.axvline(dip_result['center_mjd'], c='C0', label='Dip location')
        ax.axvline(dip_result['start_mjd'], c='C0', ls='--', label='Dip edges')
        ax.axvline(dip_result['end_mjd'], c='C0', ls='--')
        ax.axvline(window_start_mjd, c='k', ls='--', label='Window')
        ax.axvline(window_end_mjd, c='k', ls='--')

        if zoom:
            pad = (window_end_mjd - window_start_mjd)
            ax.set_xlim(window_start_mjd - pad, window_end_mjd + pad)

        ax.legend()


def plot_interactive(rows):
    """Generate an interactive plot for a set of rows.

    Parameters
    ----------
    rows : List of spark rows
        A list of spark rows where each row contains all of the observation data for a
        light curve.
    """
    from ipywidgets import interact, IntSlider

    max_idx = len(rows) - 1

    def interact_light_curve(idx, parsed=True, label_dip=True, zoom=False,
                             verbose=True, both_zoom=False):
        if both_zoom or not zoom:
            plot_light_curve(rows[idx], parsed=parsed, label_dip=label_dip, zoom=False,
                             verbose=verbose)
        if both_zoom or zoom:
            plot_light_curve(rows[idx], parsed=parsed, label_dip=label_dip, zoom=True,
                             verbose=verbose)

    interact(interact_light_curve, idx=IntSlider(0, 0, max_idx))


def plot_dip(row):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    dip_result = measure_dip_row(row, return_parsed_observations=True)

    mjd = dip_result['parsed_mjd']
    mag = dip_result['parsed_mag']
    magerr = dip_result['parsed_magerr']

    ax.scatter(mjd, mag, s=5, c='k', zorder=3, label='Individual observations')

    ax.fill_between(mjd, mag, step='mid', alpha=0.2, label='Used profile')
    ax.plot(mjd, mag, drawstyle='steps-mid', alpha=0.5)

    ax.axvline(dip_result['start_mjd'], c='C2', lw=2, label='Dip boundary')
    ax.axvline(dip_result['end_mjd'], c='C2', lw=2)
    ax.axvline(dip_result['center_mjd'], c='C1', lw=2, label='Dip center')

    ax.set_xlim(dip_result['start_mjd'] - 30, dip_result['end_mjd'] + 30)
    ax.legend()
    ax.invert_yaxis()
    ax.set_xlabel('MJD')
    ax.set_ylabel('Magnitude + offset')
