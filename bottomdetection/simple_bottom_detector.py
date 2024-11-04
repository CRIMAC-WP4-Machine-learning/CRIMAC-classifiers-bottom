# -*- coding: utf-8 -*-
"""
A simple threshold-based bottom detection.

Copyright (c) 2021, Contributors to the CRIMAC project.
Licensed under the MIT license.
"""

import numpy as np
import xarray as xr

from bottomdetection import bottom_utils
from bottomdetection.parameters import Parameters


def detect_bottom(zarr_data: xr.Dataset, parameters: Parameters = Parameters()) -> xr.DataArray:
    channel_index = 0
    channel_sv = zarr_data['sv'][channel_index]
    threshold_sv = 10 ** (parameters.threshold_log_sv / 10)

    heave_corrected_transducer_depth = zarr_data['heave'] + zarr_data['transducer_draft'][channel_index]

    pulse_duration = float(zarr_data['pulse_length'][channel_index].values[0])
    depth_ranges, indices = bottom_utils.detect_bottom_single_channel(channel_sv, threshold_sv, heave_corrected_transducer_depth, pulse_duration, parameters.minimum_range)

    depth_ranges_back_step, indices_back_step = back_step(channel_sv, indices, heave_corrected_transducer_depth,
                                                          0.001)

    bottom_depths = heave_corrected_transducer_depth + depth_ranges_back_step - parameters.offset
    bottom_depths = xr.DataArray(name='bottom_depth', data=bottom_depths, dims=['ping_time'],
                                 coords={'ping_time': zarr_data['ping_time']})
    return bottom_depths


def _back_step_inner(v, v_prev, v_next, shift, shift_prev, shift_next, di, min_depth_value_fraction: float,
                     segment_nsamples):
    if di < 0:
        return -1
    vi = v[di]
    vs = bottom_utils.stack_max(v, v_prev, v_next, shift, shift_prev, shift_next)
    back_step_offset = 1
    segment_end = di
    while segment_end > 0:
        segment_start = max(0, segment_end - segment_nsamples)
        actual_length = segment_end - segment_start
        # stack shifted arrays in range direction and take max
        a = np.nanmax(np.stack([np.asarray(vs[segment_start:segment_end])[::-1],
                                np.asarray(bottom_utils.shift_arr(vs, 1)[segment_start:segment_end])[::-1],
                                np.asarray(bottom_utils.shift_arr(vs, -1)[segment_start:segment_end])[::-1]]), axis=0)
        back_step_offset_segment = (a <= min_depth_value_fraction * vi).argmax()
        back_step_offset_segment = back_step_offset_segment \
            if a[back_step_offset_segment] <= min_depth_value_fraction * vi else actual_length
        back_step_offset += back_step_offset_segment
        segment_end -= segment_nsamples
        if back_step_offset_segment < actual_length:
            break
    # forward step on an unsmoothed array
    forward_step_offset = 0
    if v[di - back_step_offset] < min_depth_value_fraction * vi:
        forward_step_offset = (np.asarray(v[di - back_step_offset + 1:di]) > min_depth_value_fraction * vi).argmax()
    return di - back_step_offset + forward_step_offset


def back_step(sv_array: xr.DataArray, depths_indices: xr.DataArray, depth_correction, min_depth_value_fraction: float):
    """
    Find minimum bottom depths by back stepping

    :param sv_array: an array of sv values for a channel
    :param depths_indices: sample indices of detected depth
    :param depth_correction: the recorded depth correction (heave plus transducer draft)
    :param min_depth_value_fraction: a fraction of the detected bottom echo strength
    :return: a data array of minimum bottom depths and the indices of the minimum bottom depths
    """
    segment_distance = 5
    segment_nsamples: int = int(segment_distance / (sv_array.range[1] - sv_array.range[0]))
    range_shift = depth_correction / (sv_array.range[1] - sv_array.range[0])

    back_step_indices = xr.apply_ufunc(_back_step_inner,
                                       sv_array,
                                       sv_array.shift(ping_time=1),
                                       sv_array.shift(ping_time=-1),
                                       range_shift,
                                       range_shift.shift(ping_time=1),
                                       range_shift.shift(ping_time=-1),
                                       depths_indices,
                                       input_core_dims=[['range'], ['range'], ['range'],
                                                        [], [], [], []],
                                       kwargs={'min_depth_value_fraction': min_depth_value_fraction,
                                               'segment_nsamples': segment_nsamples},
                                       vectorize=True,
                                       dask='parallelized',
                                       output_dtypes=[np.int64]
                                       )

    bottom_depths = sv_array.range[back_step_indices].where(back_step_indices >= 0, np.nan)
    return xr.DataArray(name='bottom_depth_backstep', data=bottom_depths, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time}), \
           xr.DataArray(name='bottom_index_backstep', data=back_step_indices, dims=['ping_time'],
                        coords={'ping_time': sv_array.ping_time})
