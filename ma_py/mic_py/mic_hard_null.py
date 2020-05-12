# -*- coding: utf-8 -*-
import copy
import numpy as np
from mic_py.mic_ds_beamforming import ds_beamforming
from mic_py.mic_adaptfilt import *
from mic_py.mic_adaptfilt_time_domain import *
from mic_py.feats import istft


def hard_null_filter(stft_arr, d_arr_sp, d_arr_inf, alg_type = 1):
    """

    :param stft_arr:   - spectr for each sensors - shape (bins, num_sensors, frames)
    :param d_arr_sp:   - steering vector in speaker direction - shape (bins, num_sensors)
    :param d_arr_inf:  - steering vector in inference direction - shape (bins, num_sensors)
    :param alg_type:   - type adaptive algorithm

             alg_type = 0  - compensate ref channel
             alg_type = 1  - spectral substract
             alg_type = 2  - Stolbov filter
    :return:
        result_spec - result spectral  - shape (bins, frames)
    """
    (n_bins, n_num_sensors, n_frames) = stft_arr.shape

    spec_sp  = ds_beamforming(stft_arr, d_arr_sp.T)
    spec_inf = ds_beamforming(stft_arr, d_arr_inf.T)

    if alg_type == 0:
        result_spec = compensate_ref_ch_filter(stft_main=spec_sp, stft_ref=spec_inf, alfa=0.7)
    elif alg_type == 1:
        result_spec = spectral_substract_filter(stft_main=spec_sp, stft_ref=spec_inf, alfa_PX=0.01, alfa_PN=0.099)
    elif alg_type == 2:
        result_spec = smb_filter(stft_main=spec_sp, stft_ref=spec_inf, gain_max=18)
    else:
        raise ValueError('hard_null_filter_ex: unsupported alg_type = {}'.format(alg_type))

    return result_spec


def hard_null_filter_time_domain(stft_arr, d_arr_sp, d_arr_inf, alg_type=4, overlap=2):
    """

    :param stft_arr:   - spectr for each sensors - shape (bins, num_sensors, frames)
    :param d_arr_sp:   - steering vector in speaker direction - shape (bins, num_sensors)
    :param d_arr_inf:  - steering vector in inference direction - shape (bins, num_sensors)
    :param alg_type:   - type adaptive algorithm

             alg_type = 3  - LMS
             alg_type = 4  - NLMS
             alg_type = 5  - AP

    :return:

        sig_result - result signal in time domain  - shape (samples,)
    """
    (n_bins, n_num_sensors, n_frames) = stft_arr.shape

    spec_sp = ds_beamforming(stft_arr, d_arr_sp.T)
    spec_inf = ds_beamforming(stft_arr, d_arr_inf.T)

    sig_sp = istft(spec_sp.transpose((1, 0)), overlap=overlap)
    sig_inf = istft(spec_inf.transpose((1, 0)), overlap=overlap)

    if alg_type == 3:

        # LMS filter

        M = 200
        step = 0.05
        leak = 0.0
        delay = -5
        #delay = 0

        sig_inf = np.roll(sig_inf, delay)
        sig_result = lms_filter(main=sig_sp, ref=sig_inf, M=M, step=step, leak=leak, norm=False)

    elif alg_type == 4:

        # NMLS filter
        M = 200
        step = 0.05
        leak = 0.09
        delay = -5
        #delay = 0

        sig_inf = np.roll(sig_inf, delay)
        sig_result = lms_filter(main=sig_sp, ref=sig_inf, M=M, step=step, leak=leak, norm=True)


    elif alg_type == 5:

        # Affine projection filter
        M = 200
        step = 0.05
        L = 5
        leak = 0.0
        delay = -5

        sig_inf = np.roll(sig_inf, delay)
        sig_result = affine_projection_filter(main=sig_sp, ref=sig_inf, M=M, step=step, L=L, leak=leak)

    else:
        raise ValueError('hard_null_filter_time_domain: unsupported alg_type = {}'.format(alg_type))

    return sig_result










