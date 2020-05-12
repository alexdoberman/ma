# -*- coding: utf-8 -*-
import numpy as np
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align


def pmwf_filter(stft_arr_data_noise, stft_arr_data_mix, stft_arr_data_sp, d_arr, beta):
    """
    Implementation algorithm:
    On Optimal Frequency-Domain Multichannel Linear Filtering for Noise Reduction
    Mehrez Souden, Student Member, IEEE, Jacob Benesty, Senior Member, IEEE, and Sofiène Affes, Senior Member, IEEE

    :param stft_arr_data_noise: -  spectrum for each sensors, noise  only period     - shape(bins, num_sensors, frames)
    :param stft_arr_data_mix:   -  spectrum for each sensors, noise + speech period  - shape(bins, num_sensors, frames)
    :param stft_arr_data_sp:    -  spectrum for each sensors, only speech period     - shape(bins, num_sensors, frames)
    :param d_arr:               -  steering vector  - shape (bins, num_sensors)
    :param beta:                -  parameter

    :return:
    """

    (n_bins, n_num_sensors, n_frames_mix) = stft_arr_data_mix.shape
    reg_const = 0.001
    n0        = 0
    u_n0      = np.zeros((n_num_sensors))
    u_n0[n0]  = 1

    #################################################################
    # 1  - Calc psd matrix
    psd_vv = get_power_spectral_density_matrix(stft_arr_data_noise)
    psd_yy = get_power_spectral_density_matrix(stft_arr_data_mix)

    if stft_arr_data_sp is None:
        # This true if uncorrelation of the desired speech and the noise
        psd_xx = psd_yy - psd_vv
    else:
        psd_xx = get_power_spectral_density_matrix(stft_arr_data_sp)

    #################################################################
    # 2 - Regularisation psd matrix
    psd_vv = psd_vv + reg_const*np.identity(psd_vv.shape[-1])

    #################################################################
    # 3 - Calc PMWF filter
    h_W    = np.zeros((n_bins, n_num_sensors), dtype=np.complex)
    I      = np.identity(n_num_sensors)

    for freq_ind in range(0, n_bins):
        F = np.dot(np.linalg.inv(psd_vv[freq_ind, :, :]), psd_xx[freq_ind, :, :])
        lamb = np.trace(F) - n_num_sensors
        F = (F - I)/(beta + lamb)
        h_W[freq_ind, :] = np.dot(F, u_n0)


    #################################################################
    # 4 - Calc filter output
    result_spec = apply_beamforming_vector(h_W, stft_arr_data_mix)

    return result_spec


