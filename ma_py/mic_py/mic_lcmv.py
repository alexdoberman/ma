# -*- coding: utf-8 -*-
import numpy as np
import copy
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix

def calc_lcmv_weights(SN, C, G):
    """

    :param SN:  - Power spectral density matrix for noise - shape (bins, num_sensors, num_sensors)
    :param C:   - Constraints matrix - shape (bins, num_sensors, num_constraints)
    :param G:   - Constraints vector - shape (1, num_constraints)
    :return:
        lcmv_weights - LCMV weights -shape (bins, num_sensors)
    """

    bins, num_sensors, _ = C.shape
    lcmv_weights = np.zeros((bins, num_sensors), dtype=np.complex)

    # Inverse cross-spectr
    SN_inv = np.linalg.inv(SN)

    # Inverse regularizaion denominator
    ird_const = 0.1

    for freq_ind in range(0, bins, 1):
        C_H = np.conjugate(C[freq_ind, :, :]).T

        numerator = np.dot(C_H, SN_inv[freq_ind, :, :])
        denominator = np.dot(numerator, C[freq_ind, :, :])

        det = np.linalg.det(denominator)
        #print('freq_ind = {}, det = {}'.format(freq_ind, np.abs(det)))

        # Stability  matrix inversion
        if np.abs(det) < ird_const:
            denominator += ird_const * np.identity(denominator.shape[-1])

        denominator_inv = np.linalg.inv(denominator)
        w = np.dot(np.dot(G, denominator_inv), numerator)

        norm_w = abs(np.dot(np.conjugate(w), w.T))
        #print('freq_ind = {}, ||w||^2 = {}'.format(freq_ind, norm_w))

        lcmv_weights[freq_ind, :] = w.conj()

    return lcmv_weights

def lcmv_filter(stft_mix, stft_noise, d_arr_sp, d_arr_inf):
    """

    :param stft_mix:    - spectr mix for each sensors - shape (bins, num_sensors, frames)
    :param stft_noise:  - spectr noise only for each sensors - shape (bins, num_sensors, frames)
    :param d_arr_sp:    - steering vector in speaker direction - shape (bins, num_sensors)
    :param d_arr_inf:   - steering vector in inference direction - shape (bins, num_sensors)

    :return:
        result_spec - result spectral  - shape (bins, frames)
        lcmv_weights - LCMV weights -shape (bins, num_sensors)
    """

    bins, num_sensors, frames = stft_mix.shape

    mvdr_reg_const = 0.001
    psd = get_power_spectral_density_matrix(stft_noise)
    psd = psd + mvdr_reg_const * np.identity(psd.shape[-1])

    # Define constraint matrix
    C = np.zeros((bins, num_sensors, 2), dtype=np.complex)
    C[:, :, 0] = d_arr_sp
    C[:, :, 1] = d_arr_inf

    # Define constraint vector
    g = np.zeros((1, 2), dtype=np.complex)
    g[0, 0] = 1
    g[0, 1] = 0


    # # Define constraint matrix
    # C = np.zeros((bins, num_sensors, 1), dtype=np.complex)
    # C[:, :, 0] = d_arr_sp
    #
    # # Define constraint vector
    # g = np.zeros((1, 1), dtype=np.complex)
    # g[0, 0] = 1

    lcmv_weights = calc_lcmv_weights(psd, C, g)
    output = apply_beamforming_vector(lcmv_weights, stft_mix)

    return output, lcmv_weights

def lcmv_filter_debug(stft_mix, d_arr_sp, d_arr_inf):
    """

    :param stft_mix:    - spectr mix for each sensors - shape (bins, num_sensors, frames)
    :param d_arr_sp:    - steering vector in speaker direction - shape (bins, num_sensors)
    :param d_arr_inf:   - steering vector in inference direction - shape (bins, num_sensors)

    :return:
        result_spec - result spectral  - shape (bins, frames)
    """

    bins, num_sensors, frames = stft_mix.shape

    mvdr_reg_const = 1.0
    psd = mvdr_reg_const * copy.copy(np.broadcast_to (np.identity(num_sensors) , (bins, num_sensors, num_sensors)))

    # Define constraint matrix
    C = np.zeros((bins, num_sensors, 2), dtype=np.complex)
    C[:, :, 0] = d_arr_sp
    C[:, :, 1] = d_arr_inf

    # Define constraint vector
    g = np.zeros((1, 2), dtype=np.complex)
    g[0, 0] = 1
    g[0, 1] = 0

    # # Define constraint matrix
    # C = np.zeros((bins, num_sensors, 1), dtype=np.complex)
    # C[:, :, 0] = d_arr_sp
    #
    # # Define constraint vector
    # g = np.zeros((1, 1), dtype=np.complex)
    # g[0, 0] = 1

    lcmv_weights = calc_lcmv_weights(psd, C, g)
    output = apply_beamforming_vector(lcmv_weights, stft_mix)

    return output

