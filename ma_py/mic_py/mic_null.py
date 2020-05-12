# -*- coding: utf-8 -*-
import numpy as np
import copy
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix

def calc_null_weights(d_arr, C):
    """

    :param d_arr:    - Steering vector in desired direction - shape (bins, num_sensors)
    :param C:        - Constraints matrix - shape (bins, num_sensors, num_constraints)
    :return:
        null_weights - NULL steering weights -shape (bins, num_sensors)
    """


    bins, num_sensors, num_constraints = C.shape
    null_weights = np.zeros((bins, num_sensors), dtype=np.complex)


    # Inverse regularizaion denominator
    ird_const = 0.001
    d_arr = d_arr/num_sensors

    for freq_ind in range(0, bins, 1):

        C_H = np.conjugate(C[freq_ind, :, :]).T
        denominator = np.dot(C_H, C[freq_ind, :, :])

        det = np.linalg.det(denominator)
        # Stability  matrix inversion
        if np.abs(det) < ird_const:
            denominator += ird_const * np.identity(num_constraints)
        denominator_inv = np.linalg.inv(denominator)

        P_ort = np.identity(num_sensors)  - np.dot ( np.dot(C[freq_ind, :, :] , denominator_inv), C_H)
        w = np.dot(d_arr[freq_ind, :].conj(), P_ort)
        null_weights[freq_ind, :] = w.conj()


    return null_weights

def null_filter(stft_mix, d_arr_sp, d_arr_inf):
    """

    :param stft_mix:    - spectr mix for each sensors - shape (bins, num_sensors, frames)
    :param d_arr_sp:    - steering vector in speaker direction - shape (bins, num_sensors)
    :param d_arr_inf:   - steering vector in inference direction - shape (bins, num_sensors)

    :return:
        result_spec - result spectral  - shape (bins, frames)
        null_weights - LCMV weights -shape (bins, num_sensors)
    """

    bins, num_sensors, frames = stft_mix.shape

    # Define constraint matrix
    C = np.zeros((bins, num_sensors, 1), dtype=np.complex)
    C[:, :, 0] = d_arr_inf

    null_weights =  calc_null_weights(d_arr_sp, C)
    output = apply_beamforming_vector(null_weights, stft_mix)

    return output, null_weights

def null_filter_ex(stft_mix, d_arr_sp, lst_d_arr_inf):
    """

    :param stft_mix:    - spectr mix for each sensors - shape (bins, num_sensors, frames)
    :param d_arr_sp:    - steering vector in speaker direction - shape (bins, num_sensors)
    :param lst_d_arr_inf:   - list steering vector in inference direction - shape (bins, num_sensors)

    :return:
        result_spec - result spectral  - shape (bins, frames)
        null_weights - LCMV weights -shape (bins, num_sensors)
    """

    bins, num_sensors, frames = stft_mix.shape

    # Define constraint matrix
    num_constrains = len(lst_d_arr_inf)
    C = np.zeros((bins, num_sensors, num_constrains), dtype=np.complex)
    for i in range(num_constrains):
        C[:, :, i] = lst_d_arr_inf[i]

    null_weights =  calc_null_weights(d_arr_sp, C)
    output = apply_beamforming_vector(null_weights, stft_mix)

    return output, null_weights