# -*- coding: utf-8 -*-
import numpy as np
from numba import jit


@jit(nopython=True)
def estimate_psd_hendriks(stft_mix, d_arr_sp, reg_const = 0.1):

    """
    Estimate PSD matrix : Noise Correlation Matrix Estimation for Multi-Microphone Speech Enhancement,
                          Richard C. Hendriks and Timo Gerkmann

    :param stft_mix:       - spectr mix for each sensors - shape (bins, num_sensors, frames)
    :param d_arr_sp:       - steering vector in speaker direction - shape (bins, num_sensors)
    :param reg_const:      - reg const
    :return:
    """

    (n_bins, n_sensors, n_frames) = stft_mix.shape
    psd = np.zeros((n_bins, n_sensors, n_sensors), dtype=np.complex64)

    d_n_m = np.zeros((n_sensors, n_sensors, n_bins), dtype=np.complex64)
    for n in range(n_sensors):
        for m in range(n_sensors):
            d_n_m[n,m] = d_arr_sp[:, n] / (d_arr_sp[:, m] + reg_const)

    # for n in range(n_sensors):
    #     for m in range(n_sensors):
    #         reg = (np.abs(d_arr_sp[:, m]) < reg_const)*reg_const
    #         sp_m = d_arr_sp[:, m] + reg
    #         d_n_m[n,m] = d_arr_sp[:, n] / sp_m

    for frame in range(n_frames):
        if frame % 100 == 0:
            print('     frame = ', frame)

        for n in range(n_sensors):
            for m in range(n_sensors):

                Y_n = stft_mix[:, n, frame]
                Y_m = stft_mix[:, m, frame]

                P_n_m = Y_n - np.conj(d_n_m[n,m]) * Y_m
                #P_n_m = Y_n - d_n_m[n,m] * Y_m
                #P_n_m = Y_n

                psd[:, n,m] += P_n_m * np.conj(Y_m)
    psd /= n_frames

    for bin in range(n_bins):
        psd[bin, :, :] = (psd[bin, :, :] + np.conj(psd[bin, :, :]).T) / 2

    return psd

