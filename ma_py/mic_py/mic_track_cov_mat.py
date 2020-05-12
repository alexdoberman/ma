# -*- coding: utf-8 -*-
import numpy as np
from  mic_py.mic_blocking_matrix import calc_blocking_matrix_from_steering
from mic_py.beamforming import  get_power_spectral_density_matrix2
import matplotlib.pyplot as plt


def calc_psd_v0(stft_noise_arr):
    psd_V0 = get_power_spectral_density_matrix2(stft_noise_arr)
    return psd_V0

def track_cov_mat_filter_2(stft_arr, d_arr):
    """

    :param stft_arr: - spectr for each sensors - shape (bins, num_sensors, frames)
    :param d_arr:    - steering vector         - shape (bins, num_sensors)
    :return:
    """


    (bins, num_sensors, frames) = stft_arr.shape

    # Calc blockin matrix
    # B      - blocking_matrix         - shape (num_sensors, num_sensors - num_constrain, bins)
    B = calc_blocking_matrix_from_steering(d_arr.T)

    # Calc Z - output from blocking matrix
    Z = np.zeros((bins, num_sensors - 1, frames), dtype=np.complex)
    for frame_ind in range(0, frames):
        for freq_ind in range(0, bins):

            # Get  output of blocking matrix.
            XK = stft_arr[freq_ind, : ,frame_ind]
            ZK = np.dot(np.conjugate(B[:,:,freq_ind]).T, XK)
            Z[freq_ind, : ,frame_ind] = ZK

    psd_Z = get_power_spectral_density_matrix2(Z)

    # Calc psd_Z another
    psd_Z2 = np.zeros((bins, num_sensors - 1, num_sensors - 1), dtype=np.complex)

    # shape (bins, num_sensors, num_sensors)
    psd_V = get_power_spectral_density_matrix2(stft_arr)
    for freq_ind in range(0, bins):
        # Get  output of blocking matrix.
        P1 = np.dot(np.conjugate(B[:, :, freq_ind]).T, psd_V[freq_ind, :, :])
        P2 = np.dot(P1, B[:, :, freq_ind])
        psd_Z2[freq_ind, :, :] = P2

    diff = psd_Z - psd_Z2


    # Inverse nose psd
    psd_inv_Z2 = np.zeros((bins, num_sensors - 1, num_sensors - 1), dtype=np.complex)
    eps = 0.0000000000001

    for freq_ind in range(0, bins):
        P = psd_Z2[freq_ind, :, :] + eps* np.identity(psd_Z2.shape[-1])
        psd_inv_Z2[freq_ind, :, :] = np.linalg.inv(P)

    # Estimate c^2
    for freq_ind in range(0, bins):
        c = 0
        for frame_ind in range(0, frames):
            ZK = Z[freq_ind, : ,frame_ind]
            ZKH = np.conjugate(ZK)

            _c  = np.dot(ZKH, psd_inv_Z2[freq_ind, :, :])
            _cc = np.inner(_c, ZK)

            c = c + _cc

        c/=frames
        c/=(num_sensors-1)
        print ("freq = ", freq_ind, " c = ", np.real(c))

    # Estimate c^2 exponential average

    # freq_ind = 25
    # alfa = 0.75
    #
    # cc = np.zeros(frames)
    # for frame_ind in range(0, frames):
    #     ZK = Z[freq_ind, : ,frame_ind]
    #     ZKH = np.conjugate(ZK)
    #     _c  = np.dot(ZKH, psd_inv_Z2[freq_ind, :, :])
    #     _cc = np.inner(_c, ZK)
    #     _cc /= (num_sensors - 1)
    #
    #     cc[frame_ind] = alfa * cc[frame_ind - 1] + (1-alfa)* np.real(_cc)
    #
    # plt.plot(range(0, frames), cc)
    # plt.show()



    print(psd_Z.shape)

def track_cov_mat_filter(stft_arr, d_arr, psd_V0):
    """

    :param stft_arr: - spectr for each sensors - shape (bins, num_sensors, frames)
    :param d_arr:    - steering vector         - shape (bins, num_sensors)
    :param psd_V0:   - shape (bins, num_sensors, num_sensors)
    :return:
    """


    (bins, num_sensors, frames) = stft_arr.shape

    # Calc blockin matrix
    # B      - blocking_matrix         - shape (num_sensors, num_sensors - num_constrain, bins)
    B = calc_blocking_matrix_from_steering(d_arr.T)

    # Calc Z - output from blocking matrix
    Z = np.zeros((bins, num_sensors - 1, frames), dtype=np.complex)
    for frame_ind in range(0, frames):
        for freq_ind in range(0, bins):

            # Get  output of blocking matrix.
            XK = stft_arr[freq_ind, : ,frame_ind]
            ZK = np.dot(np.conjugate(B[:,:,freq_ind]).T, XK)
            Z[freq_ind, : ,frame_ind] = ZK

    #psd_Z = get_power_spectral_density_matrix2(Z)

    # Calc psd_Z another
    psd_Z2 = np.zeros((bins, num_sensors - 1, num_sensors - 1), dtype=np.complex)

    # shape (bins, num_sensors, num_sensors)
    for freq_ind in range(0, bins):
        # Get  output of blocking matrix.
        P1 = np.dot(np.conjugate(B[:, :, freq_ind]).T, psd_V0[freq_ind, :, :])
        P2 = np.dot(P1, B[:, :, freq_ind])
        psd_Z2[freq_ind, :, :] = P2


    # Inverse nose psd
    psd_inv_Z2 = np.zeros((bins, num_sensors - 1, num_sensors - 1), dtype=np.complex)
    eps = 0.0000000000001

    for freq_ind in range(0, bins):
        P = psd_Z2[freq_ind, :, :] + eps* np.identity(psd_Z2.shape[-1])
        psd_inv_Z2[freq_ind, :, :] = np.linalg.inv(P)

    # # Estimate c^2
    # for freq_ind in range(0, bins):
    #     c = 0
    #     for frame_ind in range(0, frames):
    #         ZK = Z[freq_ind, : ,frame_ind]
    #         ZKH = np.conjugate(ZK)
    #
    #         _c  = np.dot(ZKH, psd_inv_Z2[freq_ind, :, :])
    #         _cc = np.inner(_c, ZK)
    #
    #         c = c + _cc
    #
    #     c/=frames
    #     c/=(num_sensors-1)
    #     print ("freq = ", freq_ind, " c = ", np.real(c))

    # Estimate c^2 exponential average

    freq_ind = 150
    alfa = 0.97

    cc = np.zeros(frames)
    for frame_ind in range(0, frames):
        ZK = Z[freq_ind, : ,frame_ind]
        ZKH = np.conjugate(ZK)
        _c  = np.dot(ZKH, psd_inv_Z2[freq_ind, :, :])
        _cc = np.inner(_c, ZK)
        _cc /= (num_sensors - 1)

        cc[frame_ind] = alfa * cc[frame_ind - 1] + (1-alfa)* np.real(_cc)

    t  =np.array(range(0, frames))*256.0/16000.0
    freq = freq_ind * 16000 / 512
    plt.plot(t, cc, label="{} HZ".format(freq))
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

