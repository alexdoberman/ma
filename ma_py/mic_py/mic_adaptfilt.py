# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


eps = 0.000001
def compensate_ref_ch_filter(stft_main, stft_ref, alfa = .75):

    """
    ADAPTIVE FREQUENCY COMPENSATOR

        Model:
        X1(f, t) = N(f, t)
        X2(f, t) = S(f, t) + N(f, t)W(f, t)

        Result:
        E(f, t) = X2(f, t) – H(f, t) X1(f, t)

        Algorithm:
        Ho(f, t) = <X1 (f, t) X2* (f, t)>/<|X1(f, t)|^2> = P12(f, t)/P11(f, t)
        P12(f, t) = (1- alfa) P12(f, t -1) + alfa X1 (f, t) X2* (f, t)
        P11(f, t) = (1- alfa) P22(f, t -1) + alfa |X1(f, t)|^2

    :stft_main: - spectr  X1  - shape (bins, frames)
    :stft_ref:  - spectr  X2  - shape (bins, frames)  
    :alfa:      - smooth factor, range: 0 .. 1

    :return:  
        output - result compensate  - shape (bins, frames)  
    """

    (bins, frames) = stft_main.shape

    if stft_main.ndim != 2 or stft_ref.ndim != 2:
        raise ValueError('compensate_ref_ch_filter: error stft_main.ndim = {} stft_ref.ndim = {}'.format(stft_main.ndim, stft_ref.ndim))

    if (bins != stft_ref.shape[0] or frames != stft_ref.shape[1]):
        raise ValueError('compensate_ref_ch_filter: error stft_ref.shape = {}'.format(stft_ref.shape))

    output  = np.zeros((bins, frames), dtype=np.complex) 

    P_MR = np.zeros((bins, frames), np.complex)
    P_RR = np.zeros((bins, frames), np.float)
    #TODO proper init

    for frame_ind in range(0, frames):
        for freq_ind in range(0, bins):
#            P_MR[freq_ind, frame_ind ] = alfa * P_MR[freq_ind, frame_ind - 1] + (1.0 - alfa) * stft_main[freq_ind, frame_ind] * np.conjugate(stft_ref[freq_ind, frame_ind])
#            P_RR[freq_ind, frame_ind ] = alfa * P_RR[freq_ind, frame_ind - 1] + (1.0 - alfa) * np.real(stft_ref[freq_ind, frame_ind] * np.conjugate(stft_ref[freq_ind, frame_ind]))

            P_MR[freq_ind, frame_ind ] = alfa * P_MR[freq_ind, frame_ind - 1] + (1.0 - alfa) * stft_ref[freq_ind, frame_ind] * np.conjugate(stft_main[freq_ind, frame_ind])
            P_RR[freq_ind, frame_ind ] = alfa * P_RR[freq_ind, frame_ind - 1] + (1.0 - alfa) * np.real(stft_ref[freq_ind, frame_ind] * np.conjugate(stft_ref[freq_ind, frame_ind]))

        # # Dump debugging info.
        # if  frame_ind % 50 == 0:
        #     print ('')
        #     print ('Sample %d' %(frame_ind))
        #     abs_P_MR = abs(P_MR[:, frame_ind ])
        #     abs_P_RR = abs(P_RR[:, frame_ind ])
        #     abs_H    = abs_P_MR / (abs_P_RR + 0.0001)
        #     print ('||abs_H||       = {}'.format(abs_H))


    H = P_MR / (P_RR + eps)

    result = stft_main - H * stft_ref 
    return result

 


def spectral_substract_filter(stft_main, stft_ref, alfa_PX = 0.01, alfa_PN = 0.99):

    """
    spectral subtraction filter

    :stft_main: - spectr  main signal  - shape (bins, frames)
    :stft_ref:  - spectr  ref signal   - shape (bins, frames)  
    :alfa_PX:   - smooth factor, range: 0 .. 1
    :alfa_PN:   - smooth factor, range: 0 .. 1

    :return:  
        output - spectral subtraction compensate  - shape (bins, frames)  
    """

    X_mag     = np.absolute(stft_main)
    N_mag     = np.absolute(stft_ref)

    PX = X_mag**2
    PN = N_mag**2

    def exp_average(X, Alpha):
        nLen = X.shape[0]

        Y = np.zeros(X.shape)
        for i in range(0, nLen - 1, 1):
            Y[i+1,:] = Alpha*Y[i,:] + (1-Alpha)*X[i+1]
        return Y

    PX = exp_average(PX, alfa_PX)    # 0   .. 0.5
    PN = exp_average(PN, alfa_PN)    # 0.5 .. 1

    # Wiener filter
    alfa  = 0.5
    beta  = 1.0
    gamma = 1.0
    Gain = np.maximum(1.0 - (PN/(PX + eps)*gamma)**beta, 0.001)**alfa

    result = stft_main*Gain
    return result


def spectral_substract_ref_psd_filter(stft_main, ref_psd, alfa_PX = 0.01, alfa_PN = 0.99):

    """
    spectral subtraction filter

    :stft_main: - spectr  main signal  - shape (bins, frames)
    :ref_psd:   - psd ref signal   - shape (bins, frames)
    :alfa_PX:   - smooth factor, range: 0 .. 1
    :alfa_PN:   - smooth factor, range: 0 .. 1

    :return:
        output - spectral subtraction compensate  - shape (bins, frames)
    """

    X_mag     = np.absolute(stft_main)

    PX = X_mag**2
    PN = ref_psd

    def exp_average(X, Alpha):
        nLen = X.shape[0]

        Y = np.zeros(X.shape)
        for i in range(0, nLen - 1, 1):
            Y[i+1,:] = Alpha*Y[i,:] + (1-Alpha)*X[i+1]
        return Y

    PX = exp_average(PX, alfa_PX)    # 0   .. 0.5
    PN = exp_average(PN, alfa_PN)    # 0.5 .. 1

    # Wiener filter
    alfa  = 0.5
    beta  = 1.0
    gamma = 1.0
    Gain = np.maximum(1.0 - (PN/(PX + eps)*gamma)**beta, 0.01)**alfa

    result = stft_main*Gain
    return result


def smb_filter(stft_main, stft_ref, gain_max=18):
    """
       spectral subtraction filter

       :stft_main: - spectr  main signal  - shape (bins, frames)
       :stft_ref:  - spectr  ref signal   - shape (bins, frames)
       :gain_max:  -

       :return:
           output - spectral subtraction compensate  - shape (bins, frames)
    """
    bins, frames = stft_main.shape
    W = np.ones(shape=(bins, frames))

    p_main = np.absolute(stft_main)
    p_ref = np.absolute(stft_ref)

    # alpha
    release_coeff = 1 - 0.033
    # beta
    attack_coeff = 1 + 0.033

    # W_max = 0.1
    stft_new = np.zeros(shape=stft_main.shape, dtype=np.complex)

    c = 1

    gain_min = 10 ** (-0.05*gain_max)
    for i in range(1, frames):

        w_buffer = np.ones(bins)
        ref_power = 0
        for j in range(0, bins):
            ref_power += p_ref[j, i]**2
            n_current = W[j, i-1]*p_ref[j, i]

            if p_main[j, i] > n_current:
                w_buffer[j] = attack_coeff*W[j, i-1]
            elif p_main[j, i] < n_current * release_coeff:
                w_buffer[j] = release_coeff*W[j, i-1]

            # W[j, i] = min(W_max, W[j, i])
            p_s = p_main[j, i] - n_current
            snr = p_s/(n_current + 1e-6)
            gain = c * snr**2
            GAIN = min(1, max(gain_min, gain))
            stft_new[j, i] = stft_main[j, i]*GAIN

        if ref_power > 100:
            W[:, i] = w_buffer

    return stft_new