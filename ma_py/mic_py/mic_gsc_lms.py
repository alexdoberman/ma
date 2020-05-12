# -*- coding: utf-8 -*-
import numpy as np
import copy
from mic_py.mic_blocking_matrix import calc_blocking_matrix_from_steering


def gsc_lms_filter(stft_arr, d_arr, alfa = 0.05):

    """
    GSC LMS filter

    :stft_arr: - spectr for each sensors - shape (bins, num_sensors, frames)
    :d_arr:    - steering vector         - shape (bins, num_sensors)
    :alfa:     - smooth factor

    :return:
        ds_spec - result spectral  - shape (bins, frames)
    """


    n_bins, n_sensors, n_frames = stft_arr.shape

    _wq = np.transpose(d_arr, (1, 0)) / n_sensors
    B = calc_blocking_matrix_from_steering(d_arr.T)

    if (n_bins != d_arr.shape[0] or n_sensors != d_arr.shape[1]):
        raise ValueError('gsc_lms_filter: error d_arr.shape = {}'.format(d_arr.shape))

    output = np.zeros((n_bins, n_frames), dtype=np.complex)

    gamma = 0.04
    betta =  0.99
    _alpha2 = 0.005

    for freq_ind in range(0, n_bins):


        waKH = np.zeros((n_sensors - 1), dtype=np.complex)
        wq = _wq[:, freq_ind]
        wqH = np.conjugate(wq)

        sigma2X = 1.0
        for frame_ind in range(0, n_frames):

            XK  = stft_arr[freq_ind, :, frame_ind]

            sigma2X = betta * sigma2X + (1 - betta)* np.inner(np.conjugate(XK),XK).real
            alfa = gamma / (sigma2X)
            #print ("    frame = {} alfa = {}".format(frame_ind, alfa))

            YcK = np.dot(wqH, XK)
            ZKH = np.dot(np.conjugate(B[:, :, freq_ind]).T, XK)

            # Calc error
            EK = YcK - np.dot(waKH, ZKH)

            # Update waKH
            waKH = waKH + alfa*np.dot(ZKH, EK)

            # Calc output
            Y = YcK - np.dot(waKH, ZKH)

            output[freq_ind, frame_ind] = EK

            watK2 = np.dot(np.conjugate(waKH), waKH).real
            if (watK2 > _alpha2):
                waKH = (waKH/np.sqrt(watK2))*_alpha2


            # Dump debugging info.
            watK2 = np.dot(np.conjugate(waKH), waKH).real
            if freq_ind == 50 and frame_ind % 50 == 0:
                print('')
                print('Sample %d' % (frame_ind))
                print('waK^2        = %8.4e' % (watK2))

    return output


