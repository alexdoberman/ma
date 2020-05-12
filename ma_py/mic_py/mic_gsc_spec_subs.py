# -*- coding: utf-8 -*-
import numpy as np
import copy
from mic_py.mic_adaptfilt import *
from mic_py.mic_blocking_matrix import calc_blocking_matrix_from_steering


def gsc_filter_spec_subs_all(stft_arr, d_arr):

    """
    GSC filter

    :stft_arr: - spectr for each sensors - shape (bins, num_sensors, frames)
    :d_arr:    - steering vector         - shape (bins, num_sensors)  


    :return:  
        ds_spec - result spectral  - shape (bins, frames)  
    """

    _sigma0 = 1.0
    _mu     = 0.9999

    bins, num_sensors, frames =  stft_arr.shape

    if (bins != d_arr.shape[0] or num_sensors != d_arr.shape[1]):
        raise ValueError('gsc_filter: error d_arr.shape = {}'.format(d_arr.shape))

    output  = np.zeros((bins, frames), dtype=np.complex) 

    # Calc blocking  matrix, shape= (num_sensors, num_sensors - num_constrain, bins)
    B = calc_blocking_matrix_from_steering(d_arr.T)

    print ("B.shape = ", B.shape)
    print ("bins, num_sensors, frames = " , bins, num_sensors, frames)

    num_constrain = 1
    noise_branch  = np.zeros((bins, frames, num_sensors - num_constrain), dtype=np.complex)
    signal_branch = np.zeros((bins, frames), dtype=np.complex)

    for frame_ind in range(0, frames):

        sigmaK = abs(np.dot(np.conjugate(stft_arr[:,0,frame_ind]), stft_arr[:,0,frame_ind]))
        for freq_ind in range(0, bins):

            # Get  output of blocking matrix.
            XK = stft_arr[freq_ind, : ,frame_ind]
            ZK = np.dot(np.conjugate(B[:,:,freq_ind]).T, XK)

            # Get output of upper branch.
            wqH = np.conjugate(d_arr[freq_ind, :])
            YcK = np.dot(wqH, XK)/num_sensors

            # Dump debugging info.
            if freq_ind == 100 and frame_ind % 50 == 0:
                print ('')
                print ('Sample %d' %(frame_ind))
                print ('SigmaK          = %8.4e' %(sigmaK))
                print ('ZK.shape        = {}'.format(ZK.shape))
                print ('YcK.shape       = {}'.format(YcK.shape))

            noise_branch[freq_ind, frame_ind, :]  = ZK
            signal_branch[freq_ind, frame_ind]    = YcK  

    for i in range(num_sensors - num_constrain):
        output +=  spectral_substract_filter(stft_main= signal_branch , stft_ref= noise_branch[:,:,i], alfa_PX = 0.01, alfa_PN = 0.99)

    output /= (num_sensors - num_constrain)

    return output



def gsc_filter_spec_subs_only_one(stft_arr, d_arr):

    """
    GSC filter

    :stft_arr: - spectr for each sensors - shape (bins, num_sensors, frames)
    :d_arr:    - steering vector         - shape (bins, num_sensors)  


    :return:  
        ds_spec - result spectral  - shape (bins, frames)  
    """

    _sigma0 = 1.0
    _mu     = 0.9999

    bins, num_sensors, frames =  stft_arr.shape

    if (bins != d_arr.shape[0] or num_sensors != d_arr.shape[1]):
        raise ValueError('gsc_filter: error d_arr.shape = {}'.format(d_arr.shape))

    output  = np.zeros((bins, frames), dtype=np.complex) 

    # Calc blocking  matrix, shape= (num_sensors, num_sensors - num_constrain, bins)
    B = calc_blocking_matrix_from_steering(d_arr.T)

    print ("B.shape = ", B.shape)
    print ("bins, num_sensors, frames = " , bins, num_sensors, frames)

    noise_branch  = np.zeros((bins, frames), dtype=np.complex)
    signal_branch = np.zeros((bins, frames), dtype=np.complex)

    for frame_ind in range(0, frames):

        sigmaK = abs(np.dot(np.conjugate(stft_arr[:,0,frame_ind]), stft_arr[:,0,frame_ind]))
        for freq_ind in range(0, bins):

            # Get  output of blocking matrix.
            XK = stft_arr[freq_ind, : ,frame_ind]
            ZK = np.dot(np.conjugate(B[:,:,freq_ind]).T, XK)

            # Get output of upper branch.
            wqH = np.conjugate(d_arr[freq_ind, :])
            YcK = np.dot(wqH, XK)/num_sensors

            # Dump debugging info.
            if freq_ind == 100 and frame_ind % 50 == 0:
                print ('')
                print ('Sample %d' %(frame_ind))
                print ('SigmaK          = %8.4e' %(sigmaK))
                print ('ZK.shape        = {}'.format(ZK.shape))
                print ('YcK.shape       = {}'.format(YcK.shape))

            # take only first zero component
            noise_branch[freq_ind, frame_ind]  = ZK[0]   
            signal_branch[freq_ind, frame_ind] = YcK  

    output =  spectral_substract_filter(stft_main= signal_branch , stft_ref= noise_branch, alfa_PX = 0.01, alfa_PN = 0.99)

    return output


        




    






