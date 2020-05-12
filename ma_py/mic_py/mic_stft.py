# -*- coding: utf-8 -*-
import numpy as np
from mic_py.feats import stft, istft


def stft_arr(x_arr, fftsize, overlap = 2):
    """
    STFT for each microphone

    :x_arr: data from microphone array. shape (sensors, samples) 
    :fftsize: 

    :return:  
        shape (bins, sensors, frames)
    """
    n_channels = x_arr.shape[0]
    n_samples  = x_arr.shape[1]

    stft_arr = []
    for ch in range(n_channels):
        ch_spec  = stft(x_arr[ch,:], fftsize = fftsize, overlap = overlap)
        stft_arr.append(ch_spec)
    return  np.stack(stft_arr).transpose((2, 0, 1))
