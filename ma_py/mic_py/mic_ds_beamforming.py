# -*- coding: utf-8 -*-
import numpy as np

def ds_beamforming(stft_arr, d_arr):
    """
    Delay- summ beamformng

    :stft_arr: - spectr for each sensors - shape (bins, sensors, frames)
    :d_arr:    - steering vector         - shape (bins, sensors)  
    :return:  
        delay summ beamforming result    - shape (bins, frames)  
    """

    '''
    (bins, sensors, frames) = stft_arr.shape   
    steering = np.expand_dims(d_arr, axis=2)
    prod     = stft_arr*steering
    result_spec = prod.sum(axis=1)/sensors
    return result_spec 
    '''                                                        
    (bins, sensors, frames) = stft_arr.shape       
    return np.einsum('...a,...at->...t', d_arr.conj(), stft_arr)/sensors

def ds_align(stft_arr, d_arr):
    """
    Align signal

    :stft_arr: - spectr for each sensors - shape (bins, sensors, frames)
    :d_arr:    - steering vector         - shape (bins, sensors)  
    :return:  
        delay summ beamforming result    - shape (bins, sensors, frames)
    """

    steering = np.expand_dims(d_arr, axis=2)
    result   = stft_arr*steering.conj()
    return result 


