# -*- coding: utf-8 -*-
import numpy as np


def zelin_noise_coherence(bins, sensors_count):
    '''
    Calc zelin noise_coherence function
    :bins:           - fft bins count
    :sensors_count:  

    :return:  
        G_IJ - shape (sensors_count, sensors_count, bins)
    '''

    G_IJ            = np.zeros((sensors_count, sensors_count, bins)) 
    for i in range(sensors_count):
        G_IJ[i,i,:] = 1

    return G_IJ


def diffuse_noise_coherence(bins, freq, D_IJ):
    '''
    Calc noise_coherence function
    :bins:  - fft bins count
    :freq:  - freq Hz
    :D_IJ:  - pair distance - shape (sensors_count, sensors_count)

    :return:  
        G_IJ - shape (sensors_count, sensors_count, bins)
    '''

    sensors_count,sensors_count  = D_IJ.shape
    freqs           = np.linspace(0.0, freq/2, num=bins)
    c               = 343
    G_IJ            = np.zeros((sensors_count, sensors_count, bins)) 

    for i in range(sensors_count):
        for j in range(sensors_count):
            G_IJ[i,j] = np.sinc(2*freqs*D_IJ[i,j]/c)
    return G_IJ

def localized_noise_coherence(steering_vector_noise_direction):

    '''
    Calc noise_coherence function for localized source
    :steering_vector_noise_direction:  - shape (num_sensors, bins)

    :return:  
        G_IJ - shape (sensors_count, sensors_count, bins)
    '''

    (sensors_count, bins) = steering_vector_noise_direction.shape
    G_IJ                  = np.zeros((sensors_count, sensors_count, bins), dtype=np.complex) 

    for i in range(sensors_count):
        for j in range(sensors_count):
                D_i = steering_vector_noise_direction[i,:]
                D_j = steering_vector_noise_direction[j,:]
                G_IJ[i,j] = D_i*D_j.conj() 
    return G_IJ


def localized_noise_coherence2(steering_vector_noise_direction):

    '''
    Calc noise_coherence function for localized source
    :steering_vector_noise_direction:  - shape (num_sensors, bins)

    :return:  
        G_IJ - shape (sensors_count, sensors_count, bins)
    '''

    (sensors_count, bins) = steering_vector_noise_direction.shape
    G_IJ                  = np.zeros((sensors_count, sensors_count, bins), dtype=np.complex) 

    for k in range(bins):
        D = steering_vector_noise_direction[:,k]
        G_IJ[:,:,k] =  np.outer(D, D.conj())
    return G_IJ



def real_noise_coherence(stft_arr):

    '''
    Calc noise_coherence function for real noise signal
    :stft_arr:  - shape (bins, sensors_count, frames)

    :return:  
        G_IJ - shape (sensors_count, sensors_count, bins)
    '''

    print (stft_arr.shape)

    (bins, sensors_count, frames) = stft_arr.shape
    G_IJ                          = np.zeros((sensors_count, sensors_count, bins), dtype=np.complex) 

    tmp = np.zeros((sensors_count, sensors_count, bins, frames), dtype=np.complex)
    for k in range(0, frames):
        for i in range(sensors_count):
            for j in range(sensors_count):
                tmp[i,j,:,k] = stft_arr[:,i,k]*stft_arr[:,j,k].conj()

    C_xx = np.mean(tmp, axis = -1)

    for i in range(sensors_count):
        for j in range(sensors_count):
                G_IJ[i,j,:] = C_xx[i,j,:]/ np.sqrt(C_xx[i,i,:] * C_xx[j,j,:])
    return G_IJ







