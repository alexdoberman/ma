# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal as sg
from numpy import linalg as lg
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_sensor_positions_kinect
from mic_py.mic_steering import  propagation_vector_free_field

def compute_autocovariance(stft_arr):
    """
    Calculates power spectral density matrix.

    This does not yet work with more than one target mask.

    :param stft_arr: - Complex observations with shape (bins, sensors, frames)
    :return: PSD matrix with shape (bins, sensors, sensors)
    """

    bins, sensors, frames = stft_arr.shape
    psd                   = np.zeros((bins, sensors, sensors), dtype=np.complex) 

    for t in range(frames):
        for k in range(bins):
            D = stft_arr[k,:,t]
            psd[k,:,:] =  psd[k,:,:] + np.outer(D, D.conj())

    psd = psd / frames
    return psd

def pseudospectrum_MUSIC(stft_arr, L, arr_angle_h, arr_angle_v,
        vert_mic_count = 6,
        hor_mic_count  = 11,
        dHor           = 0.035,
        dVert          = 0.05,
        n_fft          = 512,
        sr             = 16000):

    """ This function compute the MUSIC pseudospectrum.

        :param stft_arr: - Complex observations with shape (bins, sensors, frames)
        :param L:  int. Number of components to be extracted.
        :param arr_angle_h:  - Range hor angles 
        :param arr_angle_v:  - Range vert angles 

        :returns: ndarray shape : (len(arr_angle_h), len(arr_angle_v), n_bins)
        
    """
    n_bins, n_sensors, n_frames = stft_arr.shape

    # Extract noise subspace shape: (bins, sensors, sensors)
    R = compute_autocovariance(stft_arr)

    lst_G = []
    for freq_ind in range(n_bins):
        U,S,V = lg.svd(R[freq_ind,:,:])
        G = U[:,L:]
        lst_G.append(G)


    arr_d_arr   = np.zeros((len(arr_angle_h), len(arr_angle_v), n_sensors, n_bins), dtype=np.complex)
    arr_cost    = np.zeros((len(arr_angle_h), len(arr_angle_v), n_bins), dtype=np.float)

    print ("Begin steering calc ...")
    for i , angle_h in enumerate (arr_angle_h):
        for j , angle_v in enumerate (arr_angle_v):
            sensor_positions   = get_sensor_positions(hor_mic_count, vert_mic_count, dHor = dHor, dVert = dVert)
            source_position    = get_source_position(angle_h, angle_v)
            arr_d_arr[i,j,:,:] = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    print ("Steering calc done!")


    #compute MUSIC pseudo spectrum
    for freq_ind in range(n_bins):
        for i , angle_h in enumerate (arr_angle_h):
            for j , angle_v in enumerate (arr_angle_v):

                G = lst_G[freq_ind]
                a = arr_d_arr[i,j,:,freq_ind]

                a = np.transpose(np.matrix(a))
                G = np.matrix(G)

                cost = 1./lg.norm((G.H)*a)
                arr_cost[i,j,freq_ind] = cost

    return arr_cost




def pseudospectrum_MUSIC_kinect(stft_arr, L, arr_angle_h, arr_angle_v,
        n_fft          = 512,
        sr             = 16000):

    """ This function compute the MUSIC pseudospectrum.

        :param stft_arr: - Complex observations with shape (bins, sensors, frames)
        :param L:  int. Number of components to be extracted.
        :param arr_angle_h:  - Range hor angles 
        :param arr_angle_v:  - Range vert angles 

        :returns: ndarray shape : (len(arr_angle_h), len(arr_angle_v), n_bins)
        
    """
    n_bins, n_sensors, n_frames = stft_arr.shape

    # Extract noise subspace shape: (bins, sensors, sensors)
    R = compute_autocovariance(stft_arr)

    lst_G = []
    for freq_ind in range(n_bins):
        U,S,V = lg.svd(R[freq_ind,:,:])
        G = U[:,L:]
        lst_G.append(G)


    arr_d_arr   = np.zeros((len(arr_angle_h), len(arr_angle_v), n_sensors, n_bins), dtype=np.complex)
    arr_cost    = np.zeros((len(arr_angle_h), len(arr_angle_v), n_bins), dtype=np.float)

    print ("Begin steering calc ...")
    for i , angle_h in enumerate (arr_angle_h):
        for j , angle_v in enumerate (arr_angle_v):
            sensor_positions   = get_sensor_positions_kinect()
            source_position    = get_source_position(angle_h, angle_v)
            arr_d_arr[i,j,:,:] = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    print ("Steering calc done!")


    #compute MUSIC pseudo spectrum
    for freq_ind in range(n_bins):
        for i , angle_h in enumerate (arr_angle_h):
            for j , angle_v in enumerate (arr_angle_v):

                G = lst_G[freq_ind]
                a = arr_d_arr[i,j,:,freq_ind]

                a = np.transpose(np.matrix(a))
                G = np.matrix(G)

                cost = 1./lg.norm((G.H)*a)
                arr_cost[i,j,freq_ind] = cost

    return arr_cost

















