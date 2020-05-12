# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_zelin import cross_spectral, calc_beta, zelin_filter


import matplotlib.pyplot as plt


if __name__ == '__main__':

    #sp  [2017-11-23 19:59:26] [0x00002404] :  [INF] MicGridProcessor::SetDirectionAngles, angleFiHorz = -30.9141,  angleFiVert = 12.483
    #################################################################
    # 1.0 - _nastya_alex PROFILE

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 45
    n_fft          = 256

    (angle_hor_log, angle_vert_log) = (-30.9141, 12.483)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_nastya_alex/'
    out_wav_path   = r'./data/out/'
    #################################################################

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    #################################################################
    # 2 - Do STFT
    stft_arr =  stft_arr(x_all_arr, fftsize = n_fft)
    (n_bins, n_sensors, n_frames) = stft_arr.shape

    print ("STFT calc done!")
    print ("    n_bins     = ", n_bins)
    print ("    n_sensors  = ", n_sensors)
    print ("    n_frames   = ", n_frames)

    #################################################################
    # 3 - Calc  steering vector

    angle_step = 2
    frame_step = 2

    print ("Begin steering calc ...")
    arr_angle_h = range(-90, 90, angle_step)
    arr_angle_v = range(-90, 90, angle_step)
    arr_d_arr   = np.zeros((len(arr_angle_h), len(arr_angle_v), n_sensors, n_bins), dtype=np.complex)

    print ('arr_d_arr = ' , arr_d_arr.shape)

    for i , angle_h in enumerate (arr_angle_h):
        for j , angle_v in enumerate (arr_angle_v):
            sensor_positions   = get_sensor_positions(hor_mic_count, vert_mic_count, dHor = dHor, dVert = dVert)
            source_position    = get_source_position(angle_h, angle_v)
            arr_d_arr[i,j,:,:] = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    print ("Steering calc done!")


    #################################################################
    # 3 - Calc  map
    POW   = np.zeros(    (len(arr_angle_h), len(arr_angle_v))      )

    print ("Begin calc map ...")
    col = 0 
    for k in range(0, n_frames, frame_step):
        print ("Process frame {} .. {},  max {}".format(k, k+frame_step, n_frames))

        for i , angle_h in enumerate (arr_angle_h):
            for j , angle_v in enumerate (arr_angle_v):
                d_arr = arr_d_arr[i,j,:,:]
                # 4 - Do beamforming
                inst_spec = ds_beamforming(stft_arr[:,:,k:k+frame_step], d_arr.T)
                inst_pow  = np.sum(inst_spec ** 2) / frame_step
                POW[i,j] = inst_pow

        # Display matrix
        plt.matshow(POW.T)
        plt.savefig('./out/png/mic_{}.png'.format(col))
        plt.close()
        col+=1
    print ("Calc map done!")

    #################################################################
    # 4 - Do align 
    #align_stft_arr = ds_align(stft_arr, d_arr.T)





