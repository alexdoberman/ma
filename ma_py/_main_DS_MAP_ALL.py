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

    #################################################################
    # 1.0 - _wav_wbn45_dict0 PROFILE

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 45
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (0.0, 0.0)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_room_simul/'
    out_wav_path   = r'./data/out/'
    #################################################################

    _mix_start     = 10
    _mix_end       = 40.0

    #################################################################

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
#    x_all_arr     = x_all_arr[:, (np.int32)(_mix_start * sr):(np.int32)(_mix_end * sr)]

    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    #################################################################
    # 2 - Do STFT
    stft_all =  stft_arr(x_all_arr, fftsize = n_fft)
    (n_bins, n_sensors, n_frames) = stft_all.shape

    print ("STFT calc done!")
    print ("    n_bins     = ", n_bins)
    print ("    n_sensors  = ", n_sensors)
    print ("    n_frames   = ", n_frames)

    #################################################################
    # 3 - Calc  steering vector

    angle_step = 1
    frame_step = 1

    print ("Begin steering calc ...")
    arr_angle_h = range(-70, 70, angle_step)
    arr_angle_v = range(-70, 70, angle_step)
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
    for i , angle_h in enumerate (arr_angle_h):
        for j , angle_v in enumerate (arr_angle_v):

            print ("    process angle_h = {}, angle_v = {}".format(angle_h, angle_v))

            d_arr = arr_d_arr[i,j,:,:]

            # DS beamforming
            result_spec = ds_beamforming(stft_all, d_arr.T)
            POW[i,j] += np.real(np.sum(result_spec *np.conjugate(result_spec)) / n_frames)

    print ("Calc map done!")
    np.save('pow', POW)

    # Display matrix
    plt.matshow(POW.T)
    plt.savefig('./out/ds_map.png')
    plt.show()
    plt.close()


    #################################################################
    # 4 - Do align 
    #align_stft_arr = ds_align(stft_arr, d_arr.T)





