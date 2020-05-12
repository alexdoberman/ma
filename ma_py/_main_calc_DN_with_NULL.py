# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_gsc import *
from mic_py.mic_ds_beamforming import *
from mic_py.mic_lcmv import lcmv_filter_debug, lcmv_filter
from mic_py.mic_null import null_filter, null_filter_ex

import matplotlib.pyplot as plt


if __name__ == '__main__':

    #################################################################
    # 1.0 - _no_echo_dist_1_0_angle_60  PROFILE MVDR
    vert_mic_count = 1
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 60
    n_fft          = 512

#    in_wav_path    = r'./data/_wav_wbn45_dict0/'
    in_wav_path    = r'./data/_wav_wbn45/'
#    in_wav_path    = r'./data/_WBN_0_0_66/'
#    in_wav_path    = r'./data/_wgn_30_wgn_-30/'
#    in_wav_path    =  r'./data/_gor_no_echo_dist_3_3_1min/'
#    in_wav_path  =  r'./data/_spk_0_du_hast_30/'

    out_wav_path   = r'./data/out/'

    _noise_start   = 0
    _noise_end     = 20

    _mix_start     = 0
    _mix_end       = 20
    #################################################################

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    x_all_arr     = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

    x_noise_arr = x_all_arr[:,(np.int32)(_noise_start*sr):(np.int32)(_noise_end*sr)]
    x_mix_arr   = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

    print ("Array data read done!")
    print ("    x_noise_arr.shape  = ", x_noise_arr.shape)
    print ("    x_mix_arr.shape    = ", x_mix_arr.shape)


    #################################################################
    # 2 - Do STFT
    stft_noise_arr =  stft_arr(x_noise_arr, fftsize = n_fft)
    stft_mix_arr   =  stft_arr(x_mix_arr, fftsize = n_fft)

    (n_bins, n_sensors, n_frames) = stft_mix_arr.shape

    print ("STFT calc done!")
    print ("    n_bins               = ", n_bins)
    print ("    n_sensors            = ", n_sensors)
    print ("    stft_noise_arr.shape = ", stft_noise_arr.shape)
    print ("    stft_mix_arr.shape   = ", stft_mix_arr.shape)


    #################################################################
    # 3.0 - Calc  steering vector for null direction

    angle_null_h = -30.0
    angle_null_v = -0.0
    sensor_positions     = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position_null = get_source_position(angle_null_h, angle_null_v, radius=6.0)
    source_position_null_d1 = get_source_position(angle_null_h + 3.0, angle_null_v, radius=6.0)
    source_position_null_d2 = get_source_position(angle_null_h - 3.0, angle_null_v, radius=6.0)

    d_arr_null           = propagation_vector_free_field(sensor_positions, source_position_null, N_fft = n_fft, F_s = sr)
    d_arr_null_d1        = propagation_vector_free_field(sensor_positions, source_position_null_d1, N_fft = n_fft, F_s = sr)
    d_arr_null_d2        = propagation_vector_free_field(sensor_positions, source_position_null_d2, N_fft = n_fft, F_s = sr)

    #################################################################
    # 3.1 - Calc  steering vector

    angle_step = 1

    print ("Begin steering calc ...")
    arr_angle_h = range(-90, 90, angle_step)
    angle_v     = 0.0
    arr_d_arr   = np.zeros((len(arr_angle_h), n_sensors, n_bins), dtype=np.complex)

    print ('arr_d_arr = ' , arr_d_arr.shape)

    for i , angle_h in enumerate (arr_angle_h):
        source_position    = get_source_position(angle_h, angle_v)
        arr_d_arr[i,:,:] = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    print ("Steering calc done!")


    #################################################################
    # 4 - Calc  map
    POW   = np.zeros(    (len(arr_angle_h))      )

    print ("Begin calc map ...")

    for i , angle_h in enumerate (arr_angle_h):
        print ("    process angle =  {}".format(angle_h))
        d_arr = arr_d_arr[i,:,:]

        # GSC beamforming
        #result_spec = gsc_filter(stft_arr , d_arr.T)

        # DS beamforming
        result_spec = ds_beamforming(stft_mix_arr, d_arr.T)

        # LCVM debug beamforming
        #result_spec = lcmv_filter_debug(stft_arr, d_arr.T, d_arr_null.T)

        # NULL beamforming
        #result_spec, _ = null_filter(stft_mix_arr,  d_arr_sp=d_arr.T, d_arr_inf=d_arr_null.T)

        # broad NULL 1 beamforming
        #result_spec, _ = null_filter_ex(stft_mix_arr, d_arr_sp=d_arr.T,
        #                                lst_d_arr_inf=[d_arr_null.T, d_arr_null_d1.T, d_arr_null_d2.T])

        # # LCVM beamforming
        # result_spec, _ = lcmv_filter(stft_mix_arr, stft_noise_arr, d_arr_sp=d_arr.T, d_arr_inf=d_arr_null.T)


        POW[i]      = np.real(np.sum(result_spec *np.conjugate(result_spec)) / n_frames)
    print ("Calc map done!")

    #################################################################
    # 5 - Scale to power ch_0_0
    P0    = np.sum(stft_mix_arr[:,0,:] *np.conjugate(stft_mix_arr[:,0,:])) / n_frames # power ch_0_0
    POW = POW/P0
    #POW = 10*np.log10(POW/P0)

    #################################################################
    # 6 - Plot DN
    plt.plot(arr_angle_h, POW)
    plt.xlabel('angle_h (s)')
    plt.ylabel('pow_res/pow_s0')
    plt.title('DS alg')
    plt.grid(True)
    plt.savefig(r".\out\DS_LVCM_NULL.png")
    plt.show()






