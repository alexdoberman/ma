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
from mic_py.beamforming import apply_beamforming_vector, get_mvdr_vector, get_power_spectral_density_matrix

from mic_py.mic_lcmv_2 import lcmv_filter_delta, lcmv_filter_der_hor
from mic_py.mic_exp_steering import get_der_steering_2


import matplotlib.pyplot as plt


if __name__ == '__main__':

    #################################################################
    # 1.0 - _no_echo_dist_1_0_angle_60  PROFILE MVDR
    # vert_mic_count = 1
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 60
    n_fft          = 512

    in_wav_path    = r'./data/_wav_wbn45_dict0/'
#    in_wav_path    = r'./data/_wav_wbn45/'
#    in_wav_path    = r'./data/_WBN_0_0_66/'
#    in_wav_path    = r'./data/_wgn_30_wgn_-30/'
    # in_wav_path = r'./data/_wgn_0.0_0.0'

    out_wav_path   = r'./data/out/'

    _mix_start     = 0
    _mix_end       = 9
    #################################################################

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    x_all_arr     = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

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

    angle_step = 1

    print ("Begin steering calc ...")
    arr_angle_h = range(-90, 90, angle_step)
    angle_v     = 0.0
    arr_d_arr   = np.zeros((len(arr_angle_h), n_sensors, n_bins), dtype=np.complex)

    print ('arr_d_arr = ' , arr_d_arr.shape)

    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position = get_source_position(30, angle_v)
    d_arr_inf = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)

    d_arr_inf_h_der = get_der_steering_2(sensor_positions, source_position, 512, 16000, 30, angle_v)

    for i , angle_h in enumerate (arr_angle_h):
        sensor_positions   = get_sensor_positions(hor_mic_count, vert_mic_count, dHor = dHor, dVert = dVert)
        source_position    = get_source_position(angle_h, angle_v)
        arr_d_arr[i,:,:] = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    print ("Steering calc done!")

    #################################################################
    # 4 - Calc  map
    POW   = np.zeros(    (len(arr_angle_h))      )

    print ("Begin calc map ...")

    psd_noise_matrix = get_power_spectral_density_matrix(stft_arr)

    psd_noise_matrix = psd_noise_matrix + 0.01 * np.identity(psd_noise_matrix.shape[-1])

    #################################################################
    # 6 - Apply MVDR
    for i , angle_h in enumerate (arr_angle_h):
        print ("    process angle =  {}".format(angle_h))
        d_arr = arr_d_arr[i,:,:]

        # GSC beamforming
        # result_spec = gsc_filter(stft_arr , d_arr.T)

        # DS beamforming
        # result_spec = ds_beamforming(stft_arr, d_arr.T)

        w = get_mvdr_vector(d_arr.T, psd_noise_matrix)
        result_spec = apply_beamforming_vector(w, stft_arr)

        # if angle_h == 30:
        #    POW[i] = 0
        #    continue

        # result_spec, _ = lcmv_filter_delta(stft_mix=stft_arr, stft_noise=stft_arr, d_arr_sp=d_arr.T)

        # result_spec, _ = lcmv_filter_delta(stft_arr, stft_arr, d_arr_sp=d_arr.T, d_arr_inf=d_arr_inf.T)

        # result_spec, _ = lcmv_filter_der_hor(stft_mix=stft_arr, stft_noise=stft_arr, d_arr_sp=d_arr.T,
        #                                     d_arr_inf=d_arr_inf.T, d_arr_inf_der_hor=d_arr_inf_h_der.T)

        POW[i]      = np.real(np.sum(result_spec *np.conjugate(result_spec)) / n_frames)

    print ("Calc map done!")

    #################################################################
    # 5 - Scale to power ch_0_0
    P0    = np.sum(stft_arr[:,0,:] *np.conjugate(stft_arr[:,0,:])) / n_frames # power ch_0_0
    # POW = POW/P0

    POW = 10*np.log10(POW/P0)

    #################################################################
    # 6 - Plot DN
    plt.plot(arr_angle_h, POW)
    plt.xlabel('angle_h (s)')
    plt.ylabel('pow_res/pow_s0')
    plt.title('DS alg')
    plt.grid(True)
    plt.savefig(r".\out\DS.png")
    plt.show()

