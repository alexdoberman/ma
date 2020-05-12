# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import math

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_gsc import *
from mic_py.mic_chebyshev import chebyshev_weights
from mic_py.mic_chebyshev import get_chebyshev_weights_for_amplitudes

import matplotlib.pyplot as plt



if __name__ == '__main__':

    #################################################################
    # 1.0 - _sol PROFILE MVDR
    vert_mic_count = 1
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512


    (angle_hor_log, angle_vert_log) = (13.9677, 5.65098)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    # in_wav_path    = r'./data/_5/'
    # in_wav_path = r'./data/_wav_wbn45_dict0/'
    #in_wav_path = r'./data/_du_hast/'
    #in_wav_path = r'./data/_sol/'
    in_wav_path = r'./data/_rameses/'
    out_wav_path = r'./data/out/'

    _mix_start = 26
    _mix_end = 46

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
    stft_all =  stft_arr(x_all_arr, fftsize = n_fft)
    (n_bins, n_sensors, n_frames) = stft_all.shape

    print ("STFT calc done!")
    print ("    n_bins     = ", n_bins)
    print ("    n_sensors  = ", n_sensors)
    print ("    n_frames   = ", n_frames)

    #################################################################
    # 3 - Calc Chebyshev weights

    weights = get_chebyshev_weights_for_amplitudes(vert_mic_count, hor_mic_count, n_bins)

    # 4 - Calc  steering vector with Chebyshev weights
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position          = get_source_position(angle_h, angle_v, radius = 6.0)
    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    d_arr_cheb = d_arr * weights

    #################################################################
    # 4 - Calc DS filter output
    result_spec_cheb = ds_beamforming(stft_all, d_arr_cheb.T)

    np.save('Chebyshev', result_spec_cheb)

    #################################################################
    # 5 inverse STFT and save
    sig_out = istft(result_spec_cheb.transpose((1,0)), overlap = 2)
    sf.write(r"out/out_rameses_cheb.wav", sig_out, sr)


