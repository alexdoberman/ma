# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_lcmv import lcmv_filter
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
import matplotlib.pyplot as plt
from mic_py.mic_null import null_filter



if __name__ == '__main__':

    # ################################################################
    # # 1.0 - _wav_wbn45_dict0 PROFILE
    #
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 45
    # n_fft          = 512
    #
    # (angle_hor_log, angle_vert_log) = (0.0, 0.0)
    # (angle_inf_hor_log, angle_inf_vert_log) = (45, 0)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # angle_inf_h = -angle_inf_hor_log
    # angle_inf_v = -angle_inf_vert_log
    #
    # in_wav_path    = r'./data/_wav_wbn45_dict0/'
    # out_wav_path   = r'./data/out/'
    #
    # _noise_start   = 0
    # _noise_end     = 25
    #
    # _mix_start     = 0
    # _mix_end       = 25
    # #################################################################



    #################################################################
    # 1.0 - _rameses PROFILE

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 45
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (13.9677, 5.65098)
    (angle_inf_hor_log, angle_inf_vert_log) = (-15.05, -0.31)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    angle_inf_h = -angle_inf_hor_log
    angle_inf_v = -angle_inf_vert_log

    in_wav_path = r'./data/_rameses/'
    out_wav_path   = r'./data/out/'

    _noise_start   = 26
    _noise_end     = 60

    _mix_start     = 26
    _mix_end       = 60
    #################################################################



    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    x_noise_arr = x_all_arr[:,(np.int32)(_noise_start*sr):(np.int32)(_noise_end*sr)]
    x_mix_arr   = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

    print ("Array data read done!")
    print ("    x_noise_arr.shape  = ", x_noise_arr.shape)
    print ("    x_mix_arr.shape    = ", x_mix_arr.shape)

    #################################################################
    # 2 - Do STFT
    stft_noise_arr =  stft_arr(x_noise_arr, fftsize = n_fft)
    stft_mix_arr   =  stft_arr(x_mix_arr, fftsize = n_fft)

    (n_bins, n_sensors, n_frames) = stft_noise_arr.shape

    print ("STFT calc done!")
    print ("    n_bins               = ", n_bins)
    print ("    n_sensors            = ", n_sensors)
    print ("    stft_noise_arr.shape = ", stft_noise_arr.shape)
    print ("    stft_mix_arr.shape   = ", stft_mix_arr.shape)


    #################################################################
    # 3 - Calc  steering vector
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    print ('    (angle_inf_h, angle_inf_v) = ', angle_inf_h, angle_inf_v)
    sensor_positions    = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position     = get_source_position(angle_h, angle_v, radius=6.0)
    source_position_inf = get_source_position(angle_inf_h, angle_inf_v, radius=6.0)

    d_arr               = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    d_arr_inf           = propagation_vector_free_field(sensor_positions, source_position_inf, N_fft = n_fft, F_s = sr)

    #################################################################
    # 4 - LCMV filter output

    #result_spec, _  = null_filter(stft_mix_arr, d_arr_sp=d_arr.T, d_arr_inf=d_arr_inf.T)
    result_spec, lcmv_weights  = lcmv_filter(stft_mix_arr, stft_noise_arr, d_arr_sp=d_arr.T, d_arr_inf=d_arr_inf.T)

    #################################################################
    # 5 inverse STFT and save
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/out_LCMV.wav", sig_out, sr)

    #################################################################
    # 6.1 - Do align save ds output sp
    align_stft_arr = ds_align(stft_mix_arr, d_arr.T)
    result_spec    = align_stft_arr.sum(axis=1) / (hor_mic_count * vert_mic_count)
    sig_out = istft(result_spec.transpose((1, 0)), overlap=2)
    sf.write(r"out/ds_sp.wav", sig_out, sr)

    #################################################################
    # 6.2 - Do align save ds output inf
    align_stft_arr = ds_align(stft_mix_arr, d_arr_inf.T)
    result_spec = align_stft_arr.sum(axis=1) / (hor_mic_count * vert_mic_count)
    sig_out = istft(result_spec.transpose((1, 0)), overlap=2)
    sf.write(r"out/ds_inf.wav", sig_out, sr)

