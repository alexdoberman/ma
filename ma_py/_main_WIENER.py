# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_wiener import wiener_filter
from mic_py.mic_gsc import *

import matplotlib.pyplot as plt


if __name__ == '__main__':

    #################################################################
    # 1.0 - _wav_wbn45_dict0 PROFILE

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 69
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (4.03158, 9.03258)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_buble_n2/'
    out_wav_path   = r'./data/out/'
    #################################################################

    # #################################################################
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
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # in_wav_path    = r'./data/_wav_wbn45_dict0/'
    # out_wav_path   = r'./data/out/'
    # #################################################################

    # # #################################################################
    # # 1.0 - _mus1+spk2_snr_-10_geom3 PROFILE MVDR
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 3*60
    # n_fft          = 512
    #
    # (angle_hor_log, angle_vert_log) = (-20.56, -6.42)
    # (angle_inf_hor_log, angle_inf_vert_log) = (-14.93, 3.18)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # angle_inf_h = -angle_inf_hor_log
    # angle_inf_v = -angle_inf_vert_log
    #
    # in_wav_path    = r'./data/_mus1+spk2_snr_-10_geom3/mix/'
    # out_wav_path   = r'./data/out/'
    #
    # _noise_start = 0
    # _noise_end     = 64
    #
    # _mix_start  = 0
    # _mix_end       = 64
    #
    # #################################################################

    # #################################################################
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
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # in_wav_path    = r'./data/_wav_wbn45_dict0/'
    # out_wav_path   = r'./data/out/'
    # ################################################################

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec=max_len_sec)
    # x_all_arr     = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

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
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position          = get_source_position(angle_h, angle_v, radius=6.0)
    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)

    #################################################################
    # 4 - Calc DS filter output
    result_spec = wiener_filter(stft_all, d_arr.T)

    #################################################################
    # 5 inverse STFT and save
    sig_out = istft(result_spec.transpose((1, 0)), overlap=2)
    sf.write(r"out/out_mic_wiener.wav", sig_out, sr)

    wiener_from_mic, _ = sf.read('out/wiener_from_mic.wav')

    print(wiener_from_mic.shape)
    print(sig_out.shape)
