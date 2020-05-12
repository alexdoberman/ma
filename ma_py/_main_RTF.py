# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_gsc import *
from mic_py.mic_gannot import rtf_filter


if __name__ == '__main__':

    """
    #################################################################
    # 1.0 - _du_hast PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count = 11
    dHor = 0.035
    dVert = 0.05
    max_len_sec = 3 * 60
    n_fft = 512

    (angle_hor_log, angle_vert_log) = (12.051, 5.88161)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path = r'./data/_du_hast/'
    out_wav_path = r'./data/out/'

    _noise_start = 8
    _noise_end = 17

    _mix_start = 17
    _mix_end = 84

    _sp_start = 84
    _sp_end = 102
    #################################################################
    """
    #################################################################
    # 1.0 - _sol PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count = 11
    dHor = 0.035
    dVert = 0.05
    max_len_sec = 3 * 60
    n_fft = 512

    (angle_hor_log, angle_vert_log) = (13.8845, 6.60824)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path = r'./data/_sol/'
    out_wav_path = r'./data/out/'

    _noise_start = 9
    _noise_end = 28

    _mix_start = 28
    _mix_end = 98

    _sp_start = 98
    _sp_end = 112
    #################################################################

    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec=max_len_sec)
    (n_channels, n_samples) = x_all_arr.shape

    print("Array data read done!")
    print("    n_channels  = ", n_channels)
    print("    n_samples   = ", n_samples)
    print("    freq        = ", sr)

    x_speech = x_all_arr[:, sr*_sp_start:sr*_sp_end]

    #################################################################
    # 2 - Do STFT
    stft_arr_ = stft_arr(x_all_arr, fftsize=n_fft)
    (n_bins, n_sensors, n_frames) = stft_arr_.shape

    print("STFT calc done!")
    print("    n_bins     = ", n_bins)
    print("    n_sensors  = ", n_sensors)
    print("    n_frames   = ", n_frames)

    stft_speech = stft_arr(x_speech, fftsize=n_fft)
    #################################################################
    # 3 - Calc  steering vector
    print('Calc  steering vector!')
    print('    (angle_h, angle_v) = ', angle_h, angle_v)

    result_spec_rtf = rtf_filter(stft_arr_, stft_speech, 'simple')

    sig_out = istft(result_spec_rtf.transpose((1, 0)), overlap=2)
    sf.write(r"out/out_rtf.wav", sig_out, sr)