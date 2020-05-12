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

    # #################################################################
    # # 1.0 - _sol PROFILE MVDR
    # vert_mic_count = 6
    # hor_mic_count = 11
    # dHor = 0.035
    # dVert = 0.05
    # max_len_sec = 3 * 60
    # n_fft          = 512
    # n_overlap      = 2
    #
    #
    # (angle_hor_log, angle_vert_log) = (13.8845, 6.60824)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # in_wav_path = r'./data/_sol/'
    # out_wav_path = r'./data/out/'
    #
    # _noise_start = 9
    # _noise_end = 28
    #
    # _mix_start = 28
    # _mix_end = 98
    #
    # _sp_start = 98
    # _sp_end = 112
    # #################################################################

    # #################################################################
    # 1.0 - chirp_mix PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512
    n_overlap      = 2

    (angle_hor_log, angle_vert_log) = (-50.7695, 35.7263)
    (angle_inf_hor_log, angle_inf_vert_log) = (-47.9698, -39.8491)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    angle_inf_h = -angle_inf_hor_log
    angle_inf_v = -angle_inf_vert_log


    in_wav_path    = r'./data/chirp_mix/wav_-10/'
    out_wav_path   = r'./data/out/'


    _chirp_noise_start  = 0
    _chirp_noise_end    = 11

    _chirp_sp_start     = 11
    _chirp_sp_end       = 23

    _noise_start   = 23
    _noise_end     = 38

    _mix_start     = 38
    _mix_end       = 69

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec=max_len_sec)
    (n_channels, n_samples) = x_all_arr.shape

    print("Array data read done!")
    print("    n_channels  = ", n_channels)
    print("    n_samples   = ", n_samples)
    print("    freq        = ", sr)

    x_noise_arr = x_all_arr[:, (np.int32)(_noise_start * sr):(np.int32)(_noise_end * sr)]
    x_mix_arr = x_all_arr[:, (np.int32)(_mix_start * sr):(np.int32)(_mix_end * sr)]

    x_chirp_noise_arr = x_all_arr[:, (np.int32)(_chirp_noise_start * sr):(np.int32)(_chirp_noise_end * sr)]
    x_chirp_sp_arr = x_all_arr[:, (np.int32)(_chirp_sp_start * sr):(np.int32)(_chirp_sp_end * sr)]

    #################################################################
    # 2 - Do STFT

    stft_noise_arr = stft_arr(x_noise_arr, fftsize=n_fft, overlap=n_overlap)
    stft_mix = stft_arr(x_mix_arr, fftsize=n_fft, overlap=n_overlap)

    stft_chirp_noise_arr = stft_arr(x_chirp_noise_arr, fftsize=n_fft, overlap=n_overlap)
    stft_chirp_sp_arr = stft_arr(x_chirp_sp_arr, fftsize=n_fft, overlap=n_overlap)

    (n_bins, n_sensors, n_frames) = stft_mix.shape

    print("STFT calc done!")
    print("    n_bins     = ", n_bins)
    print("    n_sensors  = ", n_sensors)
    print("    n_frames   = ", n_frames)

    #################################################################
    # 3 - Calc  steering vector
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    print ('    (angle_inf_h, angle_inf_v) = ', angle_inf_h, angle_inf_v)
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position          = get_source_position(angle_h, angle_v, radius = 6.0)
    source_position_inf      = get_source_position(angle_inf_h, angle_inf_v, radius=6.0)

    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    d_arr_inf                = propagation_vector_free_field(sensor_positions, source_position_inf, N_fft = n_fft, F_s = sr)

    #################################################################
    # 4 - Calc  RTF steering vector
    print('Calc  RTF steering vector!')
    #result_spec_rtf = rtf_filter(stft_mix, stft_chirp_sp_arr, 'simple')
    result_spec_rtf = rtf_filter(stft_mix, stft_chirp_noise_arr, 'simple')


    #################################################################
    # 5 - Do DS beamforming
    result_sp    = ds_beamforming(stft_mix, d_arr.T)
    result_noise = ds_beamforming(stft_mix, d_arr_inf.T)


    #################################################################
    # 5 - Write result

    sig_out = istft(result_spec_rtf.transpose((1, 0)), overlap=n_overlap)
    sf.write(r"out/out_DS_rtf.wav", sig_out, sr)

    sig_out = istft(result_sp.transpose((1, 0)), overlap=n_overlap)
    sf.write(r"out/out_DS_sp.wav", sig_out, sr)

    sig_out = istft(result_noise.transpose((1, 0)), overlap=n_overlap)
    sf.write(r"out/out_DS_noise.wav", sig_out, sr)
