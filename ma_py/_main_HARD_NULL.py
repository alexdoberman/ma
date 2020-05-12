# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_hard_null import hard_null_filter, hard_null_filter_time_domain
import matplotlib.pyplot as plt
from mic_py.mic_ds_beamforming import ds_beamforming
from mic_py.mic_adaptfilt import *


if __name__ == '__main__':


    # #################################################################
    # # 1.0 - _wav_wbn45_dict0 PROFILE
    #
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 45
    # n_fft          = 512
    # n_overlap = 2
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
    # _mix_start     = 0
    # _mix_end       = 20
    # #################################################################


    #################################################################
    # 1.0 - _sol PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512
    n_overlap      = 2

    (angle_hor_log, angle_vert_log)         = (13.8845, 6.60824)
    (angle_inf_hor_log, angle_inf_vert_log) = (-15.06, -0.31)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    angle_inf_h = -angle_inf_hor_log
    angle_inf_v = -angle_inf_vert_log


    in_wav_path    = r'./data/_sol/'
    out_wav_path   = r'./data/out/'

    _mix_start     = 28
    _mix_end       = 58
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
    # 3 - Calc  steering vector
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    print ('    (angle_inf_h, angle_inf_v) = ', angle_inf_h, angle_inf_v)
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position          = get_source_position(angle_h, angle_v, radius = 6.0)
    source_position_inf      = get_source_position(angle_inf_h, angle_inf_v, radius=6.0)

    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    d_arr_inf                = propagation_vector_free_field(sensor_positions, source_position_inf, N_fft = n_fft, F_s = sr)

    # time_domain_filter = True
    # if time_domain_filter:
    #     #################################################################
    #     # 4 - HARD NULL filter output
    #
    #     alg_type = 5  # 3-LMS,4-NML,5-AP
    #     sig_out = hard_null_filter_time_domain(stft_all, d_arr_sp=d_arr, d_arr_inf=d_arr_inf, alg_type=alg_type)
    #
    #     #################################################################
    #     # 5 - Save result
    #     sf.write(r"out/out_HARD_NULL_{}.wav".format(alg_type), sig_out, sr)
    # else:
    #
    #     #################################################################
    #     # 4 - HARD NULL filter output
    #     alg_type    = 2 #0-compensate ref, 1-spec subs, 2- SMB filter
    #     result_spec = hard_null_filter(stft_all, d_arr_sp=d_arr, d_arr_inf=d_arr_inf, alg_type=alg_type)
    #
    #     #################################################################
    #     # 5 - Inverse STFT and save
    #     sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    #     sf.write(r"out/out_HARD_NULL_{}.wav".format(alg_type), sig_out, sr)


    (n_bins, n_num_sensors, n_frames) = stft_all.shape

    spec_sp = ds_beamforming(stft_all, d_arr.T)
    spec_inf = ds_beamforming(stft_all, d_arr_inf.T)

    gain_correction = np.zeros((n_bins),dtype=np.complex)

    for i in range(n_bins):
        d1 = d_arr[:, i]
        d2 = d_arr_inf[:, i]
        eq_i = np.dot(np.conj(d2), d1)
        gain_correction[i] = eq_i/n_sensors

    spec_corr_inf = np.expand_dims(gain_correction, axis=1) * spec_inf

    #result_spec = spectral_substract_filter(stft_main=spec_sp, stft_ref=spec_inf, alfa_PX=0.01, alfa_PN=0.099)
    result_spec = spectral_substract_filter(stft_main=spec_sp, stft_ref=spec_corr_inf, alfa_PX=0.01, alfa_PN=0.099)
    #result_spec = smb_filter(stft_main=spec_sp, stft_ref=spec_corr_inf, gain_max=18)


    sig_sp = istft(spec_sp.transpose((1, 0)), overlap=n_overlap)
    sig_inf = istft(spec_inf.transpose((1, 0)), overlap=n_overlap)
    sig_corr_inf = istft(spec_corr_inf.transpose((1, 0)), overlap=n_overlap)
    sig_result = istft(result_spec.transpose((1, 0)), overlap=n_overlap)



    sf.write(r"out/out_HARD_spk.wav", sig_sp, sr)
    sf.write(r"out/out_HARD_inf.wav", sig_inf, sr)
    sf.write(r"out/out_HARD_corr_inf.wav", sig_corr_inf, sr)
    sf.write(r"out/out_HARD_result.wav", sig_result, sr)

