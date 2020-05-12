# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.mic_hendriks_psd_estim import estimate_psd_hendriks


if __name__ == '__main__':

    # #################################################################
    # # 1.0 - _wgn_-25_dict_30_snr_-5 PROFILE
    #
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 45
    # n_fft          = 512
    # n_overlap      = 2
    #
    # (angle_hor_log, angle_vert_log) = (-30.0, 0.0)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # in_wav_path    = r'./data/_wgn_-25_dict_30_snr_-5/'
    # out_wav_path   = r'./data/out/'
    #
    # _mix_start     = 0.0
    # _mix_end       = 20.0
    # #################################################################

    # #################################################################
    # # 1.0 - _wav_wbn45_dict0 PROFILE
    #
    # vert_mic_count = 1
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 45
    # n_fft          = 512
    # n_overlap      = 2
    #
    # (angle_hor_log, angle_vert_log) = (0.0, 0.0)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # in_wav_path    = r'./data/_wav_wbn45_dict0/'
    # out_wav_path   = r'./data/out/'
    #
    # _mix_start     = 0.0
    # _mix_end       = 20.0
    # #################################################################

    #################################################################
    # 1.0 - _du_hast PROFILE MVDR

    vert_mic_count = 6
    hor_mic_count = 11
    dHor = 0.035
    dVert = 0.05
    max_len_sec = 2 * 60
    n_fft = 512
    n_overlap = 2

    (angle_hor_log, angle_vert_log) = (12.051, 5.88161)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path = r'./data/_du_hast/'
    out_wav_path = r'./data/out/'

    _mix_start = 17
    _mix_end = 37


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
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position          = get_source_position(angle_h, angle_v, radius = 6.0)
    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)


    # #################################################################
    # 4 - Estimate HENDRIKS PSD matrix

    psd = estimate_psd_hendriks(stft_mix=stft_all, d_arr_sp=d_arr.T, reg_const = 0.1)
    #psd = get_power_spectral_density_matrix(stft_all)

    # Regularisation
    psd = psd + 0.01 * np.identity(psd.shape[-1])
    w = get_mvdr_vector(d_arr.T, psd)
    result_spec = apply_beamforming_vector(w, stft_all)

    #result_spec = ds_beamforming(stft_all, d_arr.T)
    #################################################################
    # 5 inverse STFT and save
    sig_out = istft(result_spec.transpose((1,0)), overlap = n_overlap)
    sf.write(r"out/out_MVDR_HENDRIKS.wav", 5*sig_out, sr)


