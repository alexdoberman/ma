# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

import sys
sys.path.append('../')
import os
from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_cov_matrix_tracking import cov_matrix_tracking
from scipy.signal import medfilt
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.mic_zelin import cross_spectral, calc_beta, zelin_filter


if __name__ == '__main__':

    #################################################################
    # read mic config
    # vert_mic_count = int(mic_cfg['vert_mic_count'])
    '''
    # du_hast profile
    vert_mic_count = 6
    hor_mic_count = 11
    mic_count = hor_mic_count*vert_mic_count
    dHor = 0.035
    dVert = 0.05
    max_len_sec = 2*60
    n_fft = 512

    (angle_hor_log, angle_vert_log) = (12.051, 5.88161)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path = r'./data/_du_hast/'
    out_wav_path = r'./data/out/'
    start_noise_time = 8
    end_noise_time = 17
    '''
    '''
    # ramees
    vert_mic_count = 6
    hor_mic_count = 11
    mic_count = 11*6
    dHor = 0.035
    dVert = 0.05
    max_len_sec = 2 * 60
    n_fft = 512

    (angle_hor_log, angle_vert_log) = (13.9677, 5.65098)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path = r'./data/_rameses/'
    out_wav_path = r'./data/out/'

    start_noise_time = 8
    end_noise_time = 26

    _mix_start = 26
    _mix_end = 104

    _sp_start = 104
    _sp_end = 128
    '''
    # sol
    vert_mic_count = 6
    hor_mic_count = 11
    mic_count = vert_mic_count*hor_mic_count
    dHor = 0.035
    dVert = 0.05
    max_len_sec = 3 * 60
    n_fft = 512

    (angle_hor_log, angle_vert_log) = (13.8845, 6.60824)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path = r'./data/_sol/'
    out_wav_path = r'./data/out/'

    start_noise_time = 9
    end_noise_time = 28

    _mix_start = 28
    _mix_end = 98

    _sp_start = 98
    _sp_end = 112
    #################################################################

    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    x_pref = x_all_arr[:end_noise_time*sr]
    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    #################################################################
    # 2 - Do STFT
    stft_all =  stft_arr(x_all_arr, fftsize = n_fft)
    stft_noise = stft_arr(x_all_arr[:, int(start_noise_time)*sr:int(end_noise_time)*sr], fftsize = n_fft)
    stft_mix = stft_arr(x_all_arr[:, int(end_noise_time)*sr:], fftsize = n_fft)
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

    psd_noise_matrix = get_power_spectral_density_matrix(stft_noise)
    psd_noise_matrix = psd_noise_matrix + 0.01*np.identity(psd_noise_matrix.shape[-1])
    w = get_mvdr_vector(d_arr.T,  psd_noise_matrix)
    #################################################################
    # 4 - Calc filter output
    result_spec = cov_matrix_tracking(stft_noise, stft_mix, w*mic_count, filter_type='blocking_matrix')

    #################################################################
    # 5 - Do align
    align_stft_arr = ds_align(stft_mix, d_arr.T)
    print('Align mix_arr done!')

    #################################################################
    # 6 - Calc zelin filter output
    result_spec_, H = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)
    print('Calc zelin filter output done!')

    #################################################################
    # 7 - Calc tracking + Zelin filter output
    result_spec = result_spec * H

    #################################################################
    # 8 inverse STFT and save
    sig_out = istft(result_spec.T, overlap=2)

    sf.write(r"out/out_cov_matrix_tracking_zelin.wav", sig_out, sr)
