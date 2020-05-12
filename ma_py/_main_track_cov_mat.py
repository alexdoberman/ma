# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_track_cov_mat import track_cov_mat_filter, calc_psd_v0



if __name__ == '__main__':


    #################################################################
    # 1.0 - _wav_wbn45_dict0 PROFILE

    vert_mic_count = 1
    hor_mic_count  = 2
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 45
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (0.0, 0.0)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    #in_wav_path = r'./data/_mix_wgn_45_x2/'
    in_wav_path = r'./data/_mix_wgn_20_x2/'

    #in_wav_path = r'./data/_mix_sp_noise_2_0_0/'

    out_wav_path   = r'./data/out/'

    _nose_start   = 0
    _nose_end     = 10

    #################################################################


    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    x_noise_arr   = x_all_arr[:,(np.int32)(_nose_start*sr):(np.int32)(_nose_end*sr)]
    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    #################################################################
    # 2 - Do STFT
    stft_all_arr = stft_arr(x_all_arr, fftsize=n_fft)
    stft_noise_arr = stft_arr(x_noise_arr, fftsize=n_fft)
    (n_bins, n_sensors, n_frames) = stft_all_arr.shape

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

    #################################################################
    # 4 - Calc GSC filter output

    psd_V0 = calc_psd_v0(stft_noise_arr=stft_noise_arr)
    track_cov_mat_filter(stft_arr=stft_all_arr, d_arr=d_arr.T, psd_V0=psd_V0)

    #################################################################
    # 5 inverse STFT and save
    #sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    #sf.write(r"out/out_GSC.wav", sig_out, sr)



