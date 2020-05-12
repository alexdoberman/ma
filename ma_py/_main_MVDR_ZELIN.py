# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector
from mic_py.mic_zelin import  zelin_filter


if __name__ == '__main__':

    # 0 - Define params 
    #################################################################
    # 1.0 - _du_hast PROFILE MVDR

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 2*60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (12.051, 5.88161)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_du_hast/'
    out_wav_path   = r'./data/out/'

    _noise_start   = 8
    _noise_end     = 17

    _mix_start     = 17
    _mix_end       = 84

    _sp_start      = 84
    _sp_end        = 102
    #################################################################

    '''
    #################################################################
    # 1.0 - _rameses PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (13.9677, 5.65098)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_rameses/'
    out_wav_path   = r'./data/out/'


    _noise_start   = 8
    _noise_end     = 26

    _mix_start     = 26
    _mix_end       = 104

    _sp_start      = 104
    _sp_end        = 128
    #################################################################
    '''

    '''
    #back 0-7
    #sp [2017-12-01 13:46:23] [0x00000344] :  [INF] MicGridProcessor::SetDirectionAngles, angleFiHorz = 2.37685,  angleFiVert = -4.85129
    #################################################################
    # 1.0 - _5 PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 2*60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (2.37685, -4.85129)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_5/'
    out_wav_path   = r'./data/out/'


    _noise_start   = 0
    _noise_end     = 7

    _mix_start     = 7
    _mix_end       = 88

    _sp_start      = 88
    _sp_end        = 90
    #################################################################
    '''

    '''
    #################################################################
    # 1.0 - _sol PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (13.8845, 6.60824)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_sol/'
    out_wav_path   = r'./data/out/'


    _noise_start   = 9
    _noise_end     = 28

    _mix_start     = 28
    _mix_end       = 98

    _sp_start      = 98
    _sp_end        = 112
    #################################################################
    '''



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
    x_sp_arr    = x_all_arr[:,(np.int32)(_sp_start*sr):(np.int32)(_sp_end*sr)]

    print ("Array data read done!")
    print ("    x_noise_arr.shape  = ", x_noise_arr.shape)
    print ("    x_mix_arr.shape    = ", x_mix_arr.shape)
    print ("    x_sp_arr.shape     = ", x_sp_arr.shape)


    #################################################################
    # 2 - Do STFT
    stft_noise_arr =  stft_arr(x_noise_arr, fftsize = n_fft)
    stft_mix_arr   =  stft_arr(x_mix_arr, fftsize = n_fft)
    stft_sp_arr    =  stft_arr(x_sp_arr, fftsize = n_fft)

    (n_bins, n_sensors, n_frames) = stft_noise_arr.shape

    print ("STFT calc done!")
    print ("    n_bins               = ", n_bins)
    print ("    n_sensors            = ", n_sensors)
    print ("    stft_noise_arr.shape = ", stft_noise_arr.shape)
    print ("    stft_mix_arr.shape   = ", stft_mix_arr.shape)
    print ("    stft_sp_arr.shape    = ", stft_sp_arr.shape)


    #################################################################
    # 3 - Calc  steering vector
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    sensor_positions   = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position    = get_source_position(angle_h, angle_v)
    d_arr              = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)

    #################################################################
    # 4  - Calc psd matrix
    psd_noise_matrix = get_power_spectral_density_matrix(stft_noise_arr)
    print ('Calc psd matrix done!')
    print ('    psd_noise_matrix.shape = ', psd_noise_matrix.shape)
    np.save('psd_noise_mat', psd_noise_matrix)

    #################################################################
    # 5 - Regularisation psd matrix
    psd_noise_matrix = psd_noise_matrix + 0.001*np.identity(psd_noise_matrix.shape[-1])

    #################################################################
    # 6 - Apply MVDR
    w = get_mvdr_vector(d_arr.T,  psd_noise_matrix)
    result_mvdr_spec = apply_beamforming_vector(w, stft_mix_arr)
    print ('Apply MVDR done!')

    #################################################################
    # 7 - Do align 
    align_stft_arr = ds_align(stft_mix_arr, d_arr.T)
    print ('Align mix_arr done!')

    #################################################################
    # 8 - Calc zelin filter output
    result_spec, H = zelin_filter(stft_arr = align_stft_arr, alfa = 0.7, alg_type = 0)
    print ('Calc zelin filter output done!')

    #################################################################
    # 9 - Calc MVDR + Zelin filter output
    result_spec = result_mvdr_spec*H

    #################################################################
    # 10 inverse STFT and save
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/out_mvdr_zelin.wav", sig_out, sr)

    #################################################################
    # 11 save ds output
    result_spec = align_stft_arr.sum(axis=1)/(hor_mic_count*vert_mic_count)
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/ds.wav", sig_out, sr)



