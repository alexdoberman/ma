# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector
from mic_py.mic_zelin import  zelin_filter
from mic_py.mic_mccowan import *
from mic_py.mic_noise_coherence import diffuse_noise_coherence, localized_noise_coherence, real_noise_coherence, zelin_noise_coherence



if __name__ == '__main__':
    '''
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
    pair_distance      = get_pair_mic_distance(sensor_positions)


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
    # 8 - Calc mccowan filter output

    # 8.1 - Get coherence shape - (sensors_count, sensors_count, bins)
    G = diffuse_noise_coherence(bins = n_bins, freq = sr, D_IJ = pair_distance)

    # 8.2 - Calc mccowan filter output
    result_spec, H  = mccowan_filter(align_stft_arr, alfa = 0.7, G_IJ = G)
    print ('Calc mccowan filter output done!')


    #################################################################
    # 9 - Calc MVDR + Mccowan filter output
    result_mvdr_spec =  result_mvdr_spec*10.0
    result_spec = result_mvdr_spec*H

    #################################################################
    # 10 inverse STFT and save
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/out_mvdr_mccowan.wav", sig_out, sr)

    #################################################################
    # 11 save ds output
    result_spec = align_stft_arr.sum(axis=1)/(hor_mic_count*vert_mic_count)
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/ds.wav", sig_out, sr)



