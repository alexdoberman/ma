# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import time

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_zelin import cross_spectral, calc_beta, zelin_filter
from mic_py.mic_mccowan import *
from mic_py.mic_noise_coherence import diffuse_noise_coherence, localized_noise_coherence, real_noise_coherence, zelin_noise_coherence

import matplotlib.pyplot as plt


if __name__ == '__main__':

    '''
    #################################################################
    # 1.0 - LANDA PROFILE
    vert_mic_count = 1
    hor_mic_count  = 8
    dHor           = 0.05
    dVert          = 0.05
    max_len_sec    = 60
    n_fft          = 512

    (angle_h, angle_v) = (-16,0)

    in_wav_path    = r'./data/_landa/'
    out_wav_path   = r'./data/out/'
    #################################################################
    '''

    '''
    #sp  [2017-11-23 19:59:26] [0x00002404] :  [INF] MicGridProcessor::SetDirectionAngles, angleFiHorz = -30.9141,  angleFiVert = 12.483
    #################################################################
    # 1.0 - _nastya_alex PROFILE

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 45
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (-30.9141, 12.483)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_nastya_alex/'
    out_wav_path   = r'./data/out/'
    #################################################################
    '''


    '''
    #sp [2017-11-07 17:15:41] [0x00000590] :  [INF] MicGridProcessor::SetDirectionAngles, angleFiHorz = 12.051,  angleFiVert = 5.88161
    #################################################################
    # 1.0 - _du_hast PROFILE

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (12.051, 5.88161)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_du_hast/'
    out_wav_path   = r'./data/out/'
    #################################################################
    '''

    #MicGridProcessor::SetDirectionAngles, angleFiHorz = 13.9677,  angleFiVert = 5.65098
    #################################################################
    # 1.0 - _rameses PROFILE

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (13.9677, 5.65098)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_rameses/'
    out_wav_path   = r'./data/out/'
    #################################################################



    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    #################################################################
    # 2 - Do STFT
    stft_arr =  stft_arr(x_all_arr, fftsize = n_fft)
    (n_bins, n_sensors, n_frames) = stft_arr.shape

    print ("STFT calc done!")
    print ("    n_bins     = ", n_bins)
    print ("    n_sensors  = ", n_sensors)
    print ("    n_frames   = ", n_frames)

    #################################################################
    # 3 - Calc  steering vector
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor = dHor, dVert = dVert)
    source_position  = get_source_position(angle_h, angle_v)
    d_arr            = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    pair_distance    = get_pair_mic_distance(sensor_positions)

    #################################################################
    # 4 - Do align 
    align_stft_arr = ds_align(stft_arr, d_arr.T)

    #################################################################
    # 5 - Calc filter output
    start_time = time.time()

    # 5.1 - Get coherence shape - (sensors_count, sensors_count, bins)
    G = diffuse_noise_coherence(bins = n_bins, freq = sr, D_IJ = pair_distance)
    #G = zelin_noise_coherence(bins = n_bins, sensors_count = vert_mic_count*hor_mic_count)

    # 5.2 - Calc mccowan filter output
    result_spec, _ = mccowan_filter(align_stft_arr, alfa = 0.7, G_IJ = G)

    print("--- %s seconds ---" % (time.time() - start_time))

    #################################################################
    # 6 inverse STFT and save
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/out_mccowan.wav", sig_out, sr)

    #################################################################
    # 7 save ds output
    result_spec = align_stft_arr.sum(axis=1)/(hor_mic_count*vert_mic_count)
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(r"out/ds.wav", sig_out, sr)

