# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

import sys
sys.path.append('../../')

from mic_py.calc_metric import calc_metric
from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align


def delay_summ(in_wav_path, angle_hor_log, angle_vert_log, out_wav_path):

    #################################################################
    # 0.0 - load PROFILE

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log
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

    #################################################################
    # 4 - Calc GSC filter output
    result_spec = ds_beamforming(stft_all, d_arr.T)

    #################################################################
    # 5 inverse STFT and save
    sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    sf.write(out_wav_path, sig_out, sr)


def prepare_main_ref_signal(wav_path, angle_hor_log, angle_vert_log, out_path):

    # DS in sp 
    delay_summ(wav_path + r'\spk', angle_hor_log, angle_vert_log, out_path + r'\ds_spk.wav')

    # DS in mus
    delay_summ(wav_path + r'\mus', angle_hor_log, angle_vert_log, out_path + r'\ds_mus.wav')

    # DS in mix
    delay_summ(wav_path + r'\mix', angle_hor_log, angle_vert_log, out_path + r'\ds_mix.wav')


if __name__ == '__main__':

    '''
    wav_path        = r'..\..\data\_sdr_test\out_mus1_spk1_snr_-20'
    angle_hor_log   = 7.1539
    angle_vert_log  = 7.39515
    out_path        = r'.\temp'
    prepare_main_ref_signal(wav_path, angle_hor_log, angle_vert_log, out_path)
    '''


    wav_path        = r'..\..\data\_sdr_test\test'
    angle_hor_log   = -15.1388
    angle_vert_log  = -4.32925
    out_path        = r'..\..\data\_sdr_test\test'
    prepare_main_ref_signal(wav_path, angle_hor_log, angle_vert_log, out_path)


