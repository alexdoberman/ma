# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_zelin import cross_spectral, calc_beta, zelin_filter
from mic_py.mic_localisation import pseudospectrum_MUSIC

import matplotlib.pyplot as plt


if __name__ == '__main__':

    #################################################################
    # 1.0 - _no_echo_dist_1_0_angle_60  PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 8
    dHor           = 0.05
    dVert          = 0.05
    max_len_sec    = 80
    n_fft          = 512

    in_wav_path    = r'./data/_speach_noise_12db/'
#    in_wav_path    = r'./data/_wav_wbn45_dict0/'
#    in_wav_path    = r'./data/_wav_wbn45/'
#    out_wav_path   = r'./data/out/'

    _mix_start   = 0
    _mix_end     = 30
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
    stft_arr =  stft_arr(x_all_arr, fftsize = n_fft)
    (n_bins, n_sensors, n_frames) = stft_arr.shape

    print ("STFT calc done!")
    print ("    n_bins     = ", n_bins)
    print ("    n_sensors  = ", n_sensors)
    print ("    n_frames   = ", n_frames)

    #################################################################
    # 3 - Do localisation
    L = 2
    angle_step  = 1
    arr_angle_h = range(-90, 90, angle_step)
    #arr_angle_v = range(-90, 90, angle_step)
    arr_angle_v = np.array([0])

    POW =  pseudospectrum_MUSIC(stft_arr, L, arr_angle_h, arr_angle_v,
                vert_mic_count = vert_mic_count,
                hor_mic_count  = hor_mic_count,
                dHor           = dHor,
                dVert          = dVert,
                n_fft          = n_fft,
                sr             = sr)

    # (len(arr_angle_h), len(arr_angle_v), n_bins)
    print ("POW.shape = {}".format(POW.shape))

    #################################################################
    # 4 - Plot DN
    lst_freq_bins = [10, 20, 30, 40, 150]

    for f_bin in lst_freq_bins:
        freq  = f_bin*sr/n_fft
        plt.plot(arr_angle_h, POW[:, 0, f_bin], label="{} HZ".format(freq))

    plt.plot(arr_angle_h, np.mean(POW[:, 0, 10:50], axis=1), label="AVERAGE")

    plt.xlabel('angle_h (s)')
    plt.ylabel('MUSIC POW')
    plt.title('MUSIC alg')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(r".\out\MUSIC.png")
    plt.show()


