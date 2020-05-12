# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
import matplotlib.pyplot as plt
from mic_py.mic_null import null_filter, null_filter_ex
from mic_py.mic_phase_vad import phase_vad


if __name__ == '__main__':
    #
    # ################################################################
    # # 1.0 - _wav_wbn45_dict0 PROFILE
    #
    # vert_mic_count = 1
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 45
    # n_fft          = 512
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
    # _noise_start   = 0
    # _noise_end     = 25
    #
    # _mix_start     = 0
    # _mix_end       = 25
    # #################################################################

    # #################################################################
    # 1.0 - _mus1+spk2_snr_-10_geom3 PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (20.56, -6.42)
    (angle_inf_hor_log, angle_inf_vert_log) = (-14.93, 3.18)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    angle_inf_h = -angle_inf_hor_log
    angle_inf_v = -angle_inf_vert_log


    in_wav_path    = r'./data/_mus1+spk2_snr_-10_geom3/mix/'
    out_wav_path   = r'./data/out/'


    #_noise_start   = 20
    _noise_start = 0
    _noise_end     = 64

    _mix_start  = 0
    _mix_end       = 64

    #################################################################

    # #################################################################
    # # 1.0 - LANDA PROFILE
    # vert_mic_count = 1
    # hor_mic_count  = 8
    # dHor           = 0.05
    # dVert          = 0.05
    # max_len_sec    = 60
    # n_fft          = 512
    #
    # (angle_hor_log, angle_vert_log) = (16.0, 0.0)
    # (angle_inf_hor_log, angle_inf_vert_log) = (-58, 0)
    #
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # angle_inf_h = -angle_inf_hor_log
    # angle_inf_v = -angle_inf_vert_log
    #
    # in_wav_path    = r'./data/_landa/'
    # out_wav_path   = r'./data/out/'
    #
    # _noise_start   = 0
    # _noise_end     = 50
    #
    # _mix_start     = 0
    # _mix_end       = 50
    # #################################################################

    # # #################################################################
    # # 1.0 - _sol PROFILE MVDR
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 3*60
    # n_fft          = 512
    #
    # (angle_hor_log, angle_vert_log) = (13.8845, 6.60824)
    # (angle_inf_hor_log, angle_inf_vert_log) = (-15.06, -0.31)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # angle_inf_h = -angle_inf_hor_log
    # angle_inf_v = -angle_inf_vert_log
    #
    #
    # in_wav_path    = r'./data/_sol/'
    # out_wav_path   = r'./data/out/'
    #
    #
    # #_noise_start   = 20
    # _noise_start = 9
    # _noise_end     = 102
    #
    # _mix_start  = 9
    # _mix_end       = 102
    #
    # #################################################################

    # # #################################################################
    # # 1.0 - _rameses PROFILE MVDR
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 3*60
    # n_fft          = 512
    #
    # (angle_hor_log, angle_vert_log) = (13.9677, 5.65098)
    # (angle_inf_hor_log, angle_inf_vert_log) = (-15.06, -0.31)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # angle_inf_h = -angle_inf_hor_log
    # angle_inf_v = -angle_inf_vert_log
    #
    #
    # in_wav_path    = r'./data/_rameses/'
    # out_wav_path   = r'./data/out/'
    #
    #
    # _noise_start   = 8
    # _noise_end     = 102
    #
    # _mix_start     = 8
    # _mix_end       = 102
    #
    # #################################################################

    # # #################################################################
    # # 1.0 - _du_hast PROFILE MVDR
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 3*60
    # n_fft          = 512
    #
    # (angle_hor_log, angle_vert_log) = (12.051, 5.88161)
    # (angle_inf_hor_log, angle_inf_vert_log) = (-15.06, -0.31)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # angle_inf_h = -angle_inf_hor_log
    # angle_inf_v = -angle_inf_vert_log
    #
    #
    # in_wav_path    = r'./data/_du_hast/'
    # out_wav_path   = r'./data/out/'
    #
    #
    # #_noise_start   = 17
    # _noise_start = 8
    # _noise_end     = 84
    #
    # #_mix_start = 17
    # _mix_start = 8
    # _mix_end       = 84
    #
    # #################################################################

    # # #################################################################
    # # 1.0 - chirp_mix PROFILE MVDR
    #
    # vert_mic_count = 6
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 3*60
    # n_fft          = 512
    #
    # (angle_hor_log, angle_vert_log) = (15.0, 0.0)
    # (angle_inf_hor_log, angle_inf_vert_log) = (-15.0, 0.0)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # angle_inf_h = -angle_inf_hor_log
    # angle_inf_v = -angle_inf_vert_log
    #
    #
    # #in_wav_path    = r'./data/chirp_mix/wav_-10/'
    # in_wav_path = r'./data/_vad_test/_speech+prodigy_-10dB/mix/'
    # out_wav_path   = r'./data/out/'
    #
    #
    # _noise_start   = 0
    # _noise_end     = 60
    #
    # _mix_start     = 0
    # _mix_end       = 60
    # #################################################################







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

    print ("Array data read done!")
    print ("    x_noise_arr.shape  = ", x_noise_arr.shape)
    print ("    x_mix_arr.shape    = ", x_mix_arr.shape)

    #################################################################
    # 2 - Do STFT
    stft_noise_arr =  stft_arr(x_noise_arr, fftsize = n_fft)
    stft_mix_arr   =  stft_arr(x_mix_arr, fftsize = n_fft)

    (n_bins, n_sensors, n_frames) = stft_noise_arr.shape

    print ("STFT calc done!")
    print ("    n_bins               = ", n_bins)
    print ("    n_sensors            = ", n_sensors)
    print ("    stft_noise_arr.shape = ", stft_noise_arr.shape)
    print ("    stft_mix_arr.shape   = ", stft_mix_arr.shape)


    #################################################################
    # 3 - Calc  steering vector
    print ('Calc  steering vector!')
    print ('    (angle_h, angle_v) = ', angle_h, angle_v)
    print ('    (angle_inf_h, angle_inf_v) = ', angle_inf_h, angle_inf_v)
    sensor_positions    = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position     = get_source_position(angle_h, angle_v, radius=6.0)
    source_position_inf = get_source_position(angle_inf_h, angle_inf_v, radius=6.0)

    d_arr               = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)
    d_arr_inf           = propagation_vector_free_field(sensor_positions, source_position_inf, N_fft = n_fft, F_s = sr)

    #################################################################
    # 4 - NULL filter output
    #result_spec, _  = null_filter(stft_mix_arr, d_arr_sp=d_arr.T, d_arr_inf=d_arr_inf.T)

    H = np.zeros((n_bins, n_sensors, n_frames), dtype=np.complex)
    for i in range(0, n_sensors):
        H[:, i, :] = stft_mix_arr[:, i, :] / stft_mix_arr[:, 0, :]


    def cosine_similarity(a, b):
        """
        Calc cosine distance between vectors

        :param a: (M, N) - complex matrix, M - count vectors, N - size vector
        :param b: (M, N) - complex matrix, M - count vectors, N - size vector
        :return:
            cos distance - (M)
        """
        return (np.sum(a * b.conj(), axis=-1)) / ((np.sum(a * a.conj(), axis=-1) ** 0.5) * (np.sum(b * b.conj(), axis=-1) ** 0.5))


    S1 = np.zeros((n_bins, n_frames))
    S2 = np.zeros((n_bins, n_frames))

    for t in range(0, n_frames):
        d1 = d_arr.T
        d2 = d_arr_inf.T

        S1[:, t] = np.abs(cosine_similarity(H[:, :, t], d1))
        S2[:, t] = np.abs(cosine_similarity(H[:, :, t], d2))

    R1 = np.mean(S1, axis=0)
    R2 = np.mean(S2, axis=0)



    def double_exp_average(X, sr, win_average_begin=0.060, win_average_end=0.060):
        nLen = X.shape[0]

        En = X

        Y = np.zeros(X.shape)
        Z = np.zeros(X.shape)
        Alpha = 1.0 - 1.0 / (win_average_begin * sr)
        Beta = 1.0 - 1.0 / (win_average_end * sr)

        for i in range(0, nLen - 1, 1):
            Y[i + 1] = Alpha * Y[i] + (1 - Alpha) * En[i + 1]

        for i in range(nLen - 1, 0, -1):
            Z[i - 1] = Beta * Z[i] + (1 - Beta) * Y[i - 1]

        return Z


    P = double_exp_average(R1, sr = sr/256.0, win_average_begin = 0.1, win_average_end = 0.1)
    hist, bin_edges = np.histogram(P, bins = 100)
    base = bin_edges[np.argmax(hist)]
    print ('base = ' , base)

    threshold = base + 0.01
    VAD = np.zeros((n_frames))
    VAD[P>threshold] = 0.5


    np.save(file=r'./out/P', arr=P)
    #VAD2 = phase_vad(stft_mix=stft_mix_arr, d_arr_sp=d_arr.T, sr= sr, fft_hop_size = 256)

    #################################################################
    # 6 - Plot DN
    T = np.arange(0, n_frames)*256.0/sr

    #################################################################
    # 6 - Plot hist

    # An "interface" to matplotlib.axes.Axes.hist() method
    # n, bins, patches = plt.hist(x=P, bins='auto', color='#0504aa',
    #                             alpha=0.7, rwidth=0.85)
    #

    n, bins, patches = plt.hist(x=P, bins=200, color='#0504aa', range = (0.1, 0.4),
                                alpha=0.7, rwidth=0.85)

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

    plt.plot(T, R1, T, P, T, VAD)
    #plt.plot(T, VAD, T, VAD2)
    #plt.plot(T, R2)


    #plt.grid(True)
    plt.show()




    # print(d_arr.shape)
    # print (H[50, :, 10])
    # print (d_arr.T[50,:])
    # print (d_arr_inf.T[50,:])

