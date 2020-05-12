# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from  mic_py.mic_mk import MKBeamformer

import matplotlib.pyplot as plt



if __name__ == '__main__':


    #################################################################
    # 1.0 - _wav_wbn45_dict0 PROFILE

    vert_mic_count = 1
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 45
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (0.0, 0.0)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_wav_wbn45_dict0/'
    out_wav_path   = r'./data/out/'
    #################################################################

    # #################################################################
    # # 1.0 - _du_hast PROFILE
    #
    # vert_mic_count = 1
    # hor_mic_count  = 11
    # dHor           = 0.035
    # dVert          = 0.05
    # max_len_sec    = 46
    # n_fft          = 512
    #
    # (angle_hor_log, angle_vert_log) = (12.051, 5.88161)
    #
    # angle_h = -angle_hor_log
    # angle_v = -angle_vert_log
    #
    # in_wav_path    = r'./data/_du_hast/'
    # out_wav_path   = r'./data/out/'
    # _mix_start   = 21
    # _mix_end     = 45
    #
    # #################################################################



    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    #x_all_arr     = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

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
    sensor_positions         = get_sensor_positions(hor_mic_count, vert_mic_count, dHor  = dHor, dVert = dVert)
    source_position          = get_source_position(angle_h, angle_v, radius = 6.0)
    d_arr                    = propagation_vector_free_field(sensor_positions, source_position, N_fft = n_fft, F_s = sr)

    # #################################################################
    # # 4 - Calc MK filter output
    #
    # MK_filter = MEKSubbandBeamformer_pr(stft_arr, NC=1, alpha=1.0E-02, beta=3.0)
    # MK_filter.set_steering_vector(d_arr)
    #
    # MK_filter.accumObservations(sFrame = 0, eFrame = 1000)
    # MK_filter.calcCov()
    #
    # stft_out = []
    # for fbinX in range (0, n_bins):
    #     wa_res = MK_filter.estimateActiveWeights(fbinX, MAXITNS=40)
    #     print("fbinX = {} , wa = {}".format(fbinX, wa_res))
    #     Y = MK_filter.calc_output(fbinX)
    #     stft_out.append(Y)
    #
    # result_spec = np.array(stft_out)



    # #################################################################
    # # 4 - Calc MK filter output
    #
    # alpha        = 0.0
    # normalise_wa = False
    #
    # MK_filter = MKBeamformer(stft_arr, alpha = alpha, normalise_wa = normalise_wa )
    # MK_filter.set_steering_vector(d_arr)
    # MK_filter.accum_observations(start_frame = 0, end_frame = n_frames)
    # MK_filter.calc_cov()
    #
    # stft_out = []
    # for freq_ind in range(0, n_bins):
    #     wa_res = MK_filter.estimate_active_weights(freq_ind, max_iter = 40)
    #     Y = MK_filter.calc_output(freq_ind)
    #     stft_out.append(Y)
    #
    # result_spec = np.array(stft_out)



    #################################################################
    # 4 - Calc MK filter output

    alpha        = 0.01
    normalise_wa = True

    MK_filter = MKBeamformer(stft_arr, alpha = alpha, normalise_wa = normalise_wa )
    MK_filter.set_steering_vector(d_arr)
    MK_filter.accum_observations(start_frame = 0, end_frame = n_frames)
    MK_filter.calc_cov()


    stft_out = []
    for freq_ind in range(0, n_bins):

        # do filtering only speech freq
        if freq_ind in range(5, n_bins - 55):
            wa_res = MK_filter.estimate_active_weights(freq_ind, max_iter = 10)
            print("freq_ind = {}  wa_res = {}".format(freq_ind, wa_res))

        Y = MK_filter.calc_output(freq_ind)
        stft_out.append(Y)

    result_spec = np.array(stft_out)


        #
    # wa = np.array([1. + 1j ])
    # k = MK_filter.calc_kurtosis(freq_ind, wa)
    # print ("k = ", k)

    #wa = MK_filter.estimate_active_weights(freq_ind, max_iter = 40)
    #print ("wa = {} ".format(wa))
    # Y = MK_filter.calc_output(freq_ind)


    #################################################################
    # 5 inverse STFT and save
    sig_out = istft(result_spec.transpose((1, 0)), overlap=2)
    sf.write(r"out/out_MK.wav", sig_out, sr)











