# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_mn import maximum_negentropy_filter
from mic_py.mic_mn_fast import maximum_negentropy_fast_filter
from mic_py.mic_mn_batch import batch_maximum_negentropy_filter
from mic_py.mic_zelin import zelin_filter_ex


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

    in_wav_path    = r'./data/_spk_0_du_hast_30/'
    out_wav_path   = r'./data/out/'
    #################################################################


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

    #################################################################
    # 4 - Calc MN filter output
    do_zelin_postproc = False
    alpha        = 0.1
    beta         = 1.0
    normalise_wa = False
    max_iter     = 30

    speech_distribution_coeff_path = r'mic_utils/alg_data/clean_ru_speech_gg_params_freq_f_scale.npy'
    result_spec = maximum_negentropy_filter(stft_arr = stft_arr, d_arr = d_arr, alpha=alpha, beta=beta, normalise_wa=normalise_wa, max_iter=max_iter, speech_distribution_coeff_path = speech_distribution_coeff_path)
    #result_spec = maximum_negentropy_fast_filter(stft_arr = stft_arr, d_arr = d_arr, alpha=alpha, beta=beta, normalise_wa=normalise_wa, max_iter=max_iter)
    #result_spec = batch_maximum_negentropy_filter(stft_arr = stft_arr, d_arr = d_arr, alpha=alpha, normalise_wa=normalise_wa, max_iter=max_iter)


    if do_zelin_postproc:
        #################################################################
        # 4.1 - Do zelin filter
        alpha_zelin = 0.7
        _, H = zelin_filter_ex(stft_arr = stft_arr, d_arr = d_arr, alfa = alpha_zelin, alg_type = 0)

        #################################################################
        # 4.2 - Calc MN + Zelin filter output
        result_spec = result_spec*H

    #################################################################
    # 5 inverse STFT and save
    sig_out = istft(result_spec.transpose((1, 0)), overlap=2)
    sf.write(r"out/out_MN.wav", sig_out, sr)
















