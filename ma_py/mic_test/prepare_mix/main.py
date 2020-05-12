# -*- coding: utf-8 -*-
import numpy as np
import os
from mic_io import read_mic_wav_from_folder, write_mic_wav_to_folder


def main(mus_wav_path, sp_wav_path, mus_time_beg, mus_time_end, sp_time_beg, sp_time_end, NeedSNR, 
        out_mix_wav_path, out_mus_wav_path, out_spk_wav_path):

    if not os.path.exists(out_mix_wav_path):
        os.makedirs(out_mix_wav_path)
    if not os.path.exists(out_mus_wav_path):
        os.makedirs(out_mus_wav_path)
    if not os.path.exists(out_spk_wav_path):
        os.makedirs(out_spk_wav_path)

    vert_mic_count = 6
    hor_mic_count  = 11
    max_len_sec    = 120
    time_train     = 10
    time_mix       = 29

    #################################################################
    # 1.0 - Read signal
    mus_all_arr, sr = read_mic_wav_from_folder(mus_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    (n_channels, n_samples) = mus_all_arr.shape

    print ("Music read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    sp_all_arr, sr = read_mic_wav_from_folder(sp_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    (n_channels, n_samples) = sp_all_arr.shape

    print ("Speaker read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    #################################################################
    # 1.0 - Cut signal
    mus_all_arr = mus_all_arr[:,(np.int32)(mus_time_beg*sr):(np.int32)(mus_time_end*sr)]
    sp_all_arr  = sp_all_arr[:,(np.int32)(sp_time_beg*sr):(np.int32)(sp_time_end*sr)]

    #################################################################
    # 2.0 - Calc SNR by chanell 0
    en_mus  = np.sum((mus_all_arr[0,:])**2)/float (mus_all_arr.shape[1])
    en_sp   = np.sum((sp_all_arr[0,:])**2)/float (sp_all_arr.shape[1])
    SNR_actual    = 10*np.log10(en_sp/ en_mus)

    print ("SNR by chanell 0")
    print ("    SNR  = ", SNR_actual)


    #################################################################
    # 2.0 - Scale to need SNR
    alfa = pow(10.0, (SNR_actual- NeedSNR)/20.0)

    # Suppress main signal
    mus_all_arr = mus_all_arr* alfa

    #################################################################
    # 2.0 - Result signal
    len_mus   = mus_all_arr.shape[1]
    len_sp    = sp_all_arr.shape[1]
    len_train = (np.int32)(time_train*sr)
    len_mix   = (np.int32)(time_mix*sr)

    min_len = min (len_mus - len_train, len_sp, len_mix)

    result_arr = np.zeros((n_channels, min_len + len_train))
    result_arr[:,0:len_train] = mus_all_arr[:,0:len_train]

    mus_arr = mus_all_arr[:,len_train: len_train + min_len]
    spk_arr = sp_all_arr[:,0:min_len]
    result_arr[:,len_train:]  = mus_arr + spk_arr

    #################################################################
    # 3.0 - Write signal
    write_mic_wav_to_folder(out_mix_wav_path, result_arr, vert_mic_count = 6, hor_mic_count = 11, sr = sr)
    write_mic_wav_to_folder(out_mus_wav_path, mus_arr, vert_mic_count = 6, hor_mic_count = 11, sr = sr)
    write_mic_wav_to_folder(out_spk_wav_path, spk_arr, vert_mic_count = 6, hor_mic_count = 11, sr = sr)


if __name__ == "__main__":

    #mus2 - angleFiHorz = -16.0721,  angleFiVert = -0.163439  time - 6 ... 53

    snr_out = -10
#    main('./in/mus2', './in/spk1', 6, 52, 7, 90, snr_out, './out/mix', './out/mus' ,'./out/spk' )
#    main('./in/mus1', './in/spk1', 6, 52, 7, 90, snr_out, './out/mix', './out/mus' ,'./out/spk' )

    main('./in/_buble_n3', './in/_sp_n3', 0, 40, 0, 40, snr_out, './out/mix', './out/mus' ,'./out/spk' )

