# -*- coding: utf-8 -*-
import numpy as np
import os
from mic_io import read_mic_wav_from_folder, write_mic_wav_to_folder


def main_mix(in_1_path, in_2_path, out_path, NeedSNR=None):

    if not os.path.exists(in_1_path):
        raise ValueError('Path {} not exist!'.format(in_1_path))

    if not os.path.exists(in_2_path):
        raise ValueError('Path {} not exist!'.format(in_2_path))

    if not os.path.exists(out_path):
        os.makedirs(out_path)


    vert_mic_count = 6
    hor_mic_count  = 11
    max_len_sec    = 10*60

    #################################################################
    # 1.0 - Read signal
    sig_1_all_arr, sr1      = read_mic_wav_from_folder(in_1_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    (n_channels1, n_samples1) = sig_1_all_arr.shape

    print ("Mic 1 signals  read done!")
    print ("    n_channels  = ", n_channels1)
    print ("    n_samples   = ", n_samples1)
    print ("    freq        = ", sr1)

    sig_2_all_arr, sr2      = read_mic_wav_from_folder(in_2_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    (n_channels2, n_samples2) = sig_2_all_arr.shape

    print ("Mic 2 signals  read done!")
    print ("    n_channels  = ", n_channels2)
    print ("    n_samples   = ", n_samples2)
    print ("    freq        = ", sr2)

    if sr1 != sr2 or n_channels1 != n_channels2:
        raise ValueError('sr1 != sr2 or n_channels1 != n_channels2')

    #################################################################
    # 2.0 - Cut signal

    n_samples = min(n_samples1, n_samples2)
    
    sig_1_all_arr = sig_1_all_arr[:,0:n_samples]
    sig_2_all_arr = sig_2_all_arr[:,0:n_samples]

    #################################################################
    # 3.0 - Calc SNR by chanell 0
    en_sig1  = np.sum((sig_1_all_arr[0,:])**2)/float (sig_1_all_arr.shape[1])
    en_sig2  = np.sum((sig_2_all_arr[0,:])**2)/float (sig_2_all_arr.shape[1])
    SNR_actual    = 10*np.log10(en_sig1 / en_sig2)

    print ("SNR in by chanell 0")
    print ("    SNR  = ", SNR_actual)

    #################################################################
    # 3.0 - Scale to need SNR

    if NeedSNR is not None:
        alfa = pow(10.0, (SNR_actual- NeedSNR)/20.0)

        # Suppress main signal
        sig_2_all_arr = sig_2_all_arr* alfa

    #################################################################
    # 3.1 - Calc SNR out by chanell 0
    en_sig1  = np.sum((sig_1_all_arr[0,:])**2)/float (sig_1_all_arr.shape[1])
    en_sig2  = np.sum((sig_2_all_arr[0,:])**2)/float (sig_2_all_arr.shape[1])
    SNR_actual    = 10*np.log10(en_sig1 / en_sig2)

    print ("SNR out by chanell 0")
    print ("    SNR  = ", SNR_actual)


    result_arr = sig_1_all_arr + sig_2_all_arr

    #################################################################
    # 4.0 - Write signal
    write_mic_wav_to_folder(out_path, result_arr, vert_mic_count = vert_mic_count, hor_mic_count = hor_mic_count, sr = sr1)


if __name__ == "__main__":


    snr_out = -5
#    snr_out = None
    main_mix(in_1_path = './in_1', in_2_path = './in_2', out_path = './out', NeedSNR = snr_out)

