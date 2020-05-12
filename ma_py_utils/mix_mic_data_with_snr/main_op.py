# -*- coding: utf-8 -*-
import numpy as np
import os
from mic_io import read_mic_wav_from_folder, write_mic_wav_to_folder


def main_mix(in_1_path, out_path):

    if not os.path.exists(in_1_path):
        raise ValueError('Path {} not exist!'.format(in_1_path))

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


    #################################################################
    # 3.0 - Scale to need SNR
    sig_1_all_arr = sig_1_all_arr / 4

    #################################################################
    # 4.0 - Write signal
    write_mic_wav_to_folder(out_path, sig_1_all_arr, vert_mic_count = vert_mic_count, hor_mic_count = hor_mic_count, sr = sr1)


if __name__ == "__main__":

    main_mix(in_1_path = './in_2', out_path = './out_1')

