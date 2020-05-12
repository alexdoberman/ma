# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import os
import matplotlib.pyplot as plt

from mic_py.feats import *
from mic_py.mic_io import read_mic_wav_from_lst
from mic_py.mic_gcc_phat import gcc_phat

def main():

    #################################################################
    # 0.0 - profile
    path_to_data="E:\STORAGE_PROJECT_2\AMI\data\EN2001b_1min"
    mic_count = 8
    max_len_sec = 60
    _mix_start = 0
    _mix_end = 40
    gcc_window_sec = .5
    gcc_hop_sec = 0.25

    #################################################################
    # 1.0 - Read signal
    lst_files_ = ["EN2001b.Array1-0{}.wav".format(n) for n in range(1,mic_count +1)]
    lst_files = [os.path.join(path_to_data, f) for f in lst_files_]
    x_all_arr, sr = read_mic_wav_from_lst(lst_files, max_len_sec)
    x_all_arr     = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

    (n_channels, n_samples) = x_all_arr.shape
    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    # #################################################################
    # # 2.0 - Calc feat
    # tau, cc = gcc_phat(sig = x_all_arr[1,:], refsig= x_all_arr[0,:], fs=sr, max_tau=None, interp=16)
    # print("tau , cc = ", tau, cc)


    #################################################################
    # 2.0 - Calc feats slice
    gcc_window_size = int(sr * gcc_window_sec)
    gcc_hop_size = int(sr * gcc_hop_sec)
    id_main_ch = 2
    id_ref_ch = 0

    lst_tau = []
    for i in range(0, n_samples - gcc_window_size, gcc_hop_size):
        print("     i:i+win", i, i + gcc_window_size)
        tau, cc = gcc_phat(sig=x_all_arr[id_main_ch, i:i + gcc_window_size], refsig=x_all_arr[id_ref_ch, i:i + gcc_window_size], fs=sr, max_tau=None, interp=16)
        lst_tau.append(tau)
        ind = np.argsort(np.abs(cc))[-4:]
        print(cc[ind])





    lst_tau = np.array(lst_tau)
    time    = np.arange(0, len(lst_tau)*gcc_hop_sec, gcc_hop_sec)
    #################################################################
    # 3.0 - Plot DN
    plt.plot(time, lst_tau)
    plt.xlabel('time')
    plt.ylabel('cc')
    plt.title('GCC_PHAT')
    plt.grid(True)
    plt.savefig(r".\out\DS.png")
    plt.show()


    # #################################################################
    # # 3.0 - Plot DN
    # plt.plot(cc)
    # plt.xlabel('k')
    # plt.ylabel('cc')
    # plt.title('GCC_PHAT')
    # plt.grid(True)
    # plt.savefig(r".\out\DS.png")
    # plt.show()

if __name__ == '__main__':

    main()
