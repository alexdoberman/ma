# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import glob
import copy

import sys
sys.path.append('../../')

from mic_py.feats import stft, istft
from matplotlib.colors import LogNorm



def get_all_samples(file_pattern, freq_bin):

    x_all = None
    for filename in glob.iglob(file_pattern):

        print("freq_bin = {} filename = {}".format(freq_bin, filename))

        n_fft = 512
        #################################################################
        # 1.0 - Read signal
        sig, sr  = sf.read(filename, dtype = np.float64)

        #################################################################
        # 2.0 - STFT signal
        stft_sig = stft(sig, fftsize = n_fft, overlap = 2)

        #################################################################
        # 3.0 - Update hist
        x = stft_sig[:, freq_bin]

        if x_all is None:
            x_all = x
        else:
            x_all = np.hstack((x_all, x))
    return x_all
    
def plot_hist_2d(y, freq_bin):

    xmin = -0.5
    xmax = 0.5
    ymin = -0.5
    ymax = 0.5

    plt.hist2d(np.real(y), np.imag(y), range =  [[xmin, xmax], [ymin, ymax]], bins = 400, norm=LogNorm())
    if freq_bin == 0:
        plt.colorbar()
    plt.savefig('./out_pic/bin_{}.png'.format(freq_bin))


if __name__ == '__main__':

#    file_pattern = r'F:\DC\DATABASE\ruspeech_test_16\*.wav' 
#    file_pattern = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\mic_utils\hist_speech\in\*.wav' 
    file_pattern = r'.\in\*.wav' 

    n_fft = 512
    for freq_bin in range(0, int(n_fft/2) + 1):
        y = get_all_samples(file_pattern, freq_bin)
        np.save("./out_bin/bin_{}".format(freq_bin), y )
        plot_hist_2d(y, freq_bin)

