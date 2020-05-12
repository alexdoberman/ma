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



def gather_hist():

    datamin = -0.5
    datamax = 0.5
    numbins = 200
    mybins = np.linspace(datamin, datamax, numbins)
    myhist = np.zeros(numbins-1, dtype='int32')


    freq_bin = 28
    sr       = 0

#    file_pattern = r'F:\DC\DATABASE\ruspeech_test\*.wav' 
#    file_pattern = r'F:\DC\DATABASE\classic_music_test_slice\*.wav'
    file_pattern = r'D:\REP\svn_MicArrAlgorithm2\MA_PY\mic_utils\hist_speech\in\*.wav'

    x_all = None

    for filename in glob.iglob(file_pattern):

        print(filename)

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

        htemp, jnk = np.histogram(x, mybins)
        myhist += htemp

    # Plot


    np.save("400hz.dat", x_all )

    xmin = -0.5
    xmax = 0.5
    ymin = -0.5
    ymax = 0.5

    plt.hist2d(np.real(x_all), np.imag(x_all), range =  [[xmin, xmax], [ymin, ymax]], bins=400, norm=LogNorm())
    plt.colorbar()
    plt.show()

    """
    myhist = myhist/np.sum(myhist)

    width = 0.7 * (mybins[1] - mybins[0])
    center = (mybins[:-1] + mybins[1:]) / 2
    plt.bar(center, myhist, align='center', width=width)

    freq_hz  = freq_bin*sr/n_fft

    plt.xlabel('stft.real')
    plt.ylabel('Probability')
    plt.title('Histogram of stft.real freq = {} hz'.format(freq_hz))
    plt.grid(True)
    plt.show()
    """



def plot_hist():

    y = np.load("bin_50.npy" )

    xmin = -0.5
    xmax = 0.5
    ymin = -0.5
    ymax = 0.5

    plt.hist2d(np.real(y), np.imag(y), range =  [[xmin, xmax], [ymin, ymax]], bins = 400, norm=LogNorm())
    plt.colorbar()
    plt.show()





if __name__ == '__main__':

    plot_hist()
