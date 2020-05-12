# -*- coding: utf-8 -*-

import soundfile as sf
import numpy as np
import os
import fnmatch
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


def find_files(directory, pattern):
    """
    Search file in directory

    for f in find_files(in_path, '*.txt'):

    :param directory:
    :param pattern:
    :return:
    """
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                yield filename

def snr_check(path):

    eps = 1e-7
    N = len(list(find_files(path, '*_mix.wav')))
    data = []
    for i in range(N):
        mix   = "{}_{}.wav".format(i, 'mix')
        spk   = os.path.join(path, "{}_{}.wav".format(i, 'sp'))
        noise = os.path.join(path, "{}_{}.wav".format(i, 'mus'))

        sig_spk, rate = sf.read(spk)
        sig_noise, rate = sf.read(noise)

        pow_sp = np.sum((sig_spk) ** 2) / float(len(sig_spk))
        pow_noise = np.sum((sig_noise) ** 2) / float(len(sig_noise))
        actual_snr = 10 * np.log10(pow_sp / (pow_noise + eps))
        print ('    {}  - {}'.format(mix, actual_snr))
        data.append(actual_snr)
    return data


def plot_hist(data):
    x = np.asarray(data)

    # the histogram of the data
    n, bins, patches = plt.hist(x, 50,  facecolor='g', alpha=0.75)

    plt.xlabel('SNR')
    plt.ylabel('Count')
    plt.title('Histogram of SNR')
    plt.grid(True)
    plt.savefig(r"hist_snr.png")



data = snr_check('./v8')
plot_hist(data)