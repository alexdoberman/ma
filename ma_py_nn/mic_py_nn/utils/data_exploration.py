import os
import argparse
import numpy as np
import soundfile as sf


from mic_py_nn.features.feats import stft

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def get_hist(arg):
    root_path = arg.root_path
    d_type = arg.type

    dir_name = os.path.normpath(root_path).split(os.sep)[-2]
    freq = 100

    files_lst = os.listdir(root_path)

    flatten_stat = np.zeros(0)

    for file in files_lst:
        if file.startswith('{}_'.format(d_type)):
            file_path = os.path.join(root_path, file)
            data, rate = sf.read(file_path)

            stft_data = np.real(stft(data, 512, 2))

            flatten_stat = np.hstack((flatten_stat, stft_data[:, freq]))

    plt.hist(flatten_stat, bins=100)
    plt.title('hist for {} - {}'.format(d_type, freq))
    plt.grid(True)
    plt.savefig('./temp/hist_{}_{}.png'.format(d_type, dir_name))


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        '-r', '--root_path',
        metavar='R',
        default='None',
        help='Path to the dataset')

    arg_parser.add_argument(
        '-t', '--type',
        metavar='T',
        default='None',
        help='Data type')

    args_obj = arg_parser.parse_args()

    get_hist(args_obj)
