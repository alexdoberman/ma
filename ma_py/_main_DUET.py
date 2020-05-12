# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_gsc import *

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def xxx():

    x = np.array([0, 0, 1, 1, 1, 1])
    y = np.array([0, 0.1, 1, 1, 1, 1])

    H, xedges, yedges = np.histogram2d(x, y, bins=4, range=[[-1, 3], [-1, 3]])

    X, Y = np.meshgrid(xedges, yedges)

    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(131, title='imshow: square bins')
    plt.imshow(H, interpolation='nearest', origin='low',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    ax = fig.add_subplot(132, title='pcolormesh: actual edges', aspect='equal')
    X, Y = np.meshgrid(xedges, yedges)
    ax.pcolormesh(X, Y, H)

    # X, Y = np.meshgrid(xedges, yedges)
    # ax.pcolormesh(X, Y, H)

    # # Make data.
    # X = np.arange(-5, 5, 0.25)
    # Y = np.arange(-5, 5, 0.25)
    # X, Y = np.meshgrid(X, Y)
    # R = np.sqrt(X**2 + Y**2)
    # Z = np.sin(R)

    # print ('X.shape = {},  Y.shape = {}, Z.shape = {}'.format(X.shape, Y.shape, hist.shape))
    #
    # # Plot the surface.
    # surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    #
    # # Customize the z axis.
    # ax.set_zlim(0.01, 2.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()



if __name__ == '__main__':
    # xxx()
    # exit()

    #################################################################
    # 1.0 - _wav_wbn45_dict0 PROFILE

    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 45*5
    n_fft          = 512

    (angle_hor_log, angle_vert_log) = (0.0, 0.0)

    angle_h = -angle_hor_log
    angle_v = -angle_vert_log

    in_wav_path    = r'./data/_du_hast/'
    out_wav_path   = r'./data/out/'

    _mix_start = 8
    _mix_end = 17

    #################################################################


    #################################################################
    # 1.0 - Read signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    x_all_arr     = x_all_arr[:,(np.int32)(_mix_start*sr):(np.int32)(_mix_end*sr)]

    (n_channels, n_samples) = x_all_arr.shape

    print ("Array data read done!")
    print ("    n_channels  = ", n_channels)
    print ("    n_samples   = ", n_samples)
    print ("    freq        = ", sr)

    #################################################################
    # 2 - Do STFT
    stft_all =  stft_arr(x_all_arr, fftsize = n_fft)
    (n_bins, n_sensors, n_frames) = stft_all.shape

    print ("STFT calc done!")
    print ("    n_bins     = ", n_bins)
    print ("    n_sensors  = ", n_sensors)
    print ("    n_frames   = ", n_frames)

    #################################################################
    #
    x1 = stft_all[:, 0, :]
    x2 = stft_all[:, 1, :]

    a = np.zeros((n_frames, n_bins), dtype=np.float32)
    delta = np.zeros((n_frames, n_bins), dtype=np.float32)

    for t in range(n_frames):
        for w in range(n_bins):
            R_21 = x1[w, t] / x2[w, t]
            a[t, w] = np.abs(R_21)
            delta[t, w] = -np.angle(R_21)/( np.pi * w /(n_bins + 0.001) + 0.001)

    a = a.flatten()
    delta = delta.flatten()

    print('a.mean = {}, a.min = {}, a.max = {}'.format(np.mean(a), np.min(a), np.max(a)))
    print('delta.mean = {}, delta.min = {}, delta.max = {}'.format(np.mean(delta), np.min(delta), np.max(delta)))



    H, xedges, yedges = np.histogram2d(a, delta, bins=200, range=[[0.1, 2.0], [-2.0, 2.0]], normed =True)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(X, Y, H, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0.01, 2.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)


    #fig = plt.figure(figsize=(7, 3))
    # ax = fig.add_subplot(131, title='imshow: square bins')
    # plt.imshow(H, interpolation='nearest', origin='low',  extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # plt.show()

    # ax = fig.add_subplot(132, title='pcolormesh: actual edges', aspect='equal')
    # X, Y = np.meshgrid(xedges, yedges)
    # ax.pcolormesh(X, Y, H)

    plt.show()




    # #################################################################
    # # 5 inverse STFT and save
    # sig_out = istft(result_spec.transpose((1,0)), overlap = 2)
    # sf.write(r"out/out_DS.wav", sig_out, sr)


