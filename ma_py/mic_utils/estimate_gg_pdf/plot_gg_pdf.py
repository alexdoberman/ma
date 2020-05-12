# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import glob
import math
from scipy import optimize
import scipy.stats as stats


if __name__ == '__main__':
    gg_params_0  = np.load('0_lavr_gg_params_freq_f_scale.npy')
    gg_params_1  = np.load('1_lavr_gg_params_freq_f_scale.npy')

    ######################################

    f_0 = []
    s_0 = []

    for i in gg_params_0:
        f_0.append(i[1])
        s_0.append(i[2])

    f_0 = np.array(f_0)
    s_0 = np.array(s_0)

    ######################################
    f_1 = []
    s_1 = []

    for i in gg_params_1:
        f_1.append(i[1])
        s_1.append(i[2])

    f_1 = np.array(f_1)
    s_1 = np.array(s_1)
    ######################################


    plt.plot(f_0[5:])
    plt.plot(f_1[5:])
    plt.show()
