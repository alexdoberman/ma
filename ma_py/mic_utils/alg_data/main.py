# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    gg_params      = np.load('clean_ru_speech_gg_params_freq_f_scale.npy')
    gg_params_old  = np.load('lavr_gg_params_freq_f_scale.npy')
    
    ind = []
    f = []
    s = []

    f_old = []
    s_old = []

    for i in gg_params:
        ind.append(i[0])
        f.append(i[1])
        s.append(i[2])

    for i in gg_params_old:
        ind.append(i[0])
        f_old.append(i[1])
        s_old.append(i[2])

    ind = np.array(ind)
    f = np.array(f)
    s = np.array(s)

    f_old = np.array(f_old)
    s_old = np.array(s_old)

    plt.plot(f, label="f params")
    plt.plot(f_old, label="f params_old")
    plt.legend(loc='best')
    plt.show()

    """
    plt.plot(s, label="s params")
    plt.plot(s_old, label="s params_old")
    plt.legend(loc='best')
    plt.show()
    """