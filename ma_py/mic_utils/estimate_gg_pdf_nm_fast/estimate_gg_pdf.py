# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import glob
import math
from scipy import optimize
import scipy.stats as stats


def fun_ML_c(f, *args):
    """
    Calc log likelihood for complex data

    :param f: - shape
    :param args:
    :return:
    """

    (scale, y) = args
    K = y.shape[0]

    B = math.gamma(1.0/f) / math.gamma(2.0/f) 
    p1 = K*(np.log(f) - np.log(np.pi * math.gamma(1.0/f) *B *scale))
    p2 = np.sum(np.power( (np.abs(y)**2)/(B*scale), f))
    R = p1 - p2
    return - R

def estimate_shape_factor_c_(y, scale):
    """
    Estimate shape factor for complex data
    :param y:  - complex array
    :param scale:
    :return:
    """
    args = (scale, y)
    minimum = optimize.brent(fun_ML_c, args=args, brack=(0.02, .3))
    return minimum

def estimate_scale_c(y, shape_factor):
    """
    Estimate scale for complex data
    :param y:
    :param shape_factor:
    :return:
    """

    K = y.shape[0]
    B =  math.gamma(1.0/shape_factor) / math.gamma(2.0/shape_factor)
    scale = np.power( np.sum(np.power(np.abs(y), 2*shape_factor))*shape_factor/K, 1.0/shape_factor) / B
    return scale

def estimate_gg_pdf_param_c(y, tol = 0.0000001):
    """
    Estim GG pdf params for complex data
    :param y:
    :param tol:
    :return:
    """

    shape_factor_prev  = 1
    scale_prev         = np.mean(np.power(np.abs(y), 2))
    max_iter           = 200
    print ('scale_prev = {}'.format(scale_prev))

    for _iter in range(max_iter):
        shape_factor = estimate_shape_factor_c(y, scale_prev)
        scale        = estimate_scale_c(y, shape_factor)
        print ("    iter = {} shape = {} scale = {}".format(_iter, shape_factor, scale))

        if (np.abs(scale - scale_prev) < tol and np.abs(shape_factor - shape_factor_prev) < tol):
            return shape_factor, scale

        scale_prev = scale
        shape_factor_prev = shape_factor

    print("Warning: estimate_gg_pdf_param_c - not convergent!")
    return None, None

def main():

    n_fft = 512
    gg_params = []
    for freq_bin in range(1, int(n_fft / 2)):
        print('Process freq_ind = {}'.format(freq_bin))
        path = "./out_bin/bin_{}.npy".format(freq_bin)
        y = np.load(path)
        f, scale = estimate_gg_pdf_param_c(y)
        gg_params.append([freq_bin, f, scale])
        np.save("gg_params_freq_f_scale", np.array(gg_params))

    np.save("gg_params_freq_f_scale", np.array(gg_params))

def estimate_shape_factor_c(y, scale):
    """
    Estimate shape factor for complex data
    :param y:  - complex array
    :param scale:
    :return:
    """
    args = (scale, y)


    ff = np.linspace(0.02, 0.9, 200)
    L = []
    for i in ff:
        args = (scale, y)
        L.append(fun_ML_c(i, *args))
    L = np.array(L)
    min_index = np.argmin(L)

    l_min = np.min(min_index - 5, 0)
    r_min = min_index + 5

    a = ff[l_min]
    b = ff[r_min]
    c = ff[min_index]

    minimum = optimize.brent(fun_ML_c, args=args, brack=(a, b))
    return minimum
    #return L[min_index]


def debug_run():

    freq_bin = 1

    print('Process freq_ind = {}'.format(freq_bin))
    path = "./out_bin/bin_{}.npy".format(freq_bin)
    y = np.load(path)
    # f, scale = estimate_gg_pdf_param_c(y)
    # print (f, scale)

    ff = np.linspace(0.02, 0.9, 200)
    L = []
    for i in ff:
        args = (0.04692564477433535, y)
        L.append(fun_ML_c(i, *args))
    L = np.array(L)
    min_index = np.argmin(L)

    l_min = np.min(min_index - 5, 0)
    r_min = min_index + 5

    a = ff[l_min]
    b = ff[r_min]
    c = ff[min_index ]
    print (l_min,min_index,r_min)
    print (a,c,b)


    plt.plot(ff, L, label="L")
    plt.legend(loc='best')
    plt.show()

if __name__ == '__main__':
    #debug_run()
    main()


