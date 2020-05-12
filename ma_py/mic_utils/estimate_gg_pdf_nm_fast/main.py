# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import glob
import math
from scipy import optimize
import scipy.stats as stats


def fun_ML(f, *args):
    """
    Calc log likelihood for real data

    :param f: - shape
    :param args:
    :return:
    """

    (scale, y) = args

    K = y.shape[0]
    B = np.sqrt( math.gamma(1.0/f) / math.gamma(3.0/f) )

    p1 = K* np.log(1 / (2 * scale * B * math.gamma(1 + 1.0/f) ))
    p2 = (1/(np.power(B * scale, f)))
    p3 = np.sum(np.power(np.abs(y), f))
    R = p1 - p2*p3
    return - R

def fun_ML_c(f, *args):
    """
    Calc log likelihood for complex data

    :param f: - shape
    :param args:
    :return:
    """

    (scale, y) = args

    K = y.shape[0]
    B = np.sqrt( math.gamma(2.0/f) / math.gamma(4.0/f) )
    B2 = B*B
    s2 = scale*scale

    p1 = K* np.log(f / (2*np.pi * s2 * B2 * math.gamma(2.0/f) ))
    p2 = (1/(  np.power(B * scale, f)))
    p3 = np.sum(np.power(np.abs(y), f))
    R = p1 - p2*p3
    return - R

def estimate_shape_factor(y, scale):
    """
    Estimate shape factor for real data
    :param y:
    :param scale:
    :return:
    """
    args = (scale, y)
    minimum = optimize.brent(fun_ML, args=args, brack=(0.1, 1))
    return minimum

def estimate_shape_factor_c(y, scale):
    """
    Estimate shape factor for complex data
    :param y:  - complex array
    :param scale:
    :return:
    """
    args = (scale, y)
    minimum = optimize.brent(fun_ML_c, args=args, brack=(0.1, 0.2))
    return minimum

def estimate_scale(y, shape_factor):
    """
    Estimate scale for real data
    :param y:
    :param shape_factor:
    :return:
    """

    K = y.shape[0]
    B = np.sqrt(math.gamma(1.0 / shape_factor) / math.gamma(3.0 / shape_factor))

    scale = np.power(np.sum(np.power(np.abs(y), shape_factor)) * shape_factor / K, 1.0 / shape_factor) / B
    return scale

def estimate_scale_c(y, shape_factor):
    """
    Estimate scale for complex data
    :param y:
    :param shape_factor:
    :return:
    """

    K = y.shape[0]
    B = np.sqrt( math.gamma(2.0/shape_factor) / math.gamma(4.0/shape_factor) )
    scale = np.power( np.sum(np.power(np.abs(y), shape_factor))*shape_factor/(2.0*K), 1.0/shape_factor) / B
    return scale

def estimate_gg_pdf_param(y, tol = 0.0000001):
    """
    Estim GG pdf params for real data
    :param y:
    :param tol:
    :return:
    """

    shape_factor_prev  = 1
    scale_prev         = 0
    max_iter           = 200

    for _iter in range(max_iter):

        scale        = estimate_scale(y, shape_factor_prev)
        shape_factor = estimate_shape_factor(y, scale)
        print ("    iter = {} scale = {} shape = {}".format(_iter, scale, shape_factor))

        if (np.abs(scale - scale_prev) < tol and np.abs(shape_factor - shape_factor_prev) < tol):
            return shape_factor, scale

        scale_prev = scale
        shape_factor_prev = shape_factor

    print("Warning: estimate_gg_pdf_param - not convergent!")
    return None, None

def estimate_gg_pdf_param_c(y, tol = 0.0000001):
    """
    Estim GG pdf params for complex data
    :param y:
    :param tol:
    :return:
    """

    shape_factor_prev  = 1
    scale_prev         = 0
    max_iter           = 200

    for _iter in range(max_iter):

        scale        = estimate_scale_c(y, shape_factor_prev)
        shape_factor = estimate_shape_factor_c(y, scale)
        print ("    iter = {} scale = {} shape = {}".format(_iter, scale, shape_factor))

        if (np.abs(scale - scale_prev) < tol and np.abs(shape_factor - shape_factor_prev) < tol):
            return shape_factor, scale

        scale_prev = scale
        shape_factor_prev = shape_factor

    print("Warning: estimate_gg_pdf_param_c - not convergent!")
    return None, None

def gg_pdf(f, scale, y):
    """
    Calc GG pdf
    :param f:
    :param scale:
    :param y:
    :return:
    """

    B = np.sqrt( math.gamma(1.0/f) / math.gamma(3.0/f) )

    p1 = 1 / (2 * math.gamma(1.0 + 1.0 / f) * B * scale)
    p2 = np.exp(- np.power(np.abs(y/(scale*B)) , f))
    R = p1*p2
    return R

def ggc_pdf(f, scale, z):
    """

    Calc complex GG pdf
    :param f:
    :param scale:
    :param z:
    :return:
    """

    B = np.sqrt( math.gamma(2.0/f) / math.gamma(4.0/f) )
    B2 = B*B
    s2 = scale*scale

    p1 = f / (2 * np.pi * s2 * B2 * math.gamma(2.0 / f))
    p2 = np.exp(- np.power(np.abs(z/(scale*B)) , f))
    R = p1*p2
    return R


if __name__ == '__main__':
    gg_params  = np.load('gg_params_freq_f_scale.npy')

    ind = []
    f = []
    s = []

    for i in gg_params:
        ind.append(i[0])
        f.append(i[1])
        s.append(i[2])

    ind = np.array(ind)
    f = np.array(f)
    s = np.array(s)

    plt.plot(f[5:])
    plt.show()
    # #y = np.load('bin_28.npy')
    # #y = np.load('bin_50.npy')
    # y = np.load('bin_1 .npy')
    # print (y.shape)
    # f, scale = estimate_gg_pdf_param_c(y)

    # n_fft = 512
    # gg_params = []
    # for freq_bin in range(1, int(n_fft / 2)):
    #     print('Process freq_ind = {}'.format(freq_bin))
    #     path = "./in_bin/bin_{}.npy".format(freq_bin)
    #     y = np.load(path)
    #     f, scale = estimate_gg_pdf_param_c(y)
    #     gg_params.append([freq_bin, f, scale])
    #
    # np.save("gg_params_freq_f_scale", np.array(gg_params) )





# #####################################
    #
    # datamin = -0.5
    # datamax = 0.5
    # numbins = 200
    # mybins = np.linspace(datamin, datamax, numbins)
    # gg_pdf  = np.array([gg_pdf(f, scale, z) for z in mybins])
    #
    # htemp, jnk = np.histogram(y, mybins, density = True)
    # center = (mybins[:-1] + mybins[1:]) / 2
    #
    # s = 0
    # for v in htemp:
    #     s = s + v*(datamax-datamin)/numbins
    # print ('s = {}'.format(s))
    #
    #
    # plt.plot(mybins, gg_pdf, 'b-')
    # plt.plot(center, htemp, 'g-')
    #
    #
    # plt.show()


    #####################################

    # h = y
    # h.sort()
    #
    # fit = stats.norm.pdf(h, np.mean(h), np.std(h))  # this is a fitting indeed
    #
    # plt.plot(h, fit, '-o')
    # plt.hist(h,bins = 300, normed=True)  # use this to draw histogram of your data
    #
    # plt.show()
