# -*- coding: utf-8 -*-

import numpy as np
import math
from scipy import optimize

eps = 0.0000001

def fun_MN(x, *args):
    # Calculate the objective function for the gradient algorithm

    (n_bin, MNSubstraction_ptr) = args

    # Unpack weights : 
    wa = x[0] + 1j * x[1]

    # Normalization weigth
    wa = MNSubstraction_ptr.normalize_wa(n_bin, wa)

    # Calculate  negentropy
    negentropy = MNSubstraction_ptr.calc_negentropy(n_bin, wa)

    #print ("x = {} negentropy = {}".format(wa, negentropy))
    return -negentropy

def _pdf_gg(y, shape, scale):
    """
    Calculate pdf for GG complex distribution

    :param y:  - complex value
    :param shape:
    :param scale:
    :return:
    """

    B = np.sqrt(math.gamma(2.0 / shape) / math.gamma(4.0 / shape))
    B2 = B * B
    s2 = scale * scale

    p1 = shape / (2 * np.pi * s2 * B2 * math.gamma(2.0 / shape))
    p2 = np.exp(- np.power(np.abs(y / (scale * B)), shape))
    pdf = p1 * p2
    return pdf

def _pdf_gauss(y, sigma):
    """
    Calculate pdf for gauss complex distribution

    :param y:  - complex value
    :param sigma:
    :return:
    """
    pdf = 1 / (np.pi * sigma * sigma) * np.exp(- y * np.conj(y) / (sigma * sigma))
    return pdf

def _estimate_gauss_sigma(y):
    """
    Estimate Gauss sigma for complex data
    :param y:
    :return:
    """

    K = y.shape[0]
    sigma = np.sqrt(np.sum(np.power(np.abs(y), 2))/K)
    return sigma

def _estimate_gg_scale(y, shape):
    """
    Estimate GG scale for complex data
    :param y:
    :param shape:
    :return:
    """

    K = y.shape[0]
    B = np.sqrt( math.gamma(2.0/shape) / math.gamma(4.0/shape))
    scale = np.power(np.sum(np.power(np.abs(y), shape))*shape/(2.0*K), 1.0/shape) / B
    return scale

def _calc_H_gauss(y, sigma):
    """
    Calc entropy for gauss distribution
    :param y:
    :param sigma:
    :return:
    """

    H = -np.mean(np.log(_pdf_gauss(y, sigma=sigma) + eps))
    return H

def _calc_H_gg(y, shape, scale):
    """
    Calc entropy for gg distribution
    :param y:
    :param shape:
    :param scale:
    :return:
    """

    H = -np.mean(np.log(_pdf_gg(y, shape=shape, scale=scale) + eps))
    return H

def _calc_negentropy(y, shape, beta = 1):
    """
    Calc negentropy for gg distribution
    :param y:
    :param shape:
    :return:
    """
    sigma =_estimate_gauss_sigma(y)
    scale = _estimate_gg_scale(y, shape=shape)
    H_gauss = _calc_H_gauss(y, sigma=sigma)
    H_gg = _calc_H_gg(y, shape=shape, scale=scale)
    J = H_gauss - beta * H_gg
    return np.real(J)


class MNSubstraction:

    def __init__(self, stft_main, stft_ref, alfa , normalise_wa, speech_distribution_coeff_path):
        """
        Maximum negentropy adaptive substraction

        Y = Y_main - wq*Y_ref
        H(Y_gauss) - H(Y_gg) - alfa*wq^2  -> max

        :param stft_main: - spectr  main signal  - shape (bins, frames)
        :param stft_ref: - spectr  ref signal   - shape (bins, frames)
        :param alfa: - regularisation const
        :param speech_distribution_coeff_path: - path
        """

        self._normalise_wa = normalise_wa
        self._stft_main = stft_main
        self._stft_ref  = stft_ref
        self._alfa      = alfa
        self._bins, self._frames = stft_main.shape

        # Shape param speech distribution coeff
        self._sp_f = np.zeros((self._bins), dtype=np.float64)
        # Scale param speech distribution coeff
        self._sp_s = np.zeros((self._bins), dtype=np.float64)
        # Load speech distribution coeff
        gg_params = np.load(speech_distribution_coeff_path)
        # if len(gg_params) + 2 != self._bins:
        #     raise ValueError('Check fft size when estimate speech distribute')

        for item in gg_params:
            freq_bin = (int)(item[0])
            self._sp_f[freq_bin] = item[1]
            self._sp_s[freq_bin] = item[2]


    def _calc_substraction_output(self, n_bin, wq):
        """

        :param n_bin: - freq bin
        :param wq: - weigth complex value
        :return:
        """
        Y = self._stft_main[n_bin,:] - wq*self._stft_ref[n_bin,:]
        return Y

    def normalize_wa(self, n_bin, wa):
        """
        Normalization active weights
        :param n_bin:
        :param wa:
        :return:
        """

        if (not self._normalise_wa):
            return wa

        nrm_wa2 = np.inner(wa, np.conjugate(wa))
        nrm_wa = np.sqrt(nrm_wa2.real)

        gamma = 1.0
        if nrm_wa > abs(gamma):  # >= 1.0:
            wa = abs(gamma) * wa / nrm_wa
        return wa

    def calc_negentropy(self, n_bin, wq):
        """

        :param n_bin: - freq bin
        :param wq: - weigth complex value
        :return:
        """

        Y = self._calc_substraction_output(n_bin, wq)

        J = _calc_negentropy(y=Y, shape=self._sp_f[n_bin])
        negentropy = J - self._alfa *np.real( wq * np.conjugate(wq))
        return negentropy

    def calc_output(self, wq):
        """

        :param wq: - weigth complex value
        :return:
        """
        wq = np.expand_dims(wq , axis = 1)
        Y = self._stft_main - wq * self._stft_ref
        return Y.T

    def estimate_weights(self, n_bin, maxiter = 40):
        """

        :param n_bin:  - freq bin
        :param maxiter:
        :return:
        """

        x0 = np.zeros(2)
        args = (n_bin, self)

        #use this badly
        res = optimize.minimize(fun_MN, x0, args=args)['x']

        return res[0] + 1j * res[1]

def maximize_negentropy_filter(stft_main, stft_ref, speech_distribution_coeff_path):
    """
    Spectral subtraction filter
    :param stft_main: - spectr  main signal  - shape (bins, frames)
    :param stft_ref: - spectr  ref signal   - shape (bins, frames)
    :return:
        output - spectral subtraction compensate  - shape (bins, frames)
    """

    (frames, bins) = stft_main.shape

    MN_filter = MNSubstraction(stft_main.T, stft_ref.T, alfa=0.01, normalise_wa=False, speech_distribution_coeff_path=speech_distribution_coeff_path)

    wq = np.zeros((bins), dtype=np.complex)
    for n_bin in range(0, bins):
        wq[n_bin] = MN_filter.estimate_weights(n_bin, maxiter=40)

        # Normalization weigth
        wq[n_bin] = MN_filter.normalize_wa(n_bin, wq[n_bin])

        neg = MN_filter.calc_negentropy(n_bin, wq[n_bin])

        print("n_bin = {}, negentropy = {},  wq = {}".format(n_bin, neg, wq[n_bin]))

    return MN_filter.calc_output(wq)

def maximize_negentropy_filter_dbg(stft_main, stft_ref):
    """
    Spectral subtraction filter
    :param stft_main: - spectr  main signal  - shape (bins, frames)
    :param stft_ref: - spectr  ref signal   - shape (bins, frames)
    :return:
        output - spectral subtraction compensate  - shape (bins, frames)
    """

    (frames, bins) = stft_main.shape

    MN_filter = MNSubstraction(stft_main.T, stft_ref.T, alfa=0.0, normalise_wa=False)

    n_bin = 20

    wq = np.zeros((bins), dtype=np.complex)
    wq[n_bin] = MN_filter.estimate_weights(n_bin, maxiter=40)
    print("wq[n_bin]", wq[n_bin])

    # wq = np.zeros((bins), dtype=np.complex)
    # for n_bin in range(0, bins):
    #     wq[n_bin] = MN_filter.estimate_weights(n_bin, maxiter=40)
    #     kurt = MN_filter.calc_kurtosis(n_bin, wq[n_bin])
    #
    #     print("n_bin = {}, kurt = {},  wq = {}".format(n_bin, kurt, wq[n_bin]))
    #
    # return MN_filter.calc_output(wq)