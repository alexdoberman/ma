# -*- coding: utf-8 -*-

import numpy as np
from scipy import optimize


def fun_MK(x, *args):
    # Calculate the objective function for the gradient algorithm

    (n_bin, MKBeamformer_ptr) = args

    # Unpack weights : 
    wa = x[0] + 1j * x[1]

    # Normalization weigth
    wa = MKBeamformer_ptr.normalize_wa(n_bin, wa)

    # Calculate  kurtosis
    nkurt = -MKBeamformer_ptr.calc_kurtosis(n_bin, wa)
    #nkurt = MKSubstractionPtr.calc_kurtosis(n_bin, wa)

    return nkurt



def dfun_MK(x, *args):
    # Calculate the derivatives of the objective function for the gradient algorithm

    (n_bin, MKBeamformer_ptr) = args

    # Unpack  gradient
    wa = x[0] + 1j * x[1]

    # Normalization weigth
    wa = MKBeamformer_ptr.normalize_wa(n_bin, wa)

    # Calculate a gradient
    deltaWa = - MKBeamformer_ptr.calc_gradient(n_bin, wa)
    #deltaWa =  MKSubstractionPtr.calc_gradient(n_bin, wa)

    # Pack gradient
    grad = np.zeros(2)
    grad[0] = deltaWa.real
    grad[1] = deltaWa.imag

    return grad


class MKSubstraction:

    def __init__(self, stft_main, stft_ref, alfa , normalise_wa):
        """
        MK adaptive substraction
        KURT(stft_main - _wq*stft_ref) - > max

        :param stft_main: - spectr  main signal  - shape (bins, frames)
        :param stft_ref: - spectr  ref signal   - shape (bins, frames)
        :param alfa: - regularisation const
        """

        self._normalise_wa = normalise_wa
        self._stft_main = stft_main
        self._stft_ref  = stft_ref
        self._alfa      = alfa
        self._bins, self._frames = stft_main.shape

    def _calc_kurtosis(self, Y):
        """

        :param Y: signal - shape (bins, frames)
        :return:
        """

        frames = Y.shape[0]

        exY4 = 0
        exY2 = 0

        for n_frame in range(frames):
            _Y = Y[n_frame]

            Y2 = _Y * np.conjugate(_Y)
            Y4 = Y2 * Y2
    
            exY2 += (Y2.real )
            exY4 += (Y4.real )

        exY2 = exY2 / frames
        exY4 = exY4 / frames

        kurt = exY4 - 3 * exY2 * exY2
        return kurt

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



    def calc_kurtosis(self, n_bin, wq):
        """

        :param n_bin: - freq bin
        :param wq: - weigth complex value
        :return:
        """

        Y = self._calc_substraction_output(n_bin, wq)
        kurt = self._calc_kurtosis(Y) + self._alfa *np.real( wq * np.conjugate(wq))
        return kurt


    def calc_output(self, wq):
        """

        :param wq: - weigth complex value
        :return:
        """
        wq = np.expand_dims(wq , axis = 1)
        Y = self._stft_main - wq * self._stft_ref
        return Y.T

    def calc_gradient(self, n_bin, wq):
        """

        :param n_bin: - freq bin
        :param wq: - weigth complex value
        :return:
        """

        # X,Y,Z - shape  frames
        Y = self._calc_substraction_output(n_bin, wq)
        X = self._stft_main[n_bin,:]
        Z = self._stft_ref[n_bin,:]

        L1 = 0 
        L2 = 0 
        L3 = 0 

        # for n_frame in range(self._frames):
        #     L1 += Y[n_frame]*np.conjugate(Y[n_frame])*(-X[n_frame]*np.conjugate(Z[n_frame]) + wq*Z[n_frame]*np.conjugate(Z[n_frame]) )
        #     L2 += Y[n_frame]*np.conjugate(Y[n_frame])
        #     L3 += (-X[n_frame]*np.conjugate(Z[n_frame]) + wq*Z[n_frame]*np.conjugate(Z[n_frame]) )


        L1 = np.sum((Y * np.conjugate(Y) * (-X * np.conjugate(Z) + wq*Z*np.conjugate(Z))))
        L2 = np.sum(Y * np.conjugate(Y))
        L3 = np.sum(-X * np.conjugate(Z) + wq*Z*np.conjugate(Z))

        L1 /= self._frames
        L2 /= self._frames
        L3 /= self._frames

        delta = L1 - 6*L2*L3 + self._alfa * wq
        return delta

    def estimate_weights(self, n_bin, maxiter = 40):
        """

        :param n_bin:  - freq bin
        :param maxiter:
        :return:
        """

        x0 = np.zeros(2)
        args = (n_bin, self)
        res = optimize.fmin_cg(fun_MK, x0, fprime=dfun_MK, args=args, maxiter = maxiter )

        #use this badly
        #res = optimize.minimize(fun_MK, x0, args=args)['x']

        return res[0] + 1j * res[1]



def maximize_kutrosis_filter(stft_main, stft_ref):
    """
    Spectral subtraction filter
    :param stft_main: - spectr  main signal  - shape (bins, frames)
    :param stft_ref: - spectr  ref signal   - shape (bins, frames)
    :return:
        output - spectral subtraction compensate  - shape (bins, frames)
    """

    (frames, bins) = stft_main.shape

    MK_filter = MKSubstraction(stft_main.T, stft_ref.T, alfa = 0.01, normalise_wa = True)

    wq = np.zeros((bins), dtype = np.complex)
    for n_bin in range (0, bins):
        wq[n_bin] = MK_filter.estimate_weights(n_bin, maxiter = 40)
        kurt = MK_filter.calc_kurtosis(n_bin, wq[n_bin])

        print("n_bin = {}, kurt = {},  wq = {}".format(n_bin, kurt, wq[n_bin]))

    return MK_filter.calc_output(wq)




def maximize_kutrosis_filter_dbg(stft_main, stft_ref):
    """
    Spectral subtraction filter
    :param stft_main: - spectr  main signal  - shape (bins, frames)
    :param stft_ref: - spectr  ref signal   - shape (bins, frames)
    :return:
        output - spectral subtraction compensate  - shape (bins, frames)
    """

    (frames, bins) = stft_main.shape

    MK_filter = MKSubstraction(stft_main.T, stft_ref.T, alfa = 0.0)

    # n_bin = 50
    # wa =  0.09291984 + 1j * 0.40222919
    # print( "n_bin = {}  wa = {} kutr = {} ".format ( n_bin, wa, MK_filter.calc_kurtosis(n_bin, wa)))
    #
    # n_bin = 50
    # wa =  1. + 1j * 0.0
    # print( "n_bin = {}  wa = {} kutr = {} ".format ( n_bin, wa, MK_filter.calc_kurtosis(n_bin, wa)))
    #
    #
    # n_bin = 50
    # wa =  -1. + 1j * 0.0
    # print( "n_bin = {}  wa = {} kutr = {} ".format ( n_bin, wa, MK_filter.calc_kurtosis(n_bin, wa)))

    n_bin = 20
    #wa = np.array([1. - 1j ])
    #wa = np.array([-2.81899258e+53 -1.23579650e+55j])
    #k = MK_filter.calc_kurtosis(n_bin, wa[0])
    #print ("k = ", k)


    wq = np.zeros((bins), dtype = np.complex)
    wq[n_bin] = MK_filter.estimate_weights(n_bin, maxiter = 40)
    print ("wq[n_bin]", wq[n_bin])


