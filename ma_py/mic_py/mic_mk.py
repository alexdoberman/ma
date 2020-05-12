# -*- coding: utf-8 -*-
import copy
import numpy as np
from scipy import optimize
from mic_py.mic_blocking_matrix import calc_blocking_matrix_from_steering


def fun_MK(x, *args):
    # Calculate the objective function for the gradient algorithm

    (n_bin, MKBeamformer_ptr) = args

    n_sensors = MKBeamformer_ptr.get_n_sensors()
    nc        = MKBeamformer_ptr.get_nc()

    # Unpack weights : 
    wa = np.zeros(n_sensors - nc, np.complex)
    for chan in range(n_sensors - nc):
        wa[chan] = x[2 * chan ] + 1j * x[2 * chan + 1]

    # Normalization weigth
    wa = MKBeamformer_ptr.normalize_wa(n_bin, wa)

    # Calculate  kurtosis
    nkurt = -MKBeamformer_ptr.calc_kurtosis(n_bin, wa)

    # Regularization
    alpha = MKBeamformer_ptr.get_alpha()
    nkurt += alpha * np.inner(wa, np.conjugate(wa))

    nkurt = np.real(nkurt)

    #print ("    fbinX = {}, nkurt = {}".format(n_bin, nkurt))
    return nkurt

def dfun_MK(x, *args):
    # Calculate the derivatives of the objective function for the gradient algorithm

    (n_bin, MKBeamformer_ptr) = args

    n_sensors = MKBeamformer_ptr.get_n_sensors()
    nc        = MKBeamformer_ptr.get_nc()


    # Unpack weights :
    wa = np.zeros(n_sensors - nc, np.complex)
    for chan in range(n_sensors - nc):
        wa[chan] = x[2 * chan ] + 1j * x[2 * chan + 1]

    #print("    wa = {}".format(wa))

    # Normalization weigth
    wa = MKBeamformer_ptr.normalize_wa(n_bin, wa)

    # Calculate a gradient
    delta_wa = - MKBeamformer_ptr.calc_gradient(n_bin, wa)

    # Add the derivative of the regularization term
    alpha = MKBeamformer_ptr.get_alpha()
    delta_wa += alpha * wa

    # Pack gradient
    grad = np.zeros(2 *  (n_sensors - nc), np.float)
    for chan in range(n_sensors - nc):
        grad[2 * chan ]     = delta_wa[chan].real
        grad[2 * chan + 1 ] = delta_wa[chan].imag

    return grad


class MKBeamformer:

    def __init__(self, stft_arr, alpha=0.01, normalise_wa = True):

        """
        :stft_main: - spectr  main signal  - shape (n_bins, n_sensors, n_frames)
        :alfa:      - regularisation const
        """

        self._stft_arr = stft_arr
        self._alpha     = alpha
        self._n_bins, self._n_sensors, self._n_frames =  stft_arr.shape
        self._normalise_wa = normalise_wa

        # Number constaraint
        self._nc = 1
        # Quiescent vectors
        self._wq = np.zeros((self._n_bins, self._n_sensors), dtype=np.complex)
        # Active vectors
        self._wa = np.zeros((self._n_bins, self._n_sensors - self._nc), dtype=np.complex)
        # Output vectors
        self._wo = np.zeros((self._n_bins, self._n_sensors), dtype=np.complex)
        # Blocking matrix
        self._B  = np.zeros((self._n_bins, self._n_sensors, self._n_sensors - self._nc), dtype=np.complex)

        # Input vectors - shape (n_bins, n_sensors, n_frames)
        self._observations = None

        # Covariance matrix of input vectors -  shape (self._n_bins, self._n_sensors, self._n_sensors)
        self._sigma_x = None

    def set_steering_vector(self, d_arr):
        """
        Set steering vector and calc blocking matrix
        :param d_arr:  Steering vector - shape (num_sensors, bins)
        :return:
        """

        self._wq = np.transpose(d_arr, (1, 0)) / self._n_sensors

        B = calc_blocking_matrix_from_steering(d_arr)
        self._B = np.transpose(B, (2, 0, 1))

    def get_n_bins(self):
        return self._n_bins

    def get_n_sensors(self):
        return self._n_sensors

    def get_nc(self):
        return self._nc

    def get_alpha(self):
        return self._alpha

    def accum_observations(self, start_frame, end_frame):
        """
        Accumulate observation from start_frame to end_frame
        :param start_frame:
        :param end_frame:
        :return: self._observations - shape (n_bins, n_sensors, n_frames)
        """

        if end_frame <= start_frame:
            raise ValueError('accum_observations: end_frame <= start_frame')

        self._observations = copy.deepcopy(self._stft_arr[:,:,start_frame:end_frame])
        return self._observations

    def calc_cov(self):
        """
        Calculate covariance matrix for observation
        :return: covariance matrix - shape (bins, num_sensors, num_sensors)
        """

        bins, num_sensors, frames = self._observations.shape

        if frames < 1 or bins < 1 or num_sensors < 2:
            raise ValueError('calc_cov: frames < 1 or bins < 1 or num_sensors < 2')

        if bins != self._n_bins or num_sensors != self._n_sensors:
            raise ValueError('calc_cov: bins != self._n_bins or num_sensors != self._n_sensors')

        sigma_x = np.zeros((bins, num_sensors, num_sensors), dtype=np.complex)

        for freq_ind in range(0, bins):
            for frame_ind in range(0, frames):
                sigma_x[freq_ind] += np.outer(self._observations[freq_ind, :, frame_ind],
                                              np.conjugate(self._observations[freq_ind, :, frame_ind]))

        for freq_ind in range(0, bins):
            sigma_x[freq_ind] /= frames

        self._sigma_x = sigma_x
        return self._sigma_x

    def _calc_GSC_output(self, n_bin, wa, X):
        """

        :param n_bin: - Freq bin
        :param wa:    - Current active weights - shape (self._n_sensors - self._nc)
        :param X:     -
        :return:
        """

        self._wo[n_bin] = self._wq[n_bin] - np.dot(self._B[n_bin], wa)

        _woH = np.transpose(np.conjugate(self._wo[n_bin]))
        Yt = np.dot(_woH, X)

        return Yt

    def normalize_wa(self, n_bin, wa):
        """
        Normalization active weights
        :param n_bin:
        :param wa:
        :return:
        """

        wa = wa.real

        if (not self._normalise_wa):
            return wa

        nrm_wa2 = np.inner(wa, np.conjugate(wa))
        nrm_wa = np.sqrt(nrm_wa2.real)

        # if self._gamma < 0:
        #     gamma = sqrt(numpy.inner(self._wq[srcX][fbinX], conjugate(self._wq[srcX][fbinX])))
        # else:
        #     gamma = self._gamma

        gamma = 1.0
        if nrm_wa > abs(gamma):  # >= 1.0:
            wa = abs(gamma) * wa / nrm_wa
        return wa

    def calc_kurtosis(self, n_bin, wa):
        """

        :param n_bin:
        :param wa: Current active weights - shape (self._n_sensors - self._nc)
        :return:
        """

        exY4 = 0
        exY2 = 0
        n_observation_frames = self._observations.shape[2]

        for frame_ind in range(0, n_observation_frames):
            X = self._observations[n_bin,:,frame_ind]
            Y = self._calc_GSC_output(n_bin, wa, X)
            Y2 = Y * np.conjugate(Y)
            Y4 = Y2 * Y2
            exY2 += Y2.real
            exY4 += Y4.real

        exY2 /= n_observation_frames
        exY4 /= n_observation_frames

        kurt = exY4 - 3.0 * exY2 * exY2
        return kurt

    def calc_gradient_2(self,  n_bin, wa):
        """

        :param n_bin:
        :param wa: Current active weights - shape (self._n_sensors - self._nc)
        :return:
        """

        n_observation_frames = self._observations.shape[2]

        dexY2 = np.zeros((self._n_sensors - self._nc), np.complex)
        dexY4 = np.zeros((self._n_sensors - self._nc), np.complex)
        exY2  = 0

        BH = np.transpose(np.conjugate(self._B[n_bin]))
        for frame_ind in range(0, n_observation_frames):
            X = self._observations[n_bin,:,frame_ind]
            Y = self._calc_GSC_output(n_bin, wa, X)

            BHX = - np.dot(BH, X)  # BH * X
            Y2 = Y * np.conjugate(Y)

            dexY4 += 2 * Y2 * BHX * np.conjugate(Y)
            dexY2 += BHX * np.conjugate(Y)
            exY2 +=  Y2.real

        dexY4 /= n_observation_frames
        dexY2 /= n_observation_frames
        exY2 /= n_observation_frames

        delta_kurt = dexY4 - 6.0 * exY2 * dexY2
        return delta_kurt

    def calc_gradient(self,  n_bin, wa):
        """

        :param n_bin:
        :param wa: Current active weights - shape (self._n_sensors - self._nc)
        :return:
        """

        n_observation_frames = self._observations.shape[2]

        dexY2 = np.zeros((self._n_sensors - self._nc), np.complex)
        dexY4 = np.zeros((self._n_sensors - self._nc), np.complex)
        exY2  = 0

        BH = np.transpose(np.conjugate(self._B[n_bin]))
        for frame_ind in range(0, n_observation_frames):
            X = self._observations[n_bin,:,frame_ind]
            Y = self._calc_GSC_output(n_bin, wa, X)

            # shape (m-1,1)
            BHX = np.dot(BH, X)
            Y2 = Y * np.conjugate(Y)
            exY2 +=  Y2.real

            BHXY = BHX * np.conjugate(Y)

            dexY4 += -Y2 * BHXY
            dexY2 += 3.0 * BHXY

        exY2 /= n_observation_frames

        dexY4 /= n_observation_frames
        dexY2 /= n_observation_frames

        delta_kurt = 2*dexY4 + 2 * exY2 * dexY2
        return delta_kurt

    def estimate_active_weights(self, n_bin, max_iter = 40):
        """

        :param n_bin:
        :param max_iter:
        :return:

        """

        # initialize optimize procedure
        ndim = 2 * (self._n_sensors - self._nc)
        #x0 = np.zeros(ndim)
        x0 = np.ones(ndim)
        args = (n_bin, self)

        res = optimize.fmin_cg(fun_MK, x0, fprime=dfun_MK, args=args, maxiter = max_iter)
        # res = optimize.minimize(fun_MK, x0, args=args)['x']

        # Unpack weights
        for chan in range(self._n_sensors - self._nc):
            self._wa[n_bin, chan] = res[2 * chan] + 1j * res[2 * chan + 1]

        # Normalization weigth
        self._wa[n_bin, :] = self.normalize_wa(n_bin, self._wa[n_bin, :])

        return self._wa[n_bin, :]

    def calc_output(self, n_bin):
        """

        :param n_bin:
        :return:
        """

        result = []
        n_observation_frames = self._observations.shape[2]
        for frame_ind in range(0, n_observation_frames):
            X = self._observations[n_bin,:,frame_ind]
            Y = self._calc_GSC_output(n_bin, self._wa[n_bin,:], X)
            result.append(Y)
        return np.array(result)


    def calc_output_dbg(self, n_bin):
        """

        :param n_bin:
        :return:
        """

        result = []
        n_observation_frames = self._observations.shape[2]
        for frame_ind in range(0, n_observation_frames):
            X = self._observations[n_bin,:,frame_ind]
            Y = self._calc_GSC_output(n_bin, self._wa[n_bin,:], X)
            result.append(Y)
        return np.array(result)





