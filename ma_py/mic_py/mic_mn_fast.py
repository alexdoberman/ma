# -*- coding: utf-8 -*-
import copy
import numpy as np
import math
from scipy import optimize
from mic_py.mic_blocking_matrix import calc_blocking_matrix_from_steering

def fun_MN(x, *args):
    # Calculate the objective function for the gradient algorithm

    (n_bin, MNFastBeamformer_ptr) = args

    n_sensors = MNFastBeamformer_ptr.get_n_sensors()
    nc        = MNFastBeamformer_ptr.get_nc()

    # Unpack weights :
    wa = np.zeros(n_sensors - nc, np.complex)
    for chan in range(n_sensors - nc):
        wa[chan] = x[2 * chan ] + 1j * x[2 * chan + 1]

    # Normalization weigth
    wa = MNFastBeamformer_ptr.normalize_wa(n_bin, wa)

    # Calculate  negentropy
    negentropy = -MNFastBeamformer_ptr.calc_negentropy(n_bin, wa)

    # Regularization
    alpha = MNFastBeamformer_ptr.get_alpha()
    negentropy += alpha * np.inner(wa, np.conjugate(wa))

    negentropy = np.real(negentropy)

    #print ("    fbinX = {}, negentropy = {}".format(n_bin, -negentropy))
    return negentropy

def dfun_MN(x, *args):
    # Calculate the derivatives of the objective function for the gradient algorithm

    (n_bin, MNFastBeamformer_ptr) = args

    n_sensors = MNFastBeamformer_ptr.get_n_sensors()
    nc        = MNFastBeamformer_ptr.get_nc()


    # Unpack weights :
    wa = np.zeros(n_sensors - nc, np.complex)
    for chan in range(n_sensors - nc):
        wa[chan] = x[2 * chan ] + 1j * x[2 * chan + 1]

    #print("    wa = {}".format(wa))

    # Normalization weigth
    wa = MNFastBeamformer_ptr.normalize_wa(n_bin, wa)

    # Calculate a gradient
    delta_wa = - MNFastBeamformer_ptr.calc_gradient(n_bin, wa)

    # Add the derivative of the regularization term
    alpha = MNFastBeamformer_ptr.get_alpha()
    delta_wa += alpha * wa

    # Pack gradient
    grad = np.zeros(2 *  (n_sensors - nc), np.float)
    for chan in range(n_sensors - nc):
        grad[2 * chan ]     = delta_wa[chan].real
        grad[2 * chan + 1 ] = delta_wa[chan].imag

    return grad

class MNFastBeamformer:
    """
    Maximum negentropy beamforming
    """

    def __init__(self, stft_arr, speech_distribution_coeff_path, alpha=0.01, beta=1.0, normalise_wa = False):

        """
        :stft_main: - spectr  main signal  - shape (n_bins, n_sensors, n_frames)
        :speech_distribution_coeff_path: - path to    *.npy file that contain speech distribution coeff: shape, scale
        :alfa:      - regularisation const
        :beta:      - H(Y_gauss) - beta*H_gg(Y)
        """

        self._stft_arr = stft_arr
        self._alpha     = alpha
        self._beta      = beta
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

        # Shape param speech distribution coeff
        self._sp_f = np.zeros((self._n_bins),dtype=np.float64)
        # Scale param speech distribution coeff
        self._sp_s = np.zeros((self._n_bins),dtype=np.float64)
        # Load speech distribution coeff
        gg_params = np.load(speech_distribution_coeff_path)
        if len(gg_params) + 2 !=  self._n_bins:
            raise ValueError('Check fft size when estimate speech distribute')

        for item in gg_params:
            freq_bin = (int)(item[0])
            self._sp_f[freq_bin] = item[1]
            self._sp_s[freq_bin] = item[2]

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

    def cal_GSC_output(self, n_bin, wa, X):
        """

        :param n_bin: - Freq bin
        :param wa: - Current active weights - shape (self._n_sensors - self._nc)
        :param X: - Observations - shape (n_sensors, n_frames)
        :return:
        """
        self._wo[n_bin] = self._wq[n_bin] - np.dot(self._B[n_bin], wa)
        _woH = np.transpose(np.conjugate(self._wo[n_bin]))
        Yt = np.dot(np.transpose(X, (1,0)), _woH)
        return Yt

    def normalize_wa(self, n_bin, wa):
        """
        Normalization active weights
        :param n_bin:
        :param wa:
        :return:
        """

        if (not self._normalise_wa):
            return wa

        nrm_wa = np.sqrt(np.inner(wa, np.conjugate(wa)).real)
        gamma = 1.0
        if nrm_wa > abs(gamma):  # >= 1.0:
            wa = abs(gamma) * wa / nrm_wa
        return wa

    def _calc_negentropy(self, y, shape, beta):
        """

        :param y: - comple array
        :param f: - shape factor
        :return:
        """

        K = y.shape[0]
        B =  math.gamma(1.0/shape) / math.gamma(2.0/shape)
        sigma2 = np.mean(np.power(np.abs(y), 2))
        scale  = np.power( np.sum(np.power(np.abs(y), 2*shape))*shape/K, 1.0/shape) / B

        p1 = np.log(np.pi*sigma2) + 1
        p2 = np.log(np.pi* math.gamma(1 + 1.0/shape)*B*scale ) + 1/shape
        R = p1 - beta * p2

        return R

    def calc_negentropy(self, n_bin, wa):
        """

        :param n_bin:
        :param wa: Current active weights - shape (self._n_sensors - self._nc)
        :return:
        """

        # Calc output GSC filter
        Y_out = self.cal_GSC_output(n_bin, wa, self._observations[n_bin,:,:])

        # Negentropy output GSC filter
        negentropy = self._calc_negentropy(y=Y_out, shape=self._sp_f[n_bin], beta = self._beta)

        return negentropy

    def calc_gradient(self, n_bin, wa):
        """

        :param n_bin:
        :param wa:  - Current active weights - shape (self._n_sensors - self._nc)
        :return:
        """

        # Calc output GSC filter
        n_observation_frames = self._observations.shape[2]

        # Calc output GSC filter
        Y_out = self.cal_GSC_output(n_bin, wa, self._observations[n_bin,:,:])

        delta_neg = np.zeros((self._n_sensors - self._nc), np.complex)
        beta  = self._beta
        shape = self._sp_f[n_bin]

        K = Y_out.shape[0]
        B =  math.gamma(1.0/shape) / math.gamma(2.0/shape)
        sigma2 = np.mean(np.power(np.abs(Y_out), 2))
        scale  = np.power(np.sum(np.power(np.abs(Y_out), 2*shape))*shape/K, 1.0/shape) / B

        BH = np.transpose(np.conjugate(self._B[n_bin]))
        f = self._sp_f[n_bin]

        for frame_ind in range(0, n_observation_frames):
            X = self._observations[n_bin, :, frame_ind]
            Y = Y_out[frame_ind]
            BHX = np.dot(BH, X)
            BHXY = BHX * np.conjugate(Y)

            p = (-1.0/(sigma2) + beta*f*np.power(np.abs(Y), 2*f-2)/(np.power(B*scale, f)))
            delta_neg = delta_neg + p*BHXY

        delta_neg /= n_observation_frames
        return delta_neg

    def estimate_active_weights(self, n_bin, max_iter = 40):
        """

        :param n_bin:
        :param max_iter:
        :return:

        """

        # initialize optimize procedure
        ndim = 2 * (self._n_sensors - self._nc)

        x0 = np.zeros(ndim)
        #x0 = np.ones(ndim)

        args = (n_bin, self)

        res = optimize.fmin_cg(fun_MN, x0, fprime=dfun_MN, args=args, maxiter = max_iter)
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

        Y_out = self.cal_GSC_output(n_bin, self._wa[n_bin,:], self._observations[n_bin, :, :])
        return Y_out

    def calc_output_all(self, n_bin):
        """

        :param n_bin:
        :return:
        """

        Y_out = self.cal_GSC_output(n_bin, self._wa[n_bin,:], self._stft_arr[n_bin, :, :])
        return Y_out

def maximum_negentropy_fast_filter(stft_arr, d_arr, alpha = 0.01, beta = 1.0, normalise_wa = False, max_iter = 10,
                              speech_distribution_coeff_path = r'mic_utils\alg_data\gg_params_freq_f_scale.npy'):
    """

    :param stft_arr: - spectr for each sensors - shape (bins, num_sensors, frames)
    :param d_arr: - steering vector         - shape (bins, num_sensors)
    :param alpha:
    :param normalise_wa:
    :param beta:
    :return:
        result_spec - result spectral  - shape (bins, frames)
    """
    (n_bins, n_num_sensors, n_frames) = stft_arr.shape

    MN_filter = MNFastBeamformer(stft_arr, speech_distribution_coeff_path=speech_distribution_coeff_path, alpha=alpha,
                             normalise_wa=normalise_wa, beta = beta)
    MN_filter.set_steering_vector(d_arr)
    MN_filter.accum_observations(start_frame=0, end_frame=n_frames)

    stft_out = []
    for freq_ind in range(0, n_bins):

        # do filtering only speech freq
        if freq_ind in range(2, n_bins - 5):
            wa_res = MN_filter.estimate_active_weights(freq_ind, max_iter=max_iter)

            nrm_wa = np.inner(wa_res, np.conjugate(wa_res)).real

            print("freq_ind = {} nrm_wa = {} wa_res = {}".format(freq_ind, nrm_wa, wa_res))

        Y = MN_filter.calc_output(freq_ind)
        stft_out.append(Y)
    result_spec = np.array(stft_out)

    return result_spec





