# -*- coding: utf-8 -*-
import numpy as np
from sklearn import mixture

def phase_sad(stft_mix, d_arr_sp, sr = 16000.0, fft_hop_size = 256, threshold_type = 'hist', **kwargs):
    """
    Phase speech absence detector

    :param stft_mix:       - spectr mix for each sensors - shape (bins, num_sensors, frames)
    :param d_arr_sp:       - steering vector in speaker direction - shape (bins, num_sensors)
    :param sr:             - sample rate
    :param fft_hop_size:   - overlap in samples
    :param threshold_type: - type algorithm: 'hist', 'gmm'
    :param kwargs:         - params for algorithm

    :return:
        SAD                - SAD mark  - shape (frames) , 1.0 - speaker present else 0.0
    """

    bins, num_sensors, frames = stft_mix.shape

    # Calc RTF
    eps = 0.00000001
    H = np.zeros((bins, num_sensors, frames), dtype=np.complex)
    for i in range(0, num_sensors):
        H[:, i, :] = stft_mix[:, i, :] / (stft_mix[:, 0, :] + eps)

    # Calc cos distance between RTF and steering vector
    S = np.zeros((bins, frames))
    for t in range(0, frames):
        S[:, t] = np.abs(_cosine_similarity(H[:, :, t], d_arr_sp))
    R = np.mean(S, axis=0)

    # Average distance
    win_average = 0.1 # sec
    P = _double_exp_average(R, sr = sr/fft_hop_size, win_average_begin = win_average, win_average_end = win_average)

    threshold = -1
    if threshold_type == 'hist':
        threshold = _find_best_threshold_hist(P, **kwargs)
        print('    phase_sad: type = {}, threshold = {}'.format(threshold_type, threshold))

    elif threshold_type == 'gmm':
        # threshold, best_gmm_n_components, gmm_after_pruning, first_gauss_mean, first_gauss_covariances \
        #     = _find_best_threshold_model_gmm(P, delta=0.0, min_gauss=2, max_gauss=10, threshold_for_garbage_gauss=0.1)
        threshold, best_gmm_n_components, gmm_after_pruning, first_gauss_mean, first_gauss_covariances \
            = _find_best_threshold_model_gmm(P, **kwargs)
        print('    phase_sad: type = {}, threshold = {}, best_gmm_n_components = {}, gmm_after_pruning = {}, '
              'first_gauss_mean = {}, first_gauss_covariances={}'.format(threshold_type, threshold, best_gmm_n_components,
                                                                         gmm_after_pruning, first_gauss_mean,
                                                                         first_gauss_covariances))
    else:
        assert False, 'type_sad = {} unsupported'.format(threshold_type)


    SAD = np.zeros((frames))
    SAD[P>threshold] = 1.0

    return 1 - SAD

def _cosine_similarity(a, b):
    """
    Calc cosine distance between vectors

    :param a: (M, N) - complex matrix, M - count vectors, N - size vector
    :param b: (M, N) - complex matrix, M - count vectors, N - size vector
    :return:
        cos distance - (M)
    """
    return (np.sum(a * b.conj(), axis=-1)) / (
    (np.sum(a * a.conj(), axis=-1) ** 0.5) * (np.sum(b * b.conj(), axis=-1) ** 0.5))

def _double_exp_average(X, sr, win_average_begin=0.060, win_average_end=0.060):
    """

    :param X:
    :param sr:
    :param win_average_begin:
    :param win_average_end:
    :return:
    """
    nLen = X.shape[0]

    En = X

    Y = np.zeros(X.shape)
    Z = np.zeros(X.shape)
    Alpha = 1.0 - 1.0 / (win_average_begin * sr)
    Beta = 1.0 - 1.0 / (win_average_end * sr)

    for i in range(0, nLen - 1, 1):
        Y[i + 1] = Alpha * Y[i] + (1 - Alpha) * En[i + 1]

    for i in range(nLen - 1, 0, -1):
        Z[i - 1] = Beta * Z[i] + (1 - Beta) * Y[i - 1]

    return Z

def _find_best_threshold_hist(P, bias_for_base_level = 0.01):
    """
    Find best threshold for PHASE VAD using max hist edges

    :param P:
    :return:
        threshold - best threshold
    """
    # Find peak on histogramm <=> base level noise
    hist, bin_edges = np.histogram(P, bins=100)
    nose_base_level = bin_edges[np.argmax(hist)]

    # Compare to threshold
    threshold = nose_base_level + bias_for_base_level
    return threshold

def _find_best_threshold_model_gmm(P, delta = 0.0, min_gauss = 2, max_gauss = 10, threshold_for_garbage_gauss = 0.1):
    """
    Find best threshold for PHASE VAD using model selection

    :param P: - average cos distance between steering and RTF from data - np.array
    :param delta: - count for sigma
    :param min_gauss:
    :param mas_gauss:
    :param threshold_for_garbage_gauss:
    :return:
             threshold, best_gmm_n_components, gmm_after_pruning, first_gauss_mean, first_gauss_covariances
             threshold - best threshold
    """

    assert min_gauss > 0 and min_gauss <= max_gauss, \
        '_find_best_threshold: min_gauss > 0 and min_gauss <= max_gauss'
    assert threshold_for_garbage_gauss >= 0. and threshold_for_garbage_gauss <= 1., \
        '_find_best_threshold: threshold_for_garbage_gauss >= 0. and threshold_for_garbage_gauss <= 1.'

    # Reshape for use in GaussianMixture
    P = P.reshape(-1, 1)

    # Model selection
    lowest_bic = np.infty
    n_components_range = range(min_gauss, max_gauss+1)
    best_gmm = None
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        # best_gmm.weights_, best_gmm.means_, best_gmm.covariances_
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type='spherical',
                                      random_state=23)
        gmm.fit(P)
        bic = gmm.bic(P)

        # Select model with smallest BIC
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    # Pruning gmm
    j = np.where(best_gmm.weights_ > threshold_for_garbage_gauss)

    weights_ = best_gmm.weights_[j]
    means_ = np.squeeze(best_gmm.means_[j])
    covariances_ = best_gmm.covariances_[j]

    j = np.argsort(means_)
    weights_ = weights_[j]
    means_ = means_[j]
    covariances_ = covariances_[j]

    threshold =  means_[0] +  delta * covariances_[0]

    best_gmm_n_components = best_gmm.n_components
    gmm_after_pruning = len(weights_)
    first_gauss_mean = means_[0]
    first_gauss_covariances = covariances_[0]

    return threshold, best_gmm_n_components, gmm_after_pruning, first_gauss_mean, first_gauss_covariances


