# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from scipy import linalg
from sklearn import mixture
import matplotlib.pyplot as plt


def double_exp_average(X, sr, win_average_begin=0.060, win_average_end=0.060):
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

def exp_average(X, sr, win_average=0.060):

    nLen = X.shape[0]
    En = X
    Y = np.zeros(X.shape)
    Alpha = 1.0 - 1.0 / (win_average * sr)

    for i in range(0, nLen - 1, 1):
        Y[i + 1] = Alpha * Y[i] + (1 - Alpha) * En[i + 1]

    return Y

def double_exp_average_with_start_value(X, start_value, sr, win_average_begin=0.060, win_average_end=0.060):
    nLen = X.shape[0]

    En = X

    Y = np.zeros(X.shape)
    Z = np.zeros(X.shape)
    Alpha = 1.0 - 1.0 / (win_average_begin * sr)
    Beta = 1.0 - 1.0 / (win_average_end * sr)

    Y[0] = Alpha * start_value + (1 - Alpha) * En[0]
    for i in range(0, nLen - 1, 1):
        Y[i + 1] = Alpha * Y[i] + (1 - Alpha) * En[i + 1]

    Z[nLen - 1] = Y[nLen - 1]
    for i in range(nLen - 1, 0, -1):
        Z[i - 1] = Beta * Z[i] + (1 - Beta) * Y[i - 1]

    return Z

def double_exp_average_with_start_value2(X, start_value, sr, win_average_begin=0.060, win_average_end=0.060):
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


def plot_mark():
    f_offline = r'D:\REP\test\offline_sad_mark.txt'
    f_online = r'D:\REP\test\online_sad_mark.txt'


    with open(f_offline, 'r') as file:
        data_offline = file.read()
    with open(f_online, 'r') as file:
        data_online = file.read()

    data_off = np.frombuffer(data_offline.encode('utf-8'), dtype=np.uint8) - 48
    data_on = np.frombuffer(data_online.encode('utf-8'), dtype=np.uint8) - 48

    m = min(len(data_on), len(data_off))
    data_off = data_off[:m].astype(float)*0.5
    data_on = data_on[:m].astype(float)

    x = np.abs(data_on - data_off)

    #plt.plot(x, 'm')
    plt.plot(data_on, 'm')
    plt.plot(data_off, 'g')
    plt.show()


def test4():
    f_offline = r'D:\REP\test\offline_sad_R.txt'
    f_online = r'D:\REP\test\online_sad_R.txt'

    with open(f_offline, 'r') as file:
        data_offline = file.read()
    with open(f_online, 'r') as file:
        data_online = file.read()


    data_off = np.fromstring(data_offline, dtype=float, sep=', ')
    data_on = np.fromstring(data_online, dtype=float, sep=', ')

    m = min(len(data_on), len(data_off))
    data_off = data_off[:m]
    data_on = data_on[:m]

    # plt.plot(x, 'm')
    plt.plot(data_on, 'm')
    plt.plot(data_off, 'g')
    plt.show()


def test2():
    R1 = np.load(file=r'./out/R1.npy')
    P  = np.load(file=r'./out/P.npy')

    win_len     = 100
    start_value = 0.0
    partial_av  = np.empty((0))
    sr          = 16000.0/256.0
    win_average = .1

    P = double_exp_average(R1, sr, win_average, win_average)

    for i in range(0, len(R1), win_len):
        begin = i
        end   = min(i + win_len, len(R1))
        print(begin, end)

        x = R1[begin:end]
        y = double_exp_average_with_start_value(x, start_value, sr, win_average, win_average)

        start_value = y[-1]
        partial_av = np.hstack((partial_av, y))


    print (len(R1))


    plt.plot(R1, 'm')
    plt.plot(P, '-k')
    plt.plot(partial_av, '-g')
    plt.show()

    count, bins, ignored = plt.hist(P, 500, normed=True)
    count, bins, ignored = plt.hist(partial_av, 500, normed=True)
    plt.show()

def test():

    # lst_path = [r'./out/P_rameses.npy', r'./out/P_du_hast.npy', r'./out/P_sol.npy',
    #             r'./out/P_rameses_all.npy', r'./out/P_du_hast_all.npy', r'./out/P_sol_all.npy']
    #
    #
    # for f in lst_path:
    #     P = np.load(file=f)
    #     threshold, best_gmm_n_components, gmm_after_pruning, first_gauss_mean, first_gauss_covariances = _find_best_threshold(P)
    #     print ('    {},  t = {}, best_gmm_n_components = {}, gmm_after_pruning = {}, m = {} cov = {}'.format(
    #         f, threshold, best_gmm_n_components, gmm_after_pruning, first_gauss_mean, first_gauss_covariances))


    # load data
    #P = np.load(file=r'./out/P_rameses.npy')
    #P = np.load(file=r'./out/P_du_hast.npy')

    R1 = np.load(file=r'./out/R1.npy')
    T = exp_average(R1, sr = 16000.0/256.0, win_average = .1)
    shift = (int )((0.2 / (256.0/16000.0) ) / 2)
    TR = np.roll(T, shift = -shift)
    P = np.load(file=r'./out/P.npy')

    plt.plot(R1, 'm')
    plt.plot(P, '-k')
    plt.plot(T, '-g')
    #plt.plot(TR, '-b')
    plt.show()


    #P = np.load(file=r'./out/P_rameses_all.npy')
    #P = np.load(file=r'./out/P_du_hast_all.npy')
    #P = np.load(file=r'./out/P_sol_all.npy')

    print (stats.describe(P))
    # mu, sigma = 0, 1.0  # mean and standard deviation
    # P1 = np.random.normal(mu, sigma, 10000)
    # P2 = np.random.normal(mu+5, sigma, 10000)
    # #P = np.concatenate((P1,P2))
    # P = P1

    P = P.reshape(-1, 1)

    threshold, best_gmm_n_components, gmm_after_pruning, first_gauss_mean, first_gauss_covariances = _find_best_threshold(P)

    # model selection
    lowest_bic = np.infty
    n_components_range = range(2, 10)
    #cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_types = ['spherical']


    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(P)
            bic = gmm.bic(P)
            print('n_components = {}, covariance_type = {}, bic = {} '.format(n_components, cv_type, bic))
            #print('weights_ = {}, means_ = {}, covariances_ = {} '.format(gmm.weights_, gmm.means_, gmm.covariances_))

            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm

    print('Best GMM:')
    print('     weights_ = {}, means_ = {}, covariances_ = {} '.format(best_gmm.weights_, best_gmm.means_, best_gmm.covariances_))

    # Plot
    #count, bins, ignored = plt.hist(P, 100, density=True)
    #plt.show()
    count, bins, ignored = plt.hist(P, 1000, normed=True)

    x = np.linspace(0, 1.0, 1000)
    logprob = best_gmm._estimate_log_prob(x.reshape(-1, 1))
    pdf = np.exp(logprob)*best_gmm.weights_
    pdf = np.sum(pdf, axis= 1)

    plt.plot(x, pdf, '-k')
    #pdf_individual =  pdf[:, np.newaxis]
    plt.show()


def _find_best_threshold(P, delta = 0.0, min_gauss = 2, max_gauss = 10, threshold_for_garbage_gauss = 0.1):
    """
    Find best threshold for PHASE VAD

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
                                      covariance_type='spherical')
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






if __name__ == '__main__':
    test4()






