# -*- coding: utf-8 -*-
import numpy as np
import copy
import scipy.io as sio


def rcond(X):
    """

    :param X:
    :return:
    """
    rcond = 1.0 / np.linalg.cond(X, p=1)
    return rcond

def complex_gauss_log_pdfs(y, phi, cov):
    """

    :param y:
    :param phi:
    :param cov:
    :return:
    """
    M = y.shape[0]
    lpdf = - (np.dot(y.conj(), np.linalg.solve(cov, y)) / phi + M * (np.log(np.pi) + np.log(phi)) + np.log(
        np.linalg.det(cov)))
    return np.real(lpdf)

def covar_entropy(cov):
    """
    Return entropy eigenvalues of correlation matrix
    :param cov:
    :return:
    """

    egval, _ = np.linalg.eig(cov)
    real_eigen = egval.real / egval.real.sum()
    entropy = -(real_eigen * np.log(real_eigen)).sum()
    return entropy

def est_cgmm(stft_arr, num_iters = 10):
    """

    :param stft_arr:  spectr for each sensors - shape(bins, num_sensors, frames)
    :return: mask, R
    """

    rcond_min = 1e-5

    n_bins, n_sensors, n_frames = stft_arr.shape
    mask = np.zeros((2, n_bins, n_frames))

    mask[0, :, :] = np.random.rand(n_bins, n_frames)
    mask[1, :, :] = np.ones((n_bins, n_frames)) - mask[0, :, :]

    R_noise = np.zeros((n_bins, n_sensors, n_sensors), dtype=np.complex)
    R_noisy = np.zeros((n_bins, n_sensors, n_sensors), dtype=np.complex)

    psd = np.einsum('...it,...jt->...ij', stft_arr, stft_arr.conj())
    psd /= n_frames

    for f in range(n_bins):
        R_noise[f, :, :] = np.identity(n_sensors)
        R_noisy[f, :, :] = psd[f, :,:]

    R = np.zeros((2, n_bins, n_sensors, n_sensors), dtype=np.complex)
    R[0, :, :, :] = R_noisy
    R[1, :, :, :] = R_noise

    phi  = np.zeros((2, n_bins, n_frames))
    alfa = np.ones((n_bins, 2)) * 1. / 2

    mask_cur = copy.deepcopy(mask)
    for iter in range(num_iters):

        log_likelihood = 0;
        for k in range(2):  # speech/noise
            for f in range(n_bins):

                # M - step
                Rf = R[k, f, :, :]

                gamma = 1e-10
                while (rcond(Rf) < rcond_min):
                    Rf += gamma * np.identity(n_sensors)
                    gamma *= 10
                #Rf += 0.01 * np.identity(n_sensors)

                Rf_next = np.zeros((n_sensors, n_sensors), dtype=np.complex)
                alfa_next = 0
                for t in range(n_frames):
                    phi_k_f_t = np.real(1.0 / n_sensors * np.dot(stft_arr[f, :, t].conj(), np.linalg.solve(Rf, stft_arr[f, :, t])))
                    #phi[k, f, t] = phi_k_f_t + 0.0001
                    phi[k, f, t] = phi_k_f_t
                    Rf_next += (mask[k, f, t] / phi[k, f, t]) * np.outer(stft_arr[f, :, t], stft_arr[f, :, t].conj())
                    alfa_next += mask[k, f, t]

                #R[k, f, :, :] = Rf_next / (alfa_next + 0.0001)
                R[k, f, :, :] = Rf_next / alfa_next
                alfa[f, k] = alfa_next / n_frames


                Rf = R[k, f, :, :]
                gamma = 1e-10
                while (rcond(Rf) < rcond_min):
                    Rf += gamma * np.identity(n_sensors)
                    gamma *= 10
                # Rf += 0.01 * np.identity(n_sensors)

                # E - step

                for t in range(n_frames):
                    mask[k, f, t] = np.log(alfa[f, k]) + complex_gauss_log_pdfs(stft_arr[f, :, t], phi[k, f, t], Rf)
                    log_likelihood += mask[k, f, t]

        # Normalization mask
        max_mask = np.maximum(mask[0, :, :], mask[1, :, :])
        for k in range(2):
            mask[k, :, :] = np.exp(mask[k,:,:] - max_mask)
        mask_sum = np.sum(mask, axis = 0)

        for k in range(2):
            mask[k, :, :] = mask[k, :, :] / mask_sum



        maxd = np.max(abs(mask[1, :, :] - mask_cur[1, :, :]))
        meand = np.mean(abs(mask[1, :, :] - mask_cur[1, :, :]))

        mask_cur = copy.deepcopy(mask)
        print('iter = {} max = {} mean = {}'.format(iter, maxd, meand))

    return mask, R

def permute_mask(mask, R):
    """

    :param mask:  - mask, shape  - (2, n_bins, n_frames)
    :param R:     - cov matrix,  shape - (2, n_bins, n_sensors, n_sensors)
    :return:      - perm_mask   shape - (2, n_bins, n_frames)
    """
    _, n_bins, n_frames = mask.shape
    _, _, _, n_sensors = R.shape

    perm_mask = np.zeros((2, n_bins, n_frames))

    for f in range(n_bins):
        entropy1 = covar_entropy(R[0, f, :, :])
        entropy2 = covar_entropy(R[1, f, :, :])
        print ("f = {} ent1 = {} ent2 = {} ".format(f, entropy1, entropy2))

        if (entropy1 > entropy2):
            perm_mask[0, f, :] = mask[1, f, :]
            perm_mask[1, f, :] = mask[0, f, :]
        else:
            perm_mask[0, f, :] = mask[0, f, :]
            perm_mask[1, f, :] = mask[1, f, :]

    return perm_mask

def est_cgmm_ex(stft_arr, psd_1, psd_2, num_iters = 10, allow_cov_update = False):
    """

    :param stft_arr: spectr for each sensors - shape(bins, num_sensors, frames)
    :param psd_1:
    :param psd_2:
    :param num_iters:
    :return: mask, R
    """

    rcond_min = 1e-5

    n_bins, n_sensors, n_frames = stft_arr.shape
    mask = np.zeros((2, n_bins, n_frames))

    mask[0, :, :] = np.random.rand(n_bins, n_frames)
    mask[1, :, :] = np.ones((n_bins, n_frames)) - mask[0, :, :]

    R_noise = np.zeros((n_bins, n_sensors, n_sensors), dtype=np.complex)
    R_noisy = np.zeros((n_bins, n_sensors, n_sensors), dtype=np.complex)

    eps_reg = 0.001
    for f in range(n_bins):
        R_noise[f, :, :] = psd_1[f, :, :] + eps_reg * np.identity(n_sensors)
        R_noisy[f, :, :] = psd_2[f, :, :] + eps_reg * np.identity(n_sensors)

    R = np.zeros((2, n_bins, n_sensors, n_sensors), dtype=np.complex)
    R[0, :, :, :] = R_noisy
    R[1, :, :, :] = R_noise

    phi  = np.zeros((2, n_bins, n_frames))
    alfa = np.ones((n_bins, 2)) * 1. / 2

    mask_cur = copy.deepcopy(mask)
    for iter in range(num_iters):

        log_likelihood = 0
        for k in range(2):  # speech/noise
            for f in range(n_bins):

                # M - step
                Rf = R[k, f, :, :]

                gamma = 1e-10
                while (rcond(Rf) < rcond_min):
                    Rf += gamma * np.identity(n_sensors)
                    gamma *= 10

                Rf_next = np.zeros((n_sensors, n_sensors), dtype=np.complex)
                alfa_next = 0
                for t in range(n_frames):
                    phi_k_f_t = np.real(1.0 / n_sensors * np.dot(stft_arr[f, :, t].conj(), np.linalg.solve(Rf, stft_arr[f, :, t])))
                    phi[k, f, t] = phi_k_f_t

                    if allow_cov_update:
                        Rf_next += (mask[k, f, t] / phi[k, f, t]) * np.outer(stft_arr[f, :, t], stft_arr[f, :, t].conj())

                    alfa_next += mask[k, f, t]

                if allow_cov_update:
                    R[k, f, :, :] = Rf_next / alfa_next

                alfa[f, k] = alfa_next / n_frames

                Rf = R[k, f, :, :]
                gamma = 1e-10
                while (rcond(Rf) < rcond_min):
                    Rf += gamma * np.identity(n_sensors)
                    gamma *= 10

                # E - step

                for t in range(n_frames):
                    mask[k, f, t] = np.log(alfa[f, k]) + complex_gauss_log_pdfs(stft_arr[f, :, t], phi[k, f, t], Rf)
                    log_likelihood += mask[k, f, t]

        # Normalization mask
        max_mask = np.maximum(mask[0, :, :], mask[1, :, :])
        for k in range(2):
            mask[k, :, :] = np.exp(mask[k,:,:] - max_mask)
        mask_sum = np.sum(mask, axis = 0)

        for k in range(2):
            mask[k, :, :] = mask[k, :, :] / mask_sum



        maxd = np.max(abs(mask[1, :, :] - mask_cur[1, :, :]))
        meand = np.mean(abs(mask[1, :, :] - mask_cur[1, :, :]))

        mask_cur = copy.deepcopy(mask)
        print('iter = {} max = {} mean = {}'.format(iter, maxd, meand))

    return mask, R



