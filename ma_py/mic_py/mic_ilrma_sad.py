import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from mic_py.feats import istft
from mic_py.mic_double_exp_averaging import double_exp_average
from mic_py.mic_make_mask import make_mask, mask_to_frames
from numba import jit



def _find_best_threshold_hist(P, bias_for_base_level = 0.01, bins_number=100):
    """
    Find best threshold for PHASE VAD using max hist edges

    :param P:
    :return:
        threshold - best threshold
    """
    # Find peak on histogramm <=> base level noise
    hist, bin_edges = np.histogram(P, bins=bins_number)
    nose_base_level = bin_edges[np.argmax(hist)]

    # Compare to threshold
    threshold = nose_base_level + bias_for_base_level
    return threshold

@jit()
def ilrma_sad(stft_arr, sr, n_fft, overlap):
    stft_all_arr_ilrma = np.concatenate(
        (np.transpose(stft_arr, (2, 0, 1))[:, :, :7], np.transpose(stft_arr, (2, 0, 1))[:, :, 11:18]),
        axis=-1)

    # 5 - Do ILRMA

    weights = np.load(r'./weights14.npy')


    res = ilrma(stft_all_arr_ilrma, n_iter=15, n_components=2, W0=weights, seed=0)

    # 6 - Calculate entropy for each signal
    entr = np.zeros(res.shape[-1])
    for i in range(res.shape[-1]):
        for j in range(res.shape[1]):
            entr[i] += entropy(np.real(res[:, j, i] * np.conj(res[:, j, i])))

    # 7 - extract signal with the minimum entropy

    resulting_sig = istft(res[:, :, np.argmin(entr)], overlap=overlap)
    resulting_sig[:300] = resulting_sig[300]
    resulting_sig[-300:] = resulting_sig[-300]


    # 8 - Do exponential smoothing over resulting signal

    average_sig = double_exp_average(resulting_sig, sr)
    average_sig[-300:] = average_sig[-300]
    average_sig[:300] = average_sig[300]


    # 9 - Make mask

    # mask = make_mask(average_sig, percent_threshold=percent_thrs)
    average_sig = (average_sig - np.min(average_sig)) / (np.max(average_sig) - np.min(average_sig))
    thrs = _find_best_threshold_hist(average_sig, bins_number=100, bias_for_base_level=0.01)
    mask = np.array(average_sig < thrs, dtype='float32')
    mask_frames = mask_to_frames(mask, int(n_fft), int(n_fft / overlap))

    return mask_frames


def ilrma(X, n_src=None, n_iter=20, proj_back=False, W0=None, T0=None, V0=None,
        n_components=2,
        return_filters=0,
        callback=None, seed=0):


    n_frames, n_freq, n_chan = X.shape

    # default to determined case
    if n_src is None:
        n_src = X.shape[2]

    # Only supports determined case

    # if n_chan != n_src:
    #     raise AssertionError

    # initialize the demixing matrices
    # The demixing matrix has the following dimensions (nfrequencies, nchannels, nsources),
    if W0 is None:
        W = np.array([np.eye(n_chan, n_src) for f in range(n_freq)], dtype=np.complex128)
    else:
        W = W0.copy()
    np.random.seed(seed)
    # initialize the nonnegative matrixes with random values
    if T0 is None:
        T = np.array(np.random.rand(n_freq, n_components, n_src))
    else:
        T = T0.copy()
    if V0 is None:
        V = np.array(np.random.rand(n_components, n_frames, n_src))
    else:
        V = V0.copy()
    Y = np.zeros((n_frames, n_freq, n_src), dtype=np.complex128)
    R = np.zeros((n_freq, V.shape[1], n_src))
    I = np.eye(n_src, n_src)
    U = np.zeros((n_freq, n_src, n_chan, n_chan), dtype=np.complex128)
    product = np.zeros((n_freq, n_chan, n_chan), dtype=np.complex128)
    lambda_aux = np.zeros(n_src)
    machine_epsilon = np.finfo(float).eps

    for n in range(0, n_src):
        R[:, :, n] = np.dot(T[:,:, n], V[:,:,n])

    # Compute the demixed output
    def demix(Y, X, W):
        for f in range(n_freq):
            Y[:,f,:] = np.dot(X[:,f,:], np.conj(W[f,:,:]))

    demix(Y, X, W)
    P = np.power(abs(Y), 2.)



    for epoch in tqdm(range(n_iter), desc='current_iter:'):



        if callback is not None and epoch % 1 == 0:
            print("Iteration: " + str(epoch))

            if proj_back:
                pass

            else:
                callback(Y)

        # simple loop as a start
        for s in range(n_src):
            iR = 1 / R[:,:,s]
            T[:,:,s] *= np.sqrt( np.dot(P[:,:,s].T * iR ** 2, V[:,:,s].T) / np.dot(iR, V[:,:,s].T) )
            T[T < machine_epsilon] = machine_epsilon

            R[:, :, s] = np.dot(T[:, :, s], V[:, :, s])

            iR = 1 / R[:,:,s]
            V[:,:,s] *= np.sqrt( np.dot(T[:,:,s].T, P[:,:,s].T * iR ** 2) / np.dot(T[:,:,s].T, iR) )
            V[V < machine_epsilon] = machine_epsilon

            R[:, :, s] = np.dot(T[:, :, s], V[:, :, s])

            # Compute Auxiliary Variable and update the demixing matrix
            for f in range(n_freq):
                U[f,s,:,:] = np.dot(X[:,f,:].T, np.conj(X[:,f,:]) / R[f,:,None,s]) / n_frames
                product[f,:,:] = np.dot(np.conj(W[f,:,:].T), U[f,s,:,:])
                W[f,:,s] = np.linalg.solve(product[f,:,:], I[s,:])

            # print('before_normalization W_mean is {}'.format(np.mean(np.real(W[:50,:,:]))))



                w_Unorm = np.inner(np.conj(W[f,:,s]), np.dot(U[f,s,:,:], W[f,:,s]))
                W[f,:,s] /= np.power(w_Unorm, 0.5)


            # print('after_normalization W_mean is {}'.format(np.mean(np.real(W[:50, :, :]))))

        demix(Y, X, W)
        P = np.abs(Y) ** 2

        for s in range(n_src):
            lambda_aux[s] = 1 / np.sqrt(np.mean(P[:,:,s]))

            W[:,:,s] *= lambda_aux[s]
            P[:,:,s] *= lambda_aux[s] ** 2
            R[:,:,s] *= lambda_aux[s] ** 2
            T[:,:,s] *= lambda_aux[s] ** 2

    if return_filters:
        return Y, W, T, V
    else:
        return Y