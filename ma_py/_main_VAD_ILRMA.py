import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import pyroomacoustics as pra
from pyroomacoustics.bss.ilrma import ilrma
from scipy.stats import entropy

from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft



if __name__ == '__main__':

    # 0 - Define params
    #################################################################
    # 1.0 - _du_hast PROFILE MVDR
    vert_mic_count = 6
    hor_mic_count  = 11
    dHor           = 0.035
    dVert          = 0.05
    max_len_sec    = 3*60
    n_fft          = 512


    in_wav_path    = r'./data/_sdr_test/out_mus1_spk1_snr_-10/mix'
    # in_wav_path = r'./data/simulation/_speech+prodigy_-5dB/'




    #################################################################

    ########################################################################
    # make a mask first

    #################################################################
    # 1.0 - Read signal,  do STFT and extract data for ILRMA apply ILRMA weights and extract cleaned signal
    x_all_arr, sr = read_mic_wav_from_folder(in_wav_path, vert_mic_count, hor_mic_count, max_len_sec = max_len_sec)
    (n_channels, n_samples) = x_all_arr.shape

    stft_all_arr = stft_arr(x_all_arr, fftsize = n_fft)

    stft_all_arr_ilrma = np.concatenate((np.transpose(stft_all_arr, (2, 0, 1))[:,:, :7], np.transpose(stft_all_arr, (2, 0, 1))[:,:, 11:18]), axis=-1)

    # 2.0 - Apply ILRMA and extract cleaned signal
    weights = np.load(r'./mic_utils/room_simulation\room_simulation_Gleb\weights14.npy')
    n_iter = 15
    n_comp = 2

    res = ilrma(stft_all_arr_ilrma, n_iter=n_iter, n_components=n_comp, W0=weights)
    entr = np.zeros(res.shape[-1])

    for i in range(res.shape[-1]):
        for j in range(res.shape[1]):
            entr[i] += entropy(np.real(res[:,j,i]*np.conj(res[:,j,i])))



    resulting_sig = istft(res[:,:,np.argmin(entr)], overlap=2)



    # def demix(X, W):
    #     Y = np.zeros(X.shape, dtype=X.dtype)
    #     for f in range(X.shape[1]):
    #         Y[:,f,:] = np.dot(X[:,f,:], np.conj(W[f,:,:]))
    #     return Y
    #
    # Y = demix(stft_all_arr_ilrma, weights)

    # resulting_sig = istft(Y[:,:,8], overlap=2)

    # sf.write(r'./out/Simulation/MVDR+ILRMA/test.wav', resulting_sig, sr)

    # 3.0 Calculate average amd make a mask
    def double_exp_average(X, sr, vad_win_average_begin=0.1, vad_win_average_end=0.1):
        nLen = X.shape[0]

        En = X ** 2

        Y = np.zeros(X.shape)
        Z = np.zeros(X.shape)
        Alpha = 1.0 - 1.0 / (vad_win_average_begin * sr)
        Beta = 1.0 - 1.0 / (vad_win_average_end * sr)

        for i in range(0, nLen - 1, 1):
            Y[i + 1] = Alpha * Y[i] + (1 - Alpha) * En[i + 1]

        for i in range(nLen - 1, 0, -1):
            Z[i - 1] = Beta * Z[i] + (1 - Beta) * Y[i - 1]

        return Z


    average_sig = double_exp_average(resulting_sig, sr)
    average_sig[-300:] = average_sig[-300]

    def make_mask(average_sig, percent_threshold=80):
        return np.array((average_sig-np.min(average_sig)) < np.max((average_sig)-np.min(average_sig))/percent_threshold,
                        dtype='float32')



    mask = make_mask(average_sig, percent_threshold=60)
    plt.plot(average_sig)
    plt.plot(mask*np.max(average_sig))
    # plt.hlines(np.max(average_sig)/50, xmin=0, xmax=len(average_sig))






    def make_noise_borders(mask):
        noise_borders = []
        j = 0
        for i, el in enumerate(mask):
            if el == 1:
                j += 1
            elif el != 1 and j != 0:
                noise_borders.append([i-j-1, i-1])
                j = 0
        return np.array((noise_borders))


    noise_borders = make_noise_borders(mask)

    for el in noise_borders:
        print(el)

    plt.show()


