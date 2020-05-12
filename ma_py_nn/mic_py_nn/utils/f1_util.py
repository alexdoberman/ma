import glob
import soundfile as sf
import numpy as np
import os

from sklearn.metrics import f1_score

from mic_py_nn.features.preprocessing import energy_mask
from mic_py_nn.features.feats import stft


def average_f1_score(true_mask_arr, predicted_mask_arr, return_mse=False, return_all_info=False):

    assert len(true_mask_arr) == len(predicted_mask_arr), 'Arrays shapes don\'t match!'

    num_ex = len(true_mask_arr)

    av_f1 = 0
    av_mse = 0

    av_TN_accuracy = 0
    av_FR_part = 0

    for i in range(num_ex):
        av_f1 += f1_score(true_mask_arr[i], predicted_mask_arr[i])
        av_mse += np.mean((true_mask_arr[i] - predicted_mask_arr[i])**2)
        print(np.mean(true_mask_arr[i] - predicted_mask_arr[i])**2)

        av_TN_accuracy += np.sum(((true_mask_arr[i] + predicted_mask_arr[i]) == 0))/(len(true_mask_arr[i]) -
                                                                                     np.count_nonzero(true_mask_arr[i]))
        av_FR_part += np.sum(((true_mask_arr[i] - predicted_mask_arr[i]) == 1))/np.sum((true_mask_arr[i] == 1))

    if return_all_info:
        return av_f1 / num_ex, av_mse / num_ex, av_TN_accuracy / num_ex, av_FR_part / num_ex

    if return_mse:
        return av_f1 / num_ex, av_mse / num_ex

    return av_f1 / num_ex


def get_dist(predicted_mask_arr):

    predicted_mask_flatten = predicted_mask_arr.flatten()
    bins, vals = np.histogram(predicted_mask_flatten, bins=50)
    return bins, vals


def process_files(true_files_path, predicted_mask_path, fft_size=512, overlap=2, bin_thr=0.5, return_mse=False,
                  return_all_info=False, return_out_vec_dist=False):

    true_mask_arr = []
    predicted_mask_arr = []
    predicted_soft_mask_arr = []

    files = glob.glob(pathname=os.path.join(true_files_path, '*_mix.wav'))

    st_cst = 0
    for file in files:

        _, name = os.path.split(file)
        npy_mask = np.load(os.path.join(predicted_mask_path, name + '.npy'))

        if st_cst == 0:
            st_cst = npy_mask.shape[0]
        if npy_mask.shape[0] == st_cst:
            predicted_soft_mask_arr.append(npy_mask)

        thr_mask = (npy_mask > bin_thr).astype(int)
        predicted_mask_arr.append(thr_mask)

        data, rate = sf.read(file)
        stft_data = stft(data, fft_size, overlap)
        mask = energy_mask(stft_data)
        true_mask_arr.append(mask)
        # print(npy_mask, mask)

    if return_out_vec_dist:
        return average_f1_score(true_mask_arr, predicted_mask_arr, return_mse, return_all_info), \
               get_dist(np.stack(predicted_soft_mask_arr))
    return average_f1_score(true_mask_arr, predicted_mask_arr, return_mse)

