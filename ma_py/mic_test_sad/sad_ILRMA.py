# -*- coding: utf-8 -*-
import sys
import logging
sys.path.append('../')

import numpy as np
from mic_py.mic_ilrma import ilrma
from scipy.stats import entropy
from base_sad import BaseSAD


from mic_py.mic_double_exp_averaging import double_exp_average
from mic_py.mic_make_mask import make_mask, mask_to_frames


class ILRMA_SAD(BaseSAD):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(ILRMA_SAD, self).__init__(mic_config, logger_flag, logger_level)

    def do_sad(self, in_wav_path, filter_cfg, **kwargs):
        threshold = kwargs.get('threshold', 0.0)
        logging.debug('\n threshold = {}'.format(threshold))

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        # 4 - Extract signals for ILRMA
        stft_all_arr_ilrma = np.concatenate(
            (np.transpose(stft_all, (2, 0, 1))[:, :, :7], np.transpose(stft_all, (2, 0, 1))[:, :, 11:18]),
            axis=-1)



        # 5 - Do ILRMA
        weights = np.load('weights14.npy')
        res = ilrma(stft_all_arr_ilrma, n_iter=10, n_components=2, W0=weights, seed=10)

        # 6 - Calculate entropy for each signal
        entr = np.zeros(res.shape[-1])
        for i in range(res.shape[-1]):
            for j in range(res.shape[1]):
                entr[i] += entropy(np.real(res[:, j, i] * np.conj(res[:, j, i])))

        # 7 - extract signal with the minimum entropy

        resulting_sig = self.get_istft(res[:, :, np.argmin(entr)])

        # 8 - Do exponential smothing over resulting signal

        average_sig = double_exp_average(resulting_sig, sr)
        average_sig[-300:] = average_sig[-300]
        average_sig[:300] = average_sig[300]

        # 9 - Make mask

        mask = make_mask(average_sig, percent_threshold=100)
        mask_frames = mask_to_frames(mask, int(self.mic_config.n_fft), int(self.mic_config.n_fft/self.mic_config.overlap) )



        return mask_frames