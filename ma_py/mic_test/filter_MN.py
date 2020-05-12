# -*- coding: utf-8 -*-
import numpy as np

import sys
sys.path.append('../')

from base_filter import BaseFilter

from mic_py.mic_mn import maximum_negentropy_filter, maximum_negentropy_filter_ex
from mic_py.mic_zelin import zelin_filter_ex

from calc_metric import *


class MN_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(MN_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        # 4 - Calc MN filter output
        do_zelin_postproc = False
        alpha = 0.1
        beta = 1.0
        normalise_wa = False
        max_iter = 30
        speech_distribution_coeff_path = r'../mic_utils/alg_data/clean_ru_speech_gg_params_freq_f_scale.npy'

        _mix_start_frame = (np.int32)(filter_cfg.mix_start_time * sr / (mic_cfg.n_fft / mic_cfg.overlap))
        _mix_end_frame = (np.int32)(filter_cfg.mix_end_time * sr / (mic_cfg.n_fft / mic_cfg.overlap)) - 1

        result_spec = maximum_negentropy_filter_ex(stft_arr=stft_all, d_arr=d_arr, start_frame=_mix_start_frame,
                                                   end_frame=_mix_end_frame, alpha=alpha, beta=beta,
                                                   normalise_wa=normalise_wa, max_iter=max_iter,
                                                   speech_distribution_coeff_path=speech_distribution_coeff_path)

        if do_zelin_postproc:
            # 4.1 - Do zelin filter
            alpha_zelin = 0.7
            _, H = zelin_filter_ex(stft_arr=stft_all, d_arr=d_arr, alfa=alpha_zelin, alg_type=0)

            # 4.2 - Calc MN + Zelin filter output
            result_spec = result_spec * H

        # 5 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
