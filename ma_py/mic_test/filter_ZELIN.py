# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

from base_filter import BaseFilter
from mic_py.mic_ds_beamforming import ds_align
from mic_py.mic_zelin import zelin_filter

from calc_metric import *


class ZELIN_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(ZELIN_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        # Zelin filter
        align_stft_arr = ds_align(stft_all, d_arr.T)
        result_spec, _ = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)

        # 5 inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
