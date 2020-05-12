# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.append('../')

from base_filter import BaseFilter
from mic_py.mic_ds_beamforming import ds_beamforming
from mic_py.mic_exp_steering import get_steering_linear_array, get_steering


class EXP_STEERING_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(EXP_STEERING_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        d_arr_2 = get_steering(hor_angle=filter_cfg.angle_h, vert_angle=filter_cfg.angle_v)

        print('Norm: {}'.format(np.linalg.norm(d_arr-d_arr_2)))

        # DS filter
        result_spec = ds_beamforming(stft_all, d_arr_2.T)

        # 5 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)