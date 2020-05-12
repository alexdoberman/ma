# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

from base_filter import BaseFilter
from mic_py.mic_hard_null import hard_null_filter, hard_null_filter_time_domain


class HARD_NULL_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(HARD_NULL_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        #################################################################
        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        #################################################################
        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)

        #################################################################
        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)
        d_arr_inf = self.get_steering(filter_cfg.angle_inf_h, filter_cfg.angle_inf_v, sr)

        #################################################################
        # 4 - HARD NULL filter

        # 0-compensate ref, 1-spec subs, 2- SMB filter
        # 3-LMS,4-NML,5-AP
        alg_type = kwargs.pop('alg_type', 1)
        time_domain_filter = kwargs.pop('time_domain_filter', False)

        if time_domain_filter:
            #################################################################
            # 4 - HARD NULL filter output
            sig_out = hard_null_filter_time_domain(stft_all, d_arr_sp=d_arr, d_arr_inf=d_arr_inf,
                                                   alg_type=alg_type, overlap=self.mic_config.overlap)


            #################################################################
            # 5 - Save result
            self.write_result(out_wav_path, sig_out, sr)
        else:

            #################################################################
            # 4 - HARD NULL filter output
            result_spec = hard_null_filter(stft_all, d_arr_sp=d_arr, d_arr_inf=d_arr_inf, alg_type=alg_type)

            #################################################################
            # 5 - inverse STFT and save
            sig_out = self.get_istft(result_spec, overlap=self.mic_config.overlap)
            self.write_result(out_wav_path, sig_out, sr)
