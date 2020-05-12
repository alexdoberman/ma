# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import math

import sys
sys.path.append('../')

from base_filter import BaseFilter

from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_chebyshev import chebyshev_weights
from mic_py.mic_chebyshev import chebyshev_weights_zeros
from mic_py.mic_chebyshev import get_chebyshev_weights_for_nulls

from calc_metric import *

class CHEBYSHEV_NULLS_FILTER(BaseFilter):
    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(CHEBYSHEV_NULLS_FILTER, self).__init__(mic_config, logger_flag, logger_level)


    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        (bins, num_sensors, frames) = stft_all.shape

        ##################################################################
        # 3 Calc Chebyshev weights

        weights = get_chebyshev_weights_for_nulls(self.mic_config.vert_mic_count,
                                                  self.mic_config.hor_mic_count,
                                                  self.mic_config.dHor,
                                                  self.mic_config.dVert,
                                                  bins,
                                                  filter_cfg.angle_inf_h)

        #################################################################
        # 4 - Calc  steering vector multiplied by Chebyshev weights

        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        d_arr_cheb = d_arr * weights

        ##################################################################
        # 5 - Calc filter output
        # Chebyshev+DS filter
        result_spec = ds_beamforming(stft_all, d_arr_cheb.T)

        # result_spec[int(800/8000*257):257, :] = 0
        # result_spec[0:5, :] = 0

        ##################################################################
        # 6 inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)

