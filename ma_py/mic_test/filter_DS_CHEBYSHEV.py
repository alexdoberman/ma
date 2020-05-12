# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf
import math

import sys
sys.path.append('../')

from base_filter import BaseFilter
from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_gsc import *
from mic_py.mic_gsc_griffiths import gsc_griffiths_filter
from mic_py.mic_gsc_spec_subs import *
from mic_py.mic_zelin import cross_spectral, calc_beta, zelin_filter
from mic_py.mic_chebyshev import chebyshev_weights
from mic_py.mic_chebyshev import get_chebyshev_weights_for_amplitudes
from calc_metric import *



class DS_CHEBYSHEV_FILTER(BaseFilter):
    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(DS_CHEBYSHEV_FILTER, self).__init__(mic_config, logger_flag, logger_level)


    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        (bins, num_sensors, frames) = stft_all.shape

        ##################################################################
        # 3 Calc Chebyshev weights

        weights = get_chebyshev_weights_for_amplitudes(self.mic_config.vert_mic_count,
                                                       self.mic_config.hor_mic_count,
                                                       bins)

        ##################################################################
        # 4 - Calc  steering vector multiplied by Chebyshev weights

        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)


        d_arr_cheb = d_arr * weights
        d_arr_ds_cheb = d_arr_cheb
        d_arr_ds_cheb[:, 0:20] = d_arr[:, 0:20] # DS+Chebyshev combination

        ##################################################################
        # 5 - Calc filter output
        # Chebyshev+DS filter
        result_spec = ds_beamforming(stft_all, d_arr_ds_cheb.T)


        # result_spec[int(800/8000*257):257, :] = 0
        # result_spec[0:5, :] = 0

        ##################################################################
        # 6 inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)

