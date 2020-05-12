# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.append('../')

from base_filter import BaseFilter
from mic_py.mic_ds_beamforming import ds_beamforming
from mic_py.mic_null import null_filter
from mic_py.mic_lcmv_2 import lcmv_filter, lcmv_filter_delta, lcmv_filter_der, lcmv_filter_der_hor
from mic_py.mic_exp_steering import get_steering, get_der_steering_2
# from mic_py.mic_lcmv import lcmv_filter

from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector

from mic_py.mic_geometry import get_source_position, get_sensor_positions


class LCMV_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(LCMV_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)

        stft_noise = self.get_stft(x_all_arr[:,(np.int32)(filter_cfg.start_noise_time
                                                        *sr):(np.int32)(filter_cfg.end_noise_time*sr)])

        stft_mix = self.get_stft(x_all_arr[:,(np.int32)(filter_cfg.mix_start_time
                                                        *sr):(np.int32)(filter_cfg.mix_end_time*sr)])

        # 3 - Calc  steering vector

        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)
        d_arr_inf = self.get_steering(filter_cfg.angle_inf_h, filter_cfg.angle_inf_v, sr)

        sensor_positions = get_sensor_positions(self.mic_config.hor_mic_count, self.mic_config.vert_mic_count,
                                                dHor=self.mic_config.dHor, dVert=self.mic_config.dVert)

        source_position = get_source_position(filter_cfg.angle_inf_h, filter_cfg.angle_inf_v, radius=6.0)

        d_arr_h_der, d_arr_v_der = get_der_steering_2(sensor_positions, source_position, 512, 16000,
                                                      filter_cfg.angle_inf_h, filter_cfg.angle_inf_v)

        result_spec, _ = lcmv_filter_der(stft_mix=stft_mix, stft_noise=stft_noise, d_arr_sp=d_arr.T,
                                         d_arr_inf=d_arr_inf.T, d_arr_inf_der_hor=d_arr_h_der.T,
                                         d_arr_inf_der_vert=d_arr_v_der.T)

        # 5 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
