# -*- coding: utf-8 -*-
import sys
import numpy as np
sys.path.append('../')

from base_filter import BaseFilter

from mic_py.mic_cov_marix_taper import cov_matrix_tapper_linear_array, cov_matrix_tapper_mean_steering, \
    cov_matrix_tapper_interf_steering, cov_matrix_tapper_bandwidth, cov_matrix_tapper_interf_steering_bandwidth
from mic_py.mic_exp_steering import get_steering_linear_array

from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector


class CMT_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(CMT_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)
        noise_arr = x_all_arr[:, (np.int32)(filter_cfg.start_noise_time *
                                            sr):(np.int32)(filter_cfg.end_noise_time * sr)]

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        stft_noise_arr = self.get_stft(noise_arr)

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)
        # d_arr = get_steering_linear_array(filter_cfg.angle_h, mic_cfg.dHor, 11, sr, 512)

        # 4  - Calc psd matrix
        # psd_noise_matrix = cov_matrix_tapper_linear_array(stft_noise_arr, mic_count=mic_cfg.hor_mic_count,
        #                                                  d=mic_cfg.dHor, angle=filter_cfg.angle_h, sr=16000)
        '''
        psd_noise_matrix = cov_matrix_tapper_mean_steering(stft_noise_arr, mic_cfg.hor_mic_count,
                                                           mic_cfg.vert_mic_count, dHor=mic_cfg.dHor,
                                                           dVert=mic_cfg.dVert, angle_v=filter_cfg.angle_v,
                                                           angle_h=filter_cfg.angle_h, sr=sr)

        '''
        '''
        psd_noise_matrix = cov_matrix_tapper_interf_steering(mic_cfg.hor_mic_count,
                                                             mic_cfg.vert_mic_count, dHor=mic_cfg.dHor,
                                                             dVert=mic_cfg.dVert, angle_v=filter_cfg.angle_inf_v,
                                                             angle_h=filter_cfg.angle_inf_h, sr=sr)
        '''

        '''
        psd_noise_matrix, _ = cov_matrix_tapper_interf_steering_bandwidth(mic_cfg.hor_mic_count,
                                                                       mic_cfg.vert_mic_count, dHor=mic_cfg.dHor,
                                                                       dVert=mic_cfg.dVert,
                                                                       angle_v=filter_cfg.angle_inf_v,
                                                                       angle_h=filter_cfg.angle_inf_h,
                                                                       angle_inf_h=filter_cfg.angle_inf_h,
                                                                       angle_inf_v=filter_cfg.angle_inf_v,
                                                                       sr=sr)
        '''

        psd_noise_matrix, _ = cov_matrix_tapper_bandwidth(stft_noise_arr, mic_cfg.hor_mic_count,
                                                          mic_cfg.vert_mic_count, dHor=mic_cfg.dHor,
                                                          dVert=mic_cfg.dVert, angle_v=filter_cfg.angle_v,
                                                          angle_h=filter_cfg.angle_h, sr=sr)
        # 5 - Regularisation psd matrix
        # psd_noise_matrix = get_power_spectral_density_matrix(stft_noise_arr)
        # psd_noise_matrix = psd_noise_matrix + 0.001 * np.identity(psd_noise_matrix.shape[-1])

        # 6 - Apply MVDR
        w = get_mvdr_vector(d_arr.T, psd_noise_matrix)
        result_spec = apply_beamforming_vector(w, stft_all)

        # 7 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
