# -*- coding: utf-8 -*-
from base_filter import BaseFilter

from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.mic_cov_matrix_tracking import cov_matrix_tracking
from mic_py.beamforming import get_power_spectral_density_matrix

import numpy as np


class COV_MATRIX_TRACKING_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):

        super(COV_MATRIX_TRACKING_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        stft_noise = self.get_stft(x_all_arr[:, int(filter_cfg.start_noise_time) * sr:int(filter_cfg.end_noise_time)
                                                                                      * sr])
        stft_mix = self.get_stft(x_all_arr[:, int(filter_cfg.end_noise_time) * sr:])

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        psd_noise_matrix = get_power_spectral_density_matrix(stft_noise)
        psd_noise_matrix = psd_noise_matrix + 0.01 * np.identity(psd_noise_matrix.shape[-1])

        w = get_mvdr_vector(d_arr.T, psd_noise_matrix)
        w *= self.mic_config.mic_count

        # 4 - Calc filter output
        result_spec = cov_matrix_tracking(stft_noise, stft_mix, w, filter_type='blocking_matrix')

        # 5 inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)

        # sf.write(os.path.join('temp', in_wav_path.replace('\\', '_').replace('/', '_') + '.wav'), sig_out, sr)
        self.write_result(out_wav_path, sig_out, sr)

