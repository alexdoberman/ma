# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

from base_filter import BaseFilter
from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector


class MVDR_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(MVDR_FILTER, self).__init__(mic_config, logger_flag, logger_level)

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

        # 4  - Calc psd matrix
        psd_noise_matrix = get_power_spectral_density_matrix(stft_noise_arr)

        # 5 - Regularisation psd matrix
        psd_noise_matrix = psd_noise_matrix + 0.01 * np.identity(psd_noise_matrix.shape[-1])

        # 6 - Apply MVDR
        w = get_mvdr_vector(d_arr.T, psd_noise_matrix)
        result_spec = apply_beamforming_vector(w, stft_all)

        # 7 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
