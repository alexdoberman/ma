# -*- coding: utf-8 -*-
import sys

sys.path.append('../')

from base_filter import BaseFilter

from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix


class MVDR_MIX_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(MVDR_MIX_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        stft_mix = self.get_stft(x_all_arr[:, (np.int32)(filter_cfg.end_noise_time*sr):])

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        # 4 - filter output
        psd = get_power_spectral_density_matrix(stft_mix)
        psd = psd + 0.001 * np.identity(psd.shape[-1])

        w = get_mvdr_vector(d_arr.T, psd)

        result_spec = apply_beamforming_vector(w, stft_mix)

        # 5 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
