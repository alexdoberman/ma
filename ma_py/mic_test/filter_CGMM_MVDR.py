# -*- coding: utf-8 -*-
import sys
sys.path.append('../')

from base_filter import BaseFilter

from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.beamforming import get_mvdr_vector ,apply_beamforming_vector
from mic_py.mic_cgmm import est_cgmm, permute_mask, est_cgmm_ex


class CGMM_MVDR_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(CGMM_MVDR_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        num_iters = 2

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        stft_mix = self.get_stft(x_all_arr[:, (np.int32)(filter_cfg.end_noise_time * sr):])

        (n_bins, n_sensors, n_frames) = stft_all.shape

        # 3 - Calc  steering vector
        d_arr_sp = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)
        d_arr_noise = self.get_steering(filter_cfg.angle_inf_h, filter_cfg.angle_inf_v, sr)

        psd_sp = np.einsum('i...,j...->...ij', d_arr_sp, d_arr_sp.conj())
        psd_noise = np.einsum('i...,j...->...ij', d_arr_noise, d_arr_noise.conj())

        # 4 - Calc cgmm mask
        print('Estimate CGMM')

        mask, R = est_cgmm_ex(stft_mix, psd_sp, psd_noise, num_iters=num_iters, allow_cov_update=False)

        stft_all_noise = copy.deepcopy(stft_mix)
        for i in range(0, n_sensors):
            stft_all_noise[:, i, :] *= mask[0, :, :]

        psd_noise_matrix = get_power_spectral_density_matrix(stft_all_noise)

        # 5 -Regularisation psd matrix
        psd_noise_matrix = psd_noise_matrix + 0.01 * np.identity(psd_noise_matrix.shape[-1])

        # 6 - Apply MVDR
        w = get_mvdr_vector(d_arr_sp.T, psd_noise_matrix)
        result_spec = apply_beamforming_vector(w, stft_all)

        # 5 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
