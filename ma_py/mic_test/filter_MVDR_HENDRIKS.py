# -*- coding: utf-8 -*-
import sys

sys.path.append('../')

from base_filter import BaseFilter

from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.mic_phase_sad import phase_sad
from mic_py.mic_ilrma_sad import ilrma_sad
from mic_py.mic_cov_marix_taper import get_taper
from mic_py.mic_zelin import zelin_filter
from mic_py.mic_ds_beamforming import ds_align
from mic_py.mic_hendriks_psd_estim import estimate_psd_hendriks


class MVDR_HENDRIKS_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(MVDR_HENDRIKS_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        #################################################################
        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        #################################################################
        # 2 - Do STFT
        stft_mix = self.get_stft(x_all_arr[:, (np.int32)(filter_cfg.end_noise_time*sr):])

        #################################################################
        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        #################################################################
        # 4 - Do MVDR + HENDRIKS
        bins, num_sensors, frames = stft_mix.shape
        fft_hop_size = self.mic_config.n_fft / self.mic_config.overlap

        use_cmt = kwargs.pop('use_cmt', False)
        use_zelin = kwargs.pop('use_zelin', False)
        reg_const_hendriks = kwargs.pop('reg_const_hendriks', 0.1)
        reg_const_mvdr = kwargs.pop('reg_const_mvdr', 0.01)

        #################################################################
        # 5 - Estimate HENDRIKS PSD matrix
        psd = estimate_psd_hendriks(stft_mix=stft_mix, d_arr_sp=d_arr.T, reg_const=reg_const_hendriks)

        #################################################################
        # 6 - Use Tapper
        if use_cmt:
            T_matrix = get_taper(hor_mic_count=self.mic_config.hor_mic_count,
                                     vert_mic_count=self.mic_config.vert_mic_count,
                                     dHor=self.mic_config.dHor,
                                     dVert=self.mic_config.dVert,
                                     angle_h=filter_cfg.angle_h,
                                     angle_v=filter_cfg.angle_v,
                                     sr=sr,
                                     fft_size=self.mic_config.n_fft,
                                     bandwidth=0.5)
            for i in range(bins):
                psd[i, :, :] = np.multiply(psd[i, :, :], T_matrix[i])

        # Regularisation
        psd = psd + reg_const_mvdr * np.identity(psd.shape[-1])
        w = get_mvdr_vector(d_arr.T, psd)
        result_spec = apply_beamforming_vector(w, stft_mix)

        #################################################################
        # 7 - Use Zelin

        if use_zelin:
            # 7.1 - Do align
            align_stft_arr = ds_align(stft_mix, d_arr.T)

            # 7.2 - Calc zelin filter output
            _, H = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)
            print('Calc zelin filter output done!')

            # 7.3 - Calc MVDR + Zelin filter output
            result_spec = result_spec * H

        #################################################################
        # 8 - Inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)



