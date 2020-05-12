# -*- coding: utf-8 -*-
import numpy as np
import soundfile as sf

import sys
import os
import re

sys.path.append('../')

from base_filter import BaseFilter

from mic_py.feats import stft, istft
from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.mic_zelin import cross_spectral, calc_beta, zelin_filter
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align


class MVDR_IBM_ZELIN_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(MVDR_IBM_ZELIN_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        wav_path_cut = in_wav_path[:-3]
        ds_noise, _ = sf.read(os.path.join(wav_path_cut, 'ds_mus.wav'))
        ds_speech, _ = sf.read(os.path.join(wav_path_cut, 'ds_spk.wav'))

        # 2 - Do STFT

        stft_mix = self.get_stft(x_all_arr[:, (np.int32)(filter_cfg.end_noise_time * sr):])

        # 2.1 - Do STFT one channel

        stft_noise = stft(ds_noise, mic_cfg.n_fft, 2)
        stft_speech = stft(ds_speech, mic_cfg.n_fft, 2)

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        # 4 - filter output
        IBM_mask = (abs(stft_noise) > abs(stft_speech)).astype(int).T

        psd = get_power_spectral_density_matrix(stft_mix[:, :, :-2], IBM_mask)

        psd = psd + 0.001 * np.identity(psd.shape[-1])

        w = get_mvdr_vector(d_arr.T, psd)

        result_mvdr_spec = apply_beamforming_vector(w, stft_mix)

        align_stft_arr = ds_align(stft_mix, d_arr.T)

        # 4.1 - Zelin postproc
        result_spec, H = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)
        print('Calc zelin filter output done!')

        result_spec = result_mvdr_spec * H

        # 5 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
