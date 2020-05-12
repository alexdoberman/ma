# -*- coding: utf-8 -*-
import soundfile as sf

import sys

sys.path.append('../')
sys.path.append('../../svn_MA_PY_NN')

from base_filter import BaseFilter

import os
from mic_py.feats import istft
from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_zelin import cross_spectral, calc_beta, zelin_filter

from mic_py_nn.mains.sample_predict import ChimeraPredict


class NN_MVDR2_MIX_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(NN_MVDR2_MIX_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        stft_mix = self.get_stft(x_all_arr[:, (np.int32)(filter_cfg.end_noise_time*sr):])

        (n_bins, n_sensors, n_frames) = stft_all.shape

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        # 4 - DS or MVDR filter
        psd = get_power_spectral_density_matrix(stft_mix)
        psd = psd + 0.001 * np.identity(psd.shape[-1])
        w = get_mvdr_vector(d_arr.T, psd)
        result_spec = apply_beamforming_vector(w, stft_mix)

        # result_spec = ds_beamforming(stft_mix, d_arr.T)

        # 5 - inverse STFT and save
        sig_out = istft(result_spec.transpose((1, 0)), overlap=2)
        out_ds_path = r'{}/tmp_mvdr.wav'.format(os.path.dirname(out_wav_path))
        sf.write(out_ds_path, sig_out, sr)

        # 6 - get noise mask
        print('Get noise mask form neural net!')

        # shape - (time, freq, 2)
        config_path = r'/home/stc/MA_ALG/datasets/test_ma/8_chimera_r09_em_30_a05_ctx_100_tanh_snr_3_size_4x500.json'
        in_model_path = r'/home/stc/MA_ALG/datasets/test_ma/checkpoint/'
        mask = ChimeraPredict(config_path, in_model_path).predict_mask(out_ds_path, os.path.dirname(out_wav_path))

        # 7  - Calc psd matrix
        # bin x frames x mask
        mask = np.transpose(mask, (1, 0, 2))

        print('Calc psd matrix!')
        print('     mask.shape      = {}'.format(mask.shape))
        print('     stft_mix.shape  = {}'.format(stft_mix.shape))

        # TODO in deep clustering need choice mask
        actual_mask = mask[:, :, 1]

        # stft_mix_noise = copy.deepcopy(stft_mix)
        stft_mix_noise = copy.deepcopy(stft_mix[:, :, 0:-2])
        for i in range(0, n_sensors):
            stft_mix_noise[:, i, :] *= actual_mask
        psd_noise_matrix = get_power_spectral_density_matrix(stft_mix_noise)

        #################################################################
        # 8 -Regularisation psd matrix
        psd_noise_matrix = psd_noise_matrix + 0.001 * np.identity(psd_noise_matrix.shape[-1])

        #################################################################
        # 9 - Apply MVDR
        w = get_mvdr_vector(d_arr.T, psd_noise_matrix)
        result_mvdr_spec = apply_beamforming_vector(w, stft_mix)

        #################################################################
        # 10 - Apply Zelin filter
        align_stft_arr = ds_align(stft_mix, d_arr.T)
        _, H = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)
        print('Calc zelin filter output done!')
        result_spec = result_mvdr_spec * H

        # 11 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
