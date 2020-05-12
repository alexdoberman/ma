# -*- coding: utf-8 -*-
import sys

sys.path.append('../')
sys.path.append('../../svn_MA_PY_NN')

from base_filter import BaseFilter

import os
import soundfile as sf
from mic_py.feats import istft
from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.mic_zelin import cross_spectral, calc_beta, zelin_filter

from mic_py_nn.mains.sample_predict import UnetPredict


class UNET_MVDR_MIX_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(UNET_MVDR_MIX_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        stft_mix = self.get_stft(x_all_arr[:, int(filter_cfg.end_noise_time) * sr:])

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_inf_h, filter_cfg.angle_inf_v, sr)

        # 4 - DS filter
        result_spec = ds_beamforming(stft_mix, d_arr.T)

        # 5 - inverse STFT and save
        sig_out = istft(result_spec.transpose((1, 0)), overlap=2)
        out_ds_path = r'{}/tmp_ds.wav'.format(os.path.dirname(out_wav_path))
        sf.write(out_ds_path, sig_out, sr)

        # 6 - get noise mask
        print('Get noise mask form neural net!')

        # shape - (time, freq, 2)
        config_path = r'/home/stc/MA_ALG/datasets/test_ma/unet_v12_f/unet.json'
        in_model_path = r'/home/stc/MA_ALG/datasets/test_ma/unet_v12_f/checkpoint/'

        mask = UnetPredict(config_path, in_model_path).predict_mask(out_ds_path)

        # 7  - Calc psd matrix

        print('Calc psd matrix!')
        print('     mask.shape      = {}'.format(mask.shape))
        print('     stft_mix.shape  = {}'.format(stft_mix.shape))

        stft_mix_noise = copy.deepcopy(stft_mix[:, :, 0:-2])

        psd_noise_matrix = get_power_spectral_density_matrix(stft_mix_noise, mask)

        # 5 - Regularisation psd matrix
        psd_noise_matrix = psd_noise_matrix + 0.001 * np.identity(psd_noise_matrix.shape[-1])

        # 6 - Apply MVDR
        w = get_mvdr_vector(d_arr.T, psd_noise_matrix)
        result_mvdr_spec = apply_beamforming_vector(w, stft_mix)

        # 7 - Apply Zelin filter
        align_stft_arr = ds_align(stft_mix, d_arr.T)
        _, H = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)
        print('Calc zelin filter output done!')
        result_spec = result_mvdr_spec * H

        # 8 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)
