# -*- coding: utf-8 -*-
import sys
import os
import soundfile as sf

sys.path.append('../')

from base_filter import BaseFilter

from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.mic_ds_beamforming import ds_beamforming, ds_align
from mic_py.feats import istft
from mic_py.mic_cov_marix_taper import get_taper

from mic_py_nn.mains.sample_predict import MADPredict


np.set_printoptions(threshold=np.nan)


class MVDR_MIX_MAD_MASK_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(MVDR_MIX_MAD_MASK_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        out_wav_path_2 = kwargs.get('out_wav_path_2', None)

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        stft_mix = self.get_stft(x_all_arr[:, (np.int32)(filter_cfg.end_noise_time*sr):])

        bins, sensors, frames = stft_mix.shape

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        # 4 - DS filter
        result_spec = ds_beamforming(stft_mix, d_arr.T)

        # 5 - inverse STFT and save
        sig_out = istft(result_spec.transpose((1, 0)), overlap=2)

        out_ds_path = r'{}/tmp_ds.wav'.format(os.path.dirname(out_wav_path))
        sf.write(out_ds_path, sig_out, sr)

        # out_ds_path = r'{}/ds_mus.wav'.format(os.path.dirname(out_wav_path))
        # config_path = r'/home/superuser/MA_ALG/datasets/test_ma/mad_td_cnn/mad.json'
        # in_model_path = r'/home/superuser/MA_ALG/datasets/test_ma/mad_td_cnn/checkpoint/'

        config_path = r'/home/stc/MA_ALG/datasets/test_ma/mad_work!!/mad.json'
        in_model_path = r'/home/stc/MA_ALG/datasets/test_ma/mad_work!!/checkpoint/'
        model_name = ''

        # mask = MADPredict(config_path, in_model_path).predict_mask(out_ds_path)
        mask = MADPredict(config_path, in_model_path, model_name).predict_mask(out_ds_path)
        # print(mask)
        threshold = 0.5
        print(mask)

        mask = (mask < threshold).astype(int)
        print(np.count_nonzero(mask))
        # print(mask.shape)

        bins_mask = np.tile(mask, (bins, 1))
        stft_mix_masked = np.zeros(shape=(bins, sensors, frames-2), dtype=np.complex)
        print('Got mask! shape: {}'.format(bins_mask.shape))

        for i in range(sensors):
            stft_mix_masked[:, i, :] = stft_mix[:, i, :-2]*bins_mask

        noise_out = self.get_istft(stft_mix_masked[:, 0, :], overlap=2)

        # 4 - filter output
        psd = get_power_spectral_density_matrix(stft_mix_masked)
        psd = psd + 0.001 * np.identity(psd.shape[-1])
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

        w = get_mvdr_vector(d_arr.T, psd)

        result_spec = apply_beamforming_vector(w, stft_mix)

        # 5 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)

        if out_wav_path_2 is not None:
            sf.write(out_wav_path_2, noise_out, sr)
