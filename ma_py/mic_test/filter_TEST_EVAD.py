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
from mic_py.feats import istft, stft

from mic_py_nn.features import preprocessing


np.set_printoptions(threshold=np.nan)


class TEST_EVAD_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(TEST_EVAD_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, spk_ds_wav_path, params=None,
                  out_wav_path_2=None):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        stft_mix = self.get_stft(x_all_arr[:, (np.int32)(filter_cfg.end_noise_time*sr):])

        bins, sensors, frames = stft_mix.shape

        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        ds_spk, _ = sf.read(spk_ds_wav_path)
        stft_spk = stft(ds_spk, fftsize=mic_cfg.n_fft, overlap=mic_cfg.overlap)
        if params is None:
            row_mask = np.ones(frames-2) - preprocessing.energy_mask(stft_spk)
        else:
            row_mask = np.ones(frames-2) - preprocessing.energy_mask(stft_spk, params[0], params[1])

        bins_mask = np.tile(row_mask, (bins, 1))
        stft_mix_masked = np.zeros(shape=(bins, sensors, frames-2), dtype=np.complex)
        print('Got mask! shape: {}'.format(bins_mask.shape))

        for i in range(sensors):
            stft_mix_masked[:, i, :] = stft_mix[:, i, :-2]*bins_mask

        noise_out = self.get_istft(stft_mix_masked[:, 0, :], overlap=2)

        # 4 - filter output
        psd = get_power_spectral_density_matrix(stft_mix_masked)
        psd = psd + 0.001 * np.identity(psd.shape[-1])

        w = get_mvdr_vector(d_arr.T, psd)

        result_spec = apply_beamforming_vector(w, stft_mix)

        # 5 - inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)

        if out_wav_path_2 is not None:
            sf.write(out_wav_path_2, noise_out, sr)
