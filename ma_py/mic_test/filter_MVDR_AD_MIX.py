# -*- coding: utf-8 -*-
import sys
import math

sys.path.append('../')

from base_filter import BaseFilter

from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector


EXP_AV_COEF = 0.99
REG_COEF = 0.001
MVDR_TIME_STEP = 5
TIME_STEP = 5


class MVDR_AD_MIX_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(MVDR_AD_MIX_FILTER, self).__init__(mic_config, logger_flag, logger_level)

    def do_filter(self, in_wav_path, out_wav_path, filter_cfg, mic_cfg, **kwargs):

        # 1.0 - Read signal
        x_all_arr, sr = self.get_raw_wav(in_wav_path)

        # 2 - Do STFT
        stft_all = self.get_stft(x_all_arr)
        stft_mix = self.get_stft(x_all_arr[:, int(filter_cfg.end_noise_time) * sr:])

        # 3 - Calc  steering vector
        d_arr = self.get_steering(filter_cfg.angle_h, filter_cfg.angle_v, sr)

        # 4 - filter output
        (n_bins, n_sensors, n_frames) = stft_all.shape
        (n_bins_mix, n_sensors_mix, n_frames_mix) = stft_mix.shape

        frame_step = self.time_to_frame(TIME_STEP, sr, mic_cfg.n_fft, 2)

        EXP_AV_COEF = 1 - (1 / frame_step)

        res_spec = None
        psd = np.zeros(shape=(257, 66, 66), dtype=np.complex)

        mvdr_step = self.time_to_frame(MVDR_TIME_STEP, sr, mic_cfg.n_fft, 2)

        for i in range(0, n_frames_mix, mvdr_step):

            for j in range(min(mvdr_step, n_frames_mix - i - 1)):

                psd_curr = np.zeros((257, 66, 66), dtype=np.complex)
                for k in range(n_bins):
                    psd_curr[k] = np.outer(stft_mix[k, :, i + j], stft_mix[k, :, i + j].conj())

                psd = EXP_AV_COEF * psd + (1 - EXP_AV_COEF) * psd_curr

            psd_reg = psd + REG_COEF * np.identity(psd.shape[-1])

            w = get_mvdr_vector(d_arr.T, psd_reg)

            if i == 0:
                pp = apply_beamforming_vector(w, stft_mix[:, :, i:i + mvdr_step])
                res_spec = pp

            pp = apply_beamforming_vector(w, stft_mix[:, :, i + mvdr_step:i + 2 * mvdr_step])

            if res_spec is None:
                res_spec = pp
            else:
                res_spec = np.hstack((res_spec, pp))

        print('Result shape: {}'.format(res_spec.shape))

        # 5 - inverse STFT and save
        sig_out = self.get_istft(res_spec, overlap=2)

        self.write_result(out_wav_path, sig_out, sr)

    def time_to_frame(self, time, sr, n_fft, overlap):
        hop_size = n_fft // overlap
        return math.floor(time * sr / hop_size)
