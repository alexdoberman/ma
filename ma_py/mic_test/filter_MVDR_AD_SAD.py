# -*- coding: utf-8 -*-
import sys
import math
from tqdm import tqdm

sys.path.append('../')

from base_filter import BaseFilter

from mic_py.mic_gsc_spec_subs import *
from mic_py.beamforming import get_mvdr_vector, apply_beamforming_vector
from mic_py.mic_cov_marix_taper import get_taper
from mic_py.mic_phase_sad import phase_sad


EXP_AV_COEF = 0.99
REG_COEF = 0.001
MVDR_TIME_STEP = 5
TIME_STEP = 5


class MVDR_AD_SAD_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(MVDR_AD_SAD_FILTER, self).__init__(mic_config, logger_flag, logger_level)

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
        use_taper = True

        res_spec = None
        psd = np.zeros(shape=(257, 66, 66), dtype=np.complex)

        mvdr_step = self.time_to_frame(MVDR_TIME_STEP, sr, mic_cfg.n_fft, 2)

        hop_size = mic_cfg.n_fft / mic_cfg.overlap

        frame_segm = phase_sad(stft_mix=stft_mix, d_arr_sp=d_arr.T, sr=sr, fft_hop_size=hop_size,
                               **kwargs)
        mask = np.ones((n_bins_mix, n_frames_mix)) * frame_segm

        print('get mask')
        stft_mix_noise = copy.deepcopy(stft_mix)
        # stft_mix_noise = copy.deepcopy(stft_mix[:,:,0:-2])

        (n_bins_mix, n_sensors_mix, n_frames_mix) = stft_mix.shape

        for i in range(0, n_sensors):
            stft_mix_noise[:, i, :] *= mask

        for i in tqdm(range(0, n_frames_mix, mvdr_step)):
            noise_time = (np.sum((frame_segm[i: i+mvdr_step])) * hop_size) / sr
            print('Current noise: {}'.format(noise_time))
            for j in range(min(mvdr_step, n_frames_mix - i - 1)):

                psd_curr = np.zeros((257, 66, 66), dtype=np.complex)
                for k in range(n_bins):
                    psd_curr[k] = np.outer(stft_mix_noise[k, :, i + j], stft_mix_noise[k, :, i + j].conj())

                if use_taper:
                    taper = get_taper(hor_mic_count=mic_cfg.hor_mic_count,
                                      vert_mic_count=mic_cfg.vert_mic_count,
                                      dHor=mic_cfg.dHor,
                                      dVert=mic_cfg.dVert,
                                      angle_h=filter_cfg.angle_h,
                                      angle_v=filter_cfg.angle_v,
                                      sr=sr,
                                      fft_size=mic_cfg.n_fft,
                                      bandwidth=0.5)
                    for k in range(n_bins):
                        psd[k, :, :] = np.multiply(psd[k, :, :], taper[k])

                psd = EXP_AV_COEF * psd + (1 - EXP_AV_COEF) * psd_curr

            psd_reg = psd + REG_COEF * np.identity(psd.shape[-1])

            w = get_mvdr_vector(d_arr.T, psd_reg)

            if i == 0:
                psd_start = np.array([np.eye(66) for _ in range(257)])
                w_start = get_mvdr_vector(d_arr.T, psd_start)
                pp = apply_beamforming_vector(w_start, stft_mix[:, :, i:i + mvdr_step])
                # pp = ds_beamforming(stft_mix[:, :, i:i + mvdr_step], d_arr.T)
                res_spec = pp

            pp = apply_beamforming_vector(w, stft_mix[:, :, i + mvdr_step:i + 2 * mvdr_step])

            # if res_spec is None:
            #    res_spec = pp
            # else:
            res_spec = np.hstack((res_spec, pp))

        print('Result shape: {}'.format(res_spec.shape))

        # 5 - inverse STFT and save
        sig_out = self.get_istft(res_spec, overlap=2)

        self.write_result(out_wav_path, sig_out, sr)

    def time_to_frame(self, time, sr, n_fft, overlap):
        hop_size = n_fft // overlap
        return math.floor(time * sr / hop_size)