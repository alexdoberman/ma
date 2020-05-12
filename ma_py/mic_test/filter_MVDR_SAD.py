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
from mic_py.mic_mcra import mcra_filter


class MVDR_SAD_FILTER(BaseFilter):

    def __init__(self, mic_config, logger_flag=True, logger_level='DEBUG'):
        super(MVDR_SAD_FILTER, self).__init__(mic_config, logger_flag, logger_level)

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
        # 4 - Do SAD
        bins, num_sensors, frames = stft_mix.shape
        fft_hop_size = self.mic_config.n_fft / self.mic_config.overlap

        type_sad = kwargs.pop('type_sad', None)
        use_cmt = kwargs.pop('use_cmt', False)
        use_zelin = kwargs.pop('use_zelin', False)
        use_mcra = kwargs.pop('use_mcra', False)

        if type_sad == 'phase':
            frame_segm = phase_sad(stft_mix=stft_mix, d_arr_sp=d_arr.T, sr=sr, fft_hop_size=fft_hop_size,
                                   **kwargs)
        elif type_sad == 'ilrma':
            frame_segm = ilrma_sad(stft_mix, sr, self.mic_config.n_fft, self.mic_config.overlap)
        elif type_sad == 'all':
            frame_segm = np.ones((frames))
        else:
            assert False, 'type_sad = {} unsupported'.format(type_sad)

        noise_time = (np.sum((frame_segm)) *fft_hop_size) / sr
        print('MVDR_SAD_FILTER type = {} , detect noise only period = {} sec.'.format(type_sad, noise_time))

        #################################################################
        # Create mask by SAD segm
        (n_bins, n_sensors, n_frames) = stft_mix.shape
        mask = np.ones((n_bins, n_frames)) * frame_segm


        #################################################################
        # 4 - Filter output
        psd = get_power_spectral_density_matrix(stft_mix, mask = mask)
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
        psd = psd + 0.001 * np.identity(psd.shape[-1])
        w = get_mvdr_vector(d_arr.T, psd)
        result_spec = apply_beamforming_vector(w, stft_mix)


        if use_mcra:
            result_spec = mcra_filter(stft_arr=result_spec)

        if use_zelin:
            # 7 - Do align
            align_stft_arr = ds_align(stft_mix, d_arr.T)

            # 8 - Calc zelin filter output
            _, H = zelin_filter(stft_arr=align_stft_arr, alfa=0.7, alg_type=0)
            print('Calc zelin filter output done!')

            # 9 - Calc MVDR + Zelin filter output
            result_spec = result_spec * H


        #################################################################
        # 5 - Inverse STFT and save
        sig_out = self.get_istft(result_spec, overlap=2)
        self.write_result(out_wav_path, sig_out, sr)



