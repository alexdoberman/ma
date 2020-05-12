# -*- coding: utf-8 -*-
import numpy as np
from mic_py.mic_mn import MNBeamformer

def batch_maximum_negentropy_filter(stft_arr, d_arr, batch_size = 300, alpha = 0.01, normalise_wa = False, max_iter = 10):
    """

    :param stft_arr: - spectr for each sensors - shape (bins, num_sensors, frames)
    :param d_arr: - steering vector         - shape (bins, num_sensors)
    :param alpha:
    :param normalise_wa:
    :return:
        result_spec - result spectral  - shape (bins, frames)
    """
    (n_bins, n_num_sensors, n_frames) = stft_arr.shape

    stft_out = np.zeros((n_bins, n_frames), dtype=np.complex)

    speech_distribution_coeff_path = r'mic_utils\alg_data\gg_params_freq_f_scale.npy'
    MN_filter = MNBeamformer(stft_arr, speech_distribution_coeff_path=speech_distribution_coeff_path, alpha=alpha,
                             normalise_wa=normalise_wa)
    MN_filter.set_steering_vector(d_arr)

    batch_count = (int)(n_frames/batch_size) + 1

    for batch_num in range(0, batch_count):
        start_frame = batch_num*batch_size
        end_frame   = min(start_frame + batch_size, n_frames)

        MN_filter.accum_observations(start_frame=start_frame, end_frame=end_frame)

        for freq_ind in range(0, n_bins):

            # do filtering only speech freq
            if freq_ind in range(2, n_bins - 5):
                wa_res = MN_filter.estimate_active_weights(freq_ind, max_iter=max_iter)
                print("batch_num = {} freq_ind = {}  wa_res = {}".format(batch_num, freq_ind, wa_res))

            stft_out[freq_ind, start_frame:end_frame] = MN_filter.calc_output(freq_ind)

    return stft_out