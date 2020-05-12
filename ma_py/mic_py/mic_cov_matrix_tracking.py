import numpy as np
from mic_py import beamforming
from mic_py.mic_io import read_mic_wav_from_folder
from mic_py.beamforming import get_power_spectral_density_matrix, get_power_spectral_density_matrix2
from mic_py.mic_ds_beamforming import ds_beamforming
from mic_py import mic_geometry
from mic_py import mic_steering
from mic_py.feats import istft
from mic_py.mic_adaptfilt import spectral_substract_ref_psd_filter
from mic_py import feats
from mic_py import mic_gsc_spec_subs
from mic_py.beamforming import get_mvdr_vector
from mic_py import mic_gsc_griffiths


def cov_matrix_tracking(stft_arr_data_noise, stft_arr_data_mix, d_arr, filter_type='blocking_matrix', c_type='uniq',
                        bf_type='ds'):
    """
    Filter based on covariance matrix tracking
    (MAXIMUM LIKELIHOOD BASED NOISE COVARIANCE MATRIX ESTIMATION FOR MULTI-MICROPHONE SPEECH ENHANCEMENT)

    :stft_arr_data_noise:    - only noise part of specter for each sensors - shape (bins, num_sensors, frames)
    :stft_arr_data_mix:      - specter for each sensors - shape (bins, num_sensors, frames)
    :d_arr:                  - steering vector - shape (bins, num_sensors)
    :filter_type:            - algorithm for c coefficient estimation: 'griffiths'
                                                                       'blocking_matrix'
    :c_type:                 - type of c coefficient: 'uniq' - unique for every bin
                                                      'same' - one for all bins(mean for blocking matrix)
                                                      'bandpass'
    :bf_type:                 - type of beamforming:  'ds'   - delay sum
                                                      'mvdr'

    :return:
        ds_spec - result spectral  - shape (bins, frames)
    """
    mix_one_ch = ds_beamforming(stft_arr_data_mix, d_arr)

    # coefficient for exponential averaging
    alpha = 0.97

    if filter_type == 'blocking_matrix':
        psd_estimated = blocking_tracking(stft_arr_data_noise, stft_arr_data_mix, d_arr, c_type, alpha)
    elif filter_type == 'griffiths':
        psd_estimated = gsc_tracking(stft_arr_data_noise, stft_arr_data_mix, d_arr, c_type, alpha)

    psd_estimated = np.array(psd_estimated)
    frames, bins = psd_estimated.shape

    return spectral_substract_ref_psd_filter(mix_one_ch[:, :frames], np.array(psd_estimated).T, 0.1, 0.9)


def blocking_tracking(stft_arr_data_noise, stft_arr_data_mix, d_arr, c_type='uniq', alpha=0.9):

    _, mic_count = d_arr.shape
    bins, mic_count, frames = stft_arr_data_noise.shape
    coefficient = 1 / (mic_count - 1)
    steering_vector = d_arr

    steering_vector_norm = np.array([steering_vector[i] / np.sqrt(abs(np.inner(steering_vector[i],
                                                                               steering_vector[i].conj())))
                                     for i in range(257)])

    steering_vector_norm = steering_vector_norm[:, :, np.newaxis]
    steering_vector_norm_H = np.transpose(steering_vector_norm.conj(), axes=(0, 2, 1))

    steering_vector = steering_vector[:, :, np.newaxis]
    steering_vector_H = np.transpose(steering_vector, (0, 2, 1))
    # I - d^Hxd = [B(l) h(j)] (2)
    i_dh_d = np.array([(np.eye(mic_count, dtype=np.complex)) -
                       np.matmul(steering_vector_norm[i], steering_vector_norm_H[i]) /
                       np.inner(steering_vector_norm[i].squeeze(), steering_vector_norm_H[i].squeeze())
                       for i in range(257)])

    # get first n-1 columns
    b_matrix = i_dh_d[:, :, :-1]

    bins, dim_1, dim_2 = b_matrix.shape

    # orthonalization

    for fr in range(bins):
        buffer = np.zeros((mic_count - 1, mic_count), np.complex)
        for i in range(dim_2):
            a = b_matrix[fr, :, i]
            for j in range(i):
                b = b_matrix[fr, :, j]
                b_dot = np.inner(np.conj(b), b)
                ab_dot = np.inner(np.conj(b), a)
                a -= b * (ab_dot / b_dot)

            b_norm = np.sqrt(abs(np.inner(np.conj(a), a)))
            buffer[i] = a / b_norm
        b_matrix[fr] = buffer.T

    b_matrix_H = np.transpose(b_matrix.conj(), axes=(0, 2, 1))

    # estimate noise cov matrix
    F_l_0 = get_power_spectral_density_matrix(stft_arr_data_noise)

    # w = get_mvdr_vector(steering_vector.squeeze(),  F_l_0)
    # w = w[:, :, np.newaxis]
    # w_H = np.transpose(w.conj(), axes=(0, 2, 1))

    # B^H*F*B (4)
    bfb = np.einsum('...ik,...kj,...jm->...im', b_matrix_H, F_l_0, b_matrix)

    eps = 1e-12
    bfb_inv = np.linalg.inv(bfb + eps * np.identity(bfb.shape[-1]))
    # bfb_inv = np.linalg.inv(bfb)

    mix_bins, mix_sens, mix_frames = stft_arr_data_mix.shape

    V = stft_arr_data_mix[:, :, np.newaxis, :]
    Z = np.einsum('...ik,...kjn->...ijn', b_matrix_H, V)

    b, s, _, f = Z.shape

    psd_estimated = []

    c_uniq_blocking = np.zeros((mix_frames, bins))

    for i in range(0, mix_frames):

        z = Z[:, :, :, i]
        z_H = np.transpose(z.conj(), axes=(0, 2, 1))

        c_uniq = np.real(np.einsum('...ij,...jn,...nm->...im', z_H, bfb_inv, z).squeeze() * coefficient)

        if c_type == 'uniq':
            c_current = c_uniq
        elif c_type == 'same':
            c_current = np.ones(shape=bins) * np.mean(c_uniq)
        elif c_type == 'bandpass':
            c_current = np.ones(shape=bins)
            target_50 = np.mean(c_uniq[:50])
            target_100 = np.mean(c_uniq[50:100])
            target_150 = np.mean(c_uniq[100:])
            c_current[:50] = c_current[:50] * target_50
            c_current[50:100] = c_current[50:100]*target_100
            c_current[100:] = c_current[100:]*target_150
        else:
            raise Exception('{}: such type for c is not supported yet'.format(c_type))

        if i > 0:
            c_uniq_blocking[i] = alpha * c_uniq_blocking[i - 1] + (1 - alpha) * c_current
        else:
            c_uniq_blocking[i] = c_current

        F_new = np.zeros(F_l_0.shape)

        for kk in range(257):
            F_new[kk] = c_uniq_blocking[i][kk] * F_l_0[kk]

        current_psd = np.einsum('...ij,...jk,...km->...im', steering_vector_H, F_new, steering_vector).squeeze() \
                      / (mic_count ** 2)
        psd_estimated.append(current_psd)

    return psd_estimated


def gsc_tracking(stft_arr_data_noise, stft_arr_data_mix, d_arr, c_type='uniq', alpha=0.9):

    _, mic_count = d_arr.shape
    bins, mic_count, frames = stft_arr_data_noise.shape
    steering_vector = d_arr

    steering_vector = steering_vector[:, :, np.newaxis]
    steering_vector_H = np.transpose(steering_vector, (0, 2, 1))

    # estimate noise cov matrix
    F_l_0 = get_power_spectral_density_matrix(stft_arr_data_noise)

    # w = get_mvdr_vector(steering_vector.squeeze(),  F_l_0)
    # w = w[:, :, np.newaxis]
    # w_H = np.transpose(w.conj(), axes=(0, 2, 1))

    mix_bins, mix_sens, mix_frames = stft_arr_data_mix.shape

    Z_gr_init = mic_gsc_griffiths.gsc_griffiths_get_noise_specter(stft_arr_data_noise, steering_vector.squeeze())

    Z_griffiths = np.sum(np.einsum('...i,...i->...i', Z_gr_init, Z_gr_init.conj()), axis=1) / Z_gr_init.shape[1]

    Z_griffiths_power = np.sum(Z_griffiths)

    Z_griffiths_mix = mic_gsc_griffiths.gsc_griffiths_get_noise_specter(stft_arr_data_mix, steering_vector.squeeze())

    psd_estimated = []

    c_griffiths = np.zeros((mix_frames, bins))

    for i in range(0, mix_frames):

        buffer = np.real(Z_griffiths_mix[:, i] * (Z_griffiths_mix[:, i].conj()))
        buffer_power = np.sum(buffer)

        c_uniq = buffer / Z_griffiths
        if c_type == 'uniq':
            c_current = c_uniq
        elif c_type == 'same':
            c_current = np.ones(shape=bins) * (buffer_power / Z_griffiths_power)
        elif c_type == 'bandpass':
            c_current = np.ones(shape=bins)
            target_50 = np.mean(c_uniq[:50])
            target_100 = np.mean(c_uniq[50:100])
            target_150 = np.mean(c_uniq[100:])
            c_current[:50] = c_current[:50] * target_50
            c_current[50:100] = c_current[50:100] * target_100
            c_current[100:] = c_current[100:] * target_150
        else:
            raise Exception('{}: such type for c is not supported yet'.format(c_type))

        if i > 0:
            c_griffiths[i] = alpha * c_griffiths[i-1] + (1 - alpha) * c_current
        else:
            c_griffiths[i] = c_current

        F_new = np.zeros(F_l_0.shape)

        for kk in range(257):
            F_new[kk] = c_griffiths[i][kk] * F_l_0[kk]

        current_psd = np.einsum('...ij,...jk,...km->...im', steering_vector_H, F_new, steering_vector).squeeze() \
                      / mic_count**2
        psd_estimated.append(current_psd)

    return psd_estimated
