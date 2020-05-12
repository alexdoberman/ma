import numpy as np
from mic_py.beamforming import get_power_spectral_density_matrix
from mic_py.mic_ds_beamforming import ds_beamforming


def frame_cross_psd(v1, v2):
    return v1*v2


def rtf_filter(stft_arr_data_mix, stft_arr_data_sp, filter_type='simple'):
    """

    :param stft_arr_data_mix:   -  spectrum for each sensors, noise + speech period  - shape(bins, num_sensors, frames)
    :param stft_arr_data_sp:    -  spectrum for each sensors, only speech period     - shape(bins, num_sensors, frames)
    :param filter_type:         -  possible types: 'simple'
                                                   'gannot'
    :return result_spec:        -  shape(bins, frames)
    """
    possible_types = ['simple', 'gannot']
    assert filter_type in possible_types, 'Such type is not supported!'

    if filter_type == 'simple':
        rtf_arr = tf_ratio_simple(stft_arr_data_sp)
    elif filter_type == 'gannot':
        rtf_arr = tf_ratio_gannot(stft_arr_data_sp)

    result_spec = ds_beamforming(stft_arr_data_mix, rtf_arr.T)
    return result_spec


def rtf_vector(stft_arr_data_sp, filter_type='simple'):
    """

    :param stft_arr_data_sp:    -  spectrum for each sensors, only speech period     - shape(bins, num_sensors, frames)
    :param filter_type:         -  possible types: 'simple'
                                                   'gannot'
    :return result_spec:        -  shape(bins, frames)
    """
    possible_types = ['simple', 'gannot']
    assert filter_type in possible_types, 'Such type is not supported!'

    if filter_type == 'simple':
        rtf_arr = tf_ratio_simple(stft_arr_data_sp)
    elif filter_type == 'gannot':
        rtf_arr = tf_ratio_gannot(stft_arr_data_sp)

    return rtf_arr


def tf_ratio_gannot(stft_arr):
    """

    :param stft_arr:   -  input spectrum -  shape(bins, sensors, frames)
    :return h_arr:     -  shape=(sensors, bins)
    """
    bins, sensors, frames = stft_arr.shape
    h_arr = np.zeros(shape=(bins, sensors))

    h_arr[:, 0] = np.ones(shape=(bins), dtype=np.complex)

    f11_vec = frame_cross_psd(stft_arr[:, 0, :], stft_arr[:, 0, :].conj())
    denominator = np.mean(f11_vec ** 2) - np.mean(f11_vec)**2
    for i in range(1, sensors):
        fi1_vec = frame_cross_psd(stft_arr[:, i, :], stft_arr[:, 0, :].conj())
        numerator = np.mean(f11_vec * fi1_vec) - np.mean(f11_vec)*np.mean(fi1_vec)
        h_arr[:, i] = numerator / denominator

    return h_arr.T


def tf_ratio_simple(stft_arr):
    """

    :param stft_arr:   -  input spectrum -  shape(bins, sensors, frames)
    :return h_arr:     -  shape=(sensors, bins)
    """
    bins, sensors, frames = stft_arr.shape
    h_arr = np.zeros(shape=(bins, sensors), dtype=np.complex)
    h_arr[:, 0] = np.ones(shape=(bins), dtype=np.complex)

    psd = get_power_spectral_density_matrix(stft_arr)

    for i in range(1, sensors):
        h_arr[:, i] = psd[:, i, 0] / psd[:, 0, 0]

    return h_arr.T
