import numpy as np
from mic_py.beamforming import *
from mic_py import mic_geometry
from mic_py import mic_steering
from mic_py.beamforming import get_power_spectral_density_matrix, get_mvdr_vector, apply_beamforming_vector


def calc(steering_vector_spk, steering_vector_noise, stft_mix):
    """
    Calculate mvdr based on covariance matrix estimated through steering vector

    :param steering_vector_spk:   steering vector in speaker direction (bins, num_sensors)
    :param steering_vector_noise: steering vector in noise direction (bins, num_sensors)
    :param stft_mix:              mix specter

    :return: result_spec :        result specter (bins, frames)
    """
    bins, num_sensors = steering_vector_spk.shape
    psd_matrix = np.zeros(shape=(bins, num_sensors, num_sensors), dtype=np.complex)

    for i in range(bins):
        buffer = np.outer(steering_vector_noise[i].conj(), steering_vector_noise[i])
        psd_matrix[i] = buffer

    psd_matrix = psd_matrix + 0.001 * np.identity(psd_matrix.shape[-1])

    mvdr_vector = get_mvdr_vector(steering_vector_spk, psd_matrix)

    result_spec = apply_beamforming_vector(mvdr_vector, stft_mix)

    return result_spec
