import numpy as np


from mic_py.beamforming import apply_beamforming_vector, get_power_spectral_density_matrix
from mic_py.mic_geometry import get_sensor_positions, get_source_position, get_pair_mic_distance
from mic_py.mic_steering import time_delay_of_arrival, propagation_vector_free_field

from mic_py.mic_gannot import rtf_vector


'''
    Based on 'Theory and Application of Covariance Matrix Tapers for Robust Adaptive Beamforming' - Joseph R. Guerci
'''


def cov_matrix_tapper_linear_array(stft_arr, delta=0.01):
    """
        Estimates psd matrix with taper for linear array

    :param stft_arr: stft array for covariance matrix estimation - shape (bins, sensors, frames)
    :param delta: parameter for taper

    :return: psd matrix multiplied by taper - shape (bins, sensors, sensors)

    """
    bins, sensors, frames = stft_arr.shape

    psd = get_power_spectral_density_matrix(stft_arr)

    T_cov_matrix = np.zeros((bins, sensors, sensors), dtype=np.complex)

    for i in range(sensors):
        for j in range(sensors):
            if i == j:
                T_cov_matrix[:, i, j] = np.ones(257)
            else:
                T_cov_matrix[:, i, j] = np.sin((i - j)*delta) / ((i - j)*delta)

    for i in range(bins):
        psd[i, :, :] = np.multiply(psd[i, :, :], T_cov_matrix[i])

    psd = psd + 0.001 * np.identity(psd.shape[-1])

    return psd


def cov_matrix_tapper_mean_steering(stft_arr, hor_mic_count, vert_mic_count, dHor, dVert, sr,
                                    delta=0.57, n_fft=512, num_it=500):
    """
        Estimates psd matrix with taper for planar array based on steering averaging

    :param stft_arr: stft array for covariance matrix estimation - shape (bins, sensors, frames)
    :param hor_mic_count:
    :param vert_mic_count:
    :param dHor:
    :param dVert:
    :param sr: sample rate
    :param n_fft:
    :param num_it: num of iterations for steering averaging
    :param delta: parameter for taper

    :return: psd matrix multiplied by taper - shape (bins, sensors, sensors)

    """

    bins, sensors, frames = stft_arr.shape

    psd = get_power_spectral_density_matrix(stft_arr)

    d_arr_mean = np.zeros(shape=(bins, sensors), dtype=np.complex)

    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)

    for i in range(num_it):
        [angle_h_delta, angle_v_delta] = np.random.uniform(-delta, delta, 2)
        source_position = get_source_position(angle_h_delta, angle_v_delta, radius=6.0)
        d_arr = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)
        d_arr_mean += d_arr.T

        if i % 100 == 0 and i != 0:
            print('{} vectors processed!'.format(i))

    d_arr_mean /= num_it

    T_cov_matrix = np.zeros((bins, sensors, sensors), dtype=np.complex)

    for j in range(bins):
        T_cov_matrix[j, :, :] = np.outer(d_arr_mean[j, :], d_arr_mean[j, :].conj())

    for i in range(bins):
        psd[i, :, :] = np.multiply(psd[i, :, :], T_cov_matrix[i])

    psd = psd + 0.001 * np.identity(psd.shape[-1])

    return psd


def cov_matrix_tapper_interf_steering(hor_mic_count, vert_mic_count, dHor, dVert, angle_h, angle_v, sr,
                                    delta=0.5, n_fft=512, num_it=1000):

    bins = int(n_fft / 2) + 1
    sensors = vert_mic_count*hor_mic_count
    d_arr_mean = np.zeros(shape=(bins, hor_mic_count*vert_mic_count), dtype=np.complex)

    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)

    for i in range(num_it):
        [angle_h_delta, angle_v_delta] = np.random.uniform(-delta, delta, 2)
        source_position = get_source_position(angle_h + angle_h_delta, angle_v + angle_v_delta, radius=6.0)
        d_arr = propagation_vector_free_field(sensor_positions, source_position, N_fft=n_fft, F_s=sr)
        d_arr_mean += d_arr.T

        if i % 100 == 0 and i != 0:
            print('{} vectors processed!'.format(i))

    d_arr_mean /= num_it

    T_cov_matrix = np.zeros((bins, sensors, sensors), dtype=np.complex)

    for j in range(bins):
        T_cov_matrix[j, :, :] = np.outer(d_arr_mean[j, :], d_arr_mean[j, :].conj())

    psd = T_cov_matrix

    psd = psd + 0.001 * np.identity(psd.shape[-1])

    return psd


def cov_matrix_tapper_interf_steering_bandwidth(hor_mic_count, vert_mic_count, dHor, dVert, angle_h, angle_v,
                                                angle_inf_h, angle_inf_v, sr,
                                                delta=0.5, n_fft=512, bandwidth=0.5):

    bins = n_fft//2 + 1
    sensors = hor_mic_count*vert_mic_count

    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)

    inf_source_position = get_source_position(angle_inf_h, angle_inf_v, radius=6.0)
    d_arr_inf = propagation_vector_free_field(sensor_positions, inf_source_position, N_fft=n_fft, F_s=sr).T

    psd = np.zeros((bins, sensors, sensors), dtype=np.complex)

    # get psd by steering in inference direction
    for j in range(bins):
        psd[j, :, :] = np.outer(d_arr_inf[j, :], d_arr_inf[j, :].conj())

    source_position = get_source_position(angle_h, angle_v, radius=6.0)
    time_delays, a = time_delay_of_arrival(sensor_positions, source_position, sr)
    freq_array = 2 * np.pi * np.arange(0, bins, dtype=np.float32) / ((bins - 1) * 2)

    T_cov_matrix = np.zeros((bins, sensors, sensors), dtype=np.complex)

    for i in range(sensors):
        for j in range(sensors):
            if i == j:
                T_cov_matrix[:, i, j] = 1
                continue
            delay = time_delays[i] - time_delays[j]
            T_cov_matrix[1, i, j] = 1
            T_cov_matrix[1:, i, j] = np.sin(freq_array[1:] * bandwidth * delay * sr) / \
                                     (delay * bandwidth * freq_array[1:] * sr)

    for i in range(bins):
        psd[i, :, :] = np.multiply(psd[i, :, :], T_cov_matrix[i])

    psd = psd + 0.001 * np.identity(psd.shape[-1])

    return psd, T_cov_matrix


def cov_matrix_tapper_bandwidth(stft_arr, hor_mic_count, vert_mic_count, dHor, dVert, angle_h, angle_v, sr,
                                bandwidth=0.5):
    """
        Estimates psd matrix with taper for planar array based on bandwidth

    :param stft_arr: stft array for covariance matrix estimation - shape (bins, sensors, frames)
    :param hor_mic_count:
    :param vert_mic_count:
    :param dHor:
    :param dVert:
    :param sr: sample rate
    :param bandwidth: taper parameter

    :return: psd matrix multiplied by taper - shape (bins, sensors, sensors)

    """
    bins, sensors, frames = stft_arr.shape
    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position = get_source_position(angle_h, angle_v, radius=6.0)

    time_delays, a = time_delay_of_arrival(sensor_positions, source_position, sr)

    psd = get_power_spectral_density_matrix(stft_arr)

    freq_array = 2*np.pi*np.arange(0, bins, dtype=np.float32)/((bins-1)*2)
    T_cov_matrix = np.zeros((bins, sensors, sensors), dtype=np.complex)

    for i in range(sensors):
        for j in range(sensors):
            if i == j:
                T_cov_matrix[:, i, j] = 1
                continue
            delay = time_delays[i] - time_delays[j]
            T_cov_matrix[1, i, j] = 1
            T_cov_matrix[1:, i, j] = np.sin(freq_array[1:]*bandwidth*delay*sr)/(delay*bandwidth*freq_array[1:]*sr)

    for i in range(bins):
        psd[i, :, :] = np.multiply(psd[i, :, :], T_cov_matrix[i])

    psd = psd + 0.001 * np.identity(psd.shape[-1])

    return psd, T_cov_matrix


def interf_steering_rtf_psd(stft_noise):

    bins, sensors, frames = stft_noise.shape

    d_arr_inf = rtf_vector(stft_noise, filter_type='simple').T

    psd = np.zeros((bins, sensors, sensors), dtype=np.complex)

    # get psd by steering in inference direction
    for j in range(bins):
        psd[j, :, :] = np.outer(d_arr_inf[j, :], d_arr_inf[j, :].conj())

    psd = psd + 0.001 * np.identity(psd.shape[-1])

    return psd


def get_taper(hor_mic_count, vert_mic_count, dHor, dVert, angle_h, angle_v, sr, fft_size, bandwidth=0.5):
    """
        Estimates taper

    :param hor_mic_count:
    :param vert_mic_count:
    :param dHor:
    :param dVert:
    :param sr: sample rate
    :param bandwidth: taper parameter

    :return: taper - shape (bins, sensors, sensors)

    """
    bins = fft_size // 2 + 1
    sensors = hor_mic_count * vert_mic_count
    sensor_positions = get_sensor_positions(hor_mic_count, vert_mic_count, dHor=dHor, dVert=dVert)
    source_position = get_source_position(angle_h, angle_v, radius=6.0)

    time_delays, a = time_delay_of_arrival(sensor_positions, source_position, sr)

    freq_array = 2 * np.pi * np.arange(0, bins, dtype=np.float32) / ((bins - 1) * 2)
    T_cov_matrix = np.zeros((bins, sensors, sensors), dtype=np.complex)

    for i in range(sensors):
        for j in range(sensors):
            if i == j:
                T_cov_matrix[:, i, j] = 1
                continue
            delay = time_delays[i] - time_delays[j]
            T_cov_matrix[1, i, j] = 1
            T_cov_matrix[1:, i, j] = np.sin(freq_array[1:] * bandwidth * delay * sr) / (
            delay * bandwidth * freq_array[1:] * sr)

    return T_cov_matrix
