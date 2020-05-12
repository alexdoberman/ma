import numpy as np

from mic_py.mic_geometry import get_sensor_positions, get_source_position


def get_steering(hor_angle, vert_angle, dHor=0.035, dVert=0.05, c=343.0, hor_mic_count=11, vert_mic_count=6, n_fft=512,
                 sr=16000, num_mic=66):

    sensors_coords = get_sensor_positions(hor_mic_count, vert_mic_count, dHor, dVert)

    source_pos = get_source_position(hor_angle, vert_angle)

    theta = np.arccos(source_pos[2] / np.sqrt(source_pos[0]**2 + source_pos[1]**2 + source_pos[2]**2))

    if source_pos[0] != 0:
        phi = np.arctan(source_pos[1]/source_pos[0])
    else:
        phi = 0

    _, mic_count = sensors_coords.shape

    td = [1 / c * (sensors_coords[0][i] * np.sin(theta) * np.cos(phi) + sensors_coords[1][i] *
                    np.sin(theta) * np.sin(phi)) for i in range(mic_count)]

    freq_array = np.arange(0, n_fft/2 + 1, dtype=np.float32)/n_fft

    out_vec = [np.exp(-1j * td[i] * 2*np.pi * sr * freq_array) for i in range(num_mic)]

    return np.array(out_vec)


def time_delay_of_arrival(sensor_positions, source_position, F_s, c=343.0):
    """
    TDOA

    :sensor_positions:  shape -  (3, num_sensors)
    :source_position:   shape - (3,)
    :return:
        T - shape (num_sensors)   array of delays in sec
        a - shape (num_sensors)   array of gain (in near field)
    """

    num_sensors = sensor_positions.shape[1]
    r_sensors_source = np.zeros((num_sensors))
    a = np.ones((num_sensors))
    for sensor_index in range(num_sensors):
        r_sensors_source[sensor_index] = np.sqrt(np.sum((sensor_positions[:, sensor_index] - source_position) ** 2))

    r_sensors_source_bias = r_sensors_source - r_sensors_source[0]
    T = r_sensors_source_bias / c

    return T, a, r_sensors_source


def propagation_vector_free_field(sensor_positions, source_position, N_fft, F_s):
    """
    Calc steering vector
        :sensor_positions:  shape -  (3, num_sensors)
        :source_position:   shape - (3,)

    :return:
        D - shape (num_sensors, bin)
    """

    D = np.zeros((sensor_positions.shape[1], (int)(N_fft / 2) + 1))
    D = D + 0j
    T, a, _ = time_delay_of_arrival(sensor_positions, source_position, F_s)

    freq_array = np.arange(0, N_fft / 2 + 1, dtype=np.float32) / N_fft

    for sensor_index in range(sensor_positions.shape[1]):
        # D[sensor_index, :] = a[sensor_index] * numpy.exp(-1j * 2*numpy.pi*freq_array * T[sensor_index])
        D[sensor_index, :] = np.exp(-1j * 2 * np.pi * freq_array * T[sensor_index] * F_s)
    return D


def get_der_steering_2(sensor_positions, source_position, N_fft, F_s, hor, vert, c=343.0):
    """
        Calc steering vector derivate
            :sensor_positions:  shape -  (3, num_sensors)
            :source_position:   shape - (3,)

        :return:
            D - shape (num_sensors, bin)
        """

    mic_count = sensor_positions.shape[1]

    D_hor = np.zeros((mic_count, (int)(N_fft / 2) + 1), dtype=np.complex)
    D_hor[0, :] = np.zeros((int)(N_fft / 2) + 1)

    D_vert = np.zeros((mic_count, (int)(N_fft / 2) + 1), dtype=np.complex)
    D_vert[0, :] = np.zeros((int)(N_fft / 2) + 1)

    hor_rad = np.deg2rad(hor)
    vert_rad = np.deg2rad(vert)

    freq_array = np.arange(0, N_fft / 2 + 1, dtype=np.float32) / N_fft
    const = freq_array * -1j * 2 * np.pi * F_s

    _, _, time_delays = time_delay_of_arrival(sensor_positions, source_position, F_s)

    steering = propagation_vector_free_field(sensor_positions, source_position, N_fft, F_s)

    eq = np.tan(hor_rad) ** 2 + np.tan(vert_rad) ** 2 + 1
    denom = 1/np.sqrt(eq)

    x_hor_der = -6 * (
    1 / np.cos(hor_rad) ** 2 * denom + np.tan(hor_rad) * -eq ** -1.5 * 2 * np.tan(hor_rad) * 1 / np.cos(hor_rad) ** 2)

    y_hor_der = -6 * (np.tan(vert_rad) * -eq ** -1.5 * 2 * np.tan(hor_rad) * 1 / np.cos(hor_rad) ** 2)

    z_hor_der = -6 * -eq ** -1.5 * 2 * np.tan(hor_rad) * 1 / np.cos(hor_rad) ** 2

    x_vert_der = -6 * (np.tan(hor_rad) * -eq ** -1.5 * 2 * np.tan(vert_rad) * 1 / np.cos(vert_rad) ** 2)

    y_vert_der = -6 * (
    1 / np.cos(vert_rad) ** 2 * denom + np.tan(vert_rad) * -eq ** -1.5 * 2 * np.tan(vert_rad) * 1 / np.cos(vert_rad) ** 2)

    z_vert_der = -6 * -eq ** -1.5 * 2 * np.tan(vert_rad) * 1 / np.cos(vert_rad) ** 2

    for i in range(1, mic_count):
        D_hor[i, :] = steering[i] * const * 1/c * (x_hor_der + y_hor_der + z_hor_der) *\
                                                (1/time_delays[i] - 1/time_delays[0])

        D_vert[i, :] = steering[i] * const * 1 / c * (x_vert_der + y_vert_der + z_vert_der) * \
                      (1 / time_delays[i] - 1 / time_delays[0])

    return D_hor, D_vert


def get_steering_linear_array(angle, d, num_mic, sr, n_fft, c=343.0):
    td = d*np.sin(np.deg2rad(angle))/c

    freq_array = np.arange(0, n_fft/2 + 1, dtype=np.float32)/n_fft

    out_vec = [np.exp(-1j * td * 2*np.pi * i * sr * freq_array) for i in range(num_mic)]

    return np.array(out_vec)


def get_der_steering_linear_array(angle, d, num_mic, sr, c=343.0):
    td = d*np.sin(np.deg2rad(angle))/c

    out_vec = [np.exp(-1j*td*2*np.pi*i*sr)*(-1j*d*np.cos(np.deg2rad(angle))) for i in range(num_mic)]

    return out_vec
