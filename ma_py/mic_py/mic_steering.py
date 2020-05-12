import numpy as np

#331.46
def time_delay_of_arrival_2(sensor_positions, source_position, F_s, c = 343.0):
    
    """
    TDOA

    :sensor_positions:  shape -  (3, num_sensors)
    :source_position:   shape - (3,)
    :return:  
        T - shape (num_sensors, 1)   array of delays in samples not in sec
        a - shape (num_sensors, 1)   array of gain (in near field)
    """

    num_sensors = sensor_positions.shape[1]
    r_source    = np.sqrt(np.sum(source_position**2,0))
    r_sensors_source = np.zeros((num_sensors,1))
    for sensor_index in range(num_sensors-1,-1,-1):
        r_sensors_source[sensor_index,0] = np.sqrt(np.sum((sensor_positions[:,sensor_index]-source_position)**2))
    T = (r_sensors_source-r_source) / c * F_s
    a = r_source / r_sensors_source
    return T,a

def propagation_vector_free_field_2(sensor_positions, source_position, N_fft, F_s):
    """
    Calc steering vector 
    :return:  
        D - shape (num_sensors, bin)  
    """


    D = np.zeros((sensor_positions.shape[1], (int)(N_fft/2) + 1))
    D = D+0j
    T, a = time_delay_of_arrival(sensor_positions, source_position, F_s)

    freq_array = np.arange(0 , N_fft/2 + 1, dtype = np.float32)/N_fft

    for sensor_index in range(sensor_positions.shape[1]-1, -1, -1):
        #D[sensor_index, :] = a[sensor_index] * numpy.exp(-1j * 2*numpy.pi*freq_array * T[sensor_index])
        D[sensor_index, :] =  np.exp(-1j * 2 * np.pi * freq_array * T[sensor_index])
    return D







def time_delay_of_arrival(sensor_positions, source_position, F_s, c = 343.0):
    
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
    a                = np.ones((num_sensors))
    for sensor_index in range(num_sensors): 
        r_sensors_source[sensor_index] = np.sqrt(np.sum((sensor_positions[:,sensor_index]-source_position)**2))

    r_sensors_source = r_sensors_source - r_sensors_source[0]
    T = r_sensors_source/c

    return T,a

def propagation_vector_free_field(sensor_positions, source_position, N_fft, F_s):
    """
    Calc steering vector 
        :sensor_positions:  shape -  (3, num_sensors)
        :source_position:   shape - (3,)

    :return:  
        D - shape (num_sensors, bin)  
    """

    D = np.zeros((sensor_positions.shape[1], (int)(N_fft/2) + 1))
    D = D+0j
    T, a = time_delay_of_arrival(sensor_positions, source_position, F_s)

    freq_array = np.arange(0 , N_fft/2 + 1, dtype = np.float32)/N_fft

    for sensor_index in range(sensor_positions.shape[1]):
        #D[sensor_index, :] = a[sensor_index] * numpy.exp(-1j * 2*numpy.pi*freq_array * T[sensor_index])
        D[sensor_index, :] =  np.exp(-1j * 2 * np.pi * freq_array * T[sensor_index] * F_s)
    return D
