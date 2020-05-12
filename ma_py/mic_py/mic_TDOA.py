import numpy

def time_delay_of_arrival(sensor_positions, source_position, F_s, c = 331.46):
    """
    TDOA

    :sensor_positions:  shape -  (3, num_sensors)
    :source_position:   shape - (3,)
    :return:  
        T - shape (num_sensors, 1)   array of delays in samples not in sec
        a - shape (num_sensors, 1)   array of gain (in near field)
    """

    num_sensors = sensor_positions.shape[1]
    r_source    = numpy.sqrt(numpy.sum(source_position**2,0))
    r_sensors_source = numpy.zeros((num_sensors,1))
    for sensor_index in range(num_sensors-1,-1,-1):
        r_sensors_source[sensor_index,0] = numpy.sqrt(numpy.sum((sensor_positions[:,sensor_index]-source_position)**2))
    T = (r_sensors_source-r_source) / c * F_s
    a = r_source / r_sensors_source
    return T,a
