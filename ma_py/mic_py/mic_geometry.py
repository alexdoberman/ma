# -*- coding: utf-8 -*-

import numpy as np

def get_sensor_positions_kinect():
    sensors = []

    sensors.append([-0.113,0,0])
    sensors.append([-0.076,0,0])
    sensors.append([-0.036,0,0])
    sensors.append([0.113,0,0])

    sensors = np.array(sensors).T
    return sensors

def get_sensor_positions_respeaker(a=1, n=6):

    '''
    функция рассчитывает координаты микрофонов, которые расположены в вершинах правильных 2-,4- и
    6-угольников. центр системы координат выбирается в центре решетки (центре описанной около фигуры окружности),
    ось y проводится через начало координат и первый микрофон. система координат соответствует правой тройке,
    т.е. ось z направлена перпендикулярно решетке вверх


    Межмикрофонное расстояние для решетки для 4-х микрофонов 0.0457 м
                                          для 2-х 0.05
                                          для 6 мик 0.051 м или 0.0463

    :param a: intermic distance in m
    :param n: number of microphones in array. can be 2, 4 or 6
    :return:
    '''

    if n == 2:
        r = a/2.
    elif n == 4:
        r = a/np.sqrt(2)
    elif n == 6:
        r = a
    else:
        print('cannot do anything for {}-side polygon. only 2, 4, or 6. sorry, mate'.format(n))
        return
    sensors = []
    alpha = 90.
    ang_step = 360./n

    for i in range(n):
        x = r*np.cos(np.deg2rad(alpha+i*ang_step))
        y = r*np.sin(np.deg2rad(alpha+i*ang_step))
        sensors.append([x,y,0])

    sensors = np.array(sensors).T
    return sensors

def get_source_position_respeaker(azimuth, polar, radius=6):
    '''
    функция рассчитывает координаты источника в декартовой системе координат. система координат предполагается заранее заданной
    относительно центра микрофонной решетки и первого микрофона. функция принимает на вход углы сферической системы координат -
    азимутальный и полярный. возвращает координты источника в декартовой системе координат. азимутальный угол 0 соответствует
    положительному направлению оси оу
    :param azimuth: азимутальный угол. отсчитывается от положительного направления оси оу против часовой стрелки
                    изменяется от 0 до 360
    :param polar: полярный угол. отчитывается от положительно направления оси z. изменяется от 0 до 180
    :param radius: расстояние до источника в м
    :return:
    '''

    alpha = 90

    x = radius*np.sin(np.deg2rad(polar))*np.cos(np.deg2rad(alpha+azimuth))
    y = radius * np.sin(np.deg2rad(polar)) * np.sin(np.deg2rad(alpha + azimuth))
    z = radius * np.cos(np.deg2rad(polar))

    source = np.array([x, y, z])
    return source

def get_sensor_positions_circular(r=0.1, n=8):

    '''
    функция рассчитывает координаты микрофонов, которые расположены в вершинах правильных 2-,4- и
    6-угольников. центр системы координат выбирается в центре решетки (центре описанной около фигуры окружности),
    ось y проводится через начало координат и первый микрофон. система координат соответствует правой тройке,
    т.е. ось z направлена перпендикулярно решетке вверх


    Межмикрофонное расстояние для решетки для 4-х микрофонов 0.0457 м
                                          для 2-х 0.05
                                          для 6 мик 0.051 м или 0.0463

    :param a: intermic distance in m
    :param n: number of microphones in array. can be 2, 4 or 6
    :return:
    '''

    sensors = []
    alpha = 0.
    ang_step = 360./n

    for i in range(n):
        x = r*np.cos(np.deg2rad(alpha+i*ang_step))
        y = r*np.sin(np.deg2rad(alpha+i*ang_step))
        sensors.append([x,y,0])

    sensors = np.array(sensors).T
    return sensors

def get_source_position_circular(azimuth, polar, radius=10.):
    '''
    функция рассчитывает координаты источника в декартовой системе координат. система координат предполагается заранее заданной
    относительно центра микрофонной решетки и первого микрофона. функция принимает на вход углы сферической системы координат -
    азимутальный и полярный. возвращает координты источника в декартовой системе координат. азимутальный угол 0 соответствует
    положительному направлению оси оу
    :param azimuth: азимутальный угол. отсчитывается от положительного направления оси оу против часовой стрелки
                    изменяется от 0 до 360
    :param polar: полярный угол. отчитывается от положительно направления оси z. изменяется от 0 до 180
    :param radius: расстояние до источника в м
    :return:
    '''

    alpha = 0.

    x = radius*np.sin(np.deg2rad(polar))*np.cos(np.deg2rad(alpha+azimuth))
    y = radius * np.sin(np.deg2rad(polar)) * np.sin(np.deg2rad(alpha + azimuth))
    z = radius * np.cos(np.deg2rad(polar))

    source = np.array([x, y, z])
    return source


# sensor_positions: the positions of the sensors. sensor_positions(:, n)
#                   are the coordinates of the n-th sensor.
def get_sensor_positions(Hor_mic_count  = 11, Vert_mic_count = 6, dHor  = 0.035, dVert = 0.05):
    """
    Calc sensors position for rectangular mic array 

    :Hor_mic_count  :
    :Vert_mic_count :
    :dHor :  distance between mic by horizont axis   
    :dVert : distance between mic by vertical axis   
        Example array:

            *  *  *  *  *  *
            *  *  *  *  *  *
            *  *  *  *  *  * 
            *  *  *  *  *  * 
            Hor_mic_count  = 6
            Vert_mic_count = 4
    :return:  
        shape (3, Hor_mic_count*Vert_mic_count)

    Example output:        
        x1 x2 ... xN
        y1 y2 ... yN
        z1 z2 ... zN

    """

    Half_Width_H = Hor_mic_count * dHor / 2
    Half_Width_V = Vert_mic_count * dVert / 2

    sensors = []
    for v in range(0,Vert_mic_count,1):
        for h in range(0,Hor_mic_count,1):
            z = 0
            x = - Half_Width_H + h * dHor
            y = - Half_Width_V + v * dVert
            sensors.append([x,y,z])

    sensors = np.array(sensors).T
    return sensors

# source_positions: 3d - vector in direction source
def get_source_position(angle_Hor, angle_Vert, radius = 6.0):
    """
    Calc source position vector 

    :angle_Hor  : angle = -90 .. 90
    :angle_Vert : angle = -90 .. 90
    :radius: radius. default 6m
    :return:  
        shape (3, )
    """


    angle_Hor  = np.pi*angle_Hor/180.0
    angle_Vert  = np.pi*angle_Vert/180.0

    Angle_alfa = angle_Hor
    Angle_beta = angle_Vert

    z_norm = 1.0/np.sqrt(np.tan(Angle_alfa)*np.tan(Angle_alfa) + np.tan(Angle_beta)*np.tan(Angle_beta) +1)
    x_norm = np.tan(Angle_alfa)*z_norm
    y_norm = np.tan(Angle_beta)*z_norm

    
    x_pos  = radius*x_norm
    y_pos  = radius*y_norm
    z_pos  = radius*z_norm

    source = np.array([x_pos, y_pos, z_pos])
    return source

def get_pair_mic_distance(sensors):
    """
    Calc source position vector 

    :sensors: - sensors position - shape (3, Hor_mic_count*Vert_mic_count)
    :return:  
        D_IJ - shape (sensors_count, sensors_count)
    """

    dim, sensors_count  = sensors.shape
    D_IJ = np.zeros((sensors_count, sensors_count))
    for i in range(sensors_count):
        for j in range(sensors_count):
            D_IJ[i,j] = np.sqrt(np.sum((sensors[:,i] - sensors[:,j])**2))

    return D_IJ
