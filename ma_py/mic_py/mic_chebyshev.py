import numpy as np
import math

def get_chebyshev_weights_for_amplitudes(vert_mic_count, hor_mic_count, n_bins):

    '''

    :param vert_mic_count: number of sensors along vertical axes
    :param hor_mic_count: number os sensors along horizontal axes
    :param n_bins: number of frequency bins
    :return: array of chebyshev weights with shape (vert_mic_count, hor_mic_count)
    '''

    weights = np.zeros((vert_mic_count * hor_mic_count, n_bins))
    ampls = [-12, -12, -12, -15, -15, -15, -15, -15, -15, -12, -12, -12, -9, -9, -9,
             -3, -3, -3, 0, 0, 0, 3, 3, 3, 6, 6, 6, 9, 9, 9, 12, 12, 12, 15, 15, 15,
             15, 15, 15, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 15, 15, 15,
             15, 15, 15, 15, 15, 15, 18, 18, 18, 21, 21, 21, 21, 21, 21, 21, 21, 21,
             24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
             24, 24, 24, 24, 24, 24, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
             30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 33, 33, 33,
             33, 33, 33, 33, 33, 33, 33, 33, 33, 36, 36, 36, 36, 36, 36, 36, 36, 36,
             36, 36, 36, 36, 36, 36, 36, 36, 36, 39, 39, 39, 39, 39, 39, 39, 39, 39,
             42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42,
             45, 45, 45, 45, 45, 45, 45, 45, 45, 48, 48, 48, 48, 48, 48, 48, 48, 48,
             48, 48, 48, 48, 48, 48, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
             51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
             51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51,
             51, 51, 51, 48, 48]

    if vert_mic_count != 1 and hor_mic_count != 1:
        for i in range(weights.shape[1]):
            weights_hor = chebyshev_weights(hor_mic_count, ampls[i])
            weights_vert = chebyshev_weights(vert_mic_count, ampls[i])
            weights_i = np.broadcast_to(weights_hor[np.newaxis, :],
                                        (vert_mic_count, hor_mic_count)) \
                        * weights_vert[:, np.newaxis]
            weights_i = weights_i.flatten()
            weights[:, i] = weights_i
    else:
        if vert_mic_count == 1:
            for i in range(weights.shape[1]):
                weights_hor = chebyshev_weights(hor_mic_count, ampls[i])
                weights[:, i] = weights_hor
        elif hor_mic_count == 1:
            for i in range(weights.shape[1]):
                weights_vert = chebyshev_weights(vert_mic_count, ampls[i])
                weights[:, i] = weights_vert


    return weights


def get_chebyshev_weights_for_nulls(vert_mic_count, hor_mic_count, dHor, dVert, n_bins, zeros):

    '''
    :param vert_mic_count: number of sensors along vertical axes
    :param hor_mic_count: number os sensors along horizontal axes
    :param dHor: horizontal distance between sensors
    :param dVert: vertical distance between sensors
    :param n_bins: number of frequency bins
    :param zeros: position of nulls
    :return: array of chebyshev weights with shape (vert_mic_count, hor_mic_count)
    '''

    weights = np.zeros((vert_mic_count * hor_mic_count, n_bins))
    for i in range(weights.shape[1]):
        weights11 = chebyshev_weights_zeros(hor_mic_count, i / 256 * 8000, dHor, zeros)
        weights6 = chebyshev_weights_zeros(vert_mic_count, i / 256 * 8000, dVert, zeros)
        weights_i = np.broadcast_to(weights11[np.newaxis, :], (vert_mic_count, hor_mic_count)) * weights6[:,
                                                                                                 np.newaxis]
        weights_i = weights_i.flatten()
        weights[:, i] = weights_i

    return weights



def chebyshev_weights(N, amplification):


    '''
    Calculates Chebyshev weights

    :param N: number of sensors
    :param amplification: desired amplification of the main lobe relative to the side lobes in dB
    :return: numpy array with shape (N,)

    '''

    t = 10 ** (amplification / 20)
    if t >= 1:
        a_0 = np.cosh(1 / (N - 1) * np.arccosh(t))
    else:
        a_0 = np.cos(1 / (N - 1) * np.arccos(t))

    if N % 2 == 1:
        M = int((N - 1) / 2)
        I = np.zeros(M + 1)
        for m in range(len(I)):
            for k in range(m, M + 1):
                I[m] += (-1) ** (M - k) * a_0 ** (2 * k) * 2 * M * math.factorial(k + M - 1) / (
                        math.factorial(k - m) * math.factorial(k + m) * math.factorial(M - k))
    elif N % 2 == 0:
        M = N // 2
        I = np.zeros(M + 1)
        for m in range(1, M + 1):
            for k in range(m, M + 1):
                I[m] += (-1) ** (M - k) * a_0 ** (2 * k - 1) * (2 * M - 1) * math.factorial(k + M - 2) / (
                        math.factorial(k - m) * math.factorial(k + m - 1) * math.factorial(M - k))



    if N % 2 == 1:
        I = np.asarray(list(I[len(I):0:-1]) + list(I))
    else:
        I = np.asarray(list(I[1:][::-1]) + list(I[1:]))

    I = I/np.max(I)
    return I


def chebyshev_weights_zeros(N, freq, distance, zeros_position, c=343.0):


    '''
    Calculates Chebyshev weights

    :param N: number of sensors
    :param freq: frequency of signal
    :param distance: distance between sensors
    :param zeros_position: desired position of first nulls in degrees
    :param c: speed of sound
    :return: numpy array with shape (N,)

    '''

    a_0 = np.cos(np.pi / (2 * N - 2)) / np.cos(np.pi * distance * freq / c * (np.sin(zeros_position * np.pi / 180)))


    if N % 2 == 1:
        M = int((N - 1) / 2)
        I = np.zeros(M + 1)
        for m in range(len(I)):
            for k in range(m, M + 1):
                I[m] += (-1) ** (M - k) * a_0 ** (2 * k) * 2 * M * math.factorial(k + M - 1) / (
                        math.factorial(k - m) * math.factorial(k + m) * math.factorial(M - k))
    elif N % 2 == 0:
        M = N // 2
        I = np.zeros(M + 1)
        for m in range(1, M + 1):
            for k in range(m, M + 1):
                I[m] += (-1) ** (M - k) * a_0 ** (2 * k - 1) * (2 * M - 1) * math.factorial(k + M - 2) / (
                        math.factorial(k - m) * math.factorial(k + m - 1) * math.factorial(M - k))



    if N % 2 == 1:
        I = np.asarray(list(I[len(I):0:-1]) + list(I))
    else:
        I = np.asarray(list(I[1:][::-1]) + list(I[1:]))

    I = I/np.max(I)
    return I

