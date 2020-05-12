# -*- coding: utf-8 -*-
import numpy as np


eps = 1e-9
ENERGY_THRESHOLD = 0.1
FRAME_THRESHOLD = 6


def min_max_scale(mix):
    # Scale the inputs
    mix = (mix - mix.min()) / (mix.max() - mix.min() + eps)
    return mix


def normalize_signal(sig):

    sig = sig - np.mean(sig)
    sig = sig / (np.max(np.abs(sig)) + eps)
    return sig


def normalize_signal_std(y):

    y = y - np.mean(y)
    return y/np.std(y)


def dc_preprocess(stft_sig):

    mix = np.log10(np.abs(stft_sig) + eps)
    return mix


def dcce_preprocess(stft_sig):

    mix = np.sqrt(np.abs(stft_sig))
    return min_max_scale(mix)


def dan_preprocess(stft_sig):

    mix = np.log10(np.abs(stft_sig) + eps)
    return mix


def chimera_preprocess(stft_sig):

    mix = np.log10(np.abs(stft_sig) + eps)
    return mix


def preemphasis(signal, coeff=0.95):
    """
    Perform preemphasis
    https://github.com/jameslyons/python_speech_features/blob/e51df9e484da6c52d30b043f735f92580b9134b4/python_speech_features/sigproc.py#L133
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :return: filtered signal
    """
    if coeff == 0.0:
        return signal

    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def undo_preemphasis(preemphasized_signal, coeff=0.95):
    """
    Undo the preemphasis of an input signal

    :param preemphasized_signal:
    :param coeff:
    :return: numpy array containing the signal without preemphasis
    """

    if coeff == 0.0:
        return preemphasized_signal

    # Get the length of the input and preallocate the output array
    length = preemphasized_signal.shape[0]
    signal = np.zeros(length)

    # Set the initial element of the signal
    signal[0] = preemphasized_signal[0]

    # Use the recursion relation to compute the output signal
    for i in range(1, length):
        signal[i] = preemphasized_signal[i] + coeff*signal[i-1]

    return signal


def energy_mask(stft_data, thr=ENERGY_THRESHOLD, frame_thr=FRAME_THRESHOLD):
    """
    Return binary mask from stft array by energy threshold

    :param stft_data:
    :param thr:

    :return: energy mask - shape (frames)
    """

    dim = len(stft_data.shape)
    if dim > 2:
        raise Exception('Input data should be two dimensional array. Got {} instead'.format(dim))

    frames, bins = stft_data.shape
    output = np.zeros(shape=frames)

    xx_max = np.mean(np.sum(stft_data * stft_data.conj(), axis=1))
    xx_bound = xx_max * thr

    nn = 0
    last_idx = 0
    for fr in range(frames):
        if np.sum(stft_data[fr] * stft_data[fr].conj()) < xx_bound:
            nn += 1
        else:
            if 0 <= nn < frame_thr:
                output[last_idx:fr] = np.ones(shape=(fr - last_idx))
                last_idx = fr
            else:
                output[fr] = 1
                last_idx = fr
            nn = 0

    return output
