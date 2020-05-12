import numpy as np
import os
from utils import wav_handler


np.set_printoptions(threshold=np.nan)


def sqrt_hann(data):
    return np.sqrt(np.hanning(data))


def stft(x, fft_size, overlap):
    hop = overlap
    w = sqrt_hann(fft_size)
    out = np.array([np.fft.rfft(w*x[i:i+fft_size])
                    for i in range(0, len(x) - fft_size, hop)])
    return out


def istft(X, overlap):
    fftsize = (X.shape[1] - 1) * 2
    hop = overlap
    w = sqrt_hann(fftsize)
    x = np.zeros(X.shape[0] * hop)
    wsum = np.zeros(X.shape[0] * hop)
    for n, i in enumerate(range(0, len(x) - fftsize, hop)):
        x[i:i + fftsize] += np.real(np.fft.irfft(X[n])) * w  # overlap-add
        wsum[i:i + fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]

    return x


def get_spectogram(wav_amplitudes, size=512, overlap=128, rate=0):
    # print(wav_amplitudes[100:200])
    ft = stft(wav_amplitudes, size, overlap)
    # dt = istft(ft, 256, rate)
    # print(dt[100:200])
    spectogram = np.transpose(np.absolute(ft))
    return spectogram


def get_spectogram_and_ft(wav_amplitudes, size=512, overlap=128):
    ft = stft(wav_amplitudes, size, overlap)
    spectogram = np.transpose(np.absolute(ft))
    return spectogram, ft


def get_spectogram_and_phase(wav_amplitudes, size=512, overlap=128):
    ft = stft(wav_amplitudes, size, overlap)
    spectogram = np.transpose(np.absolute(ft))
    return spectogram, ft/np.abs(ft)


def get_energy(spectogram):
    return np.sum(spectogram**2)
