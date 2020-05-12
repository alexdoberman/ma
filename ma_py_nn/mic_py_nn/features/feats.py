# -*- coding: utf-8 -*-
import numpy as np

# FFT parameters, self-explanatory
FRAME_RATE = 16000
FRAME_LENGTH = .032
FRAME_SHIFT = .008

def sqrt_hann(M):
    return np.sqrt(np.hanning(M))


def stft(x, fftsize=int(FRAME_LENGTH*FRAME_RATE),
         overlap=FRAME_LENGTH//FRAME_SHIFT):
    """
    Short-time fourier transform.
        x:
        input waveform (1D array of samples)

        fftsize:
        in samples, size of the fft window

        overlap:
        should be a divisor of fftsize, represents the rate of
        window superposition (window displacement=fftsize/overlap)

        return: linear domain spectrum (2D complex array)
    """
    hop = int(np.round(fftsize / overlap))
    w = sqrt_hann(fftsize)
    out = np.array([np.fft.rfft(w*x[i:i+fftsize])
                    for i in range(0, len(x)-fftsize, hop)])
    return out


def istft(X, overlap=FRAME_LENGTH//FRAME_SHIFT):
    """
    Inverse short-time fourier transform.
        X:
        input spectrum (2D complex array)

        overlap:
        should be a divisor of (X.shape[1] - 1) * 2, represents the rate of
        window superposition (window displacement=fftsize/overlap)

        return: floating-point waveform samples (1D array)
    """
    fftsize = (X.shape[1] - 1) * 2
    hop = int(np.round(fftsize / overlap))
    w = sqrt_hann(fftsize)
    x = np.zeros(X.shape[0]*hop)
    wsum = np.zeros(X.shape[0]*hop)
    for n, i in enumerate(range(0, len(x)-fftsize, hop)):
        x[i:i+fftsize] += np.real(np.fft.irfft(X[n])) * w   # overlap-add
        wsum[i:i+fftsize] += w ** 2.
    pos = wsum != 0
    x[pos] /= wsum[pos]
    return x



