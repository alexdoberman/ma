import numpy as np
import sys
from scipy.signal import medfilt, zpk2tf, freqs, zpk2sos, sosfilt


if sys.platform == 'linux':
    import pydsm

stat_cst = 0.225


def sif_filter(stft_arr, const=None):
    psd = stft_arr*stft_arr.conj()
    # h = stat_cst/np.mean(psd, axis=0)
    # stft_arr = np.einsum('b,ab->ab', np.sqrt(h), stft_arr)
    if const is None:
        cst = stat_cst
    else:
        cst = const
    h = cst/np.mean(psd)
    stft_arr = np.sqrt(h)*stft_arr

    return stft_arr


def med_filter(raw, rate, ms):

    ks = int(rate/1000 * ms) + 1
    return medfilt(raw, kernel_size=ks)


def speech_filter(stft_arr, rate, global_high_bound=None, enable_smooth=False):
    alf = 0.95
    freq_low_bound = 500.0
    freq_high_bound = 1000.0

    bins, _ = stft_arr.shape
    fft_size = (bins - 1) * 2

    hz_scale = rate / fft_size
    factor = 1 / (1 + 0.5 * alf)

    freq_norm = 2.0 * np.pi / rate
    const1 = 1 + alf ** 2 - 2 * alf * np.cos(freq_norm * freq_high_bound)
    const2 = 1 + alf * 2 - 2 * alf * np.cos(freq_norm * (2 * freq_high_bound))

    factor1 = factor * np.sqrt(const2 / const1)

    id_start = int(freq_low_bound * fft_size / rate)

    if global_high_bound is None:
        id_end = bins
    else:
        id_end = int(global_high_bound * fft_size / rate)

    gain = np.ones(bins)
    for i in range(id_start, id_end):
        hz_freq = i*hz_scale

        if hz_freq < freq_high_bound:
            hz_freq = freq_high_bound

        gain[i] = factor * np.sqrt(const2 / (1 + alf**2 - 2 * alf * np.cos(freq_norm * hz_freq)))

    for i in range(id_start):
        hz_freq = i*hz_scale
        hz_freq /= freq_low_bound
        gain[i] = hz_freq * factor1

    if enable_smooth:
        gain = smoothing(gain)
        gain = smoothing(gain)

    stft_out = np.einsum('a, ab->ab', gain, stft_arr)

    return stft_out


def smoothing(data):

    data = np.array(data)
    data_len = data.shape[0]
    d0 = 0.5 * (data[0] + data[1])
    dn = 0.5 * (data[data_len-1] + data[data_len-2])

    for i in range(1, data_len-1):
        data[i-1] = 0.25 * (data[i-1] + data[i+1]) + 0.5 * data[i]

    for i in range(data_len-1, 1, -1):
        data[i] = data[i-1]

    data[0] = d0
    data[-1] = dn

    return data

'''
    These equations are described in ANSI Standards S1.4-1983 and S1.42-2001.
'''


def a_weighting(stft_arr, rate, normalization=True, power=False):

    bins, _ = stft_arr.shape
    norm_coef = 10 ** 0.1

    gain = np.zeros(bins)
    fft_size = (bins - 1)*2

    for i in range(bins):
        fr_hz = i * (rate/fft_size)

        gain[i] = 12200**2 * fr_hz**4 / ((fr_hz**2 + 20.6**2)*(fr_hz**2 + 12200**2)*np.sqrt(fr_hz**2 + 107.7**2)*
                                         np.sqrt(fr_hz**2 + 737.9**2))
        if normalization:
            gain[i] *= norm_coef

        if power:
            gain[i] **= 2

    stft_out = np.einsum('a, ab->ab', gain, stft_arr)

    return stft_out


def check_a_weighting_coef(fft_size=512, rate=16000, normalization=True, power=False):

    norm_coef = 10 ** 0.1
    bins = fft_size//2 + 1

    dsm_coef = np.zeros(bins)
    est_coef = np.zeros(bins)

    for i in range(bins):
        fr_hz = i * (rate / fft_size)

        dsm_coef[i] = 12200 ** 2 * fr_hz ** 4 / ((fr_hz ** 2 + 20.6 ** 2) * (fr_hz ** 2 + 12200 ** 2) *
                                                 np.sqrt(fr_hz ** 2 + 107.7 ** 2) * np.sqrt(fr_hz ** 2 + 737.9 ** 2))
        if normalization:
            dsm_coef[i] *= norm_coef

        if power:
            dsm_coef[i] **= 2

        est_coef[i] = pydsm.audio_weightings.a_weighting(fr_hz, normal=normalization, power=power)

    print('Check a-weighting formula: {}'.format(np.mean((est_coef - dsm_coef)**2)))


'''
    ????
'''


def a_weighting2(raw_signal, rate):

    # s-domain tf
    zeros = np.array([0, 0, 0, 0])
    poles = np.array([-2*np.pi*20.6, -2*np.pi*20.6, -2*np.pi*12200, -2*np.pi*12200, -2*np.pi*107.7, -2*np.pi*737.9])

    # poly coefficients
    b, a = zpk2tf(zeros, poles, 1)

    gain = 1/abs(freqs(b=b, a=a, worN=[2*np.pi*1000])[1][0])
    print(freqs(b=b, a=a, worN=[2*np.pi*1000]))
    # params for digital filter
    z_d, p_d, k_d = _zpkbilinear(zeros, poles, gain, rate)
    sos = zpk2sos(z_d, p_d, k_d)

    return sosfilt(sos, raw_signal)


# https://github.com/endolith/waveform_analysis
def _relative_degree(z, p):
    """
    Return relative degree of transfer function from zeros and poles
    """
    degree = len(p) - len(z)
    if degree < 0:
        raise ValueError("Improper transfer function. "
                         "Must have at least as many poles as zeros.")
    else:
        return degree


def _zpkbilinear(z, p, k, fs):
    """
    Return a digital filter from an analog one using a bilinear transform.
    Transform a set of poles and zeros from the analog s-plane to the digital
    z-plane using Tustin's method, which substitutes ``(z-1) / (z+1)`` for
    ``s``, maintaining the shape of the frequency response.
    Parameters
    ----------
    z : array_like
        Zeros of the analog IIR filter transfer function.
    p : array_like
        Poles of the analog IIR filter transfer function.
    k : float
        System gain of the analog IIR filter transfer function.
    fs : float
        Sample rate, as ordinary frequency (e.g. hertz). No prewarping is
        done in this function.
    Returns
    -------
    z : ndarray
        Zeros of the transformed digital filter transfer function.
    p : ndarray
        Poles of the transformed digital filter transfer function.
    k : float
        System gain of the transformed digital filter.
    """
    z = np.atleast_1d(z)
    p = np.atleast_1d(p)

    degree = _relative_degree(z, p)

    fs2 = 2.0*fs

    # Bilinear transform the poles and zeros
    z_z = (fs2 + z) / (fs2 - z)
    p_z = (fs2 + p) / (fs2 - p)

    # Any zeros that were at infinity get moved to the Nyquist frequency
    z_z = np.append(z_z, -np.ones(degree))

    # Compensate for gain change
    k_z = k * np.real(np.prod(fs2 - z) / np.prod(fs2 - p))

    return z_z, p_z, k_z
