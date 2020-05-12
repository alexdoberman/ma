import soundfile as sf
import numpy as np
import scipy
import scipy.io
import scipy.signal


# def reverb_matlab2(input_signal_filename, input_filter_path, custom_filter=True):
#
#     signal, rate = sf.read(input_signal_filename)
#
#     mat = scipy.io.loadmat(input_filter_path)
#
#     if custom_filter:
#         imp_sig = np.squeeze(mat['data'])
#         rir_rate = 16000
#     else:
#         imp_sig = np.squeeze(mat['h_air'])
#         rir_rate = mat['air_info']['fs'][0][0][0][0]
#
#     new_len = int((len(imp_sig)/float(rir_rate))*rate)
#
#     imp_sig_rate1 = imp_sig
#
#     if len(imp_sig) != new_len:
#         imp_sig_rate1 = scipy.signal.resample(imp_sig, new_len)
#
#     y = scipy.signal.fftconvolve(imp_sig_rate1, signal)
#     max_ = max(abs(y))
#     y_norm = y / max_
#
#     return signal, y_norm

def reverb_matlab(signal_clean, rate, input_filter_filename):
    """
    Apply reverberation. Work only on 16000 Hz !!!

    :param signal_clean:
    :param rate:
    :param input_filter_filename:
    :return:
    """

    signal_clean = signal_clean.astype(np.float64)

    # Open Impulse Response (IR)
    mat = scipy.io.loadmat(input_filter_filename)
    IR = np.squeeze(mat['data'])
    rir_rate = 16000

    if int(rate) != rir_rate:
        new_len_RIR = int((len(IR) / float(rir_rate)) * rate)
        IR = scipy.signal.resample(IR, new_len_RIR)

    signal_rev = scipy.signal.fftconvolve(signal_clean, IR, mode='full')

    # # Normalization
    # eps = 1e-7
    # signal_rev = signal_rev / (np.max(np.abs(signal_rev)) + eps)

    # Cut reverberated signal (same length as clean sig)
    signal_rev = signal_rev[0:signal_clean.shape[0]]

    return signal_rev

