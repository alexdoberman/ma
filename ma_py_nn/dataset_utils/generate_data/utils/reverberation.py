from utils import wav_handler

import scipy.io
import scipy.signal
import numpy as np
np.set_printoptions(threshold=np.nan)


def reverb_matlab(input_signal_filename, root, input_filter_path, custom_filter=True):

    signal, rate = wav_handler.load_wav(input_signal_filename, root=root)

    mat = scipy.io.loadmat(input_filter_path)

    if custom_filter:
        imp_sig = np.squeeze(mat['data'])
        rir_rate = 16000
        # print(mat['reverberation_time'])
    else:
        imp_sig = np.squeeze(mat['h_air'])
        rir_rate = mat['air_info']['fs'][0][0][0][0]

    new_len = int((len(imp_sig)/float(rir_rate))*rate)

    imp_sig_rate1 = imp_sig

    if len(imp_sig) != new_len:
        imp_sig_rate1 = scipy.signal.resample(imp_sig, new_len)

    y_simple = scipy.signal.fftconvolve(imp_sig_rate1, signal)

    y = y_simple / max(abs(y_simple))
    """"
    # just for test
    ss = spectogram.get_spectogram(signal)
    sy = spectogram.get_spectogram(y)

    print('Energy before: {}'.format(spectogram.get_energy(ss)))
    print('Energy after: {}'.format(spectogram.get_energy(sy)))

    wav_handler.save_wav(y_simple, rate, 'before_normalization_', '.')
    wav_handler.save_wav(y, rate, 'after_normalization_', '.')
    """

    return signal, y
