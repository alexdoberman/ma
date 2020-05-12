import math
import numpy as np
from utils import wav_handler as wh

EPS = 1e-6


def get_sq_sum(data):
    assert len(data.shape) == 1, 'Invalid shape'
    return sum(data**2)


def get_snr(signal, noise, noise_name='fake'):
    sq_sum_noise = get_sq_sum(noise)
    if sq_sum_noise == 0:
        # print("we have big problems!!!!")
        return 10 * math.log10(get_sq_sum(signal) / EPS)
    return 10*math.log10(get_sq_sum(signal) / get_sq_sum(noise))


def get_alpha(snr, value):
    return math.pow(10, (1/20)*(value - snr)), np.sign(value - snr)


def process(signal, noise, value, noise_name='fake'):

    current_snr = get_snr(signal, noise, noise_name)

    alpha, sign = get_alpha(current_snr, value)

    if sign > 0:
        new_signal = signal*alpha
    else:
        new_signal = -signal*alpha

    new_noised_signal = new_signal + noise

    new_snr = get_snr(new_signal, noise, noise_name)
    return new_snr, new_noised_signal, new_signal


def generate_noised_signal_with_snr(filename_source, filename_noise, snr_value, root, dataset_name='DC_V6',
                                    is_for_valid_set=False,
                                    is_for_test_set=False,
                                    different_roots=False,
                                    second_root=''):

    if dataset_name == 'DC_V6':
        if is_for_valid_set:
            samples_filename = 'samples_valid/'
            noise_filename = 'noise_valid/'
        elif is_for_test_set:
            samples_filename = 'samples_test/'
            noise_filename = 'noise_test/'
        else:
            samples_filename = 'samples/'
            noise_filename = 'noise/'
    else:
        if is_for_test_set:
            samples_filename = 'audio_test/'
            noise_filename = 'audio_test/'
        else:
            samples_filename = 'audio/'
            noise_filename = 'audio/'

    if different_roots:
        noise_root = second_root
    else:
        noise_root = root

    signal, r_0 = wh.load_wav(samples_filename+filename_source, root)
    noise, r_1 = wh.load_wav(noise_filename+filename_noise, noise_root)

    assert r_0 == r_1, 'We have some problems with rate'

    size = min(signal.shape[0], noise.shape[0])
    signal = signal[:size]
    noise = noise[:size]

    _, _, new_signal = process(signal, noise, snr_value, filename_noise)

    return new_signal, noise, signal, r_0


def generate_noised_signal_with_snr_from_files(signal, noise, snr_value):

    size = min(signal.shape[0], noise.shape[0])
    signal = signal[:size]
    noise = noise[:size]

    _, _, new_signal = process(signal, noise, snr_value)

    return new_signal, noise, signal
