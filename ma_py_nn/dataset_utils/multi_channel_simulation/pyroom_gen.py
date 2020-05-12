import math
import numpy as np
import soundfile as sf
import pyroomacoustics as py_room
import os
import sys


from shutil import copytree, rmtree


from mic_py.mic_io import read_mic_wav_from_lst, read_mic_wav_from_folder
from mic_py.mic_stft import stft_arr
from mic_py.feats import istft
from mic_py.mic_geometry import get_sensor_positions, get_source_position
from mic_py.mic_steering import  propagation_vector_free_field
from mic_py.mic_gsc import *
from mic_py.mic_ds_beamforming import *


MIN_X = 3.5
MAX_X = 10
MIN_Y = 4
MAX_Y = 8
MIN_Z = 2.5
MAX_Z = 4
BIAS = 1.2


def process_ru_speech(path_to_base, out_base_dir_path, num_it):

    list_dir = os.listdir(path_to_base)
    for _ in range(num_it):
        spk_1_dir = np.random.choice(list_dir)
        spk_2_dir = np.random.choice(list_dir)

        while spk_1_dir == spk_2_dir:
            spk_2_dir = np.random.choice(list_dir)

        i = 0
        while os.path.exists(os.path.join(out_base_dir_path, '{}_{}_{}'.format(spk_1_dir, spk_2_dir, i))):
            i += 1
        out_path = os.path.join(out_base_dir_path, '{}_{}_{}'.format(spk_1_dir, spk_2_dir, i))
        os.mkdir(out_path)

        wav_1 = np.random.choice(os.listdir(os.path.join(path_to_base, spk_1_dir)))
        wav_2 = np.random.choice(os.listdir(os.path.join(path_to_base, spk_2_dir)))

        process_files([os.path.join(spk_1_dir, wav_1), os.path.join(spk_2_dir, wav_2)], path_to_base, out_path,
                      [spk_1_dir, spk_2_dir])


def split_set(root_path):
    data_path = os.path.join(root_path, '4_channel')
    meta_path = os.path.join(root_path, 'meta')
    test_path = os.path.join(root_path, 'audio_test')

    all_files = os.listdir(data_path)

    num_files = len(all_files)

    train_bound = int(num_files * 0.7)
    valid_bound = int(num_files * 0.9)

    np.random.shuffle(all_files)

    with open(os.path.join(meta_path, 'train'), 'w+') as train_meta:
        for i in range(train_bound):
            train_meta.write(all_files[i] + '\n')

    with open(os.path.join(meta_path, 'valid'), 'w+') as valid_meta:
        for i in range(train_bound, valid_bound):
            valid_meta.write(all_files[i] + '\n')

    for i in range(valid_bound, num_files):
        copytree(os.path.join(data_path, all_files[i]), os.path.join(test_path, all_files[i]))
        rmtree(os.path.join(data_path, all_files[i]))


def process_files(files_lst, root_path, out_path, spk_lst):
    files_lst = list(map(lambda x: os.path.join(root_path, x), files_lst))
    out_wavs, rate = room_simulation(files_lst)

    num_sources, num_ch, sig_len = out_wavs.shape
    mix = np.zeros(shape=(num_ch, sig_len))

    for idx, wav_ch in enumerate(out_wavs):

        os.mkdir(os.path.join(out_path, spk_lst[idx]))
        for i in range(len(wav_ch)):
            sf.write(os.path.join(out_path, spk_lst[idx], '{}_ch.wav'.format(i)), wav_ch[i], rate)
            mix[i] += wav_ch[i]
    os.mkdir(os.path.join(out_path, 'mix'))
    for i in range(num_ch):
        sf.write(os.path.join(out_path, 'mix', '{}_ch.wav'.format(i)), mix[i], rate)


def room_simulation(wav_paths, d_hor=0.035*10, d_vert=0.05*5, n_vert=2, n_hor=2):

    length, width, height = (np.random.uniform(MIN_Z, MAX_Z), np.random.uniform(MIN_X, MAX_X),
                             np.random.uniform(MIN_Y, MAX_Y))

    x_center, y_center, z_center = (np.random.uniform(BIAS, length - BIAS), np.random.uniform(BIAS, width - BIAS),
                                    np.random.uniform(BIAS, height - BIAS))

    abs_coef = np.random.uniform(0.1, 0.9)
    absorption = {'west': abs_coef, 'east': abs_coef,
                  'south': abs_coef * (2 - abs_coef), 'north': abs_coef,
                  'ceiling': abs_coef + 0.05, 'floor': abs_coef + 0.05}

    reflection = int(np.log(1 / 1000) / np.log(1 - abs_coef))  # number of reflections from the walls, ceiling, floor.

    if reflection > 30:
        reflection = 30

    # T60 = 0.1611 * (length * width * height) / (width * height * abs_coef * (2 - abs_coef) + length * width * 2 *
    #                                            abs_coef + length * height * abs_coef + length * height * 2 * abs_coef)

    '''
    ang_v_sp = np.arctan((speaker_pos[2] - z_center) / (speaker_pos[1] - y_center)) * 180 / np.pi
    ang_h_sp = np.arctan((speaker_pos[0] - x_center) / (speaker_pos[1] - y_center)) * 180 / np.pi
    ang_v_inf = np.arctan((inference_pos[2] - z_center) / (inference_pos[1] - y_center)) * 180 / np.pi
    ang_h_inf = np.arctan((inference_pos[0] - x_center) / (inference_pos[1] - y_center)) * 180 / np.pi
    '''

    row_wav_arr = []
    source_position_arr = []
    num_sources = len(wav_paths)
    rate = 0
    for path in wav_paths:
        rad = 1
        angle_h = np.random.uniform(-90, 90)
        angle_v = np.random.uniform(-90, 90)
        source_position_arr.append(get_position(x_center, y_center, z_center, rad, angle_h, angle_v))

        data, curr_rate = sf.read(path)
        if rate == 0:
            rate = curr_rate

        else:
            if rate != curr_rate:
                raise Exception('Sample rate don\'t match for files: {}'.format(wav_paths))
        row_wav_arr.append(data)

    if len(row_wav_arr) == 2:
        snr = np.random.uniform(-5, 5)
        new_s1, s2, _ = mix_with_snr(row_wav_arr[0], row_wav_arr[1], snr)
        row_wav_arr[0] = new_s1
        row_wav_arr[1] = s2
    # for i in range(num_sources):
    #    room.add_source(source_position_arr[i], signal=row_wav_arr[i], delay=0)
    mic_array = get_mic_array_position(n_hor, n_vert, d_hor, d_vert, x_center, y_center, z_center)

    wav2mch_rec = []

    # TODO: align!!

    min_len = sys.maxsize
    for i in range(num_sources):
        room = py_room.ShoeBox([length, width, height], absorption=absorption, max_order=reflection, fs=rate)
        room.add_source(source_position_arr[i], signal=row_wav_arr[i], delay=0)

        room.add_microphone_array(py_room.MicrophoneArray(mic_array, room.fs))
        room.image_source_model(use_libroom=True)
        room.simulate()
        if room.mic_array.signals.shape[1] < min_len:
            min_len = room.mic_array.signals.shape[1]
        wav2mch_rec.append(room.mic_array.signals)

    for i in range(len(wav2mch_rec)):
        wav2mch_rec[i] = wav2mch_rec[i][:, :min_len]

    return np.stack(wav2mch_rec), rate


def get_mic_array_position(n_hor, n_vert, d_hor, d_vert, x_center, y_center, z_center):
    x_0 = x_center - d_hor * (n_hor - 1) / 2
    x = list(np.linspace(x_0, x_0 + d_hor * (n_hor - 1), n_hor))
    x = np.array(x * n_vert)

    z_0 = z_center - d_vert * (n_vert - 1) / 2
    z = list(np.linspace(z_0, z_0 + d_vert * (n_vert - 1), n_vert))
    z = np.array([[i] * n_hor for i in z])
    z = z.flatten()

    y = np.array([y_center] * n_vert * n_hor)

    return np.array([x, y, z])


def get_position(x_center, y_center, z_center, r, ang_h, ang_v):
    y_sp = np.round(
        r / np.sqrt(1 + (np.tan(ang_h * np.pi / 180)) ** 2 + (np.tan(ang_v * np.pi / 180)) ** 2) + y_center, 4)
    x_sp = np.round((y_sp - y_center) * np.tan(ang_h * np.pi / 180) + x_center, 4)
    z_sp = np.round((y_sp - y_center) * np.tan(ang_v * np.pi / 180) + z_center, 4)
    return [x_sp, y_sp, z_sp]


def mix_with_snr(signal, noise, snr_value):

    size = min(signal.shape[0], noise.shape[0])
    signal = signal[:size]
    noise = noise[:size]

    _, _, new_signal = process(signal, noise, snr_value)

    return new_signal, noise, signal


def get_sq_sum(data):
    assert len(data.shape) == 1, 'Invalid shape'
    return sum(data**2)


def get_snr(signal, noise):
    sq_sum_noise = get_sq_sum(noise)
    if sq_sum_noise == 0:
        # print("we have big problems!!!!")
        return 10 * math.log10(get_sq_sum(signal) / 1e-6)
    return 10*math.log10(get_sq_sum(signal) / get_sq_sum(noise))


def get_alpha(snr, value):
    return math.pow(10, (1/20)*(value - snr)), np.sign(value - snr)


def process(signal, noise, value):

    current_snr = get_snr(signal, noise)

    alpha, sign = get_alpha(current_snr, value)

    if sign > 0:
        new_signal = signal*alpha
    else:
        new_signal = -signal*alpha

    new_noised_signal = new_signal + noise

    new_snr = get_snr(new_signal, noise)
    return new_snr, new_noised_signal, new_signal


if __name__ == '__main__':
    # process_ru_speech('/home/superuser/MA_ALG/datasets/ru_speech/one_channel',
    #                  '/home/superuser/MA_ALG/datasets/ru_speech/4_channel', 3500)
    split_set('/home/superuser/MA_ALG/datasets/ru_speech/')
