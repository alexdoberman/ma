import numpy as np
import math


def wave_ass_seg(mask, n_fft, overlap, file_name):

    classes = ['Pause', 'Signal']
    counts = [1, 1]

    first_class, ch_points = mask_to_fr_seg(mask, n_fft, overlap)
    curr_cl = first_class

    n_labels = len(ch_points) + 1
    with open(file_name + '.seg', 'w+') as wr_file:
        wave_ass_head = '[PARAMETERS]\nSAMPLING_FREQ = 16000\nBYTE_PER_SAMPLE = 2\nCODE = 0\n' \
                        'N_CHANNEL = 1\nN_LABEL = {}\n[LABELS]\n'.format(n_labels)
        wr_file.write(wave_ass_head)
        wr_file.write('{},1,{}_0\n'.format(0, classes[first_class]))

        for i in range(len(ch_points)):
            curr_cl = int(not curr_cl)
            counts[curr_cl] += 1
            wr_file.write('{},1,{}_{}\n'.format(ch_points[i]*2, classes[curr_cl], counts[curr_cl]))


def mask_to_fr_seg(mask, n_fft, overlap):

    mask = np.array(mask)
    assert len(mask.shape), 'Mask must be 1D'

    n_frames = mask.shape[0]

    cur_label = mask[0]
    first_label = cur_label

    ch_points = []
    for i in range(1, n_frames):

        if cur_label != mask[i]:
            cur_label = mask[i]
            ch_points.append(frame_to_time2(i, n_fft, overlap))

    return first_label, ch_points


def time_to_frame(time, sr, n_fft, overlap):
    hop_size = n_fft // overlap
    return int(math.floor(time * sr / hop_size))


def frame_to_time2(frame, n_fft, overlap):
    hop_size = n_fft // overlap

    return frame*hop_size


# simple test

import soundfile as sf

from mic_py_nn.features.preprocessing import energy_mask
from mic_py_nn.features.feats import stft


if __name__ == '__main__':

    # wav_path = '../../data/data_s/audio/f_1.wav'
    # data, rate = sf.read(wav_path)

    fft_size = 512
    overlap = 2

    # stft_arr = stft(data, fft_size, overlap)
    # mask = energy_mask(stft_data=stft_arr).astype(int)
    path = './temp/v42/0_mix.wav.npy'
    mask = (np.load(path) > 0.1).astype(int)
    print(mask)

    wave_ass_seg(mask, fft_size, overlap, '0_mix')
