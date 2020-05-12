import numpy as np


def mask_to_frames(mask, n_fft, hop):
    frames = []
    i = 0
    while i < len(mask):
        if i + n_fft < len(mask):
            frames.append(np.round(np.mean(mask[i:i + n_fft])))
            i += hop
        else:
            frames.append(np.round(np.mean(mask[i:])))
            i += hop
    return np.array(frames, dtype='int32')


def make_mask(average_sig, percent_threshold=60):
    return np.array \
        ((average_sig -np.min(average_sig)) < np.max((average_sig ) -np.min(average_sig) ) /percent_threshold,
                    dtype='float32')

def make_voice_borders(mask):
    voice_borders = []
    j = 0
    for i, el in enumerate(mask):
        if el == 1:
            j += 1
        elif el != 1 and j != 0:
            voice_borders.append([i-j-1, i-1])
            j = 0
    return np.array((voice_borders), dtype='float64')








