import numpy as np
import soundfile as sf
import os
import sys
sys.path.append('../../MA_PY')

from mic_py.mic_steering import propagation_vector_free_field
from mic_py.mic_geometry import get_source_position, get_sensor_positions
from mic_py.feats import stft, istft


FFT_SIZE = 512
OVERLAP = 2


def generate(filename, angle_hor, angle_vert, dir_to_save='', hor_mic_count=11, vert_mic_count=6, dHor=0.035, dVert=0.05,
             dir_name='gen_sig'):
    """
    Generate signal in given direction.

    :filename: input wav file
    :angle_hor: horizontal angle
    :angle_vert: vertical angle
    :dir_to_save: directory to store result, default - current dir

    :return:
    """
    angle_hor_init = angle_hor
    angle_vert_init = angle_vert
    angle_hor = -angle_hor
    angle_vert = -angle_vert

    amplitudes, rate = sf.read(filename)
    input_stft = stft(amplitudes, FFT_SIZE, OVERLAP)

    sensor_positions = get_sensor_positions(Hor_mic_count=hor_mic_count, Vert_mic_count=vert_mic_count,
                                            dHor=dHor, dVert=dVert)

    source_position = get_source_position(angle_Hor=angle_hor, angle_Vert=angle_vert)

    steering_vector = propagation_vector_free_field(sensor_positions, source_position,
                                                    N_fft=512, F_s=rate).transpose(1, 0)

    mic_array_stft = np.einsum('...i,ij->...ij', input_stft, steering_vector)

    save_signal(mic_array_stft, angle_hor_init, angle_vert_init, rate, dir_to_save=dir_to_save,
                hor_mic_count=hor_mic_count, vert_mic_count=vert_mic_count, dir_name=dir_name)


def save_signal(stft_array, angle_hor, angle_vert, rate, dir_to_save, dir_name, hor_mic_count=11, vert_mic_count=6):
    dir_path = dir_to_save + '{0}_{1}_{2}'.format(dir_name, angle_hor, angle_vert)

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    for i in range(vert_mic_count):
        for j in range(hor_mic_count):
            ch_istft = istft(stft_array[:, :, i*hor_mic_count + j], overlap=OVERLAP)
            sf.write(os.path.join(dir_path, 'ch_{0}_{1}.wav'.format(i, j)), ch_istft, rate)
